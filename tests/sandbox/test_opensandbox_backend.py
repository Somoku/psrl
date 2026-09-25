"""The OpenSandbox backend, driven through a fake provider.

Two rules carry the suite. A data endpoint is transport state and is re-resolved
after every resume, and a pause is asynchronous so the backend waits for the state
to settle rather than assuming the transition happened.
"""

from __future__ import annotations

import contextlib
import json
from urllib.parse import unquote, urlsplit

import pytest
from psrl.sandbox import (
    CredentialAuth,
    CredentialAuthType,
    CredentialBinding,
    CredentialRef,
    CredentialSubstitution,
    EgressAction,
    EgressPolicy,
    EgressRule,
    ExecMode,
    PauseMode,
    ResourceSpec,
    ResumeLevel,
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
)
from psrl.sandbox.backends.opensandbox import (
    LIFECYCLE_API_KEY_HEADER,
    OpenSandboxBackend,
    OpenSandboxConfig,
    OpenSandboxError,
)

pytestmark = pytest.mark.cpu_test

EXECD_TOKEN_HEADER = "X-EXECD-ACCESS-TOKEN"
EGRESS_AUTH_HEADER = "OPENSANDBOX-EGRESS-AUTH"


class FakeProvider:
    """An OpenSandbox whose lifecycle, execution, and egress planes the test controls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self.state = "running"
        self.states: list[str] = []
        # The provider publishes execd on this fixed container port and reports it the
        # same way as any other sandbox port.
        self.endpoint = "http://execd-1:44772"
        self.egress_endpoint = "http://egress-1:18080"
        # The headers the provider tells a caller to forward, per plane.
        self.exec_headers: dict[str, str] = {}
        self.egress_headers: dict[str, str] = {}
        self.uploads: list[tuple[str, dict]] = []
        self.file_infos: dict = {}
        self.snapshots: list[str] = []
        self.snapshot_state = "ready"
        self.vaults: list[dict] = []
        self.sessions_deleted: list[str] = []
        # Built templates, keyed by the id the provider mints, as the real server does.
        self.templates: list[dict] = []
        self.template_phase = "succeeded"
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.stream: list[str] = []
        self.files: dict[str, bytes] = {}
        self.next_sandbox = 0
        self.allocation: dict | None = None
        # How many further inspections still report a transient state, so a test can
        # drive the async settle rather than assume it happened.
        self.pending_for_inspects = 0

    def sandbox_payload(self, sandbox_id: str, state: str) -> dict:
        """Return one sandbox as the provider reports it.

        The lifecycle state is nested inside `status`, and reading it from the top level
        is the mistake that made every create wait for a running state it could never see.
        """
        return {"id": sandbox_id, "status": {"state": state}}

    # The transport surface the backend consumes.
    async def request(self, method, url, *, expected, payload=None, headers=None):
        self.calls.append((method, url, dict(payload or {}), dict(headers or {})))
        path = urlsplit(url).path
        if method == "POST" and path == "/v1/sandboxes":
            self.next_sandbox += 1
            self.created.append(dict(payload or {}))
            sandbox_id = f"sandbox-{self.next_sandbox}"
            response = self.sandbox_payload(sandbox_id, "running")
            if self.allocation is not None:
                response["allocation"] = self.allocation
            return response
        if method == "GET" and "/endpoints/" in path:
            port = path.rsplit("/", 1)[-1]
            if port == "18080":
                return {"endpoint": self.egress_endpoint, "headers": dict(self.egress_headers)}
            return {"endpoint": self.endpoint, "headers": dict(self.exec_headers)}
        if path == "/credential-vault" and method == "POST":
            self.vaults.append(dict(payload or {}))
            return {"revision": 1, "credentials": [], "bindings": []}
        if path == "/credential-vault" and method == "GET":
            return {"revision": 1, "credentials": [], "bindings": []}
        if method == "GET" and path.startswith("/v1/sandboxes/"):
            sandbox_id = path.rsplit("/", 1)[-1]
            if self.pending_for_inspects > 0:
                self.pending_for_inspects -= 1
                return self.sandbox_payload(sandbox_id, "pending")
            return self.sandbox_payload(sandbox_id, self.state)
        if method == "POST" and path.endswith("/pause"):
            self.state = "paused"
            self.states.append("paused")
            return None
        if method == "POST" and path.endswith("/resume"):
            self.state = "running"
            self.states.append("running")
            return None
        if method == "POST" and path.endswith("/snapshots"):
            snapshot = f"snapshot-{len(self.snapshots) + 1}"
            self.snapshots.append(snapshot)
            return {"id": snapshot, "sandboxId": path.split("/")[3], "status": {"state": "creating"}}
        if method == "GET" and path.startswith("/v1/snapshots/"):
            return {"id": path.rsplit("/", 1)[-1], "status": {"state": self.snapshot_state}}
        if method == "DELETE" and "/v1/snapshots/" in path:
            return None
        if method == "DELETE" and "/session/" in path:
            # A bash session, not the sandbox itself.
            self.sessions_deleted.append(path)
            return None
        if method == "DELETE":
            self.deleted.append(path)
            self.state = "terminated"
            return None
        if method == "POST" and path == "/v1/templates":
            template_id = f"tpl-{len(self.templates) + 1}"
            self.templates.append(
                {
                    "templateId": template_id,
                    "image": str((payload or {}).get("image")),
                    "publish": str((payload or {}).get("publish")),
                    "status": {"phase": self.template_phase},
                }
            )
            return self.templates[-1]
        if method == "GET" and path == "/v1/templates":
            return {"items": list(self.templates), "pagination": {"page": 1, "pageSize": 20}}
        if method == "GET" and "/v1/templates/" in path:
            wanted = path.rsplit("/", 1)[-1]
            for template in self.templates:
                if template["templateId"] == wanted:
                    return template
            return {"templateId": wanted, "status": {"phase": "missing"}}
        if path.endswith("/renew-expiration"):
            return None
        if path.endswith("/session") and method == "POST":
            return {"session_id": "shell-1"}
        if path.endswith("/files/info"):
            return dict(self.file_infos)
        if path.endswith("/metrics"):
            # The provider's own field names and units: MiB, a CPU share, no peak.
            return {
                "cpu_count": 4.0,
                "cpu_used_pct": 45.5,
                "mem_total_mib": 8192.0,
                "mem_used_mib": 4096.0,
                "timestamp": 1700000000000,
            }
        return None

    async def read_file(self, url, *, headers=None):
        # Recorded like any other call, so a test can assert the query a download used.
        self.calls.append(("GET", url, {}, dict(headers or {})))
        return self.files.get(unquote(url.split("path=", 1)[-1]), b"")

    async def upload_file(self, url, data, *, metadata, headers=None):
        # The provider's upload is multipart: a metadata part naming the destination,
        # then the bytes. The path is in the metadata, not in the URL.
        self.uploads.append((str(metadata.get("path")), dict(headers or {})))
        self.files[str(metadata.get("path"))] = data

    async def stream_lines(self, method, url, *, payload=None, headers=None):
        # Recorded like any other call, so a test can assert on the endpoint and the
        # payload a streaming command used.
        self.calls.append((method, url, dict(payload or {}), dict(headers or {})))
        for line in self.stream:
            yield line

    async def close(self):
        return None


class ExecTransport(FakeProvider):
    """The exec plane, reusing the same fake so state is shared."""

    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    def __getattr__(self, item):
        return getattr(self.provider, item)


def _config(**overrides) -> OpenSandboxConfig:
    payload = {
        "api_url": "http://lifecycle:8080",
        "api_key": "lifecycle-key",
        "poll_interval_s": 0.001,
        "pause_timeout_s": 1.0,
        "isolation_runtime": "gvisor",
    }
    payload.update(overrides)
    return OpenSandboxConfig(**payload)


def _backend(provider: FakeProvider, **overrides) -> OpenSandboxBackend:
    return OpenSandboxBackend(
        _config(**overrides),
        transport=provider,
        exec_transport_factory=lambda: ExecTransport(provider),
    )


def _spec(**overrides) -> SandboxSpec:
    payload = {"source": SandboxSource.image("python:3.11")}
    payload.update(overrides)
    return SandboxSpec(**payload)


def _events(provider: FakeProvider) -> list[str]:
    return [f"{method} {url}" for method, url, _, _ in provider.calls]


async def test_a_create_without_a_publish_target_never_asks_for_a_template() -> None:
    # The template API answers 501 on a runtime that cannot build golden images, so a
    # deployment that configured no publish target must not touch it on the create path.
    provider = FakeProvider()

    await _backend(provider).create(_spec())

    assert not [url for _, url, _, _ in provider.calls if "/v1/templates" in url]
    assert provider.created[0]["image"] == {"uri": "python:3.11"}


def test_the_declared_feature_set_matches_what_the_provider_does() -> None:
    # A template or a warm pool is declared only where it is configured, because the
    # provider answers 501 for templates and has no pool otherwise.
    capabilities = _backend(
        FakeProvider(), template_publish="s3://bucket/publish", warm_pool_ref="pool-1"
    ).capabilities

    for feature in (
        SandboxFeature.HIBERNATE,
        SandboxFeature.FILESYSTEM_SNAPSHOT,
        SandboxFeature.RESTORE,
        SandboxFeature.RESUME_ANYWHERE,
        SandboxFeature.WARM_POOL,
        SandboxFeature.IMAGE_BLOCK_DELIVERY,
        SandboxFeature.TEMPLATE_BUILD,
        SandboxFeature.ISOLATION_RUNTIME,
        SandboxFeature.EGRESS_POLICY,
        SandboxFeature.CREDENTIAL_INJECTION,
    ):
        assert capabilities.supports(feature), feature
    # It forks through the manager's checkpoint path, and pause releases compute.
    assert not capabilities.supports(SandboxFeature.NATIVE_FORK)
    assert not capabilities.supports(SandboxFeature.FREEZE)
    assert not capabilities.supports(SandboxFeature.IMAGE_ON_DEMAND)
    # A snapshot commits the rootfs, so claiming full state would let a caller read a
    # workspace resume as proof that its harness survives a move.
    assert not capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
    assert capabilities.resume_level is ResumeLevel.FILESYSTEM


def test_features_that_need_configuration_are_not_declared_without_it() -> None:
    capabilities = _backend(FakeProvider(), isolation_runtime=None).capabilities

    assert not capabilities.supports(SandboxFeature.TEMPLATE_BUILD)
    assert not capabilities.supports(SandboxFeature.WARM_POOL)
    assert not capabilities.supports(SandboxFeature.ISOLATION_RUNTIME)


async def test_the_lifecycle_plane_authenticates_with_its_own_header() -> None:
    provider = FakeProvider()
    await _backend(provider).create(_spec())

    lifecycle = [headers for _, url, _, headers in provider.calls if "/v1/sandboxes" in url]
    assert lifecycle
    assert lifecycle[0][LIFECYCLE_API_KEY_HEADER] == "lifecycle-key"


async def test_the_exec_plane_forwards_the_headers_its_endpoint_returned() -> None:
    # The provider returns the headers a caller must forward. A deployment whose execd requires
    # one must not be silently unauthenticated, and the name is omitted because the provider decides it.
    provider = FakeProvider()
    provider.exec_headers = {EXECD_TOKEN_HEADER: "execd-token"}
    provider.stream = ['{"type": "execution_complete"}']
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.ONE_SHOT))

    await session.exec("echo hi")

    headers = next(headers for _, url, _, headers in provider.calls if url.endswith("/command"))
    assert headers[EXECD_TOKEN_HEADER] == "execd-token"


async def test_the_exec_endpoint_is_resolved_on_the_port_execd_listens_on() -> None:
    # execd is injected at a fixed container port, so resolving any other port would
    # address a service that is not listening.
    provider = FakeProvider()

    await _backend(provider).create(_spec())

    assert any("/endpoints/44772" in url for _, url, _, _ in provider.calls)


async def test_a_resolved_proxy_route_is_not_used_as_the_plane_base() -> None:
    # A real server answers with `host:port/proxy/44772`, and the plane's own API is at
    # the root. Keeping the suffix sends every call somewhere that is not an endpoint.
    provider = FakeProvider()
    provider.endpoint = "execd-1:44772/proxy/44772"

    session = await _backend(provider).create(_spec())
    await session.exec("echo hi")

    assert any(url == "http://execd-1:44772/session" for _, url, _, _ in provider.calls)
    assert not any("/proxy/" in url for _, url, _, _ in provider.calls if "execd-1" in url)


async def test_an_endpoint_without_headers_is_still_usable() -> None:
    # The Docker runtime returns an endpoint and no headers at all, so a backend that
    # required one would refuse every sandbox that deployment creates.
    provider = FakeProvider()
    provider.exec_headers = {}

    session = await _backend(provider).create(_spec())

    assert (await session.exec("echo hi")).exit_code == 0


async def test_the_default_deployment_builds_its_own_exec_transport() -> None:
    # A deployment configures a URL and a key and nothing else. Leaving this unset raised
    # on the first command, which no test saw because each injected a factory.
    backend = OpenSandboxBackend(_config())

    assert callable(backend._exec_transport_factory)
    assert backend._exec_transport_factory() is not None


async def test_an_upload_sends_its_metadata_as_a_file_part() -> None:
    # The server reads `metadata` as a file part and answers INVALID_FILE_METADATA for a
    # plain field, so the filename is what makes the request well formed.
    from psrl.sandbox.backends.opensandbox import AiohttpTransport

    sent: dict[str, object] = {}

    class RecordingForm:
        def add_field(self, name, value, **kwargs):
            sent[name] = kwargs

    transport = AiohttpTransport()

    class Session:
        def post(self, url, data=None, headers=None):
            raise AssertionError("The form is inspected before the request is made.")

    async def fake_session():
        return Session()

    transport._get_session = fake_session
    import aiohttp

    original, aiohttp.FormData = aiohttp.FormData, RecordingForm
    try:
        with contextlib.suppress(AssertionError):
            await transport.upload_file("http://execd/files/upload", b"x", metadata={"path": "/tmp/a.txt"})
    finally:
        aiohttp.FormData = original

    assert sent["metadata"]["filename"] == "metadata.json"


async def test_a_prepared_template_is_what_a_create_admits_against() -> None:
    # prepare builds the template off the critical path and create uses it. The provider
    # mints the id, so the build is found by listing and matching its source image.
    provider = FakeProvider()
    backend = _backend(provider, template_publish="s3://bucket/publish")
    await backend.prepare(_spec())

    await backend.create(_spec(resources=ResourceSpec(cpu_count=2, memory_mb=4096)))

    created = provider.created[0]
    assert created["templateId"] == "tpl-1"
    assert "image" not in created
    # A template fixes the workload shape, so the provider rejects the fields that would
    # describe one, and it requires a timeout instead.
    assert "resourceLimits" not in created
    assert "entrypoint" not in created
    assert created["timeout"] > 0


async def test_a_template_build_sends_the_publish_target_the_provider_requires() -> None:
    # Both fields are required and the schema rejects anything else, so a build that
    # omits the publish target is refused rather than queued.
    provider = FakeProvider()
    backend = _backend(provider, template_publish="s3://bucket/publish")

    await backend.prepare(_spec())

    builds = [
        payload for method, url, payload, _ in provider.calls if method == "POST" and url.endswith("/v1/templates")
    ]
    assert builds == [{"image": "python:3.11", "publish": "s3://bucket/publish"}]


async def test_a_template_build_without_a_publish_target_is_refused_locally() -> None:
    # The template API is 501 off a fast-sandbox runtime, so a deployment that cannot
    # serve it must not reach it at all.
    provider = FakeProvider()
    backend = _backend(provider)

    with pytest.raises(OpenSandboxError, match="publish target"):
        await backend.control.build_template("python:3.11")

    assert not [url for _, url, _, _ in provider.calls if "/v1/templates" in url]


async def test_a_confirmed_pool_allocation_is_counted_as_a_warm_claim() -> None:
    provider = FakeProvider()
    provider.allocation = {"pool": "warm-1"}
    backend = _backend(provider)

    await backend.create(_spec())

    assert backend.metrics_snapshot().operations["warm_pool_claim"].count == 1


async def test_a_configured_warm_pool_is_what_a_create_claims_from() -> None:
    # A warm sandbox comes from a pool the operator pre-created. Without the claim the
    # capability would be declared and never used.
    provider = FakeProvider()

    await _backend(provider, warm_pool_ref="pool-1").create(_spec())

    created = provider.created[0]
    assert created["extensions"] == {"poolRef": "pool-1"}
    # A pool's pod defines its own shape, so the request carries none of its own.
    assert "image" not in created
    assert "resourceLimits" not in created
    assert "entrypoint" not in created


async def test_a_spec_the_pool_cannot_serve_is_created_from_its_image() -> None:
    # The provider rejects a network policy beside a pool reference, so a spec that needs
    # one must not be routed through the pool.
    provider = FakeProvider()
    spec = _spec(egress=EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "api.example.com"),)))

    await _backend(provider, warm_pool_ref="pool-1").create(spec)

    created = provider.created[0]
    assert "extensions" not in created
    assert created["image"] == {"uri": "python:3.11"}
    assert created["networkPolicy"]["defaultAction"] == "deny"


def _vault_spec(**overrides) -> SandboxSpec:
    """A spec that brokers one credential to one host."""
    payload = {
        "credentials": (CredentialRef(source_env="PSRL_TEST_SECRET", target_env="API_TOKEN"),),
        "egress": EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "api.example.com"),)),
        "credential_bindings": (CredentialBinding(name="api", hosts=("api.example.com",), credential="API_TOKEN"),),
    }
    payload.update(overrides)
    return _spec(**payload)


async def test_a_brokered_credential_is_written_to_the_sidecar_vault(monkeypatch) -> None:
    # The value is brokered at the egress boundary, so it is written to the vault and
    # the create payload only asks for the proxy to be switched on.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    provider = FakeProvider()
    backend = _backend(provider)

    await backend.create(_vault_spec())

    assert provider.created[0]["credentialProxy"] == {"enabled": True}
    assert provider.vaults == [
        {
            "credentials": [{"name": "API_TOKEN", "source": {"type": "inline", "value": "s3cr3t"}}],
            "bindings": [
                {
                    "name": "api",
                    "match": {
                        "schemes": ["https"],
                        "hosts": ["api.example.com"],
                        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        "paths": ["/*"],
                    },
                    "auth": {"type": "bearer", "credential": "API_TOKEN"},
                }
            ],
        }
    ]


async def test_the_sandbox_holds_a_placeholder_rather_than_the_credential(monkeypatch) -> None:
    # Existing tools read their credential from the environment, so something has to be
    # there, but the real value must not be that something.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    provider = FakeProvider()

    await _backend(provider).create(_vault_spec())

    environment = provider.created[0]["env"]
    assert environment["API_TOKEN"] == "psrl-vault:API_TOKEN"
    # The value never travels in a create payload or a metric surface.
    assert "s3cr3t" not in json.dumps(provider.created)


async def test_the_vault_is_reached_through_the_egress_endpoint(monkeypatch) -> None:
    # The vault is served by the egress sidecar, not the lifecycle plane, so the call
    # goes to a resolved endpoint and forwards the headers resolution returned.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    provider = FakeProvider()
    provider.egress_headers = {EGRESS_AUTH_HEADER: "egress-auth"}

    await _backend(provider).create(_vault_spec())

    method, url, _, headers = next(call for call in provider.calls if call[1].endswith("/credential-vault"))
    assert method == "POST"
    assert url.startswith(provider.egress_endpoint)
    assert headers[EGRESS_AUTH_HEADER] == "egress-auth"
    assert any("/endpoints/18080" in call[1] for call in provider.calls)


async def test_a_brokered_credential_does_not_use_the_warm_template(monkeypatch) -> None:
    # A template-backed sandbox has no egress sidecar, so a spec that brokers a
    # credential is created from its image or the vault could not exist at all.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    provider = FakeProvider()
    backend = _backend(provider)
    await backend.prepare(_vault_spec())

    await backend.create(_vault_spec())

    assert "template" not in provider.created[0]
    # The image travels as an object with a `uri`, which is the only shape the create
    # request accepts. A bare string is refused by the server.
    assert provider.created[0]["image"] == {"uri": "python:3.11"}


async def test_a_binding_whose_host_the_policy_blocks_is_refused(monkeypatch) -> None:
    # The sidecar injects only into a request it may forward, so a blocked host would
    # fail with a network error that says nothing about the missing allow rule.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    spec = _vault_spec(egress=EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "other.example.com"),)))

    with pytest.raises(OpenSandboxError, match="does not allow"):
        await _backend(FakeProvider()).create(spec)


async def test_a_binding_without_any_policy_is_refused(monkeypatch) -> None:
    # A spec that brokers a credential needs a policy, since with no policy the sidecar
    # has nothing to inject into.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")

    with pytest.raises(OpenSandboxError, match="requires an outbound policy"):
        await _backend(FakeProvider()).create(_vault_spec(egress=None))


async def test_a_credential_this_process_does_not_hold_fails_the_create(monkeypatch) -> None:
    # A task that silently runs unauthenticated produces a reward that looks valid and
    # is not, so the missing secret is a failure rather than an empty injection.
    monkeypatch.delenv("PSRL_TEST_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="not set in this process"):
        await _backend(FakeProvider()).create(_vault_spec())


def test_an_api_key_auth_requires_the_header_it_injects_into() -> None:
    with pytest.raises(ValueError, match="header name"):
        CredentialAuth(type=CredentialAuthType.API_KEY)


def test_a_passthrough_auth_requires_a_substitution() -> None:
    with pytest.raises(ValueError, match="passthrough"):
        CredentialAuth(type=CredentialAuthType.PASSTHROUGH)


async def test_a_substituting_binding_is_rendered_for_the_provider(monkeypatch) -> None:
    # Some upstreams take a credential in a path or a body rather than a header, and
    # the provider rewrites an exact placeholder on the listed surfaces.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    provider = FakeProvider()
    binding = CredentialBinding(
        name="token-request",
        hosts=("api.example.com",),
        credential="API_TOKEN",
        auth=CredentialAuth(
            type=CredentialAuthType.PASSTHROUGH,
            substitutions=(
                CredentialSubstitution(credential="API_TOKEN", placeholder="__token__", surfaces=("body", "query")),
            ),
        ),
    )

    await _backend(provider).create(_vault_spec(credential_bindings=(binding,)))

    auth = provider.vaults[0]["bindings"][0]["auth"]
    assert auth == {
        "type": "passthrough",
        "credential": "API_TOKEN",
        "substitutions": [{"credential": "API_TOKEN", "placeholder": "__token__", "in": ["body", "query"]}],
    }


async def test_connect_waits_out_a_transient_state() -> None:
    # The control plane reports a transient state before a sandbox accepts commands,
    # so connect must not hand back one that cannot run.
    provider = FakeProvider()
    provider.state = "running"
    provider.pending_for_inspects = 1

    session = await _backend(provider).connect("sandbox-9")

    assert session.ref.sandbox_id == "sandbox-9"


async def test_a_create_maps_resources_to_the_providers_quantity_map() -> None:
    # The provider takes Kubernetes-style quantities, so every value is a string and the
    # whole request is one field, not four.
    from psrl.sandbox import ResourceSpec

    provider = FakeProvider()
    backend = _backend(provider)
    spec = _spec(resources=ResourceSpec(cpu_count=2, memory_mb=4096, gpu_count=1))

    session = await backend.create(spec)

    assert session.ref.sandbox_id == "sandbox-1"
    created = provider.created[0]
    assert created["image"] == {"uri": "python:3.11"}
    assert created["entrypoint"] == ["tail", "-f", "/dev/null"]
    assert created["resourceLimits"] == {"cpu": "2", "memory": "4096Mi", "gpu": "1"}


async def test_a_fractional_cpu_request_becomes_millicores() -> None:
    # `1.5` is not a Kubernetes quantity, so a fraction has to be rendered in millicores.
    from psrl.sandbox import ResourceSpec

    provider = FakeProvider()
    await _backend(provider).create(_spec(resources=ResourceSpec(cpu_count=1.5)))

    assert provider.created[0]["resourceLimits"]["cpu"] == "1500m"


async def test_a_spec_that_asks_for_nothing_still_sends_the_providers_required_limits() -> None:
    # The provider requires resource limits on every non-template create, so a spec with
    # no request cannot leave the field out.
    provider = FakeProvider()
    await _backend(provider).create(_spec())

    assert provider.created[0]["resourceLimits"] == {"cpu": "1", "memory": "2Gi"}


async def test_a_disk_request_is_refused_rather_than_dropped() -> None:
    # The provider documents no disk quantity, so a disk request cannot be honoured and
    # dropping it would silently give the task less disk than it asked for.
    from psrl.sandbox import ResourceSpec

    provider = FakeProvider()

    with pytest.raises(OpenSandboxError, match="no disk resource quantity"):
        await _backend(provider).create(_spec(resources=ResourceSpec(cpu_count=1, disk_mb=8192)))

    assert provider.created == []


async def test_a_timeout_below_the_providers_floor_is_refused() -> None:
    provider = FakeProvider()

    with pytest.raises(OpenSandboxError, match="at least 60"):
        await _backend(provider).create(_spec(lifetime_timeout_s=30))


async def test_a_create_carries_volumes_and_metadata() -> None:
    from psrl.sandbox import VolumeSpec

    provider = FakeProvider()
    await _backend(provider).create(
        _spec(
            volumes=(VolumeSpec(name="cache", target="/cache"),),
            metadata={"task": "1"},
            idempotency_key="task-1:rollout",
        )
    )

    created = provider.created[0]
    assert created["volumes"] == [{"name": "cache", "mountPath": "/cache", "readOnly": False}]
    assert created["metadata"]["psrl.idempotency_key"] == "task-1:rollout"


async def test_a_template_source_creates_from_the_template_and_skips_resources() -> None:
    provider = FakeProvider()
    await _backend(provider).create(_spec(source=SandboxSource.template("task-template")))

    created = provider.created[0]
    assert created["templateId"] == "task-template"
    assert "image" not in created
    assert "resourceLimits" not in created


async def test_the_outbound_policy_travels_on_the_create_request() -> None:
    # The provider configures the policy during provisioning, so it is not patched in
    # afterwards: a policy applied later would leave a window with no policy at all.
    provider = FakeProvider()
    policy = EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "api.example.com"),))

    await _backend(provider).create(_spec(egress=policy))

    assert provider.created[0]["networkPolicy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "api.example.com"}],
    }
    assert not any("/networkpolicy" in url for _, url, _, _ in provider.calls)


async def test_a_policy_with_no_rules_is_a_deny_all_policy() -> None:
    provider = FakeProvider()
    await _backend(provider).create(_spec(egress=EgressPolicy()))

    assert provider.created[0]["networkPolicy"] == {"defaultAction": "deny", "egress": []}


async def test_a_rule_that_names_a_port_is_refused() -> None:
    # The provider derives the port from the scheme and rejects a rule that names one,
    # so refusing here is better than sending a policy it will reject.
    provider = FakeProvider()
    policy = EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "10.0.0.5", (443,)),))

    with pytest.raises(OpenSandboxError, match="derives an egress rule's port"):
        await _backend(provider).create(_spec(egress=policy))


async def test_an_isolation_requirement_without_a_configured_runtime_is_refused() -> None:
    provider = FakeProvider()
    backend = OpenSandboxBackend(
        OpenSandboxConfig(api_url="http://lifecycle:8080", poll_interval_s=0.001),
        transport=provider,
        exec_transport_factory=lambda: ExecTransport(provider),
    )

    with pytest.raises(OpenSandboxError, match="belongs to the"):
        await backend.create(_spec(required_features=frozenset({SandboxFeature.ISOLATION_RUNTIME})))


async def test_an_isolation_boundary_is_a_deployment_property_not_a_request_field() -> None:
    # The provider has no per-sandbox runtime field: the boundary belongs to the server,
    # so a deployment that provides one declares it and the create carries nothing.
    provider = FakeProvider()
    backend = _backend(provider)
    assert backend.capabilities.supports(SandboxFeature.ISOLATION_RUNTIME)

    await backend.create(_spec(required_features=frozenset({SandboxFeature.ISOLATION_RUNTIME})))

    assert "runtime" not in provider.created[0]


def test_a_deployment_without_a_boundary_does_not_declare_one() -> None:
    backend = OpenSandboxBackend(
        OpenSandboxConfig(api_url="http://lifecycle:8080", poll_interval_s=0.001),
        transport=FakeProvider(),
        exec_transport_factory=lambda: ExecTransport(FakeProvider()),
    )

    assert not backend.capabilities.supports(SandboxFeature.ISOLATION_RUNTIME)


async def test_the_persistent_shell_is_the_providers_own_session() -> None:
    # The provider's events carry their text in `text`, and there is no exit event, so a
    # run that reports no error is a success.
    provider = FakeProvider()
    provider.stream = ['{"type": "stdout", "text": "hi\\n"}', '{"type": "execution_complete", "execution_time": 12}']
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.PERSISTENT))

    result = await session.exec("echo hi")

    assert result.exit_code == 0
    assert result.stdout == "hi\n"
    create = next(
        payload for method, url, payload, _ in provider.calls if url.endswith("/session") and method == "POST"
    )
    # There is no shell to choose, and the body is optional.
    assert create == {}
    assert any("/session/shell-1/run" in url for _, url, _, _ in provider.calls)


async def test_the_session_run_reports_an_error_event_as_a_failure() -> None:
    # The stream has no exit code, so an error event is the only failure signal there is.
    provider = FakeProvider()
    provider.stream = [
        '{"type": "stdout", "text": "partial"}',
        '{"type": "error", "error": {"ename": "NameError", "evalue": "boom", "traceback": ["line 1"]}}',
    ]
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.PERSISTENT))

    result = await session.exec("python -c boom")

    assert result.exit_code == 1
    assert "NameError" in result.stderr and "boom" in result.stderr and "line 1" in result.stderr


async def test_a_result_event_is_read_from_its_mime_map() -> None:
    # A code-interpreter style result arrives under a MIME map rather than as plain text.
    provider = FakeProvider()
    provider.stream = ['{"type": "result", "results": {"text/plain": "4"}}']
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.PERSISTENT))

    result = await session.exec("1 + 3")

    assert result.stdout == "4"


async def test_a_session_is_deleted_through_the_providers_own_route() -> None:
    provider = FakeProvider()
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.PERSISTENT))
    await session.exec("echo hi")
    client = session._exec_client

    await client.delete_session("shell-1")

    assert provider.sessions_deleted == ["/session/shell-1"]


async def test_the_one_shot_mode_uses_the_command_endpoint() -> None:
    provider = FakeProvider()
    provider.stream = ['{"type": "stderr", "text": "nope\\n"}', '{"type": "execution_complete"}']
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.ONE_SHOT))

    result = await session.exec("exit 3")

    assert result.stdout == ""
    assert result.stderr == "nope\n"
    payload = next(payload for _, url, payload, _ in provider.calls if url.endswith("/command"))
    assert payload["command"] == "exit 3"
    # The provider's one-shot field is `envs`, and there is none to send.
    assert "env" not in payload


async def test_a_command_exceeding_its_deadline_carries_the_deadline_to_the_provider() -> None:
    provider = FakeProvider()
    provider.stream = ['{"type": "execution_complete"}']
    session = await _backend(provider).create(_spec(exec_mode=ExecMode.ONE_SHOT))

    await session.exec("sleep 1", timeout_s=5)

    payload = next(payload for method, url, payload, _ in provider.calls if url.endswith("/command"))
    # The provider's timeout is milliseconds, which is the unit it enforces.
    assert payload["timeout"] == 5000


async def test_files_move_through_the_execution_plane() -> None:
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    await session.write_bytes("/work/out.txt", b"payload")

    assert await session.read_bytes("/work/out.txt") == b"payload"


async def test_usage_is_unknown_because_the_plane_measures_the_host() -> None:
    # `GET /metrics` reports the node the plane runs on rather than the sandbox, so an
    # operator sizing an envelope from it would be wrong by orders of magnitude.
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    usage = await session.stats()

    assert usage.memory_bytes is None
    assert usage.peak_memory_bytes is None
    assert usage.cpu_total_ns is None


async def test_a_pause_waits_for_the_state_to_settle() -> None:
    # The control plane accepts the intent and reports an intermediate state, so
    # returning immediately would let a caller command a sandbox that is stopping.
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    await session.pause(PauseMode.HIBERNATE)

    assert provider.states == ["paused"]
    assert await session.status() is SandboxStatus.PAUSED


async def test_a_freeze_is_refused_because_pause_releases_compute() -> None:
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    with pytest.raises(RuntimeError, match="hibernation"):
        await session.pause(PauseMode.FREEZE)


async def test_a_resume_resolves_the_endpoint_again() -> None:
    # A checkpoint and resume can move the sandbox, so a cached address is stale.
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())
    await session.exec("echo one")
    resolved_before = sum(1 for _, url, _, _ in provider.calls if "/endpoints/" in url)

    await session.resume()
    provider.endpoint = "http://execd-2:8080"
    await session.exec("echo two")

    resolved_after = sum(1 for _, url, _, _ in provider.calls if "/endpoints/" in url)
    assert resolved_after > resolved_before


async def test_a_resume_reinstalls_the_credential_vault(monkeypatch) -> None:
    # A resume rebuilds the egress sidecar with an empty vault, so without reinstalling
    # it the workload issues unauthenticated requests.
    monkeypatch.setenv("PSRL_TEST_SECRET", "s3cr3t")
    provider = FakeProvider()
    session = await _backend(provider).create(_vault_spec())
    assert len(provider.vaults) == 1

    await session.pause(PauseMode.HIBERNATE)
    await session.resume()

    assert len(provider.vaults) == 2


async def test_a_snapshot_is_named_and_promises_a_filesystem_resume() -> None:
    # The provider commits the rootfs, so reporting full state would let a caller read a
    # workspace resume as proof that its harness survived a move.
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    snapshot = await session.snapshot(SnapshotKind.FILESYSTEM)

    assert snapshot.snapshot_id == "snapshot-1"
    assert snapshot.kind is SnapshotKind.FILESYSTEM
    assert snapshot.resume_level is ResumeLevel.FILESYSTEM
    assert snapshot.metadata["psrl.snapshot.name"].startswith("psrl-")


async def test_a_snapshot_waits_until_its_artifact_is_ready() -> None:
    # The create is accepted with a `Creating` snapshot, so a restore issued against the
    # returned id would race the capture.
    provider = FakeProvider()
    provider.snapshot_state = "creating"
    session = await _backend(provider, snapshot_timeout_s=0.01).create(_spec())

    with pytest.raises(OpenSandboxError, match="still 'creating'"):
        await session.snapshot(SnapshotKind.FILESYSTEM)

    assert any("/v1/snapshots/" in call[1] for call in provider.calls)


async def test_a_full_state_snapshot_is_refused() -> None:
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    with pytest.raises(RuntimeError, match="commits the sandbox filesystem"):
        await session.snapshot(SnapshotKind.FULL_STATE)


async def test_a_restore_creates_from_the_snapshot_far_from_its_source() -> None:
    provider = FakeProvider()
    backend = _backend(provider)
    snapshot = await (await backend.create(_spec())).snapshot(SnapshotKind.FILESYSTEM)

    restored = await backend.restore(snapshot, _spec())

    assert restored.ref.sandbox_id == "sandbox-2"
    assert provider.created[-1]["snapshotId"] == "snapshot-1"


async def test_terminate_deletes_the_sandbox_and_reports_it_gone() -> None:
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    await session.terminate()

    assert provider.deleted
    assert await session.status() is SandboxStatus.TERMINATED
    await session.terminate()  # a second terminate is not an error


async def test_connect_resumes_a_paused_sandbox() -> None:
    provider = FakeProvider()
    provider.state = "paused"
    backend = _backend(provider)

    session = await backend.connect("sandbox-9")

    assert provider.states == ["running"]
    assert session.ref.sandbox_id == "sandbox-9"


async def test_connect_refuses_a_sandbox_in_a_terminal_state() -> None:
    provider = FakeProvider()
    provider.state = "terminated"

    with pytest.raises(OpenSandboxError, match="cannot accept commands"):
        await _backend(provider).connect("sandbox-9")


async def test_prepare_builds_a_template_once_and_verifies_it_afterwards() -> None:
    provider = FakeProvider()
    backend = _backend(provider, template_publish="s3://bucket/publish")
    spec = _spec()

    await backend.prepare(spec)
    await backend.prepare(spec)

    builds = [url for method, url, _, _ in provider.calls if method == "POST" and url.endswith("/v1/templates")]
    assert len(builds) == 1


async def test_prepare_without_a_publish_target_builds_nothing() -> None:
    # A template is an optimization rather than a prerequisite, and the template API is
    # 501 off a fast-sandbox runtime, so a deployment that cannot serve one never asks.
    provider = FakeProvider()

    await _backend(provider).prepare(_spec())

    assert not [url for _, url, _, _ in provider.calls if "/v1/templates" in url]


async def test_prepare_ignores_a_template_source() -> None:
    provider = FakeProvider()

    await _backend(provider, template_publish="s3://bucket/publish").prepare(
        _spec(source=SandboxSource.template("task-template"))
    )

    assert not [url for _, url, _, _ in provider.calls if "/v1/templates" in url]


async def test_shutdown_closes_the_lifecycle_plane() -> None:
    provider = FakeProvider()
    backend = _backend(provider)

    await backend.shutdown()

    assert backend.metrics_snapshot() is not None


def test_the_configuration_refuses_windows_it_cannot_honor() -> None:
    with pytest.raises(ValueError, match="api_url"):
        OpenSandboxConfig(api_url="")
    with pytest.raises(ValueError, match="polling"):
        OpenSandboxConfig(api_url="http://lifecycle:8080", poll_interval_s=0)


def test_a_binding_must_name_a_host() -> None:
    # A binding that matched every host would hand the credential to whatever the
    # workload asked for, which is the opposite of what brokering is for.
    with pytest.raises(ValueError, match="at least one host"):
        CredentialBinding(name="api", hosts=(), credential="API_TOKEN")


def test_a_binding_must_name_a_credential_the_spec_defines() -> None:
    with pytest.raises(ValueError, match="do not define"):
        SandboxSpec(
            SandboxSource.image("image"),
            credentials=(CredentialRef(source_env="A", target_env="B"),),
            credential_bindings=(CredentialBinding(name="api", hosts=("h",), credential="MISSING"),),
        )


def test_a_substitution_names_a_credential_the_spec_defines() -> None:
    # The auth rule can read a second credential, so the check has to look there too.
    with pytest.raises(ValueError, match="do not define"):
        SandboxSpec(
            SandboxSource.image("image"),
            credentials=(CredentialRef(source_env="A", target_env="API_TOKEN"),),
            credential_bindings=(
                CredentialBinding(
                    name="api",
                    hosts=("h",),
                    credential="API_TOKEN",
                    auth=CredentialAuth(
                        type=CredentialAuthType.PASSTHROUGH,
                        substitutions=(CredentialSubstitution(credential="OTHER", placeholder="__x__"),),
                    ),
                ),
            ),
        )


async def test_a_file_is_written_as_the_providers_two_part_multipart() -> None:
    # The provider reads a JSON metadata part naming the destination and then the bytes,
    # so the path is in the body rather than in the URL.
    provider = FakeProvider()
    session = await _backend(provider).create(_spec())

    await session.write_bytes("/workspace/a b.txt", b"payload")

    assert provider.uploads == [("/workspace/a b.txt", {})]
    assert provider.files["/workspace/a b.txt"] == b"payload"


async def test_a_file_path_is_escaped_into_the_download_query() -> None:
    # A path with a space or a `+` has to survive the query string, or the download asks
    # the sandbox for a different file than the caller named.
    provider = FakeProvider()
    provider.files["/workspace/a b.txt"] = b"payload"
    session = await _backend(provider).create(_spec())

    assert await session.read_bytes("/workspace/a b.txt") == b"payload"

    url = next(url for method, url, _, _ in provider.calls if "/files/download" in url and method == "GET")
    assert "a%20b.txt" in url


async def test_a_file_info_read_returns_the_providers_map() -> None:
    provider = FakeProvider()
    provider.file_infos = {"/workspace/a.txt": {"path": "/workspace/a.txt", "mode": 644}}
    client = await _backend(provider).create(_spec())

    info = await (await client._client()).file_info("/workspace/a.txt")

    assert info == {"/workspace/a.txt": {"path": "/workspace/a.txt", "mode": 644}}

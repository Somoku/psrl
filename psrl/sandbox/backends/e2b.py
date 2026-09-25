"""E2B data plane with provider-specific AgentEnv and Cube control planes.

**AgentEnv goes through the provider's own SDK**, whose source is the contract: create,
connect (which resumes), pause, fork, snapshot, and delete are all public SDK methods, so
PSRL calls them rather than a REST shape it would have to guess. Two things stay outside
it, both template management rather than sandbox lifecycle: importing the template an
image resolves to, and the listing that reclaims a lost create.

**CubeSandbox still uses a native control plane**, namely `ProviderSDKClientFactory` and
its `cpuCount`/`memoryMB`/`diskSizeMB` mapping, for which no public API reference was found.
That mapping is unverified, and it is not evidence that AgentEnv works the same way.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

import aiohttp

from psrl.sandbox.async_utils import acquire_nowait
from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResumeLevel,
    SandboxBackend,
    SandboxBusyError,
    SandboxCapabilities,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
    require_batch_count,
)
from psrl.sandbox.metrics import SandboxMetrics, SandboxMetricsSnapshot


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _close_client(client: Any) -> None:
    """Best-effort close of provider transport state without killing the VM."""
    close = getattr(client, "close", None)
    if close is not None:
        try:
            await _await(close())
        except Exception:
            pass


# The provider documents the longest life a sandbox may be given: 24 hours on the higher
# plan and one hour on the lower, and rejects a request above it rather than clamping it.
_MAX_SANDBOX_TTL_S = 86_400.0


def _validated_ttl(value: float | None) -> float | None:
    """
    Return a requested sandbox life, refusing one the provider cannot grant.
    """
    if value is not None and value > _MAX_SANDBOX_TTL_S:
        raise ValueError(
            f"Provider sandboxes live at most {_MAX_SANDBOX_TTL_S:g}s, so the requested {value:g}s cannot be "
            "honoured. A longer episode belongs in a resumable loop rather than in one sandbox's lifetime."
        )
    return value


def _requested_life_s(session: SandboxSession) -> float | None:
    """Return the life the caller asked for, so a provider sandbox does not expire early.

    A resume and a fork both start the provider's expiry clock over. Sending nothing
    would let the new sandbox expire against a deadline the caller never asked for,
    and the caller would see an expiry it did not configure.
    """
    spec = getattr(session, "spec", None)
    if spec is None:
        return None
    return _validated_ttl(spec.lifetime_timeout_s or spec.idle_timeout_s)


def _resource_payload(spec: SandboxSpec) -> dict[str, int]:
    """Map portable resources to E2B-compatible control-plane fields."""
    payload: dict[str, int] = {}
    if spec.resources.cpu_count is not None:
        if not float(spec.resources.cpu_count).is_integer():
            raise ValueError("MicroVM backends require an integral cpu_count.")
        payload["cpuCount"] = int(spec.resources.cpu_count)
    if spec.resources.memory_mb is not None:
        payload["memoryMB"] = spec.resources.memory_mb
    if spec.resources.disk_mb is not None:
        payload["diskSizeMB"] = spec.resources.disk_mb
    return payload


def _metadata(spec: SandboxSpec) -> dict[str, str]:
    metadata = dict(spec.metadata)
    if spec.idempotency_key:
        metadata["psrl.idempotency_key"] = spec.idempotency_key
    return metadata


def _validate_provider_spec(spec: SandboxSpec) -> None:
    if spec.policy_profile is not None:
        raise RuntimeError("Provider backends do not accept Docker policy profiles.")


class E2BClientFactory(Protocol):
    """Minimal factory boundary around an E2B-compatible SDK."""

    async def create(self, spec: SandboxSpec) -> Any: ...

    async def connect(self, sandbox_id: str, *, timeout: int | None = None) -> Any: ...

    async def delete_snapshot(self, snapshot_id: str) -> None: ...

    async def close(self) -> None: ...


class E2BStateDriver(Protocol):
    """Provider extension point for native state operations."""

    @property
    def capabilities(self) -> SandboxCapabilities: ...

    async def pause(self, session: E2BSession, mode: PauseMode) -> None: ...

    async def resume(self, session: E2BSession) -> None: ...

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef: ...

    async def fork(self, session: E2BSession, count: int) -> Sequence[SandboxSession]: ...

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession: ...

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None: ...

    async def close(self) -> None: ...


async def _adopt_fork_children(
    backend: E2BBackend,
    session: E2BSession,
    raw_children: Any,
    count: int,
) -> list[E2BSession]:
    """Adopt a fork's children, or destroy what was adopted and fail.

    The SDK returns one entry per requested fork, each being either a sandbox or the
    exception that fork failed with, because forks succeed and fail independently. A
    partial result is not a smaller group, so the children that did start are destroyed
    before the failure surfaces and the caller is left with nothing to clean up.

    Args:
        backend (E2BBackend): The backend the children belong to.
        session (E2BSession): The parent, whose spec the children inherit.
        raw_children (Any): The SDK's per-entry result list.
        count (int): How many children were requested.

    Returns:
        list[E2BSession]: The adopted children, in the order the provider returned them.

    Raises:
        RuntimeError: When the provider returned a short list or any entry failed.
    """
    children = list(raw_children or ())
    if len(children) != count:
        raise RuntimeError(f"Provider native fork produced {len(children)} entry(s) for a request of {count}.")
    started: list[E2BSession] = []
    failure: BaseException | None = None
    for child in children:
        if isinstance(child, BaseException):
            failure = failure or child
            continue
        backend.metrics.session_started()
        started.append(E2BSession(backend, child, spec=session.spec))
    if failure is not None:
        # Every successful entry is a live sandbox on the provider, even one that appeared
        # after the failure, so all are destroyed rather than only those reached so far.
        await asyncio.gather(*(entry.terminate() for entry in started), return_exceptions=True)
        raise RuntimeError(f"Provider native fork failed: {failure}.") from failure
    return started


class ProviderControlClient:
    """Persistent HTTP client for E2B-compatible provider control planes."""

    def __init__(
        self,
        api_url: str,
        *,
        api_key: str | None = None,
        request_timeout_s: float = 300.0,
        connection_limit: int = 128,
    ) -> None:
        if not api_url:
            raise ValueError("Provider api_url is required.")
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout_s = request_timeout_s
        self.connection_limit = connection_limit
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
                connector=aiohttp.TCPConnector(limit=self.connection_limit),
            )
        return self._session

    async def request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...],
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        session = await self._get_session()
        async with session.request(method, f"{self.api_url}{path}", json=payload) as response:
            if response.status not in expected:
                body = await response.text()
                raise RuntimeError(f"Provider returned HTTP {response.status} for {path!r}: {body.strip()}.")
            if response.status == 204:
                return None
            return await response.json()

    async def delete_idempotent(self, path: str) -> None:
        session = await self._get_session()
        async with session.delete(f"{self.api_url}{path}") as response:
            if response.status not in (204, 404):
                body = await response.text()
                raise RuntimeError(f"Provider returned HTTP {response.status} for {path!r}: {body.strip()}.")

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class E2BSDKClientFactory:
    """Lazy wrapper for the official `e2b` Python package."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        create_kwargs: Mapping[str, Any] | None = None,
        connect_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.create_kwargs = dict(create_kwargs or {})
        self.connect_kwargs = dict(connect_kwargs or {})

    @staticmethod
    def _sandbox_class():
        try:
            from e2b import AsyncSandbox
        except ImportError as exc:
            raise RuntimeError(
                "The E2B backend requires the optional e2b package. Install PSRL with sandbox-e2b support."
            ) from exc
        return AsyncSandbox

    def _sdk_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.api_key:
            options["api_key"] = self.api_key
        if self.api_url:
            options["api_url"] = self.api_url
        return options

    async def create(self, spec: SandboxSpec) -> Any:
        _validate_provider_spec(spec)
        if spec.source.kind != SandboxSourceKind.TEMPLATE:
            raise RuntimeError("E2B requires a template source; direct OCI image creation is unsupported.")
        if spec.resources != type(spec.resources)():
            raise RuntimeError(
                "E2B runtime resources are fixed by the template and cannot be overridden at create time."
            )
        sandbox_class = self._sandbox_class()
        kwargs = {**self.create_kwargs, **self._sdk_options()}
        kwargs["template"] = spec.source.reference
        metadata = _metadata(spec)
        if metadata:
            kwargs["metadata"] = metadata
        if spec.env:
            kwargs["envs"] = dict(spec.env)
        # The provider's `timeout` is the sandbox's own life, so an absolute lifetime is the
        # right source. Reading the idle window first would expire it against a stale deadline.
        timeout = _validated_ttl(spec.lifetime_timeout_s or spec.idle_timeout_s)
        if timeout is not None:
            kwargs["timeout"] = max(1, math.ceil(timeout))
        return await _await(sandbox_class.create(**kwargs))

    async def connect(self, sandbox_id: str, *, timeout: int | None = None) -> Any:
        """Connect, which the provider documents as auto-resuming a paused sandbox.

        The optional timeout is the sandbox's new life. The provider only applies it when
        it is longer than the remaining one, so passing the caller's requested life is how
        a resumed sandbox avoids expiring against its pre-pause deadline.
        """
        sandbox_class = self._sandbox_class()
        kwargs = {**self.connect_kwargs, **self._sdk_options()}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await _await(sandbox_class.connect(sandbox_id, **kwargs))

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete an E2B snapshot idempotently through the SDK control plane."""
        sandbox_class = self._sandbox_class()
        delete_snapshot = getattr(sandbox_class, "delete_snapshot", None)
        if delete_snapshot is None:
            raise RuntimeError("The installed e2b SDK does not support snapshot deletion; upgrade e2b.")
        await _await(delete_snapshot(snapshot_id, **self._sdk_options()))

    async def close(self) -> None:
        return None


class ProviderSDKClientFactory(E2BSDKClientFactory):
    """Create via a provider API, then attach the E2B data-plane SDK."""

    def __init__(self, *, control: ProviderControlClient, **kwargs: Any) -> None:
        super().__init__(api_key=control.api_key, api_url=control.api_url, **kwargs)
        self.control = control

    async def _connect_created(self, response: Mapping[str, Any]) -> Any:
        sandbox_id = response.get("sandboxID") or response.get("sandbox_id")
        if not sandbox_id:
            raise RuntimeError("Provider create response did not contain sandboxID.")
        return await self.connect(str(sandbox_id))

    async def close(self) -> None:
        await self.control.close()


def _template_reference(spec: SandboxSpec) -> str:
    """Return the template or snapshot a create names.

    A template source names itself, and so does a snapshot: this provider creates a
    sandbox from a template, and a snapshot id is accepted wherever a template id is. An
    image source names the template derived from its reference, which is the one
    `prepare` imports. An image becomes a template rather than being a create-time
    source, so a spec whose image was never prepared fails on the provider's
    missing-template error rather than on a request shape PSRL invented.
    """
    if spec.source.kind in {SandboxSourceKind.TEMPLATE, SandboxSourceKind.IMAGE}:
        if spec.source.kind is SandboxSourceKind.IMAGE:
            return template_name_for_image(spec.source.reference)
        return spec.source.reference
    raise RuntimeError(f"Provider create cannot start from source kind {spec.source.kind.value!r}.")


class AgentEnvClientFactory(E2BSDKClientFactory):
    """AgentEnv's create, on the provider's own SDK.

    Two decisions the SDK's own create does not make for us.

    - **An image is not a create-time source.** This provider creates a sandbox from a
      template, and an image becomes a template. So an image source resolves to the
      template name derived from it, which is the one `prepare` imports.
    - **The shape belongs to the template.** Resources and an entrypoint are fixed by the
      template's build, so a spec that asks to override them is refused here rather than
      sent for the server to reject.
    """

    async def create(self, spec: SandboxSpec) -> Any:
        _validate_provider_spec(spec)
        if spec.resources != type(spec.resources)():
            raise RuntimeError(
                "AgentEnv resources are fixed by the template a sandbox is created from, so a spec that "
                "overrides them cannot be served. The template prepare imports carries the shape instead."
            )
        if spec.mounts:
            raise RuntimeError("AgentEnv has no host bind mount, so a spec that names one cannot be served.")
        if spec.volumes:
            # The SDK takes a `volume_mounts` value whose shape the provider does not publish,
            # so rendering one would be a guess. The capability is undeclared, so admission refuses it.
            raise RuntimeError(
                "AgentEnv volumes are not wired: the SDK's volume-mount value has no published shape, and "
                "guessing it would send a request shape nobody can verify. The backend does not declare the "
                "capability, so a spec that needs provider storage is refused at admission."
            )
        sandbox_class = self._sandbox_class()
        kwargs = {**self.create_kwargs, **self._sdk_options()}
        kwargs["template"] = _template_reference(spec)
        metadata = _metadata(spec)
        if metadata:
            kwargs["metadata"] = metadata
        if spec.env:
            kwargs["envs"] = dict(spec.env)
        timeout = _validated_ttl(spec.lifetime_timeout_s or spec.idle_timeout_s)
        if timeout is not None:
            kwargs["timeout"] = max(1, math.ceil(timeout))
        return await _await(sandbox_class.create(**kwargs))


class CubeSandboxClientFactory(ProviderSDKClientFactory):
    """CubeSandbox control-plane mapping for template-backed microVMs."""

    async def create(self, spec: SandboxSpec) -> Any:
        _validate_provider_spec(spec)
        if spec.source.kind != SandboxSourceKind.TEMPLATE:
            raise RuntimeError("CubeSandbox requires a built template; direct OCI image creation is unsupported.")
        payload: dict[str, Any] = {
            "templateID": spec.source.reference,
            **_resource_payload(spec),
            **self.create_kwargs,
        }
        # The provider's `timeout` is the sandbox's own life, so an absolute lifetime wins
        # over an idle window when a spec declares both.
        lifetime_s = spec.lifetime_timeout_s or spec.idle_timeout_s
        if lifetime_s is not None:
            payload["timeout"] = max(1, math.ceil(lifetime_s))
        if spec.env:
            payload["envVars"] = dict(spec.env)
        metadata = _metadata(spec)
        if metadata:
            payload["metadata"] = metadata
        response = await self.control.request("POST", "/sandboxes", expected=(201,), payload=payload)
        return await self._connect_created(response)


class E2BHibernateDriver:
    """State driver for the standard E2B pause/connect lifecycle."""

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(frozenset({SandboxFeature.HIBERNATE}))

    async def pause(self, session: E2BSession, mode: PauseMode) -> None:
        if mode != PauseMode.HIBERNATE:
            raise RuntimeError("E2B pause has hibernation semantics, not process freeze semantics.")
        await _await(session.client.pause(keep_memory=True))

    async def resume(self, session: E2BSession) -> None:
        await session.refresh_transport()

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef:
        raise NotImplementedError("No snapshot state driver is configured for this E2B-compatible backend.")

    async def fork(self, session: E2BSession, count: int) -> Sequence[SandboxSession]:
        raise NotImplementedError("No native fork state driver is configured for this E2B-compatible backend.")

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        raise NotImplementedError("No restore state driver is configured for this E2B-compatible backend.")

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        raise NotImplementedError("No snapshot state driver is configured for this E2B-compatible backend.")

    async def close(self) -> None:
        return None


class E2BNativeStateDriver(E2BHibernateDriver):
    """State operations provided by the official E2B SDK."""

    def __init__(
        self,
        client_factory: E2BClientFactory,
        *,
        snapshot_name_prefix: str = "psrl",
        snapshot_request_timeout_s: float | None = None,
    ) -> None:
        self.client_factory = client_factory
        self.snapshot_name_prefix = snapshot_name_prefix
        self.snapshot_request_timeout_s = snapshot_request_timeout_s

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            frozenset(
                {
                    SandboxFeature.HIBERNATE,
                    SandboxFeature.FULL_STATE_SNAPSHOT,
                    SandboxFeature.RESTORE,
                    SandboxFeature.NATIVE_FORK,
                }
            )
        )

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef:
        """Capture the sandbox, named so a restore can address it by name.

        The provider pauses the sandbox while it captures and can take minutes over it, so
        the SDK request timeout is raised rather than left at its own default.
        """
        if kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("E2B create_snapshot captures full execution state, not filesystem-only state.")
        name = f"{self.snapshot_name_prefix}-{session.sandbox_id}-{uuid.uuid4().hex[:12]}"
        options: dict[str, Any] = {}
        if self.snapshot_request_timeout_s is not None:
            options["request_timeout"] = self.snapshot_request_timeout_s
        snapshot = await _await(session.client.create_snapshot(name=name, **options))
        return SnapshotRef(
            backend=session.backend.name,
            snapshot_id=str(snapshot.snapshot_id),
            kind=SnapshotKind.FULL_STATE,
            metadata={"psrl.snapshot.name": name, "names": list(snapshot.names)},
        )

    async def fork(self, session: E2BSession, count: int) -> Sequence[SandboxSession]:
        """Branch the sandbox through the SDK's one-shot count.

        A group size is a configuration value, so one call asks for the whole
        group instead of one call per member.
        """
        require_batch_count(count)
        raw_children = await _await(session.client.fork(count=count))
        return await _adopt_fork_children(session.backend, session, raw_children, count)

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        if snapshot.kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("E2B native restore requires a full-state snapshot.")
        restore_spec = spec or SandboxSpec(source=SandboxSource.template(snapshot.snapshot_id))
        restore_spec = replace(restore_spec, source=SandboxSource.template(snapshot.snapshot_id))
        client = await backend.client_factory.create(restore_spec)
        return E2BSession(backend, client, spec=restore_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        await self.client_factory.delete_snapshot(snapshot.snapshot_id)


class ProviderStateDriver(E2BHibernateDriver):
    """Full-state provider APIs shared by AgentEnv and CubeSandbox.

    Every operation here has a documented E2B SDK equivalent, so the SDK is what is
    called: pause, connect (which resumes), fork, and create_snapshot are all public SDK
    methods with published semantics, and reaching past them to a provider REST shape
    would be guessing at a contract nobody publishes.

    The declared feature set is a parameter rather than a guess, because a provider
    that only restores a snapshot must not be handed a spec that asked for more.
    """

    def __init__(
        self,
        client_factory: E2BClientFactory,
        *,
        native_fork: bool,
        fork_timeout_s: float | None = None,
        resume_timeout_s: float | None = None,
        snapshot_name_prefix: str = "psrl",
        snapshot_request_timeout_s: float | None = None,
        resume_level: ResumeLevel | None = None,
        extra_features: frozenset[SandboxFeature] = frozenset(),
    ) -> None:
        self.client_factory = client_factory
        self.native_fork = native_fork
        # The provider renews the child's TTL from the fork, so the workflow's own
        # deadline is what has to travel with the request.
        self.fork_timeout_s = fork_timeout_s
        # A resume renews the sandbox TTL from the moment it resumes, so a squatted
        # sandbox does not expire against its pre pause deadline.
        self.resume_timeout_s = resume_timeout_s
        self.snapshot_name_prefix = snapshot_name_prefix
        # A snapshot captures memory, so the provider can take minutes over it and the
        # SDK's own request timeout is far shorter than that.
        self.snapshot_request_timeout_s = snapshot_request_timeout_s
        self.resume_level = resume_level
        self.extra_features = extra_features

    @property
    def capabilities(self) -> SandboxCapabilities:
        features = {
            SandboxFeature.HIBERNATE,
            SandboxFeature.FULL_STATE_SNAPSHOT,
            SandboxFeature.RESTORE,
        }
        if self.native_fork:
            features.add(SandboxFeature.NATIVE_FORK)
        features |= self.extra_features
        level = self.resume_level if SandboxFeature.RESUME_ANYWHERE in features else None
        return SandboxCapabilities(frozenset(features), resume_level=level)

    async def resume(self, session: E2BSession) -> None:
        """Resume by reconnecting, which the provider documents as auto-resuming.

        There is no separate resume call: connecting to a paused sandbox resumes it, and
        the data plane has to be rebuilt anyway because the connection the pause invalidated
        is the one it held. The caller's requested life travels with the reconnect, because
        the provider restarts the expiry clock here and the sandbox would otherwise die
        against its pre-pause deadline.
        """
        timeout = self.resume_timeout_s if self.resume_timeout_s is not None else _requested_life_s(session)
        await session.refresh_transport(timeout=None if timeout is None else max(1, math.ceil(timeout)))

    async def pause(self, session: E2BSession, mode: PauseMode) -> None:
        if mode != PauseMode.HIBERNATE:
            raise RuntimeError("Provider pause has hibernation semantics, not process freeze semantics.")
        await _await(session.client.pause())

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef:
        """Capture the sandbox, named so a restore can address it by name.

        The provider pauses the sandbox while it captures, and returns both the id and the
        names it assigned. The name travels in the metadata because a qualified name
        survives a re-import where a bare id does not.
        """
        if kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("Provider snapshots contain filesystem, process, and memory state.")
        name = f"{self.snapshot_name_prefix}-{session.sandbox_id}-{uuid.uuid4().hex[:12]}"
        options: dict[str, Any] = {}
        if self.snapshot_request_timeout_s is not None:
            options["request_timeout"] = self.snapshot_request_timeout_s
        info = await _await(session.client.create_snapshot(name=name, **options))
        return SnapshotRef(
            backend=session.backend.name,
            snapshot_id=str(info.snapshot_id),
            kind=SnapshotKind.FULL_STATE,
            metadata={"psrl.snapshot.name": name, "names": list(info.names)},
            resume_level=self.resume_level,
        )

    async def fork(self, session: E2BSession, count: int) -> Sequence[SandboxSession]:
        """Branch the sandbox through the SDK's own fork.

        A partial result fails the whole group. The group size is a configuration value, so
        a short group is not a smaller group: it is a group the caller cannot form an
        advantage over correctly. Children that were adopted are destroyed before the
        failure surfaces, so a refused group leaves nothing behind.

        The SDK returns one entry per requested fork, each a sandbox or the exception that
        fork failed with, so there is no batch size to negotiate: the provider takes the
        whole request and reports per entry.
        """
        if not self.native_fork:
            raise NotImplementedError("This provider uses snapshot and restore rather than native fork.")
        require_batch_count(count)
        # A fork starts the child's provider clock over, so the group's children get the
        # life the members asked for instead of the provider's default.
        timeout = self.fork_timeout_s if self.fork_timeout_s is not None else _requested_life_s(session)
        raw_children = await _await(
            session.client.fork(count=count, timeout=None if timeout is None else math.ceil(timeout))
        )
        return await _adopt_fork_children(session.backend, session, raw_children, count)

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        if snapshot.kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("Provider restore requires a full-state snapshot.")
        restore_spec = spec or SandboxSpec(source=SandboxSource.template(snapshot.snapshot_id))
        restore_spec = replace(restore_spec, source=SandboxSource.template(snapshot.snapshot_id))
        client = await backend.client_factory.create(restore_spec)
        return E2BSession(backend, client, spec=restore_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete the snapshot object, which is what a restore addresses.

        A published template is a separate object derived from a snapshot, so deleting
        the snapshot is the authoritative removal rather than deleting one of its
        derivatives and leaving the source behind.
        """
        await self.client_factory.delete_snapshot(snapshot.snapshot_id)


class AgentEnvTemplateBuilder:
    """Verifies or imports the snapshot-backed template a spec will acquire from.

    A template is what makes an acquire a warm claim: the sandbox resumes a captured
    state instead of booting a guest. Importing one is asynchronous at the provider,
    so this polls the watch status inside a deadline and turns a failed build into an
    admission failure rather than a create failure.

    A Dockerfile build needs a build context, which a spec does not carry, so a
    context build stays a recipe-owned step through the provider's own CLI. This
    covers the two paths a rollout actually needs: verify a named template, and
    import one from the image a task already names.
    """

    def __init__(
        self,
        control: ProviderControlClient,
        *,
        poll_interval_s: float = 5.0,
        build_timeout_s: float = 1800.0,
        auto_import: bool = True,
    ) -> None:
        if poll_interval_s <= 0 or build_timeout_s <= 0:
            raise ValueError("AgentEnv template polling and build timeout must be greater than zero.")
        self.control = control
        self.poll_interval_s = poll_interval_s
        self.build_timeout_s = build_timeout_s
        self.auto_import = auto_import

    async def ensure(self, spec: SandboxSpec) -> None:
        """Make sure the template this spec names exists and is ready.

        Raises:
            RuntimeError: When the template is missing and cannot be imported, or when
                the provider reports a failed build.
        """
        if spec.source.kind == SandboxSourceKind.TEMPLATE:
            await self._await_ready(spec.source.reference)
            return
        if spec.source.kind != SandboxSourceKind.IMAGE or not self.auto_import:
            return
        await self._import_image(spec)

    async def _status(self, name: str) -> str | None:
        try:
            response = await self.control.request("GET", f"/templates/{name}", expected=(200,))
        except RuntimeError:
            return None
        return str(response.get("status") or response.get("state") or "ready").lower()

    async def _await_ready(self, name: str) -> None:
        deadline = time.monotonic() + self.build_timeout_s
        while True:
            status = await self._status(name)
            if status is None:
                raise RuntimeError(
                    f"AgentEnv template {name!r} does not exist. Build or import it before the rollout, or "
                    "let the backend import it from the task image."
                )
            if status == "ready":
                return
            if status == "error":
                raise RuntimeError(f"AgentEnv template {name!r} failed to build. Fix the image or the build.")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"AgentEnv template {name!r} was still {status!r} after {self.build_timeout_s:g}s, so the "
                    "rollout cannot start from it."
                )
            await asyncio.sleep(self.poll_interval_s)

    async def _import_image(self, spec: SandboxSpec) -> None:
        name = template_name_for_image(spec.source.reference)
        status = await self._status(name)
        if status == "ready":
            return
        if status is not None and status != "error":
            await self._await_ready(name)
            return
        await self.control.request(
            "POST",
            "/templates",
            expected=(201, 202),
            payload={"image": spec.source.reference, "name": name},
        )
        await self._await_ready(name)


def template_name_for_image(image: str) -> str:
    """
    Derive a stable template name from an image reference.

    Stable so a second prepare for the same image verifies what the first one built
    instead of importing a duplicate.
    """
    digest = hashlib.sha256(image.encode()).hexdigest()[:16]
    return f"psrl-image-{digest}"


class AgentEnvStateDriver(ProviderStateDriver):
    """AgentEnv's native state surface.

    Every entry here is a provider feature PSRL exposes rather than reimplements:
    fork, snapshot-backed templates, a resume that lands on any node, lazy image
    layers, provider-managed volumes, and a warm claim.
    """

    def __init__(
        self,
        client_factory: E2BClientFactory,
        *,
        fork_timeout_s: float | None = None,
        resume_timeout_s: float | None = None,
        snapshot_name_prefix: str = "psrl",
        snapshot_request_timeout_s: float | None = None,
        warm_pool: bool = True,
    ) -> None:
        features = {
            SandboxFeature.RESUME_ANYWHERE,
            SandboxFeature.TEMPLATE_BUILD,
            SandboxFeature.IMAGE_ON_DEMAND,
        }
        if warm_pool:
            # A snapshot-backed template is a warm claim: the sandbox resumes a
            # captured state instead of booting a fresh guest.
            features.add(SandboxFeature.WARM_POOL)
        super().__init__(
            client_factory,
            native_fork=True,
            fork_timeout_s=fork_timeout_s,
            resume_timeout_s=resume_timeout_s,
            snapshot_name_prefix=snapshot_name_prefix,
            snapshot_request_timeout_s=snapshot_request_timeout_s,
            resume_level=ResumeLevel.FULL_STATE,
            extra_features=frozenset(features),
        )

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        """Restore a captured sandbox, which accepts only a narrow override list.

        The provider allows a lifetime, user metadata, the network policy, and its own
        secure-access and auto-pause switches on a snapshot restore. Environment
        variables, volumes, and mounts are fixed by the snapshot and the provider
        *rejects* a restore that passes them, so they are cleared here rather than sent
        for the provider to refuse. That is the provider's semantics, not a dropped
        request: a capture already carries the environment it was taken with.

        A restore addresses the snapshot by the name it was captured under when one is
        recorded, because a qualified name survives a re-import where a bare id does not.
        """
        reference = str(snapshot.metadata.get("psrl.snapshot.name") or snapshot.snapshot_id)
        restore_spec = spec or SandboxSpec(source=SandboxSource.template(reference))
        restore_spec = replace(
            restore_spec,
            source=SandboxSource.template(reference),
            resources=type(restore_spec.resources)(),
            env={},
            volumes=(),
            mounts=(),
        )
        client = await backend.client_factory.create(restore_spec)
        return E2BSession(backend, client, spec=restore_spec)


class CubeSandboxStateDriver(ProviderStateDriver):
    """CubeSandbox snapshot and restore semantics."""

    def __init__(self, client_factory: E2BClientFactory) -> None:
        super().__init__(client_factory, native_fork=False)


class E2BBackend(SandboxBackend):
    """Backend for E2B-compatible SDK endpoints.

    TODO(claude): Raise `SandboxProvisionError` when a create can leave a provider sandbox
    behind. Confirming which SDK failures do that needs a live endpoint, so until then a
    lost create response leaks until the provider's own idle timeout.
    """

    def __init__(
        self,
        *,
        name: str = "e2b",
        api_key: str | None = None,
        api_url: str | None = None,
        create_kwargs: Mapping[str, Any] | None = None,
        connect_kwargs: Mapping[str, Any] | None = None,
        client_factory: E2BClientFactory | None = None,
        state_driver: E2BStateDriver | None = None,
    ) -> None:
        self._name = name
        self.client_factory = client_factory or E2BSDKClientFactory(
            api_key=api_key,
            api_url=api_url,
            create_kwargs=create_kwargs,
            connect_kwargs=connect_kwargs,
        )
        self.state_driver = state_driver or E2BNativeStateDriver(self.client_factory)
        self.metrics = SandboxMetrics()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.state_driver.capabilities

    def metrics_snapshot(self) -> SandboxMetricsSnapshot:
        return self.metrics.snapshot()

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        _validated_ttl(spec.lifetime_timeout_s or spec.idle_timeout_s)
        try:
            with self.metrics.measure("create"):
                client = await self.client_factory.create(spec)
        except Exception as exc:
            raise RuntimeError(f"Could not create sandbox {spec!r}: {exc}.") from exc
        self.metrics.session_started()
        return E2BSession(self, client, spec=spec)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        try:
            with self.metrics.measure("connect"):
                client = await self.client_factory.connect(sandbox_id)
        except Exception as exc:
            raise RuntimeError(f"Could not connect to sandbox {sandbox_id!r}: {exc}.") from exc
        self.metrics.session_started()
        return E2BSession(self, client, sandbox_id=sandbox_id)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        if spec is not None:
            _validated_ttl(spec.lifetime_timeout_s or spec.idle_timeout_s)
        self.capabilities.require(SandboxFeature.RESTORE)
        with self.metrics.measure("restore"):
            session = await self.state_driver.restore(self, snapshot, spec)
        self.metrics.session_started()
        return session

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        with self.metrics.measure("delete_snapshot"):
            await self.state_driver.delete_snapshot(snapshot)

    async def shutdown(self) -> None:
        await self.client_factory.close()
        await self.state_driver.close()


class AgentEnvBackend(E2BBackend):
    """AgentEnv backend: the provider's SDK for every sandbox-scoped call.

    Two things stay outside the SDK, and both are template management rather than sandbox
    lifecycle: importing or verifying the template an image resolves to, and the listing
    that reclaims a create whose reply was lost.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        client_factory: E2BClientFactory | None = None,
        state_driver: E2BStateDriver | None = None,
        control_client: ProviderControlClient | None = None,
        templates: AgentEnvTemplateBuilder | None = None,
        template_build_timeout_s: float = 1800.0,
        snapshot_request_timeout_s: float = 300.0,
        **kwargs: Any,
    ) -> None:
        self.templates: AgentEnvTemplateBuilder | None = templates
        if client_factory is None:
            # The template surface is not in the SDK, so it keeps its own client.
            control = control_client or ProviderControlClient(str(api_url or ""), api_key=api_key)
            self.control = control
            client_factory = AgentEnvClientFactory(api_url=api_url, api_key=api_key, **kwargs)
            state_driver = state_driver or AgentEnvStateDriver(
                client_factory, snapshot_request_timeout_s=snapshot_request_timeout_s
            )
            self.templates = self.templates or AgentEnvTemplateBuilder(
                control, build_timeout_s=template_build_timeout_s
            )
            kwargs = {}
        else:
            self.control = getattr(client_factory, "control", None)
            state_driver = state_driver or E2BHibernateDriver()
        super().__init__(name="agentenv", client_factory=client_factory, state_driver=state_driver, **kwargs)

    async def prepare(self, spec: SandboxSpec) -> None:
        """Make the template this spec will acquire from exist and be ready.

        A build that fails here is an admission failure rather than a create failure,
        which keeps a slow or broken build off the rollout critical path.
        """
        if self.templates is None:
            return
        with self.metrics.measure("prepare_template"):
            await self.templates.ensure(spec)

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        """Create, reclaiming a lost response rather than leaking a sandbox.

        A create that loses its reply can still have produced a sandbox. The metadata
        carries the idempotency key, so a filtered list finds it and reclaims it
        before the failure surfaces.
        """
        try:
            return await super().create(spec)
        except Exception:
            if spec.idempotency_key and self.control is not None:
                await self._reclaim_by_idempotency(spec)
            raise

    async def _reclaim_by_idempotency(self, spec: SandboxSpec) -> int:
        """
        Delete every sandbox this worker created for one idempotency key.
        """
        try:
            listed = await self.control.request(
                "GET",
                f"/sandboxes?metadata.psrl.idempotency_key={spec.idempotency_key}",
                expected=(200,),
            )
        except Exception:
            self.metrics.count("lost_create_reclaim_failed")
            return 0
        reclaimed = 0
        for entry in listed or ():
            sandbox_id = entry.get("sandboxID") or entry.get("sandbox_id")
            if not sandbox_id:
                continue
            try:
                await self.control.delete_idempotent(f"/sandboxes/{sandbox_id}")
                reclaimed += 1
            except Exception:
                continue
        if reclaimed:
            self.metrics.count("lost_create_reclaimed")
        return reclaimed


class CubeSandboxBackend(E2BBackend):
    """CubeSandbox backend with provider-native resources and snapshots."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        client_factory: E2BClientFactory | None = None,
        state_driver: E2BStateDriver | None = None,
        control_client: ProviderControlClient | None = None,
        **kwargs: Any,
    ) -> None:
        if client_factory is None:
            control = control_client or ProviderControlClient(str(api_url or ""), api_key=api_key)
            client_factory = CubeSandboxClientFactory(control=control, **kwargs)
            state_driver = state_driver or CubeSandboxStateDriver(client_factory)
            kwargs = {}
        else:
            state_driver = state_driver or E2BHibernateDriver()
        super().__init__(name="cubesandbox", client_factory=client_factory, state_driver=state_driver, **kwargs)


class E2BSession(SandboxSession):
    """One E2B-compatible SDK session."""

    def __init__(
        self,
        backend: E2BBackend,
        client: Any,
        *,
        sandbox_id: str | None = None,
        spec: SandboxSpec | None = None,
    ) -> None:
        self.backend = backend
        self.client = client
        self.sandbox_id = sandbox_id or str(client.sandbox_id)
        self._spec = spec
        self._command_count = 0
        self._status = SandboxStatus.RUNNING
        self._terminate_lock = asyncio.Lock()
        # The in-flight signal a fork and an idle pause both read. Commands serialize on it,
        # so taking it without waiting is what tells either of them that the guest is busy.
        self._exec_lock = asyncio.Lock()
        self._busy = False
        # Idle is two conditions, and this is the second. A session reporting no activity
        # at all is never idle, so the pause that releases a provider's compute never fires.
        self._last_activity_at = time.monotonic()

    @property
    def busy(self) -> bool:
        """
        Return whether a command is executing right now.
        """
        return self._busy

    @property
    def last_activity_at(self) -> float | None:
        """Return the time of the last command boundary, or the session's creation.

        Stamped at both the start and the return of a command, because a stamp written
        only on return stays stale for the whole of a long command and an idle pass
        reading it would pause a sandbox that is working.
        """
        return self._last_activity_at

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.backend.name, self.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.backend.capabilities

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def command_count(self) -> int:
        return self._command_count

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        # Commands serialize on the exec lock, which is also the in-flight signal a fork and
        # an idle pause read. Overlap would make both the guest state and the signal useless.
        async with self._exec_lock:
            self._command_count += 1
            self._busy = True
            self._last_activity_at = time.monotonic()
            try:
                return await self._exec_locked(command, cwd=cwd, env=env, timeout_s=timeout_s)
            finally:
                self._busy = False
                self._last_activity_at = time.monotonic()

    async def _exec_locked(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
    ) -> ExecResult:
        # A provider sandbox reports its own progress, and the SDK timeout is the only deadline
        # it can honor. A silence window would need enforcement inside a guest we do not own.
        kwargs: dict[str, Any] = {}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if env:
            kwargs["envs"] = dict(env)
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        with self.backend.metrics.measure("exec"):
            try:
                result = await _await(self.client.commands.run(command, **kwargs))
            except Exception as exc:
                if all(hasattr(exc, attribute) for attribute in ("exit_code", "stdout", "stderr")):
                    result = exc
                elif isinstance(exc, TimeoutError) or type(exc).__name__ in {
                    "TimeoutException",
                    "TimeoutError",
                }:
                    raise TimeoutError(f"E2B command exceeded {timeout_s} seconds.") from exc
                else:
                    raise
        # The result carries a message of its own when the command could not be run at
        # all, and dropping it would leave a failed command looking like a silent one.
        error = getattr(result, "error", None)
        stderr = str(getattr(result, "stderr", ""))
        if error:
            stderr = f"{stderr}\n{error}" if stderr else str(error)
        return ExecResult(
            exit_code=int(getattr(result, "exit_code", 0)),
            stdout=str(getattr(result, "stdout", "")),
            stderr=stderr,
        )

    async def read_bytes(self, path: str) -> bytes:
        with self.backend.metrics.measure("read_bytes"):
            data = await _await(self.client.files.read(path, format="bytes"))
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        return str(data).encode()

    async def write_bytes(self, path: str, data: bytes) -> None:
        with self.backend.metrics.measure("write_bytes"):
            await _await(self.client.files.write(path, data))

    async def status(self) -> SandboxStatus:
        if self._status in {SandboxStatus.PAUSED, SandboxStatus.TERMINATED}:
            return self._status
        with self.backend.metrics.measure("status"):
            is_running = await _await(self.client.is_running())
        return SandboxStatus.RUNNING if is_running else SandboxStatus.UNKNOWN

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._status == SandboxStatus.TERMINATED:
                return
            with self.backend.metrics.measure("terminate"):
                try:
                    await _await(self.client.kill())
                except Exception as exc:
                    if "not found" not in str(exc).lower():
                        raise
                await _close_client(self.client)
            self._status = SandboxStatus.TERMINATED
            self.backend.metrics.session_stopped()

    async def pause(self, mode: PauseMode) -> None:
        # Pausing with a command in flight would freeze the guest halfway through it, so the
        # lock is taken without waiting and a busy guest is reported instead.
        if not await acquire_nowait(self._exec_lock):
            raise SandboxBusyError(f"Provider sandbox {self.sandbox_id!r} has a command in flight, so it is not idle.")
        try:
            with self.backend.metrics.measure("pause"):
                await self.backend.state_driver.pause(self, mode)
        finally:
            self._exec_lock.release()
        self._status = SandboxStatus.PAUSED

    async def resume(self) -> None:
        with self.backend.metrics.measure("resume"):
            await self.backend.state_driver.resume(self)
        self._status = SandboxStatus.RUNNING

    async def refresh_transport(self, *, timeout: int | None = None) -> None:
        """Reconnect after a resume so stale TCP pools are never reused.

        Connecting is also what resumes a paused sandbox, and the provider only extends a
        life when the new value is longer, so a requested life travels here rather than in
        a separate renewal call.
        """
        old_client = self.client
        with self.backend.metrics.measure("refresh_transport"):
            self.client = await self.backend.client_factory.connect(self.sandbox_id, timeout=timeout)
            await _close_client(old_client)
        self._status = SandboxStatus.RUNNING

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        with self.backend.metrics.measure("snapshot"):
            return await self.backend.state_driver.snapshot(self, kind)

    async def fork(self, count: int = 1) -> Sequence[SandboxSession]:
        require_batch_count(count)
        # The source is briefly paused while its state is captured, so a fork must not overlap
        # a command. Finding the guest busy forks later rather than holding the group open.
        if not await acquire_nowait(self._exec_lock):
            raise SandboxBusyError(
                f"Provider sandbox {self.sandbox_id!r} has a command in flight, so it cannot be forked."
            )
        try:
            with self.backend.metrics.measure("fork"):
                return await self.backend.state_driver.fork(self, count)
        finally:
            self._exec_lock.release()

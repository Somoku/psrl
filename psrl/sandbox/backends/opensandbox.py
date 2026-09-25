"""OpenSandbox backend over its published lifecycle, execution, and egress planes.

OpenSandbox is not E2B compatible, so unlike AgentEnv it is a backend with its own
session type rather than a client factory under `E2BBackend`. It has a different
execution plane, a different state model, and a different image model.

Three properties make it worth a backend of its own.

- The same API serves a laptop and a cluster, so the local dev loop and the cluster
  run share one code path.
- Its pause releases all compute and its checkpoint restores on any host, which is
  the cross node resume primitive.
- It enforces a per-sandbox egress policy and a credential vault itself, so PSRL
  writes policy and never reimplements enforcement.

Two rules shape the code here.

- **A data endpoint is transport state, never configuration.** The `execd` address
  is resolved per sandbox and re-resolved after every resume, because a checkpoint
  and resume can move the sandbox.
- **A pause is asynchronous.** The control plane reports intent, so the backend
  polls until the state settles rather than assuming the transition happened.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from urllib.parse import quote

import aiohttp

from psrl.sandbox.async_utils import acquire_nowait
from psrl.sandbox.core import (
    ExecMode,
    ExecResult,
    PauseMode,
    ResumeLevel,
    SandboxBackend,
    SandboxBusyError,
    SandboxCapabilities,
    SandboxDiagnostics,
    SandboxExitReason,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
    resolve_credential,
)

psrl_logger = logging.getLogger(__file__)

LIFECYCLE_API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"

# The execution plane's port. Every sandbox gets `execd` on this fixed container port,
# so a caller resolves it without configuration.
DEFAULT_EXECD_PORT = 44772

# The egress sidecar's port. The sidecar serves the network policy and the credential
# vault, and a sandbox endpoint for this port is how a caller reaches them.
DEFAULT_EGRESS_PORT = 18080

# Provider states that mean a caller cannot command the sandbox yet.
_TRANSIENT_STATES = frozenset({"pending", "pausing", "resuming", "stopping"})

_STATUS_MAP = {
    "running": SandboxStatus.RUNNING,
    "paused": SandboxStatus.PAUSED,
    "terminated": SandboxStatus.TERMINATED,
    "failed": SandboxStatus.UNKNOWN,
    "pending": SandboxStatus.UNKNOWN,
    "pausing": SandboxStatus.UNKNOWN,
    "resuming": SandboxStatus.UNKNOWN,
    "stopping": SandboxStatus.UNKNOWN,
}

# Snapshot states, as the provider reports them in `status.state`.
_SNAPSHOT_READY = "ready"
_SNAPSHOT_FAILED = "failed"

# What OpenSandbox can do, declared once. There is no native fork, and pause is a
# hibernation that releases compute.
#
# Snapshots commit the sandbox's filesystem, so a restore brings the workspace back and
# every process starts fresh. FULL_STATE_SNAPSHOT would read as a stronger promise.
_OPEN_SANDBOX_FEATURES = frozenset(
    {
        SandboxFeature.HIBERNATE,
        SandboxFeature.FILESYSTEM_SNAPSHOT,
        SandboxFeature.RESTORE,
        SandboxFeature.RESUME_ANYWHERE,
        SandboxFeature.IMAGE_BLOCK_DELIVERY,
        SandboxFeature.VOLUME,
        SandboxFeature.EGRESS_POLICY,
        SandboxFeature.CREDENTIAL_INJECTION,
    }
)


class OpenSandboxError(RuntimeError):
    """
    Raised when the provider refused a request rather than the workload failing.
    """


@dataclass(frozen=True)
class SandboxEndpoint:
    """One resolved sandbox endpoint: where it is, and the headers it requires.

    The provider returns the headers a caller must forward, so they travel with the
    address rather than being reconstructed. Both are transport state: a resume can
    move the sandbox, and a moved sandbox can require different headers.
    """

    address: str
    headers: Mapping[str, str] = field(default_factory=dict)

    def authorization(self, name: str) -> str | None:
        """
        Return one required header's value, case-insensitively.
        """
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


class OpenSandboxTransport(Protocol):
    """
    The HTTP surface the backend needs, so a test can drive it without a server.
    """

    async def request(
        self,
        method: str,
        url: str,
        *,
        expected: Sequence[int],
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def read_file(self, url: str, *, headers: Mapping[str, str] | None = None) -> bytes: ...

    async def upload_file(
        self,
        url: str,
        data: bytes,
        *,
        metadata: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> None: ...

    async def stream_lines(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


class AiohttpTransport:
    """
    One connection pool for one plane.
    """

    def __init__(self, *, request_timeout_s: float = 300.0, connection_limit: int = 64) -> None:
        self.request_timeout_s = request_timeout_s
        self.connection_limit = connection_limit
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
                connector=aiohttp.TCPConnector(limit=self.connection_limit),
            )
        return self._session

    async def request(
        self,
        method: str,
        url: str,
        *,
        expected: Sequence[int],
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        session = await self._get_session()
        async with session.request(method, url, json=payload, headers=dict(headers or {})) as response:
            body = await response.text()
            if response.status not in expected:
                raise OpenSandboxError(f"OpenSandbox returned HTTP {response.status} for {url}: {body.strip()}.")
            if not body.strip():
                return None
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body

    async def read_file(self, url: str, *, headers: Mapping[str, str] | None = None) -> bytes:
        session = await self._get_session()
        async with session.get(url, headers=dict(headers or {})) as response:
            if response.status != 200:
                raise OpenSandboxError(
                    f"OpenSandbox file download returned HTTP {response.status} for {url}: "
                    f"{(await response.text()).strip()}."
                )
            return await response.read()

    async def upload_file(
        self,
        url: str,
        data: bytes,
        *,
        metadata: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Upload one file as the provider's two-part multipart body.

        The metadata part is JSON and comes first, then the file bytes, which is the
        order the server reads them in. aiohttp sets the multipart boundary, so the
        caller's headers must not carry a content type.

        Both parts are sent with a filename. The server reads `metadata` as a file part
        and answers `INVALID_FILE_METADATA` for a plain field, so the filename is what
        makes the request well formed rather than a cosmetic detail.
        """
        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field(
            "metadata",
            json.dumps(dict(metadata)),
            filename="metadata.json",
            content_type="application/json",
        )
        form.add_field(
            "file",
            data,
            filename=str(metadata.get("path") or "file").rsplit("/", 1)[-1] or "file",
            content_type="application/octet-stream",
        )
        async with session.post(url, data=form, headers=dict(headers or {})) as response:
            if response.status not in (200, 201, 204):
                raise OpenSandboxError(
                    f"OpenSandbox file upload returned HTTP {response.status} for {url}: "
                    f"{(await response.text()).strip()}."
                )

    async def stream_lines(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        async with session.request(method, url, json=payload, headers=dict(headers or {})) as response:
            if response.status != 200:
                raise OpenSandboxError(
                    f"OpenSandbox stream returned HTTP {response.status} for {url}: {(await response.text()).strip()}."
                )
            async for line in response.content:
                if line.strip():
                    yield line.decode(errors="replace").strip()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


@dataclass(frozen=True)
class OpenSandboxConfig:
    """
    Endpoints, credentials, and provider identity for one deployment.
    """

    api_url: str
    api_key: str | None = None
    # Template and snapshot names are namespaced so two runs cannot collide.
    namespace: str = "psrl"
    execd_port: int = DEFAULT_EXECD_PORT
    # The egress sidecar's port. The network policy and the credential vault both live
    # on that sidecar, reached by resolving the sandbox endpoint for this port.
    egress_port: int = DEFAULT_EGRESS_PORT
    # Seconds to wait for an asynchronous pause to settle.
    pause_timeout_s: float = 300.0
    # Seconds to wait for a create's endpoint to be published and answer.
    ready_timeout_s: float = 120.0
    poll_interval_s: float = 2.0
    # A template build is asynchronous, and a failed one has to fail admission.
    template_timeout_s: float = 1800.0
    # A snapshot is accepted before its artifact exists, so a capture is polled to
    # readiness. Snapshotting a large workspace is slow by nature.
    snapshot_timeout_s: float = 900.0
    # The entry process a sandbox starts with. The provider requires an entrypoint with an
    # image and defaults it to a sleeping keepalive that keeps the sandbox alive for commands.
    entrypoint: tuple[str, ...] = ("tail", "-f", "/dev/null")
    # The provider requires resource limits on every non-template create, so a spec asking
    # for nothing still sends these. They match the provider's own SDK defaults.
    default_resources: Mapping[str, str] = field(default_factory=lambda: {"cpu": "1", "memory": "2Gi"})
    # The provider requires a timeout on a template-mode create. A spec that declares no
    # lifetime falls back to this rather than to the provider's own expiry.
    default_timeout_s: float = 3600.0
    # The isolation boundary this deployment provides. It is a server property rather than a
    # per-sandbox request, so a spec requiring a runtime is refused unless the deployment has one.
    isolation_runtime: str | None = None
    # An S3-compatible target for golden-image artifacts. Setting it enables the template
    # path, which needs a fast-sandbox runtime (the provider answers 501 elsewhere).
    template_publish: str | None = None
    # The on-demand pool a warm sandbox is claimed from. The operator pre-creates it, and
    # its pods carry a fixed shape the provider will not let a request redefine.
    warm_pool_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.api_url:
            raise ValueError("OpenSandbox api_url is required.")
        if self.poll_interval_s <= 0 or self.pause_timeout_s <= 0 or self.template_timeout_s <= 0:
            raise ValueError("OpenSandbox polling and timeouts must be greater than zero.")
        if self.snapshot_timeout_s <= 0:
            raise ValueError("OpenSandbox snapshot_timeout_s must be greater than zero.")
        if self.ready_timeout_s <= 0:
            raise ValueError("OpenSandbox ready_timeout_s must be greater than zero.")
        if not self.entrypoint:
            raise ValueError("OpenSandbox requires an entrypoint, because the provider requires one with an image.")
        if self.default_timeout_s < _MIN_SANDBOX_TIMEOUT_S:
            raise ValueError(
                f"OpenSandbox default_timeout_s must be at least {_MIN_SANDBOX_TIMEOUT_S:g}s, because the "
                "provider rejects a shorter sandbox timeout."
            )
        for name, value in (("template_publish", self.template_publish), ("warm_pool_ref", self.warm_pool_ref)):
            if value is not None and not value.strip():
                raise ValueError(f"OpenSandbox {name} cannot be empty when set.")

    @property
    def base_url(self) -> str:
        """
        Return the lifecycle plane's base URL.
        """
        return self.api_url.rstrip("/")

    @classmethod
    def from_value(cls, value: OpenSandboxConfig | Mapping[str, Any]) -> OpenSandboxConfig:
        """
        Normalize a Hydra mapping into immutable configuration.
        """
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


# The provider reports memory in MiB, the portable contract in bytes.
_BYTES_PER_MIB = 1024 * 1024

# The provider rejects a sandbox timeout below a minute.
_MIN_SANDBOX_TIMEOUT_S = 60.0


def _provider_state(payload: Mapping[str, Any]) -> str:
    """Return a sandbox's lifecycle state from its inspect payload, lowercased.

    The state is nested inside `status`, and a payload carrying neither is reported as
    unknown rather than as a state, so a caller never treats a missing field as ready.
    """
    status = payload.get("status")
    if isinstance(status, Mapping):
        state = status.get("state")
    else:
        state = None
    return str(state or "unknown").lower()


def _snapshot_state(payload: Mapping[str, Any]) -> str:
    """
    Return a snapshot's lifecycle state from its record, lowercased.
    """
    status = payload.get("status")
    if isinstance(status, Mapping):
        state = status.get("state")
    else:
        state = None
    return str(state or "unknown").lower()


def _template_phase(payload: Mapping[str, Any]) -> str:
    """
    Return a template build's phase from its record, lowercased.
    """
    status = payload.get("status")
    if isinstance(status, Mapping):
        phase = status.get("phase")
    else:
        phase = None
    return str(phase or "unknown").lower()


def _requested_timeout_s(spec: SandboxSpec) -> float | None:
    """Return the lifetime a spec asked for, or None when it asked for none.

    The provider's two ways to express this are an absolute second count or an omitted
    field, and omitting it is what turns off automatic expiry. A request below the
    provider's floor is refused here rather than by the server, because a rejected
    create reads as a provider fault rather than a configuration one.
    """
    requested = spec.lifetime_timeout_s or spec.idle_timeout_s
    if requested is None:
        return None
    if requested < _MIN_SANDBOX_TIMEOUT_S:
        raise OpenSandboxError(
            f"OpenSandbox requires a sandbox timeout of at least {_MIN_SANDBOX_TIMEOUT_S:g}s, so the "
            f"requested {requested:g}s cannot be honoured."
        )
    return requested


def _resource_limits(spec: SandboxSpec, defaults: Mapping[str, str]) -> dict[str, str]:
    """Map portable resources onto the provider's `resourceLimits`.

    The provider takes a map of Kubernetes-style quantities, so every value is a string:
    `cpu` in millicores, `memory` in bytes, `gpu` as a device count. There is no
    documented disk quantity, so a disk request is refused rather than dropped.

    Args:
        spec (SandboxSpec): The portable request.
        defaults (Mapping[str, str]): What to send when the spec asks for nothing, because
            the provider requires the field on every create it is not rejecting outright.

    Returns:
        dict[str, str]: The provider's resource map.
    """
    if spec.resources.disk_mb is not None:
        raise OpenSandboxError(
            "OpenSandbox documents no disk resource quantity, so a disk request cannot be honoured. "
            "Drop resources.disk_mb for this backend or use one that accounts disk."
        )
    limits = dict(defaults)
    cpu = spec.resources.cpu_count
    if cpu is not None:
        # A whole core is a valid quantity on its own. A fraction has to be millicores,
        # because `1.5` is not a Kubernetes quantity.
        limits["cpu"] = str(int(cpu)) if float(cpu).is_integer() else f"{int(round(cpu * 1000))}m"
    if spec.resources.memory_mb is not None:
        limits["memory"] = f"{spec.resources.memory_mb}Mi"
    if spec.resources.gpu_count:
        limits["gpu"] = str(spec.resources.gpu_count)
    return limits


def _metadata_payload(spec: SandboxSpec) -> dict[str, str]:
    """Render the sandbox metadata, refusing a key the provider reserves.

    A key under `opensandbox.io/` is system managed and the server rejects the request,
    so the whole create would fail for a reason the caller cannot see.
    """
    reserved = sorted(key for key in spec.metadata if key.startswith("opensandbox.io/"))
    if reserved:
        raise OpenSandboxError(
            f"Sandbox metadata {reserved} uses the provider's reserved `opensandbox.io/` prefix, which the "
            "server rejects."
        )
    return {
        **dict(spec.metadata),
        **({"psrl.idempotency_key": spec.idempotency_key} if spec.idempotency_key else {}),
    }


def _stream_event(line: str) -> dict[str, Any] | None:
    """Parse one frame of the provider's event stream, or return None.

    The provider documents `text/event-stream` carrying its own event object as JSON. An
    unparsable frame is protocol noise rather than output: folding it into stdout would
    invent content the workload never wrote.
    """
    frame = line[len("data:") :].strip() if line.startswith("data:") else line.strip()
    if not frame:
        return None
    try:
        parsed = json.loads(frame)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_text(event: Mapping[str, Any]) -> str:
    """
    Render an error event, which carries a name, a value, and a traceback.
    """
    error = event.get("error") or {}
    if not isinstance(error, Mapping):
        return str(event.get("text") or error)
    parts = [str(error.get("ename") or ""), str(error.get("evalue") or "")]
    traceback = error.get("traceback")
    if isinstance(traceback, list):
        parts.append("\n".join(str(entry) for entry in traceback))
    return "\n".join(part for part in parts if part)


def _result_text(event: Mapping[str, Any]) -> str:
    """
    Render a result event, whose body is a MIME map.
    """
    results = event.get("results")
    if not isinstance(results, Mapping):
        return ""
    return str(results.get("text/plain") or "")


def _status_from(value: Any) -> SandboxStatus:
    """
    Map a provider state onto the portable one.
    """
    return _STATUS_MAP.get(str(value).lower(), SandboxStatus.UNKNOWN)


def _plane_base_url(address: str, port: int) -> str:
    """Return the base a plane's own API is served from.

    A resolved endpoint may end in the lifecycle server's `/proxy/<port>` route, which
    reaches a service inside the sandbox rather than being the plane's root. Keeping it
    would send every request to `/proxy/<port>/session` instead of `/session`.

    Args:
        address (str): The resolved endpoint, with a scheme.
        port (int): The container port that was resolved.

    Returns:
        str: The address to build this plane's request URLs from.
    """
    suffix = f"/proxy/{port}"
    trimmed = address.rstrip("/")
    if trimmed.endswith(suffix):
        return trimmed[: -len(suffix)]
    return trimmed


class OpenSandboxControlClient:
    """The lifecycle plane: create, pause, resume, snapshot, template, policy."""

    def __init__(
        self,
        config: OpenSandboxConfig,
        *,
        transport: OpenSandboxTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or AiohttpTransport()

    def _headers(self) -> dict[str, str]:
        return {LIFECYCLE_API_KEY_HEADER: self.config.api_key} if self.config.api_key else {}

    async def request(
        self,
        method: str,
        path: str,
        *,
        expected: Sequence[int] = (200,),
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """
        Send one lifecycle request.
        """
        merged = self._headers()
        merged.update(dict(headers or {}))
        return await self.transport.request(
            method,
            f"{self.config.base_url}{path}",
            expected=expected,
            payload=payload,
            headers=merged,
        )

    async def create(
        self,
        spec: SandboxSpec,
        *,
        source: Mapping[str, Any] | None = None,
        template: str | None = None,
        pool_ref: str | None = None,
        network_policy: Mapping[str, Any] | None = None,
        credential_proxy: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        """Create a sandbox from an image, a snapshot, a prepared template, or a pool.

        The provider constrains four shapes differently, and the difference is not
        cosmetic:

        - **An image** is the general shape. An entrypoint is required, the resource
          request travels with it, and a timeout is optional (omitting it disables
          automatic expiry). The image is an object with a `uri`, not a bare string.
        - **A template** is a golden image whose workload shape is fixed by its build, so
          the provider rejects an entrypoint, env, resources, volumes, the credential
          proxy, and lifecycle hooks outright and **requires** a timeout.
        - **A snapshot** restores a captured sandbox, so its entrypoint is optional.
        - **A pool** hands out a pre-created pod, so the provider rejects a snapshot, a
          network policy, a platform constraint, volumes, and the credential proxy
          beside it rather than ignoring them.

        There is no per-sandbox workdir field, so `spec.workdir` reaches the sandbox as
        the `cwd` of each command, and no per-sandbox isolation-runtime field, so that
        boundary is a deployment property rather than a request one.
        """
        payload: dict[str, Any] = {"metadata": _metadata_payload(spec)}
        timeout_s = _requested_timeout_s(spec)
        # A prepared golden-image template and a spec that names one are the same shape,
        # so they resolve to one id before the branches below.
        template_id = template
        if template_id is None and spec.source.kind == SandboxSourceKind.TEMPLATE:
            template_id = spec.source.reference
        if pool_ref is not None:
            # The provider rejects anything that would redefine a pool pod's shape, so
            # refusing here names the caller's mistake instead of surfacing its 400.
            conflicting = [
                name
                for name, value in (
                    ("source", source),
                    ("templateId", template_id),
                    ("networkPolicy", network_policy),
                    ("credentialProxy", credential_proxy or None),
                    ("volumes", spec.volumes or None),
                )
                if value
            ]
            if conflicting:
                raise OpenSandboxError(
                    f"An OpenSandbox pool allocation cannot carry {conflicting}, because the pool's pod "
                    "already defines them. Use an image or template create for this spec."
                )
            payload["extensions"] = {"poolRef": pool_ref}
        elif source is not None:
            payload.update(dict(source))
            # A snapshot fixes the captured process, so the entrypoint is left to the provider's
            # own default, but resource limits are still a request concern the provider requires.
            payload["resourceLimits"] = _resource_limits(spec, self.config.default_resources)
            environment = dict(spec.env if env is None else env)
            if environment:
                payload["env"] = environment
        elif template_id is not None:
            payload["templateId"] = template_id
            payload["timeout"] = timeout_s or self.config.default_timeout_s
        elif spec.source.kind == SandboxSourceKind.IMAGE:
            payload["image"] = {"uri": spec.source.reference}
            payload["entrypoint"] = list(self.config.entrypoint)
            payload["resourceLimits"] = _resource_limits(spec, self.config.default_resources)
            environment = dict(spec.env if env is None else env)
            if environment:
                payload["env"] = environment
            if spec.volumes:
                payload["volumes"] = [
                    {"name": volume.name, "mountPath": volume.target, "readOnly": volume.read_only}
                    for volume in spec.volumes
                ]
        else:
            raise OpenSandboxError(
                f"OpenSandbox does not support source kind {spec.source.kind.value!r}. A restore names a "
                "snapshot, a warm start names a template, and a cold start names an image."
            )
        # The policy is allowed in every shape but a pool allocation, and it rides the
        # request rather than being patched in, because a vault needs an active sandbox.
        if network_policy is not None:
            payload["networkPolicy"] = dict(network_policy)
        if credential_proxy:
            # Explicit opt-in for the transparent MITM the vault injects through.
            payload["credentialProxy"] = {"enabled": True}
        if timeout_s is not None and "timeout" not in payload:
            payload["timeout"] = timeout_s
        response = await self.request("POST", "/v1/sandboxes", expected=(200, 201, 202), payload=payload)
        return response or {}

    async def inspect(self, sandbox_id: str) -> Mapping[str, Any]:
        """
        Return one sandbox's current state.
        """
        response = await self.request("GET", f"/v1/sandboxes/{sandbox_id}")
        return response or {}

    async def state(self, sandbox_id: str) -> str:
        """Return one sandbox's provider state, lowercased.

        The lifecycle state sits inside a status object rather than at the top level of
        the sandbox, so reading the wrong key yields a status object whose string form
        matches no known state and every caller then waits for a `running` that never
        arrives.
        """
        inspection = await self.inspect(sandbox_id)
        return _provider_state(inspection)

    async def delete(self, sandbox_id: str) -> None:
        """
        Delete a sandbox, treating a missing one as already gone.
        """
        try:
            await self.request("DELETE", f"/v1/sandboxes/{sandbox_id}", expected=(200, 202, 204))
        except OpenSandboxError as exc:
            if "404" not in str(exc):
                raise

    async def pause(self, sandbox_id: str) -> None:
        """Pause a sandbox and wait for the state to settle.

        The control plane accepts the intent and reports `Pausing` before it is
        paused, so returning immediately would let a caller command a sandbox that is
        about to stop accepting commands.
        """
        await self.request("POST", f"/v1/sandboxes/{sandbox_id}/pause", expected=(200, 202, 204))
        await self.await_state(sandbox_id, {"paused"}, failed_states={"failed"})

    async def resume(self, sandbox_id: str) -> None:
        """
        Resume a paused sandbox and wait until it accepts commands.
        """
        await self.request("POST", f"/v1/sandboxes/{sandbox_id}/resume", expected=(200, 202, 204))
        await self.await_state(sandbox_id, {"running"}, failed_states={"failed"})

    async def await_state(self, sandbox_id: str, wanted: set[str], *, failed_states: set[str]) -> None:
        """
        Poll one sandbox until its state settles, or the deadline expires.
        """
        deadline = time.monotonic() + self.config.pause_timeout_s
        while True:
            state = await self.state(sandbox_id)
            if state in wanted:
                return
            if state in failed_states:
                raise OpenSandboxError(f"OpenSandbox sandbox {sandbox_id!r} entered state {state!r}.")
            if time.monotonic() >= deadline:
                raise OpenSandboxError(
                    f"OpenSandbox sandbox {sandbox_id!r} was still {state!r} after "
                    f"{self.config.pause_timeout_s:g}s waiting for {sorted(wanted)}."
                )
            await asyncio.sleep(self.config.poll_interval_s)

    async def renew(self, sandbox_id: str, *, seconds: float) -> None:
        """
        Renew a sandbox's expiration from now.
        """
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1.0, seconds))
        await self.request(
            "POST",
            f"/v1/sandboxes/{sandbox_id}/renew-expiration",
            expected=(200, 204),
            payload={"expiresAt": expires_at.isoformat(timespec="seconds").replace("+00:00", "Z")},
        )

    async def snapshot(self, sandbox_id: str, *, name: str) -> Mapping[str, Any]:
        """Capture one sandbox's filesystem and wait until the capture is durable.

        The create is accepted with a `Creating` snapshot and returns before the
        artifact exists, so a restore issued against the id it returned can race the
        capture. Waiting here is what makes the returned reference safe to restore.
        """
        response = await self.request(
            "POST",
            f"/v1/sandboxes/{sandbox_id}/snapshots",
            expected=(200, 201, 202),
            payload={"name": name},
        )
        record = response or {}
        snapshot_id = str(record.get("id") or record.get("snapshotId") or "")
        if not snapshot_id:
            raise OpenSandboxError(f"OpenSandbox did not return a snapshot id for sandbox {sandbox_id!r}.")
        await self.await_snapshot(snapshot_id)
        return record

    async def await_snapshot(self, snapshot_id: str) -> None:
        """
        Poll one snapshot until it is ready to restore from.
        """
        deadline = time.monotonic() + self.config.snapshot_timeout_s
        while True:
            state = _snapshot_state(await self.get_snapshot(snapshot_id))
            if state == _SNAPSHOT_READY:
                return
            if state == _SNAPSHOT_FAILED:
                raise OpenSandboxError(f"OpenSandbox snapshot {snapshot_id!r} failed to capture.")
            if time.monotonic() >= deadline:
                raise OpenSandboxError(
                    f"OpenSandbox snapshot {snapshot_id!r} was still {state!r} after "
                    f"{self.config.snapshot_timeout_s:g}s."
                )
            await asyncio.sleep(self.config.poll_interval_s)

    async def get_snapshot(self, snapshot_id: str) -> Mapping[str, Any]:
        """
        Return one snapshot's record.
        """
        response = await self.request("GET", f"/v1/snapshots/{snapshot_id}")
        return response or {}

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """
        Delete a snapshot, treating a missing one as already gone.
        """
        try:
            await self.request("DELETE", f"/v1/snapshots/{snapshot_id}", expected=(200, 202, 204))
        except OpenSandboxError as exc:
            if "404" not in str(exc):
                raise

    async def build_template(self, image: str) -> str:
        """Start a golden-image build and wait until it succeeds, returning its id.

        Polling here rather than in the rollout is what turns a slow or broken build
        into an admission failure instead of a create failure. Both the publish target
        and the image are required by the provider, and a build is keyed by the id the
        server mints rather than by any name a caller chooses.
        """
        if self.config.template_publish is None:
            raise OpenSandboxError(
                "OpenSandbox template builds need an S3-compatible publish target. Configure "
                "`template_publish` on a fast-sandbox deployment, or let sandboxes create from their image."
            )
        started = await self.request(
            "POST",
            "/v1/templates",
            expected=(200, 201, 202),
            payload={"image": image, "publish": self.config.template_publish},
        )
        template_id = str((started or {}).get("templateId") or (started or {}).get("id") or "")
        if not template_id:
            raise OpenSandboxError(f"OpenSandbox did not return a template id for {image!r}.")
        deadline = time.monotonic() + self.config.template_timeout_s
        while True:
            phase = _template_phase(await self.request("GET", f"/v1/templates/{template_id}", expected=(200,)))
            if phase == "succeeded":
                return template_id
            if phase == "failed":
                raise OpenSandboxError(f"OpenSandbox template build for {image!r} failed.")
            if time.monotonic() >= deadline:
                raise OpenSandboxError(
                    f"OpenSandbox template build for {image!r} was still {phase!r} after "
                    f"{self.config.template_timeout_s:g}s."
                )
            await asyncio.sleep(self.config.poll_interval_s)

    async def find_template(self, image: str) -> str | None:
        """Return the id of a built template for one image, or None.

        The provider has no name-keyed lookup and never returns the name a caller
        invented, so a built template is found by listing and matching its source image.
        """
        response = await self.request("GET", "/v1/templates", expected=(200,))
        items = (response or {}).get("items") or []
        for item in items:
            if not isinstance(item, Mapping) or item.get("image") != image:
                continue
            if _template_phase(item) == "succeeded":
                template_id = str(item.get("templateId") or "")
                if template_id:
                    return template_id
        return None

    async def resolve_endpoint(self, sandbox_id: str, port: int) -> SandboxEndpoint:
        """Resolve one sandbox endpoint, with the headers it requires.

        Resolved rather than configured, because the address is transport state: a
        resume can land the sandbox somewhere else. The provider returns the headers a
        caller has to forward when it needs any, so they are carried rather than guessed.

        The resolved value can carry a `/proxy/<port>` path. That path is the lifecycle
        server's own route for reaching a service inside the sandbox, and the plane's
        API is served at the root of the address, so treating the whole value as a base
        would prefix every call with a route that is not a plane.
        """
        response = await self.request("GET", f"/v1/sandboxes/{sandbox_id}/endpoints/{port}")
        payload = response or {}
        endpoint = payload.get("endpoint") or payload.get("url") or payload.get("address")
        if not endpoint:
            raise OpenSandboxError(
                f"OpenSandbox did not return an endpoint for sandbox {sandbox_id!r} on port {port}."
            )
        address = str(endpoint)
        if not address.startswith("http"):
            address = f"http://{address}"
        headers = {str(name): str(value) for name, value in (payload.get("headers") or {}).items()}
        return SandboxEndpoint(address=_plane_base_url(address, port), headers=headers)

    async def exec_endpoint(self, sandbox_id: str) -> SandboxEndpoint:
        """
        Resolve the execution plane's endpoint for one sandbox.
        """
        return await self.resolve_endpoint(sandbox_id, self.config.execd_port)

    async def egress_endpoint(self, sandbox_id: str) -> SandboxEndpoint:
        """
        Resolve the egress sidecar's endpoint for one sandbox.
        """
        return await self.resolve_endpoint(sandbox_id, self.config.egress_port)

    async def close(self) -> None:
        """
        Close the lifecycle connection pool.
        """
        await self.transport.close()


def _egress_payload(policy) -> dict[str, Any]:
    """Render an outbound policy for the provider's `networkPolicy` field.

    The shape follows the egress sidecar's own `/policy` body, so the policy set at
    create and the policy a runtime mutation would set are the same document. The
    provider derives the port from the scheme and rejects a rule that names one, so a
    rule that carries ports is refused here rather than silently dropped.
    """
    rules: list[dict[str, str]] = []
    for rule in policy.rules:
        if rule.ports:
            raise OpenSandboxError(
                f"OpenSandbox derives an egress rule's port from its scheme and cannot honour the ports "
                f"{list(rule.ports)} requested for {rule.target!r}."
            )
        rules.append({"action": rule.action.value, "target": rule.target})
    return {"defaultAction": policy.default_action.value, "egress": rules}


class OpenSandboxEgressClient:
    """The egress sidecar for one sandbox: the credential vault.

    It is reached the way the execution plane is, by resolving the sandbox endpoint for
    the egress port and forwarding the headers that resolution returned. The provider
    brokers credentials here, so a value is written once and never read back.
    """

    def __init__(
        self,
        endpoint: SandboxEndpoint,
        *,
        transport: OpenSandboxTransport | None = None,
    ) -> None:
        if not endpoint.address:
            raise ValueError("OpenSandbox egress endpoint is required.")
        self.endpoint = endpoint.address.rstrip("/")
        self.headers = dict(endpoint.headers)
        self.transport = transport or AiohttpTransport()

    async def create_vault(self, credentials: Sequence[Mapping[str, Any]], bindings: Sequence[Mapping[str, Any]]):
        """Write credentials and their bindings, activating the vault.

        Write-only by contract: inline values are never returned, by this call or any
        later read, so a secret crosses this boundary and is not recoverable from it.
        """
        return await self.transport.request(
            "POST",
            f"{self.endpoint}/credential-vault",
            expected=(200, 201),
            payload={"credentials": [dict(item) for item in credentials], "bindings": list(bindings)},
            headers=self.headers,
        )

    async def vault_state(self) -> Mapping[str, Any]:
        """
        Read the vault's sanitized state, which carries names and revisions but no values.
        """
        response = await self.transport.request(
            "GET", f"{self.endpoint}/credential-vault", expected=(200,), headers=self.headers
        )
        return response or {}

    async def close(self) -> None:
        """
        Close the egress connection pool.
        """
        await self.transport.close()


class OpenSandboxExecClient:
    """The execution plane for one sandbox: commands, sessions, files, metrics."""

    def __init__(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        transport: OpenSandboxTransport | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("OpenSandbox exec endpoint is required.")
        self.endpoint = endpoint.rstrip("/")
        self.headers = dict(headers or {})
        self.transport = transport or AiohttpTransport()

    def _headers(self) -> dict[str, str]:
        # The provider returns the headers its endpoint requires, so they are forwarded
        # rather than reconstructed from a token named here.
        return dict(self.headers)

    async def start_session(self, *, cwd: str | None = None) -> str:
        """Open a persistent shell and return its session id.

        The provider's session create takes no shell to choose and its body is optional,
        so the request carries only a working directory when one is asked for.
        """
        response = await self.transport.request(
            "POST",
            f"{self.endpoint}/session",
            expected=(200, 201),
            payload={"cwd": cwd} if cwd else None,
            headers=self._headers(),
        )
        session_id = (response or {}).get("session_id")
        if not session_id:
            raise OpenSandboxError("OpenSandbox did not return a session_id for a persistent shell.")
        return str(session_id)

    async def delete_session(self, session_id: str) -> None:
        """
        Terminate a persistent shell and release its process.
        """
        await self.transport.request(
            "DELETE",
            f"{self.endpoint}/session/{session_id}",
            expected=(200, 204),
            headers=self._headers(),
        )

    async def run_in_session(
        self,
        session_id: str,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command in a shell session and stream its output.

        The session path reports no exit code: the provider's event stream carries output
        and an error, and a session run has no status endpoint to ask afterwards. A run
        that reports an error is a failure. A non-zero exit that produces no error event
        is not distinguishable from success over this API.
        """
        payload: dict[str, Any] = {"command": command}
        if cwd:
            payload["cwd"] = cwd
        if timeout_s is not None:
            # The provider's timeout is milliseconds, and it is enforced in the guest,
            # which is the only deadline either side of this call can honour.
            payload["timeout"] = max(0, int(timeout_s * 1000))
        return await self._consume(
            self.transport.stream_lines(
                "POST",
                f"{self.endpoint}/session/{session_id}/run",
                payload=payload,
                headers=self._headers(),
            )
        )

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command with no shell state, streaming until it ends.

        The one-shot path has the same exit-code limit as the session path, for the same
        reason. What it adds is `envs`, which the session request does not have, so a
        command that needs an environment uses this mode.
        """
        payload: dict[str, Any] = {"command": command}
        if cwd:
            payload["cwd"] = cwd
        if env:
            payload["envs"] = dict(env)
        if timeout_s is not None:
            payload["timeout"] = max(0, int(timeout_s * 1000))
        return await self._consume(
            self.transport.stream_lines(
                "POST",
                f"{self.endpoint}/command",
                payload=payload,
                headers=self._headers(),
            )
        )

    async def _consume(self, lines: AsyncIterator[str]) -> ExecResult:
        """Fold a provider event stream into one result.

        The event shape is `{type, text, results, error}`: `stdout` and `stderr` carry
        their text, `result` carries its body under a MIME map, and `error` carries a
        name, a value, and a traceback. Nothing in the stream is an exit code, so an
        error event is the only failure signal there is.
        """
        stdout: list[str] = []
        stderr: list[str] = []
        failed = False
        async for line in lines:
            event = _stream_event(line)
            if event is None:
                continue
            kind = str(event.get("type") or "")
            if kind == "error":
                failed = True
                stderr.append(_error_text(event))
                continue
            text = str(event.get("text") or "")
            if kind == "stderr":
                stderr.append(text)
            elif kind in {"stdout", "result"}:
                stdout.append(text or _result_text(event))
        return ExecResult(exit_code=1 if failed else 0, stdout="".join(stdout), stderr="".join(stderr))

    async def download(self, path: str) -> bytes:
        """
        Read one file out of the sandbox.
        """
        return await self.transport.read_file(
            f"{self.endpoint}/files/download?path={quote(path, safe='')}",
            headers=self._headers(),
        )

    async def upload(self, path: str, data: bytes) -> None:
        """Write one file into the sandbox.

        The provider's upload is multipart: a JSON metadata part naming the destination,
        then the file itself. The path is inside the metadata rather than in the URL.
        """
        await self.transport.upload_file(
            f"{self.endpoint}/files/upload",
            data,
            metadata={"path": path},
            headers=self._headers(),
        )

    async def file_info(self, path: str) -> Mapping[str, Any]:
        """
        Return one file's metadata, as the provider describes it.
        """
        response = await self.transport.request(
            "GET",
            f"{self.endpoint}/files/info?path={quote(path, safe='')}",
            expected=(200,),
            headers=self._headers(),
        )
        return dict(response or {})

    async def metrics(self) -> Mapping[str, Any]:
        """
        Return one sandbox's resource telemetry.
        """
        response = await self.transport.request(
            "GET",
            f"{self.endpoint}/metrics",
            expected=(200,),
            headers=self._headers(),
        )
        return response or {}

    async def close(self) -> None:
        """
        Close this plane's connection pool.
        """
        await self.transport.close()


class OpenSandboxSession(SandboxSession):
    """One OpenSandbox sandbox, with its resolved execution endpoint."""

    def __init__(
        self,
        backend: OpenSandboxBackend,
        sandbox_id: str,
        *,
        spec: SandboxSpec | None = None,
        state: str = "running",
    ) -> None:
        self.backend = backend
        self.sandbox_id = sandbox_id
        self._spec = spec
        self._state = state
        self._exec_client: OpenSandboxExecClient | None = None
        self._shell_session_id: str | None = None
        self._command_count = 0
        self._busy = False
        self._last_activity_at: float | None = None
        self._terminate_lock = asyncio.Lock()
        # Commands serialize on this, which is also the in-flight signal a pause reads.
        # Without it two overlapping commands share one provider shell.
        self._exec_lock = asyncio.Lock()
        self._exit_reason = SandboxExitReason.UNKNOWN

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

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def last_activity_at(self) -> float | None:
        return self._last_activity_at

    @property
    def exit_reason(self) -> SandboxExitReason:
        return self._exit_reason

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command, through the mode the spec selected.

        The execution plane offers both modes natively, so the persistent shell is
        the provider's own shell rather than a protocol PSRL emulates. Commands are
        serialized, which the session contract states as a promise: on the persistent
        path two overlapping commands would interleave in one provider shell, and the
        `busy` an idle pass reads would mean nothing.
        """
        del silence_timeout_s  # The provider owns progress reporting in the guest.
        async with self._exec_lock:
            self._command_count += 1
            self._busy = True
            self._last_activity_at = time.monotonic()
            try:
                client = await self._client()
                if self._exec_mode() is ExecMode.ONE_SHOT:
                    return await client.run_command(command, cwd=cwd, env=env, timeout_s=timeout_s)
                session_id = await self._shell(client)
                if env:
                    # The session run has no environment field, so the environment becomes part of
                    # the command. POSIX quoting, not a Python repr: `'it\'s'` is a shell quoting error.
                    prefix = "".join(f"export {key}={shlex.quote(value)};" for key, value in env.items())
                    command = prefix + command
                # A working directory does have a field on the run, so it is not faked with a
                # `cd` the caller cannot see.
                return await client.run_in_session(session_id, command, cwd=cwd, timeout_s=timeout_s)
            finally:
                self._busy = False
                self._last_activity_at = time.monotonic()

    def _exec_mode(self) -> ExecMode:
        """
        Return the exec mode this sandbox uses.
        """
        if self._spec is None or self._spec.exec_mode is None:
            return ExecMode.PERSISTENT
        return self._spec.exec_mode

    async def _client(self) -> OpenSandboxExecClient:
        """
        Return the execution client, resolving its endpoint when it is unknown.
        """
        if self._exec_client is None:
            self._exec_client = await self.backend.open_exec_client(self.sandbox_id)
        return self._exec_client

    async def _shell(self, client: OpenSandboxExecClient) -> str:
        """
        Return the persistent shell's session id, opening one when needed.
        """
        if self._shell_session_id is None:
            self._shell_session_id = await client.start_session()
        return self._shell_session_id

    async def read_bytes(self, path: str) -> bytes:
        return await (await self._client()).download(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await (await self._client()).upload(path, data)

    async def status(self) -> SandboxStatus:
        if self._state in {"terminated", "failed"}:
            return _status_from(self._state)
        state = await self.backend.control.state(self.sandbox_id)
        self._state = state
        return _status_from(state)

    async def stats(self):
        """Return the sandbox's resource usage, which this plane cannot measure.

        `GET /metrics` on the execution plane reports the **host** it runs on rather
        than the sandbox: it answers with the node's whole core count and memory, not
        the sandbox's limit or footprint. Reporting that as a sandbox figure would make
        every sandbox look like it used the entire node, and an operator sizing an
        envelope from it would be wrong by orders of magnitude.

        Every field is therefore unknown, which is what the contract's None means. A
        measured value needs a provider metric that is scoped to one sandbox.
        """
        from psrl.sandbox.core import ResourceUsage

        return ResourceUsage()

    async def diagnostics(self) -> SandboxDiagnostics:
        """Return read-only evidence, from the provider rather than a local daemon."""
        inspection = await self.backend.control.inspect(self.sandbox_id)
        return SandboxDiagnostics(
            ref=self.ref,
            status=_status_from(self._state),
            exit_reason=self._exit_reason,
            inspect=dict(inspection),
        )

    async def pause(self, mode: PauseMode) -> None:
        if mode is not PauseMode.HIBERNATE:
            raise RuntimeError("OpenSandbox pause releases compute, so it is a hibernation rather than a freeze.")
        # Pausing with a command in flight stops the guest halfway through it. The lock is
        # taken without waiting, because waiting would hold an idle pass open for that command.
        if not await acquire_nowait(self._exec_lock):
            raise SandboxBusyError(
                f"OpenSandbox sandbox {self.sandbox_id!r} has a command in flight, so it is not idle."
            )
        try:
            await self.backend.control.pause(self.sandbox_id)
        finally:
            self._exec_lock.release()
        self._state = "paused"
        self._shell_session_id = None
        self._exec_client = None

    async def resume(self) -> None:
        """Resume a paused sandbox and re-establish everything the pause released.

        A resume rebuilds the sandbox from its captured rootfs, so the processes are
        new, the transport moved, and the egress sidecar starts with an empty vault.
        Reinstalling the vault here is what keeps a resumed episode from quietly
        issuing unauthenticated requests whose rewards still look valid.
        """
        await self.backend.control.resume(self.sandbox_id)
        self._state = "running"
        # A resume can move the sandbox, so the endpoint is resolved again rather
        # than reused.
        self._exec_client = None
        self._shell_session_id = None
        if self.spec is not None and self.spec.credential_bindings:
            await self.backend.install_vault(self.sandbox_id, self.spec)

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """Capture the sandbox's filesystem as a durable snapshot.

        The provider commits the rootfs, so the capture is a filesystem snapshot. A
        restore brings the workspace back and starts every process fresh, which is the
        level reported here rather than the stronger one the caller may have wanted.
        """
        if kind is not SnapshotKind.FILESYSTEM:
            raise RuntimeError(
                "OpenSandbox commits the sandbox filesystem, so it captures a filesystem snapshot rather "
                "than memory and running processes."
            )
        name = self.backend.snapshot_name(self.sandbox_id)
        response = await self.backend.control.snapshot(self.sandbox_id, name=name)
        snapshot_id = response.get("snapshotId") or response.get("id")
        if not snapshot_id:
            raise OpenSandboxError("OpenSandbox did not return a snapshot id for a checkpoint.")
        return SnapshotRef(
            backend=self.backend.name,
            snapshot_id=str(snapshot_id),
            kind=SnapshotKind.FILESYSTEM,
            metadata={"psrl.snapshot.name": name},
            resume_level=ResumeLevel.FILESYSTEM,
        )

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._state in {"terminated", "failed"}:
                return
            await self.backend.control.delete(self.sandbox_id)
            if self._exec_client is not None:
                await self._exec_client.close()
                self._exec_client = None
            self._state = "terminated"
            if self._exit_reason is SandboxExitReason.UNKNOWN:
                self._exit_reason = SandboxExitReason.RELEASED
            self.backend.metrics.session_stopped()


class OpenSandboxBackend(SandboxBackend):
    """OpenSandbox as a first class backend.

    No sandbox consumes this worker's node, so admission is skipped: the provider
    owns its own scheduling, which is the whole point of an external backend.
    """

    def __init__(
        self,
        config: OpenSandboxConfig | Mapping[str, Any],
        *,
        name: str = "opensandbox",
        transport: OpenSandboxTransport | None = None,
        exec_transport_factory=None,
        snapshot_name_prefix: str = "psrl",
    ) -> None:
        self.config = OpenSandboxConfig.from_value(config)
        self._name = name
        self.control = OpenSandboxControlClient(self.config, transport=transport)
        self._transport = transport
        # Each data plane gets its own pool, so a saturated command stream cannot starve
        # lifecycle calls. Leaving it unset raised on the first command instead.
        self._exec_transport_factory = exec_transport_factory or AiohttpTransport
        self.snapshot_name_prefix = snapshot_name_prefix
        from psrl.sandbox.metrics import SandboxMetrics

        self.metrics = SandboxMetrics()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare what this deployment can do.

        Three of these are deployment properties rather than API properties, so each is
        declared only where the deployment actually provides it:

        - The isolation boundary is server-side and has no per-sandbox request field.
        - A golden image is built only where a publish target is configured, and the
          provider answers 501 for the template API elsewhere.
        - A warm sandbox is claimed from a pool the operator pre-created.

        The resume level is a filesystem resume, because a snapshot commits the rootfs
        and restores it on any host while every process starts fresh.
        """
        features = _OPEN_SANDBOX_FEATURES
        if self.config.isolation_runtime is not None:
            features = features | {SandboxFeature.ISOLATION_RUNTIME}
        if self.config.template_publish is not None:
            features = features | {SandboxFeature.TEMPLATE_BUILD}
        if self.config.warm_pool_ref is not None:
            features = features | {SandboxFeature.WARM_POOL}
        return SandboxCapabilities(features, resume_level=ResumeLevel.FILESYSTEM)

    async def open_exec_client(self, sandbox_id: str) -> OpenSandboxExecClient:
        """
        Resolve the execution endpoint for one sandbox and build its client.
        """
        endpoint = await self.control.exec_endpoint(sandbox_id)
        return OpenSandboxExecClient(
            endpoint.address,
            headers=endpoint.headers,
            transport=self._exec_transport_factory(),
        )

    def snapshot_name(self, sandbox_id: str) -> str:
        """
        Return a namespaced snapshot name, so two runs cannot collide.
        """
        return f"{self.config.namespace}-{sandbox_id}-{int(time.time() * 1000)}"

    async def prepare(self, spec: SandboxSpec) -> None:
        """Build or verify the golden image a task image resolves to.

        A template is an optimization, not a prerequisite: a sandbox created from an
        image needs no template at all. So this is a no-op unless the deployment
        configured a publish target, which is also the only configuration where the
        provider serves the template API.

        A failed build fails admission here rather than the first create, which keeps a
        slow or broken build off the rollout critical path.
        """
        if spec.source.kind != SandboxSourceKind.IMAGE or self.config.template_publish is None:
            return
        if await self.control.find_template(spec.source.reference) is not None:
            return
        with self.metrics.measure("prepare_template"):
            await self.control.build_template(spec.source.reference)

    def _pool_ref_for(self, spec: SandboxSpec) -> str | None:
        """Return the pool to claim from, or None when this spec cannot use one.

        A pool's pod is pre-created with a fixed shape, so the provider rejects a
        network policy, a credential proxy, and volumes beside it. A spec that asks for
        any of them is created from an image instead, which is the only shape that can
        carry them.
        """
        if self.config.warm_pool_ref is None:
            return None
        if spec.egress is not None or spec.credential_bindings or spec.volumes:
            return None
        if spec.source.kind is not SandboxSourceKind.IMAGE:
            return None
        return self.config.warm_pool_ref

    def _require_isolation(self, spec: SandboxSpec) -> None:
        """Refuse a spec that requires a boundary this deployment does not provide.

        The provider has no per-sandbox isolation-runtime request field, so the boundary
        is a property of the server. Saying so is better than admitting the spec and
        running it with less isolation than it asked for.
        """
        if SandboxFeature.ISOLATION_RUNTIME in spec.required_features and self.config.isolation_runtime is None:
            raise OpenSandboxError(
                "This spec requires a stronger isolation boundary, and the boundary belongs to the "
                "OpenSandbox deployment rather than to one sandbox. Configure `isolation_runtime` when this "
                "server provides one."
            )

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        self._require_isolation(spec)
        pool_ref = self._pool_ref_for(spec)
        template: str | None = None
        if (
            pool_ref is None
            and self.config.template_publish is not None
            and spec.source.kind == SandboxSourceKind.IMAGE
            and not spec.credential_bindings
        ):
            # A golden image is worth claiming where the deployment builds one and the spec
            # needs no egress sidecar, which a template-backed sandbox does not have.
            template = await self.control.find_template(spec.source.reference)
        self._require_policy_for_bindings(spec)
        environment = self._environment_for(spec)
        with self.metrics.measure("create"):
            response = await self.control.create(
                spec,
                template=template,
                pool_ref=pool_ref,
                network_policy=_egress_payload(spec.egress) if spec.egress is not None else None,
                credential_proxy=bool(spec.credential_bindings),
                env=environment,
            )
        sandbox_id = self._sandbox_id(response)
        if response.get("allocation"):
            # The runtime confirmed a concrete pool allocation, which is the provider's
            # warm start signal and the only way a caller learns the claim was warm.
            self.metrics.count("warm_pool_claim")
        if spec.credential_bindings:
            await self.install_vault(sandbox_id, spec)
        await self._await_ready(sandbox_id)
        self.metrics.session_started()
        return OpenSandboxSession(self, sandbox_id, spec=spec)

    async def _await_ready(self, sandbox_id: str) -> None:
        """Wait until the execution plane answers, not merely until it has an address.

        The provider publishes the endpoint asynchronously and the `execd` process
        inside the sandbox starts listening after that, so a resolved address is not yet
        a sandbox a caller can command. Probing the plane itself keeps a startup failure
        in the create rather than turning it into a confusing first command, which is
        the same reason the internal backend runs a command before handing a container out.
        """
        deadline = time.monotonic() + self.config.ready_timeout_s
        state = "unknown"
        while True:
            state = await self.control.state(sandbox_id)
            if state in {"failed", "terminated"}:
                raise OpenSandboxError(f"OpenSandbox sandbox {sandbox_id!r} reached {state!r} while starting.")
            if state == "running" and await self._exec_plane_answers(sandbox_id):
                return
            if time.monotonic() >= deadline:
                raise OpenSandboxError(
                    f"OpenSandbox sandbox {sandbox_id!r} was still {state!r} without a reachable execution "
                    f"endpoint after {self.config.ready_timeout_s:g}s."
                )
            await asyncio.sleep(self.config.poll_interval_s)

    async def _exec_plane_answers(self, sandbox_id: str) -> bool:
        """Return whether the sandbox's execution plane is serving requests.

        Any answer proves the plane is listening, including one that reports an error,
        so the status is not inspected. What is being ruled out is a connection that is
        refused or reset, which is what an unstarted `execd` does.
        """
        try:
            endpoint = await self.control.exec_endpoint(sandbox_id)
        except OpenSandboxError:
            # Running but not yet routable: the endpoint is published separately.
            return False
        client = OpenSandboxExecClient(
            endpoint.address,
            headers=endpoint.headers,
            transport=self._exec_transport_factory(),
        )
        try:
            await client.metrics()
        except OpenSandboxError:
            # The plane answered and refused, which still proves it is listening.
            return True
        except Exception:
            return False
        finally:
            await client.close()
        return True

    @staticmethod
    def _environment_for(spec: SandboxSpec) -> dict[str, str]:
        """Return the environment the sandbox starts with.

        A brokered credential is not in it. The provider's sidecar injects the real
        value on the way out, and a tool that reads its credential from the environment
        needs something there, so the variable carries an explicit placeholder instead
        of a value. The placeholder names the vault entry rather than pretending to be
        a secret, so a leaked environment says where the secret went, not what it is.
        """
        environment = dict(spec.env)
        for credential in spec.credentials:
            if spec.credential_bindings:
                environment.setdefault(credential.target_env, f"psrl-vault:{credential.vault_name}")
        return environment

    @staticmethod
    def _require_policy_for_bindings(spec: SandboxSpec) -> None:
        """Refuse a brokered credential whose host the outbound policy does not reach.

        The sidecar injects only into a request it is allowed to forward, so a binding
        whose host is blocked would never receive its credential: the workload would
        fail with a network error that says nothing about the missing allow rule.
        """
        if not spec.credential_bindings:
            return
        policy = spec.egress
        if policy is None:
            raise OpenSandboxError(
                "A brokered credential requires an outbound policy that allows its hosts, because the "
                "credential is injected only into a request the sidecar was allowed to forward."
            )
        blocked = sorted(
            {host for binding in spec.credential_bindings for host in binding.hosts if not policy.allows(host)}
        )
        if blocked:
            raise OpenSandboxError(
                f"Sandbox credential bindings name {blocked}, which the outbound policy does not allow. Add an "
                "allow rule for each host, or the credential is never injected."
            )

    async def install_vault(self, sandbox_id: str, spec: SandboxSpec) -> None:
        """Write this spec's credentials and bindings into the sidecar's vault.

        The values come from this process's environment and never from the spec, and the
        sandbox only ever holds the placeholder. PSRL reads back nothing but sanitized
        metadata, so a value crosses this boundary once and is not recoverable from it.
        """
        credentials = [
            {
                "name": credential.vault_name,
                "source": {"type": "inline", "value": resolve_credential(credential)},
            }
            for credential in spec.credentials
        ]
        bindings = [binding.to_payload() for binding in spec.credential_bindings]
        endpoint = await self.control.egress_endpoint(sandbox_id)
        client = OpenSandboxEgressClient(endpoint, transport=self._exec_transport_factory())
        try:
            with self.metrics.measure("credential_vault"):
                await client.create_vault(credentials, bindings)
        finally:
            await client.close()
        self.metrics.count("credential_bound")

    async def connect(self, sandbox_id: str) -> SandboxSession:
        with self.metrics.measure("connect"):
            state = await self.control.state(sandbox_id)
        if state == "paused":
            await self.control.resume(sandbox_id)
            state = "running"
        elif state in _TRANSIENT_STATES:
            # The control plane reports intent before a sandbox accepts commands, so a
            # connect waits for running rather than handing back one that cannot run.
            await self.control.await_state(sandbox_id, {"running"}, failed_states={"failed", "terminated"})
            state = "running"
        if state != "running":
            raise OpenSandboxError(f"OpenSandbox sandbox {sandbox_id!r} is {state!r}, so it cannot accept commands.")
        self.metrics.session_started()
        return OpenSandboxSession(self, sandbox_id, state=state)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """Create from a snapshot, which lands on any host the provider chooses.

        A restore is a new sandbox with a new id, so a caller must treat the returned
        session as new rather than reconnecting to the old one. The vault is
        sandbox-local and does not travel with a snapshot, so a spec that brokers
        credentials has them written into the restored sandbox's own vault.
        """
        restore_spec = spec if spec is not None else SandboxSpec(source=SandboxSource.image("unused"))
        self._require_isolation(restore_spec)
        self._require_policy_for_bindings(restore_spec)
        with self.metrics.measure("restore"):
            response = await self.control.create(
                restore_spec,
                source={"snapshotId": snapshot.snapshot_id},
                network_policy=_egress_payload(restore_spec.egress) if restore_spec.egress is not None else None,
                credential_proxy=bool(restore_spec.credential_bindings),
                env=self._environment_for(restore_spec),
            )
        sandbox_id = self._sandbox_id(response)
        if restore_spec.credential_bindings:
            await self.install_vault(sandbox_id, restore_spec)
        await self._await_ready(sandbox_id)
        self.metrics.session_started()
        return OpenSandboxSession(self, sandbox_id, spec=spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        with self.metrics.measure("delete_snapshot"):
            await self.control.delete_snapshot(snapshot.snapshot_id)

    @staticmethod
    def _sandbox_id(response: Mapping[str, Any]) -> str:
        """
        Read the sandbox identity out of a create response.
        """
        sandbox_id = (
            response.get("sandboxId")
            or response.get("sandboxID")
            or response.get("id")
            or (response.get("sandbox") or {}).get("id")
        )
        if not sandbox_id:
            raise OpenSandboxError("OpenSandbox create response did not carry a sandbox id.")
        return str(sandbox_id)

    def metrics_snapshot(self):
        """
        Return lifecycle latency and session counters.
        """
        return self.metrics.snapshot()

    async def shutdown(self) -> None:
        """
        Close the lifecycle connection pool.
        """
        await self.control.close()

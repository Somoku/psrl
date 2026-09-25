"""Per-sandbox egress enforcement on Docker networks.

An agent that reaches an unintended host corrupts its own reward before it
becomes a security incident, so an egress policy is a training data integrity
control rather than a hardening step.

The Engine API can express exactly two things on its own: reach everything, or
reach nothing (`NetworkMode: none`). A destination allowlist therefore needs a
firewall rule on the host, and this module programs it in the `DOCKER-USER`
chain, which is the chain Docker leaves for local policy and never rewrites.

One rule shapes the whole design: when a policy cannot be enforced, the sandbox
is refused. A silently open sandbox would keep producing data that looks valid,
and nothing downstream could tell.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from psrl.sandbox.core import EgressAction, EgressPolicy

psrl_logger = logging.getLogger(__file__)

# Docker's own chain for local policy. Rules here survive a daemon restart and are
# not touched by Docker's own rule management.
_POLICY_CHAIN = "DOCKER-USER"

# The network mode that denies every route out of the container.
ISOLATED_NETWORK_MODE = "none"


class EgressEnforcementError(RuntimeError):
    """
    Raised when an egress policy cannot be applied.

    The sandbox is refused rather than started open, because an unenforced policy
    is worse than no policy: the caller would believe the sandbox was contained.
    """


@dataclass(frozen=True)
class EgressPlan:
    """
    What the backend must do to enforce one policy.
    """

    # Docker's `NetworkMode` value for the container.
    network_mode: str | None = None
    # Destinations to allow, as CIDR or literal addresses resolved on the host.
    allow: tuple[str, ...] = ()
    # Ports to allow for every destination. Empty allows every port.
    ports: tuple[int, ...] = ()
    # Whether a firewall rule is needed at all.
    needs_firewall: bool = False

    @property
    def denies_everything(self) -> bool:
        """
        Return whether the policy is expressible without a firewall.
        """
        return self.network_mode == ISOLATED_NETWORK_MODE


@dataclass
class EgressRules:
    """
    One sandbox's installed firewall rules, so they can be removed exactly.
    """

    container_ip: str
    arguments: list[list[str]] = field(default_factory=list)


def plan_egress(policy: EgressPolicy | None, *, default_network_mode: str | None = None) -> EgressPlan:
    """
    Translate a policy into a network mode and, when needed, a firewall plan.
    """
    if policy is None:
        return EgressPlan(network_mode=default_network_mode)
    if policy.allow_all:
        if any(rule.action is EgressAction.DENY for rule in policy.rules):
            # The firewall is programmed as an allowlist, which cannot express allow-by-default
            # with an exception. Refusing beats enforcing a policy other than the one asked for.
            raise EgressEnforcementError(
                "This Docker egress enforcer programs an allowlist and cannot express an allow-by-default "
                "policy with a deny rule. Use a deny-by-default policy and allow what the task needs."
            )
        return EgressPlan(network_mode=default_network_mode)
    allowed = policy.allowed_targets
    if not allowed:
        return EgressPlan(network_mode=ISOLATED_NETWORK_MODE)
    return EgressPlan(
        network_mode=default_network_mode,
        allow=allowed,
        ports=policy.allowed_ports,
        needs_firewall=True,
    )


class DockerEgressEnforcer:
    """
    Install and remove one sandbox's egress rules in the host firewall.

    The firewall is programmed through the host's `iptables`, which needs to run
    on the host rather than in the daemon. A deployment without it can still use
    the deny-everything policy, and cannot use a destination allowlist.
    """

    def __init__(
        self,
        command: Sequence[str] = ("iptables",),
        *,
        chain: str = _POLICY_CHAIN,
        exists=None,
    ) -> None:
        if not command:
            raise ValueError("Docker egress enforcement requires an iptables command.")
        self.command = tuple(command)
        self.chain = chain
        self._exists = exists or (lambda name: shutil.which(name) is not None)

    def available(self) -> bool:
        """
        Return whether the firewall can be programmed on this host.
        """
        return bool(self._exists(self.command[0]))

    async def install(self, container_ip: str, plan: EgressPlan) -> EgressRules:
        """Allow the planned destinations from one address and drop the rest.

        Every rule is inserted at the head of the chain, so the terminating drop is
        installed **before** the allows: each later insert pushes it further down, and
        the chain then reads allows first, then the drop. Installing the drop last
        would leave it at the head instead, matching every packet from the sandbox
        before any allow is consulted and turning the allowlist into deny-all.

        Args:
            container_ip (str): The container's address on its bridge network.
            plan (EgressPlan): The allowlist to apply.

        Returns:
            EgressRules: The installed rules, for exact removal later.

        Raises:
            EgressEnforcementError: When the policy cannot be applied.
        """
        if not plan.needs_firewall:
            return EgressRules(container_ip=container_ip)
        if not self.available():
            raise EgressEnforcementError(
                f"Sandbox egress policy lists destinations but {self.command[0]!r} is not available on this "
                "node, so the allowlist cannot be enforced. Install the firewall or deny egress entirely."
            )
        address = _parse_address(container_ip)
        rules = EgressRules(container_ip=str(address))
        try:
            rules.arguments.append(await self._insert(self._drop_arguments(address)))
            for destination in plan.allow:
                # A hostname has to be resolved before a rule can name it, and the
                # resolver blocks, so it runs off the event loop.
                resolved = await asyncio.to_thread(_normalize_destination, destination)
                rules.arguments.append(await self._insert(self._allow_arguments(address, resolved, plan.ports)))
        except BaseException:
            await self.remove(rules)
            raise
        return rules

    async def remove(self, rules: EgressRules) -> None:
        """
        Remove one sandbox's rules, ignoring rules that were never installed.
        """
        for arguments in reversed(rules.arguments):
            try:
                await self._run(["-D", *arguments])
            except EgressEnforcementError:
                continue
        rules.arguments.clear()

    def _allow_arguments(self, address: ipaddress._BaseAddress, destination: str, ports: tuple[int, ...]) -> list[str]:
        base = [self.chain, "-s", str(address), "-d", _normalize_destination(destination)]
        if not ports:
            return [*base, "-j", "ACCEPT"]
        dports = ",".join(str(port) for port in ports)
        return [*base, "-p", "tcp", "-m", "multiport", "--dports", dports, "-j", "ACCEPT"]

    def _drop_arguments(self, address: ipaddress._BaseAddress) -> list[str]:
        return [self.chain, "-s", str(address), "-j", "DROP"]

    async def _insert(self, arguments: Sequence[str]) -> list[str]:
        """
        Install one rule at the head of the chain and return its specification.
        """
        await self._run(["-I", *arguments])
        return list(arguments)

    async def _run(self, arguments: Sequence[str]) -> None:
        argv = [*self.command, *arguments]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise EgressEnforcementError(
                f"Sandbox egress rule failed: {' '.join(argv)} returned {process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}."
            )


def container_ipv4(inspection: Mapping[str, object]) -> str | None:
    """
    Return a container's address on its bridge network, or None.
    """
    networks = (inspection.get("NetworkSettings") or {}).get("Networks") or {}
    if not isinstance(networks, Mapping):
        return None
    for settings in networks.values():
        if not isinstance(settings, Mapping):
            continue
        address = settings.get("IPAddress")
        if address:
            return str(address)
    return None


def _normalize_destination(destination: str) -> str:
    """
    Validate one allowlist entry and return it in firewall form.

    Blocking, because an entry may be a hostname. Callers run it in a thread.
    """
    entry = destination.strip()
    if not entry:
        raise EgressEnforcementError("Sandbox egress allowlist contains an empty destination.")
    if "/" in entry:
        try:
            return str(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            raise EgressEnforcementError(f"Sandbox egress destination {entry!r} is not a valid network.") from exc
    if entry == "0.0.0.0" or _is_domain(entry):
        # A hostname has to be resolved before a rule can name it, and resolving
        # inside the enforcer would silently follow a record the agent can change.
        resolved = _resolve_hostname(entry)
        if not resolved:
            raise EgressEnforcementError(
                f"Sandbox egress destination {entry!r} did not resolve, so the allowlist cannot be enforced."
            )
        return resolved
    try:
        return str(ipaddress.ip_address(entry))
    except ValueError as exc:
        raise EgressEnforcementError(
            f"Sandbox egress destination {entry!r} is not an address, a network, or a resolvable hostname."
        ) from exc


def _is_domain(entry: str) -> bool:
    return any(character.isalpha() for character in entry)


def _resolve_hostname(hostname: str) -> str | None:
    """
    Resolve one hostname to a single address on the host.

    The first answer is used deliberately: a rule can name one address, and
    following a rotating record would let a sandbox reach somewhere the operator
    never allowed.
    """
    import socket

    try:
        answers = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return None
    for answer in answers:
        address = answer[4][0]
        if address:
            return str(ipaddress.ip_address(address))
    return None


def _parse_address(value: str) -> ipaddress._BaseAddress:
    try:
        return ipaddress.ip_address(value)
    except ValueError as exc:
        raise EgressEnforcementError(
            f"Sandbox has no usable network address ({value!r}) for an egress policy."
        ) from exc


def resolve_iptables_command(configured: Sequence[str] | str | None) -> tuple[str, ...]:
    """
    Normalize a configured firewall command into an argv prefix.
    """
    if configured is None:
        return ("iptables",)
    if isinstance(configured, str):
        parts = configured.split()
        if not parts:
            raise ValueError("Docker egress firewall command cannot be empty.")
        return tuple(parts)
    command = tuple(configured)
    if not command:
        raise ValueError("Docker egress firewall command cannot be empty.")
    return command

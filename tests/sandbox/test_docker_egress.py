"""Per-sandbox egress policy, which is a data integrity control before it is a security one.

An agent that reaches an unintended host corrupts its own reward. The rule this
suite pins down is that a policy which cannot be enforced refuses the sandbox
rather than starting it open.
"""

from __future__ import annotations

import pytest
from psrl.sandbox.backends.docker.egress import (
    ISOLATED_NETWORK_MODE,
    DockerEgressEnforcer,
    EgressEnforcementError,
    container_ipv4,
    plan_egress,
    resolve_iptables_command,
)
from psrl.sandbox.core import EgressAction, EgressPolicy, EgressRule


def allowlist(*rules: EgressRule):
    """Return the plan for a deny-by-default policy carrying the given allow rules."""
    return plan_egress(EgressPolicy(rules=rules))

pytestmark = pytest.mark.cpu_test


class FakeFirewall:
    """An iptables that records argv and can refuse.

    A refused call is not recorded, because a firewall that returns non-zero did not
    apply the rule. Recording the attempt would make a rollback assertion believe a
    rule the host never took.
    """

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    async def run(self, arguments: list[str]) -> None:
        if self.fail_on is not None and self.fail_on in " ".join(arguments):
            raise EgressEnforcementError(f"refused: {' '.join(arguments)}")
        self.calls.append(list(arguments))


class RecordingEnforcer(DockerEgressEnforcer):
    """An enforcer whose firewall the test controls."""

    def __init__(self, firewall: FakeFirewall, *, available: bool = True) -> None:
        super().__init__(("iptables",), exists=lambda name: available)
        self.firewall = firewall

    async def _run(self, arguments) -> None:
        await self.firewall.run(list(arguments))


def test_no_policy_leaves_the_container_on_its_default_network() -> None:
    plan = plan_egress(None, default_network_mode="bridge")

    assert plan.network_mode == "bridge"
    assert not plan.needs_firewall


def test_an_explicit_allow_all_is_the_default_network() -> None:
    plan = plan_egress(EgressPolicy(default_action=EgressAction.ALLOW))

    assert plan.network_mode is None
    assert not plan.needs_firewall


def test_an_empty_allowlist_needs_no_firewall_at_all() -> None:
    # Docker can express deny-everything on its own, which is why the strongest
    # policy is also the cheapest one to enforce.
    plan = plan_egress(EgressPolicy())

    assert plan.network_mode == ISOLATED_NETWORK_MODE
    assert plan.denies_everything
    assert not plan.needs_firewall


def test_a_destination_list_needs_the_host_firewall() -> None:
    plan = plan_egress(EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "10.0.0.5", (443,)),)))

    assert plan.needs_firewall
    assert plan.allow == ("10.0.0.5",)
    assert plan.ports == (443,)


async def test_a_destination_list_without_a_firewall_is_refused() -> None:
    # Starting the sandbox open would keep producing data that looks valid, and
    # nothing downstream could tell.
    enforcer = RecordingEnforcer(FakeFirewall(), available=False)

    with pytest.raises(EgressEnforcementError, match="cannot be enforced"):
        await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "10.0.0.5")))


async def test_allowed_destinations_are_accepted_and_the_rest_is_dropped() -> None:
    firewall = FakeFirewall()
    enforcer = RecordingEnforcer(firewall)

    rules = await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "10.0.0.5")))

    joined = [" ".join(call) for call in firewall.calls]
    assert any("-d 10.0.0.5" in entry and "-j ACCEPT" in entry for entry in joined)
    assert any("-s 172.17.0.2 -j DROP" in entry for entry in joined)
    assert len(rules.arguments) == len(firewall.calls)


def effective_chain(calls: list[list[str]]) -> list[str]:
    """Return the chain a firewall would end up with, head first.

    Every install is an insert at the head, so the kernel's rule order is the
    reverse of the install order. Replaying that is the only way to see the order a
    packet is actually matched against, which existence assertions cannot catch.
    """
    chain: list[str] = []
    for call in calls:
        if call[0] == "-I":
            chain.insert(0, " ".join(call[1:]))
        elif call[0] == "-D":
            chain.remove(" ".join(call[1:]))
    return chain


async def test_the_drop_is_matched_only_after_every_allow() -> None:
    # An allowlist whose drop sits above its allows is deny-all, which is the
    # opposite of the policy the caller asked for and is otherwise silent.
    firewall = FakeFirewall()
    enforcer = RecordingEnforcer(firewall)
    policy = EgressPolicy(
        rules=(
            EgressRule(EgressAction.ALLOW, "10.0.0.5"),
            EgressRule(EgressAction.ALLOW, "10.0.0.6"),
        )
    )

    await enforcer.install("172.17.0.2", plan_egress(policy))

    chain = effective_chain(firewall.calls)
    assert chain[-1] == "DOCKER-USER -s 172.17.0.2 -j DROP", chain
    assert all("-j ACCEPT" in entry for entry in chain[:-1]), chain
    assert len([entry for entry in chain if "-j ACCEPT" in entry]) == 2, chain


async def test_a_refused_allow_leaves_no_drop_behind() -> None:
    # The drop is installed first, so a later failure must roll it back or the
    # sandbox's whole address stays blocked by a policy that was never applied.
    firewall = FakeFirewall(fail_on="-d 10.0.0.5")
    enforcer = RecordingEnforcer(firewall)

    with pytest.raises(EgressEnforcementError):
        await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "10.0.0.5")))

    assert effective_chain(firewall.calls) == []


async def test_ports_are_named_when_the_policy_limits_them() -> None:
    firewall = FakeFirewall()
    enforcer = RecordingEnforcer(firewall)

    await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "10.0.0.5", (443, 8080))))

    assert any("--dports 443,8080" in " ".join(call) for call in firewall.calls)


async def test_a_network_entry_is_accepted_as_a_destination() -> None:
    firewall = FakeFirewall()
    enforcer = RecordingEnforcer(firewall)

    await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "10.0.0.0/24")))

    assert any("-d 10.0.0.0/24" in " ".join(call) for call in firewall.calls)


async def test_an_unresolvable_hostname_is_refused() -> None:
    enforcer = RecordingEnforcer(FakeFirewall())

    with pytest.raises(EgressEnforcementError, match="did not resolve"):
        await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "no-such-host.invalid")))


async def test_rules_are_removed_exactly_and_a_second_removal_is_quiet() -> None:
    firewall = FakeFirewall()
    enforcer = RecordingEnforcer(firewall)
    rules = await enforcer.install("172.17.0.2", allowlist(EgressRule(EgressAction.ALLOW, "10.0.0.5")))
    installed = len(rules.arguments)

    await enforcer.remove(rules)
    assert len([call for call in firewall.calls if call[0] == "-D"]) == installed
    assert rules.arguments == []

    await enforcer.remove(rules)
    assert len([call for call in firewall.calls if call[0] == "-D"]) == installed


async def test_a_partially_installed_policy_is_rolled_back() -> None:
    # The first allow and the drop are already in the chain when the second allow is
    # refused, so both have to come back out.
    firewall = FakeFirewall(fail_on="-d 10.0.0.6")
    enforcer = RecordingEnforcer(firewall)
    policy = EgressPolicy(
        rules=(
            EgressRule(EgressAction.ALLOW, "10.0.0.5"),
            EgressRule(EgressAction.ALLOW, "10.0.0.6"),
        )
    )

    with pytest.raises(EgressEnforcementError):
        await enforcer.install("172.17.0.2", plan_egress(policy))

    assert any(call[0] == "-D" for call in firewall.calls)
    assert effective_chain(firewall.calls) == []


async def test_removing_rules_a_deployment_never_installed_is_not_an_error() -> None:
    enforcer = RecordingEnforcer(FakeFirewall(fail_on="-D"))

    from psrl.sandbox.backends.docker.egress import EgressRules

    await enforcer.remove(EgressRules(container_ip="172.17.0.2", arguments=[["DOCKER-USER", "-s", "1.2.3.4"]]))


def test_the_container_address_is_read_from_the_first_bridge_network() -> None:
    inspection = {"NetworkSettings": {"Networks": {"bridge": {"IPAddress": "172.17.0.7"}}}}

    assert container_ipv4(inspection) == "172.17.0.7"
    assert container_ipv4({}) is None
    assert container_ipv4({"NetworkSettings": {"Networks": {"bridge": {}}}}) is None


async def test_a_container_with_no_address_is_refused() -> None:
    enforcer = RecordingEnforcer(FakeFirewall())

    with pytest.raises(EgressEnforcementError, match="no usable network address"):
        await enforcer.install("", plan_egress(EgressPolicy(rules=(EgressRule(EgressAction.ALLOW, "10.0.0.5"),))))


def test_the_firewall_command_defaults_to_iptables() -> None:
    assert resolve_iptables_command(None) == ("iptables",)
    assert resolve_iptables_command("nft iptables") == ("nft", "iptables")
    assert resolve_iptables_command(["iptables", "-w"]) == ("iptables", "-w")


def test_the_firewall_command_refuses_to_be_empty() -> None:
    with pytest.raises(ValueError):
        resolve_iptables_command("   ")


def test_an_open_policy_with_an_exception_is_refused() -> None:
    # The enforcer programs an allowlist, so allow-by-default with a deny cannot be
    # enforced. Refusing beats enforcing a policy that is not the one asked for.
    policy = EgressPolicy(
        default_action=EgressAction.ALLOW,
        rules=(EgressRule(EgressAction.DENY, "bad.example.com"),),
    )

    with pytest.raises(EgressEnforcementError, match="allowlist"):
        plan_egress(policy)


def test_a_rule_refuses_a_port_outside_the_usable_range() -> None:
    with pytest.raises(ValueError, match="1..65535"):
        EgressRule(EgressAction.ALLOW, "10.0.0.5", (0,))


def test_a_policy_allows_only_what_its_rules_allow() -> None:
    policy = EgressPolicy(
        rules=(
            EgressRule(EgressAction.ALLOW, "api.example.com"),
            EgressRule(EgressAction.ALLOW, "*.packages.example.com"),
            EgressRule(EgressAction.DENY, "evil.example.com"),
        )
    )

    assert policy.allows("api.example.com")
    assert policy.allows("cdn.packages.example.com")
    # A wildcard covers a subdomain and not the bare domain, which is what a rule
    # naming `*.host` means everywhere else it is written.
    assert not policy.allows("packages.example.com")
    assert not policy.allows("evil.example.com")
    assert not policy.allows("elsewhere.invalid")
    assert policy.allowed_targets == ("api.example.com", "*.packages.example.com")


def test_a_default_allow_policy_allows_anything_unlisted() -> None:
    policy = EgressPolicy(default_action=EgressAction.ALLOW)

    assert policy.allows("anything.invalid")
    assert policy.allowed_targets == ()

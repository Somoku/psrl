import logging
import os
import pickle
import time
from copy import deepcopy

from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.common.worker_naming import model_role_of
from psrl.utils.nixl.block_fp8 import SCALE_SUFFIX
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLCommPlan, merge_comm_plans
from psrl.utils.nixl.handshake_notif import NotifKind, decode
from psrl.utils.nixl.join_admission import JoinDecision, JoinVerdict, classify_join
from psrl.utils.nixl.membership import Membership, MembershipChange, MembershipEvent
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


def _mesh_total(mesh: dict[int, int]) -> int:
    """Total shard count a mesh implies, i.e. the product over its dimensions."""
    total = 1
    for count in mesh.values():
        total *= count
    return total


class NIXLMetaServer:
    def __init__(self, server_name: str, nixl_config: DictConfig, broadcast_init_enabled: bool = False):
        self.server_name = server_name
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        # Whether the planner groups same node rollouts that need identical shards.
        self._peer_diffusion = bool(getattr(nixl_config, "peer_diffusion", True))
        # How long each handshake phase waits for its clients.
        #
        # The default is a cluster-dependent guess, and when it is wrong the failure is
        # misleading: a run that merely started slowly dies in `wait_for_client_shardings`
        # with "Timeout waiting for agents", which reads as a connectivity fault rather
        # than as vLLM still capturing CUDA graphs. Measured on this cluster, reaching the
        # handshake took 18.6 min on one run and 23.5 min on the next -- the second
        # exceeded the 600 s budget and lost ~30 minutes of cluster time to a run that was
        # working.
        self._handshake_timeout = float(getattr(nixl_config, "handshake_timeout_s", 600.0))
        self.agent = nixl_agent(self.server_name, nixl_agent_config(True, True, self.server_port))
        self.connected_clients: dict[
            str, list[str]
        ] = {}  # agent_name -> [client_name1, client_name2, ...], one agent can bind to multiple clients
        self.client_sharding_dicts: dict[str, dict[str, NIXLSharding]] = {}
        self.client_infos: dict[str, NIXLClientInfo] = {}

        self.client_unified_sharding_dicts: dict[str, dict[str, NIXLSharding]] = {}
        #: key -> the established finest (LCM) shard mesh. Written by
        #: make_unified_sharding, read by join admission and the observability log.
        self.finest_shard_meshes: dict[str, dict[int, int]] = {}
        self.comm_plan: NIXLCommPlan | None = None
        self._client_temp_mappings: dict[str, dict] = {}

        # Per-kind rosters rather than three "handshake finished" booleans. A boolean
        # can only say the first handshake completed, and that is the answer that makes
        # a later join invisible; a roster still grows when one arrives.
        self._agents_reported: dict[NotifKind, set[str]] = {kind: set() for kind in NotifKind}
        # Admission is injected rather than reached for, so there is exactly one place
        # that decides admissibility -- A7's `classify_incoming_client`.
        self._membership = Membership(admit=self._classify_joiner)

        # NOTE(claude): Broadcast initialization makes PS workers direct peers, so
        # each receives every PS GPU descriptor.
        self._broadcast_init_enabled = broadcast_init_enabled

    def _add_client(self, agent_name: str, client_name: str):
        if agent_name not in self.connected_clients:
            self.connected_clients[agent_name] = []
        if client_name not in self.connected_clients[agent_name]:
            self.connected_clients[agent_name].append(client_name)

    def _assert_reported(self, kind: NotifKind, what: str) -> None:
        """Guard the plan-building paths against running before anyone has reported.

        Asserts the roster is non-empty rather than that the handshake "finished".
        Under elastic membership there is no such moment -- clients keep arriving -- so
        the honest precondition is that there is something to plan with.
        """
        assert self._agents_reported[kind], f"No client has sent {what} yet."

    def _drain_notifs(self) -> None:
        """Consume every pending notification once, routing each to its own registry.

        One drain for all three kinds, because `get_new_notifs` is **destructive**: it
        hands each notification to exactly one caller. Three loops polling it
        independently means whichever happens to be running swallows the others' mail.
        While the handshake was lock-step that was invisible -- the phases never
        overlapped -- but it is precisely what an out-of-phase join trips over.

        Unknown payloads are left alone rather than raising: other subsystems publish on
        the same agent stream, and `decode` returns None for anything foreign.
        """
        for agent_name, notif_list in self.agent.get_new_notifs().items():
            for raw in notif_list:
                notif = decode(raw)
                if notif is None:
                    continue
                for client_name, payload in notif.entries.items():
                    if notif.kind is NotifKind.SHARDING:
                        self.client_sharding_dicts[client_name] = payload
                    elif notif.kind is NotifKind.CLIENT_INFO:
                        self.client_infos[client_name] = NIXLClientInfo.deserialize(payload)
                    elif notif.kind is NotifKind.TEMP_MAPPING:
                        self._client_temp_mappings[client_name] = payload
                    else:
                        # Resharding re-sends both at once.
                        self.client_infos[client_name] = NIXLClientInfo.deserialize(payload["info"])
                        self._client_temp_mappings[client_name] = payload["temp_mapping"]
                    self._add_client(agent_name, client_name)
                    self._agents_reported[notif.kind].add(agent_name)

    def _wait_for(
        self,
        kind: NotifKind,
        expected_agents: int,
        timeout: float | None,
        what: str,
    ) -> None:
        """Block until `expected_agents` have reported `kind`.

        Counts *agents*, not clients: one agent may host several clients and reports for
        all of them in a single notification.

        There is deliberately no "already done, return early" short circuit. That is what
        made membership fixed at whatever the first handshake saw -- a later caller
        asking for more agents than last time is exactly how an elastic join announces
        itself, and an early return would answer it with the old roster.
        """
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        reported = self._agents_reported[kind]
        psrl_logger.info(f"Waiting for {expected_agents} agents to send {what}...")

        while len(reported) < expected_agents:
            self._drain_notifs()
            if len(reported) >= expected_agents:
                break
            if time.time() - start > budget:
                raise TimeoutError(
                    f"Timeout waiting for {expected_agents} agents to send {what}; "
                    f"{len(reported)} reported in {time.time() - start:.1f}s."
                )
            time.sleep(0.01)

        psrl_logger.info(f"{len(reported)} agents sent {what} after {time.time() - start:.1f} seconds.")

    def wait_for_client_shardings(self, expected_agents: int = 1, timeout: float | None = None):
        """Wait for `expected_agents` to send their sharding."""
        self._wait_for(NotifKind.SHARDING, expected_agents, timeout, "sharding")

    def wait_for_client_infos(self, expected_agents: int = 1, timeout: float | None = None):
        """Wait for `expected_agents` to send their client infos."""
        self._wait_for(NotifKind.CLIENT_INFO, expected_agents, timeout, "client infos")

    def wait_for_client_temp_mappings(self, expected_agents: int = 1, timeout: float | None = None):
        """Wait for `expected_agents` to send their temporary mappings."""
        self._wait_for(NotifKind.TEMP_MAPPING, expected_agents, timeout, "temp mappings")

    def registered_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Global shape per key, reconstructed from a registered member.

        A7 needs *global* shapes: whether a finer mesh can divide a tensor is a property
        of the tensor, not of the meshes. Without them `classify_join` has nothing to
        test divisibility against, so it can only ever answer RESHARD -- a join that
        should be rejected would be admitted, and every existing client would then raise
        when it re-registered against the new mesh.

        Nothing stores the global shape, but it is recoverable: a shard covers
        ``shape[d]`` of dimension ``d`` and the mesh says how many shards tile it, so the
        global extent is the product. Taken from whichever member declares a key, since
        they all describe the same tensor.
        """
        shapes: dict[str, tuple[int, ...]] = {}
        for client_name, info in self.client_infos.items():
            sharding_dict = self.client_sharding_dicts.get(client_name) or {}
            for key, tensor_info in info.tensor_infos.items():
                if key in shapes:
                    continue
                meta = next((m for m in tensor_info.shard_meta_infos if m is not None), None)
                if meta is None:
                    continue
                mesh = dict(getattr(sharding_dict.get(key), "shard_mesh", {}) or {})
                shapes[key] = tuple(extent * mesh.get(dim, 1) for dim, extent in enumerate(meta.shape))
        return shapes

    def _classify_joiner(self, client: str) -> tuple[bool, bool, str]:
        """A7's verdict for one joiner, reduced to what `Membership` needs.

        A single REJECT on any key refuses the whole client. Admitting it would make
        every *existing* client raise during re-registration, so a partial admission is
        worse than none: the joiner gains nothing and the cluster loses the members it
        had.
        """
        decisions = self.classify_incoming_client(
            self.client_sharding_dicts.get(client, {}),
            tensor_shapes=self.registered_tensor_shapes(),
        )
        rejected = {k: d for k, d in decisions.items() if d.verdict is JoinVerdict.REJECT}
        if rejected:
            key, decision = next(iter(rejected.items()))
            return False, False, f"{key}: {decision.describe()}"
        needs_reshard = any(d.requires_reshard for d in decisions.values())
        reason = ""
        if needs_reshard:
            key, decision = next((k, d) for k, d in decisions.items() if d.requires_reshard)
            reason = f"{key}: {decision.describe()}"
        return True, needs_reshard, reason

    def admit_joiners(self, timeout: float | None = None) -> list[MembershipChange]:
        """Absorb clients that registered after the initial handshake.

        This is the entry point `TODO(lhy): Support elastic clients` named and the
        handshake never had. The notification plumbing now delivers a late sharding to
        the right registry (see `handshake_notif`), but delivery alone changes nothing:
        the unified sharding and the comm plan were built once at startup, so a joiner
        would sit in `client_sharding_dicts` and appear in no plan.

        **Admission is A7's answer, not a second one.** `classify_join` already decides
        whether a mesh divides the established one; repeating that judgement here would
        give the cluster two places to disagree about who is allowed in. What this adds
        is *when* it is asked and what the cluster owes afterwards.

        Returns one change per joiner. An empty list means nothing new arrived, which is
        the normal case and is not an error -- callers may poll this.
        """
        self._drain_notifs()
        joiners = sorted(set(self.client_sharding_dicts) - self._membership.members)
        if not joiners:
            return []

        changes = [self._membership.join(client) for client in joiners]
        admitted = [c for c in changes if c.event is MembershipEvent.JOINED]
        for refused in (c for c in changes if c.event is MembershipEvent.REFUSED):
            # Refusal is fatal for the joiner and a no-op for everyone else, so the
            # refused client's sharding must not reach the replan below -- leaving it
            # there would admit by the back door what A7 just rejected.
            self.client_sharding_dicts.pop(refused.client, None)
            psrl_logger.warning(f"Refused join from {refused.client}: {refused.reason}")
        if not admitted:
            return changes

        psrl_logger.info(
            f"Admitting {len(admitted)} joiner(s): {[c.client for c in admitted]}. "
            f"Replanning for {len(self._membership.members)} member(s)."
        )
        self.make_unified_sharding(rebuild=True)
        self.notify_all_client_shardings()
        self._wait_for(
            NotifKind.CLIENT_INFO,
            len(self._agents_reported[NotifKind.CLIENT_INFO]) + 1,
            timeout,
            "client infos from the joiner",
        )
        self.make_comm_plan(rebuild=True)
        self.notify_all_client_infos_and_comm_plan()
        return changes

    def make_unified_sharding(self, *, rebuild: bool = False):
        """Make unified sharding for all clients.

        `rebuild` has to be asked for. Building twice during the initial handshake is a
        bug -- it means a step ran out of order -- while rebuilding after a join is the
        whole point of elasticity. A bare "already made" guard cannot tell those apart,
        so the caller states which one it is.
        """
        self._assert_reported(NotifKind.SHARDING, "sharding")
        assert rebuild or not self.client_unified_sharding_dicts, (
            "Unified sharding already made; pass rebuild=True to replan after a join."
        )
        _t_start = time.time()
        # Models train side by side (actor, critic) and their keys are named identically, so
        # a key is only ever unified within one model's clients. Mixing them would give a
        # rollout a plan that reads the other model's bytes for a correctly shaped tensor.
        by_role: dict[str, dict] = {}
        for client_name, sharding_dict in self.client_sharding_dicts.items():
            by_role.setdefault(model_role_of(client_name), {})[client_name] = sharding_dict
        for role, role_dicts in sorted(by_role.items()):
            self._unify_role(role, role_dicts)
        psrl_logger.info(
            f"[timing] make_unified_sharding done: total={time.time() - _t_start:.3f}s, "
            f"roles={sorted(by_role)}, n_clients={len(self.client_sharding_dicts)}"
        )
        self.log_finest_mesh_summary()

    def _unify_role(self, role: str, client_sharding_dicts: dict) -> None:
        """Build the finest common sharding of one model's keys, over that model's clients only."""
        _t_start = time.time()
        # We first need to guarantee that all client shardings have the same keys
        all_keys = set()
        for client_name, sharding_dict in client_sharding_dicts.items():
            all_keys.update(sharding_dict.keys())
        _t_keys = time.time()
        # A block scale shards exactly like its weight, so a client holding only the weight produces its scale,
        # on the scale tile the rollouts declared.
        for key in [k for k in all_keys if k.endswith(SCALE_SUFFIX)]:
            base = key[: -len(SCALE_SUFFIX)]
            declared = {
                d[key].scale_tile
                for d in client_sharding_dicts.values()
                if key in d and getattr(d[key], "scale_tile", None) is not None
            }
            assert len(declared) <= 1, f"{key}: rollouts declare different scale tiles {sorted(declared)}."
            tile = next(iter(declared), None)
            for sharding_dict in client_sharding_dicts.values():
                held = sharding_dict.get(key)
                if (held is None or held.is_empty) and base in sharding_dict and not sharding_dict[base].is_empty:
                    sharding_dict[key] = deepcopy(sharding_dict[base])
                    sharding_dict[key].scale_tile = tile
        # Build each key's finest common sharding representation.
        for key in all_keys:
            shard_mesh_list = []
            for client_name, sharding_dict in client_sharding_dicts.items():
                if key not in sharding_dict:
                    # Pipeline stages may omit keys but still need an empty shard descriptor.
                    sharding_dict[key] = NIXLSharding.empty()
                shard_mesh_list.append(sharding_dict[key].shard_mesh)
            finest_shard_mesh = NIXLSharding.find_finest_shard_mesh(shard_mesh_list)
            self.finest_shard_meshes[key] = dict(finest_shard_mesh)
            for client_name, sharding_dict in client_sharding_dicts.items():
                if client_name not in self.client_unified_sharding_dicts:
                    self.client_unified_sharding_dicts[client_name] = {}
                # NOTE(claude): Mutation is safe because `client_sharding_dicts` is
                # unused after unified sharding is built.
                sharding_dict[key].refactor_based_on_finer_shard_mesh(finest_shard_mesh)
                self.client_unified_sharding_dicts[client_name][key] = sharding_dict[key]
        psrl_logger.info(
            f"[timing] make_unified_sharding done: total={time.time() - _t_start:.3f}s, "
            f"collect_keys={_t_keys - _t_start:.3f}s, "
            f"refactor={time.time() - _t_keys:.3f}s, "
            f"n_clients={len(self.client_sharding_dicts)}, n_keys={len(all_keys)}"
        )
        self.log_finest_mesh_summary()

    def log_finest_mesh_summary(self, top_n: int = 5) -> None:
        """Report the finest-mesh sizes so metadata blow-up is visible at handshake.

        The global LCM is the second metadata amplifier after MoE key expansion: a
        single client with a coprime parallelism degree multiplies every key's shard
        count. Without this line, that only shows up later as a descriptor count
        nobody can attribute. Keeping `fsdp_size` within a node is the usual fix.
        """
        if not self.finest_shard_meshes:
            return
        worst = sorted(
            self.finest_shard_meshes.items(),
            key=lambda kv: -_mesh_total(kv[1]),
        )[:top_n]
        total_shards = sum(_mesh_total(m) for m in self.finest_shard_meshes.values())
        psrl_logger.info(
            "[finest mesh] n_keys=%d total_shards=%d; largest: %s",
            len(self.finest_shard_meshes),
            total_shards,
            ", ".join(f"{k}={dict(m)} ({_mesh_total(m)} shards)" for k, m in worst),
        )

    def classify_incoming_client(
        self,
        new_sharding_dict: dict[str, NIXLSharding],
        tensor_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> dict[str, JoinDecision]:
        """Classify a would-be joiner against the established finest mesh, per key.

        Returns one `JoinDecision` per key whose verdict is not FAST_PATH, so an
        empty result means the join costs nothing. Callers must refuse the join if
        any verdict is REJECT -- admitting it makes every *existing* client raise
        during re-registration (design doc 3.3.1d).

        Used by `admit_joiners` as the single source of admissibility, so elasticity
        and the initial handshake cannot disagree about who is allowed in.
        """
        decisions: dict[str, JoinDecision] = {}
        for key, sharding in new_sharding_dict.items():
            established = self.finest_shard_meshes.get(key)
            if established is None:
                continue
            shapes = {}
            if tensor_shapes and key in tensor_shapes:
                shapes[key] = tensor_shapes[key]
            decision = classify_join(established, sharding.shard_mesh, shapes)
            if decision.verdict is not JoinVerdict.FAST_PATH:
                decisions[key] = decision
        return decisions

    def make_comm_plan(self, *, rebuild: bool = False):
        """Make communication plan for all clients.

        See `make_unified_sharding` for why `rebuild` is explicit rather than inferred.
        """
        self._assert_reported(NotifKind.CLIENT_INFO, "client infos")
        assert rebuild or not self.comm_plan, (
            "Communication plan already made; pass rebuild=True to replan after a join."
        )

        psrl_logger.info("Making communication plan...")
        start = time.time()
        try:
            # One plan per model, merged. A transfer never crosses models, so planning them
            # together would only let a reader be matched to the wrong model's source.
            planner = CommunicationPlanner(peer_diffusion=self._peer_diffusion)
            by_role: dict[str, dict] = {}
            for name, info in self.client_infos.items():
                by_role.setdefault(model_role_of(name), {})[name] = info
            plans = [planner.make_comm_plan(infos) for _, infos in sorted(by_role.items())]
            self.comm_plan = plans[0] if len(plans) == 1 else merge_comm_plans(plans)
        except Exception as e:
            psrl_logger.error(f"Failed to make communication plan: {e}")
            raise e
        psrl_logger.info(f"Communication plan made after {time.time() - start} seconds.")

    def notify_all_client_shardings(self):
        """
        Notify all connected clients with their sharding.
        """
        self._assert_reported(NotifKind.SHARDING, "sharding")
        assert self.client_unified_sharding_dicts, "Unified sharding not made yet."
        for agent_name, client_names in self.connected_clients.items():
            client_sharding_dicts = {}
            for client_name in client_names:
                assert client_name in self.client_unified_sharding_dicts, (
                    f"Client {client_name} not found in unified sharding dicts."
                )
                sharding_dict = self.client_unified_sharding_dicts[client_name]
                client_sharding_dicts[client_name] = sharding_dict
            self.agent.send_notif(agent_name, pickle.dumps(client_sharding_dicts))

    def _get_relevant_client_names_for_agent(self, agent_name: str) -> set[str]:
        """
        Return the names of clients that a given agent actually needs to communicate with.
        Used to filter the client_infos broadcast so that each agent only receives the entries
        it needs rather than the full set of all registered clients.

        Only the initiating side needs the remote descriptors, so the rules are:
          - PUSH_SIDE needs: all PS_FOR_PUSH (train pushes to PS)
          - PULL_SIDE needs: all PS_FOR_PULL (gen pulls from PS), or all PUSH_SIDE when
            no PS exists (trainer direct supply, item D3)
          - PS clients do not initiate transfers and need no remote infos beyond their own,
            unless broadcast_init is enabled. In that case PS workers write to each other
            and therefore need every other PS worker's descriptors.
        """
        my_clients: set[str] = set(self.connected_clients[agent_name])
        relevant: set[str] = set(my_clients)  # always include own clients

        # Determine which types this agent's clients have.
        my_types: set[NIXLClientType] = {self.client_infos[c].type for c in my_clients if c in self.client_infos}

        # Initiating side → counterpart type mapping.
        type_needs: dict[NIXLClientType, NIXLClientType] = {
            NIXLClientType.PUSH_SIDE: NIXLClientType.PS_FOR_PUSH,
            NIXLClientType.PULL_SIDE: NIXLClientType.PS_FOR_PULL,
        }
        # D3: with no PS registered, a rollout reads the train clients, so those are the
        # descriptors it needs. Deciding by what is registered -- rather than by a mode
        # flag -- keeps this in step with the planner, which dispatches the same way.
        # Without it a direct-supply rollout is handed no remote descriptors at all and
        # its first read fails with "target client not found in client infos", which
        # reads as a handshake bug rather than a filter that was never updated.
        has_ps_for_pull = any(info.type is NIXLClientType.PS_FOR_PULL for info in self.client_infos.values())
        if not has_ps_for_pull:
            type_needs[NIXLClientType.PULL_SIDE] = NIXLClientType.PUSH_SIDE

        needed_types: set[NIXLClientType] = {dst for src, dst in type_needs.items() if src in my_types}

        for client_name, client_info in self.client_infos.items():
            if client_info.type in needed_types:
                relevant.add(client_name)

        # D1: a group member reads its designate and notifies its followers, so it needs every peer of its groups.
        if self.comm_plan is not None:
            for client_name in my_clients:
                for wanters in self.comm_plan.peer_groups.get(client_name, {}).values():
                    relevant.update(wanters)

        # Broadcast initialization uses direct PS writes, so include every PS
        # descriptor in the existing info exchange.
        if self._broadcast_init_enabled:
            ps_types = {NIXLClientType.PS_FOR_PUSH, NIXLClientType.PS_FOR_PULL}
            if my_types & ps_types:
                for client_name, client_info in self.client_infos.items():
                    if client_info.type in ps_types:
                        relevant.add(client_name)

        return relevant

    def notify_all_client_infos_and_comm_plan(self):
        """
        Notify all connected agents with relevant client infos and the comm plan.

        Each agent only receives the client infos it actually communicates with according
        to the comm plan, rather than the full set of all registered clients.
        """
        self._assert_reported(NotifKind.CLIENT_INFO, "client infos")
        assert self.comm_plan, "Communication plan not made yet."

        # Serialize each client info exactly once up front (avoid repeated serialization per agent)
        all_serialized: dict[str, bytes] = {name: info.serialize() for name, info in self.client_infos.items()}
        comm_plan_bytes: bytes | None = self.comm_plan.serialize() if self.comm_plan else None

        for agent_name in self.connected_clients:
            # Filter to only the client infos this agent actually needs
            relevant = self._get_relevant_client_names_for_agent(agent_name)
            payload = pickle.dumps(
                {
                    "client_infos": {n: all_serialized[n] for n in relevant if n in all_serialized},
                    "comm_plan": comm_plan_bytes,
                }
            )
            self.agent.send_notif(agent_name, payload)

    def notify_all_client_temp_mappings(self):
        """
        Notify all connected clients with all temp mappings.
        """
        self._assert_reported(NotifKind.TEMP_MAPPING, "temp mappings")
        # Prepare notification data with all clients' temp mappings
        payload = pickle.dumps(self._client_temp_mappings)
        for agent_name in self.connected_clients:
            # Send notification with all temp mappings
            self.agent.send_notif(agent_name, payload)

    def wait_for_update_infos(self, expected_agents: int, timeout: float | None = None):
        """Wait for `expected_agents` to re-send their infos after a resharding.

        The roster is cleared first, unlike the handshake kinds. Those accumulate,
        because a client that joined stays joined; this one asks "who has reported *this
        round*", and carrying last round's answer forward would return immediately with
        nobody having re-registered.
        """
        self._agents_reported[NotifKind.UPDATE_INFO].clear()
        self._wait_for(NotifKind.UPDATE_INFO, expected_agents, timeout, "updated client infos")

    def broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        """
        Broadcast updated client infos to specified agents.

        Args:
            dst_agent_names (List[str]): List of destination agent names
            update_client_names (List[str]): List of updated client names to broadcast
        """
        # Prepare notification data with updated client infos
        payload_dict = {}
        for client_name in update_client_names:
            payload_dict[client_name] = {
                "info": self.client_infos[client_name].serialize(),
                "temp_mapping": self._client_temp_mappings[client_name],
            }
        payload = pickle.dumps(payload_dict)

        for agent_name in dst_agent_names:
            # Send notification with updated client infos
            try:
                self.agent.send_notif(agent_name, payload)
            except Exception as e:
                raise RuntimeError(
                    f"{self.server_name}: Failed to send update client infos to agent {agent_name}: {e}, "
                    f"connected clients: {self.connected_clients}, "
                    f"dst agent names: {dst_agent_names}, "
                    f"update client names: {update_client_names}"
                ) from e

        psrl_logger.debug(
            f"Broadcast update client infos to agents: {dst_agent_names}, include clients: {update_client_names}"
        )

    def shutdown(self):
        """
        Shutdown the meta server.
        """
        for agent_name in self.connected_clients:
            self.agent.remove_remote_agent(agent_name)
        self.agent.invalidate_local_metadata(self.server_ip, self.server_port)

"""
NIXL Communication Planning Module

This module provides intelligent load-balanced communication planning for NIXL,
analyzing network topology and tensor distribution to generate optimal
PUSH_SIDE to PS and PULL_SIDE from PS communication plans.
"""

import pickle
from dataclasses import dataclass, field

from psrl.utils.nixl.chunk_size import choose_chunk_bytes
from psrl.utils.nixl.chunking import CommitGroup, plan_chunks
from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY
from psrl.utils.nixl.nixl_spec import NIXLClientInfo, NIXLClientType


@dataclass
class NIXLCommPlan:
    """Communication plan defining communication scheme for each client's keys"""

    # PUSH_SIDE -> PS write plan
    # {push_client: {key: {ps_client: [shard_indices]}}}
    push_to_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

    # PULL_SIDE <- PS read plan
    # {pull_client: {key: {ps_client: [shard_indices]}}}
    rollout_pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

    # PUSH_SIDE <- PS read plan
    # {pull_client: {key: {ps_client: [shard_indices]}}
    train_pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

    # Per-pull-client chunk plan (implementation item B1).
    # {pull_client: (chunks, commit_groups)}
    #
    # `chunk` and `commit_group` are deliberately separate concepts (design 3.6.1): a
    # chunk is a fixed byte budget and may split one key, while a commit group is what
    # the engine loader requires atomically and may span several chunks. They are
    # many-to-many, so collapsing them into one unit makes either the pipeline or
    # version visibility wrong.
    #
    # Empty when no size information was available at planning time; consumers must
    # treat an absent entry as "no chunking", not as an error.
    pull_chunk_plan: dict[str, tuple[list, list]] = field(default_factory=dict)

    def serialize(self):
        """Serialize communication plan"""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        """Deserialize communication plan"""
        return pickle.loads(data)

    def get_push_plan(self, push_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get write plan for specific PUSH_SIDE client and key"""
        return self.push_to_ps_plan.get(push_client, {}).get(key, {})

    def get_rollout_pull_plan(self, pull_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PULL_SIDE client and key"""
        return self.rollout_pull_from_ps_plan.get(pull_client, {}).get(key, {})

    def get_train_pull_plan(self, pull_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PUSH_SIDE client and key"""
        return self.train_pull_from_ps_plan.get(pull_client, {}).get(key, {})

    def get_ps_write_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get write plan for specific PS client and key (receiving from PUSH_SIDE)"""
        result = {}
        for push_client, key_plans in self.push_to_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[push_client] = key_plans[key][ps_client]
        return result

    def get_rollout_ps_read_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PS client and key (sending to PULL_SIDE)"""
        result = {}
        for pull_client, key_plans in self.rollout_pull_from_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[pull_client] = key_plans[key][ps_client]
        return result

    def get_pull_chunks(self, pull_client: str) -> tuple[list, list]:
        """Chunks and commit groups this pull client should transfer, in order.

        Returns empty lists when the client has no chunk plan -- the planner had no size
        metadata for it. Callers fall back to per-key transfers, which is the behaviour
        that predates B1.
        """
        return self.pull_chunk_plan.get(pull_client, ([], []))

    def get_train_ps_read_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PS client and key (sending to PUSH_SIDE)"""
        result = {}
        for pull_client, key_plans in self.train_pull_from_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[pull_client] = key_plans[key][ps_client]
        return result


class CommunicationPlanner:
    """Intelligent communication planning with load balancing"""

    #: Default chunk budget. 512MB is the design's recommendation for a 170 GB/s node
    #: (3.5.6): large enough that `check_xfer_state` polling is a small fraction of the
    #: per-chunk transfer time, small enough to leave the pipeline several rounds.
    #: B3 replaces this constant with the cost model.
    DEFAULT_CHUNK_BYTES = 512 * 1024 * 1024

    def __init__(self, restrict_client_group_comm: bool = False, chunk_bytes: int | None = None):
        """Initialize the communication planner"""
        # NOTE(lhy): Restrict communication to matching client groups when enabled.
        self.restrict_client_group_comm = restrict_client_group_comm
        # None means "ask the cost model per client" (B3). A value pins every client
        # to that budget, which is what the chunk-size sweep needs.
        self.chunk_bytes = chunk_bytes

    def make_comm_plan(self, clients: dict[str, NIXLClientInfo]) -> NIXLCommPlan:
        """
        Generate communication plan for all clients

        Args:
            clients: Dictionary mapping client names to client info

        Returns:
            NIXLCommPlan: Generated communication plan
        """
        # Register all clients to network topology
        for client_name, client_info in clients.items():
            GLOBAL_TOPOLOGY.register_client(client_name, client_info.node_ip, client_info.node_gpu_id)

        # Classify clients
        push_client_groups = {}
        ps_for_push_client_groups = {}
        rollout_pull_client_groups = {}
        ps_for_rollout_pull_client_groups = {}
        train_pull_client_groups = {}
        ps_for_train_pull_client_groups = {}

        for client_name, client_info in clients.items():
            if client_info.type == NIXLClientType.PUSH_SIDE:
                if client_info.client_group_id not in push_client_groups:
                    push_client_groups[client_info.client_group_id] = []
                push_client_groups[client_info.client_group_id].append(client_name)
                if client_info.client_group_id not in train_pull_client_groups:
                    train_pull_client_groups[client_info.client_group_id] = []
                train_pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PUSH:
                if client_info.client_group_id not in ps_for_push_client_groups:
                    ps_for_push_client_groups[client_info.client_group_id] = []
                ps_for_push_client_groups[client_info.client_group_id].append(client_name)
                if client_info.client_group_id not in ps_for_train_pull_client_groups:
                    ps_for_train_pull_client_groups[client_info.client_group_id] = []
                ps_for_train_pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PULL_SIDE:
                if client_info.client_group_id not in rollout_pull_client_groups:
                    rollout_pull_client_groups[client_info.client_group_id] = []
                rollout_pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PULL:
                if client_info.client_group_id not in ps_for_rollout_pull_client_groups:
                    ps_for_rollout_pull_client_groups[client_info.client_group_id] = []
                ps_for_rollout_pull_client_groups[client_info.client_group_id].append(client_name)
            else:
                raise ValueError(f"Unknown client type: {client_info.type}")

        # Initialize communication plans
        push_to_ps_plan = {client: {} for client_group in push_client_groups.values() for client in client_group}
        rollout_pull_from_ps_plan = {
            client: {} for client_group in rollout_pull_client_groups.values() for client in client_group
        }
        train_pull_from_ps_plan = {
            client: {} for client_group in train_pull_client_groups.values() for client in client_group
        }

        # Generate PUSH_SIDE -> PS_FOR_PUSH write plan
        if push_client_groups and ps_for_push_client_groups:
            self._make_push_to_ps_plan(clients, push_client_groups, ps_for_push_client_groups, push_to_ps_plan)

        # Generate PUSH_SIDE <- PS_FOR_PUSH read plan
        if train_pull_client_groups and ps_for_train_pull_client_groups:
            self._make_pull_from_ps_plan(
                clients, train_pull_client_groups, ps_for_train_pull_client_groups, train_pull_from_ps_plan
            )

        # Generate PULL_SIDE <- PS_FOR_PULL read plan
        if rollout_pull_client_groups and ps_for_rollout_pull_client_groups:
            self._make_pull_from_ps_plan(
                clients, rollout_pull_client_groups, ps_for_rollout_pull_client_groups, rollout_pull_from_ps_plan
            )
        elif rollout_pull_client_groups and push_client_groups:
            # D3: trainer direct supply. With no PS in the cluster the rollout reads
            # from the train clients themselves, which is the same planner with a
            # different source set rather than a second code path -- the thing being
            # removed is the intermediate copy, not the planning.
            #
            # Dispatching on "is there a PS" rather than on a mode flag keeps the two
            # arrangements from drifting: a cluster either has PS clients registered or
            # it does not, and that fact is already in `clients`.
            self._make_pull_from_ps_plan(
                clients, rollout_pull_client_groups, push_client_groups, rollout_pull_from_ps_plan
            )

        # B1: chunk plan for the rollout pull side, where the pipeline and the wire
        # buffer live. The push side writes whole shards into PS buffers and has no
        # chunk pipeline, so planning chunks for it would be dead metadata.
        pull_chunk_plan = self._build_pull_chunk_plan(
            clients, rollout_pull_from_ps_plan, self.chunk_bytes
        )

        return NIXLCommPlan(
            push_to_ps_plan=push_to_ps_plan,
            rollout_pull_from_ps_plan=rollout_pull_from_ps_plan,
            train_pull_from_ps_plan=train_pull_from_ps_plan,
            pull_chunk_plan=pull_chunk_plan,
        )

    def _build_pull_chunk_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        pull_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
        chunk_bytes: int | None,
    ) -> dict[str, tuple[list, list]]:
        """Derive each pull client's chunk plan from what it is scheduled to read.

        One commit group per key: that is the granularity the engine loader requires
        atomically (weight and its scale, a fused QKV). Chunks are then cut across those
        groups by byte budget, so a key larger than the budget spans several chunks and
        several small keys share one -- the many-to-many relationship B1 exists to
        express.

        Sizes come from the *reader's* own metadata rather than the sender's: that is
        what the reader will actually land, and it is the side whose buffer the chunk
        budget is protecting. A key the reader has no metadata for is skipped rather
        than guessed at, since a wrong size silently mis-cuts every later chunk.
        """
        chunk_plan: dict[str, tuple[list, list]] = {}

        for pull_client, key_plans in pull_plan.items():
            info = clients.get(pull_client)
            if info is None:
                continue

            groups: list[CommitGroup] = []
            shard_bytes: dict[tuple[str, tuple[int, ...]], int] = {}

            for key in sorted(key_plans):
                tensor_info = info.tensor_infos.get(key)
                if tensor_info is None:
                    continue
                shard_positions = {s: i for i, s in enumerate(tensor_info.sharding.shard_indices)}

                shards: list[tuple[str, tuple[int, ...]]] = []
                for shard_indices in key_plans[key].values():
                    for shard_idx in shard_indices:
                        local_pos = shard_positions.get(shard_idx)
                        if local_pos is None:
                            continue
                        ref = (key, shard_idx)
                        if ref in shard_bytes:
                            continue
                        shard_bytes[ref] = tensor_info.get_shard_size_bytes(local_pos)
                        shards.append(ref)

                if shards:
                    groups.append(CommitGroup(name=key, shards=tuple(shards)))

            if not groups:
                continue

            # B3: size the chunk from this client's actual payload rather than a
            # constant. The cost model balances two opposing pressures -- enough rounds
            # that format cost hides behind RDMA, chunks large enough that completion
            # polling is not most of the per-chunk time -- and both depend on how many
            # bytes and keys this particular reader has. A fixed budget is wrong in both
            # directions as the model or the parallelism changes.
            #
            # An explicit chunk_bytes still wins: tests and the chunk-size sweep
            # (verification item 15b) need to pin it.
            effective = chunk_bytes
            if effective is None:
                total_bytes = sum(shard_bytes.values())
                effective = choose_chunk_bytes(
                    model_bytes=total_bytes, n_grouped_keys=len(groups)
                ).chunk_bytes

            chunks, resolved = plan_chunks(groups, shard_bytes, chunk_bytes=effective)
            chunk_plan[pull_client] = (chunks, resolved)

        return chunk_plan

    def _make_push_to_ps_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        push_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        ps_for_push_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        push_to_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
    ):
        """Generate PUSH_SIDE to PS write plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_client_groups=push_client_groups,
            target_client_groups=ps_for_push_client_groups,
            comm_plan=push_to_ps_plan,
            is_push_to_ps=True,
        )

    def _make_pull_from_ps_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        pull_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        ps_for_pull_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
    ):
        """Generate PULL_SIDE from PS read plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_client_groups=ps_for_pull_client_groups,
            target_client_groups=pull_client_groups,
            comm_plan=pull_from_ps_plan,
            is_push_to_ps=False,
        )

    def _make_comm_plan_generic(
        self,
        clients: dict[str, NIXLClientInfo],
        source_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        target_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        comm_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
        is_push_to_ps: bool,
    ):
        """
        Generic communication plan generation with intelligent load balancing

        This method implements a custom sorting algorithm that prioritizes:
        1. Network connection quality (LOCAL > NVLINK > PCIE > IB > ETH)
        2. Current data volume (lower volume gets priority)

        Args:
            clients: Dictionary mapping client names to client info
            source_client_groups: Dict of source client groups (PUSH_SIDE or PS_FOR_PULL)
            target_client_groups: Dict of target client groups (PS_FOR_PUSH or PULL_SIDE)
            comm_plan: Communication plan to update
            is_push_to_ps: True if PUSH_SIDE to PS_FOR_PUSH, False if PS_FOR_PULL to PULL_SIDE
        """
        # Track data volume for each source client
        source_client_volumes = {
            client: 0.0 for client_group in source_client_groups.values() for client in client_group
        }

        restrict_target_to_source_client_group_mapping = {}
        if self.restrict_client_group_comm:
            # NOTE(lhy): Push traffic uses round robin group pairing, while pull
            # traffic remains unrestricted.
            if is_push_to_ps:
                for target_client_group_id in target_client_groups.keys():
                    source_client_group_ids = [
                        client_group_id
                        for client_group_id in source_client_groups.keys()
                        if client_group_id % len(target_client_groups)
                        == target_client_group_id % len(source_client_groups)
                    ]
                    assert len(source_client_group_ids) > 0, (
                        f"No source client group ids found for target client group "
                        f"(id: {target_client_group_id}, "
                        f"client_names: {target_client_groups[target_client_group_id]})"
                    )
                    restrict_target_to_source_client_group_mapping[target_client_group_id] = source_client_group_ids
            else:
                for target_client_group_id in target_client_groups.keys():
                    restrict_target_to_source_client_group_mapping[target_client_group_id] = list(
                        source_client_groups.keys()
                    )
        else:
            for target_client_group_id in target_client_groups.keys():
                restrict_target_to_source_client_group_mapping[target_client_group_id] = list(
                    source_client_groups.keys()
                )

        # Cache source clients by restriction set to avoid rescanning every target key.
        key_to_sources_by_restriction: dict[tuple, dict[str, list[str]]] = {}
        for target_client_group_id in target_client_groups.keys():
            restrict_ids = restrict_target_to_source_client_group_mapping[target_client_group_id]
            # Use a frozen tuple as cache key (group restriction sets are small and reused)
            restriction_key = tuple(sorted(restrict_ids))
            if restriction_key not in key_to_sources_by_restriction:
                key_to_srcs: dict[str, list[str]] = {}
                for src_group_id in restrict_ids:
                    for src_client in source_client_groups[src_group_id]:
                        for key in clients[src_client].tensor_infos:
                            key_to_srcs.setdefault(key, []).append(src_client)
                key_to_sources_by_restriction[restriction_key] = key_to_srcs

        # Lazily build {shard: local_pos} per (source_client, key) on first use:
        # O(1) lookup vs O(S) list.index() per call.
        shard_pos_maps: dict[tuple[str, str], dict] = {}  # (source_client, key) -> {shard: local_pos}

        def get_shard_pos_map(src: str, key: str) -> dict:
            cache_key = (src, key)
            if cache_key not in shard_pos_maps:
                shard_pos_maps[cache_key] = {
                    s: i for i, s in enumerate(clients[src].tensor_infos[key].sharding.shard_indices)
                }
            return shard_pos_maps[cache_key]

        # For each target client, process all its keys
        for target_client_group_id, target_client_group in target_client_groups.items():
            restrict_ids = restrict_target_to_source_client_group_mapping[target_client_group_id]
            restriction_key = tuple(sorted(restrict_ids))
            key_to_srcs = key_to_sources_by_restriction[restriction_key]

            for target_client in target_client_group:
                target_info = clients[target_client]

                for key, target_tensor_info in target_info.tensor_infos.items():
                    # Use the precomputed key->sources mapping and a set for O(1) discard,
                    # avoiding O(G × C_s) per-key scan and O(C_s) list.remove().
                    if key not in key_to_srcs:
                        client_keys_info = {}
                        for source_client_group_id in restrict_ids:
                            for source_client in source_client_groups[source_client_group_id]:
                                client_keys_info[source_client] = list(clients[source_client].tensor_infos.keys())
                        error_msg = (
                            f"No available key {key} in source client groups {source_client_groups}, "
                            f"which is required by target client {target_client}, "
                            f"keys of them are {client_keys_info}"
                        )
                        raise AssertionError(error_msg)

                    # Build a working set for this (target_client, key) assignment round:
                    # set.discard() is O(1) vs list.remove() O(C_s).
                    available_source_clients: set[str] = set(key_to_srcs[key])

                    # Get all shards needed by target client
                    needed_shards = set(target_tensor_info.sharding.shard_indices)
                    assigned_shards = set()

                    # Greedy assignment: prioritize source clients with
                    # optimal network connection and least data volume
                    while assigned_shards != needed_shards:
                        # Custom sorting: first by link type (LOCAL > NVLINK > PCIE > IB > ETH), then by data volume
                        def sort_key(source_client, target_client=target_client):
                            # Get link priority (higher is better)
                            link_priority = GLOBAL_TOPOLOGY.get_link_priority(source_client, target_client)
                            # We want best first, so we negate the value
                            link_priority = -link_priority
                            # Secondary sort by data volume (lower is better)
                            volume = source_client_volumes[source_client]
                            return (link_priority, volume)

                        # Find source client with optimal network connection and minimum data volume
                        min_volume_client = min(available_source_clients, key=sort_key)

                        source_tensor_info = clients[min_volume_client].tensor_infos[key]

                        # Find shards this client can provide (and needed by the target client)
                        available_shards = (
                            set(source_tensor_info.sharding.shard_indices) - assigned_shards
                        ) & needed_shards
                        if not available_shards:
                            # O(1) set discard vs O(C_s) list.remove()
                            available_source_clients.discard(min_volume_client)
                            if not available_source_clients:
                                break
                            continue

                        # Assign shards
                        shards_to_assign = sorted(list(available_shards))
                        assigned_shards.update(shards_to_assign)

                        # Update plan
                        if is_push_to_ps:
                            # PUSH_SIDE -> PS_FOR_PUSH: push_client -> {key -> {ps_client -> shards}}
                            if key not in comm_plan[min_volume_client]:
                                comm_plan[min_volume_client][key] = {}
                            comm_plan[min_volume_client][key][target_client] = shards_to_assign
                        else:
                            # PS_FOR_PULL -> PULL_SIDE: pull_client -> {key -> {ps_client -> shards}}
                            if key not in comm_plan[target_client]:
                                comm_plan[target_client][key] = {}
                            comm_plan[target_client][key][min_volume_client] = shards_to_assign

                        # O(1) lookup via precomputed shard_pos_map vs O(S) list.index()
                        shard_pos_map = get_shard_pos_map(min_volume_client, key)
                        local_indices = [shard_pos_map[shard] for shard in shards_to_assign]
                        shard_size_bytes = sum(
                            source_tensor_info.get_shard_size_bytes(local_idx) for local_idx in local_indices
                        )
                        bandwidth_gbps = GLOBAL_TOPOLOGY.get_bandwidth_gbps(min_volume_client, target_client)
                        volume_increase = shard_size_bytes / (bandwidth_gbps * 1e9)  # Convert to time
                        source_client_volumes[min_volume_client] += volume_increase

                        # O(1) set discard vs O(C_s) list.remove()
                        available_source_clients.discard(min_volume_client)
                        if not available_source_clients:
                            break

                    # Verify all shards are assigned
                    assert assigned_shards == needed_shards, (
                        f"Not all shards assigned for key {key} on target client {target_client}. "
                        f"Needed shards: {needed_shards}. Assigned shards: {assigned_shards}."
                    )

    def _get_link_type_for_test(self, client1: str, client2: str):
        """Helper method for testing to get link type between clients"""
        return GLOBAL_TOPOLOGY.get_link_type(client1, client2)


# Global communication planner instance
global_comm_planner = CommunicationPlanner()

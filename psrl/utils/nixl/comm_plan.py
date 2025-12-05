"""
NIXL Communication Planning Module

This module provides intelligent load-balanced communication planning for NIXL,
analyzing network topology and tensor distribution to generate optimal
PUSH_SIDE to PS and PULL_SIDE from PS communication plans.
"""

import pickle
from dataclasses import dataclass

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
    pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

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

    def get_pull_plan(self, pull_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PULL_SIDE client and key"""
        return self.pull_from_ps_plan.get(pull_client, {}).get(key, {})

    def get_ps_write_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get write plan for specific PS client and key (receiving from PUSH_SIDE)"""
        result = {}
        for push_client, key_plans in self.push_to_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[push_client] = key_plans[key][ps_client]
        return result

    def get_ps_read_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PS client and key (sending to PULL_SIDE)"""
        result = {}
        for pull_client, key_plans in self.pull_from_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[pull_client] = key_plans[key][ps_client]
        return result


class CommunicationPlanner:
    """Intelligent communication planning with load balancing"""

    def __init__(self, restrict_client_group_comm: bool = False):
        """Initialize the communication planner"""
        # NOTE(lhy): if restrict_client_group_comm is True,
        # we will restrict communciation only happens between client groups
        self.restrict_client_group_comm = restrict_client_group_comm

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
        pull_client_groups = {}
        ps_for_pull_client_groups = {}

        for client_name, client_info in clients.items():
            if client_info.type == NIXLClientType.PUSH_SIDE:
                if client_info.client_group_id not in push_client_groups:
                    push_client_groups[client_info.client_group_id] = []
                push_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PUSH:
                if client_info.client_group_id not in ps_for_push_client_groups:
                    ps_for_push_client_groups[client_info.client_group_id] = []
                ps_for_push_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PULL_SIDE:
                if client_info.client_group_id not in pull_client_groups:
                    pull_client_groups[client_info.client_group_id] = []
                pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PULL:
                if client_info.client_group_id not in ps_for_pull_client_groups:
                    ps_for_pull_client_groups[client_info.client_group_id] = []
                ps_for_pull_client_groups[client_info.client_group_id].append(client_name)
            else:
                raise ValueError(f"Unknown client type: {client_info.type}")

        # Initialize communication plans
        push_to_ps_plan = {client: {} for client_group in push_client_groups.values() for client in client_group}
        pull_from_ps_plan = {client: {} for client_group in pull_client_groups.values() for client in client_group}

        # Generate PUSH_SIDE -> PS_FOR_PUSH write plan
        if push_client_groups and ps_for_push_client_groups:
            self._make_push_to_ps_plan(clients, push_client_groups, ps_for_push_client_groups, push_to_ps_plan)

        # Generate PULL_SIDE <- PS_FOR_PULL read plan
        if pull_client_groups and ps_for_pull_client_groups:
            self._make_pull_from_ps_plan(clients, pull_client_groups, ps_for_pull_client_groups, pull_from_ps_plan)

        return NIXLCommPlan(push_to_ps_plan=push_to_ps_plan, pull_from_ps_plan=pull_from_ps_plan)

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
            # NOTE(lhy): for pull we don't have any restriction
            # for push, we currently use round robin to map target client groups to source client groups
            # this should be improved to use a more intelligent mapping
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

        # For each target client, process all its keys
        for target_client_group_id, target_client_group in target_client_groups.items():
            for target_client in target_client_group:
                target_info = clients[target_client]

                for key, target_tensor_info in target_info.tensor_infos.items():
                    # Find all source clients with the same key
                    available_source_clients = []
                    restrict_source_client_group_ids = restrict_target_to_source_client_group_mapping[
                        target_client_group_id
                    ]
                    for source_client_group_id in restrict_source_client_group_ids:
                        for source_client in source_client_groups[source_client_group_id]:
                            source_info = clients[source_client]
                            if key in source_info.tensor_infos:
                                available_source_clients.append(source_client)
                    if not available_source_clients:
                        client_keys_info = {}
                        for source_client_group_id in restrict_source_client_group_ids:
                            for source_client in source_client_groups[source_client_group_id]:
                                client_keys_info[source_client] = list(clients[source_client].tensor_infos.keys())
                        error_msg = (
                            f"No available key {key} in source client groups {source_client_groups}, "
                            f"which is required by target client {target_client}, "
                            f"keys of them are {client_keys_info}"
                        )
                        raise AssertionError(error_msg)

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

                        source_info = clients[min_volume_client]
                        source_tensor_info = source_info.tensor_infos[key]

                        # Find shards this client can provide (and needed by the target client)
                        available_shards = (
                            set(source_tensor_info.sharding.shard_indices) - assigned_shards
                        ) & needed_shards
                        if not available_shards:
                            available_source_clients.remove(min_volume_client)
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

                        # Update data volume (considering bandwidth)
                        local_indices = [
                            source_tensor_info.sharding.shard_indices.index(shard) for shard in shards_to_assign
                        ]
                        shard_size_bytes = sum(
                            source_tensor_info.get_shard_size_bytes(local_idx) for local_idx in local_indices
                        )
                        bandwidth_gbps = GLOBAL_TOPOLOGY.get_bandwidth_gbps(min_volume_client, target_client)
                        volume_increase = shard_size_bytes / (bandwidth_gbps * 1e9)  # Convert to time
                        source_client_volumes[min_volume_client] += volume_increase

                        # Remove the client from the list of available clients
                        available_source_clients.remove(min_volume_client)
                        if not available_source_clients:
                            break

                    # Verify all shards are assigned
                    assert assigned_shards == needed_shards, (
                        f"Not all shards assigned for key {key} on target client {target_client}, \
                        needed_shards: {needed_shards}, assigned_shards: {assigned_shards}"
                    )

    def _get_link_type_for_test(self, client1: str, client2: str):
        """Helper method for testing to get link type between clients"""
        return GLOBAL_TOPOLOGY.get_link_type(client1, client2)


# Global communication planner instance
global_comm_planner = CommunicationPlanner()

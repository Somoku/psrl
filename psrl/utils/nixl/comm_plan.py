"""
NIXL Communication Planning Module

This module provides intelligent load-balanced communication planning for NIXL,
analyzing network topology and tensor distribution to generate optimal
PUSH_SIDE to PS and PULL_SIDE from PS communication plans.
"""

import pickle
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLClientInfo
from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY


@dataclass
class NIXLCommPlan:
    """Communication plan defining communication scheme for each client's keys"""
    
    # PUSH_SIDE -> PS write plan
    # {push_client: {key: {ps_client: [shard_indices]}}}
    push_to_ps_plan: Dict[str, Dict[str, Dict[str, List[int]]]]
    
    # PULL_SIDE <- PS read plan
    # {pull_client: {key: {ps_client: [shard_indices]}}}
    pull_from_ps_plan: Dict[str, Dict[str, Dict[str, List[int]]]]
    
    def serialize(self):
        """Serialize communication plan"""
        return pickle.dumps(self)
    
    @staticmethod
    def deserialize(data):
        """Deserialize communication plan"""
        return pickle.loads(data)
    
    def get_push_plan(self, push_client: str, key: str) -> Dict[str, List[int]]:
        """Get write plan for specific PUSH_SIDE client and key"""
        return self.push_to_ps_plan.get(push_client, {}).get(key, {})
    
    def get_pull_plan(self, pull_client: str, key: str) -> Dict[str, List[int]]:
        """Get read plan for specific PULL_SIDE client and key"""
        return self.pull_from_ps_plan.get(pull_client, {}).get(key, {})
    
    def get_ps_write_plan(self, ps_client: str, key: str) -> Dict[str, List[int]]:
        """Get write plan for specific PS client and key (receiving from PUSH_SIDE)"""
        result = {}
        for push_client, key_plans in self.push_to_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[push_client] = key_plans[key][ps_client]
        return result
    
    def get_ps_read_plan(self, ps_client: str, key: str) -> Dict[str, List[int]]:
        """Get read plan for specific PS client and key (sending to PULL_SIDE)"""
        result = {}
        for pull_client, key_plans in self.pull_from_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[pull_client] = key_plans[key][ps_client]
        return result


class CommunicationPlanner:
    """Intelligent communication planning with load balancing"""
    
    def __init__(self):
        """Initialize the communication planner"""
        pass
    
    def make_comm_plan(self, clients: Dict[str, NIXLClientInfo]) -> NIXLCommPlan:
        """
        Generate communication plan for all clients
        
        Args:
            clients: Dictionary mapping client names to client info
            
        Returns:
            NIXLCommPlan: Generated communication plan
        """
        # Register all clients to network topology
        for client_name, client_info in clients.items():
            GLOBAL_TOPOLOGY.register_client(client_name, client_info.ip, client_info.gpu_id)
        
        # Classify clients
        push_clients = []
        ps_for_push_clients = []
        pull_clients = []
        ps_for_pull_clients = []
        
        for client_name, client_info in clients.items():
            if client_info.type == NIXLClientType.PUSH_SIDE:
                push_clients.append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PUSH:
                ps_for_push_clients.append(client_name)
            elif client_info.type == NIXLClientType.PULL_SIDE:
                pull_clients.append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PULL:
                ps_for_pull_clients.append(client_name)
        
        # Initialize communication plans
        push_to_ps_plan = {client: {} for client in push_clients}
        pull_from_ps_plan = {client: {} for client in pull_clients}
        
        # Generate PUSH_SIDE -> PS_FOR_PUSH write plan
        if push_clients and ps_for_push_clients:
            self._make_push_to_ps_plan(clients, push_clients, ps_for_push_clients, push_to_ps_plan)
        
        # Generate PULL_SIDE <- PS_FOR_PULL read plan
        if pull_clients and ps_for_pull_clients:
            self._make_pull_from_ps_plan(clients, pull_clients, ps_for_pull_clients, pull_from_ps_plan)
        
        return NIXLCommPlan(
            push_to_ps_plan=push_to_ps_plan,
            pull_from_ps_plan=pull_from_ps_plan
        )
    
    def _make_push_to_ps_plan(
        self, 
        clients: Dict[str, NIXLClientInfo], 
        push_clients: List[str], 
        ps_for_push_clients: List[str],
        push_to_ps_plan: Dict[str, Dict[str, Dict[str, List[int]]]],
    ):
        """Generate PUSH_SIDE to PS write plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_clients=push_clients,
            target_clients=ps_for_push_clients,
            comm_plan=push_to_ps_plan,
            is_push_to_ps=True
        )
    
    def _make_pull_from_ps_plan(
        self, 
        clients: Dict[str, NIXLClientInfo],
        pull_clients: List[str], 
        ps_for_pull_clients: List[str],
        pull_from_ps_plan: Dict[str, Dict[str, Dict[str, List[int]]]]
    ):
        """Generate PULL_SIDE from PS read plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_clients=ps_for_pull_clients,
            target_clients=pull_clients,
            comm_plan=pull_from_ps_plan,
            is_push_to_ps=False
        )
    
    def _make_comm_plan_generic(
        self,
        clients: Dict[str, NIXLClientInfo],
        source_clients: List[str],
        target_clients: List[str],
        comm_plan: Dict[str, Dict[str, Dict[str, List[int]]]],
        is_push_to_ps: bool
    ):
        """
        Generic communication plan generation with intelligent load balancing
        
        This method implements a custom sorting algorithm that prioritizes:
        1. Network connection quality (LOCAL > NVLINK > PCIE > IB > ETH)
        2. Current data volume (lower volume gets priority)
        
        Args:
            clients: Dictionary mapping client names to client info
            source_clients: List of source client names (PUSH_SIDE or PS_FOR_PULL)
            target_clients: List of target client names (PS_FOR_PUSH or PULL_SIDE)
            comm_plan: Communication plan to update
            is_push_to_ps: True if PUSH_SIDE to PS_FOR_PUSH, False if PS_FOR_PULL to PULL_SIDE
        """
        # Track data volume for each source client
        source_client_volumes = {client: 0.0 for client in source_clients}
        
        # For each target client, process all its keys
        for target_client in target_clients:
            target_info = clients[target_client]
            
            for key, target_tensor_info in target_info.tensor_infos.items():
                # Find all source clients with the same key
                available_source_clients = []
                for source_client in source_clients:
                    source_info = clients[source_client]
                    if key in source_info.tensor_infos:
                        available_source_clients.append(source_client)
                
                # Get all shards needed by target client
                needed_shards = set(target_tensor_info.sharding.shard_indices)
                assigned_shards = set()
                
                # Greedy assignment: prioritize source clients with optimal network connection and least data volume
                while assigned_shards != needed_shards:
                    # Custom sorting: first by link type (LOCAL > NVLINK > PCIE > IB > ETH), then by data volume
                    def sort_key(source_client):
                        # Get link priority (higher is better)
                        link_priority = GLOBAL_TOPOLOGY.get_link_priority(source_client, target_client)
                        # We want best first, so we negate the value
                        link_priority = -link_priority
                        # Secondary sort by data volume (lower is better)
                        volume = source_client_volumes[source_client]
                        return (link_priority, volume)
                    
                    # Find source client with optimal network connection and minimum data volume
                    min_volume_client = min(
                        available_source_clients, 
                        key=sort_key
                    )
                    
                    source_info = clients[min_volume_client]
                    source_tensor_info = source_info.tensor_infos[key]
                    
                    # Find shards this client can provide (and needed by the target client)
                    available_shards = (set(source_tensor_info.sharding.shard_indices) - assigned_shards) & needed_shards
                    if not available_shards:
                        available_source_clients.remove(min_volume_client)
                        if not available_source_clients:
                            break
                        continue
                    
                    # Assign shards
                    shards_to_assign = list(available_shards)
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
                    local_indices = [source_tensor_info.sharding.shard_indices.index(shard) for shard in shards_to_assign]
                    shard_size_bytes = sum(source_tensor_info.get_shard_size_bytes(local_idx) for local_idx in local_indices)
                    bandwidth_gbps = GLOBAL_TOPOLOGY.get_bandwidth_gbps(min_volume_client, target_client)
                    volume_increase = shard_size_bytes / (bandwidth_gbps * 1e9)  # Convert to time
                    source_client_volumes[min_volume_client] += volume_increase
                    
                    # Remove the client from the list of available clients
                    available_source_clients.remove(min_volume_client)
                    if not available_source_clients:
                        break
                
                # Verify all shards are assigned
                assert assigned_shards == needed_shards, f"Not all shards assigned for key {key} on PS client {target_client}, \
                    needed_shards: {needed_shards}, assigned_shards: {assigned_shards}"
    
    def _get_link_type_for_test(self, client1: str, client2: str):
        """Helper method for testing to get link type between clients"""
        return GLOBAL_TOPOLOGY.get_link_type(client1, client2)


# Global communication planner instance
global_comm_planner = CommunicationPlanner() 
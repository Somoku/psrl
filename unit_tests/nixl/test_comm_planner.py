#!/usr/bin/env python3
"""
Comprehensive test for communication planner with custom sorting algorithm
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from psrl.utils.nixl.network_topology import NetworkTopology, LinkType
from psrl.utils.nixl.comm_plan import (
    CommunicationPlanner, NIXLClientInfo, NIXLClientType, 
    NIXLTensorInfo, global_comm_planner
)


def create_test_tensor_desc_info_1():
    """Create a test tensor descriptor info"""
    # Simulate a tensor with 4 shards
    desc_bytes_list = [b"shard_0", b"shard_1", b"shard_2", b"shard_3"]
    return NIXLTensorInfo(
        desc_bytes_list=desc_bytes_list,
        shard_dim=0,
        shard_mesh=4,
        shard_indices=[0, 1],  # This client has shards 0 and 1
        dtype="float32",
        shape=[1000, 1000],
        element_size=4
    )
  
    
def create_test_tensor_desc_info_2():
    """Create a test tensor descriptor info"""
    # Simulate a tensor with 4 shards
    desc_bytes_list = [b"shard_0", b"shard_1", b"shard_2", b"shard_3"]
    return NIXLTensorInfo(
        desc_bytes_list=desc_bytes_list,
        shard_dim=0,
        shard_mesh=4,
        shard_indices=[1, 2],  # This client has shards 0 and 1
        dtype="float32",
        shape=[1000, 1000],
        element_size=4
    )
    
    
def create_test_tensor_desc_info_3():
    """Create a test tensor descriptor info"""
    # Simulate a tensor with 4 shards
    desc_bytes_list = [b"shard_0", b"shard_1", b"shard_2", b"shard_3"]
    return NIXLTensorInfo(
        desc_bytes_list=desc_bytes_list,
        shard_dim=0,
        shard_mesh=4,
        shard_indices=[0, 1, 2],  # This client has shards 0 and 1
        dtype="float32",
        shape=[1000, 1000],
        element_size=4
    )


def test_communication_planner():
    """Test the communication planner with custom sorting"""
    print("Testing communication planner with custom sorting...")
    
    # Create clients with different network topologies
    clients = {}
    
    # PUSH_SIDE clients
    # Client A: Same node as PS, same GPU (LOCAL)
    clients["push_A"] = NIXLClientInfo(
        name="push_A",
        ip="192.168.1.1",
        gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"push_meta_A"
    )
    
    # Client B: Same node as PS, different GPU (NVLINK)
    clients["push_B"] = NIXLClientInfo(
        name="push_B",
        ip="192.168.1.1",
        gpu_id=1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_2()},
        meta=b"push_meta_B"
    )
    
    # Client C: Same node as PS, CPU (PCIE)
    clients["push_C"] = NIXLClientInfo(
        name="push_C",
        ip="192.168.1.1",
        gpu_id=-1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"push_meta_C"
    )
    
    # Client D: Different node (IB)
    clients["push_D"] = NIXLClientInfo(
        name="push_D",
        ip="192.168.1.2",
        gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_3()},
        meta=b"push_meta_D"
    )
    
    # PS clients
    clients["ps_1"] = NIXLClientInfo(
        name="ps_1",
        ip="192.168.1.1",
        gpu_id=0,
        type=NIXLClientType.PS,
        tensor_infos={"weight": create_test_tensor_desc_info_3()},
        meta=b"ps_meta_1"
    )
    
    # PULL_SIDE clients
    clients["pull_1"] = NIXLClientInfo(
        name="pull_1",
        ip="192.168.1.3",
        gpu_id=0,
        type=NIXLClientType.PULL_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"pull_meta_1"
    )
    
    # Generate communication plan
    comm_plan = global_comm_planner.make_comm_plan(clients)
    
    print("\nGenerated communication plan:")
    print("=" * 50)
    
    # Print PUSH_SIDE -> PS plan
    print("PUSH_SIDE -> PS Plan:")
    for push_client, key_plans in comm_plan.push_to_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(push_client, ps_client)
                print(f"  {push_client} -> {ps_client}: {key} shards {shards} ({link_type.name})")
    
    print("\nPULL_SIDE <- PS Plan:")
    for pull_client, key_plans in comm_plan.pull_from_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(ps_client, pull_client)
                print(f"  {ps_client} -> {pull_client}: {key} shards {shards} ({link_type.name})")
    
    print("\nCommunication planner test completed!")


def test_network_topology_integration():
    """Test network topology integration with communication planner"""
    print("\nTesting network topology integration...")
    
    # Test link type determination
    topology = NetworkTopology()
    
    # Register test clients
    topology.register_client("client_A", "192.168.1.1", 0)
    topology.register_client("client_B", "192.168.1.1", 1)
    topology.register_client("client_C", "192.168.1.1", -1)
    topology.register_client("client_D", "192.168.1.2", 0)
    
    print("Link type determination:")
    print(f"  client_A -> client_A: {topology.get_link_type('client_A', 'client_A').name}")
    print(f"  client_A -> client_B: {topology.get_link_type('client_A', 'client_B').name}")
    print(f"  client_A -> client_C: {topology.get_link_type('client_A', 'client_C').name}")
    print(f"  client_A -> client_D: {topology.get_link_type('client_A', 'client_D').name}")
    
    print("\nBandwidth values:")
    print(f"  LOCAL: {topology.get_bandwidth_gbps('client_A', 'client_A')} Gbps")
    print(f"  NVLINK: {topology.get_bandwidth_gbps('client_A', 'client_B')} Gbps")
    print(f"  PCIE: {topology.get_bandwidth_gbps('client_A', 'client_C')} Gbps")
    print(f"  IB: {topology.get_bandwidth_gbps('client_A', 'client_D')} Gbps")
    
    print("\nNetwork topology integration test completed!")


if __name__ == "__main__":
    test_network_topology_integration()
    test_communication_planner()
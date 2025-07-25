import os
import time
import torch
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from psrl.utils.nixl.server_client import NIXLMetaServer, NIXLStorageClient


@ray.remote
class MetaServerActor:
    def __init__(self, server_name, listen_ip, listen_port, expected_clients):
        self.server = NIXLMetaServer(server_name, listen_ip, listen_port)
        self.expected_clients = expected_clients

    def init_is_ready(self):
        return True

    def wait_for_client_infos(self):
        self.server.wait_for_client_infos(self.expected_clients)
        return True

    def notify_all_client_infos_and_comm_plan(self):
        self.server.notify_all_client_infos_and_comm_plan()
        return True

    def shutdown(self):
        self.server.shutdown()

@ray.remote
class MetaClientActor:
    def __init__(self, client_name, server_name, server_ip, server_port, local_data):
        self.client = NIXLStorageClient(client_name, server_name, server_ip, server_port, use_gpu=False, mode="meta_server")
        tensors = {k: torch.zeros_like(torch.tensor(v, dtype=torch.float32)) for k, v in local_data.items()}
        self.client.register_local_tensors(tensors)
        self.tensors = tensors
        self.name = client_name

    def init_is_ready(self):
        return True

    def connect(self):
        self.client.connect_to_server()
        return True

    def sync_desc(self):
        self.client.wait_for_server_info()
        return True

    def do_client_read(self, target_client, key, tag):
        self.client.client_read(target_client, key, tag)
        self.client.wait(key, tag, "READ", target_client=target_client)
        return self.tensors[key].clone()

    def do_client_write(self, target_client, key, tag, new_data):
        self.tensors[key].copy_(torch.tensor(new_data, dtype=torch.float32))
        self.client.client_write(target_client, key, tag)
        self.client.wait(key, tag, "WRITE", target_client=target_client)
        return True

    def shutdown(self):
        self.client.shutdown()

    def set_tensor_value(self, key, new_data):
        self.tensors[key].copy_(torch.tensor(new_data, dtype=torch.float32))
        return True

    def get_tensor_value(self, key):
        return self.tensors[key].clone()

def test_nixl_meta_comm():
    ray.init(ignore_reinit_error=True)
    listen_ip = "29.210.128.48"
    listen_port = 23458
    state_dict_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    num_clients = 2
    server_name = "meta_server"
    client_name = "client"

    # Start meta server
    print("Starting meta server")
    ip_to_node_id = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes()}
    assert listen_ip in ip_to_node_id, f"listen_ip {listen_ip} not found in ray nodes"
    server = MetaServerActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ip_to_node_id[listen_ip],
            soft=False
        )
    ).remote(server_name, listen_ip, listen_port, num_clients)
    ray.get(server.init_is_ready.remote())

    # Start clients
    print("Starting clients")
    clients = []
    for i in range(num_clients):
        client = MetaClientActor.remote(f"{client_name}_{i}", server_name, listen_ip, listen_port, state_dict_data)
        clients.append(client)

    # Connect clients
    print("Connecting meta clients")
    ray.get([c.connect.remote() for c in clients])

    # Meta server waits for all clients and notifies them
    print("Waiting for meta clients")
    ray.get(server.wait_for_client_infos.remote())
    ray.get(server.notify_all_client_infos_and_comm_plan.remote())

    # Client syncs descs
    print("Syncing meta descs")
    ray.get([c.sync_desc.remote() for c in clients])

    # Each client writes its own tensor to a known value
    print("Initializing client tensors")
    for idx, c in enumerate(clients):
        for key in state_dict_data:
            new_data = [v + idx + 10 for v in state_dict_data[key]]
            ray.get(c.set_tensor_value.remote(key, new_data))

    # Client 0 reads its own tensor (local read)
    print("Client 0 reads its own tensor")
    for key, value in state_dict_data.items():
        expected = torch.tensor([v + 0 + 10 for v in value], dtype=torch.float32)
        tensor = ray.get(clients[0].get_tensor_value.remote(key))
        assert torch.allclose(tensor, expected), f"Local tensor read failed for {key}"

    # Client 0 reads from client 1 (remote read)
    print("Client 0 reads from client 1")
    for key, value in state_dict_data.items():
        expected = torch.tensor([v + 1 + 10 for v in value], dtype=torch.float32)
        tensor = ray.get(clients[0].do_client_read.remote(f"{client_name}_1", key, b"read1"))
        assert torch.allclose(tensor, expected), f"Meta client read failed for {key}"

    # Client 1 writes to client 0 (remote write)
    print("Client 1 writes to client 0")
    for key in state_dict_data:
        new_data = [v + 100 for v in state_dict_data[key]]
        ray.get(clients[1].do_client_write.remote(f"{client_name}_0", key, b"write1", new_data))
        # Client 0 should now have the new data (local read)
        tensor = ray.get(clients[0].get_tensor_value.remote(key))
        # print(f"tensor: {tensor}, new_data: {new_data}")
        assert torch.allclose(tensor, torch.tensor(new_data, dtype=torch.float32)), f"Meta client write failed for {key}"

    # Shutdown
    print("Shutting down meta clients and server")
    ray.get([c.shutdown.remote() for c in clients])
    ray.get(server.shutdown.remote())
    ray.shutdown()

if __name__ == "__main__":
    test_nixl_meta_comm() 
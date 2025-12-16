import ray
import torch
from psrl.utils.nixl.server_client import NIXLStorageClient, NIXLStorageServer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote
class ServerActor:
    def __init__(
        self,
        server_name,
        listen_ip,
        listen_port,
        cuda,
        state_dict_data,
        expected_clients,
    ):
        self.server = NIXLStorageServer(server_name, listen_ip, listen_port, cuda)
        # Construct state dict
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in state_dict_data.items()}
        self.server.register_state_dict(state_dict)
        self.expected_clients = expected_clients
        self.state_dict = state_dict

    def wait_for_client_infos(self):
        self.server.wait_for_client_infos(self.expected_clients)
        return True

    def notify_all_client_infos(self):
        self.server.notify_all_client_infos()
        return True

    def get_state(self, key):
        return self.server.state_dict[key].clone()

    def update_state(self, key, tensor_data):
        self.server.state_dict[key].copy_(torch.tensor(tensor_data, dtype=torch.float32))
        return True

    def shutdown(self):
        self.server.shutdown()


@ray.remote
class ClientActor:
    def __init__(self, client_name, server_name, server_ip, server_port, cuda, local_data):
        self.client = NIXLStorageClient(
            client_name,
            server_name,
            server_ip,
            server_port,
            cuda,
            mode="storage_server",
        )
        # Register local tensors
        tensors = {k: torch.zeros_like(torch.tensor(v, dtype=torch.float32)) for k, v in local_data.items()}
        self.client.register_local_tensors(tensors)
        self.tensors = tensors

    def connect(self):
        self.client.connect_to_server()
        return True

    def sync_desc(self):
        self.client.wait_for_server_info()
        return True

    def do_read(self, key, tag):
        self.client.read(key, tag)
        self.client.wait(key, tag, "READ")
        return self.tensors[key].clone()

    def do_write(self, key, tag, new_data):
        self.tensors[key].copy_(torch.tensor(new_data, dtype=torch.float32))
        self.client.write(key, tag)
        self.client.wait(key, tag, "WRITE")
        return True

    def shutdown(self):
        self.client.shutdown()


def test_nixl_comm():
    ray.init(ignore_reinit_error=True)
    listen_ip = "29.210.128.48"
    listen_port = 23457
    cuda = -1
    state_dict_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    num_clients = 2
    server_name = "server"
    client_name = "client"

    # Start server
    print("Starting server")
    ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
    assert listen_ip in ip_to_node_id, f"listen_ip {listen_ip} not found in ray nodes"
    server = ServerActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ip_to_node_id[listen_ip],  # Use the first node's ID
            soft=False,
        )
    ).remote(server_name, listen_ip, listen_port, cuda, state_dict_data, num_clients)

    # Start clients
    print("Starting clients")
    clients = []
    for i in range(num_clients):
        client = ClientActor.remote(
            f"{client_name}_{i}",
            server_name,
            listen_ip,
            listen_port,
            cuda,
            state_dict_data,
        )
        clients.append(client)

    # Connect clients
    print("Connecting clients")
    ray.get([c.connect.remote() for c in clients])

    # Server waits for all clients and notifies them
    print("Waiting for clients")
    ray.get(server.wait_for_client_infos.remote())
    ray.get(server.notify_all_client_infos.remote())

    # Client syncs descs
    print("Syncing descs")
    ray.get([c.sync_desc.remote() for c in clients])

    # Each client reads from server and verifies
    print("Reading from server")
    for c in clients:
        for key, value in state_dict_data.items():
            tensor = ray.get(c.do_read.remote(key, b"read1"))
            assert torch.allclose(tensor, torch.tensor(value, dtype=torch.float32)), f"Client read failed for {key}"

    # Each client writes new data to server
    print("Writing to server")
    for idx, c in enumerate(clients):
        for key in state_dict_data:
            new_data = [v + idx + 10 for v in state_dict_data[key]]
            ray.get(c.do_write.remote(key, b"write1", new_data))
            # Server should now have the new data
            server_tensor = ray.get(server.get_state.remote(key))
            assert torch.allclose(server_tensor, torch.tensor(new_data, dtype=torch.float32)), (
                f"Server write failed for {key}"
            )

    # Shutdown
    print("Shutting down")
    ray.get([c.shutdown.remote() for c in clients])
    ray.get(server.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    test_nixl_comm()

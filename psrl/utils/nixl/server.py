import logging
import os
import pickle
import time
from copy import deepcopy

import torch
from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.logger import deprecated
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLCommPlan
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
    NIXLTensorInfo,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@deprecated("Use NIXLMetaServer instead")
class NIXLStorageServer:
    """
    NIXL initiator (server): holds the state dict, registers tensors, and notifies all clients with its descs.
    """

    def __init__(self, server_name: str, server_ip: str, server_port: int = 23456, cuda: int = -1):
        self.server_name = server_name
        self.server_ip = server_ip
        self.server_port = server_port
        self.cuda = cuda
        self.state_dict: dict[str, torch.Tensor] = {}
        self.tensor_infos: dict[str, NIXLTensorInfo] = {}
        self.agent = nixl_agent(self.server_name, nixl_agent_config(True, True, self.server_port))
        self.client_infos: set[str] = set()
        self._init_device()

    def _init_device(self):
        if self.cuda >= 0:
            torch.set_default_device(f"cuda:{self.cuda}")
        else:
            torch.set_default_device("cpu")

    def register_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """
        Register each tensor in the state_dict with NIXL. Build key->desc mapping.
        """
        self.state_dict = state_dict
        for key, tensor in state_dict.items():
            desc = self.agent.register_memory([tensor])
            if not desc:
                raise RuntimeError(f"Memory registration failed for key {key}.")
            desc_bytes = self.agent.get_serialized_descs(desc)
            self.tensor_infos[key] = NIXLTensorInfo(
                desc_bytes_list=[desc_bytes],
                shard_dim=-1,
                shard_mesh=1,
                shard_indices=[0],
            )

    def wait_for_client_infos(self, expected_clients: int = 1, timeout: float = 600.0):
        """
        Wait for all clients to connect and synchronize metadata.
        """
        start = time.time()
        while len(self.client_infos) < expected_clients:
            notifs = self.agent.get_new_notifs()
            for client_name in notifs:
                self.client_infos.add(client_name)
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for clients.")
            time.sleep(0.1)

    def get_serialized_descs(self) -> dict[str, bytes]:
        """
        Return a dict mapping key to serialized desc for all tensors.
        """
        return {k: info.desc for k, info in self.tensor_infos.items()}

    def notify_all_client_infos(self):
        """
        Notify all connected clients with the server's infos.
        """
        # Use ClientInfo to serialize
        for client in self.client_infos:
            info = NIXLClientInfo(
                name=self.server_name,
                type=NIXLClientType.PS,
                tensor_infos=self.tensor_infos,
                meta=self.agent.get_agent_metadata(),
            )
            self.agent.send_notif(client, info.serialize())

    def shutdown(self):
        [self.agent.remove_remote_agent(client) for client in self.client_infos]
        self.agent.invalidate_local_metadata(self.server_ip, self.server_port)
        for info in self.tensor_infos.values():
            self.agent.deregister_memory(info.get_desc(self.agent, 0))


class NIXLMetaServer:
    """
    NIXL meta server: only stores client meta and desc info, not state dict.
    """

    def __init__(self, server_name: str, nixl_config: DictConfig):
        self.server_name = server_name
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.agent = nixl_agent(self.server_name, nixl_agent_config(True, True, self.server_port))
        self.connected_clients: dict[
            str, list[str]
        ] = {}  # agent_name -> [client_name1, client_name2, ...], one agent can bind to multiple clients
        self.client_sharding_dicts: dict[str, dict[str, NIXLSharding]] = {}
        self.client_infos: dict[str, NIXLClientInfo] = {}

        self.client_unified_sharding_dicts: dict[str, dict[str, NIXLSharding]] = {}
        self.comm_plan: NIXLCommPlan | None = None
        self._client_temp_mappings: dict[str, dict] = {}

        self._is_all_client_shardings_recved = False
        self._is_all_client_infos_recved = False
        self._is_all_temp_mappings_recved = False

    def _add_client(self, agent_name: str, client_name: str):
        if agent_name not in self.connected_clients:
            self.connected_clients[agent_name] = []
        if client_name not in self.connected_clients[agent_name]:
            self.connected_clients[agent_name].append(client_name)

    def wait_for_client_shardings(self, expected_agents: int = 1, timeout: float = 600.0):
        """
        Wait for all agents to connect and send sharding.
        """
        psrl_logger.info(f"Waiting for {expected_agents} agents to connect and send sharding...")
        if self._is_all_client_shardings_recved:
            # TODO(lhy): support elastic adding new clients after all clients are connected
            return True
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_shardings = pickle.loads(notif)
                        assert isinstance(multi_shardings, dict), (
                            f"Expected a dict of multi_shardings, but got {multi_shardings}"
                        )
                        for client_name, sharding_dict in multi_shardings.items():
                            self.client_sharding_dicts[client_name] = sharding_dict
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.1)
        self._is_all_client_shardings_recved = True
        psrl_logger.info(
            f"All {len(self.client_sharding_dicts)} clients of {expected_agents} agents "
            f"sent sharding after {time.time() - start} seconds."
        )

    def wait_for_client_infos(self, expected_agents: int = 1, timeout: float = 600.0):
        """
        Wait for all agents to connect and send client infos.
        """
        psrl_logger.info(f"Waiting for {expected_agents} agents to send client infos...")
        if self._is_all_client_infos_recved:
            # TODO(lhy): support elastic adding new clients after all clients are connected
            return True
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_infos = pickle.loads(notif)
                        assert isinstance(multi_infos, dict), f"Expected a dict of multi_infos, but got {multi_infos}"
                        for client_name, info in multi_infos.items():
                            self.client_infos[client_name] = NIXLClientInfo.deserialize(info)
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.1)
        self._is_all_client_infos_recved = True
        psrl_logger.info(
            f"All {len(self.client_infos)} clients of {expected_agents} agents "
            f"sent client infos after {time.time() - start} seconds."
        )

    def wait_for_client_temp_mappings(self, expected_agents: int = 1, timeout: float = 600.0):
        """
        Wait for all agents to send temporary mappings.
        """
        psrl_logger.info(f"Waiting for {expected_agents} agents to send temp mappings...")
        if self._is_all_temp_mappings_recved:
            return True
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_temp_mappings = pickle.loads(notif)
                        assert isinstance(multi_temp_mappings, dict), (
                            f"Expected a dict of multi_temp_mappings, but got {multi_temp_mappings}"
                        )
                        for client_name, temp_mapping in multi_temp_mappings.items():
                            self._client_temp_mappings[client_name] = temp_mapping
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents temp mappings.")
            time.sleep(0.1)
        self._is_all_temp_mappings_recved = True
        psrl_logger.info(
            f"All {len(self._client_temp_mappings)} clients of {expected_agents} agents "
            f"sent temp mappings after {time.time() - start} seconds."
        )

    def make_unified_sharding(self):
        """
        Make unified sharding for all clients.
        """
        assert self._is_all_client_shardings_recved, "Not all clients sent sharding yet."
        assert not self.client_unified_sharding_dicts, "Unified sharding already made."
        # We first need to guarantee that all client shardings have the same keys
        all_keys = set()
        for client_name, sharding_dict in self.client_sharding_dicts.items():
            all_keys.update(sharding_dict.keys())
        # Then we can make the unified sharding for each client
        # That is, for each key, we need to find the new representation
        # of (shard_dim, shard_mesh, shard_indices) for the mutual slice of all clients
        for key in all_keys:
            shard_mesh_list = []
            for client_name, sharding_dict in self.client_sharding_dicts.items():
                if key not in sharding_dict:
                    # raise RuntimeError(f"Key {key} not found in sharding of client {client_name}.")
                    # This handle the case that some clients do not have the key (pipeline parallelism),
                    # but we can still make the unified sharding
                    sharding_dict[key] = NIXLSharding.empty()
                shard_mesh_list.append(sharding_dict[key].shard_mesh)
            finest_shard_mesh = NIXLSharding.find_finest_shard_mesh(shard_mesh_list)
            for client_name, sharding_dict in self.client_sharding_dicts.items():
                if client_name not in self.client_unified_sharding_dicts:
                    self.client_unified_sharding_dicts[client_name] = {}
                self.client_unified_sharding_dicts[client_name][key] = deepcopy(sharding_dict[key])
                self.client_unified_sharding_dicts[client_name][key].refactor_based_on_finer_shard_mesh(
                    finest_shard_mesh
                )

    def make_comm_plan(self):
        """
        Make communication plan for all clients.
        """
        assert self._is_all_client_infos_recved, "Not all clients sent client infos yet."
        assert not self.comm_plan, "Communication plan already made."

        psrl_logger.info("Making communication plan...")
        start = time.time()
        self.comm_plan = CommunicationPlanner().make_comm_plan(self.client_infos)
        psrl_logger.info(f"Communication plan made after {time.time() - start} seconds.")

    def notify_all_client_shardings(self):
        """
        Notify all connected clients with their sharding.
        """
        assert self._is_all_client_shardings_recved, "Not all clients sent sharding yet."
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

    def notify_all_client_infos_and_comm_plan(self):
        """
        Notify all connected clients with all client infos and optional comm plan.
        """
        assert self._is_all_client_infos_recved, "Not all clients sent client infos yet."
        assert self.comm_plan, "Communication plan not made yet."
        # Prepare notification data
        notification_data = {
            "client_infos": {
                client_name: self.client_infos[client_name].serialize() for client_name in self.client_infos
            },
            "comm_plan": self.comm_plan.serialize() if self.comm_plan else None,
        }
        payload = pickle.dumps(notification_data)

        for agent_name in self.connected_clients:
            # Send notification with client infos and optional comm plan
            self.agent.send_notif(agent_name, payload)

    def notify_all_client_temp_mappings(self):
        """
        Notify all connected clients with all temp mappings.
        """
        assert self._is_all_temp_mappings_recved, "Not all clients sent temp mappings yet."
        # Prepare notification data with all clients' temp mappings
        payload = pickle.dumps(self._client_temp_mappings)
        for agent_name in self.connected_clients:
            # Send notification with all temp mappings
            self.agent.send_notif(agent_name, payload)

    def wait_for_update_infos(self, expected_agents: int, timeout: float = 600.0):
        """
        Wait for expected number of agents to send updated client infos.
        """
        psrl_logger.debug(f"Waiting for {expected_agents} agents to send updated client infos...")
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_infos = pickle.loads(notif)
                        assert isinstance(multi_infos, dict), f"Expected a dict of multi_infos, but got {multi_infos}"
                        for client_name, info_and_temp_mapping in multi_infos.items():
                            info = info_and_temp_mapping["info"]
                            client_temp_mapping = info_and_temp_mapping["temp_mapping"]
                            client_info = NIXLClientInfo.deserialize(info)
                            self.client_infos[client_name] = client_info
                            self._client_temp_mappings[client_name] = client_temp_mapping
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception as e:
                        psrl_logger.error(f"Failed to parse updated client infos from agent {agent_name}: {e}")
                        raise
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.1)

        psrl_logger.info(
            f"{self.server_name}: Successfully received all {len(already_recved_agents)}/{expected_agents} "
            f"agents in {time.time() - start:.2f} seconds"
        )

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

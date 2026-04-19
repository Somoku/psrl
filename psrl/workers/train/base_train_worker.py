import logging
import os
import threading
import time
import ray
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from dataclasses import dataclass

from psrl.utils.logger import DualOutputHandler
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import ps_agent_name
from psrl.utils.nixl import NIXLInterface

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@dataclass
class TrainInterface:
    """Info for the PSRL TrainWorker."""

    ps_manager_handle: ray.actor.ActorHandle


# NOTE(lhy): This class is used to abstract the base train worker for PSRL.
# It is used to handle the NIXL push and pull operations.
# Cannot directly call this class, please use the derived classes instead.
class PSRL_BaseTrainWorker:
    def __init__(
        self,
        worker_rank: int,
        worker_world_size: int,
        psrl_config: DictConfig,
        train_interface: TrainInterface,
        nixl_interface: NIXLInterface,
    ):
        # Basic debug
        self.worker_rank = worker_rank
        self.worker_world_size = worker_world_size
        self.psrl_config = psrl_config
        self.train_interface = train_interface
        self.nixl_interface = nixl_interface
        # NIXL
        self.node_id = None
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        self._cached_ps_nixl_agent_names = None
        self._cached_ps_nixl_train_storage_client_names = None
        self._cached_ps_worker_handles: dict[str, ray.actor.ActorHandle] = {}
        # Cache for non-persistent named buffers fetched from PS (populated lazily).
        self._cached_non_persistent_buffers: dict[str, torch.Tensor] | None = None
        # NIXL wait threads
        self.nixl_wait_thread = None  # Single thread for all wait operations
        self.nixl_wait_thread_lock = threading.Lock()
        self.nixl_wait_completed = threading.Event()

        # Build logger
        self.log_prefix = f"BaseTrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {ray.get_runtime_context().get_node_id()}.")

        # Env debug
        # log_env_info(psrl_logger, level=logging.DEBUG)

    def get_node_id(self) -> str:
        """
        Get the node id of the train worker.
        """
        if self.node_id is not None:
            return self.node_id
        self.node_id = ray.get_runtime_context().get_node_id()
        return self.node_id

    @property
    def is_train_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        pass

    def get_replica_id(self) -> int:
        """
        Get the replica id (dp id) of the train worker.
        """
        pass

    def init_nixl_client(self):
        pass

    def nixl_protocol(self, mode: str = "full"):
        pass

    def nixl_sleep(self, mode: str = "full"):
        pass

    def ray_push_model(self) -> None:
        pass

    def nixl_push_model(self) -> None:
        """
        Push the model weights to the PS via NIXL.

        Usage example:
            # Start the push operation (this will start a background wait thread)
            worker.nixl_push_model()

            # Do other work while push is happening in background...

            # Wait for all push operations to complete
            success = worker.wait_for_nixl_push_completion(timeout=60.0)
            if success:
                print("All NIsXL push operations completed successfully")
            else:
                print("Some NIXL push operations timed out")

            # Or check thread status
            status = worker.get_nixl_wait_thread_status()
            print(f"Thread alive: {status.get('alive', False)}")
        """
        assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."
        assert self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
            "push_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode, "
            f"got: {self.psrl_config.ps_mode!r}."
        )
        ps_manager_handle = self.train_interface.ps_manager_handle
        psrl_logger.debug("Getting the current PS model version...")
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote(debug_info="base_train_worker"))
        next_ps_model_version = curr_ps_model_version + 1
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = ray.get(ps_manager_handle.get_ps_nixl_agent_names.remote())
        if self._cached_ps_nixl_train_storage_client_names is None:
            self._cached_ps_nixl_train_storage_client_names = ray.get(
                ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
            )
        psrl_logger.debug(
            f"Pushing the model with version {next_ps_model_version} to the PS "
            f"via NIXL on {len(self._cached_ps_nixl_train_storage_client_names)} clients."
        )

        # Clear previous wait thread
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is not None and self.nixl_wait_thread.is_alive():
                raise RuntimeError(
                    "Previous NIXL wait thread is still running, "
                    "you should wait for it to complete before calling nixl_push_model again."
                )
            self.nixl_wait_thread = None
            self.nixl_wait_completed.clear()

        # Start a single background thread to wait for all operations
        def wait_all_operations():
            try:
                # NOTE(lhy): Now we use a dict to store the PS handle and the key and shards to transfer and merge them on the PS side.
                # This is more efficient than calling transfer_train_to_gen for each key and shard, which will cause
                # a lot of remote calls and may cause the ray actor collapse.
                ps_handle_to_precision_transfer_key_and_shards_list: dict[str, list[tuple[str, list[tuple[int, ...]]]]] = {}
                psrl_logger.debug(f"Starting to push model to the PS via NIXL for version {next_ps_model_version}...")
                for key in self.unified_state_dict:
                    wait_operations = []
                    for target_agent_name, target_client_name in zip(
                        self._cached_ps_nixl_agent_names,
                        self._cached_ps_nixl_train_storage_client_names,
                    ):
                        if target_client_name not in self._cached_ps_worker_handles:
                            self._cached_ps_worker_handles[target_client_name] = ray.get(
                                ps_manager_handle.get_ps_worker_handle.remote(target_client_name)
                            )
                        psrl_logger.debug(
                            f"Pushing key {key} to {target_client_name} for version {next_ps_model_version}"
                        )
                        try:
                            shards_to_transfer = self.nixl_storage_client.client_write(
                                target_agent_name,
                                target_client_name,
                                key,
                                f"train_push_{next_ps_model_version}",
                            )
                            # shards_to_transfer = self.nixl_storage_client.client_write(
                            #     target_agent_name, target_client_name, key, "train_push"
                            # )
                        except Exception as e:
                            psrl_logger.error(
                                f"Error pushing key {key} to {target_client_name} "
                                f"for version {next_ps_model_version}: {e}"
                            )
                            raise e
                        if len(shards_to_transfer) > 0:
                            psrl_logger.debug(
                                f"Pushing key {key} shards {shards_to_transfer} to {target_client_name} "
                                f"for version {next_ps_model_version} with {len(shards_to_transfer)} shards"
                            )
                            wait_operations.append((key, target_client_name, shards_to_transfer))
                    psrl_logger.debug(
                        f"Starting to wait for {len(wait_operations)} NIXL operations "
                        f"for version {next_ps_model_version}..."
                    )
                    for wait_key, wait_target_client_name, wait_shards_to_transfer in wait_operations:
                        try:
                            self.nixl_storage_client.wait(
                                wait_key,
                                f"train_push_{next_ps_model_version}",
                                "WRITE",
                                target_client=wait_target_client_name,
                            )
                        except Exception as e:
                            psrl_logger.error(
                                f"Error waiting for key {wait_key} to target {wait_target_client_name} "
                                f"for version {next_ps_model_version}: {e}"
                            )
                            raise e
                        psrl_logger.debug(f"Wait completed for key {wait_key} to target {wait_target_client_name}")
                        if wait_target_client_name not in ps_handle_to_precision_transfer_key_and_shards_list:
                            ps_handle_to_precision_transfer_key_and_shards_list[wait_target_client_name] = []
                        ps_handle_to_precision_transfer_key_and_shards_list[wait_target_client_name].append((wait_key, wait_shards_to_transfer))
                precision_transfer_futures = []
                for target_client_name, precision_transfer_key_and_shards_list in ps_handle_to_precision_transfer_key_and_shards_list.items():
                    precision_transfer_futures.append(self._cached_ps_worker_handles[target_client_name].transfer_train_to_gen_merged.remote(precision_transfer_key_and_shards_list))
                ray.get(precision_transfer_futures)
                psrl_logger.debug("Starting to push model tag to the PS...")
                # Ensure all workers have completed the NIXL push operations and precision transfers
                assert dist.is_initialized(), "Pytorch distributed is not initialized."
                dist.barrier()
                psrl_logger.debug("Barrier done, now pushing model tag to the PS on the representative rank...")
                if self.worker_rank == 0:
                    # Only the representative rank pushes the model tag to the PS
                    ray.get(ps_manager_handle.push_model_state_dict_nixl.remote(next_ps_model_version))
                self.nixl_storage_client.clear_intermediate_cached_data()
                self.nixl_wait_completed.set()
                psrl_logger.debug(
                    f"All NIXL push operations completed, "
                    f"model with version {next_ps_model_version} is successfully pushed to the PS."
                )
            except Exception as e:
                raise RuntimeError(f"Error in NIXL wait thread: {e}") from e

        wait_thread = threading.Thread(target=wait_all_operations, daemon=True)
        wait_thread.start()
        # Store the thread reference
        with self.nixl_wait_thread_lock:
            self.nixl_wait_thread = wait_thread

    def wait_for_nixl_push_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for the NIXL push wait thread to complete.

        Args:
            timeout (float | None): Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            bool: True if the thread completed successfully, False if timeout occurred or thread failed.
        """
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is None:
                psrl_logger.debug("No NIXL wait thread to wait for.")
                return True

            psrl_logger.debug("Waiting for NIXL wait thread to complete...")
            if timeout is not None:
                # Use the event to wait with timeout
                if self.nixl_wait_completed.wait(timeout=timeout):
                    # Event was set, check if thread actually completed successfully
                    self.nixl_wait_thread.join(timeout=1.0)  # Brief join to catch any exceptions
                    if self.nixl_wait_thread.is_alive():
                        psrl_logger.warning("NIXL wait thread is still alive after event was set.")
                        return False
                    psrl_logger.debug("NIXL wait thread completed successfully.")
                    return True
                else:
                    psrl_logger.warning("Timeout waiting for NIXL wait thread to complete.")
                    return False
            else:
                # Wait indefinitely
                self.nixl_wait_thread.join()
                if self.nixl_wait_thread.is_alive():
                    psrl_logger.warning("NIXL wait thread is still alive after join.")
                    return False
                psrl_logger.debug("NIXL wait thread completed successfully.")
                return True

    def get_nixl_wait_thread_status(self) -> dict:
        """
        Get the status of the NIXL wait thread.

        Returns:
            dict: Dictionary containing thread status information.
        """
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is None:
                return {"has_thread": False, "alive": False, "completed": True}
            return {
                "has_thread": True,
                "alive": self.nixl_wait_thread.is_alive(),
                "completed": self.nixl_wait_completed.is_set(),
            }

    def push_model(self):
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            self.ray_push_model()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            # ---- DEBUG: log train info BEFORE push ----
            # self._debug_log_train_info(label=f"TRAIN_BEFORE_PUSH_R{self.worker_rank}")
            self.nixl_push_model()
            # TODO(lhy): wait for the push to complete before the next iteration optimizer update
            # This will enable the NIXL push to be overlapped with the next iteration training
            self.wait_for_nixl_push_completion()
            # ---- DEBUG: log PS info AFTER push completes ----
            # self._debug_log_ps_info(label=f"PS_AFTER_PUSH_R{self.worker_rank}")
        else:
            raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")

    def nixl_update_local_info_to_ps(self, ps_worker_node_id_to_idxs: dict[str, int]):
        """
        Update local NIXL info to the PS workers on the same node with this train worker.
        """
        node_id = self.get_node_id()
        dst_ps_worker_idx = ps_worker_node_id_to_idxs[node_id]
        dst_agent_names = [ps_agent_name(dst_ps_worker_idx), NIXL_META_SERVER_NAME]
        self.nixl_storage_client.send_local_info_to(dst_agent_names)

    def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the specified destination agent names.

        Args:
            dst_agent_names (str | list[str]): Destination agent name(s) to send local info to.
        """
        if isinstance(dst_agent_names, str):
            dst_agent_names = [dst_agent_names]
        self.nixl_storage_client.send_local_info_to(dst_agent_names)

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for infos of updated clients for global synchronization.

        Args:
            info_num (int): Number of infos to wait for.
        """
        self.nixl_storage_client.wait_for_update_infos(info_num)

    def nixl_pull_model(self):
        """Pull the model from the NIXL storage client."""
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", (
            "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        )
        ps_manager_handle = self.train_interface.ps_manager_handle
        # Cache the agent and client names to avoid redundant ray calls
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = ray.get(ps_manager_handle.get_ps_nixl_agent_names.remote())
        if self._cached_ps_nixl_train_storage_client_names is None:
            self._cached_ps_nixl_train_storage_client_names = ray.get(
                ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
            )
        self.nixl_pull_model_core(self._cached_ps_nixl_agent_names, self._cached_ps_nixl_train_storage_client_names)

    def nixl_pull_model_core(self, ps_nixl_agent_names: list[str], ps_nixl_train_storage_client_names: list[str]):
        """
        Core logic for pulling the model from NIXL storage clients.

        Args:
            ps_nixl_agent_names (list[str]): List of PS NIXL agent names
            ps_nixl_train_storage_client_names (list[str]): List of PS NIXL train storage client names
        """
        if not hasattr(self, "pull_times"):
            self.pull_times = 0
        self.pull_times += 1
        wait_operations = []
        time_start = time.time()
        for key in self.unified_state_dict:
            for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_train_storage_client_names):
                shards_to_transfer = self.nixl_storage_client.client_read(
                    target_agent_name, target_client_name, key, f"train_pull_{self.pull_times}"
                )
                # shards_to_transfer = self.nixl_storage_client.client_read(
                #     target_agent_name, target_client_name, key, "train_pull", merge_and_cache_xfer=False
                # )
                if len(shards_to_transfer) > 0:
                    psrl_logger.debug(
                        f"Pulling key {key} shards {shards_to_transfer} from {target_client_name} "
                        f"for pull {self.pull_times} times"
                    )
                    wait_operations.append((key, target_client_name, shards_to_transfer))
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        for key, target_client_name, shards_to_transfer in wait_operations:
            self.nixl_storage_client.wait(
                key, f"train_pull_{self.pull_times}", "READ", target_client=target_client_name
            )
            # self.nixl_storage_client.wait(key, "train_pull", "READ", target_client=target_client_name)
        self.nixl_storage_client.merge_and_finish_cached_xfer()
        psrl_logger.info(
            f"{self.nixl_storage_client}: NIXL pull model core done "
            f"({self.pull_times} times). time: {time.time() - time_start}s"
        )
        torch.cuda.synchronize()
        self.nixl_storage_client.clear_intermediate_cached_data()

    def pull_model(self):
        """Pull the model from the PS via the specified mode.

        Currently we do not support `cpu` and `cpu_ref` modes for pulling the model in trainer.
        """
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            raise RuntimeError("ray_pull_model is not supported for TrainWorker in 'cpu' or 'cpu_ref' mode.")
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            # ---- DEBUG: log PS info BEFORE pull ----
            # self._debug_log_ps_info(label=f"PS_BEFORE_PULL_R{self.worker_rank}")
            self.nixl_pull_model()
            # ---- DEBUG: log train info AFTER pull ----
            # self._debug_log_train_info(label=f"TRAIN_AFTER_PULL_R{self.worker_rank}")
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        # Restore non-persistent buffers (e.g. inv_freq) that are not transferred by NIXL pull.
        self._restore_non_persistent_buffers_from_ps()

    def _restore_non_persistent_buffers_from_ps(self) -> None:
        """
        Restore non-persistent buffers from PS after pull.
        Subclasses must override this method.
        """
        raise NotImplementedError

    def _get_any_ps_worker_handle(self) -> ray.actor.ActorHandle:
        """
        Return a handle to the PS storage worker on the same node, falling back
        to the first available worker if none is co-located.

        Result is cached in _cached_ps_worker_handles after first resolution.

        Returns:
            ray.actor.ActorHandle: A handle to a PS storage worker.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        my_node_id = self.get_node_id()
        # Prefer PS worker on the same node to avoid cross-node data transfer.
        client_name = ray.get(
            ps_manager_handle.get_ps_nixl_train_storage_client_name_for_node.remote(my_node_id)
        )
        if client_name is None:
            # Fallback: pick the first available PS client.
            if self._cached_ps_nixl_train_storage_client_names is None:
                self._cached_ps_nixl_train_storage_client_names = ray.get(
                    ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
                )
            client_name = self._cached_ps_nixl_train_storage_client_names[0]
            psrl_logger.warning(
                f"[_get_any_ps_worker_handle] No PS worker found on node {my_node_id}; "
                f"falling back to client {client_name}."
            )
        if client_name not in self._cached_ps_worker_handles:
            self._cached_ps_worker_handles[client_name] = ray.get(
                ps_manager_handle.get_ps_worker_handle.remote(client_name)
            )
        return self._cached_ps_worker_handles[client_name]

    def _get_non_persistent_buffers_from_ps(self) -> dict[str, torch.Tensor]:
        """
        Fetch non-persistent named buffers from the co-located PS storage worker.
        Result is cached after the first call.

        Returns:
            dict[str, torch.Tensor]: Mapping of dotted buffer name to CPU tensor.
        """
        if self._cached_non_persistent_buffers is not None:
            return self._cached_non_persistent_buffers
        ps_handle = self._get_any_ps_worker_handle()
        self._cached_non_persistent_buffers = ray.get(
            ps_handle.get_non_persistent_named_buffers.remote()
        )
        psrl_logger.debug(
            f"[_get_non_persistent_buffers_from_ps] Fetched "
            f"{len(self._cached_non_persistent_buffers)} non-persistent buffer(s) from PS."
        )
        return self._cached_non_persistent_buffers

    def _debug_log_train_info(self, label: str):
        """Debug log the train info."""
        if self.nixl_storage_client is not None:
            self.nixl_storage_client.log_shard_info(label=label)
        self._debug_log_train_model_info(label=label)

    def _debug_log_train_model_info(self, label: str):
        """Debug log the train model info."""
        pass

    def _debug_log_ps_info(self, label: str):
        """Call debug_log_info on every PSStorageWorker via Ray RPC (rank-0 only to reduce noise)."""
        if self.worker_rank != 0:
            return
        try:
            ps_manager_handle = self.train_interface.ps_manager_handle
            if self._cached_ps_nixl_train_storage_client_names is None:
                self._cached_ps_nixl_train_storage_client_names = ray.get(
                    ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
                )
            futures = []
            for target_client_name in self._cached_ps_nixl_train_storage_client_names:
                if target_client_name not in self._cached_ps_worker_handles:
                    self._cached_ps_worker_handles[target_client_name] = ray.get(
                        ps_manager_handle.get_ps_worker_handle.remote(target_client_name)
                    )
                ps_worker_handle = self._cached_ps_worker_handles[target_client_name]
                futures.append(ps_worker_handle.debug_log_info.remote(label=label))
            ray.get(futures)
        except Exception as e:
            psrl_logger.warning(f"[{label}] Failed to log PS shard info: {e}")

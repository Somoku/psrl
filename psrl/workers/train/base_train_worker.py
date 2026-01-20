import logging
import os
import threading
from dataclasses import dataclass

import ray
import torch.distributed as dist
from omegaconf import DictConfig

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
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        self._cached_ps_nixl_agent_names = None
        self._cached_ps_nixl_train_storage_client_names = None
        self._cached_ps_worker_handles: dict[str, ray.actor.ActorHandle] = {}
        # NIXL wait threads
        self.nixl_wait_thread = None  # Single thread for all wait operations
        self.nixl_wait_thread_lock = threading.Lock()
        self.nixl_wait_completed = threading.Event()

        # Build logger
        # self.log_prefix = f"BaseTrainWorker_R{self.rank}"
        # psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        # psrl_logger.debug(f"Initialized on {get_worker_info()}.")

        # Env debug
        # log_env_info(psrl_logger, level=logging.DEBUG)

    def get_node_id(self) -> str:
        """
        Get the node id of the train worker.
        """
        return ray.get_runtime_context().get_node_id()

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

    def nixl_protocol(self):
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
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", (
            "push_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
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
                precision_transfer_futures = []
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
                            wait_operations.append((key, target_client_name, shards_to_transfer))
                    psrl_logger.debug(
                        f"Starting to wait for {len(wait_operations)} NIXL operations "
                        f"for version {next_ps_model_version}..."
                    )
                    for key, target_client_name, shards_to_transfer in wait_operations:
                        try:
                            self.nixl_storage_client.wait(
                                key,
                                f"train_push_{next_ps_model_version}",
                                "WRITE",
                                target_client=target_client_name,
                            )
                            # self.nixl_storage_client.wait(
                            #     key, "train_push", "WRITE", target_client=target_client_name
                            # )
                        except Exception as e:
                            psrl_logger.error(
                                f"Error waiting for key {key} to target {target_client_name} "
                                f"for version {next_ps_model_version}: {e}"
                            )
                            raise e
                        psrl_logger.debug(f"Wait completed for key {key} to target {target_client_name}")
                        ps_worker_handle = self._cached_ps_worker_handles[target_client_name]
                        '''
                        precision_transfer_futures.append(
                            ps_worker_handle.transfer_train_to_gen.remote(key, shards_to_transfer)
                        )
                        '''
                        psrl_logger.debug(
                            f"Transfer {shards_to_transfer} shards of {key} "
                            f"from train to gen in target {target_client_name}"
                        )
                psrl_logger.debug(
                    f"Wait NIXL xfers done, start to wait for {len(precision_transfer_futures)} "
                    f"train to gen transfers on the PS..."
                )
                # ray.get(precision_transfer_futures)
                psrl_logger.debug("Starting to push model tag to the PS...")
                # Ensure all workers have completed the NIXL push operations and precision transfers
                assert dist.is_initialized(), "Pytorch distributed is not initialized."
                dist.barrier()
                psrl_logger.debug("Barrier done, now pushing model tag to the PS on the representative rank...")
                if self.worker_rank == 0:
                    # Only the representative rank pushes the model tag to the PS
                    ray.get(ps_manager_handle.push_model_state_dict_nixl.remote(next_ps_model_version))
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

    def wait_for_nixl_push_completion(self, timeout: float = None) -> bool:
        """
        Wait for the NIXL push wait thread to complete.

        Args:
            timeout (float, optional): Maximum time to wait in seconds. If None, wait indefinitely.

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
            self.nixl_push_model()
            # TODO(lhy): wait for the push to complete before the next iteration optimizer update
            # This will enable the NIXL push to be overlapped with the next iteration training
            self.wait_for_nixl_push_completion()
        else:
            raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")

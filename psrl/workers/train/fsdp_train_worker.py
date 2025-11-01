import ray
import os
import logging
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)

from psrl.workers.train import TrainInterface, PSRL_BaseTrainWorker
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_dual_events, log_single_event, EventType
from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLStorageClient, GLOBAL_META_SERVER_NAME, GLOBAL_TRAIN_CLIENT_NAME
from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        state_dict_config = StateDictOptions(
            full_state_dict=True, 
            cpu_offload=offload_to_cpu, 
            broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


class PSRL_FSDPTrainWorker(ActorRolloutRefWorker, PSRL_BaseTrainWorker):
    def __init__(
        self, 
        config: DictConfig, 
        role: str, 
        psrl_config: DictConfig, 
        train_interface: TrainInterface, 
        nixl_interface: NIXLInterface
    ) -> None:
        ActorRolloutRefWorker.__init__(self, config, role)
        PSRL_BaseTrainWorker.__init__(self, self.rank, self.world_size, psrl_config, train_interface, nixl_interface)
        
        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")
        
    @property   
    def is_train_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        return self.rank == 0
    
    def get_replica_id(self) -> int:
        """
        Get the replica id (dp id) of the train worker.
        """
        assert hasattr(self, "device_mesh"), "device_mesh is not initialized."
        if self.device_mesh.ndim <= 1:
            return 0
        return self.device_mesh.get_local_rank(mesh_dim=0)
    
    def init_nixl_client(self):
        """Initialize the NIXL client."""
        # NOTE(lhy): the init_nixl_client is called before the initialization of the actor module now
        # Because in UCX 1.18.0, this may enhance the communication performance
        # assert self.actor_module_fsdp, "The actor module must be initialized before calling init_nixl_client."
        if self.psrl_config.nixl.server_mode == "storage_server":
            raise ValueError("Storage server mode is deprecated.")
        elif self.psrl_config.nixl.server_mode == "meta_server":
            self.nixl_storage_client = NIXLStorageClient(
                client_name=f"{GLOBAL_TRAIN_CLIENT_NAME}_{self.rank}",
                server_name=GLOBAL_META_SERVER_NAME,
                use_gpu=True,
                client_type=NIXLClientType.PUSH_SIDE,
                nixl_config=self.psrl_config.nixl,
                nixl_interface=self.nixl_interface,
                # client_group_id=self.get_replica_id()
                logging_path=self.psrl_config.logging_path,
            )
        else:
            raise ValueError(f"Invalid NIXL server mode: {self.psrl_config.nixl.server_mode}")
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")
       
    def nixl_protocol(self):
        # Register the state dict and sharding dict to the NIXL client
        psrl_logger.info(f"nixl client protocol step 0: convert_fsdp_inplace")
        unified_state_dict, local_sharding_dict = convert_fsdp_inplace(self.config.actor.strategy, self.actor_module_fsdp)
        psrl_logger.info(f"nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info(f"nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(local_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        psrl_logger.info(f"nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(unified_state_dict, unified_sharding_dict)
        psrl_logger.info(f"nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info(f"nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info(f"nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info(f"nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info(f"nixl client protocol done.")
        self.unified_state_dict = unified_state_dict
        self.unified_sharding_dict = unified_sharding_dict
        
    def ray_push_model(self) -> None:
        """
        Push the model weights to the PS via ray. In 'cpu' mode, push the full state dict. In 'cpu_ref' mode, push a ray object_ref.
        In 'cpu' mode, the PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote(debug_info="fsdp_train_worker"))
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        # assert fsdp_version(self.actor_module_fsdp) == 1, "FSDP version 2 is not supported yet."
        psrl_logger.info(f"Gathering the full state dict on the CPU of the representative rank.")
        full_state_dict = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=True, rank0_only=True)
        if self.is_train_representative_rank:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on the representative worker."
            psrl_logger.info(f"Push the model via CPU on the representative rank (async).")
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, push the full state dict (PS worker will block on transfer)
                # But the training side does not need to wait for the push to complete, as it can be overlapped with the next-iteration training
                ps_manager_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, push a ray object_ref (PS worker is non-blocking)
                # But the training side needs to wait for the push to complete, as `ray.put` is blocking
                object_ref = ray.put(full_state_dict)  # This blocks until the state dict is in the object store
                ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(next_ps_model_version, [object_ref]) # Tricky part: manually wrap the object_ref in a list to avoid ray dereferencing the full state dict
            else:
                raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        else:
            # assert len(full_state_dict) == 0, "The model state dict should be empty on non-representative workers."
            # FSDP may combined with DDP now (HSDP), so the state dict may not be empty on non-representative workers.
            pass
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            ActorRolloutRefWorker.init_model(self)
            
    def _build_rollout(self, trust_remote_code: bool = False):
        pass
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def build_rollout(self, trust_remote_code: bool = False):
        ActorRolloutRefWorker._build_rollout(self, trust_remote_code=trust_remote_code)
    
    # The log_prob in training side may need to be recomputed
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        # NOTE(lhy): compared with verl, we replace `old_log_probs` with `recomputed_log_probs` in the output.
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        with log_dual_events("Recompute log_prob", psrl_logger, event_type=EventType.OTHER):
            assert self._is_actor
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)

            # Support all hardwares
            from contextlib import nullcontext

            is_lora = data.meta_info.pop("is_lora", False)
            adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
            data = data.to(get_device_id())
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                data = self.ulysses_sharding_manager.preprocess_data(data)
                with adapter_ctx:
                    output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
                output = DataProto.from_dict(
                    tensors={"recomputed_log_probs": output, "entropys": entropys}
                )
                output = self.ulysses_sharding_manager.postprocess_data(output)

            output = output.to("cpu")

            # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
            # unshard the root FSDP module
            if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
                self.actor.actor_module._handle.reshard(True)

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

            return output
                
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
            output = ActorRolloutRefWorker.update_actor(self, data)
        torch.cuda.synchronize()
        with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
            PSRL_BaseTrainWorker.push_model(self)
        return output
            
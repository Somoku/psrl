import logging
import os
from typing import TYPE_CHECKING

import ray
import torch
from contextlib import nullcontext
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from psrl.utils.common.patch_utils import apply_tms_patch
from psrl.utils.common.dynamic_import import lazy_import_to_globals
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    MemoryLogger,
    deprecated,
    get_worker_info,
    gpu_memory_logger_decorator,
    log_dual_events,
    log_tensor,
)
from psrl.utils.ray import exclusive_push_model_context
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import train_client_name
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLInterface,
    NIXLStorageClient,
)
from psrl.workers.train import PSRL_BaseTrainWorker, TrainInterface

if TYPE_CHECKING:
    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError:
        pass
else:
    torch_memory_saver = None

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
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only),
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state_dict_config = StateDictOptions(
            full_state_dict=True,
            cpu_offload=offload_to_cpu,
            broadcast_from_rank0=not rank0_only,
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
        nixl_interface: NIXLInterface,
    ) -> None:
        ActorRolloutRefWorker.__init__(self, config, role)
        PSRL_BaseTrainWorker.__init__(
            self,
            self.rank,
            self.world_size,
            psrl_config,
            train_interface,
            nixl_interface,
        )

        if self.psrl_config.tms.range in ["train", "all"]:
            lazy_import_to_globals("torch_memory_saver", "torch_memory_saver")
            apply_tms_patch()

        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

        # Memory logger (periodic + on-demand GPU memory log, same path/prefix pattern with Mem suffix)
        if torch.cuda.is_available() and self.psrl_config.memory_logger.enable:
            self.memory_logger = MemoryLogger(
                self.psrl_config.logging_path,
                f"{self.log_prefix}_Mem",
                interval_seconds=self.psrl_config.memory_logger.interval_seconds,
            )
            self.memory_logger.start()
        else:
            self.memory_logger = None

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
        self.nixl_storage_client = NIXLStorageClient(
            client_name=train_client_name(self.rank),
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PUSH_SIDE,
            nixl_config=self.psrl_config.nixl,
            nixl_interface=self.nixl_interface,
            # client_group_id=self.get_replica_id()
            logging_path=self.psrl_config.logging_path,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self):
        """Convert the FSDP model parameters for NIXL storage client."""
        from transformers import AutoConfig
        from verl.utils.fs import copy_to_local

        model_config = AutoConfig.from_pretrained(
            copy_to_local(self.config.model.path),
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        parameter_mapping = create_parameter_mapping("FSDP", model_config)
        self.unified_state_dict, self.local_sharding_dict = convert_fsdp_inplace(
            parameter_mapping,
            self.actor_module_fsdp,
            fsdp_strategy=self.config.actor.strategy,
        )

    def nixl_protocol(self, mode: str = "full"):
        """Run the NIXL server protocol.

        Args:
            mode (str): Mode of conversion, either 'meta' or 'full'.
                'meta' mode converts to meta tensors and skip registering them.
                'full' mode converts to full tensors.

            NOTE: ps storage may init with meta tensors, the register step would be different.
        """
        # Register the state dict and sharding dict to the NIXL client
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params()
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(self.local_sharding_dict)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(
            self.unified_state_dict, unified_sharding_dict, meta_only=meta_only
        )
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")
        self.unified_sharding_dict = unified_sharding_dict

    def clear_fsdp2_grads(self):
        """Clear all FSDP2 gradient state after wake_up/pull_model.

        optimizer.zero_grad() only clears sharded_param.grad, but FSDP2 internally
        maintains unsharded_accumulated_grad on each FSDPParam object. If a val phase
        interrupts the training loop before post_backward() has a chance to consume and
        reset this field, the stale unsharded grad will be accumulated on top of the
        next backward pass, causing grad_norm to explode.

        This method clears all three grad locations:
          1. sharded_param.grad        - what the optimizer sees
          2. unsharded_accumulated_grad - FSDP2 internal accumulation buffer
          3. _unsharded_param.grad     - transient grad on the all-gathered parameter
        """
        from torch.distributed._composable.fsdp import FSDPModule

        dirty_count = 0
        for module in self.actor_module_fsdp.modules():
            if isinstance(module, FSDPModule):
                for pg in module._get_fsdp_state()._fsdp_param_groups:
                    for fp in pg.fsdp_params:
                        if fp.unsharded_accumulated_grad is not None:
                            psrl_logger.debug(
                                f"[clear_fsdp2_grads] {fp._param_fqn}: "
                                f"unsharded_accumulated_grad norm = {fp.unsharded_accumulated_grad.norm():.4f}"
                            )
                            fp.unsharded_accumulated_grad = None
                            dirty_count += 1
                        if fp._unsharded_param.grad is not None:
                            psrl_logger.debug(
                                f"[clear_fsdp2_grads] {fp._param_fqn}: "
                                f"_unsharded_param.grad norm = {fp._unsharded_param.grad.norm():.4f}"
                            )
                            fp._unsharded_param.grad = None
                            dirty_count += 1

        # Also clear sharded_param.grad (what optimizer.zero_grad() would clear)
        for p in self.actor_module_fsdp.parameters():
            if p.grad is not None:
                p.grad = None
                dirty_count += 1

        if dirty_count > 0:
            psrl_logger.warning(
                f"[clear_fsdp2_grads] Cleared {dirty_count} stale grad tensor(s) "
                f"after wake_up on rank {self.rank}. "
                f"This likely indicates val interrupted a backward pass."
            )
        else:
            psrl_logger.debug(f"[clear_fsdp2_grads] No stale grads found on rank {self.rank}.")

    def nixl_sleep(self, mode: str = "full"):
        """Deregister the model weights for NIXL and free up GPU memory."""
        self.sleep_fsdp_model()
        if mode == "meta":
            return
        self.nixl_storage_client.deregister_local_tensors()

    def sleep_fsdp_model(self):
        """
        Release GPU memory for model weights without CPU offloading.
        The model metadata (shape, dtype, etc.) is preserved for later wake_up.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} sleep")

        # NOTE(lhy): aggressive_empty_cache is used to ensure no torch reserved memory exists
        # so torch won't trigger cudaFree from the mempool side
        # otherwise it will cause double cuMemRelease (first pause, then free) in tms
        aggressive_empty_cache(force_sync=True)
        # Release GPU memory for FSDP model parameters
        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.pause()
        else:
            self._sleep_fsdp_model(self.actor_module_fsdp)
        # aggressive_empty_cache(force_sync=True)

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} sleep")

    @deprecated(
        "This method is deprecated, it's reserved as an example "
        "for FSDP manual memory management. Use torch_memory_saver instead."
    )
    def _sleep_fsdp_model(self, model):
        """
        Release GPU memory for FSDP model without CPU offloading.
        For FSDP2 models (with DTensor), resize the storage of local shards to 0.

        Args:
            model: FSDP-wrapped model
        """
        for _, param in model.named_parameters():
            if isinstance(param, DTensor):
                # Get the local shard tensor from DTensor
                local_tensor = param._local_tensor
            else:
                local_tensor = param

            if local_tensor.untyped_storage().size() > 0:
                # Store the original storage size as an attribute
                param._sleep_storage_size = local_tensor.untyped_storage().size()
                local_tensor.untyped_storage().resize_(0)

        # Also handle buffers (e.g., LayerNorm running stats)
        for _, buffer in model.named_buffers():
            if buffer is not None and buffer.untyped_storage().size() > 0:
                buffer._sleep_storage_size = buffer.untyped_storage().size()
                buffer.untyped_storage().resize_(0)

    def nixl_wake_up(self):
        """Wake up and re-register NIXL after sleep.

        This method restores GPU memory allocation and performs NIXL re-registration
        to handle memory changes after sleep/wake_up cycle.
        """
        self.wake_up_fsdp_model()
        # Reset nixl agent and reregister to handle physical memory changes
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)

    def wake_up_fsdp_model(self):
        """
        Restore GPU memory allocation for model weights without restoring data.
        The actual data will be transferred via NIXL.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} wake_up")

        # aggressive_empty_cache(force_sync=True)
        # Restore GPU memory allocation for FSDP model
        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.resume()
        else:
            self._wake_up_fsdp_model(self.actor_module_fsdp)
        # aggressive_empty_cache(force_sync=True)

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} wake_up")

    def _restore_non_persistent_buffers_from_ps(self) -> None:
        """
        Restore non-persistent named buffers (e.g. inv_freq) from PS after pull.
        """
        ps_buffers = self._get_non_persistent_buffers_from_ps()
        if not ps_buffers:
            return
        device = get_device_id()
        model = self.actor_module_fsdp
        # Apply each buffer by navigating the module tree with its dotted name.
        for full_name, cpu_tensor in ps_buffers.items():
            *module_path_parts, buf_attr = full_name.split(".")
            module = model
            for part in module_path_parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and hasattr(module, buf_attr):
                existing = getattr(module, buf_attr)
                if isinstance(existing, torch.Tensor):
                    existing.data.copy_(cpu_tensor.to(device=device, dtype=existing.dtype))
        psrl_logger.info(
            f"[_restore_non_persistent_buffers_from_ps] Restored {len(ps_buffers)} "
            f"non-persistent buffers to FSDP model."
        )

    @deprecated(
        "This method is deprecated, it's reserved as an example "
        "for FSDP manual memory management. Use torch_memory_saver instead."
    )
    def _wake_up_fsdp_model(self, model):
        """
        Restore GPU memory allocation for FSDP model without restoring data.
        For FSDP2 models (with DTensor), resize the storage of local shards back to original size.

        Args:
            model: FSDP-wrapped model
        """
        for _, param in model.named_parameters():
            if isinstance(param, DTensor):
                # Get the local shard tensor from DTensor
                local_tensor = param._local_tensor
            else:
                local_tensor = param

            if hasattr(param, "_sleep_storage_size") and local_tensor.untyped_storage().size() == 0:
                local_tensor.untyped_storage().resize_(param._sleep_storage_size)

        # Also restore buffers
        for _, buffer in model.named_buffers():
            if buffer is not None and hasattr(buffer, "_sleep_storage_size") and buffer.untyped_storage().size() == 0:
                buffer.untyped_storage().resize_(buffer._sleep_storage_size)

    def ray_push_model(self) -> None:
        """
        Push the model weights to the PS via ray.
        In 'cpu' mode, push the full state dict.
        The PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, push a ray object_ref.
        Only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote(debug_info="fsdp_train_worker"))
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        # assert fsdp_version(self.actor_module_fsdp) == 1, "FSDP version 2 is not supported yet."
        psrl_logger.info("Gathering the full state dict on the CPU of the representative rank.")
        full_state_dict = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=True, rank0_only=True)
        if self.is_train_representative_rank:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on the representative worker."
            psrl_logger.info("Push the model via CPU on the representative rank (async).")
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, push the full state dict (PS worker will block on transfer)
                # But the training side does not need to wait for the push to complete,
                # as it can be overlapped with the next-iteration training
                ps_manager_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, push a ray object_ref (PS worker is non-blocking)
                # But the training side needs to wait for the push to complete, as `ray.put` is blocking
                object_ref = ray.put(full_state_dict)  # This blocks until the state dict is in the object store
                ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(
                    next_ps_model_version, [object_ref]
                )  # Tricky part: manually wrap the object_ref in a list to avoid ray dereferencing the full state dict
            else:
                raise NotImplementedError(
                    f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet."
                )
        else:
            # assert len(full_state_dict) == 0, "The model state dict should be empty on non-representative workers."
            # FSDP may combined with DDP now (HSDP), so the state dict may not be empty on non-representative workers.
            pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def init_model(self, init_mode: str = "full"):
        """Initialize the FSDP model.

        If init_mode is 'empty', only initialize the model without loading weights.

        Args:
            init_mode (str): 'full' to load weights, 'empty' to only initialize the model with empty weights.
        """
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            skip_load_weight = init_mode == "empty"
            ActorRolloutRefWorker.init_model(self, skip_load_weight)

    def _build_rollout(self, trust_remote_code: bool = False):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def build_rollout(self, trust_remote_code: bool = False):
        ActorRolloutRefWorker._build_rollout(self, trust_remote_code=trust_remote_code)

    # The log_prob in training side may need to be recomputed
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def compute_log_prob(self, data: DataProto):
        # NOTE(lhy): compared with verl, we replace `old_log_probs` with `recomputed_log_probs` in the output.
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        with log_dual_events("Recompute log_prob", psrl_logger, event_type=EventType.OTHER):
            assert self._is_actor
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
                
            data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
            data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
            data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz

            is_lora = data.meta_info.pop("is_lora", False)
            adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
            data = data.to(get_device_id())
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                data = self.ulysses_sharding_manager.preprocess_data(data)
                with adapter_ctx:
                    output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
                output = DataProto.from_dict(tensors={"recomputed_log_probs": output, "entropys": entropys})
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
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def update_actor(self, data: DataProto):
        with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
            output = ActorRolloutRefWorker.update_actor(self, data)
        torch.cuda.synchronize()
        context_manager = exclusive_push_model_context(self.train_interface.ps_manager_handle) if self.is_train_representative_rank else nullcontext()
        with context_manager:
            with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
                PSRL_BaseTrainWorker.push_model(self)
        return output

    def _debug_log_train_model_info(self, label: str, max_elements: int = 10):
        """Debug log train model tensor statistics."""
        for tensor_type, named_tensors in (
            ("param", self.actor_module_fsdp.named_parameters()),
            ("buffer", self.actor_module_fsdp.named_buffers()),
        ):
            for name, tensor in named_tensors:
                log_tensor(
                    tensor=tensor,
                    psrl_logger=psrl_logger,
                    log_prefix=label,
                    name=f"{tensor_type}={name}",
                    max_elements=max_elements,
                )

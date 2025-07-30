import inspect
import logging
import os
import asyncio

from torch import nn
from torch.multiprocessing.reductions import reduce_tensor
from omegaconf import DictConfig

from vllm.v1.engine.async_llm import AsyncLLM

from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.debug.performance import simple_timer
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu, per_tensor_generator
from verl.utils.torch_functional import check_device_is_available
from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

class PSRL_MegatronvLLMShardingManager(MegatronVLLMShardingManager):
    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=psrl_logger)
    def __enter__(self):
        is_sleeping = self.inference_engine.llm_engine.is_sleeping()
        if not is_sleeping:
            psrl_logger.info("Rollout instance is not sleeping, skip sharding manager.")
            return
        psrl_logger.info("Rollout instance is sleeping, start sharding manager.")
        super().__enter__()

class PSRL_MegatronASyncvLLMShardingManager(MegatronVLLMShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: AsyncLLM,
        model_config: DictConfig,
        transformer_config,
        rollout_config: DictConfig,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        device_mesh,
        offload_param: bool = True,
        bridge=None,
    ):
        super().__init__(
            actor_module,
            None,
            model_config,
            rollout_config,
            transformer_config,
            layer_name_mapping,
            weight_converter,
            device_mesh,
            offload_param,
            bridge,
        )
        self.inference_engine = inference_engine

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=psrl_logger)
    async def __aenter__(self):
        is_sleeping = await self.inference_engine.is_sleeping()
        if not is_sleeping:
            psrl_logger.info("Rollout instance is not sleeping, skip sharding manager.")
            return
        psrl_logger.info("Rollout instance is sleeping, start sharding manager.")

        self.timing = {}
        with simple_timer("reshard", self.timing):
            get_torch_device().empty_cache()

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=psrl_logger)
            if self.offload_param:
                load_megatron_model_to_gpu(self.actor_module)

            if self.rollout_config.free_cache_engine:
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    await self.inference_engine.wake_up(tags=["weights"])
                else:
                    await self.inference_engine.wake_up()
            if self.bridge is not None:
                per_tensor_param = self.bridge.export_weights(self.actor_module)
            else:
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.inference_engine.collective_rpc(
                "patch_vllm_moe_model_weight_loader",
                args=tuple(),
            ))
            loaded_params = loop.run_until_complete(self.update_params_async(per_tensor_param))
            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            psrl_logger.info(info)

            if self.offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            get_torch_device().empty_cache()

            if (
                self.rollout_config.free_cache_engine
                and "tags" in inspect.signature(self.inference_engine.wake_up).parameters
            ):
                await self.inference_engine.wake_up(tags=["kv_cache"])

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=psrl_logger)
    async def __aexit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=psrl_logger)
    async def update_params_async(self, updated_params):
        # NOTE: It might be inefficient to load weights one by one,
        # but it is necessary because generator cannot be pickled and passed to collective_rpc.
        # If we pass a list instead, the memory usage will be higher.
        loaded_params = []
        for name, param in updated_params:
            per_loaded_params = await self.inference_engine.collective_rpc(
                "load_weights",
                args=((name, reduce_tensor(param.detach())), False)
            )
            loaded_params.extend(per_loaded_params)
        await self.inference_engine.collective_rpc(
            "cuda_synchronize",
            args=tuple()
        )
        return loaded_params


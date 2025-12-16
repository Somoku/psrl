from collections import OrderedDict

import torch
from mbridge import AutoBridge
from mbridge.core.parallel_states import ParallelStates
from mbridge.core.util import (
    unwrap_model,
)
from torch.nn import Parameter

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    slice_gate_up_proj,
    slice_qkv_proj_megatron,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding


class MegatronConverter(BaseConverter):
    """Convert Megatron model to a unified format (i.e., HuggingFace) and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping, mpu: ParallelStates | None = None):
        self.parameter_mapping = parameter_mapping

        self.parameter_mapping.disable_tie_word_embeddings()
        self.bridge = AutoBridge.from_config(self.parameter_mapping.config)  # mbridge will maintain its own mpu
        if mpu is not None:
            assert self.bridge.mpu == mpu, (
                "Megatron parallel states must be the same, "
                f"but got external {mpu} and mbridge internal {self.bridge.mpu}"
            )
            self.mpu = mpu
        else:
            self.mpu = self.bridge.mpu
        self.model_info = parameter_mapping.get_model_info()

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert Megatron model to unified state dict and generate sharding info.
        Args:
            model: The Megatron model instance
        Returns:
            (converted_state_dict, sharding_dict)
        """
        converted_state_dict = {}
        sharding_dict = {}

        models = [unwrap_model(sub_model) for sub_model in model]
        local_to_global_maps = [
            self.bridge._weight_name_mapping_mcore_local_to_global(model, consider_ep=True) for model in models
        ]
        # print(f"Unwrapped models: {models}")

        def get_model_chunk_generator():
            for vpp_rank, model in enumerate(models):
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield vpp_rank, name, param
                # NOTE(mbridge): there is a bug in megatron GPTModel
                # decoder.layers[n].mlp.router.expert_bias" in GPTModel
                # is not registered in named_parameter, but in state_dict().
                # for now we patch it by adding those keys to extra_keys.
                extra_keys = [
                    x
                    for x in model.state_dict()
                    if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
                ]
                for name in extra_keys:
                    yield vpp_rank, name, model.state_dict()[name]

        for vpp_rank, name, param in get_model_chunk_generator():
            # print(f"Converting parameter: {name}")
            # refactor the name to global name
            local_to_global_map = local_to_global_maps[vpp_rank]
            name = local_to_global_map[name]
            new_params = self.convert_parameter(name, param)
            sharding = self.get_sharding_for_param(name, param)
            for new_param_name, new_param in new_params.items():
                converted_state_dict[new_param_name] = new_param
                sharding_dict[new_param_name] = sharding

        # NOTE(lhy): a workaround for lm_head
        # if PP is not used and the word embedding is shared
        # we manually set the lm_head
        if self.mpu.pp_size == 1 and self.parameter_mapping.original_tie_word_embeddings:
            for original_name, new_name in self.bridge._DIRECT_MAPPING.items():
                if "embed_tokens.weight" in new_name and new_name in converted_state_dict:
                    converted_state_dict["lm_head.weight"] = converted_state_dict[new_name]
                    sharding_dict["lm_head.weight"] = sharding_dict[new_name]
                    break

        return converted_state_dict, sharding_dict

    def convert_parameter(self, full_name: str, param: Parameter) -> dict:
        """
        Convert the parameter, may need to split inplace
        if it matches a split mapping type (e.g., qkv_proj, gate_up_proj).
        """
        full_hf_names = self.bridge._weight_name_mapping_mcore_to_hf(full_name)
        # print(f"convert parameter {full_name} to {full_hf_names}")
        if len(full_hf_names) == 3:
            assert "linear_qkv" in full_name, "Only linear_qkv should have 3 corresponding hf names after split"
            try:
                sliced_params = slice_qkv_proj_megatron(
                    fused_param=param,
                    num_heads=self.model_info["num_heads"],
                    num_kv_heads=self.model_info["num_kv_heads"],
                    head_size=self.model_info["head_size"],
                    tp_size=self.mpu.tp_size,
                )
            except Exception as e:
                raise ValueError(f"Failed to slice qkv parameter {full_name} into {full_hf_names}: {e}") from e
            out = {}
            for shard_id, full_hf_name in enumerate(full_hf_names):
                assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {full_name}"
                new_param = sliced_params[shard_id]
                new_param_name = full_hf_name
                out[new_param_name] = new_param
            return out
        elif len(full_hf_names) == 2:
            assert "linear_fc1" in full_name, "Only linear_fc1 should have 2 corresponding hf names after split"
            try:
                if "shared_experts" in full_name:
                    # NOTE(zym): shared_experts use tp_size, not etp_size
                    sliced_params = slice_gate_up_proj(
                        fused_param=param,
                        output_sizes=[
                            self.model_info["shared_expert_intermediate_size"],
                            self.model_info["shared_expert_intermediate_size"],
                        ],
                        tp_size=self.mpu.tp_size,
                    )
                elif "experts" in full_name:
                    sliced_params = slice_gate_up_proj(
                        fused_param=param,
                        output_sizes=[
                            self.model_info["moe_intermediate_size"],
                            self.model_info["moe_intermediate_size"],
                        ],
                        tp_size=self.mpu.etp_size,
                    )
                else:
                    sliced_params = slice_gate_up_proj(
                        fused_param=param,
                        output_sizes=[
                            self.model_info["intermediate_size"],
                            self.model_info["intermediate_size"],
                        ],
                        tp_size=self.mpu.tp_size,
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to slice gate up proj parameter {full_name} into {full_hf_names}: {e}"
                ) from e
            out = {}
            for shard_id, full_hf_name in enumerate(full_hf_names):
                assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {full_name}"
                new_param = sliced_params[shard_id]
                new_param_name = full_hf_name
                out[new_param_name] = new_param
            return out
        else:
            assert len(full_hf_names) == 1, (
                f"Only one corresponding hf name should be transformed for {full_name}, but got {full_hf_names}"
            )
            return {full_hf_names[0]: param}

    def get_sharding_for_param(self, full_name: str, param: Parameter) -> NIXLSharding:
        """
        Generate sharding info for a parameter given its module and tp_rank.
        Returns a NIXLSharding object.
        """
        is_etp_param = "mlp.experts" in full_name and self.mpu.etp_size > 1
        # NOTE(zym): When enabling both ep and tp, ep param also has attribute "tensor_model_parallel" which is True,
        # so we need to exclude ep param when determining is_tp_param
        is_tp_param = getattr(param, "tensor_model_parallel", False) and "mlp.experts" not in full_name
        # NOTE(zym): etp param also has attribute "tensor_model_parallel" which is True,
        # so we need to first determine is_etp_param
        if is_etp_param:
            shard_size = self.mpu.etp_size
            shard_indices = [(self.mpu.etp_rank,)]
            shard_dim = 0 if "fc1" in full_name else 1
        elif is_tp_param:
            shard_size = self.mpu.tp_size
            assert hasattr(param, "partition_dim"), (
                f"Tensor parallel partition dim must be set, but got {param.partition_dim}"
            )
            shard_indices = [(self.mpu.tp_rank,)]
            shard_dim = param.partition_dim
        else:
            shard_size = 1
            shard_indices = [(0,)]
            shard_dim = 0
        kwargs = {
            "shard_mesh": OrderedDict([(shard_dim, shard_size)]),
            "shard_indices": shard_indices,
        }
        return NIXLSharding(**kwargs)


def convert_megatron_inplace(parameter_mapping: ParameterMapping, model, mpu: ParallelStates | None = None):
    """
    Convenience function to convert Megatron model to unified state dict and sharding info.
    Args:
        parameter_mapping: Parameter mapping instance for the specific model
        model: The Megatron model instance
        mpu: Megatron parallel states, if None, use mbridge internal mpu
    Returns:
        (converted_state_dict, sharding_dict)
    """
    converter = MegatronConverter(parameter_mapping, mpu=mpu)
    return converter.convert_state_and_sharding_dict(model)

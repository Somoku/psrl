from collections import OrderedDict

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    GDNLinearMappingSeparate,
    MegatronParamMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.utils import unwrap_model
from torch.nn import Parameter

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    slice_qwen3_5_in_proj,
    slice_qwen3_5_in_proj_qkv,
    slice_gate_up_proj,
    slice_qkv_proj_megatron,
    slice_attn_conv1d,
    reshape_visual_block_qkv,
)
from psrl.utils.converter.utils.parallel_states import ParallelStates
from psrl.utils.nixl.nixl_spec import NIXLSharding


class MegatronConverter(BaseConverter):
    """Convert Megatron model to a unified format (i.e., HuggingFace) and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping, mpu: ParallelStates | None = None):
        super().__init__(parameter_mapping)
        self.parameter_mapping = parameter_mapping

        self.parameter_mapping.disable_tie_word_embeddings()
        self.bridge = AutoBridge.from_hf_config(self.parameter_mapping.config)
        current_mpu = ParallelStates.get_parallel_state()
        if mpu is not None:
            assert current_mpu == mpu, (
                "Megatron parallel states must be the same, "
                f"but got external {mpu} and current Megatron state {current_mpu}"
            )
            self.mpu = mpu
        else:
            self.mpu = current_mpu

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

        model_list = model if isinstance(model, (list, tuple)) else [model]
        models = unwrap_model(list(model_list))
        conversion_tasks = self.bridge._model_bridge.build_conversion_tasks(self.parameter_mapping.config, models)
        task_by_local_name = {
            (task.vp_stage, task.param_name): task
            for task in conversion_tasks
            if task is not None and task.vp_stage is not None and task.megatron_module is not None
        }
        embedding_hf_name = next(
            (
                task.mapping.hf_param
                for task in conversion_tasks
                if task is not None
                and task.global_param_name.endswith("embedding.word_embeddings.weight")
                and isinstance(task.mapping, AutoMapping)
                and isinstance(task.mapping.hf_param, str)
            ),
            None,
        )

        def get_model_chunk_generator():
            for vpp_rank, model in enumerate(models):
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield vpp_rank, name, param
                # NOTE(megatron-bridge): there is a bug in megatron GPTModel
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
            task = task_by_local_name.get((vpp_rank, name))
            if task is None:
                raise ValueError(f"Megatron-Bridge did not build a conversion task for parameter {name!r}.")
            global_name = task.global_param_name
            new_params = self.convert_parameter(global_name, param, task.mapping)
            sharding = self.get_sharding_for_param(global_name, param)
            for new_param_name, new_param in new_params.items():
                converted_state_dict[new_param_name] = new_param
                sharding_dict[new_param_name] = sharding

        # NOTE(lhy): a workaround for lm_head
        # if PP is not used and the word embedding is shared
        # we manually set the lm_head
        if self.mpu.pp_rank == self.mpu.pp_size - 1 and self.parameter_mapping.original_tie_word_embeddings:
            if embedding_hf_name is not None and embedding_hf_name in converted_state_dict:
                converted_state_dict["lm_head.weight"] = converted_state_dict[embedding_hf_name]
                sharding_dict["lm_head.weight"] = sharding_dict[embedding_hf_name]

        return converted_state_dict, sharding_dict

    def convert_parameter(self, full_name: str, param: Parameter, mapping: MegatronParamMapping) -> dict:
        """
        Convert the parameter, may need to split inplace
        if it matches a split mapping type (e.g., qkv_proj, gate_up_proj).

        Args:
            full_name: The full parameter name in Megatron model
            param: The parameter tensor
            mapping: Megatron-Bridge parameter mapping for this parameter
        Returns:
            A dict of {new_param_name: new_param_tensor}
        """
        if isinstance(mapping, QKVMapping):
            hf_param = mapping.hf_param
            return self._convert_qkv_parameter(full_name, param, [hf_param["q"], hf_param["k"], hf_param["v"]])

        if isinstance(mapping, GDNLinearMappingSeparate):
            hf_param = mapping.hf_param
            return self._convert_qwen35_in_proj_parameter(
                full_name, param, [hf_param["qkv"], hf_param["z"], hf_param["b"], hf_param["a"]]
            )

        if isinstance(mapping, GatedMLPMapping):
            hf_param = mapping.hf_param
            return self._convert_gated_mlp_parameter(full_name, param, [hf_param["gate"], hf_param["up"]])

        if isinstance(mapping, AutoMapping):
            return self._convert_auto_mapping_parameter(full_name, param, mapping.hf_param)

        raise NotImplementedError(
            "Unsupported Megatron-Bridge mapping class for PSRL local shard conversion: "
            f"{type(mapping).__name__} {full_name=} {mapping.hf_param=}."
        )

    def _convert_qkv_parameter(self, full_name: str, param: Parameter, full_hf_names: list[str]) -> dict:
        assert "linear_qkv" in full_name, "Only linear_qkv should have 3 corresponding hf names after split"
        try:
            sliced_params = slice_qkv_proj_megatron(
                fused_param=param,
                num_heads=self.model_info["num_heads"],
                num_kv_heads=self.model_info["num_kv_heads"],
                head_size=self.model_info["head_size"],
                attn_output_gate=self.model_info["attn_output_gate"],
                tp_size=self.mpu.tp_size,
            )
        except Exception as e:
            raise ValueError(f"Failed to slice qkv parameter {full_name} into {full_hf_names}: {e}") from e
        out = {}
        for shard_id, full_hf_name in enumerate(full_hf_names):
            assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {full_name}"
            out[full_hf_name] = sliced_params[shard_id]
        return out

    def _convert_qwen35_in_proj_parameter(
        self, full_name: str, param: Parameter, full_hf_names: list[str]
    ) -> dict:
        assert "in_proj" in full_name, "Only in_proj should have 4 corresponding hf names after split"
        key_dim = self.model_info.get("linear_key_dim")
        value_dim = self.model_info.get("linear_value_dim")
        num_v_heads = self.model_info.get("linear_num_value_heads")
        if key_dim is None or value_dim is None or num_v_heads is None:
            raise ValueError(
                "Qwen3.5 linear attention dims are missing in model_info for in_proj split; "
                f"got linear_key_dim={key_dim}, linear_value_dim={value_dim}, linear_num_value_heads={num_v_heads}."
            )
        try:
            sliced_params = slice_qwen3_5_in_proj(
                fused_param=param,
                key_dim=key_dim,
                value_dim=value_dim,
                num_v_heads=num_v_heads,
                tp_size=self.mpu.tp_size,
                output_dim=0,
            )
        except Exception as e:
            raise ValueError(f"Failed to slice in_proj parameter {full_name} into {full_hf_names}: {e}") from e
        out = {}
        for shard_id, full_hf_name in enumerate(full_hf_names):
            assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {full_name}"
            new_param = sliced_params[shard_id]
            if shard_id == 0:
                qkv_names = [full_hf_name + "_q", full_hf_name + "_k", full_hf_name + "_v"]
                qkv_params = slice_qwen3_5_in_proj_qkv(
                    fused_param=new_param,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    tp_size=self.mpu.tp_size,
                )
                out.update(dict(zip(qkv_names, qkv_params)))
                continue
            out[full_hf_name] = new_param
        return out

    def _convert_gated_mlp_parameter(self, full_name: str, param: Parameter, full_hf_names: list[str]) -> dict:
        if "linear_fc1" not in full_name:
            raise NotImplementedError(
                "PSRL's local Megatron shard converter only supports GatedMLPMapping for linear_fc1 parameters, "
                f"but got {full_name=}."
            )
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
            raise ValueError(f"Failed to slice gate up proj parameter {full_name} into {full_hf_names}: {e}") from e
        out = {}
        for shard_id, full_hf_name in enumerate(full_hf_names):
            assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {full_name}"
            out[full_hf_name] = sliced_params[shard_id]
        return out

    def _convert_auto_mapping_parameter(self, full_name: str, param: Parameter, hf_name: str) -> dict:
        if "self_attention.conv1d.weight" in full_name:
            new_param_names = [hf_name + "_q", hf_name + "_k", hf_name + "_v"]
            new_params = slice_attn_conv1d(
                fused_param=param,
                num_k_heads=self.model_info["linear_num_key_heads"],
                num_v_heads=self.model_info["linear_num_value_heads"],
                k_head_size=self.model_info["linear_key_head_dim"],
                v_head_size=self.model_info["linear_value_head_dim"],
                tp_size=self.mpu.tp_size,
            )
            return dict(zip(new_param_names, new_params))
        if "visual.blocks" in hf_name and "qkv" in hf_name:
            param = reshape_visual_block_qkv(param)
            param.partition_dim = 1
        if "mlp.experts.linear_fc1.weight" in full_name:
            name_prefix = hf_name.rsplit(".", 1)[0]
            expert_id = full_name.split("weight")[1]
            new_param_names = [
                f"{name_prefix}.{expert_id}.gate_proj.weight",
                f"{name_prefix}.{expert_id}.up_proj.weight",
            ]
            new_params = slice_gate_up_proj(
                fused_param=param,
                output_sizes=[
                    self.model_info["moe_intermediate_size"],
                    self.model_info["moe_intermediate_size"],
                ],
                tp_size=self.mpu.etp_size,
            )
            return dict(zip(new_param_names, new_params))
        if "mlp.experts.linear_fc2.weight" in full_name:
            name_prefix = hf_name.rsplit(".", 1)[0]
            expert_id = full_name.split("weight")[1]
            new_param_name = f"{name_prefix}.{expert_id}.down_proj.weight"
            return {new_param_name: param}
        return {hf_name: param}

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
            if shard_dim == -1:
                # For dt_bias, A_log, and conv1d.weight, shard_dim should be 0,
                # but their partition_dim is -1
                shard_dim = 0
        else:
            shard_size = 1
            shard_indices = [(0,)]
            shard_dim = 0
        kwargs = {
            "shard_mesh": OrderedDict([(shard_dim, shard_size)]),
            "shard_indices": shard_indices,
        }
        return NIXLSharding(**kwargs)


def convert_megatron_inplace(
    parameter_mapping: ParameterMapping,
    model,
    mpu: ParallelStates | None = None,
):
    """
    Convenience function to convert Megatron model to unified state dict and sharding info.
    Args:
        parameter_mapping: Parameter mapping instance for the specific model
        model: The Megatron model instance
        mpu: Megatron parallel states, if None, use current Megatron parallel state
    Returns:
        (converted_state_dict, sharding_dict)
    """
    converter = MegatronConverter(parameter_mapping, mpu=mpu)
    return converter.convert_state_and_sharding_dict(model)

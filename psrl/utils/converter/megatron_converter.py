from collections import OrderedDict

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ChunkedMapping,
    ConcatenatedQKVMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    GDNLinearMapping,
    GDNLinearMappingSeparate,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.utils import unwrap_model
from torch.nn import Parameter

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    make_slice_parameter,
    reshape_visual_block_qkv,
    slice_attn_conv1d,
    slice_gate_up_proj,
    slice_qkv_proj_megatron,
    slice_qwen3_5_in_proj,
    slice_qwen3_5_in_proj_qkv,
)
from psrl.utils.converter.param_sync import (
    ConcatenatedQKVSync,
    ConversionResult,
    DTypeCastSync,
    ParamSyncPlan,
    ZeroCenteredGammaSync,
)
from psrl.utils.converter.utils.parallel_states import ParallelStates
from psrl.utils.nixl.nixl_spec import NIXLSharding


class MegatronConverter(BaseConverter):
    """Convert Megatron model to a unified format (i.e., HuggingFace) and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping, mpu: ParallelStates | None = None):
        super().__init__(parameter_mapping)
        self.parameter_mapping = parameter_mapping
        self.sync_plan = ParamSyncPlan()

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
        for m in models:
            if hasattr(m, "config") and not hasattr(m.config, "share_embeddings_and_output_weights"):
                m.config.share_embeddings_and_output_weights = getattr(m, "share_embeddings_and_output_weights", False)
        # Detect attention_output_gate from the actual Megatron TransformerConfig.
        # The HF config may not have this attribute, causing model_info to default
        # to False. But if the Megatron model was built with attention_output_gate=True,
        # the QKV weight layout includes the gate and we must account for it.
        if models and hasattr(models[0], "config"):
            tf_config = models[0].config
            actual_attn_output_gate = getattr(tf_config, "attention_output_gate", False)
            if actual_attn_output_gate and not self.model_info.get("attn_output_gate", False):
                self.model_info["attn_output_gate"] = True
                # num_heads must be doubled when attn_output_gate is True
                self.model_info["num_heads"] = self.model_info["num_heads"] * 2

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
                if name.endswith("output_layer.weight"):
                    # Skip output_layer.weight in the fallthrough — it's the lm_head
                    # which is handled by the tied-weight alias workaround below
                    # (exported from the embedding on the PP stage that has it).
                    continue

                if name.startswith("vision_model."):
                    raise ValueError(
                        "Megatron-Bridge did not produce a conversion task for vision parameter "
                        f"{name!r} on vp_stage={vpp_rank}. Decoder-style vision names such as "
                        "'vision_model.decoder.layers.0.self_attention.linear_qkv.weight' should be "
                        "covered by the model bridge mapping registry. If this parameter uses a "
                        "different naming convention, add an alias mapping in Megatron-Bridge instead "
                        "of translating it in PSRL."
                    )

                converted_state_dict[name] = param
                sharding_dict[name] = NIXLSharding(
                    shard_mesh=OrderedDict([(0, 1)]),
                    shard_indices=[(0,)],
                )
                continue
            global_name = task.global_param_name
            new_params = self.convert_parameter(global_name, param, task.mapping)
            sharding = self.get_sharding_for_param(global_name, param)
            for new_param_name, new_param in new_params.items():
                # Each output param must own a SEPARATE sharding object because the server
                # mutates shardings in-place during refactor_based_on_finer_shard_mesh.
                # Without copy, Q/K/V from the same QKV split share one object, and
                # refactoring Q's sharding (different unified mesh due to attn_output_gate)
                # would corrupt K/V's sharding.
                param_sharding = NIXLSharding(
                    shard_mesh=OrderedDict(sharding.shard_mesh),
                    shard_indices=list(sharding.shard_indices),
                )
                new_param, sharding_for_param = self.maybe_reshape_qkv_to_3d(new_param_name, new_param, param_sharding)
                converted_state_dict[new_param_name] = new_param
                sharding_dict[new_param_name] = sharding_for_param

        # NOTE(lhy): a workaround for lm_head with tied word embeddings.
        # When tie_word_embeddings=True, lm_head.weight == embed_tokens.weight.
        # With PP=1: the last (only) stage has both → just alias.
        # With PP>1: the embedding is on the FIRST stage but lm_head is logically
        # on the LAST stage. LinearCrossEntropyModule on the last stage may not
        # expose a separate .weight parameter. So whichever stage HAS the embedding
        # should also export lm_head.weight (they're the same tensor).
        if self.parameter_mapping.original_tie_word_embeddings:
            if embedding_hf_name is not None and embedding_hf_name in converted_state_dict:
                converted_state_dict["lm_head.weight"] = converted_state_dict[embedding_hf_name]
                sharding_dict["lm_head.weight"] = NIXLSharding(
                    shard_mesh=OrderedDict(sharding_dict[embedding_hf_name].shard_mesh),
                    shard_indices=list(sharding_dict[embedding_hf_name].shard_indices),
                )

        # Expose architecturally-constrained parameters in the dtype expected by PS/vLLM.
        # The sync action keeps this detached copy synchronized in both directions.
        fp32_patterns = self.parameter_mapping.get_external_fp32_param_patterns()
        for key, tensor in converted_state_dict.items():
            if tensor.dtype != torch.float32 and any(p in key for p in fp32_patterns):
                self.sync_plan.add(DTypeCastSync(key=key, source_param=tensor))
                converted_state_dict[key] = tensor.float()

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

        if isinstance(mapping, ConcatenatedQKVMapping):
            return self._convert_concatenated_qkv_parameter(full_name, param, mapping.hf_param)

        if isinstance(mapping, GDNLinearMappingSeparate):
            hf_param = mapping.hf_param
            return self._convert_qwen35_in_proj_parameter(
                full_name, param, [hf_param["qkv"], hf_param["z"], hf_param["b"], hf_param["a"]]
            )

        if isinstance(mapping, GatedMLPMapping):
            hf_param = mapping.hf_param
            return self._convert_gated_mlp_parameter(full_name, param, [hf_param["gate"], hf_param["up"]])

        if isinstance(mapping, (FusedGatedExpertMapping, FusedExpertMapping)):
            return self._convert_fused_expert_parameter(full_name, param, mapping.hf_param)

        if isinstance(mapping, GDNLinearMapping):
            hf_param = mapping.hf_param
            return self._convert_gdn_linear_parameter(full_name, param, hf_param)

        if isinstance(mapping, ChunkedMapping):
            # Handles GDNConv1dMapping, MambaConv1dMapping, and any future ChunkedMapping subclasses.
            # ChunkedMapping has string hf_param and needs component-wise splitting.
            return self._convert_chunked_parameter(full_name, param, mapping.hf_param)

        if isinstance(mapping, RMSNorm2ZeroCenteredRMSNormMapping):
            # This norm is stored in Megatron as (γ-1). We keep the raw Megatron tensor in
            # unified_state_dict (zero-centered format) and record the key so that push_model /
            # nixl_pull_model can apply the ±1 correction without any string-matching heuristic.
            assert isinstance(mapping.hf_param, str), (
                f"RMSNorm2ZeroCenteredRMSNormMapping hf_param must be a resolved string, "
                f"got {type(mapping.hf_param)} for {full_name}"
            )
            self.sync_plan.add(ZeroCenteredGammaSync(mapping.hf_param))
            return {mapping.hf_param: param}

        if isinstance(mapping, (AutoMapping, ReplicatedMapping)):
            return self._convert_auto_mapping_parameter(full_name, param, mapping.hf_param)

        # Catch-all for remaining mapping types with string hf_param (DirectMapping,
        # ColumnParallelMapping, RowParallelMapping, etc.) — treat as simple passthrough.
        hf_param = mapping.hf_param
        if isinstance(hf_param, str):
            return self._convert_auto_mapping_parameter(full_name, param, hf_param)

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

    def _convert_concatenated_qkv_parameter(self, full_name: str, param: Parameter, full_hf_name: str) -> dict:
        try:
            if "vision_model" in full_name:
                num_heads = self.model_info["vision_num_heads"]
                num_kv_heads = self.model_info["vision_num_kv_heads"]
                head_size = self.model_info["vision_head_size"]
            else:
                num_heads = self.model_info["num_heads"]
                num_kv_heads = self.model_info["num_kv_heads"]
                head_size = self.model_info["head_size"]
            q, k, v = slice_qkv_proj_megatron(
                fused_param=param,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                attn_output_gate=False,
                tp_size=self.mpu.tp_size,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to convert concatenated qkv parameter {full_name} into {full_hf_name}: {e}"
            ) from e

        qkv = torch.cat((q, k, v), dim=0)
        if "visual.blocks" in full_hf_name and "qkv" in full_hf_name:
            qkv = reshape_visual_block_qkv(qkv, vision_head_size=self.model_info.get("vision_head_size"))
            param.partition_dim = 1
        self.sync_plan.add(
            ConcatenatedQKVSync(
                key=full_hf_name,
                megatron_name=full_name,
                source_param=param,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                tp_size=self.mpu.tp_size,
                vision_head_size=self.model_info.get("vision_head_size"),
            )
        )
        return {full_hf_name: qkv}

    def _convert_qwen35_in_proj_parameter(self, full_name: str, param: Parameter, full_hf_names: list[str]) -> dict:
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

    def _convert_fused_expert_parameter(self, full_name: str, param: Parameter, hf_name: str) -> dict:
        """Handle FusedExpertMapping / FusedGatedExpertMapping.

        These mappings use a PREFIX-style hf_param with only one wildcard (layer index).
        The expert index is NOT embedded in hf_name — it must be extracted from the
        Megatron param name (e.g., ``...linear_fc2.weight64``) and inserted manually.

        Example (DeepSeek / GLM45 fused-expert mode):
            full_name = "decoder.layers.0.mlp.experts.linear_fc2.weight64"
            hf_name   = "model.layers.0.mlp.experts.down_proj"  (prefix, no expert index)
            result    → "model.layers.0.mlp.experts.64.down_proj.weight"
        """
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

    def _convert_auto_mapping_parameter(self, full_name: str, param: Parameter, hf_name: str) -> dict:
        """Handle AutoMapping where hf_name is already fully resolved by the bridge.

        For standard AutoMapping, the bridge's ``resolve()`` replaces ALL wildcards
        in hf_param (including expert indices), so hf_name is the correct final key.
        No manual expert-index reconstruction is needed.

        Special cases (conv1d, visual qkv) are handled here as they require
        parameter splitting/reshaping beyond simple name passthrough.
        """
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
            param = reshape_visual_block_qkv(param, vision_head_size=self.model_info.get("vision_head_size"))
            param.partition_dim = 1
        return {hf_name: param}

    def _convert_chunked_parameter(self, full_name: str, param: Parameter, hf_name: str) -> dict:
        """Handle ChunkedMapping subclasses (GDNConv1dMapping, MambaConv1dMapping).

        ChunkedMapping has a single string ``hf_param`` representing the base HF weight
        name. The fused tensor contains multiple components (e.g., q/k/v for GDN conv1d,
        or x/B/C for Mamba conv1d) that need to be split.

        For GDN conv1d, vLLM expects the weight split into ``<base>_q``, ``<base>_k``,
        ``<base>_v`` components via ``split_param`` in its weight layout builder.
        """
        if "conv1d" in full_name:
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
        # Fallback for other ChunkedMapping subclasses (e.g. MambaConv1dMapping) —
        # pass through as-is until specific handling is implemented.
        return {hf_name: param}

    def _convert_gdn_linear_parameter(self, full_name: str, param: Parameter, hf_param: dict) -> dict:
        """Handle GDNLinearMapping (fused QKVZ + BA format used by Qwen3-Next).

        GDNLinearMapping has dict hf_param with keys ``qkvz`` and ``ba``.
        The fused Megatron tensor is the concatenation of QKVZ and BA along dim 0.
        We split it based on the model's GDN linear dimensions.

        For vLLM Qwen3-Next, ``in_proj_qkvz`` and ``in_proj_ba`` are the expected keys.
        """
        qkvz_name = hf_param["qkvz"]
        ba_name = hf_param["ba"]

        key_dim = self.model_info.get("linear_key_dim")
        value_dim = self.model_info.get("linear_value_dim")
        num_v_heads = self.model_info.get("linear_num_value_heads")
        if key_dim is None or value_dim is None or num_v_heads is None:
            raise ValueError(
                "GDN linear dims missing in model_info for GDNLinearMapping split; "
                f"got linear_key_dim={key_dim}, linear_value_dim={value_dim}, "
                f"linear_num_value_heads={num_v_heads}."
            )
        # QKVZ dim = key_dim + key_dim + value_dim + hidden_size (z = hidden_size)
        # BA dim = num_v_heads * value_head_dim + num_v_heads * value_head_dim
        # But simpler: split by ratio — the bridge concatenates QKVZ first, then BA
        # The local shard is already TP-sliced, so we compute based on local sizes.
        hidden_size = self.model_info.get("hidden_size", 0)
        qkvz_size = (key_dim + key_dim + value_dim + hidden_size) // self.mpu.tp_size
        total_size = param.shape[0]
        ba_size = total_size - qkvz_size

        qkvz_param = param.data.narrow(0, 0, qkvz_size)
        ba_param = param.data.narrow(0, qkvz_size, ba_size)

        return {
            qkvz_name: make_slice_parameter(qkvz_param, param),
            ba_name: make_slice_parameter(ba_param, param),
        }

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
) -> ConversionResult:
    """
    Convenience function to convert Megatron model to unified state dict and sharding info.
    Args:
        parameter_mapping: Parameter mapping instance for the specific model
        model: The Megatron model instance
        mpu: Megatron parallel states, if None, use current Megatron parallel state
    Returns:
        ConversionResult containing the unified state dict, local sharding dict, and a
        synchronization plan for canonical tensors that are not plain aliases of train parameters.
    """
    converter = MegatronConverter(parameter_mapping, mpu=mpu)
    state_dict, sharding_dict = converter.convert_state_and_sharding_dict(model)
    return ConversionResult(state_dict=state_dict, sharding_dict=sharding_dict, sync_plan=converter.sync_plan)

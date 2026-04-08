"""Functional tests: each vLLM model returns a spec consistent with its load_weights()."""

from unittest.mock import MagicMock

import pytest
from vllm.model_executor.models.interfaces import SupportsWeightLayoutSpec, WeightLayoutSpec

pytestmark = pytest.mark.cpu_test


def make_mock_config(**kwargs):
    config = MagicMock()
    defaults = {
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_experts": 8,
        "moe_intermediate_size": 32,
        "n_routed_experts": 8,
        "num_hidden_layers": 4,
        "num_local_experts": 4,
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        setattr(config, k, v)
    # Simulate absence of optional head_dim attribute
    config.configure_mock(**{"head_dim": MagicMock(side_effect=AttributeError)})
    # Make getattr(config, "head_dim", fallback) work by deleting the attr
    try:
        del config.head_dim
    except AttributeError:
        pass
    return config


class TestPattern1StandardModels:
    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            ("vllm.model_executor.models.qwen2", "Qwen2ForCausalLM"),
            ("vllm.model_executor.models.llama", "LlamaForCausalLM"),
            ("vllm.model_executor.models.mistral", "MistralForCausalLM"),
            ("vllm.model_executor.models.phi", "PhiForCausalLM"),
            ("vllm.model_executor.models.gemma", "GemmaForCausalLM"),
        ],
    )
    def test_implements_protocol(self, module_path, class_name):
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        assert hasattr(cls, "get_weight_layout_spec"), (
            f"{class_name} must implement SupportsWeightLayoutSpec (has get_weight_layout_spec)"
        )

    def test_qwen2_spec_consistency(self):
        from vllm.model_executor.models.qwen2 import Qwen2Model

        instance = Qwen2Model.__new__(Qwen2Model)
        instance.config = make_mock_config()
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        packed_names = {e[0] for e in spec.stacked_params}
        assert "qkv_proj" in packed_names
        assert "gate_up_proj" in packed_names
        assert "num_heads" in spec.packing_metadata
        assert "num_kv_heads" in spec.packing_metadata
        assert "head_size" in spec.packing_metadata
        assert "intermediate_size" in spec.packing_metadata

    def test_qwen2_for_causal_lm_delegates(self):
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model

        outer = Qwen2ForCausalLM.__new__(Qwen2ForCausalLM)
        inner = Qwen2Model.__new__(Qwen2Model)
        inner.config = make_mock_config()
        # Use object.__setattr__ to bypass nn.Module's __setattr__ check
        # (which requires __init__ to have been called).
        object.__setattr__(outer, "model", inner)
        spec = outer.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        assert isinstance(outer, SupportsWeightLayoutSpec)

    @pytest.mark.parametrize(
        "model_module,model_cls,inner_attr",
        [
            ("vllm.model_executor.models.llama", "LlamaForCausalLM", "model"),
            ("vllm.model_executor.models.mistral", "MistralForCausalLM", "model"),
            ("vllm.model_executor.models.phi", "PhiForCausalLM", "model"),
            ("vllm.model_executor.models.gemma", "GemmaForCausalLM", "model"),
        ],
    )
    def test_pattern1_has_method_and_returns_spec(self, model_module, model_cls, inner_attr):
        import importlib

        mod = importlib.import_module(model_module)
        cls = getattr(mod, model_cls)
        assert hasattr(cls, "get_weight_layout_spec"), f"{model_cls} missing get_weight_layout_spec"
        instance = cls.__new__(cls)
        instance.config = make_mock_config()
        inner_name = model_cls.replace("ForCausalLM", "Model")
        if hasattr(mod, inner_name):
            inner = getattr(mod, inner_name).__new__(getattr(mod, inner_name))
            inner.config = instance.config
            object.__setattr__(instance, inner_attr, inner)
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        assert len(spec.stacked_params) > 0


class TestPattern1MoEModels:
    @pytest.mark.parametrize(
        "model_module,model_cls,inner_attr",
        [
            ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeForCausalLM", "model"),
            ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeForCausalLM", "model"),
            ("vllm.model_executor.models.jamba", "JambaForCausalLM", "model"),
            ("vllm.model_executor.models.olmoe", "OlmoeForCausalLM", "model"),
        ],
    )
    def test_moe_spec_has_expert_params(self, model_module, model_cls, inner_attr):
        import importlib

        mod = importlib.import_module(model_module)
        cls = getattr(mod, model_cls)
        assert hasattr(cls, "get_weight_layout_spec"), f"{model_cls} missing get_weight_layout_spec"
        instance = cls.__new__(cls)
        instance.config = make_mock_config()
        inner_name = model_cls.replace("ForCausalLM", "Model")
        if hasattr(mod, inner_name):
            inner = getattr(mod, inner_name).__new__(getattr(mod, inner_name))
            inner.config = instance.config
            inner.get_expert_mapping = MagicMock(return_value=[("experts.w13_", "experts.0.gate_proj.", 0, "w1")])
            object.__setattr__(instance, inner_attr, inner)
        instance.get_expert_mapping = MagicMock(return_value=[("experts.w13_", "experts.0.gate_proj.", 0, "w1")])
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        assert spec.expert_params is not None, f"{model_cls} spec.expert_params must not be None"
        assert "num_experts" in spec.packing_metadata

    def test_deepseek_v2_mha_path(self):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM

        assert hasattr(DeepseekV2ForCausalLM, "get_weight_layout_spec")
        instance = DeepseekV2ForCausalLM.__new__(DeepseekV2ForCausalLM)
        instance.config = make_mock_config(n_routed_experts=8, moe_intermediate_size=32, num_key_value_heads=4)
        instance.use_mha = True
        instance.get_expert_mapping = MagicMock(return_value=[("experts.w13_", "experts.0.gate_proj.", 0, "w1")])
        spec = instance.get_weight_layout_spec()
        packed = {e[0] for e in spec.stacked_params}
        assert "qkv_proj" in packed, "MHA path must include qkv_proj"
        assert "fused_qkv_a_proj" not in packed

    def test_deepseek_v2_mla_path(self):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM

        instance = DeepseekV2ForCausalLM.__new__(DeepseekV2ForCausalLM)
        instance.config = make_mock_config(n_routed_experts=8, moe_intermediate_size=32, num_key_value_heads=4)
        instance.use_mha = False
        instance.get_expert_mapping = MagicMock(return_value=[("experts.w13_", "experts.0.gate_proj.", 0, "w1")])
        spec = instance.get_weight_layout_spec()
        packed = {e[0] for e in spec.stacked_params}
        assert "fused_qkv_a_proj" in packed, "MLA path must include fused_qkv_a_proj"
        assert "qkv_proj" not in packed


class TestLoadWeightsConsistency:
    @pytest.mark.parametrize(
        "model_module,inner_class_name",
        [
            ("vllm.model_executor.models.qwen2", "Qwen2Model"),
            ("vllm.model_executor.models.llama", "LlamaModel"),
            ("vllm.model_executor.models.gemma", "GemmaModel"),
            ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeModel"),
            ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeModel"),
            ("vllm.model_executor.models.jamba", "JambaModel"),
            ("vllm.model_executor.models.gemma2", "Gemma2Model"),
            ("vllm.model_executor.models.gemma3", "Gemma3Model"),
            ("vllm.model_executor.models.mixtral", "MixtralModel"),
        ],
    )
    def test_load_weights_calls_spec_not_local_var(self, model_module, inner_class_name):
        import importlib
        import inspect

        mod = importlib.import_module(model_module)
        cls = getattr(mod, inner_class_name)
        src = inspect.getsource(cls.load_weights)
        assert "get_weight_layout_spec()" in src, (
            f"{inner_class_name}.load_weights() must call self.get_weight_layout_spec()"
        )
        assert "stacked_params_mapping = [" not in src, (
            f"{inner_class_name}.load_weights() must not re-declare stacked_params_mapping as a list literal"
        )


class TestPattern2Models:
    @pytest.mark.parametrize(
        "model_module,model_cls",
        [
            ("vllm.model_executor.models.mamba", "MambaForCausalLM"),
            ("vllm.model_executor.models.kimi_k25", "KimiK25ForConditionalGeneration"),
            ("vllm.model_executor.models.bailing_moe_linear", "BailingMoeV25ForCausalLM"),
        ],
    )
    def test_pattern2_returns_empty_spec(self, model_module, model_cls):
        import importlib

        mod = importlib.import_module(model_module)
        cls = getattr(mod, model_cls)
        assert hasattr(cls, "get_weight_layout_spec"), f"{model_cls} missing get_weight_layout_spec"
        instance = cls.__new__(cls)
        instance.config = make_mock_config()
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        assert spec.stacked_params == []
        assert spec.expert_params is None


class TestPattern3Arctic:
    def test_arctic_has_extra_stacked_params(self):
        from vllm.model_executor.models.arctic import ArcticForCausalLM

        assert hasattr(ArcticForCausalLM, "get_weight_layout_spec")
        instance = ArcticForCausalLM.__new__(ArcticForCausalLM)
        instance.config = make_mock_config(num_hidden_layers=4, num_local_experts=4)
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        assert spec.extra_stacked_params is not None
        assert len(spec.extra_stacked_params) > 0
        packed_names = {e[0] for e in spec.stacked_params}
        assert "qkv_proj" in packed_names

    def test_arctic_load_weights_uses_spec(self):
        import inspect

        from vllm.model_executor.models.arctic import ArcticForCausalLM

        src = inspect.getsource(ArcticForCausalLM.load_weights)
        assert "get_weight_layout_spec()" in src, "ArcticForCausalLM.load_weights() must call get_weight_layout_spec()"
        assert "stacked_params_mapping = [" not in src, (
            "ArcticForCausalLM.load_weights() must not re-declare stacked_params_mapping as list literal"
        )


class TestNewHighPriorityModels:
    """Tests for newly added models: qwen3, qwen3_5, gemma2, gemma3, mixtral, internlm2."""

    def test_qwen3_for_causal_lm_has_spec(self):
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model

        assert hasattr(Qwen3ForCausalLM, "get_weight_layout_spec")
        outer = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
        # Qwen3Model inherits from Qwen2Model which has get_weight_layout_spec
        inner = Qwen3Model.__new__(Qwen3Model)
        inner.config = make_mock_config()
        object.__setattr__(outer, "model", inner)
        spec = outer.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        packed = {e[0] for e in spec.stacked_params}
        assert "qkv_proj" in packed

    @pytest.mark.parametrize(
        "model_module,model_cls",
        [
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5ForCausalLM"),
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5MoeForCausalLM"),
        ],
    )
    def test_qwen3_5_empty_spec(self, model_module, model_cls):
        import importlib

        mod = importlib.import_module(model_module)
        if not hasattr(mod, model_cls):
            pytest.skip(f"{model_cls} not found in {model_module}")
        cls = getattr(mod, model_cls)
        assert hasattr(cls, "get_weight_layout_spec"), f"{model_cls} missing get_weight_layout_spec"
        instance = cls.__new__(cls)
        instance.config = make_mock_config()
        spec = instance.get_weight_layout_spec()
        assert spec.stacked_params == []
        assert spec.expert_params is None

    @pytest.mark.parametrize(
        "model_module,model_cls,inner_attr",
        [
            ("vllm.model_executor.models.gemma2", "Gemma2ForCausalLM", "model"),
            ("vllm.model_executor.models.gemma3", "Gemma3ForCausalLM", "model"),
            ("vllm.model_executor.models.mixtral", "MixtralForCausalLM", "model"),
        ],
    )
    def test_standard_model_has_spec(self, model_module, model_cls, inner_attr):
        import importlib

        mod = importlib.import_module(model_module)
        cls = getattr(mod, model_cls)
        assert hasattr(cls, "get_weight_layout_spec"), f"{model_cls} missing get_weight_layout_spec"
        instance = cls.__new__(cls)
        instance.config = make_mock_config()
        inner_name = model_cls.replace("ForCausalLM", "Model")
        if hasattr(mod, inner_name):
            inner = getattr(mod, inner_name).__new__(getattr(mod, inner_name))
            inner.config = instance.config
            if model_cls == "MixtralForCausalLM":
                inner.get_expert_mapping = MagicMock(
                    return_value=[
                        ("experts.w13_", "experts.0.w1.", 0, "w1"),
                    ]
                )
                object.__setattr__(inner, "num_redundant_experts", 0)
            object.__setattr__(instance, inner_attr, inner)
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        assert len(spec.stacked_params) > 0

    def test_internlm2_spec(self):
        from vllm.model_executor.models.internlm2 import InternLM2ForCausalLM

        assert hasattr(InternLM2ForCausalLM, "get_weight_layout_spec")
        instance = InternLM2ForCausalLM.__new__(InternLM2ForCausalLM)
        instance.config = make_mock_config()
        spec = instance.get_weight_layout_spec()
        assert isinstance(spec, WeightLayoutSpec)
        # InternLM2 uses "w1"/"w3" as HF names, not "gate_proj"/"up_proj"
        hf_names = {e[1] for e in spec.stacked_params}
        assert "w1" in hf_names
        assert "w3" in hf_names

    def test_load_weights_uses_spec(self):
        """load_weights in updated models must call get_weight_layout_spec(), not redeclare list literal."""
        import importlib
        import inspect

        cases = [
            ("vllm.model_executor.models.gemma2", "Gemma2Model"),
            ("vllm.model_executor.models.gemma3", "Gemma3Model"),
            ("vllm.model_executor.models.mixtral", "MixtralModel"),
            ("vllm.model_executor.models.internlm2", "InternLM2ForCausalLM"),
        ]
        for mod_path, cls_name in cases:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            src = inspect.getsource(cls.load_weights)
            assert "get_weight_layout_spec()" in src, f"{cls_name}.load_weights() must call get_weight_layout_spec()"
            assert "stacked_params_mapping = [" not in src, (
                f"{cls_name}.load_weights() must not re-declare stacked_params_mapping as list literal"
            )

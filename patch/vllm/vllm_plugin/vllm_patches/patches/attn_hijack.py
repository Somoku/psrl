import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Kept identical to the module the source patch used to modify. The patch is
# applied at runtime instead so the model source stays untouched.
_TARGET_MODULE = "vllm.model_executor.models.qwen2"
_TARGET_CLASS = "Qwen2Attention"
_ENV_VAR = "VLLM_DISABLE_ATTN"


def apply_attn_hijack_patch() -> None:
    """Allow bypassing attention entirely for cost-modeling runs.

    When ``VLLM_DISABLE_ATTN=1``, ``Qwen2Attention.forward`` returns ``q``
    (post-RoPE) in place of the attention output, while still running
    ``o_proj``. This isolates the cost of everything *except* attention, which
    is what the rollout cost model needs.

    Gated at runtime by the environment variable, so there is no cost when the
    flag is off. The ``forward`` body below mirrors
    ``vllm/model_executor/models/qwen2.py`` at v0.29.0 with a single extra
    branch; re-derive it on every vLLM bump.

    Only ``Qwen2Attention`` is affected, matching the previous source patch.
    """
    _patch_qwen2_attention()


def _patch_qwen2_attention() -> None:
    import importlib

    import torch

    module = importlib.import_module(_TARGET_MODULE)
    target = getattr(module, _TARGET_CLASS)

    _INIT_SENTINEL = "_psrl_attn_init_patched"
    _FWD_SENTINEL = "_psrl_attn_forward_patched"

    if not getattr(target, _INIT_SENTINEL, False):
        _original_init = target.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            self.disable_attn = os.environ.get(_ENV_VAR, "0") == "1"
            if self.disable_attn:
                psrl_logger.info(
                    "Hijacking %s attention: %s=1, attention is bypassed.",
                    _TARGET_CLASS,
                    _ENV_VAR,
                )

        target.__init__ = _patched_init
        setattr(target, _INIT_SENTINEL, True)

    if not getattr(target, _FWD_SENTINEL, False):

        def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Apply QK normalization if enabled (before RoPE)
            if self.qk_norm:
                # Reshape to apply per-head normalization
                # q shape: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
                total_tokens = q.shape[0]
                q = q.view(total_tokens, self.num_heads, self.head_dim)
                k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

                q = self.q_norm(q)
                k = self.k_norm(k)

                q = q.view(total_tokens, self.q_size)
                k = k.view(total_tokens, self.kv_size)

            q, k = self.rotary_emb(positions, q, k)
            if self.disable_attn:
                attn_output = q
            else:
                attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        target.forward = forward
        setattr(target, _FWD_SENTINEL, True)

    psrl_logger.info("Patched %s.__init__/forward for %s.", _TARGET_CLASS, _ENV_VAR)

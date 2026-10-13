"""
Logprob gap between a bf16 rollout and a block FP8 rollout fed by the trainer side quantiser.

The FP8 engine starts from dummy weights. Every canonical key the converter exposes is then
written as the trainer would publish it: block FP8 weights with their scales, other keys as
bf16. That is the byte layout the pull lands, so a correct layout gives the usual FP8 gap and
a wrong one gives garbage.

    python tests/nixl/bench_fp8_logprob_gap.py --model <hf dir> --mode bf16 --out /tmp/bf16.pt
    python tests/nixl/bench_fp8_logprob_gap.py --model <hf dir> --mode fp8 --out /tmp/fp8.pt --ref /tmp/bf16.pt
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1905 Albert Einstein published",
    "The derivative of sin(x) is",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["bf16", "fp8"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams

    kwargs = {"model": args.model, "dtype": "bfloat16", "enforce_eager": True, "max_model_len": 512}
    if args.mode == "fp8":
        kwargs["load_format"] = "dummy"
        kwargs["quantization"] = "fp8"
        kwargs["hf_overrides"] = {
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        }
    llm = LLM(gpu_memory_utilization=0.5, **kwargs)
    if args.mode == "fp8":
        from safetensors import safe_open

        from psrl.utils.converter.vllm_converter import convert_vllm_inplace
        from psrl.utils.nixl.block_fp8 import FP8_DTYPE, quantize_blockwise

        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        state, _ = convert_vllm_inplace(model, tp_rank=0, ep_rank=0, parameter_mapping=None)
        written = 0
        torch.set_grad_enabled(False)
        for path in glob.glob(os.path.join(args.model, "*.safetensors")):
            with safe_open(path, "pt") as handle:
                for key in handle.keys():
                    if key not in state:
                        continue
                    target = state[key]
                    source = handle.get_tensor(key).cuda().reshape(target.shape)
                    if target.dtype == FP8_DTYPE:
                        tile = (1,) * (target.dim() - 2) + (128, 128)
                        q, s = quantize_blockwise(source, tile)
                        target.copy_(q)
                        state[key + "_scale_inv"].copy_(s)
                    else:
                        target.copy_(source.to(target.dtype))
                    written += 1
        print(f"wrote {written} keys")
    params = SamplingParams(max_tokens=32, temperature=0.0, logprobs=1, prompt_logprobs=None)
    outs = llm.generate(PROMPTS, params, use_tqdm=False)
    result = [
        {
            "tokens": list(o.outputs[0].token_ids),
            "logprobs": [next(iter(lp.values())).logprob for lp in o.outputs[0].logprobs],
            "text": o.outputs[0].text,
        }
        for o in outs
    ]
    torch.save(result, args.out)
    for r in result:
        print(repr(r["text"][:80]))
    if args.ref:
        ref = torch.load(args.ref)
        agree = [sum(a == b for a, b in zip(x["tokens"], y["tokens"])) / len(y["tokens"]) for x, y in zip(result, ref)]
        print(f"RESULT greedy token agreement with bf16 per prompt {[round(a, 3) for a in agree]}")
        # Logprob of the reference's own greedy tokens is not available here, so compare the first token only.
        gap = [abs(x["logprobs"][0] - y["logprobs"][0]) for x, y in zip(result, ref)]
        print(f"RESULT first token logprob gap {[round(g, 4) for g in gap]}")


if __name__ == "__main__":
    main()

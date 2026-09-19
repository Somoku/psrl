# PSRL 支持 FP8 rollout + BF16 training 设计方案（v3，CPU-only PS）

> 状态：设计（不落地）。适用约束：`ps_mode = nixl_cpu`（PS 仅在 CPU，不考虑 `nixl_gpu`）。
> 目标：让 PSRL 支持「rollout 用 FP8 推理、training 用 BF16」的训练形态，其难点在于 PSRL 有 CPU 侧 PS 模块、且 rollout 侧是「从 PS 拉取权重」而非训练直连。
> 调研基线（本节点 `/home/base`）：`vllm v0.29.1rc0-200-g9ca6dbba71`、`miles`(Megatron+SGLang)、`vime`(Megatron+vLLM)、`sglang`；PSRL 源码位于 `/home/psrl`。

---

## 0. TL;DR

在 CPU-only PS 约束下，把权重更新拆成三阶段、各就其位：

1. **量化（BF16→fp8 weight + raw scale）在 train GPU**：PS 无 GPU、blockwise fp8 是 GPU-only kernel，量化必须上移到 train。
2. **存储 + 重分片在 CPU PS + nixl comm plan**：PS 退化为 **fp8+scale 的纯字节仓库**；因 push/pull 缓冲同为 fp8，`train_gen_model_share()` 为 True → **零拷贝共享单缓冲、PS 零计算、host 占用 ~0.5×**。
3. **运行时格式化（转置/tma 对齐/ue8m0 打包/融合分片 requant）在 rollout GPU（拉取后）**：分区相关、需 GPU，但是廉价的 **fp8 域**操作，全程复用 vLLM `fp8_utils`。

净效果：PS 极简、`train→PS` 与 `PS→rollout` **两段带宽均减半**、显存/host 最省、layout 权威全部交给 vLLM。

---

## 1. 现状与问题定位（基于 PSRL 源码）

PSRL 权重更新链路（区别于 miles/vime 的 trainer→engine 直连，PSRL 有 **CPU 侧 PS 中介**）：

```
train worker ──push──▶ PS.client_for_push(train buffer) ──transfer──▶ PS.client_for_pull(gen buffer) ──nixl client_read──▶ vLLM 已注册参数张量
```

关键事实：

- **双缓冲骨架已存在**。`PSStoragePlan`（`psrl/workers/ps/ps_storage_worker.py`）已区分 `train_model_dtype` / `gen_model_dtype`，`train_gen_model_share()` 判断二者是否相等。相等 → pull 缓冲**绑定同一块内存**（零拷贝）；不等 → pull 端注册独立缓冲，由 `transfer_train_to_gen` 搬运。`psrl/trainer/ppo/ray_trainer.py:1907` 已把 `gen_model_dtype` 接到 `rollout.dtype`——**配置入口天然存在**。
- **rollout 侧 nixl 是「直写已注册参数」**。`vllm_extension.py::nixl_pull_model_core` 对每个 key 调 `client_read`，RDMA 直接写进 vLLM 运行时参数缓冲（`convert_vllm_inplace` 注册的 `unified_state_dict`），**绕过 `load_weights` 与 `process_weights_after_loading`**。这是 PSRL 的性能核心，也是 FP8 的最大约束点。
- **PS 缓冲设备由 `ps_mode` 决定**：`nixl_cpu` → host RAM（`ps_storage_worker.py:115` `use_gpu=False`）。本方案锁定该模式。

**为什么现在不支持 FP8（三个硬伤）：**

1. **gen 缓冲 layout 结构性错误**。`init_model` 里 gen 缓冲用 `from_config(torch_dtype=fp8)` 构造——只是把**所有**参数朴素 cast 成 fp8，**没有 scale 张量**；而真实 vLLM FP8 模型是 `weight`(fp8) + `weight_scale`(per-tensor)/`weight_scale_inv`(blockwise)，且只量化部分线性层。参数集对不上，nixl 按 key 传输直接失配。
2. **量化语义缺失**。`transfer_train_to_gen` 只有一句 `target.copy_(src)`——BF16→FP8 朴素 round，**不算 scale**，动态范围尽失。
3. **post-load layout 未处理**。`psrl/utils/converter/weight_layout_transforms.py` 覆盖 qkv/merged_column/fused_moe/transpose 等 BF16 变换，但**无 fp8/scale 意识**；vLLM 的 `process_weights_after_loading` 会对 scale 做 transpose/requantize/ue8m0 打包，nixl 直写路径绕过了它。

---

## 2. 外部框架调研（`/home/base`）

### 2.1 miles（Megatron+SGLang）、vime（Megatron+vLLM）——几乎同源

- **量化发生在 trainer 侧**，在 megatron→HF 导出的**逐参数流水线**中（`miles/.../update_weight/hf_weight_iterator_bridge.py::_postprocess_and_quantize` → `megatron_to_hf/processors/quantizer_fp8.py::quantize_params_fp8`）。
- **只量化推理引擎跑 FP8 的那部分**：显式白名单（`self_attention.linear_{proj,qkv,q/kv_*}`、`mlp.linear_fc1/fc2`、MoE experts/shared_experts、MLA、linear-attn 等）；embedding/norm/router/bias/lm_head 保持 BF16。
- **emit `(weight_fp8, scale)` 成对**，命名对齐引擎 checkpoint：per-tensor → `...weight_scale`（`scale = amax/FP8_MAX`）；blockwise([128,128]) → `...weight_scale_inv`，按后端选 UE8M0（Blackwell/DeepGEMM）或 FP32 block scale（Hopper）。
- **原子分组（关键约束）**：`miles/.../hf_weight_iterator/atomic_groups.py` 指出——引擎 loader 会把若干 HF 权重融合成一个引擎参数（如 `wq_a+wkv→wqkv_a`），**必须同一次 load 调用到达，半到达 = 静默 no-op / 断言失败**。故 weight+scale 与模型特定融合组都要**原子分桶**。
- **传输**：`update_weights_from_tensor`(CUDA IPC, colocated) / `_from_distributed`(NCCL) / `_from_disk`。vime 直接调 **vLLM 的 `pack_tensors`** 把混合 dtype（fp8 权重 + fp32 scale）打包进单一 `uint8` buffer 走 CUDA IPC。

一句话：**「发送端量化、成对原子传输、命名对齐引擎」**。

### 2.2 vLLM 0.29 接口支持

- **一等公民权重传输子系统** `vllm/distributed/weight_transfer/`：factory 选择后端 `WeightTransferConfig.backend ∈ {"nccl","ipc","sparse_nccl","sharded_rdt"}`；会话协议 `start_weight_update → update_weights(chunk) → finish_weight_update`（`vllm/v1/worker/gpu_worker.py:1425`）。**`sharded_rdt_engine` 本身就用 NIXL**（`ray.experimental.register_nixl_memory` + `vllm.distributed.nixl_utils`）——PSRL 自研 nixl 的官方对标 / 潜在替代。
- **`packed_tensor`**：`pack_tensors`/`unpack_tensor` 原生支持混合 dtype 打包。
- **`fp8_utils`（`vllm/model_executor/layers/quantization/utils/fp8_utils.py`）全套可复用件**：
  - 量化：`per_block_cast_to_fp8`（`vllm/utils/deep_gemm.py`，GPU blockwise，vime 同款）。
  - 构造参数：`create_fp8_weight_parameter` / `create_fp8_scale_parameter`（layout、命名与引擎逐字节一致）。
  - 运行时格式化：`process_fp8_weight_{tensor,channel,block}_strategy` + `deepgemm_post_process_fp8_weight_block` + `requant_weight_ue8m0_inplace`（**即 `process_weights_after_loading` 的实现体**）。
  - 分片校验：`validate_fp8_block_shape` / `validate_fp8_block_shape_moe`。
  - DeepGEMM/UE8M0/TMA：`is_deep_gemm_e8m0_used` / `get_mn_major_tma_aligned_packed_ue8m0_tensor` / `get_tma_aligned_size`。
- **`is_weights_pre_processed()`**（`fp8.py::process_weights_after_loading` 首段）：当权重已是「运行时格式」时，post-processing 退化为近乎 no-op（只补 `input_scale=None` 给 dynamic 激活）。这是 **D-i（直写即终态）的官方开关**。

---

## 3. 约束的根本影响：为什么 v3 唯一化到「train 侧量化」

`ps_mode = nixl_cpu` ⇒ PS 缓冲在 host RAM、**PS 无 GPU、不能做任何量化计算**（blockwise fp8 是 GPU-only kernel；CPU 上对 `float8_e4m3fn` 仅支持存储与朴素 cast，无高效 per-block amax+缩放）。因此：

> **量化不能发生在 PS 上，必须上移到有 GPU 的一侧。** 早期「PS 侧量化」方案作废。

CPU PS 的性能画像是 **NIC/PCIe 带宽受限**、且**无法把量化与传输重叠**（没 GPU）。这把方案唯一化——量化放到 **上游 train GPU**，让 PS 存/传的就是 fp8，把最紧张的 NIC 字节数直接减半。

对「rollout 侧全量量化」（PS 存 bf16、rollout 拉 bf16 再量化）的排除理由：PS 需存 bf16（1×、NIC 传输翻倍），且 rollout 需**常驻 bf16 影子缓冲**（注册给 nixl，不宜频繁 dereg）——host 带宽与 inference 显存双输。故不选。

---

## 4. 架构决策：三阶段流水，各就其位

| 阶段 | 做什么 | 放哪 | 为什么 |
|---|---|---|---|
| ① 量化 | BF16 → fp8 weight + **raw scale**（canonical/block 对齐层面） | **train GPU** | 需 GPU；分区无关的 raw scale 便于干净重分片 |
| ② 存储 + 重分片 | 存 fp8+scale；train-HF 分片 → vLLM-TP 分片 | **CPU PS + nixl comm plan** | 纯字节搬运，CPU 胜任；fp8 使 NIC/host 占用减半 |
| ③ 运行时格式化 | 转置(K,N)/tma 对齐/ue8m0 打包/融合分片 requant | **rollout GPU（拉取后）** | 分区相关、需 GPU；是廉价的 **fp8 域**操作，非重量化 |

**核心红利：PS 与 train 的 push/pull 缓冲同为 fp8 ⇒ `train_gen_model_share()` 返回 True ⇒ 零拷贝共享单缓冲、`transfer_train_to_gen` 变 no-op、PS 完全无计算、host 占用 ~0.5×。** CPU-only 约束从「障碍」变成「恰好合身」。

### 端到端数据流

```
train(BF16 计算)
  └─[①GPU 量化] per_block_cast_to_fp8 / per-tensor → fp8 weight + raw fp32 scale
     (在 block 对齐的 canonical 层面; nixl_convert_params 产出 fp8+scale 的 unified dict)
  └─push(nixl, fp8+scale) ─────────────▶ CPU PS 单缓冲(fp8+scale, host RAM, ~0.5×)
                                            │ train_gen_model_share=True → 无 transfer、无计算
  └─rollout client_read(nixl, fp8+scale, CPU→GPU RDMA)
        └─[③rollout GPU 轻量格式化] 复用 vLLM:
              process_fp8_weight_{tensor,block}_strategy / deepgemm_post_process_* /
              requant_weight_ue8m0_inplace(尽量 in-place)
           → 写入 vLLM 层最终运行时 weight / weight_scale_inv
```

对比早期草案：PS 从「BF16 push + fp8 gen 双缓冲(~1.5×, 且要算量化)」变为「fp8 单缓冲(~0.5×, 零计算)」；两段带宽都减半。

---

## 5. 最大化复用 vLLM（保证可扩展性）

- **量化**：`per_block_cast_to_fp8`（blockwise）+ 现成 per-tensor 逻辑。
- **参数 layout 权威**：train 侧 unified dict 与 rollout 侧注册，均以 `create_fp8_weight_parameter`/`create_fp8_scale_parameter` 为准（shape、命名 `weight_scale`/`weight_scale_inv` 与引擎一致）。
- **阶段③**：直接调 `process_fp8_weight_tensor_strategy`（融合分片 requant + 转置）、`deepgemm_post_process_fp8_weight_block` + `requant_weight_ue8m0_inplace`（ue8m0/tma）、`is_deep_gemm_e8m0_used` 分流。与引擎行为同源。
- **零 rollout 计算的进阶变体**：当 **train 与 inference 并行度匹配（如 colocated、分区一致）** 时，让 train 直接产出「按 inference 分区的运行时格式」push，rollout 打开 **`is_weights_pre_processed()=True`** → 阶段③退化为 no-op、nixl 直写即终态。作为可选优化，不作默认（避免把 train 强耦合到 inference TP，且并行度不同即失效）。
- **量化层白名单**：从引擎 `Fp8LinearMethod` 反推，做成共享 **FP8 plan**，train/rollout 两处共用，消除漂移。

---

## 6. 分片对齐：确定性方案

- **在 block 对齐的 canonical 层面量化**（miles/vime 在 megatron→HF 导出时正是如此）：raw block-scale 形如 `(⌈N/128⌉, ⌈K/128⌉)`，对任意 **block 对齐的分区**都是干净子块，重分片（阶段②）与再切分（阶段③）**均不跨块**。
- 落地为**断言而非风险**：对每个量化 key 调 `validate_fp8_block_shape` / `_moe`；不满足整除的模型×TP 组合 **vLLM 引擎本身也会拒绝**，PSRL 报同一约束，不引入新风险面。
- train 侧优先在其**原生分片**（megatron 列并行保留完整 input 维、行并行保留完整 output 维，天然多为 block 友好）上量化；仅当某维不对齐时，对该参数做**分桶 gather** 到对齐形态（逐参数、瞬态，沿用 converter 既有 per-param 流水，不 gather 全模型）。
- **融合/MoE 边界**：merged QKV / gate_up / MoE w13·w2 沿用 vLLM 对 merged「末段可不整除」的豁免；并强制 weight 与其 scale 同属一个 **atomic group** 一起传。

---

## 7. AI-infra 优化点（v3 已内建）

1. **显存/host 冗余**：fp8 共享单缓冲，PS host 占用 ~0.5×；`transfer_train_to_gen` 无操作，消除双缓冲与逐 step PS 计算。
2. **小传输爆炸**：weight 与 scale **相邻分配**，借 `psrl/utils/nixl/client.py::_merge_regions` 合并为单 nixl 描述符；按 atomic group 组织 pull/wait，减少 wait 次数。理念对齐 vLLM `pack_tensors`。
3. **流水重叠**：量化在 train 侧**分桶流式**，与既有 param gather/push 管线重叠（参考 miles atomic-group 分桶）；阶段③在 rollout **generation 已暂停的窗口**内做，不抢生成算力。
4. **免传 input_scale**：`activation_scheme=dynamic`，激活运行时量化；rollout 侧仅补 `input_scale=None`（正是 `is_weights_pre_processed` 分支所做）。
5. **NIC 带宽**：CPU PS 是 NIC/PCIe 受限点，两段传输都走 fp8 → 最紧资源直接减半（相对「rollout 拉 bf16」方案的核心优势）。
6. **sleep/wake_up**：fp8 weight 与 scale 全部纳入 `nixl_register_after_wake_up` 重注册集合，防物理页变更后 scale 指针失效。
7. **精度失配（算法层 hook）**：rollout(fp8) 与 train(bf16) logprob 系统性偏差，需 off-policy / importance-sampling 修正；infra 保证 rollout logprob 与 fp8 权重版本对齐（复用 PSRL 版本机制）。

### CPU PS + rollout 格式化新引入的问题与对策

- **rollout 侧新增中间缓冲**：阶段③需一块 fp8+scale 中间态（拉取落点）+ 最终运行时张量。对策：**预分配、跨 step 复用**；`requant_weight_ue8m0_inplace` 走 in-place；仅转置需一次额外拷贝。总体 ~0.5× 瞬态，远小于「rollout 拉 bf16」的 1× 常驻 bf16 影子。
- **train 侧耦合**：量化织入 `nixl_convert_params`（fsdp/megatron 两路），改动面比「PS 侧量化」大；但换来 PS 极简与全链路带宽减半，在 CPU PS 下是正确取舍。

---

## 8. 落地边界 / 改动面 / 验证（仅设计）

### 改动面
- `psrl/workers/train/engine_train_worker.py::nixl_convert_params`：产出 fp8+scale 的 unified dict（阶段①，GPU 量化 + block 对齐）。
- `psrl/utils/config.py` 校验：`rollout.dtype=fp8` 时强制 `ps_mode=nixl_cpu` 走 train 侧量化；引入并下发**共享 quantization_config**（`fmt=e4m3`、`activation_scheme=dynamic`、`weight_block_size`、scale 格式）；`gen_model_dtype=fp8` 使共享缓冲生效。
- `psrl/workers/gen/vllm_extension.py::nixl_pull_model_core`：拉取后接**阶段③**运行时格式化（复用 vLLM `fp8_utils`）。
- `psrl/utils/converter/vllm_converter.py::convert_vllm_inplace`：量化参数走 identity + 注册**终态 fp8 运行时张量**（不再走 vllm→hf 变换）。
- `nixl_register_after_wake_up`：纳入 scale 张量。
- 新增**共享 FP8 plan 模块**：从引擎 `Fp8LinearMethod` 反推量化层白名单 + block_size + scheme。

### 不改
- PS 存储 / staleness 控制 / broadcast init / 版本机制（PS 反而更简单）。
- nixl 传输内核。

### 验证
1. 小模型比对「v3 三阶段产出」vs「vLLM 从 checkpoint 原生 fp8 加载 + `process_weights_after_loading`」逐张量等价（或在 ue8m0/requant 容差内）。
2. `validate_fp8_block_shape` 对目标模型×TP 全通过。
3. 端到端几步比 rollout logprob 与 bf16 参考偏差落在预期区间。
4. 实测带宽 / host 占用确认两段 ~0.5×。

---

## 9. 备选与演进

- **进阶：`is_weights_pre_processed()=True` 零 rollout 计算**——train/infer 并行度匹配时启用，阶段③消失。
- **远期：迁移到 vLLM 原生 `weight_transfer`（`sharded_rdt`=NIXL 后端）**——让 vLLM 托管 fp8 打包/加载/格式化。优点是上游维护正确性；缺点是它为 trainer→engine 直连，不含 PSRL 的 CPU 侧 PS 中介（staleness/broadcast/版本解耦是 PSRL 核心价值），整体替换代价大。建议保持接口/命名与其兼容（复用 `pack_tensors`、scale 命名、`fp8_utils`），为将来留路。

---

_附：本方案基于对 `/home/psrl`（PSRL）、`/home/base/{vllm,miles,vime,sglang}` 的源码调研；关键引用见正文行号/函数名。_

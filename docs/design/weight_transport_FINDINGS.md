# 权重传输：实测结论

本文收录权重传输数据面的实测数据，设计见 `weight_transport.md`，落地状态见
`weight_transport_implementation.md` §9。每条结论标出能复现它的最轻档位
（CPU < 单 GPU < 单节点 < 多节点），复现命令见 §8。

**环境。** H20 96 GB × 8 每节点，节点间 RoCE，8 个 mlx5 端口组成 4 个 400G bond。
vLLM 0.30.0，NIXL UCX 后端。模型 Qwen3-8B（dense，8.19 GB/每 gen TP2 rank）与
Qwen3-30B-A3B（MoE，128 experts，48 层）。

**判据。** e2e harness 在 train 侧按 `(key, version, 全局坐标)` 的哈希填写元素原始位，
gen actor 就地按自己声明的 sharding 复核，错位元素的漏检概率是 `2^-位宽`。
block FP8 格在 train 侧填「block 量化恰好可逆」的值，于是 gen 侧的 FP8 位与 scale 都能逐位判定。
每格跑两轮，第二轮换新版本号，判「每个 rollout 都拿到新版本，不是缓存」。
**第一轮含 prepped handle 构建与注册预热，稳态看第二轮。**

---

## 1. 带宽：按并发聚合测，不按端口数推

| 链路 | 单对 | 并发聚合 |
|---|---|---|
| 同节点 GPU→GPU（NVLink） | 365 GB/s | **793 GB/s**（4 对） |
| 跨节点 RoCE | 49 GB/s（一个 400G bond） | **386 GB/s**（8 对，每节点） |

- 单流只占一个 bond，用单流数字推节点聚合会低估约 8×。按端口数推（`8 × 200G × 0.85`）也会错，
  端口 bond 成了 4 条链路。cost model 的 `BW_NIC` 应取探测值。
- 同节点与跨节点的节点聚合只差约 2×。**扩散的收益因此不在 rollout 入口，在 trainer 出口**（§4）。

档位：多节点，`tests/nixl/scripts/run_pair_bw.sh`。

## 2. peer 读的粒度：gen 本地整张量

designate 与 follower 的 gen TP rank 相同，本地张量逐字节相同，两者之间不需要重分片。
沿用全局 finest-mesh cell 时，row-parallel key 的每个 cell 是 `hidden_size` 行的 strided 视图。

Qwen3-8B 一个 gen client（8.19 GB），字节逐位校验：

| 链路 | 按 finest cell | 按 gen 本地整张量 | 单 descriptor 上界 |
|---|---|---|---|
| 同节点 | 590,333 desc，2415 ms | **399 desc，22.5 ms**（365 GB/s） | 20.8 ms |
| 跨节点单对 | 679 ms | **167 ms**（49 GB/s） | 175 ms |

整张量读贴近单 descriptor 上界，剩余开销是字节而不是 descriptor。CPU 计数下 5078 个 peer 读
100% 可整张量合并。档位：CPU（计数）、单节点 2 GPU 与跨节点单对（传输），
`bench_peer_whole_tensor.py`。

## 3. 源端 strided：发布副本消除

descriptor 数取两端最大值。读方切得比 trainer 细时，trainer 的 shard 对读方是逐行 strided 的。
MoE 下 EP 一侧整 expert、TP 一侧切 expert 内部，strided cell 数随 expert 数线性增长。
trainer 为这类 shard 在发布点刷新一份连续副本后，单节点 4+4 GPU 的稳态拉取：

| 配置 | 副本前 descriptor（每 train client） | 副本前 | **现在** |
|---|---|---|---|
| Megatron TP2·PP2 → vLLM TP4 | 295,128 | 1.60 s | **0.035 s** |
| FSDP2 → TP2 | 147,456 | 1.43 s | **0.067 s** |
| HSDP 2×2 → TP2 | 294,912 | 1.65 s | **0.062 s** |
| MoE Megatron EP4 → TP4 | 12,976,128 | 66.5 s | **0.158 s** |
| MoE FSDP2 → TP4 | 12,681,216 | 70.6 s | **0.159 s** |
| MoE Megatron TP2·EP2 → TP4·EP4 | 196,608 | 0.84 s | **0.104 s** |

副本只为「被读方细切」的 shard 存在，由 `publish_copy_budget_mb` 封顶。档位：单节点。

## 4. 同节点扩散：trainer 出口

**出口随 rollout 数线性增长，扩散把它压到每节点一份。** CPU，生产 planner + 真实 shard 代数，
Qwen3-8B，train TP4·PP2 × 24 rank，gen TP2，每 gen 节点 8 个 rollout：

| gen 节点 | 无扩散 trainer 出口 | 扩散 trainer 出口 |
|---|---|---|
| 2 | 131 GB | 32.8 GB |
| 4 | 262 GB | 65.5 GB |
| 8 | 524 GB | 131 GB |

**多节点同轮 A/B。** Qwen3-8B，Megatron TP4·PP2 × 16 rank → 16 个 vLLM TP2 rollout，
同一次运行内先关扩散、再开扩散：

| | trainer 出口 | 每 rollout descriptor 上限 | 稳态 sync |
|---|---|---|---|
| 无扩散 | 131 GB | 869 | 0.67–0.84 s |
| 扩散，上游读完再读 peer（5 节点） | 32.8 GB | 537 | 0.285 s |
| **扩散，分波重叠（4 节点）** | **32.8 GB** | **537** | **0.178 s**（上游 0.136 s，peer 0.038 s） |

- 四个 instance 轮流当 designate，节点入口分散到每个 instance 的 NIC 上。
- 分波后 peer 阶段只剩最后一波的尾巴。
- 首次 sync 为 3.3 s，含 peer 读 prepped handle 的构建。
- 16 个 rollout 的 32 个 gen client 两轮均通过 layout 判据，fallback key 为 0。

档位：CPU（出口）、多节点（时间）。

## 5. 配置矩阵（单节点 4 train + 4 gen GPU，layout 判据，两轮）

| 格 | 精度 | 判据 | 第一轮 / 稳态 | trainer 出口 |
|---|---|---|---|---|
| Megatron TP2 → TP2 | bf16 | ✅ 399 keys × 4 × 2 轮 | 0.206 / 0.065 s | 32.8 GB |
| Megatron TP2·PP2 → TP4 | bf16 | ✅ | 0.292 / 0.035 s | 16.4 GB |
| Megatron TP4 → TP2，扩散 | bf16 | ✅ | 1.029 / 0.052 s | 16.4 GB |
| FSDP2 → TP2 | bf16 | ✅ | 0.879 / 0.067 s | 32.8 GB |
| HSDP 2×2 → TP2 | bf16 | ✅ | 1.035 / 0.062 s | 32.8 GB |
| FSDP2 fp32 → TP2 | fp32 → bf16 | ✅ | 0.335 / 0.057 s | 32.8 GB |
| FSDP2 fp32 → TP1，扩散 | fp32 → bf16 | ✅ | 1.341 / 0.141 s | 16.4 GB |
| colocate Megatron TP2 → TP2，扩散（8+8 共卡） | bf16 | ✅ | 2.774 / 0.058 s | 16.4 GB |
| MoE Megatron EP4 → TP4 | bf16 | ✅ 579 keys | 0.473 / 0.158 s | 61.1 GB |
| MoE Megatron TP2·EP2 → TP4·EP4 | bf16 | ✅ | 0.297 / 0.104 s | 61.1 GB |
| MoE FSDP2 → TP4 | bf16 | ✅ | 2.109 / 0.159 s | 61.1 GB |
| Megatron TP2 → TP2 | bf16 → FP8 | ✅ FP8 位与 scale 逐位 | 0.256 / 0.040 s | 18.9 GB |
| Megatron TP2·PP2 → TP4 | bf16 → FP8 | ✅ | 0.255 / 0.028 s | 9.4 GB |
| Megatron TP4 → TP2，扩散 | bf16 → FP8 | ✅ | 0.847 / 0.046 s | 9.4 GB |
| FSDP2 fp32 → TP2 | fp32 → FP8 | ✅ | 0.583 / 0.044 s | 18.9 GB |
| FSDP2 fp32 → TP1，扩散 | fp32 → FP8 | ✅ | 0.996 / 0.074 s | 9.4 GB |
| colocate Megatron TP2 → TP2，扩散 | bf16 → FP8 | ✅ | 2.403 / 0.037 s | 9.4 GB |
| MoE Megatron TP2·EP2 → TP4·EP4 | bf16 → FP8 | ✅ 579 keys | 0.675 / 0.075 s | 31.3 GB |
| MoE FSDP2 → TP4·EP4 | bf16 → FP8 | ✅ | 0.663 / 0.064 s | 31.3 GB |

- FP8 格的出口约为 bf16 的 0.58 倍：权重减半，加上 scale、norm 与 embedding 等不量化的 key。
- 扩散格的出口是一份模型（16.4 GB / FP8 9.4 GB），与节点上的 rollout 数无关。

**未支持的格，注册时拒绝：** MoE Megatron TP2·EP2 → vLLM TP4（EP1），bf16 → FP8。
Qwen3-30B-A3B 的 `moe_intermediate_size = 768`，TP4 每 rank 192 行，不是 128 的整数倍。
vLLM 0.30 此时把 expert 的 scale 网格细化到 64 × 64，把 128 × 128 的 scale 无损上采样过去。
trainer 发布的是 128 × 128 网格的 scale，它在 192 行处切不开，所以注册时拒绝并报出 key 名。
支持它需要 trainer 按 rollout 的细化网格发布 scale（值不变，逐 tile 复制），见实现规划 §11 的 W9。
目前改用 gen EP 即可绕开。

## 6. block FP8 的数值

- `block_fp8.quantize_blockwise` 与 vLLM 的 `per_block_cast_to_fp8` 逐位一致（CPU）。
- 真实 Qwen3-8B 权重经 trainer 的量化器发布、FP8 rollout 生成，与 bf16 rollout 对比
  （4 个 prompt，32 token，贪心）：贪心 token **100% 一致**，首 token logprob 差
  0.003–0.10。档位：单 GPU，`bench_fp8_logprob_gap.py`。

## 7. 环境限制（与权重传输无关）

- **FlashInfer autotune 在节点已有缓存时死锁。** vLLM 的 MoE autotune 由 rank 0 读缓存文件再广播。
  两次失败的日志里 rank 0 报 `Config cache hit`，另一个 rank 进入 `Tuning trtllm::fused_moe`，
  随后 TP all-reduce 等满 600 s 超时。卡点在 vLLM 初始化，在任何传输发生之前。
  `ray start` 前把 `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` 指向一个空目录，所有 rank 一起从头调优，该格即通过。
- **MoE rollout 的 DP > 1 在 harness 下跑不起来。** harness 把一个节点上的全部 gen rank 起成
  一个 vLLM DP 组（`external_launcher`）。MoE 层在 DP > 1 时跨 DP rank 分发 token
  （`naive_dp_ep` → `all_gatherv`），该路径断言 `pynccl_comm` 存在而失败，与 gen 是否开 EP 无关。
  所以 MoE 格只覆盖 4 张 gen 卡组成一个 rollout instance（gen TP = 4）的切法，
  MoE 下 vLLM DP > 1 时 expert 跨 DP × TP 切分的布局尚无 L2 覆盖（实现规划 §11 W11）。
- **DeepGEMM 在 CUDA < 12.9 下不编译**（`build(vllm)` 提交），FP8 MoE 的性能数字需注意这一点。

## 8. 复现

```bash
# CPU：出口与 descriptor 几何
python tests/nixl/bench_direct_supply_geometry.py --gen-nodes 2 4 8
# 带宽与 peer 读粒度（target 与 initiator 各一进程）
tests/nixl/scripts/run_pair_bw.sh <target-ip> <initiator-ip> <pairs> --modes cells whole flat
INIT_GPU_OFFSET=4 tests/nixl/scripts/run_pair_bw.sh <ip> <ip> 4 --modes whole   # 同节点
# 单节点矩阵格（节点上需单节点 Ray head）
tests/nixl/scripts/run_e2e_cell.sh <node-ip> <cell> [hydra overrides]
#   扩散 nixl.peer_diffusion=True, FP8 +model.gen_quantization=fp8, fp32 model.train_dtype=float32
# 多节点同轮 A/B
python tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_5node_direct test.num_train=16 \
    +test.rounds=2 nixl.peer_diffusion=True
# FP8 数值
python tests/nixl/bench_fp8_logprob_gap.py --model <ckpt> --mode bf16 --out /tmp/bf16.pt
python tests/nixl/bench_fp8_logprob_gap.py --model <ckpt> --mode fp8 --out /tmp/fp8.pt --ref /tmp/bf16.pt
```

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

## 1b. 一对 agent 只占一条 bond，聚合靠并发对数

一个 NIXL agent 对的 READ 稳定在 **49 GB/s**，等于一条 400G bond，且**与 `UCX_MAX_RNDV_RAILS` 无关**
（1 / 2 / 4 / 8 四档实测都是 49.0 GB/s）。原因是 GPU 与 NIC 的亲和性：`nvidia-smi topo` 下每张 GPU 只有一个
`PIX` 的 NIC，UCX 选中它之后不会把同一个 rendezvous 拆到其余 bond 上。

于是节点聚合带宽是**并发对数**的函数，不是单流调优的函数：

```
节点聚合 ≈ min(Σ 每个源的 49 GB/s, Σ 每个读方的 49 GB/s)，且只有在「源与读方是一一匹配」时取到
```

实测佐证，Qwen3-30B-A3B（48 层），train EP8 单节点 → 8 个 rollout（TP4·EP8）在另一节点，稳态 0.41 s
搬 64.3 GB，即每节点 **156 GB/s**：

| 每源出口 | 该源的并发读方数 |
|---|---|
| 46.0 GB/s（最快） | 1 |
| 24.0 GB/s（中位） | 2 |
| 18.5 GB/s（最慢） | 2 以上 |

每个源搬的字节几乎相等（7.3–9.5 GB），但速率差 2.6 倍，看上去像「一条 bond 被几个读方分」。

**但按这条假设去改 planner 被实测否决了。** 在选源时优先挑「还没有别的读方在读」的源
（即把分配往一一匹配推），同一格从 0.413 s **退化到 0.654 s**（慢 1.6 倍），
每 client 上游速率下界从 19.6 降到 12.4 GB/s。原因是 EP 下每个读方要从哪个源拿哪些字节由分片决定，
能选的只有被复制的 dense 键；把这些键推到「无人读」的源，等于让读方多开几条对，
而不是消掉争用。**按字节均衡仍是目前最好的策略**，改动已回滚。

所以聚合停在探测值 386 GB/s 的 40% 这件事，不是 planner 选源能解决的：
剩下的差距要么来自每对连接的协议开销（单对上界 49 GB/s 已是实测事实），
要么要改变「谁持有什么」，也就是 train 侧的并行度本身。

235B 规模下每节点 107 GB/s 是同一现象的更极端形态：EP 让每个 gen GPU 的专家只有一个持有者，
读方无从分散，而一个源被多个读方同时读时就退化到 bond 的几分之一。

档位：两节点。复现见 §8 的 `bench_rails.sh` 与矩阵格。

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
| FP8 rollout，无扩散（bf16 train） | 75.5 GB | 5621 | 0.33–0.83 s |
| FP8 rollout，扩散分波（bf16 train） | 18.9 GB | 4201 | 0.121 s（上游 0.095 s，peer 0.023 s） |
| FP8 rollout，扩散分波（Megatron FP8 train） | 18.9 GB | 4201 | 0.113 s（上游 0.093 s，peer 0.016 s） |

FP8 两行每个 train rank 的发布刷新（重量化进发布副本）为 22 ms（bf16 train）与 32 ms（FP8 train），
发布副本 828 MiB / rank。这些数字取自融合 kernel 之前的实现；之后同样的副本刷新快 14.8 倍（§6）。

- 四个 instance 轮流当 designate，节点入口分散到每个 instance 的 NIC 上。
- 分波后 peer 阶段只剩最后一波的尾巴。
- 首次 sync 为 3.3 s，含 peer 读 prepped handle 的构建。
- 16 个 rollout 的 32 个 gen client 两轮均通过 layout 判据，fallback key 为 0。

档位：CPU（出口）、多节点（时间）。

## 4b. 规模验证：Qwen3-235B-A22B（94 层，128 experts，470 GB bf16）

4 节点：16 个 Megatron train rank（EP16，TP1，不包 DDP）→ 16 个 vLLM rollout（TP4，EP16 跨 DP4），
每节点 2 个 instance，扩散开，两轮换版本。**16/16 个 gen client 两轮各 1131/1131 key 通过 layout 判据，fallback 0。**

| | trainer 出口 | peer | 每 rollout descriptor 上限 | 首轮 / 稳态 |
|---|---|---|---|---|
| bf16，无扩散 | 519.4 GB | 0 | 2339 | 2.41 / 3.26 s |
| bf16，扩散分波 | 486.2 GB | 33.2 GB | 2353 | 4.05 / 2.39 s |
| **bf16 → FP8**，无扩散 | 265.5 GB | 0 | 4421 | 1.31 / 1.43 s |
| **bf16 → FP8**，扩散分波 | 245.7 GB | 19.8 GB | 4421 | 5.04 / 1.53 s |

**这个布局下扩散几乎没有可省的字节，原因是结构性的。** 专家权重占模型的 97%（423 GiB），
而 vLLM 的 EP16 让 16 个 gen GPU 各持**不同**的 1/16 专家，同节点两个 instance 之间没有重复的专家字节。
可去重的只有 dense 部分（attention 与 embedding，15 GiB），算术与实测吻合：
`519.4 − 486.2 = 33.2 GB` 恰好等于 peer 字节。
**去冗余的收益上界是「被多个 rollout 重复持有的字节」**，EP = DP × TP 时这只是 dense 部分。
rollout 用 TP 而非 EP 切专家时，每个 instance 持有全部专家的一个切片，重复度才回到 100%
（单节点矩阵的 MoE 格就是这一类），但那样每 rank 的显存需求按 TP 而非 EP 下降，
235B 下放不进 96 GB 卡。

**稳态 2.39 s 不是链路限制。** 每 gen 节点入口约 107 GB/s（8 client × 13–16 GB/s），
是实测节点聚合 386 GB/s 的 28%，每 train 节点出口约 102 GB/s。瓶颈在并发度而非带宽，
train 侧只有 2 个节点供给 2 个 gen 节点，且专家键的读取粒度是每 expert 一段。

**FP8 把字节减半，但曾把发布成本放大 36 倍（已修复，见 §6）。** bf16 下每个 train rank 的发布副本 6016 MiB
（只有 attention，因为 gen TP4 把它切得比 train 的 TP1 细），刷新 14 ms。
FP8 下副本涨到 19933 MiB（专家权重也要量化成 FP8 副本），刷新 513–545 ms。
后者与一次稳态 sync（1.5 s）同量级，**所以 235B 规模下重量化是发布点的主要成本，不是传输**。
它发生在推送锁内、不在拉取路径上，但会计入 trainer 的步间开销。
`weight_transport.md` §3.4.2 说的「代价按 rank 计」在这里是每 rank 约 20 GB 与半秒。

启动耗时：meta server 100 s，16 train + 16 rollout 的 client init 1020 s。

档位：4 节点。模型只需 `config.json` 与 tokenizer，权重用 `+test.gen.load_format=dummy` 起，
传输会覆盖每一个字节。

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
| MoE Megatron TP2·EP2 → TP4（EP1，细化 64×64 scale 网格） | bf16 → FP8 | ✅ | 1.108 / 0.383 s | 31.3 GB |
| FSDP2 → TP2 | bf16 → FP8 | ✅ | 0.288 / 0.051 s | 18.9 GB |
| colocate FSDP2 → TP2，扩散 | bf16 → bf16 | ✅ | 4.360 / 0.061 s | 16.4 GB |
| colocate FSDP2 → TP2，扩散 | bf16 → FP8 | ✅ | 2.739 / 0.049 s | 9.4 GB |
| Megatron FP8 train TP2 → TP2 | FP8 → FP8 | ✅ | 0.287 / 0.035 s | 18.9 GB |
| Megatron FP8 train TP2·PP2 → TP4 | FP8 → FP8 | ✅ | 0.290 / 0.021 s | 9.4 GB |
| Megatron FP8 train TP4 → TP2，扩散 | FP8 → FP8 | ✅ | 1.007 / 0.040 s | 9.4 GB |
| colocate Megatron FP8 train TP2 → TP2，扩散 | FP8 → FP8 | ✅ | 2.260 / 0.036 s | 9.4 GB |
| MoE Megatron EP4 → TP2 × DP2（vLLM 无 EP） | bf16 | ✅ | 0.618 / 0.169 s | 64.2 GB |
| MoE Megatron EP4 → TP2 × DP2，EP4 跨 DP | bf16 | ✅ | 0.342 / 0.090 s | 64.2 GB |
| MoE FSDP2 → TP1 × DP4，EP4 跨 DP | bf16 | ✅ | 1.661 / 0.098 s | 70.3 GB |

- FP8 格的出口约为 bf16 的 0.58 倍：权重减半，加上 scale、norm 与 embedding 等不量化的 key。
- 扩散格的出口是一份模型（16.4 GB / FP8 9.4 GB），与节点上的 rollout 数无关。

- Megatron FP8 train 指 TE `fp8_param` 加 per-tensor current scaling（`train_fp8=tensorwise`）。
  trainer 从 FP8 字节反量化，再按 128 tile 重量化进发布副本，每 rank 发布刷新 64–118 ms（Qwen3-8B，TP2）。
  TE 的 blockwise recipe 需要 CUDA ≥ 12.9，本集群是 12.8，所以零拷贝的 FP8-native 路径没有测。
- 细化网格格的稳态 0.383 s 慢于其他 MoE FP8 格，是因为 EP1 下每个 expert 被 TP4 切开，
  每 rollout 37,491 个 descriptor。block scale 改走发布副本之前是 228,675 个、1.23 s。

## 5b. 同设备的源：拷贝而不是传输（colocate）

一个 NIXL agent 对在**同一张卡**上 READ 只有 **52 GB/s**，而同样字节的 `cudaMemcpy` 是 **399–447 GB/s**，
差 7.7 倍。原因是 UCX 走的是 rendezvous 协议而不是交出指针；想只开 IPC 也不行，
NIXL 的控制面需要一个 active message 传输，`UCX_TLS=cuda_ipc,cuda_copy,self` 直接被拒绝建后端。

所以同设备的源改走 CUDA IPC：train client 为**每个已注册 region** 导出一个句柄（region 本就装着许多 shard，
句柄数与模型规模无关），同卡的读方映射一次，之后按 descriptor 本来要用的地址做设备拷贝。
单节点 8 train + 8 rollout 共卡、扩散开、两轮、每轮每 client 判 399 个 key：

| 格 | 稳态（NIXL 同卡） | 稳态（IPC） | 走拷贝的字节 |
|---|---|---|---|
| Megatron bf16 → bf16 | 0.056 s | **0.044 s** | 16.38 GB（全部上游） |
| Megatron bf16 → FP8 | 0.034 s | **0.033 s** | 9.44 GB |
| FSDP2 bf16 → bf16 | 0.061 s | **0.057 s** | 2.05 GB，其余回落到 NIXL |

**收益按「同设备字节占比」缩放。** 这三格里 peer 读（跨卡、走 NVLink）占大头，所以总时间只动一两成；
FSDP2 一格只有 2.05 GB 走拷贝，因为它多数 region 不是 caching allocator 能导出的块，回落路径按设计生效。
235B 规模的 colocate 下同设备字节是每 rank 约 20 GB，52 GB/s 要 0.38 s，拷贝只要 0.05 s，差距才显出来。

跨进程导入自己导出的句柄会被 CUDA 拒绝，所以地址算术单独做纯函数测试（`locate`），
映射与拷贝由另一个进程验证（`test_device_ipc_gpu.py`）。

## 6. block FP8 的数值

- `block_fp8.quantize_blockwise` 与 vLLM 的 `per_block_cast_to_fp8` 逐位一致（CPU）。
- **发布点的融合量化 kernel**（`fp8_kernels.py`）与上面那条逐位一致，25 个形状与分布全覆盖，
  包括 strided 的输入与输出。速度 **3169 GB/s**，高于 bf16 拷贝的 880 GB/s，
  因为它每元素只搬 3 字节（读 2 写 1）而拷贝搬 4；原来的四趟实现是 133 GB/s。
  端到端实测：同样 3313 MiB 的发布副本，每 rank 的发布刷新从 **118 ms 降到 8.0 ms**（14.8 倍），
  Qwen3-8B TP2 单节点格。235B 下原为 513 ms，按同一比例落到 35 ms 量级，
  不再与一次稳态 sync 同量级。
- 真实 Qwen3-8B 权重经 trainer 的量化器发布、FP8 rollout 生成，与 bf16 rollout 对比
  （4 个 prompt，32 token，贪心）：贪心 token **100% 一致**，首 token logprob 差
  0.003–0.10。档位：单 GPU，`bench_fp8_logprob_gap.py`。

## 7. 环境限制（与权重传输无关）

- **FlashInfer autotune 在节点已有缓存时死锁。** vLLM 的 MoE autotune 由 rank 0 读缓存文件再广播。
  两次失败的日志里 rank 0 报 `Config cache hit`，另一个 rank 进入 `Tuning trtllm::fused_moe`，
  随后 TP all-reduce 等满 600 s 超时。卡点在 vLLM 初始化，在任何传输发生之前。
  `ray start` 前把 `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` 指向一个空目录，所有 rank 一起从头调优，该格即通过。
- **共享 conda 的 NCCL 2.28.9 让 vLLM 静默禁用 pynccl。** vLLM 0.30 绑定 `ncclCommSuspend`（NCCL ≥ 2.29.7），
  绑定失败时所有 group 的 pynccl 被禁用，TP all-reduce 退回 torch.distributed，MoE 在 DP > 1 时
  于 `all_gatherv` 断言失败。`scripts/nixl_env.sh` 把 `VLLM_NCCL_SO_PATH` 指向单独安装的 2.29.7，
  harness 在引擎启动前检查绑定。
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

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

**并发对数聚合带宽实测（2 节点，28.59.83.232 → 29.162.224.39，`bench_cross_node_bw.py --name`，`run_bw_initiators.sh`）：**

| 并发对数 | 各对带宽 | 聚合 | 理论上限 | 效率 |
|---|---|---|---|---|
| 1 | 49.0 GB/s | **49.0 GB/s** | 49 GB/s | 100% |
| 2 | 49.0 + 24.5 | **73.5 GB/s** | 98 GB/s | 75% |
| 4 | 49.0×2 + 20.6×2 | **139.1 GB/s** | 196 GB/s | 71% |
| 8 | 49.0×4 + ~21×4 | **278.1 GB/s** | 392 GB/s | 71% |

关键发现：**8 对并发中始终有 4 对达到 49 GB/s，另外 4 对只有约 19–25 GB/s**，且分布模式在每次运行中完全一致（GPU 0/3/4/5 满速，GPU 1/2/6/7 降速）。这不是并发争用问题，而是 UCX 对 GPU→NIC 亲和性的确定性映射：某些 GPU 的"最优"NIC 与其 NUMA 亲和性不对齐，导致 UCX 选中了带宽较低的路径。段大小完全无关（4 KB → 2 GB 全部 49.0 GB/s，已在 §1c 单独验证）。

对 W12 的含义：235B EP16 下 16 个 gen GPU 同时读 16 个 train GPU，若 GPU 分布与 NIC 亲和性不理想，实际聚合带宽可能只有 `8×49 + 8×21 = 560 GB/s`（两个 gen 节点），而非理论 `16×49 = 784 GB/s`，效率约 71%。再叠加 EP 串行化（每个 gen GPU 按 key 顺序读 36 个专家，链路并非持续满载），最终稳态 214 GB/s 是合理的。

所以聚合停在探测值 386 GB/s 的 40% 这件事，不是 planner 选源能解决的：
剩下的差距要么来自每对连接的协议开销（单对上界 49 GB/s 已是实测事实），
要么要改变「谁持有什么」，也就是 train 侧的并行度本身。

235B 规模下每节点 107 GB/s 是同一现象的更极端形态：EP 让每个 gen GPU 的专家只有一个持有者，
读方无从分散，而一个源被多个读方同时读时就退化到 bond 的几分之一。

档位：两节点。复现见 §8 的 `bench_rails.sh` 与矩阵格。

## 1c. W12：235B 稳态 107 GB/s 的根因（EP 拓扑约束，非协议开销）

**观测**：235B 稳态每 gen 节点入口约 107 GB/s，是节点聚合探测值 386 GB/s 的 28%。§1b 已排除两条候选（波数无差别，planner 一一匹配反而慢 1.6 倍）。本节记录第三条候选的排除，以及最终定性。

**第三条候选：提交开销（已排除）**。strided descriptor 下 `make_prepped_xfer` 约 0.001 ms/key（`bench_nixl_prepped_xfer.result.json`），235B 235B MoE EP→TP 格的 579 个 key × 0.001 ms = 0.58 ms，而稳态传输时间 158 ms（§4b）。比值 < 0.4%，不是瓶颈。

**根因：EP 拓扑让每对链路只有一个 gen GPU 在读**。Qwen3-235B-A22B 的实测配置是 2 train 节点 × 8 GPU + 2 gen 节点 × 8 GPU，vLLM gen 侧 EP16：每个 gen GPU 持有 1/16 的专家，且它的专家只在一个 train GPU 上有对应分片，所以每个 gen GPU 在读专家键时只与**一个** train GPU 之间有一对 NIXL agent 链路处于活跃状态。该对链路的上界是 49 GB/s（§1b 实测单对值）。

`16 gen GPU × 49 GB/s = 784 GB/s` 是 2 个 gen 节点的**理论上界**（每对都饱和时），而 2 个 gen 节点入口合计实测约 `2 × 107 = 214 GB/s`，效率 214/784 = 27%。

剩余的 27% 效率不是波数、planner 或提交开销能解决的，而是**单次传输字节数不足以让 IB 链路饱和**：每个专家 dense 矩阵约 `422 MiB / 128 experts / 16 EP = 206 KB`，在 49 GB/s 的链路上传输时间约 4 µs，而 NIXL RDMA READ 的往返开销（QP setup + DMA + completion）在这个量级下构成明显开销。dense（attention）键因 TP 分片被多个 gen GPU 复用，但它们只占总字节的 3%（§4b）。

**第四条候选排除：段大小在 NVLink 上无关（单节点实测）**。
H20 单节点 GPU0→GPU1（NVLink，2048 MB 总量），inflight=1 下从 4 KB 到 2 GB 整段全部 395–396 GB/s，524,288 个 4 KB 段与 1 个 2 GB 段吞吐完全相同。NVLink 是零拷贝 peer memory access，没有 RDMA QP setup / RTT 开销，段大小无关。

**第四条候选排除（IB 路径）：段大小在跨节点 RoCE 上同样完全无关（2 节点实测已完成）**。
28.59.83.232 GPU0（target）→ 29.162.224.39 GPU0（initiator），跨节点 RoCE，2048 MB 总量：

| 段大小 | inflight=1 | inflight=4 | inflight=8 |
|--------|-----------|-----------|-----------|
| 2 GB（1 desc） | **49.0 GB/s** | — | — |
| 262 MB（8 desc） | 49.0 | 49.0 | 49.0 |
| **206 KB（MoE 实际专家段，10,180 desc）** | **49.0** | **49.0** | **49.0** |
| 64 KB（32,768 desc） | 49.0 | 49.0 | 49.0 |
| 16 KB（131,072 desc） | 49.0 | 49.0 | 49.0 |
| 4 KB（524,288 desc） | 49.0 | 49.0 | 49.0 |

4 KB × 524,288 desc 与 1 个 2 GB 传输带宽完全相同，均为 49.0 GB/s，全部字节逐位校验正确（ok=true）。**RDMA READ 在此 fabric 上对段大小不敏感**，206 KB 的实际专家段大小没有任何效率损失，完全贴近单对链路上界。

**W12 根因彻底锁定：四条候选全部被实测排除，剩余差距是纯 EP 拓扑约束。**

| 候选 | 排除方式 |
|---|---|
| 波数（1/4/8） | 实测无差别（§1b） |
| planner 选源（一一匹配） | 反而慢 1.6 倍，已回滚（§1b） |
| 提交开销（descriptor 数） | 0.58 ms vs 158 ms，< 0.4%（本节） |
| 段大小（IB RTT overhead） | 4 KB → 2 GB 全部 49.0 GB/s，NVLink 与 IB 两路均测，无差别（本节） |

每个 gen GPU 的专家只在一个 train GPU 上有分片，整个 pull 过程中每对 NIC 链路只有 1 个 gen GPU 在读。链路本身已饱和（49 GB/s），但同时活跃的链路对数被 EP 拓扑固定为 1 per train GPU。16 对活跃链路 × 49 GB/s = 784 GB/s 理论上界，实测 214 GB/s，效率 27%。不是协议层或实现层问题。

**结论**：W12 的剩余差距是 EP 拓扑的结构性结果，与 PSRL 的 planner、传输实现、descriptor 批量化、段大小均无关。要从 27% 提升到 ≥ 60%，需要改变「谁读谁」的拓扑，而不是调协议参数：

| 候选 | 效果 | 代价 |
|---|---|---|
| 把同一 gen GPU 的所有专家读合并为单次大 READ（coalesce at MoE group level） | 每次 sync 的 RDMA 操作数从 ~579 降到几十，激活更多 train GPU 参与并发读 | 需要 vLLM gen 侧 EP 在内存中连续存放专家；`moe_grouping.py` 已有分组逻辑可复用 |
| 增大 vLLM EP 分组（EP16 → EP8），每 gen GPU 持有更大的专家块 | 每个 gen GPU 的专家来自更多 train GPU，激活更多链路对并发 | EP 越小则 gen GPU 显存需求越高，235B 下 EP8 已接近上限 |
| train 侧专家发布副本在内存中按 gen EP 分组连续排列 | 同上，允许单次 RDMA READ 覆盖整个 gen GPU 的专家份额 | 需改 `block_fp8.py` / `comm_plan.py` 的发布布局 |

这三个方向均需在真实训练配置下实测确认，段大小不是障碍。

档位：NVLink 与 IB 段大小 sweep 均已完成（单节点 + 2 节点）；改善路径需 2 节点含真实 MoE 权重配置验证。

---

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

## 5c. 缩容吊销：驱逐之后陈旧 descriptor 会被拒绝

设计（§3.4.5）担心的是：成员退出后远端仍持有可用的 descriptor，
retry 路径发起读会**读到已释放显存并成功**，与缺 `registration_epoch` 同构的静默损坏。
实测（H20，两 agent 一 region，先做一次成功的 READ，再在读方执行 `remove_remote_agent`）：

| 用法 | 结果 |
|---|---|
| 用驱逐前的 prepped dlist 新建请求 | `makeXferReq` 拒绝，`NIXL_ERR_NOT_FOUND` |
| 驱逐前已建好的 req 句柄 repost | `postXferReq` 拒绝，`NIXL_ERR_NOT_FOUND` |

机制是 NIXL 对 remote section 持弱引用并做 generation 校验
（`nixl_agent.cpp` 在 `makeXferReq` / `postXferReq` / `getXferStatus` 三处查
`remoteSection.expired()`）。**所以这条路径不需要我们再加一层 epoch**，
要保证的只是「每个持有者都确实执行过驱逐」。

**吊销不走 NIXL 的 `invalidateLocalMD`。** 读 `nixl_p2p_metadata_backend.cpp` 后，
它的全部语义是「给该 peer 发一条 `NIXLCOMM:INVL`，让它执行 `invalidateRemoteMD(我)`」——
干活的始终是持有者本地那次驱逐。它不是广播（广播形式要中心 MD 存储，PSRL 没有）、
需要对端监听端口（`NIXLClientInfo` 不发布）、且 fire-and-forget 无法确认。
而第四步释放显存的安全性恰恰取决于这个确认，所以真正的缺口是确认而非调用，
吊销改走 PSRL 已有的 notif 通道（`MEMBER_LEAVE` / `LEAVE_ACK`）。

此前两轮把这一项记为「NIXL 缺 `invalidateLocalMD` 绑定」是**探测名写错**：
真名是 `invalidate_local_metadata`，而 PSRL 的 `server.shutdown` 一直在调它。
那个「等 NIXL 补绑定就失败」的哨兵测试因此永远不会触发。

**`remove_remote_agent` 在本构建上从不报错**：对已驱逐过的 peer、
甚至对从未加载过的名字都正常返回。所以「这次驱逐是否忘掉了什么」在 NIXL 层无法回答，
接口不返回布尔值；重复吊销天然幂等，丢 ack 后的重试安全。

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

## 9. DCP 保存与 NIXL 进度线程

结论：**进度线程常开时，同步 DCP 保存与直供读不冲突；会损坏 NIXL 的是 fork 式异步保存。**
原先「`all_gather_object` 破坏 UCX endpoint」的归因没有复现。

合成负载（`tests/nixl/repro_dcp_ucx.py`）：单节点 8 rank，每个 rank 注册一块 256 MB 的显存、
开进度线程，后台线程循环读邻居 rank 并逐字节校验，主线程同时做被测动作。

| 被测动作 | 读次数 / rank | 错误 | 不一致 |
|---|---|---|---|
| 无（基线，60 s） | 约 4.4 万 | 0 | 0 |
| `all_gather_object`，64 MB 负载，40 s | 约 1.7 千 | 0 | 0 |
| 同步 DCP 保存，约 8 次，每次 512 MB / rank | 约 5.5 万 | 0 | 0 |
| 同上，每卡另占 70 GB（余约 25 GB） | 约 4.5 万 | 0 | 0 |
| 异步 DCP，fork（`mcore` 默认） | 约 2.4 万后失效 | 之后全部 `NIXL_ERR_REMOTE_DISCONNECT` | 0 |
| 同上加 `RDMAV_FORK_SAFE`、`IBV_FORK_SAFE` | 同上 | 同上 | 0 |
| 异步 DCP，mcore spawn 持久进程（`persistent=True`） | 约 8 万～17 万（96-99 GB/s） | 0 | 0 |
| 异步 DCP，nvrx 策略（`AsyncCallsQueue persistent=True`） | 约 7 万（113-119 GB/s） | 0 | 0 |

- fork 失败的形态：UCX 报 `local work queue catastrophic error`，此后读全部失败且不恢复。
- `all_gather_object` 一行的读次数低，是 64 MB 负载下 GIL 与 NCCL 争用，不是错误。
- 测试脚本的一个陷阱：读线程默认在 0 号卡上，`torch.cuda.synchronize()` 同步错了卡，
  `zero_()` 晚于网卡写入，表现为大面积「不一致」且字节全为 0。已修，规则是读线程先 `set_device`。

真实栈（`examples/grpo_trainer/qwen3_8b_megatron_direct.sh`，单节点，4 train + 4 rollout，
绕过 `use_dcp_save=True` 的断言并强制进度线程常开，每步保存，2 步）：训练与两次保存完成，
日志里没有 UCX / NIXL 传输错误。

**fork 式异步保存归 verl 上游所有，且在当前环境里走不通。** `actor.checkpoint.async_save`
（默认 `False`）与调用链都在 verl（上游提交 `9d772002`），PSRL 只在 `ray_trainer.py:2904`
读它、在逐 rank 保存里拒绝它，不创建任何 fork。fork 来自 Megatron 的 `TemporalAsyncCaller`
（`async_utils.py:277`，`mp.get_context('fork')`）。但 verl 的异步路径在 Megatron 0.19 下先于 fork
就失败：

- `megatron_checkpoint_manager.py:1144` 导入 `strategies.base.async_calls`，该名字在 Megatron 0.19 的
  `base.py` 里已不存在，实测 `ImportError`；
- verl 保存时不传 `async_strategy`，Megatron 0.19 的默认值是 `"nvrx"`，需要
  `nvidia-resiliency-ext`，本环境未安装。

所以今天打开 `async_save` 会在导入处报错，不会走到上表里 fork 损坏 NIXL 的那一步；
上表的 fork 结果是绕开 verl、直接调 Megatron 并指定 `async_strategy="mcore"` 得到的。

**装上 `nvidia-resiliency-ext` 0.7.0 后，nvrx 路径与 NIXL 并发不冲突。** 上表新增的 nvrx 行
（`repro_dcp_nvrx.py`）用 `AsyncCallsQueue(persistent=True)`，persistent worker 以 `spawn` 启动，
不继承父进程的 UCX context，30 s 内完成 1 次 checkpoint，reader 线程 113–119 GB/s、0 错误。
单独跑 nvrx（无 NIXL）也 PASS，所以之前的 SIGABRT 是测试脚本自身没有用
`if __name__ == "__main__":` 保护初始化代码导致 spawn 子进程递归 import 触发的，不是 nvrx 与 NIXL 的冲突。

**结论：冲突在 fork，不在版本或环境。** fork 继承 CUDA context / UCX endpoint / 注册显存，
是 OS 层面约束，不会因 Megatron 或 verl 升级而改变。spawn（mcore persistent 或 nvrx persistent）
都能与 NIXL 共存。真实栈要跑通 async save 需要：装 nvrx >= 0.6.0、修 verl 的接口适配、
确保 spawn worker 不会递归运行初始化代码。`PSRL_PerRankMegatronCheckpointManager` 暂不删。

没有覆盖的：多节点、更多步数、真实栈上的异步保存、训练端与读者跨节点的布局。

另一个发现：同一次真实栈运行里，**第二次拉取在 peer 读处抛 `NIXL_ERR_NOT_FOUND`**，
4 个 rollout 被摘除。关掉 DCP 且不保存的对照同样失败，所以与保存无关，见实现文档。

## 10. 写入原子性（v2）：提交校验与 peer 修复

机制见主文档 §3.4.2.2，落地状态见实现文档 E5。本节记录可复现的实测。

**稳态写门闩是 O(1)。** 没有在飞读者时，写者不枚举 holder、不发通知。
`python tests/nixl/bench_write_gate.py`（CPU，纯状态机 + 假 client info）：

| 实例数 | 在飞读者 | holder 查找 | 通知数 | 单条 payload | fanout |
|---|---|---|---|---|---|
| 1000 | 0 | 0 | 0 | 0 B | 2.1 µs |
| 1000 | 100 | 1000 | 100 | 213 B | 609 µs |
| 10000 | 0 | 0 | 0 | 0 B | **0.6 µs** |
| 10000 | 100 | 10000 | 100 | 213 B | 5.9 ms |
| 10000 | 1000 | 10000 | 1000 | 116 B | 6.8 ms |

- 查找次数恒为 `holders + readers`，与读者数无关；通知数恒为 readers。
- payload 只含一个节点的 holder（≤ 8 条），不随集群规模增长。
- 有读者时的一次写约 6 ms（10k 实例），相对训练步（约 50 s）是 0.01%；稳态为 0.6 µs。
- 写者对读者传输的等待是 **0**：只等一次控制面 ack 往返，上限 100 ms。

**正确性判据（CPU）。**
`tests/unit/test_write_state.py` 是一个穷举真值表：对 `(pull 起点, ack, repaired)`
的全部组合，只有「读取 epoch 未变」或「本 epoch ack 且全部 suspect key 从 certified
holder 修复」两种拉取能被提交；其余被拒绝。`tests/nixl/test_pull_preempt.py` 用真实
client/planner/engine 验证有 holder 时修复落地同一版本、无 holder 时
`PullPreempted(repaired=False)` 且不提交；`tests/nixl/test_pull_engine.py` 则在
`MemcpyAgent` 上建模了 NIXL 的释放语义（轮询已释放句柄抛 `BAD_STATUS`），钉住句柄
所有权。

**两节点生产 e2e。** 配置：Qwen3-8B，Megatron TP4，8 rollout + 8 validate，
`ps_mode=nixl_direct`，`await_write_gate=True`，10 个训练步
（`RAY_ADDRESS=<head>:6399 LOCAL_IP=<head> bash examples/grpo_trainer/qwen3_8b_megatron_direct.sh`）：

| 量 | 值 |
|---|---|
| 完成的训练步 / 验证轮 | **10/10**，step 5 与最终验证均完成 |
| 写门闩次数 | 10（每步一次） |
| 写者等待 | **0.03–0.09 ms**（稳态无在飞读者，O(1)） |
| 权重 pull 次数 | 26，0.91–1.70 s（中位 1.05 s） |
| PREEMPT / 修复 / ack | 0 / 0 / 0（pull 都在 optimizer 写之前完成） |
| 提交拒绝 / NIXL / 运行错误 | 0 / 0 / 0 |

- 稳态没有 pull 与 optimizer 写重叠，所以门闩路径只走了「零读者」分支；这本来就是
  设计目标（写者不等待）。抢占 / ack / holder 修复由 CPU 测试确定性覆盖
  （`test_pull_preempt.py`、`test_write_state.py` 真值表）。
- 最终 reward 为 0 是基座模型在数学题上的预期结果，不是传输或门闩的问题；传输链路的
  正确性判据是步进与验证全部完成、pull 无错误、无提交拒绝。

**e2e 过程中暴露并修复的两处非 v2 缺陷（都是生产路径才可达）：**

1. **重注册广播的 client 名映射错误。** `_publish_validate_client_infos` 与广播的
   目的地用 `WorkerKey(role, i, rank).to_nixl_client_name(...)` 计算名字，而实际
   NIXL client 名编码的是 **gateway replica id**；rollout/validate 交替编号时二者
   不同。validation wake 因此把一方的 info 发到另一方的名下，真正醒来的 validate
   客户端收不到新地址，pull 读到 `read_addrs=None`。修复为按
   `tag_to_base_worker_ids` + 每实例 rank 生成 client 名。
2. **选源未过滤没有可读地址的 peer。** peer 在握手时可能只拿到 conn-info-only
   元数据；participant 子集变化会重新选 designate，可能选中一个从未加载地址的
   peer。pull engine 现在在 `_assign`/`_pick_holder` 里过滤这类源，回退到可读
   peer 或 upstream。

3. **注册策略漏掉了「自己也是读目的地」。** W13 的 `_should_register_rdma` 只回答
   「有没有远程读者读我」，据此在纯同节点场景跳过注册；但 pull 侧的本地张量同时是
   自己 pull 的目的地，NIXL 的本地 dlist 必须已注册。sleep/wake 后的 validate 实例
   因此在上游读时以 `NIXL_INIT_AGENT` 报 `NIXL_ERR_NOT_FOUND`。修复：PULL_SIDE
   始终注册；真正的「所有读都是同设备 IPC」再作为后续细化。

## 11. 精度对照：direct vs PS（同一坐标判据）与 reward 全 0 的根因

**坐标判据两臂对照**（`tests/nixl/test_nixl_e2e.py --config-name nixl_e2e_1node`，
4 train + 4 gen，Megatron TP2 / gen TP2，两轮换版本）：

| 臂 | round 1 | round 2（新版本） | trainer egress |
|---|---|---|---|
| direct（`direct_supply=true`） | 4 clients × **399/399** | 4 clients × **399/399** | 32.76 GB |
| PS（`direct_supply=false`） | 4 clients × **399/399** | **FAIL**：`lm_head.weight` 311,159,884/311,164,928 个元素不一致 | 32.76 GB |

- **round 1 两臂逐元素一致**：direct 与 PS 搬运的是同一份字节，direct 的传输精度与
  PS 路径在该版本上等价。
- **round 2 只有 direct 通过**；PS 臂在 `lm_head` 上失败（几乎全部元素不同），说明该
  PS 配置的第二版本发布/拉取本身有缺陷，不能作为 round 2 的参考。direct 直接对
  ground-truth 坐标函数验证通过，因此**不依赖 PS 臂也成立**。

**reward 全 0 的根因（GSM8K 对照实验）**

初版结论（「8192 截断导致无答案」）只解释了离线 DAPO 探针为何判无答案，不是训练日志
全 0 的原因。换成 GSM8K 后 training/validation reward 仍然全 0，而同一段生成文本在
离线用 `gsm8k.compute_score` 打分是 1.0，说明 reward 函数根本没有被调用：

- 数据集行里没有 `reward_model_dicts`。agent loop 从 batch 取该字段时默认 `[]`，
  `_resolve_reward_manager([])` 返回空的 manager 列表，reward 等于对空列表求和 =
  **0.0**，且不报错（`grading_failed=0`），与「答案错误」不可区分。
- 修复：`PSRLRLHFDataset.__getitem__` 把数据集配置（`data.reward_model_dicts`，多数据
  集模式取各条目）附到每一行；行内已有该字段时以行内为准。
  `tests/unit/test_dataset_reward_spec.py` 覆盖三种情形。
- DAPO 在 8192 上限下确实还会因生成没有收敛到答案行而拿 0（离线探针），两个问题叠加。
  因此精度对照改用 GSM8K（非零方差）。

**PS vs direct 10 步对照（Qwen3-8B / GSM8K / 2 节点 × 8 GPU，除 `ps_mode` 外同配置）：**

| 指标 | PS baseline | direct | direct − PS（Welch 95% CI） |
|---|---|---|---|
| 完成步数 | 10/10 | 10/10 | — |
| 训练/NIXL 错误 | 0 | 0 | — |
| 训练 reward 均值 | 0.4047 | 0.4070 | +0.0023 [−0.149, +0.153] |
| entropy | 0.2134 | 0.2389 | +0.025 [−0.026, +0.077] |
| grad norm | 0.3363 | 0.2909 | −0.045 [−0.153, +0.062] |
| response length | 1606.5 | 1762.6 | +156 [−543, +855] |
| step 时间 | 72.4 s | 74.2 s | +1.8 [−53.7, +57.4] |
| 初始验证 reward | 0.3442 | 0.1698 | — |
| 最终验证 reward | 0.6611 | 0.6520 | 差 0.009（1319 样本，噪声 ≈0.019） |

- step 0 的验证值两臂不同（0.34 vs 0.17），但使用同一份初始权重的 step 1 rollout
  reward（0.727 vs 0.625，n=128）和最终验证都在噪声范围内；fp8 臂的 step 0 验证
  （0.196）也贴近 direct 臂，因此 PS 的 step-0 点更像一次性异常，对照以逐步指标和
  最终验证为准。
- direct 写门闩 10 次，最大等待 0.05 ms；PREEMPT / 修复 / 提交拒绝均为 0。
- 数据集修复后，DAPO 的 reward-0 问题也随之中断：任何需要 rule reward 的示例都必须
  携带 `reward_model_dicts`，否则 reward 静默为 0。

## 12. FP8 rollout + BF16 training（direct）

场景：trainer 保持 BF16，在发布点把权重按 128×128 block 量化为 FP8 副本（scale 随
字节一起传输），rollout 与 validate 引擎均以 `quantization=fp8` 运行。配置与
`qwen3_8b_megatron_direct.sh` 对齐，只多这一项。

**逐位精度（`tests/nixl/test_nixl_e2e.py +model.gen_quantization=fp8`）：**

- 4 个 gen client × 399 个 key × 2 轮：FP8 权重**逐位**、`_scale_inv` **逐值**与坐标
  判据一致（`check_fp8_exact`），0 失败。
- 单次 pull trainer egress **18.88 GB**（bf16 同场景 32.76 GB，−42%）。
- scale 池化改造后重跑同样通过。

**生产 10 步（Qwen3-8B / GSM8K，train 1×8 GPU TP4，rollout 1×8 GPU TP1）：**

| 指标 | bf16 direct | fp8 direct |
|---|---|---|
| 完成步数 / 错误 | 10/10 / 0 | 10/10 / 0 |
| 初始验证 reward | 0.1698 | 0.1964 |
| 最终验证 reward | 0.6520 | 0.6475 |
| 训练 reward 均值 | 0.4070 | 0.4516（差 +0.045，CI [−0.088, +0.177]） |
| 每步 pull（p50） | 1.017 s，16.38 GB | **0.693 s，9.44 GB** |
| 每步 publish | 0.01–0.04 s | 0.03–0.17 s |
| validation→trainer 切换 | 1.14–1.28 s | 9.86–10.85 s → **1.80 s**（修复后） |
| trainer→validation 切换 | 21.3–23.1 s | 26.4–27.7 s |
| step 时间 | 74.2 s | 100.4 s（CI 重叠） |

**过程中修复的两处问题：**

1. **validate 引擎与 rollouts 的量化配置必须一致。** validate 实例由
   `train_actor_rollout_ref.rollout` 构建，而发布格式由
   `gen_actor_rollout_ref.rollout.quantization` 决定。只给后者开 fp8 时，validate
   引擎以 bf16 注册，第一次拉取新版本即报
   `... is torch.float8_e4m3fn ... but torch.bfloat16 locally`，rollout 协调器把
   8 个 validate 实例标记为不可用。修复：示例脚本同时设置两处；`_resolve_publish_cast`
   在启动时断言两者一致（`tests/unit/test_publish_format_config.py`）。
2. **FP8 scale 张量的注册粒度。** `_derive_block_scales` 原先每个权重分配一个 scale
   tensor，每个 tensor 都是独立的 NIXL region：Qwen3-8B 每 rank 257 个 region，
   每次 trainer wake 注册耗时 **8.5–9.2 s**，主导了 validation→trainer 切换。
   改为按设备分配一块 float32 池、各 scale 作为连续 view 后，注册降为 **6 个 region /
   0.12–0.17 s**，切换时间回到 1.80 s（bf16 为 1.14–1.28 s）。

**剩余差异**：trainer→validation 切换仍多约 5 s。原因在 validate 侧：FP8 vLLM 的权重
和 scale 是各自独立的分配，每实例 439 个 region，唤醒时注册约 14.5 s；bf16 同路径也
需要数百个 region，因此净差约 5 s。这部分属于 vLLM 分配布局，不影响每步同步热路径。


### 12.1 FP8 training + FP8 rollout（direct）

配置：`qwen3_8b_megatron_direct_fp8gen.sh TRAIN_FP8=true`，布局与 §12 相同，训练侧额外打开 TE
`fp8=hybrid` / `fp8_recipe=tensorwise`；发布点仍是 bf16 shard → 128×128 block FP8
副本（与 FP8-gen / BF16-train 场景相同的 wire 格式，逐位判据在该场景已通过）。

**结论：10/10 步、0 错误；初始验证 0.1986，最终验证 0.6520**（与 BF16 训练臂相同）；
训练 reward 均值 0.392 vs 0.452（差 +0.059，CI [−0.178, +0.059]），entropy、grad norm、
loss、response length 全部落在噪声区间。性能：pull 9.44 GB / 0.40–0.64 s（bf16 direct
为 16.38 GB / 1.02 s），publish 0.02–0.24 s，`update_actor` 均值 21.3 s（BF16 训练
21.5 s），trainer 切换 1.25–2.12 s，validate 切换 26.3–27.4 s。

**`fp8_param` 必须关闭。** TE 的常驻 FP8 参数只有当模型在 `fp8_model_init` 内构建时
才会产生 `QuantizedTensor`；本引擎直接构建模型（`get_fp8_context(is_init=True)` 没有
调用者），所以该 flag 不会改变参数 dtype，却会改变 FP8 反向的 weight-grad 记账。
实测打开后：第 1 步指标正常（entropy 0.14、grad 0.54），第 2 步 entropy 跳到 7.4、
grad norm 到 600，随后 response length 全部撞到 8192、reward 归零。关闭后训练稳定。
`validate_config` 现在在启动期拒绝 `fp8_param=True`（`tests/unit/test_fp8_training_config.py`）。

## 13. 验证切换：去掉派生描述符、按读者广播

现象：validate 切换 26.4–27.7 s，trainer 切换 1.14–2.12 s。按阶段拆分（fp8gen，19:35 与
09:12 两次切换）：

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| vLLM 引擎唤醒（`wake up tags ['weights']`） | 0.17 s | 0.17 s |
| **validate 实例 NIXL 重注册** | **14.5–15.6 s（439 region）** | 13.2–15.6 s（未变） |
| **`UPDATE_INFO` 广播 + ack** | **6.4 s（8 条 info 发给 24 个 agent）** | ~0（8 条 info 发给 8 个读者） |
| validate pull | 0.8–1.3 s | 0.3–0.9 s |
| trainer offload / KV 唤醒 / resume | ~3–4 s | ~3 s |
| **合计** | **26.4–27.7 s** | **17.8–18.8 s（−32%）** |

根因（第一性原理）与对应修复：

1. **client info 携带的是派生数据。** strided shard 的远程 descriptor 在源端按行展开
   （单个 validate 实例 **1,188,864 个 descriptor**），而 direct-supply 的 pull 引擎在读者侧
   用 `read_addrs` + stride + `shard_meta_infos` 已经能构造同样的描述符（`pull_engine._spec`）。
   把展开结果序列化进 info 再发给所有成员，是纯冗余。修复：direct 模式不再序列化
   pairwise descriptor（`pairwise_descriptors=False`），读者侧派生；PS 的
   `client_read`/`client_write` 仍需要它们，保持原样。validate 的单实例 info 从展开的
   MB 级降到 **0.127 MB**。
2. **广播给了不消费它的人。** 只有读者会使用更新后的 info；validate 实例是 pull destination，
   没有读者时广播与 ack 是纯切换开销。修复：目的地由 comm plan 反查
   （`descriptor_readers_by_source`，含 peer-diffusion follower），并补齐读者所在实例的所有
   rank（ack 是实例级的）。本场景广播为 8 个读者 / 8 个 agent。
3. 注册成本按 **region 数** 计：validate 每个权重（FP8 再加 scale）是独立分配，439 region；
   trainer 是 Megatron flat buffer + 合并后的 scale 池，6 region / 0.2 s。这是剩余 18 s 的
   唯一主项，见 §13.1。

验证：
- 坐标判据加 `+model.pairwise_descriptors=false`（读者侧派生描述符）：4 gen client ×
  399 key × 2 轮 **逐位通过，0 失败**。
- 生产 10 步（fp8gen，GSM8K）：0 错误；初始验证 0.1941、最终 0.6535（优化前 0.1964 /
  0.6520）；训练 reward 0.4437 vs 0.4516（CI 重叠），entropy/grad/loss/长度全部在噪声内。
- 复杂度：info 载荷 `O(#shard × 常量)`（原 `O(展开行数)`）；广播 `O(readers)`（原 `O(fleet)`）。
  万卡 + TB MoE 下，meta server 的序列化从 GB 级降到 KB/MB 级，fan-out 与 ack 只与 plan 中的
  读者数成正比。

### 13.1 segment 粒度注册（已落地）

instrumentation（`_allocator_segment_report`）：validate 实例的 439 个 allocation 落在 **151 个
CUDA allocator segment** 中；trainer 的 6 个 allocation 落在 5 个 segment。注册耗时与 region 数
近似线性（~30–35 ms/region，多次运行一致）。

**实现**：`psrl.nixl.segment_registration`（默认 `auto`）。注册前对 caching allocator 做一次
`memory_snapshot()`，把每个张量 allocation 映射到其所属 segment；同一 segment 只登记一个
NIXL region。`read_bases` 与 IPC handle 仍按 allocation 保留（一个 segment 可能含多个
allocation，各自有 IPC handle），因此数据路径不变、显存占用不变。expandable segments
（`vLLM` 在 sleep 关闭时启用）的 VA 按需映射，不能整段注册；vLLM server 知道这个设置，
启动时把解析结果传给 client，snapshot 里出现 expandable 段或无法定位的 allocation
（CuMem pool、注册后才分配的发布副本）逐 allocation 回退。

**实测（Qwen3-8B，2 节点）：**

| 场景 | 优化前 | 优化后 |
|---|---|---|
| validate 实例注册 | 439 region / 13.2–15.6 s | **151 region / 5.06 s** |
| trainer 注册 | 6 region / 0.17–0.22 s | 5 region / 0.17 s |
| validate 切换（含注册、pull、offload） | 17.8–18.8 s | **7.57–8.87 s** |
| 10 步运行 | 0 错误 | 0 错误（初始 0.1948，最终 0.6141） |
| 坐标判据（433 allocation 的小模型） | 435 region / ~13 s | **113 region / 1.06 s，399×4×2 两轮逐位通过** |

注册时间从 13–15 s 降到 5 s，与“151 × 34 ms”的线性模型一致；切换从 26.4–27.7 s 累计降到
7.6–8.9 s（两轮优化合计约 −71%）。风险与回退路径：expandable segments 自动关闭该模式；
snapshot 无法覆盖的 allocation 逐 allocation 注册。

### 13.2 广播 fan-out 的大规模评估与并行化

**评估。** 广播目的地只包含读者，但读者数随 fleet 增长：trainer 唤醒时是所有读它的 rollout
实例（有 peer diffusion 时为每节点 designate，仍随节点数增长）；validate 唤醒时是同节点
peer follower。单实例 info 在 direct 模式下已是元数据级（0.12–0.23 MB），但一个 agent 串行
发 O(readers) 次：按实测每通知 ~10–40 ms 计，万卡规模（10^3–10^4 读者）需要数十秒到分钟，
是切换路径的新瓶颈。**结论：有必要并行化。**

**实现。** meta server 不再自己发送，而是生成可序列化的发送计划：每个更新源序列化一次，
每个目的地只带它读取的源子集与连接元数据（`plan_update_broadcast`）。trainer 的每个 rank
用**自己的 NIXL agent** 发送 `destinations[rank::world_size]` 这一片（`nixl_send_update_infos`），
首次发送前用 conn-info 把目的地 agent 加入本地元数据。读者仍只收到一个通知，ack 语义不变；
发送量从 O(readers) 降为 O(readers / n_train_ranks)，并可随训练并行度扩展。发送计数与计划
不一致时 trainer 直接报错，而不是让读者等到握手超时。

**实测与两个修正：**

- 8/16 个目的地的 fan-out 在 **1.68–2.38 s** 内完成（含目的地刷新），validate 切换 **8.03 s**；
- **重复切换最初丢失通知**：train rank 缓存了目的地的 agent 元数据，而目的地在上一轮 wake
  中重新注册后缓存条目不再可用，通知被静默丢弃，读者一直等到握手预算（实测 20 分钟后
  超时）。修正：**每次 fan-out 前 `remove_remote_agent` + `add_remote_agent` 刷新目的地**；
  修正后同一 6 步序列（3 次切换、含两次 trainer 唤醒）全部通过，无回退；
- **ack 有界**（`fanout_ack_timeout_ms`，默认 10 s）：若通知仍未到达，meta server 用它
  已有的完整元数据重发同一更新（保证路径），读者不会因一条丢失的通知挂起到握手超时。
  修正后的序列中该回退一次都没有触发。


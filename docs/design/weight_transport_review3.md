# 权重传输方案自审（第三轮）

> 对 [weight_transport](weight_transport) 的第三轮审核，覆盖实现可行性、代价与风险、
> 底层通信栈（CUDA kernel / NCCL / RDMA / PCIe / NIXL）、可扩展性、完整性与瓶颈，
> 以及**弹性扩缩容与 `is_rollout_mode_in_actor=True` 启动场景**六项。
>
> **结论先行：前五项均无新增阻塞，但第六项发现一个真实的正确性缺口 ——
> 当 `is_rollout_mode_in_actor=True` 时 trainer 以 `meta` 模式注册（只有元数据、
> 没有真实显存），此时「trainer 直供」没有可读的源，当前设计在这个场景下不成立。**
> §6 给出解法。

---

## 1. 实现可行性（第三轮）

前两轮的分档结论不变。本轮复核新增的四个子系统（阶段标记 W、chunk 偏移、
DP 副本接管、chunk/commit group 分离），结论：

| 新增项 | 依赖的既有机制 | 可行性 |
|---|---|---|
| W 跟随阶段 | `STAGE_META`（`strategies/base.py:22`）已有阶段分类 | 高，只是加一条单向上报 |
| chunk 偏移错开 | 无，纯本地计算 | 高，零依赖 |
| DP 副本接管 | `get_sharding_for_param` 不含 DP 维度（已核实） | 高，副本本来就存在 |
| chunk / commit group 分离 | 需重构 plan、位图、计数器三者的索引单位 | 中，是概念重构 |

**最后一项是本轮唯一需要上调工作量的**：它不是加字段，而是把一个索引单位拆成两个，
会牵动 planner、index、执行层三处的数据结构。但它是 MoE 分组的前置
（§3.6.1 已说明），所以必须做，不是可选。

---

## 2. 代价与潜在风险（第三轮）

### 2.1 本轮新增的风险

**(a) 阶段标记引入了 trainer → index 的新方向依赖。** 之前 trainer 只在版本发布时
联系 index，现在每 mini-batch 两条。虽然设计上是 fire-and-forget，
但**它让 index 持有了会过期的状态**：若 trainer 崩在 `bwd` 阶段，index 会永远认为
W 应该很小，直到 holder 被摘除。应当给阶段标记加 TTL，超时回退到 `W_base`。
这条设计里没写。

**(b) chunk 偏移与 W 的交互未验证。** 偏移让 W 个读者命中不同地址区间，
这对 trainer 出口是好事（减少同一块内存争用），但**对 rollout 侧的 wire buffer 复用是坏事**：
每个 instance 读的 chunk 序列不同，意味着 trainer 侧的 `2×chunk` 流式缓冲
要同时服务**不同进度**的读者，可能需要比 2 更深的缓冲。
这是一个真实的张力，设计里只写了「偏移是零成本的」，**不准确**。

**(c) DP 副本接管假设了 DP 组内权重严格一致。** 这在数学上成立（DP 就是数据并行，
参数同步），但**在实现上依赖 grad all-reduce 已完成**。若在 all-reduce 完成前就
从另一个 DP rank 读，会读到未同步的参数。所以接管必须限制在「版本已发布之后」,
而版本发布本来就在 optimizer step 之后，所以**实际安全，但这条推理链必须写明**，
否则后人改动发布时机就会破坏它。

---

## 3. 底层通信栈分析

### 3.1 CUDA kernel 层

**(a) 量化 kernel 与推理 kernel 的 SM 争用（colocate 场景）。**
§3.7 的 colocate 原地交接下，train 与 rollout 共享 SM。量化 kernel
（`per_block_cast_to_fp8` 或 TE 的 MXFP8 quantizer）是**显存带宽受限**的，
它会和 rollout 的 decode kernel 争带宽。设计里说「量化与 RDMA 重叠」,
但没说**量化与推理重叠时的优先级**。

建议：colocate 下量化 kernel 应当在**低优先级 stream** 上提交
（`torch.cuda.Stream(priority=-1)`），让 decode 优先。这是一行改动，收益明确。

**(b) 落地格式化 kernel 的 launch 开销在 MoE 下不可忽略。**
`process_fp8_weight_block_strategy` / `requant_weight_ue8m0_inplace` 是 per-tensor 调用。
MoE 分组后 key 数降到数百，所以**分组顺带解决了这个问题**,
但若分组未做，14 万次 kernel launch 本身就是秒级开销。这强化了分组的必要性。

**(c) FP4 的 swizzle 是 permute + contiguous，会产生一次全量拷贝。**
`swizzle_blockscale`（`nvfp4_utils.py:13`）的 `permute(...).contiguous()`
对 scale 张量做一次完整重排。scale 体积是权重的 1/16（block 16）或更小，
所以绝对量不大，但**它是不可避免的额外显存与带宽**，FP4 的收益账里应当扣掉这部分。

### 3.2 NCCL 层

**(a) 本设计不引入任何 NCCL 通信，这是相对 broadcast 方案的结构性优势。**
但它与 trainer 自身的 NCCL 有**资源竞争**，而这一点设计里只在 W 的讨论中提到：

- `grad reduce-scatter` / `param all-gather`（DistributedOptimizer）走 NCCL，
  跨节点时占 NIC。
- NIXL 的 RDMA 读也占同一张 NIC。
- **两者的流控是独立的**：NCCL 有自己的 channel 与 buffer，NIXL 走 UCX，
  互相看不见对方，所以会简单地争抢带宽。

§3.5.2.2 的阶段标记是对的方向，但**粒度偏粗**：它按 fwd/bwd/opt 分档，
而 `overlap_grad_reduce` 开启时 grad reduce 是**与 backward 交织**的，
不存在「干净的 bwd 阶段」。所以阶段因子应当按「是否有 NCCL 在飞」而非「训练阶段」判定，
更准确的信号是 Megatron 的 `grad_sync` / `param_sync` 回调。

**(b) 一个未被利用的机会：NCCL 与 NIXL 可以共享 NIC 但不能共享流控，
那就让它们在时间上互斥。** param all-gather 完成的那一刻正是权重就位的时刻,
也是 NIC 刚空闲的时刻。**把版本发布精确挂在 param all-gather 完成的回调上**,
而不是「optimizer step 返回后」,可以让 rollout 的读取精确填进 NCCL 的空档。
这比阶段标记更精确，且成本更低（一个回调而非持续上报）。

### 3.3 RDMA / NIC 层

**(a) 单边读的 QP 与 completion 开销随并发读者线性增长。**
NIXL 的读是单边的，trainer 侧不消耗 CPU，但**NIC 侧仍要为每个 QP 维护状态**。
W 个并发读者 × trainer rank 数 = QP 数量。万卡下若 W=8、trainer 2000 rank，
是 1.6 万个 QP，**可能超过 NIC 的 QP 缓存**，导致 cache miss 与性能悬崖。

设计里没有考虑 QP 数量上限。应当把它作为 W 的第四条约束
（§3.5.2.1 现有三条：NIC 不超卖、扩散收益、轮数），即
`W × trainer_ranks_per_reader ≤ QP_budget`。

**(b) 跨节点 peer 读会放大 incast。** 阶段二的跨节点 peer 下，
多个 rollout 可能同时读同一个 peer，形成 incast 到该 peer 的 NIC。
设计里有 `P`（每 peer 最多服务读者数）这个约束，**方向对，但 P 的取值没有依据**。
它应当由「peer 自己的推理对延迟的容忍度」决定，而不是一个经验值。

**(c) 一个明确的优化机会：GPUDirect RDMA 路径确认。**
rollout 直读 trainer GPU 显存，若走 GPUDirect RDMA 则零 CPU 拷贝。
但这要求两侧内存都已注册且 NIC 与 GPU 在同一 PCIe root complex 下。
PSRL 的 `NetworkTopology` 只区分 `IB` 与 `IB_PCIE`（`network_topology.py:42`），
**没有区分「GPUDirect 可用」与「需经 host bounce」**。
后者的带宽可能差一个量级。这应当在握手期用 `query_memory` 探测并记入拓扑。

### 3.4 PCIe 层

**(a) 本设计大幅减少 PCIe 流量，这是未被充分强调的收益。**
今天的链路 train GPU → CPU PS → rollout GPU 要跨 PCIe 两次（D2H + H2D），
每次 `N×M`。直供后 PCIe 流量**归零**（GPU 到 GPU，走 NVLink 或 NIC）。
1TB 模型 × 32 节点下这是数十 TB 的 PCIe 流量被消除。
设计的收益表里只记了「train 出口 `N×M → M`」,**没有记 PCIe 这一项**。

**(b) optimizer offload 仍走 PCIe，且与权重读取在时间上重叠。**
`offload_megatron_copy_params` 把 master shard 搬到 CPU，走 PCIe。
它与 NIXL 的 GPU 读**不争同一资源**（一个 PCIe、一个 NIC），
所以 §3.5.2.2 把「optimizer 数值计算」阶段的 W 放大是对的。
但若 NIC 与 GPU 共享 PCIe switch（常见于 PCIe-attached NIC），
两者**会**争 PCIe 带宽。这取决于硬件拓扑，应当作为一条可配置的保守假设。

### 3.5 NIXL 层

| 能力 | 状态 | 本轮新发现 |
|---|---|---|
| `prep_xfer_dlist` / `make_prepped_xfer` | 已规划 | 应确认 prepped handle 在 chunk 偏移下仍可复用（偏移只改索引顺序，不改 dlist，**可复用**） |
| Nx5 strided | 已规划 | MoE 分组后 expert 维度正是规则跨步，收益最大化 |
| `estimate_xfer_cost` | 已规划用于 W 标定 | 它是 `ANALYTICAL_BACKEND`，不含实时拥塞，只能做初值 |
| `query_memory` | 规划用于注册校验 | **还可用于 GPUDirect 可用性探测**（§3.3c） |
| notif 通道 | 已识别为误用 | §3.5.5 已规划指针表 |
| `capture_telemetry` | 未用 | 可作 W 的反馈信号源，省去自建埋点 |
| `invalidateRemoteMD` | **未用，但弹性场景需要** | 见 §6 |
| `num_threads`（多线程后端） | 未用（`nixl_agent_config` 默认 0） | MoE 分组后单 chunk 变大，多线程可提升单传输带宽 |

最后一行值得注意：`nixl_agent_config` 的 `num_threads` 默认 0，
而 UCX 插件支持多线程（本仓库 NIXL 的 commit 说明提到
`split_batch_size=4096 in threadpool`）。chunk 变大后，单个传输的并行度成为瓶颈，
应当实测 `num_threads > 1` 的收益。

---

## 4. 可扩展性（第三轮）

前两轮结论不变（MoE 分组是前置、FP4 需显式处理 swizzle、万卡需元数据去扇出）。
本轮新增两点：

**(a) QP 数量是万卡下的新上限**（§3.3a），应纳入 W 的约束。

**(b) 阶段标记的消息量在万卡下需要复核。** 每 mini-batch 两条 × trainer rank 数。
若每个 rank 都上报，2000 rank × 16 mini-batch × 2 = 6.4 万条／版本，
**超过了 §3.2 承诺的 `O(instances)` 量级**。
解法：**只由 DP rank 0 上报**（阶段在 DP 组内是同步的），
消息量降到 `O(minibatches)`。设计里没说明由谁上报，这是个需要补的细节。

---

## 5. 完整性与单点瓶颈（第三轮）

### 5.1 瓶颈清单更新

| 候选单点 | 严重度 | 状态 |
|---|---|---|
| NIXL 元数据扇出 | 致命 | 已规划 §3.5.5 |
| **`is_rollout_mode_in_actor=True` 下无权重源** | **致命（正确性）** | **本轮新发现，§6** |
| QP 数量上限 | 高（万卡） | **本轮新发现，未规划** |
| 轮数 × 控制面 RTT | 高 | 已规划 §3.5.6 |
| PS 单进程控制面 | 高 | 已规划 §3.2 |
| 阶段标记消息量 | 中（万卡） | **本轮新发现，解法明确（DP rank 0 上报）** |
| trainer 出口 | 高 | 已规划 W + peer |
| GPUDirect 不可用导致的 host bounce | 中 | **本轮新发现，未规划** |

### 5.2 仍缺的性能优化空间

**(a) 版本发布的时机可以更精确**（§3.2b）：挂在 param all-gather 完成回调上，
而非 optimizer step 返回后。既更早（减少 staleness），又填进 NCCL 空档。

**(b) 没有利用 chunk 之间的依赖差异。** embedding / lm_head 在 decode 的第一步就要用，
而深层 MoE expert 在 prefill 后期才用。**按「rollout 的实际使用顺序」排 chunk 优先级**,
可以让 rollout 在部分 chunk 到达后就开始生成（渐进可用）。
这与 commit group 的原子性不冲突（group 内原子，group 间可有序）。
这是一个未被考虑的方向，收益在大模型长传输时显著。

**(c) 量化结果可跨版本复用的部分未利用。** 若某些层的权重在一步内未变化
（学习率为 0、冻结层、或 MoE 中未被路由到的 expert 梯度为 0），
量化结果与上一版本相同，**可跳过量化与传输**。这是 verl `delta_sharded` 的思路，
但本设计把 delta 整体列为「跨 DC 才用」。**冻结层与未激活 expert 是免费的 delta**,
判定成本接近零（检查 grad 是否为 None 或全零）。

---

## 6. 弹性扩缩容与 `is_rollout_mode_in_actor=True`：当前设计不完整

**这是本轮最重要的发现。**

### 6.1 `is_rollout_mode_in_actor=True` 下 trainer 没有可读的权重

核实启动路径（`ray_trainer.py:2183`）：

```python
actor_protocol_mode = "meta" if self.is_rollout_mode_in_actor else "full"
...
futures.extend(self.actor_wg.execute_all_async("nixl_protocol", actor_protocol_mode))
```

`meta` 模式下 `register_local_tensors(..., meta_only=True)` **只注册元数据、
跳过真实张量注册**（`client.py:503/580/602` 的 `if meta_only` 分支）。
配合 `init_model("empty")` 与随后的 `nixl_sleep("meta")`，
此时 trainer **既没有权重数据，也没有可被 RDMA 读的注册内存**。

而当前设计的核心前提是「权重的唯一副本在 trainer GPU 上，rollout 直读它」。
**在这个场景下前提不成立**：

- 启动时全部 GPU 跑 rollout，trainer 是睡着的、meta 注册的。
- rollout 需要初始权重，但**没有 holder**。
- 今天这件事由 PS worker 承担（`preload_checkpoint_to_cpu` 读盘 →
  `broadcast_init` 扩散 → `initial_pull_from_ps` 灌给 rollout），
  而本设计删掉了 PS worker。

§3.3 末段的 bootstrap 方案说「trainer rank 0 读 checkpoint，按 shard 代数分发」,
**但在 `is_rollout_mode_in_actor=True` 下 trainer 根本没有显存可以装那份 checkpoint**
（它的显存已经让给 rollout 了）。这是一个**设计漏洞，不是实现细节**。

### 6.2 解法：bootstrap 必须有一条不依赖 trainer 显存的路径

三个候选，建议第三个：

**候选 A：rollout 自己读 checkpoint。** 把 `load_format="dummy"` 改回真实加载。
简单，但把 `broadcast_init` 当初要解决的「N 个 reader 打爆文件系统」搬回来，
且 1TB 模型下每个 rollout 节点都读一遍盘不可接受。

**候选 B：临时唤醒 trainer 做一次分发。** 启动时先让 trainer 短暂持有权重、
分发给 rollout、再睡下。正确但**与「全部 GPU 跑 rollout」的目标直接冲突**:
那一刻需要同时容纳 trainer 权重与 rollout，显存可能不够。

**候选 C（建议）：rollout rank 0 读盘，经 peer 扩散给其余 rollout。**
这完全复用本设计已有的两个机制：

```
启动（is_rollout_mode_in_actor=True）：
  rollout rank 0（或每节点一个代表）从 checkpoint 读自己需要的 shard
    -> 注册为 holder，版本号 = v0
  其余 rollout 走 §3.5.2 的同版本 peer 扩散拿到 v0
    -> 复用 broadcast.py 的树形拓扑，只是源头从 PS rank 0 换成 rollout rank 0
  trainer 后续被唤醒时（switch_to_trainer_mode）：
    从 rollout 侧 pull 一份作为训练起点（这条路径今天就存在:
    ray_trainer.py:2578 的 actor_wg.execute_all_async("pull_model")）
```

**为什么这是对的**：它让「谁是 holder」完全由运行时状态决定，
而不是假定 trainer 永远是源。这正是 §3.1 holder 抽象的初衷,
只是初稿把 `TRAIN_GPU` 当成了唯一的权威源。
**修正后 holder 种类应当是对称的**：`TRAIN_GPU` 与 `ROLLOUT_GPU` 都可以是权威源，
取决于谁当前持有最新版本。

### 6.3 弹性扩缩容：三个具体问题

**(a) NIXL server 明确不支持运行时注册变更。**
`server.py:56` 与 `:90` 两处都有
`# TODO(lhy): Support elastic clients after initial registration.`,
且 `_is_all_client_infos_recved` 一旦为 True 就直接 return。
**所以弹性增减 rollout instance 时，新 instance 无法完成 NIXL 握手。**

今天这件事被 `_broadcast_updated_client_infos_from_ps_manager` 部分绕过
（colocate 切换时用它更新 client info），但那是**固定集合内的 wake/sleep**,
不是集合变更。**真正的弹性需要 NIXL 层支持增量注册**,
对应 NIXL 的 `invalidateRemoteMD` / `loadRemoteMD` 这组 API（本仓库
`nixl.h:528` 起），当前完全未用。

**这是弹性场景的前置阻塞项**，与元数据扇出同级。

**(b) 弹性缩容时 holder 摘除与在飞读者的竞态。**
一个 rollout instance 被缩容（sleep）时，它可能正作为 peer 为他人提供读取。
§3.5.7 的 holder 摘除是**被动触发**（靠 RDMA 失败），
对主动缩容来说太晚，读者会先失败一次。
应当让缩容走与 §3.4.5 相同的协议：**先撤销 holder 资格、等在飞读者归零、再 sleep**。
设计里只对 trainer sleep 写了这条，**对 rollout 缩容没写，应当对称补上**。

**(c) 扩容进来的新 instance 的版本状态未定义。**
新 instance 没有任何版本的权重。它应当：从 index 查当前版本 → 作为「排队者」
走 §3.5.2 的解析（此时必然有 peer，因为其他 instance 已持有）→ 读完后注册为 holder。
**这条路径逻辑上成立且复用已有机制**，但设计里没有显式写出「新 instance 的冷启动」,
应当补一节。

### 6.4 一个更深的结论

§6.1 和 §6.3 指向同一件事：**初稿把「trainer 是权重源」当成了不变量，
而 PSRL 的部署形态（colocate 切换、全 rollout 启动、弹性伸缩）
决定了「谁是源」本身就是动态的。**

正确的表述应当是：

> **权重的权威副本在「当前持有最新版本且处于唤醒状态的那组 GPU」上，
> 它可能是 trainer，也可能是 rollout。index 的职责就是回答这个问题。**

这个修正不改变数据面（仍是 RDMA 直读、零额外副本），
但它让 holder 表从「trainer 注册 + rollout 追加」变成**对称的注册**,
并要求 bootstrap、colocate 切换、弹性伸缩三条路径共用同一套 holder 语义。

---

## 7. 修订建议汇总

按优先级：

1. **holder 语义对称化**（§6.4）：`TRAIN_GPU` 与 `ROLLOUT_GPU` 都可作权威源。
   这是 §6.1 正确性缺口的根治，也让三条部署路径统一。
2. **bootstrap 走候选 C**（§6.2）：rollout rank 0 读盘 + peer 扩散，
   复用 `broadcast.py` 树形拓扑，不依赖 trainer 显存。
3. **NIXL 增量注册**（§6.3a）：弹性场景的前置阻塞，需要用
   `invalidateRemoteMD` / `loadRemoteMD`。与元数据扇出同级优先。
4. **rollout 缩容的 holder 撤销协议**（§6.3b）：与 trainer sleep 对称。
5. **QP 数量作为 W 的第四条约束**（§3.3a）。
6. **阶段标记只由 DP rank 0 上报，且加 TTL**（§4b、§2.1a）。
7. **版本发布挂在 param all-gather 完成回调上**（§3.2b），更早且填 NCCL 空档。
8. **GPUDirect 可用性探测记入拓扑**（§3.3c）。
9. **colocate 下量化 kernel 走低优先级 stream**（§3.1a）。
10. **冻结层与未激活 expert 免费跳过**（§5.2c）。
11. **按 rollout 使用顺序排 chunk 优先级**，支持渐进可用（§5.2b）。
12. **PCIe 流量归零应计入收益表**（§3.4a）。

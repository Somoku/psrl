# 权重传输方案自审（第四轮）

> 对 [weight_transport](weight_transport) 的第四轮审核。本轮重点是**验证前三轮
> 发现的问题是否真的被方案解决了**，而不是寻找新问题。
>
> **结论：第三轮发现的正确性缺口（`is_rollout_mode_in_actor=True`）已修复，
> 通信栈的五条限制中三条已落实、两条只记录未解决。
> 「渐进可用」经查证 vLLM 源码确认不可行，已从方案移除而非保留为可选项。
> 本轮新发现一处精度配置的覆盖盲区（§4.3）。**

---

## 1. 实现可行性

### 1.1 本轮复核的结论

方案现在有 **10 个落地项 + 1 个待优化清单**。按「是否依赖未实现的外部能力」分类，
这是可行性的真正分界线：

| 类别 | 项 | 判断 |
|---|---|---|
| **纯 PSRL 内部改动** | prepped descriptor、strided、chunk/commit group 分离、MoE 分组、FP8/FP4 wire、写屏障、chunk 偏移、DP 副本接管、holder 对称化、bootstrap | 可行，工作量明确 |
| **依赖 vLLM 已有能力** | colocate 用 `discard(("kv_cache",))`、FP4 的 `swizzle_blockscale`、`fp8_utils` | 可行，已查证 API 存在 |
| **依赖 NIXL 已有但未用的能力** | `prep_xfer_dlist`、Nx5 strided、`query_memory`、`invalidateRemoteMD`/`loadRemoteMD` | 可行，API 存在但 PSRL 从未调用，**风险在首次使用的未知坑** |
| **依赖 Megatron 回调** | 版本发布挂 param all-gather 完成 | **需确认 Megatron-Bridge 是否暴露该回调**，未查证 |

**最后一行是本轮唯一新增的可行性不确定项。** §3.5.2.2 建议把版本发布挂在
param all-gather 完成的回调上，但没有查证 Megatron-Bridge 是否提供这个 hook。
若不提供，退化方案是「optimizer step 返回后立即发布」（即今天的时机），
收益损失有限，不构成阻塞。**但设计里把它写成了建议而没标注待确认，应当标注。**

### 1.2 工作量重估

按前三轮的分档，总量大致是：

```
低风险接线（1-5）            约占 20%
MoE 分组 + chunk/group 分离   约占 25%   <- 最大单项，且是前置
三阶段 planner                约占 25%   <- 第二大单项
index + 写屏障 + holder 对称   约占 20%
bootstrap + 弹性              约占 10%
```

**两个最大项（MoE 分组、planner）各占四分之一，且都必须做**，
这是方案的真实体量。前三轮的「可行」结论成立，但不是小工程。

---

## 2. 代价与潜在风险

### 2.1 前三轮风险的处理状态

| 风险 | 状态 |
|---|---|
| 读写冲突 | **已收窄**：量化路径无冲突，同精度路径一个计数器 + 写屏障 |
| 内存注册有效性 | **已消除**：改为 holder 注册/注销的自然结果，不再是独立不变量 |
| 元数据扇出 | **已规划**：MoE 分组（根治）+ strided + 可选 RDMA 按需拉取 |
| 控制面瓶颈 | **已规划**：plan 本地缓存、租约聚合、gRPC 读写分离 |
| trainer 故障 | **已规划**：DP 副本接管 + `CHUNK_UNAVAILABLE` |
| `is_rollout_mode_in_actor=True` | **已修复**：holder 对称化 + rollout 代表读盘 |
| QP 数量上限 | **仅记录为 W 的约束 ④，未给出 QP_budget 的测定方法** |
| GPUDirect 不可用 | **仅记录为「应探测」，未给出退化行为** |
| chunk 偏移与 buffer 深度的张力 | **已如实记录，未解决** |

**后三项是本轮要指出的**：方案把它们从「未发现」变成了「已记录」，
但「已记录」不等于「已解决」。具体地：

- **QP_budget 没有测定方法。** 不同 NIC 的 QP 缓存容量差异很大，
  方案说「`W × ranks_per_reader ≤ QP_budget`」但没说 QP_budget 从哪来。
  实际可行的做法是**不去测它，而是用「W 的上调必须伴随实测验证」兜住**:
  W 本来就是自适应的（§3.5.2.2），只要反馈信号包含「单读带宽是否劣化」,
  QP 超限会表现为带宽悬崖，被反馈自动压回去。**这个联系方案里没写明。**
- **GPUDirect 不可用时的退化行为未定义。** 若探测发现要经 host bounce，
  应当做什么？候选：降低该链路在选源中的权重（优先同节点 peer）、
  或直接禁用该路径。方案只说「记入拓扑作为一档权重」,
  但没说权重值与 host bounce 的带宽差（可能一个量级）是否匹配。

### 2.2 本轮新识别的风险

**(a) `is_origin` 位的维护在 holder 对称化后变得微妙。**
§3.5.4 禁止链式 peer，靠 `is_origin` 位（原 `from_trainer`）。
但在 bootstrap 场景下，**origin 是 rollout 代表而非 trainer**,
而它读的是 checkpoint（磁盘），不是任何 holder。
所以 `is_origin` 的定义应当是「本版本中不从任何 peer 读取而获得权重的节点」,
涵盖「从 trainer 读」「从磁盘读」「自己就是 trainer」三种来源。
方案改了名字但没重新定义语义，**这会在 bootstrap 路径上产生歧义**。

**(b) 写屏障与 chunk 偏移叠加时，trainer 侧的等待可能被放大。**
偏移让 W 个读者在任意时刻读不同 chunk。trainer 回写时要检查「该 chunk 有无在飞读者」,
而偏移意味着**任意时刻都有某个 chunk 正被读**。
最坏情况下 trainer 的每个 chunk 回写都要等一次。
量化路径下不存在这个问题（读的是 wire buffer），
**但同精度路径下偏移与写屏障是互相加剧的**，方案没有分析这个组合。
缓解是显然的：同精度路径下关闭偏移（偏移的收益是加速扩散，而同精度路径
本来就该优先用 peer），但这条规则要写出来。

---

## 3. 网络通信：优化是否充足，能否扩展到万卡 + 1TB

### 3.1 已落实的

| 优化 | 落实状态 | 对万卡/1TB 的作用 |
|---|---|---|
| 去冗余（`N×M → M`） | §3.5 三阶段 plan | **规模越大收益越大**，核心收益 |
| PCIe 流量归零 | §3.6.2b 已计入 | 1TB×32 节点消除数十 TB |
| 批量提交（prepped + strided） | §3.6 | 消除 `O(#tensors)` 开销 |
| 元数据去扇出 | §3.5.5 | **万卡的前置阻塞项**，已有方案 |
| 轮级决策本地化 | §3.5.6 | 1TB/2000 轮下避免 `2000×RTT` |
| W 跟随 NCCL 在飞状态 | §3.5.2.2 | 避免与 grad reduce 争 NIC |
| chunk 大小自适应 | §3.5.6 | 1TB 用 4GB chunk 降到 250 轮 |
| 低优先级 stream（colocate） | §3.6.2c | 量化不抢 decode 带宽 |

### 3.2 未解决的

**(a) incast 到单个 peer 的流控只有 `P` 这一个旋钮，且 P 无依据。**
跨节点 peer 下多个 rollout 同时读同一 peer 形成 incast。
RDMA 的拥塞控制（DCQCN / 或 lossless fabric 的 PFC）会处理，
但 PFC 可能引发 head-of-line blocking，影响该 peer 的推理流量。
**阶段一不开跨节点 peer，所以这个问题被推迟而非解决**，这是正确的排序,
但阶段二之前必须有实测。

**(b) 没有考虑多 NIC / 多轨道（rail）拓扑。**
万卡集群普遍是每节点多张 NIC、按 rail 组网。
`NetworkTopology` 只有 `(node_ip, gpu_id)`，**无 rail 概念**,
所以无法做「同 rail 优先」的选源。这在万卡下是真实的带宽差异来源。
UCX 本身有 multi-rail 支持（`UCX_IB_DEVICES` 等），
所以**数据面可能已经受益，但选源决策看不到它**。
这一项方案完全没提，是一个明确的空白。

**(c) 1TB 模型的首次 bootstrap 仍是瓶颈。**
稳态传输已优化到 `M`，但 bootstrap 要从磁盘读 1TB。
方案复用 `broadcast.py` 树形扩散，所以**磁盘只读一次**,
但 `O(log N)` 轮的树形扩散在 1TB 下仍是分钟级。
这是一次性成本，可接受，但方案没有给出量级预期。

### 3.3 结论

**稳态传输的优化是充足的**：去冗余 + 批量 + 重叠 + 本地化决策，
对齐了 `T_min` 下界，且三条万卡阻塞项（元数据、轮数、控制面）都有方案。

**不充足的是拓扑感知的精细度**：rail、GPUDirect、incast 三项都只做到「识别」
而未做到「利用」或「测定」。这与 §3.5.3 明确的取向一致
（「不做实时带宽探测、不做连续优化」），**是有意的权衡而非遗漏**,
但应当在文档里把「这是权衡」和「这是空白」区分开。rail 属于空白。

---

## 4. 场景覆盖

### 4.1 `is_rollout_mode_in_actor=True`：已支持

§3.3.1a 的三步链路完整：

```
启动：rollout 每节点代表读盘 -> 注册为 holder(v0) -> peer 扩散
切回 trainer：从 rollout 侧 pull（ray_trainer.py:2578 的 pull_model 已存在）
再切回 rollout：trainer 成为 holder，rollout 从 trainer 读
```

**三步都复用同一套 holder 语义**，这是 §3.1 对称化的直接收益。
验证项 12 覆盖了全流程回归。

一处遗留：**切回 trainer 时，trainer 需要的是「训练用 dtype」的权重，
而 rollout 持有的可能是量化后的**。bf16 train + fp8 rollout 下，
从 rollout pull 回来的是 fp8，**不能直接作为 bf16 训练的起点**
（反量化会损失精度，且 optimizer master params 需要 fp32 精度）。

这条链路今天能工作，是因为 PS 持有 bf16 副本。**直供后这个问题暴露出来。**
正确的处理是：`is_rollout_mode_in_actor=True` 的 bootstrap 下，
**rollout 代表读盘时应当同时保留一份训练精度的权重**，或者
**trainer 切回时直接从 checkpoint 加载而不是从 rollout pull**。
后者更简单且正确（切回是低频事件，读盘可接受）。
**方案没有处理这一点，这是 §4 发现的第一个缺口。**

### 4.2 colocate：已支持，但有一条未说明的依赖

§3.7 的原地交接依赖「train 侧把量化结果直接写进 rollout 已注册的参数显存」。
这在 **量化路径**下成立（量化产物就写进 rollout 的参数）。
但在**同精度路径**（bf16 train + bf16 rollout）下，
「trainer 参数」与「rollout 参数」是同一块内存吗？

- 若是同一块：trainer 的 optimizer 回写会**直接改写 rollout 正在用的权重**,
  这比跨进程的读写冲突更糟（没有任何屏障机会）。
- 若不是同一块：就需要一次 bf16→bf16 的拷贝，**而 §3.4.2 明确说不做这个拷贝**。

**这是一个真实的矛盾**，方案没有讨论。实际上 colocate + 同精度的组合下，
必须有一次拷贝或一次切换，不存在「零拷贝且零冲突」的解。
最简单的处理是：**colocate 场景下强制要求量化路径**（即 rollout dtype 必须
不同于 train dtype），配置校验拒绝 colocate + 同精度。
这在实践上是合理的（colocate 的动机本来就是省显存，而 fp8 rollout 正是省显存），
但必须显式写出来。

### 4.3 精度配置矩阵：本轮发现的覆盖盲区

方案按「train dtype 是否等于 rollout dtype」二分，但实际配置空间更大：

| train | rollout | 路径 | 方案是否覆盖 |
|---|---|---|---|
| bf16 | bf16 | 同精度 | 是（但 colocate 下有 §4.2 的矛盾） |
| bf16 | fp8 | 量化 | 是，推荐 |
| bf16 | fp4 | 量化 | 是（§3.8.1 的 swizzle 处理） |
| fp8 (TE) | fp8 | 同精度 | 是（scale 转码，§3.8） |
| fp8 (TE) | fp4 | 量化 | **部分**：fp8→fp4 的再量化精度损失未讨论 |
| bf16 | bf16 + colocate | 同精度 | **否**，见 §4.2 |

**`fp8 → fp4` 这一格是真实的盲区**：方案说「量化路径含 fp8→fp4」，
但从 fp8 再量化到 fp4 是**二次量化**，误差不是从 bf16 直接量化到 fp4 的误差。
TE 的 fp8 参数已经损失了精度，再降到 fp4 会放大。
正确做法是**从 optimizer 的 fp32 master params 量化到 fp4**,
而不是从 fp8 model params。这需要访问 optimizer 状态，
与 §3.4.1 的「optimizer 写 master shard」是同一块结构，**可行但方案未写**。

### 4.4 弹性扩缩容：方案完整但依赖未实现的 NIXL 能力

§3.3.1c 的三点（新 instance 冷启动、缩容撤销 holder、NIXL 增量注册）覆盖完整。
但**第三点是外部依赖**：`server.py:56/90` 的 TODO 是 PSRL 自己的代码，
需要用 NIXL 的 `invalidateRemoteMD`/`loadRemoteMD` 实现。

**风险**：这两个 API PSRL 从未调用过，且元数据交换当前是「一次性全量握手」的
设计（`_is_all_client_infos_recved` 一旦为真就 return）。
改成增量不是加个分支，而是**把握手从一次性协议改成持续协议**,
这会影响 §3.5.5 的元数据方案（指针表也要支持增量）。
两者应当一起设计，方案目前把它们分开写。

---

## 5. 完整性与瓶颈

### 5.1 本轮瓶颈清单

| 候选 | 严重度 | 状态 |
|---|---|---|
| 元数据扇出 | 致命 | 有方案（§3.5.5），**但与弹性增量注册耦合，需合并设计** |
| MoE 分组未做 | 致命（大 MoE） | 有方案，是前置项 |
| colocate + 同精度的矛盾 | **高（正确性）** | **本轮新发现，§4.2** |
| 切回 trainer 的 dtype 不匹配 | **高（正确性）** | **本轮新发现，§4.1** |
| fp8→fp4 二次量化 | 中（精度） | **本轮新发现，§4.3** |
| QP 上限 | 中 | 已记录，建议靠 W 反馈兜住 |
| rail 拓扑未感知 | 中（万卡） | **空白** |
| PS 单进程 | 高 | 有方案 |
| trainer 出口 | 高 | 有方案 |

### 5.2 性能优化空间

已覆盖：去冗余、批量、三类重叠、精度、本地性、colocate 零搬运、W 自适应、扩散。

**移除**：渐进可用（vLLM 不支持，§5.1 待优化清单已记录理由）。

**仍空白**：rail 感知选源（§3.2b）。

---

## 6. 修订建议

按优先级：

1. **colocate 强制量化路径**（§4.2）：配置校验拒绝 colocate + 同精度 dtype。
   这是正确性，不是优化。
2. **切回 trainer 时从 checkpoint 加载，不从 rollout pull**（§4.1）:
   避免 fp8 权重作为 bf16 训练起点。
3. **fp4 量化的源应当是 fp32 master params，不是 fp8 model params**（§4.3）。
4. **`is_origin` 重新定义**为「本版本中不从任何 peer 获得权重的节点」,
   涵盖从 trainer 读、从磁盘读、自己是 trainer 三种（§2.2a）。
5. **同精度路径下关闭 chunk 偏移**（§2.2b），避免与写屏障互相加剧。
6. **元数据方案与弹性增量注册合并设计**（§4.4）。
7. **把 QP 超限交给 W 的带宽反馈兜住**，不单独测 QP_budget（§2.1）。
8. **标注「版本发布挂 param all-gather 回调」为待确认**（§1.1）。
9. **rail 感知**列入待优化清单，不在本期做（§3.2b）。

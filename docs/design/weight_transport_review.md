# 权重传输方案自审

> 对 [weight_transport](weight_transport) 的实现可行性、代价与风险、读写冲突处理、
> 完整性与瓶颈四项审核。结论先行：**主干方向成立，但原文有四处事实错误和一处关键的
> 时序误判，其中「trainer 一步一次 optimizer step」的假设是错的，它直接推翻了原
> §3.4 的 quiesce 论证。** 本文给出修正后的冲突处理方案（§3）。

---

## 0. 四处必须先更正的事实错误

审核时逐条回查源码，发现原方案有四处与代码不符。它们不改变主干方向，但有两处会改变实现难度评估。

### 0.1 Megatron 不需要 gather（原 §3.3 写错了）

原文写「Megatron 的 gather 是这里唯一的真实复杂度……位置从『push 前一次性全量』改为
『按 bucket 懒 gather』」。**查 `megatron_converter.py` 全文没有任何 `all_gather`。**

真实机制是：converter 为每个参数**声明** `NIXLSharding`（`get_sharding_for_param`），
把本地 TP/PP/EP 分片如实登记，重分片交给 NIXL 的 shard 代数在传输时完成
（`find_finest_shard_mesh` / `refactor_based_on_finer_shard_mesh`）。
也就是说 **PSRL 从来没有在 push 前 gather 过**，它一直是 shard-to-shard 直传。

这是个**好消息且显著强化了方案**：trainer 直供下同样不需要 gather，
rollout 按 shard 代数直接读 trainer 的本地分片。原文担心的「按 bucket 懒 gather」
整段可以删除，`O(bucket)` 的 gather 瞬态显存也不存在。

### 0.2 但 `ParamSyncPlan` 里确实有物化副本（原文漏了）

`megatron_converter.py:176` 的 `DTypeCastSync` 做 `converted_state_dict[key] = tensor.float()`,
`:289` 的 `ConcatenatedQKVSync` 做 `torch.cat((q,k,v))`。这些是**真实的额外张量**，
由 `before_push` / `after_pull` 维护双向同步。

它们的量级很小（fp32 pattern 参数 + QKV 重排），但**它们是约束 (b) 意义上的额外 GPU 权重存储**，
原文声称「bf16 路径零额外 buffer」对 Megatron 后端不准确。正确表述是：
**零额外的「全量级」副本，但保留 converter 已有的 per-param 物化张量**，
这部分今天就存在，本设计不增加也不减少。

更要紧的是：`before_push` / `after_push` 是**成对**的（`ZeroCenteredGammaSync` 在 push 后要 `sub_(1)`
把参数改回去）。直供模式下没有「push 完成」这个时刻，**rollout 可能在任意时间读取**，
所以 `after_push` 不能再被调用，否则 rollout 读到的是已经被改回去的值。
这意味着这些 sync action 必须改成**常驻正确**（即 `state_dict` 里那份始终是对外正确的值），
而不是「push 前临时搬正、push 后还原」。这是一处实打实的改造工作量，原文完全没提。

### 0.3 rollout 用 `load_format="dummy"`，初始化不读 checkpoint（原 §3.1 写错了）

原文说「初始化改为 rollout 与 trainer 各自从 checkpoint 加载，这是它们本来就会做的事」。
**错。** `vllm_async_server.py:110` 硬编码 `self.config.load_format = "dummy"`，
trainer 也是 `init_model("empty")`（`ray_trainer.py:2116`）。
今天**唯一读 checkpoint 的就是 PS worker**（`preload_checkpoint_to_cpu`），
再经 `broadcast_init` 树形扩散、最后 `initial_pull_from_ps` 灌给两侧。

所以删掉 PS worker 后**必须新增一条 bootstrap 路径**，这不是「本来就会做的事」。
可选方案：(i) trainer rank 0 读 checkpoint 后按 shard 代数分发（复用现有 broadcast 逻辑，
但源头从 PS 换成 trainer）。(ii) 两侧都改回真实 `load_format`，各自并行读共享文件系统。
(ii) 更简单但会把 `broadcast_init` 当初要解决的「N 个 reader 打爆文件系统」问题搬回来。
**建议 (i)**，并保留 `broadcast.py` 的树形算法，只换源头角色。原文把它整体删掉是错的。

### 0.4 PP 归属已经被处理了（原 §6 的风险被高估）

原文把「Megatron PP/EP 下 holder 归属必须正确」列为重大风险，并引 SkyRL 的 `held_names()` 为据。
实际上 PSRL 的 converter 只遍历**本 rank 的 `model.named_parameters()`**，
天然只登记自己持有的参数，`is_pp_missing_parameter` 在 vllm 侧也已跳过。
comm plan 在 `_make_comm_plan_generic` 里对「target 需要但无 source 提供」的 key 会**直接 assert**
（`No available key {key} in source client groups`），所以缺失是启动期报错而非静默读垃圾。

风险等级应从「可能静默读到垃圾」下调为「启动期 assert，需保证 PP stage 集合覆盖完整」。

---

## 1. 实现可行性

按「现有代码能承载多少」分三档。

### 1.1 低风险，基本是接线（可独立落地）

| 项 | 依据 | 工作量 |
|---|---|---|
| `prep_xfer_dlist` / `make_prepped_xfer` | NIXL python API 已暴露（`_api.py:488/564`），PSRL 只是没用 | 中，需重构 `client_read/write` 的 handle 缓存 |
| 描述符合并 | `merge_and_cache_xfer` + `merge_and_finish_cached_xfer` **代码已存在**，只是默认关闭、无人传 True | 小，打开并验证 |
| Nx5 strided | NIXL 原生支持（`nixl_bindings.cpp:148` `to_stride_dlist`） | 中，替换 pinned 中转逻辑 |
| colocate 用 `discard(("kv_cache",))` | vLLM 已有 `release_kv_cache_memory()`（`core.py:981`），语义与前置断言都对 | 小 |
| FP8 wire | SkyRL 的 `quantize.py` / `vllm_format.py` 可直接移植，vLLM `fp8_utils` 现成 | 中 |

**这一档的可行性没有疑问**，而且其中两项（描述符合并、strided）本身就在削减约束 (b) 关心的存储。

### 1.2 中风险，需要新模块但有明确对标

- **三阶段 planner**：PSRL 的 shard 代数够用，但 `_make_comm_plan_generic` 要整体重写，
  且要新增「peer 作为 source」的解析。WeightBridge 没开源，需自己实现。
  **这是最大的单项工作量**，估计是整个方案的一半。
- **WeightIndex + 租约**：与 `staleness_controller.py`（1309 行）必须合并而非并存，
  否则两套版本状态会漂移。合并本身是重构既有逻辑，风险在回归而非设计。

### 1.3 高风险，有未解问题

- **`ParamSyncPlan` 的语义改造**（§0.2）：从「push 前搬正 / push 后还原」改为「常驻正确」。
  `ZeroCenteredGammaSync` 常驻 `+1` 会让 trainer 自己的 forward 读到错的 gamma，
  所以不能简单地不还原。**需要的是一份该参数的独立副本**，而这又与约束 (b) 张力。
  好在涉及的参数量极小（layernorm gamma 级别），可接受。但这条必须在设计里写明，
  不能留给实现期发现。
- **bootstrap 路径**（§0.3）：必须新建，不是「本来就有」。
- **trainer 故障后 rollout 无源**：没有副本就没有降级目标。peer rollout 能部分缓解，
  但 trainer 全体重启期间必然失败。

**总评：可行，但原文低估了两处（ParamSyncPlan、bootstrap），高估了一处（Megatron gather）。
净工作量与原估计相当，分布不同。**

---

## 2. 代价与潜在风险

### 2.1 真实代价清单

| 代价 | 量级 | 可否缓解 |
|---|---|---|
| trainer 成为数据面关键路径 | 全局唯一权重源 | peer rollout 分担读取，但不解决 trainer 重启 |
| optimizer step 与 rollout 读的冲突 | 见 §3，比原文估计**严重得多** | §3 的方案 |
| converter sync action 语义改造 | 小量参数的额外副本 | 接受 |
| 新增 bootstrap 路径 | 一次性工程 | 复用 broadcast 树 |
| planner 重写 | 最大单项 | 分阶段，先保留旧 planner 作 fallback |
| 失去「PS 作为版本快照」的调试能力 | 不可忽视 | 见下 |

**最后一条值得单独说**：今天 PS 持有一份独立的版本快照，出问题时可以对比
「PS 里的 v」与「rollout 里的 v」来定位是传输错还是训练错
（`debug_log_info` / `_debug_log_ps_info` 就是为此存在）。直供模式下这个中间观测点消失了，
一旦 rollout 读到错的值，无法区分「trainer 当时就是错的」与「传输错了」。
**建议补偿**：保留一个可选的「校验模式」，对少量 key 做 checksum 比对
（miles 的 `--check-weight-update-equal` 与 `local_weight_checksum.py` 是现成范式）。

### 2.2 被原文漏掉的风险

**(a) NIXL 注册与 Megatron DDP buffer resize 的交互。** `_sleep_megatron_model` 对
`buf.param_data.untyped_storage().resize_(0)` 释放显存，`_wake_up` 再 resize 回来。
今天这之后必然跟一次 `nixl_register_after_wake_up` 重注册。直供模式下 rollout 可能
**在任意时刻**发起读取，如果此刻 trainer 正处于 resize 后、重注册前的窗口，
远端 rkey 失效 → RDMA 读到已释放的物理页。这是**静默数据损坏**，不是报错。
必须有显式的「trainer 权重可读」状态位，index 据此拒绝 pull。原文的租约只保护了
「版本不被覆盖」，没保护「内存注册有效」。**这是两个不同的不变量。**

**(b) fp8 量化缓冲在约束 (b) 下的真实位置。** BF16 train + FP8 rollout 时需要 `O(bucket)` 缓冲。
但这块缓冲**必须是 NIXL 注册过的**，而注册是昂贵操作，所以它得常驻。
常驻 `2×512MB` 的注册缓冲在 1TB 模型场景可忽略，但在小模型 colocate 场景占比不小。
应允许 bucket 大小按模型规模自适应。

**(c) peer rollout 作为 source 会与 colocate 冲突。** colocate 下 rollout 与 train 共享 GPU，
一个 rollout 为 peer 提供读取服务时会占用它自己的 PCIe/NIC 带宽与 SM（NIXL progress thread）。
eval 期间这会直接拖慢 eval 本身。应在 colocate 模式下禁用 peer 供给。

---

## 3. 读写冲突：原方案的论证是错的，这是修正

### 3.1 为什么原 §3.4 不成立

原文写「一个训练步的时长远大于一次权重读取，rollout 在 trainer 还在算 backward 时就读完了 v，
等到 optimizer step 时租约早已清零」。

**这个论证基于一个错误的前提：trainer 一步只有一次 optimizer step。**

查 `_run_scheduled_update`（`ray_trainer.py:3061`）：它按 `ppo_mini_batch_size` 和
`ppo_epochs` 构建 schedule，**循环 `schedule.steps` 多次调用 `update_fn`**，
每次都是一次完整的 forward/backward/optimizer step。`push_model` 只在
`finalize_step and is_final_step` 为真时置位（`:3067`），也就是说
**一个训练步内权重被改写 `num_minibatches × ppo_epochs` 次，但只有最后一次对外发布版本。**

这彻底改变了冲突的性质：

- 原文设想的「quiesce 窗口在 optimizer step 前，通常已清零」**在 mini-batch 循环内根本没有空隙**。
  第 k 个 mini-batch 的 optimizer step 紧接着第 k+1 个的 forward，中间没有可供 rollout 读取的稳定期。
- 更糟：**中间状态根本不该被读到。** mini-batch 3/8 之后的权重不是任何合法策略版本，
  rollout 若读到它，得到的是一个「半步」模型,算法上无意义。
- 所以 trainer 权重的「可读窗口」实际上只有：**最后一个 mini-batch 的 optimizer step 完成之后，
  到下一个训练步的第一个 mini-batch 的 optimizer step 之前。**

而这个窗口里 trainer 在做什么？下一步的数据准备 + 第一个 mini-batch 的 forward/backward。
**窗口是存在的，但它比原文想的窄得多，而且它的终点是硬边界**：第一个 mini-batch 的
optimizer step 一旦执行，v 就被破坏。

### 3.2 窗口到底有多宽：从 RL 与预训练的开销特性推

这是问题三的核心。把一个 PSRL 训练步拆开：

```
[rollout 生成]  异步，与训练重叠（PSRL 的核心设计）
[数据准备]      advantage 计算、log_prob 重算、reward（可异步）
[训练]          num_minibatches × ppo_epochs 次 (fwd + bwd + opt_step)
```

关键的量化特征，与预训练对比：

| 特征 | 预训练 | RL（PSRL） | 对冲突的影响 |
|---|---|---|---|
| 每步 optimizer step 次数 | 1 | `num_minibatches × ppo_epochs`（典型 4~16） | 可读窗口被切碎 |
| 权重传输频率 | 无 | 每步一次 | 传输本身是瓶颈项 |
| 单步时长 | 秒级 | 数十秒到分钟级（含生成） | 窗口绝对时长尚可 |
| 权重被读取的方数 | 无 | `N_rollout_instance`，可到数十 | fan-out 是主要压力 |

结论：**窗口的绝对时长（第一个 mini-batch 的 fwd+bwd，通常数百毫秒到数秒）
与一次全量权重读取（M/B，1TB 模型跨节点可达数秒）是同一量级。**
这意味着「一定来得及」不成立，必须设计成「不需要来得及」。

### 3.3 修正方案：双缓冲发布点，而非 quiesce 整个 trainer

正确的解法不是让 trainer 等 rollout，而是**让被读取的那份字节不是 optimizer 正在写的那份**。
而约束 (b) 不许额外副本。这两者的交集只有一个点：

> **利用 mini-batch 循环的结构：在最后一个 optimizer step 之后，
> 权重已经是 v 的终态，而下一个训练步的第一个 optimizer step 之前，它都是有效的。
> 把「量化/格式化」这一步（本来就要做的工作）当作发布屏障。**

具体分三种形态，按 train/rollout dtype 组合：

**形态 A：BF16 train + FP8 rollout（有天然隔离，推荐）**

量化产物（fp8 weight + scale）**就是 rollout 要读的字节，而它不是 trainer 的参数**。
trainer 的 bf16 参数继续被 mini-batch 循环改写，互不干扰：

```
最后一个 optimizer step 完成
  └─ 按 bucket 量化 bf16 参数 -> fp8 wire buffer（常驻注册，2×bucket）
       bucket 量化完即登记可读，rollout 开始读这个 bucket
  └─ trainer 立刻进入下一步，无需等待任何 rollout
       因为 rollout 读的是 wire buffer，不是参数
```

**这个形态下读写冲突根本不存在。** 代价是 `2×bucket` 的常驻缓冲（约束 (b) 允许的最小代价），
且它买到的不只是隔离，还有带宽减半。**这应该是默认推荐形态**，原文没有意识到它顺带解决了冲突。

但注意：bucket 是流式复用的，所以「所有 rollout 读完 bucket i」才能复用它装 bucket i+1。
fan-out 大时这会成为新的等待点。缓解是 §3.5 的 peer 扩散：第一批 rollout 读完后，
后来者从 peer 读，bucket 可以尽快复用。

**形态 B：FP8 train + FP8 rollout（无隔离，需租约）**

trainer 持有的 TE fp8 参数就是 rollout 要的字节，没有中间产物可作屏障。
此时必须租约 + quiesce，但 quiesce 的粒度要对：

```
以 bucket 为租约单位，而非整个模型
  最后一个 optimizer step 完成 -> 全部 bucket 登记可读
  下一步第一个 optimizer step 前 -> 需要全部 bucket 租约清零
  未清零的 bucket：超时后撤销租约，令该 rollout 重拉下一版本
```

**关键优化**：optimizer step 对 bucket 的写入是有顺序的。若让 index 感知这个顺序，
trainer 可以**边写边放行**：写完 bucket 0 之前，bucket 0 的读者必须先清零，
但 bucket 7 的读者还有时间。这把一个全局屏障拆成流水线。实现上依赖
Megatron DDP buffer 的 bucket 划分与 index 的 bucket 划分对齐。

**形态 C：BF16 train + BF16 rollout（无隔离，无中间产物）**

最困难的组合：既没有量化产物作屏障，又要传全量字节。建议**显式不推荐**，
并在配置校验时提示「BF16 rollout 下建议保留一份 wire buffer 或接受 quiesce 等待」。

### 3.4 从算法角度：让 rollout 拿到更新的权重

问题三的后半问得很关键，冲突处理方式会直接影响 off-policy 程度。三条设计取向：

**(a) 发布粒度应该比训练步更细，而不是更粗。**

今天 `push_model` 只在最后一个 mini-batch 后置位,所以 rollout 最快也只能拿到「上一个完整训练步」的权重。
但 `fine_grain_overlap` 这个 StepStrategy 已经存在于 PSRL（`psrl.fine_grain_overlap`），
它把训练步切成 chunk 以与 rollout 重叠。**在形态 A 下，可以在每个 mini-batch 后都量化并发布一个中间版本**,
让 rollout 拿到更新的策略。

但这件事**必须由算法决定而非 infra 擅自**：mini-batch 中间状态是否是合法策略，
取决于算法是否接受。GRPO/GSPO 的 importance sampling 修正项是按「采样策略 vs 当前策略」算的，
如果采样策略是一个 mini-batch 中间态，修正项仍然定义良好（它只需要知道采样时的 logprob）。
**所以这在算法上是可行的，但会增加策略的方差。** 建议做成可配置：
`publish_granularity: step | minibatch`，默认 `step`，让算法侧实验决定。

**(b) 冲突处理绝不能以「让 rollout 读旧版本」为代价。**

这是最容易犯的错：为了避免 quiesce 等待，允许 rollout 继续读 `v-k`。
这**直接增加 off-policy 程度**，是用算法质量换 infra 简单。正确的取舍方向是相反的：
**宁可让 trainer 等，也不要让 rollout 读旧的。** 因为：

- trainer 等待是确定的、可测量的吞吐损失。
- rollout 读旧权重是算法质量损失，表现为 reward 曲线变差，而且很难归因。

所以 §3.3 形态 B 的「超时后撤销租约」必须是**令其重拉更新版本**，而不是「让它继续用旧的」。

**(c) 直供天然比经 PS 更 on-policy，这是本设计的算法红利。**

今天的链路是 `train GPU → CPU PS → rollout GPU`，两跳。rollout 看到的版本至少滞后
「一次 push 完成 + 一次 pull 完成」。直供去掉中间跳，**同一个版本可以更早被 rollout 看到**,
而且 rollout 可以在 trainer 刚登记 v 时立刻开始读，不必等 push 到 N 个 PS worker 全部完成。
原文没有强调这一点，但它是约束 (b) 带来的意外收益：**少一跳就是少一份 staleness。**

量化地说：今天 `push(N×M)` 完成才登记版本，直供下 trainer 登记版本后立刻可读，
省掉的正是 `N×M / B_train` 这段时间。N=32 时这不是小数。

---

## 4. 完整性与瓶颈

### 4.1 已覆盖的优化空间

去冗余（三阶段 plan）、批处理（prepped descriptor + round）、重叠（双 buffer）、
精度（FP8 wire）、本地性（link priority + peer 就近）、colocate 原地交接。
对照 WeightBridge 的 `T_min` 模型，trainer 出口与 rollout 入口两侧都已对齐下界。

### 4.2 未覆盖或覆盖不足的

**(a) rollout 侧的 `process_weights_after_loading` 没有被计入关键路径。**
miles 的实测数据里 Kimi-K2 的 `post_load_weights` 要 **884ms**，
而整个 RDMA 是 7227ms 的一部分。这是**纯 GPU 计算**，与传输无关，
在本设计里它发生在「落地格式化」阶段，**当前方案把它当作可忽略项，这是错的**。
应当：按 atomic group 流水，让第 i 组的格式化与第 i+1 组的 RDMA 重叠。
原文 §3.6 提了双 buffer 但只覆盖了「量化/gather 与 RDMA 重叠」,
漏了「落地格式化与 RDMA 重叠」。

**(b) 控制面消息量随 rollout 数平方增长。**
WeightBridge 明确指出其 AGST 的亚线性增长「mainly driven by larger pipeline bubbles
and the **quadratically growing number of control-plane messages**」。
本设计的 index 是单个 Ray actor（`PSManager`），每个 rollout instance 每版本都要
预约/查询/释放。数十个 instance × 数千 bucket 的租约消息会把它打满。
**这是本设计最可能的单点瓶颈，而原文完全没提。**
缓解：租约按 instance 粒度而非 bucket 粒度聚合上报。plan 下发走一次性缓存
（plan 在握手期编译，稳态下不变，rollout 本地缓存即可，无需每次查询）。
考虑把 index 的热路径从 Ray actor 换成 gRPC（`psrl/grpc/ps_manager_service.py` 已有先例）。

**(c) trainer 出口在 fan-out 大时仍是单点，peer 扩散的收敛速度未分析。**
第一批 rollout 必须从 trainer 读，之后才能扩散。若有 R 个 instance，
理想情况是 `log(R)` 轮扩散，但这需要 index 实时知道「谁已持有 v」并动态改派。
原文说「先到的从 trainer 读，后到的从 peer 读」，但没说**如何保证先到的那批足够少**。
若所有 instance 同时发起（PSRL 的 greedy sync 策略正是如此
`sync_and_mig_strategy.method: greedy` 默认「sync all instances ASAP」），
它们会同时压 trainer，peer 扩散根本没机会发生。
**必须让 sync 策略与传输 plan 协同**：greedy 策略下要显式分批，
或者让 index 在解析 holder 时对后到者强制指派 peer 而非 trainer。
这是一个**跨模块的设计缺口**，原文把 planner 和 sync 策略当成独立的，它们不是。

**(d) 没有分析 NIXL progress thread 对 rollout 推理的干扰。**
`init_nixl_client` 有 `enable_prog_thread` 参数（train 侧 DCP 场景会关掉）。
rollout 侧作为 peer source 时，progress thread 会与推理争 CPU。
PSRL 已有 `route_trace.log` 这类观测，应当量化这项干扰。

**(e) 单点瓶颈汇总**

| 候选单点 | 严重度 | 本设计是否解决 |
|---|---|---|
| PSManager/index 控制面消息 | **高**，随 instance 数增长 | **未解决**，见 (b) |
| trainer 出口带宽 | 高 | 部分解决（三阶段 + peer），但受 (c) 制约 |
| greedy sync 导致的同时冲击 | **高** | **未解决**，见 (c) |
| rollout 落地格式化 | 中 | 未计入，见 (a) |
| trainer 可用性 | 中 | 未解决，无副本可降级 |
| quiesce 等待（形态 B/C） | 中 | §3.3 流水化缓解 |

---

## 5. 结论与建议的方案修订

**主干成立**：PS 索引化 + trainer 直供在约束 (b) 下是唯一自洽的形态，
而且它顺带减少一跳 staleness（§3.4c），对算法有利。

**必须修订的五处**：

1. **删掉「Megatron 懒 gather」整段**（§0.1），它基于错误前提。shard 代数已经处理。
2. **补上 `ParamSyncPlan` 语义改造**（§0.2）与 **bootstrap 路径**（§0.3），两处都是真实工作量。
3. **重写冲突处理**（§3）：原 quiesce 论证因 mini-batch 循环而失效。
   改为「形态 A（BF16 train + FP8 rollout）用量化产物作天然隔离、无冲突」为推荐路径，
   形态 B 用 bucket 粒度流水化租约，形态 C 显式不推荐。
4. **新增两条不变量**：「版本不被覆盖」与「内存注册有效」是**两个**不变量（§2.2a），
   DDP buffer resize 窗口期必须拒绝 pull，否则静默数据损坏。
5. **补上控制面瓶颈与 sync 策略协同**（§4.2b/c），这是最可能的实际瓶颈，
   且 greedy sync 与 peer 扩散目前是互相拆台的。

**建议的落地顺序调整**：原文建议「1-5 先行，再 7→8→6」。
考虑到 §4.2 发现的瓶颈，应插入一项前置工作：
**先把控制面从 Ray actor 热路径上移下来（plan 本地缓存 + 租约聚合），
再做直供。** 否则直供成功了，瓶颈会立刻转移到 index 上。

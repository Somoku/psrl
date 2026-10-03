# PSRL 权重传输与 PS 架构设计

> 状态：设计（未落地）。本文取代 `fp8_rollout_bf16_train.md`。
> 约束（用户给定，本文据此收敛）：
> **(a) 不做自动模型接入**，沿用现有显式 layout plan。
> **(b) 不引入额外的 GPU 权重存储，也不引入额外的 CPU 权重存储**，一律从 trainer 或 rollout
> GPU 上**已经存在**的权重取数。
> **(c) TMS 仅在必不可少或能极大简化实现时使用。**
> **(d) 不依赖 SkyRL / miles 作为第三方库。** 它们在本文中只作为**设计调研对象**，
> 不产生任何 import。所需原语全部来自 PSRL 已有依赖：vLLM（`fp8_utils`、`deep_gemm`、
> `CuMemAllocator`）、Transformer Engine（MXFP8 quantizer、fp8 参数）、NIXL、Megatron-Bridge。
> 调研对象：SkyRL v0.4（`third_party/SkyRL`）、miles（`third_party/miles`）、verl（`third_party/verl`）、
> WeightBridge（arXiv 2609.25442）、NIXL（`third_party/nixl`）、vLLM（`third_party/vllm`）。
>
> **本文已按 [weight_transport_review](weight_transport_review) 的审核结论修订。**
> 审核推翻了初稿的两处事实判断（Megatron 需要 gather、rollout 初始化会读 checkpoint）
> 和一处时序论证（quiesce 窗口），修订已并入 §3.2 / §3.3 / §3.4 / §3.5 / §3.6。

---

## 0. TL;DR

约束 (b) 把设计空间收得非常窄，而这恰好让方案变锐利：**既然不许有任何一份额外副本，
那么权重的唯一副本就只能是 trainer GPU 上那份。PS 于此退化为一个不持有字节的索引服务。**

这不是把 PS 做小，而是**把 PS 从数据面彻底移出**：

```
今天      train GPU ──写满 N 份──▶ N×(CPU 全量副本) ──读──▶ rollout GPU
本设计    train GPU ◀──────────── rollout GPU 按需 READ ────────┘
          PSManager 只存 (version, key, shard) -> holder，零字节
```

| 维度 | 今天 | 本设计 |
|---|---|---|
| train 出口流量 | `N×M` | `M`（fp8 下 `M/2`） |
| 额外 GPU 权重存储 | 无 | 无（约束 b） |
| 额外 CPU 权重存储 | `N×M` | **0** |
| colocate 切换 | 全量 host→GPU 回灌 + NIXL 重握手 | 不搬权重，不重注册，不走 host |
| 传输提交 | 每 (key, shard) 一次 `initialize_xfer` | 握手期 prep 一次，每轮按 index 提交 |
| pull 阻塞 | push 期间全体 pull 阻塞 | 无锁，trainer 侧版本门闸 |
| FP8 | 不支持 | train 侧出 wire 格式，链路不感知精度 |

代价必须说在前面：**去掉 CPU 副本就去掉了吸收版本偏斜的缓冲区**，所以 trainer 的
optimizer step 与 rollout 的读取会直接冲突。两个决定性事实：一个训练步内权重被改写
`num_minibatches × ppo_epochs` 次（`ray_trainer.py:3061`），但 Megatron
`DistributedOptimizer`（默认开启）下 optimizer 写的是 **fp32 master shard**，
只有 param 回写那一瞬触碰 rollout 正在读的张量。**所以冲突窗口远比直觉窄**，
§3.4 把它统一成**一条规则**：rollout 永远读「发布缓冲」。量化路径下该缓冲就是量化产物
（零额外显存，因为量化本来就要有落点），冲突归零。同精度路径下直读 trainer 参数，
用一个 per-chunk 的在飞读者计数器加写屏障放行，不需要租约表。

PS 的新定位：**不持有字节，但它是权重的版本与位置权威**，维护当前版本号、holder 表、
在飞读者计数、routing plan，以及合并进来的 staleness 账本。全是元数据，KB 级。
§3.2 同时给出控制面不成为新瓶颈的四条措施。

**holder 是对称的**：trainer 与 rollout 都可以是权威源，取决于谁当前持有最新版本
且处于唤醒状态。这不是为了优雅，而是正确性要求，因为
`is_rollout_mode_in_actor=True` 时 trainer 以 `meta` 模式注册、没有真实显存（§3.3.1）。

**一条配置期硬约束**：colocate 必须走量化路径（train dtype != rollout dtype），
否则 trainer 的 optimizer 回写会直接改写 rollout 正在用的权重，没有屏障机会。
完整精度矩阵见 §3.8.0。

六个支柱：**PS 索引化 → trainer 直供 → 冲突隔离 → 全局去冗余规划 → 执行层批处理 → colocate 原地交接**，
FP8 作为贯穿全链路的 wire 格式。自动接入按约束 (a) 移出本设计，§8 只留接口预留说明。

---

## 1. 现状：三个量化事实

三条都从源码核实，是设计的出发点。

### 1.1 push 是 N 倍冗余的，而且冗余的是 CPU 副本

PS worker 按「rollout / actor / validate 涉及的不同节点」各起一个
（`ray_trainer.py:1980` 起的 `ps_node_ids` 收集），每个 worker 用
`NIXLSharding.default()` 注册**整张**张量（`hf_converter.py` 注释原文：
`PS holds full (non-TP-sharded) weights`）。

comm plan 对每个 (PS target, key) 把所需 shard 分给持有它的 train source
（`comm_plan.py::_make_comm_plan_generic`）。由于每个 PS worker 都需要全部 shard，
train rank 必须把自己那份 shard 写给**全部** `N` 个 PS worker
（`base_train_worker.py:156` 的 `for key ... for target_agent, target_client in zip(...)`）。

> **train 侧总出口 = `N × M`，CPU 总占用 = `N × M`。**

这正是约束 (b) 要消灭的东西。注意它不是「CPU 放不下一份」，是放不下 N 份。

pull 方向是干净的，而且**已经是 pull 模式**：comm plan 的 link priority 让每个 rollout rank
从本节点 PS worker 读自己的 TP shard。rollout 侧发起 READ 的骨架已存在
（`vllm_extension.py::nixl_pull_model_core`），这是 PSRL 相对 miles/verl 的结构性优势，必须保住。

### 1.2 热路径是逐张量提交的

`client_read` / `client_write` 对每个 (key, shard) 调一次 `agent.initialize_xfer` + `agent.transfer`
（`client.py:1173`、`client.py:1353`）。合并路径 `merge_and_cache_xfer` 实现了但**默认关闭且无人调用**，
所以 pull 结尾的 `merge_and_finish_cached_xfer()` 实际是空操作。

WeightBridge 在 30B 上实测：逐张量的格式转换 + 控制面开销单项 590ms，超过 `3×T_min`。
这部分开销与模型大小无关、只与张量数量有关，MoE 下尤其致命。

而 NIXL 已有的正确原语 PSRL 没有用：`prep_xfer_dlist`（握手期准备一次）+
`make_prepped_xfer(local_indices, remote_indices)`（按下标组合、每轮提交），
以及 Nx5 strided descriptor（`(addr, len, dev_id, stride, count)`，原生表达非连续 shard）。

### 1.3 colocate 切换在经由 host 搬全量模型

`switch_to_trainer_mode` / `switch_to_rollout_mode`（`ray_trainer.py:2453` 起）每次切换做：
NIXL dereg → 释放显存 → 对侧分配显存 → **重新握手**（`nixl_send_local_info_to` +
`nixl_wait_for_update_infos` + 全局 broadcast client infos）→ **从 CPU PS 回灌全量模型**。

三项成本：全量 host→GPU 的 PCIe 拷贝、`O(#tensors)` 的 descriptor 重建与元数据广播、显存 pause/resume。
verl / SkyRL 的 colocate 快，是因为它们**根本不把权重挪出显存**。PSRL 慢的原因不在 NIXL，
在于把「同一块物理显存在 train 与 rollout 间交接」实现成了「经由 host 的一次完整往返」。

---

## 2. 外部方案调研：各自解决了哪一块

| 系统 | 核心机制 | 本设计采纳 | 不采纳的原因 |
|---|---|---|---|
| **WeightBridge** | 全局三阶段路由（receive / external-exchange / internal-exchange）。round-based 批处理。探测 `load_weights` 自动推导对应关系 | 三阶段去冗余规划、`T_min` 下界、round 批处理 | trainer 发起、固定 worker 集合，不含 pull 与弹性。自动接入按约束 (a) 排除 |
| **miles P2P** | train rank RDMA 直写 rollout 显存。CPU 侧建 sglang 模型副本以复用其 `weight_loader`。atomic group。bucket 512MB | atomic group 不变量、bucket 流水 | **CPU replica 违反约束 (b)**。单源瓶颈：rollout DP 一涨 AGST 近线性退化 |
| **verl** | `CheckpointEngineRegistry` 可插拔后端。NIXL 后端是 prev/next 中继链，双 buffer 边收边转发。`delta_sharded` 只传变化的 `(position, value)` | 双 buffer 流水思想。delta 作为跨 DC 正交选项 | 中继链是固定环拓扑，不感知 rollout 实际需求，也不是 pull。**其 2×bucket 常驻 buffer 要计入约束 (b) 预算** |
| **SkyRL v0.4** | `sharded_rdt`：rollout **pull**，consumer 在 init 期用 `FakeRDTTensor` **烘焙静态 pull plan**。producer sidecar 以 CUDA IPC 共享 trainer 显存。serve slot 跨 replica 共享 | **pull + 预烘焙 plan**、**sidecar 以 IPC 共享而非复制 trainer 显存**、slot 共享、FP8 wire | sidecar 的 ring buffer 仍是额外显存，本设计改为直读参数（§3.3） |

**与本设计最近的是 SkyRL 的 `sharded_rdt`**：它同样是 rollout 从 trainer 侧拉取、同样预烘焙 plan、
同样用 NIXL。它的 producer sidecar 以 **CUDA IPC 共享** trainer rank 的显存而非复制，
这正是约束 (b) 想要的形态。本设计在两点上更进一步：

1. SkyRL 仍有 K×chunk 的 publish ring 与 receive ring（文档明言「RDT receive buffers live
   **outside** `gpu_memory_utilization`」）。本设计让 rollout **直读 trainer 的参数张量本身**，
   bf16 路径下零额外 buffer（§3.3）。
2. SkyRL 的 `sharded_rdt` 与 FP8 serialized wire **互斥**，且不支持 colocate。本设计两者都要。

另外两条事实值得单列：

- **FP8 wire 有两种，都已在 SkyRL 打通。** blockwise（Hopper）：128×128 tile，FP32 或 power-of-2
  scale，`.weight_scale_inv`，vLLM `quant_method=fp8`。MXFP8（SM100+）：沿 K 方向 1×32 group，
  E8M0 偏置指数存 uint8，`.weight_scale`，走 compressed-tensors。TE 的 C++ kernel 跑在显存带宽
  （B200 约 4TB/s），torch 组合实现约 100GB/s 且作为 bitwise 一致性 oracle。
- **`fp8_param` 证明 FP8-native 训练可行**：TE 持有常驻 fp8 参数 + scale，参数 HBM 省 39%~42%。
  这同时意味着 FP8 train 形态下 trainer 本来就持有 fp8 权重，**直供给 rollout 不需要任何转换缓冲**。

---

## 3. 架构：PS 索引化 + trainer 直供

### 3.1 核心抽象：PS 不再持有字节

```
控制面  WeightIndex（PSManager 内，零字节）
          (version, key, shard) -> [holder, ...]，holder 带链路成本与租约
          holder 种类：TRAIN_GPU | ROLLOUT_GPU | SELF（colocate 同卡），前两者地位对称
          职责：版本可见性、租约计数、holder 注册/失效、下发预编译 routing plan

数据面  纯 NIXL，端点就是两侧已有的权重张量
          train 侧：converter 产出的 unified_state_dict（已注册，今天就在注册）
          rollout 侧：convert_vllm_inplace 注册的 vLLM 运行时参数张量
          没有中间存储，没有 PS worker 进程
```

**holder 语义是对称的：trainer 不是唯一的权威源。**
初稿把「trainer 持有权威副本」当成不变量，这是错的。PSRL 的三种部署形态都会
让源发生转移：colocate 切换（trainer 睡下、rollout 持有）、
`is_rollout_mode_in_actor=True` 启动（trainer 以 `meta` 模式注册，没有真实显存，
见 §3.3.1）、弹性伸缩（instance 集合变化）。正确的表述是：

> **权重的权威副本在「当前持有最新版本且处于唤醒状态的那组 GPU」上，
> 它可能是 trainer，也可能是 rollout。index 的职责就是回答这个问题。**

所以 `TRAIN_GPU` 与 `ROLLOUT_GPU` 在 holder 表里**地位相同**，
区别只在链路成本与「是否为本版本的首个源」。§3.5.4 的 `from_trainer` 位
应改名 `is_origin`，定义为**「本版本中不从任何 peer 获得权重的节点」**，
涵盖三种来源：从 trainer 读、从 checkpoint 读（bootstrap 的 rollout 代表）、
自己就是 trainer。只有 `is_origin` 为真的 holder 才可被指派为 peer 源，
这样禁止链式的规则在 bootstrap 路径上也成立。
这个修正不改变数据面，但它让 bootstrap、colocate 切换、弹性伸缩三条路径
共用同一套 holder 语义，而不是各写一套。

holder 种类从四种缩到三种，`STORE_CPU` / `STORE_GPU` 按约束 (b) 删除。
`PSStorageWorker` 这个角色随之**整体消失**，`PSWorkerGroup` / `PSStoragePlan` 一并移除。
但 **`broadcast_init` 的树形算法要保留**：今天唯一读 checkpoint 的就是 PS worker
（`preload_checkpoint_to_cpu`），rollout 用 `load_format="dummy"`
（`vllm_async_server.py:110`）、trainer 用 `init_model("empty")`（`ray_trainer.py:2116`），
所以删掉 PS worker 后必须新建 bootstrap 路径（详见 §3.3 末段），
做法是让 trainer rank 0 接替 PS rank 0 的读盘与扩散角色，复用 `broadcast.py`。

一次 pull 统一表述为：**rollout 向 index 声明「我要 version v 的这些 (key, shard)」，
index 返回按链路成本排序的 holder，执行层按预编译 plan 读取。** 三种场景是同一机制的三种解析结果：

- colocate → `SELF`，同一块显存，退化为零拷贝（§3.6）
- 分离部署 → `TRAIN_GPU`，跨节点 RDMA 直读 trainer 参数
- 大 fan-out → 追加 `PEER_ROLLOUT_GPU`，已拿到 v 的 rollout 分担出口（§3.5）

### 3.2 支柱一：PS 的新定位，它维护什么

**PS 不再是存储，它是权重的「版本与位置权威」。** 保留 `PSManager` 这个角色与进程，
删除的是 `PSStorageWorker`（数据面）。它维护的全部状态如下，**全是元数据，总量 KB 级**：

| 状态 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| **当前权重版本号** | `v`，以及每个版本的发布时间戳 | trainer 发布时 | rollout、coordinator、router |
| **holder 位图** | 每 instance 一个 `Bitmap(num_chunks)` + 节点索引（**不按 key 展开**，见 §3.5.3c，64 instance 约 16KB） | trainer 注册、rollout 释放租约时搭车上报 | pull 选源 |
| **可读状态位** | 每个 trainer rank 的「权重注册有效且非中间态」 | trainer 在 resize / offload / mini-batch 边界翻转 | pull 准入 |
| **租约表** | 哪个 instance 预约了哪个版本，用于回写放行（§3.4） | rollout 预约 / 释放 | trainer 回写前检查 |
| **routing plan** | 握手期编译的每 instance 三阶段 plan | planner 一次 | rollout 首次拉取后本地缓存 |
| **staleness 账本** | Reserve/Occupy/Consume 缓冲（**合并自现有 staleness controller**） | 现有逻辑 | 现有逻辑 |

**版本号的语义必须保持连续**，不能因为冲突而跳号：`maybe_delete_buffer(version_tag - 1)`
（`ps_manager.py:1203`）和 staleness 缓冲回收都按 `v-1` 索引，跳号会让缓冲泄漏。
所以 §3.4.3 的「超时强制推进」推进的是**写权限**，不是版本号。

它与现有 staleness controller 的 Reserve/Occupy/Consume 本质同源，
**应当合并进同一个版本权威**而不是两套状态互相同步（今天
`set_ps_model_version` 要向 coordinator 单向推送，就是两套状态的症状）。

`ps_mode` 枚举（`cpu` / `cpu_ref` / `nixl_cpu` / `nixl_gpu`）删除：四个值里 `nixl_gpu`
至今是 `NotImplementedError`，其余三个都在描述「副本放哪」，而本设计没有副本。

#### 3.2.1 控制面是本设计最可能的单点瓶颈，这是解法

索引化之后，PS 不再搬字节，但**所有 rollout 的所有拉取都要经过它做准入与解析**。
WeightBridge 明确指出其端到端时间的增长主要来自 pipeline bubble 与
**控制面消息数的平方增长**。今天 `PSManager` 是单个 Ray actor，而且
`@_state_locked` 把它的状态操作全序列化了。数十 instance × 数千 chunk
的租约消息会直接把它打满。

四条措施，按「消除 > 降频 > 加速」排序：

**(a) 消除：plan 不进热路径。** plan 在握手期编译、稳态下不变，rollout **本地缓存**，
拉取时**完全不查询 PS**。只有 holder 集合变化（instance 增减、trainer 重启）
才由 PS 主动失效通知。这一条把稳态下最大的消息源直接删掉。

**(b) 降频：租约按 (instance, version) 聚合，不按 chunk。**
一个 instance 一次预约覆盖它整轮要读的全部 chunk，释放同理。
消息量从 `O(instances × chunks)` 降到 `O(instances)`。
chunk 粒度只用于 **trainer 侧的本地放行判断**：trainer 自己知道哪些 chunk 已被读完
（由 rollout 完成时的一次聚合通知携带 bitmap），不需要每 chunk 一次往返。

**(c) 降频：版本发布用推送而非轮询。** 今天 rollout coordinator 有 `status_loop`
周期性拉状态。版本发布是低频事件（每步一次），应当由 PS 主动推给 coordinator
（现有 `set_ps_model_version` 已是此形态，保持），rollout 侧不轮询 PS。

**(d) 加速：热路径走 gRPC，与 Ray actor 状态分离。**
`psrl/grpc/ps_manager_service.py` 已有先例（SMG router 就是这么读 PS 状态的）。
准入与租约这两个高频读走 gRPC + 内存快照，**不进 `@_state_locked` 的临界区**。
写路径（版本发布、holder 变更）仍走 Ray actor 保持强一致。
这是读写分离，不是双写。

**残余风险**：PS 仍是单进程单点。本设计不引入 PS 高可用，但因为它只持有 KB 级元数据，
重建成本极低（trainer 与 rollout 重新注册 holder 即可），这与「PS 持有 N×M 字节」
时代的故障语义完全不同。应确保 PS 重启后能从两侧重建状态，而不是要求全局重启。

### 3.3 支柱二：trainer 直供，零额外缓冲

**rollout 直接 READ trainer 的参数张量。** 两侧的 NIXL 端点今天就已经存在：

- train 侧 `nixl_convert_params` 产出 `unified_state_dict`，是 FSDP/Megatron 参数的 HF 视图，
  已经 `register_local_tensors`。
- rollout 侧 `convert_vllm_inplace` 注册的是 vLLM 运行时参数张量本身，`nixl_pull_model_core`
  的 `client_read` 已经是 RDMA 直写进去（绕过 `load_weights`）。

也就是说**今天的链路已经是「写进最终张量」，缺的只是把中间那跳 CPU 副本去掉**。
改动是把 pull 的 target 从 `ps_client_pull_name` 换成 train client，
并把 `NIXLClientType` 里的 `PS_FOR_PUSH` / `PS_FOR_PULL` 折叠掉。

三种 train 形态的缓冲账：

| train 形态 | rollout 读到的是 | 额外显存 |
|---|---|---|
| FP8 train + FP8 rollout | TE 常驻 fp8 参数 + scale（转码后） | **0**，直读 |
| BF16 train + BF16 rollout | FSDP/Megatron 参数本身 | **0**，直读 |
| BF16 train + FP8 rollout | 量化产出的 fp8 wire buffer | `2×chunk`（量化本来就需要的落点） |

只有第三种需要落地缓冲，且**它天然是有界的**：量化按 chunk 流式进行，
一个 chunk（默认 512MB）量化完就被 rollout 读走、立刻复用，所以显存是
`O(chunk)` 而非 `O(M)`。注意这块缓冲**必须是 NIXL 注册过的**，而注册昂贵，所以它常驻，
不能每步分配释放。chunk 大小应随模型规模自适应：1TB 场景下 `2×512MB` 可忽略，
小模型 colocate 下占比不小。

这是约束 (b) 允许的最小代价，而且它买到的不只是带宽减半（§3.8），
**还顺带消除了读写冲突**（§3.4 量化路径）。

> **为什么这比 SkyRL 的 sidecar 更省**：SkyRL 的 producer 要把 gather 结果 pack 进
> CUDA IPC ring slot，consumer 再 pull 进 receive ring，两侧各一份 K×chunk。
> 本设计在 bf16 路径下根本不 pack。参数张量自己就是传输源，descriptor 直接指向它。

**Megatron 不需要 gather，shard 代数已经处理。** `megatron_converter.py` 全文没有任何
`all_gather`：converter 为每个参数**声明** `NIXLSharding`（`get_sharding_for_param`），
把本地 TP/PP/EP 分片如实登记，重分片由 NIXL 的 shard 代数在传输时完成
（`find_finest_shard_mesh` / `refactor_based_on_finer_shard_mesh`）。PSRL 一直是 shard-to-shard
直传，所以直供下 rollout 按 shard 代数直接读 trainer 的本地分片，同样不需要 gather。

**PP 归属已经是启动期错误而非静默错误。** converter 只遍历本 rank 的 `named_parameters()`，
天然只登记自己持有的参数。comm plan 对「target 需要但无 source 提供」的 key 直接 assert
（`No available key {key} in source client groups`），所以 PP stage 覆盖不全在启动期暴露。

**但 `ParamSyncPlan` 的语义必须改造，这是一处真实工作量。** 现有 sync action 是
「push 前搬正、push 后还原」的成对钩子：`ZeroCenteredGammaSync.before_push` 做 `add_(1)`、
`after_push` 做 `sub_(1)`（`param_sync.py:65`）。直供模式下**没有「push 完成」这个时刻**，
rollout 可能在任意时间读取，所以 `after_push` 不能再被调用。这些 action 必须改成
**常驻正确**（`state_dict` 里那份始终是对外正确的值）。而 `ZeroCenteredGammaSync` 常驻 `+1`
会让 trainer 自己的 forward 读到错的 gamma，所以该参数需要一份独立副本。
涉及的参数量极小（layernorm gamma、fp32 pattern 参数、QKV 重排），
注意 `DTypeCastSync`（`megatron_converter.py:176`）与 `ConcatenatedQKVSync`（`:289`）
今天就已经物化了额外张量，本设计不增不减。

**bootstrap 必须新建，不是「本来就有」。** 今天 rollout 用 `load_format="dummy"`
（`vllm_async_server.py:110`），trainer 用 `init_model("empty")`（`ray_trainer.py:2116`），
**唯一读 checkpoint 的是 PS worker**，再经 `broadcast_init` 树形扩散与 `initial_pull_from_ps`
灌给两侧。删掉 PS worker 后改为：**trainer rank 0 读 checkpoint，按 shard 代数分发**，
保留 `broadcast.py` 的树形算法、只换源头角色。不要让两侧各自读共享文件系统，
那会把 `broadcast_init` 当初要解决的「N 个 reader 打爆文件系统」搬回来。

#### 3.3.1 bootstrap 与弹性：三条路径共用 holder 语义

本节是 §3.1 holder 对称性的落地。三个场景，一套机制。

**(a) `is_rollout_mode_in_actor=True` 启动：rollout 自己做源。**

启动时全部 GPU 跑 rollout，trainer 以 `meta` 模式注册
（`ray_trainer.py:2183` 的 `actor_protocol_mode = "meta"`），
`meta_only=True` **跳过真实张量注册**（`client.py:503` 起），
配合 `init_model("empty")` 与 `nixl_sleep("meta")`，
此时 trainer **既无权重数据，也无可被 RDMA 读的注册内存**。

所以 bootstrap 绝不能假定 trainer 是源。做法：

```
rollout 每节点一个代表从 checkpoint 读自己需要的 shard
  -> 注册为 holder，版本号 = v0
其余 rollout 走 §3.5.2 的同版本 peer 扩散拿到 v0
  -> 复用 broadcast.py 的树形拓扑，源头从 PS rank 0 换成 rollout 代表
trainer 后续被唤醒时（switch_to_trainer_mode）
  -> 从 checkpoint 加载训练起点，不从 rollout pull
```

**为什么切回 trainer 要读盘而不是从 rollout pull**：量化路径下 rollout 持有的是
fp8/fp4 权重，**不能作为 bf16 训练的起点**（反量化有损，且 optimizer master params
需要 fp32 精度）。今天这条链路能用 `pull_model`，是因为 PS 持有一份训练精度的副本,
直供后该副本不存在。切回是低频事件（每次 eval 结束一次），读盘可接受,
且这保证了训练起点的精度正确。

同精度路径下从 rollout pull 是安全的，但为了**两条路径共用一套逻辑**，
统一走读盘。若实测读盘成为瓶颈，再为同精度路径单独开 pull 捷径。

排除的两个候选：**每个 rollout 都读盘**会把 `broadcast_init` 当初要解决的
「N 个 reader 打爆文件系统」搬回来，1TB 下不可接受。
**临时唤醒 trainer 分发**与「全部 GPU 跑 rollout」直接冲突，那一刻显存可能不够。

**(b) 非 colocate 启动：trainer 代表读盘，同样走扩散。**
与 (a) 结构相同，只是源头角色换成 trainer rank 0。
**两条路径共用一套代码**，差别只是「谁被选为读盘代表」,
由 index 根据「哪一侧当前是唤醒的」决定。

**(c) 弹性伸缩：三个必须处理的点。**

- **扩容进来的新 instance 冷启动**：从 index 查当前版本 → 作为排队者走 §3.5.2 解析
  （此时必然有 peer，因为其他 instance 已持有）→ 读完注册为 holder。
  逻辑上完全复用已有机制，无需新路径。
- **缩容时必须主动撤销 holder 资格**，不能依赖 §3.5.7 的被动摘除（靠 RDMA 失败）。
  被缩容的 instance 可能正作为 peer 服务他人，被动摘除会让读者先失败一次。
  协议与 §3.4.5 的 trainer sleep **完全对称**：
  先撤销 holder 资格、等在飞读者归零、再 sleep。
- **NIXL 层需要支持增量注册，这是弹性的前置阻塞项。**
  `server.py:56` 与 `:90` 两处都标注
  `TODO(lhy): Support elastic clients after initial registration`,
  且 `_is_all_client_infos_recved` 一旦为真就直接 return。
  对应的 NIXL 能力是 `invalidateRemoteMD` / `loadRemoteMD`（`nixl.h:528` 起），
  当前完全未用。**优先级与 §3.5.5 的元数据扇出同级，且两者必须一起设计**:
  握手从「一次性全量」改成「持续增量」不是加个分支，
  §3.5.5 的指针表也必须支持增量更新，否则弹性与元数据方案会互相冲突。

### 3.4 支柱三：读写冲突，用「发布点」而非「互斥」解决

这是约束 (b) 引入的**唯一新增耦合**，也是本设计最需要讲清楚的一点。

今天 `PSManager` 是单写多读锁：push 独占，期间所有 pull 阻塞
（`_try_acquire_shared_pull_lock` 在 `_exclusive_push_locked` 时直接返回 False）。
这是一个伪装成锁的**全局同步点**，与「rollout 自行拉取」直接冲突。

#### 3.4.1 两个决定性的时序事实

**事实一：一个训练步内权重被改写多次，只有最后一次该被发布。**

`_run_scheduled_update`（`ray_trainer.py:3061`）按 `ppo_mini_batch_size` 与 `ppo_epochs`
构建 schedule 并**循环多次** `update_fn`，每次都是完整的 fwd/bwd/optimizer step。
`push_model` 只在 `finalize_step and is_final_step` 置位（`:3067`）。
所以权重被改写 `num_minibatches × ppo_epochs` 次（典型 4~16），
而 mini-batch 中间态**不是任何合法策略版本**，不该被 rollout 读到。

**事实二：Megatron `DistributedOptimizer` 下 optimizer 写的不是模型参数本身。**

`use_distributed_optimizer: True` 是默认（`engine/megatron.yaml:32`）。此模式下存在
**两份**东西：fp32 master params（optimizer 的 shard）与 bf16 model params。
optimizer step 先更新自己的 fp32 master shard，再通过
`_copy_main_params_to_model_params` + param all-gather 落回 bf16 model params
（PSRL 自己就依赖这个结构，见 `megatron_optimizer.py::sync_master_params_from_model`
调用的 `_copy_model_params_to_main_params`）。

**这个事实把冲突窗口从「整个 optimizer step」缩小到「param 回写那一瞬」。**
optimizer 的数值计算（Adam 的 moment 更新、梯度裁剪、offload 往返）全都发生在
master shard 上，**完全不触碰 rollout 正在读的 bf16 参数**。

#### 3.4.2 回答「optimizer step 过长会不会拖长 rollout 读取延迟」

会，但不是以直觉的方式，而且**有办法把它降到接近零**。先把时间轴摆清楚：

```
一个训练步：
  [mb 1] fwd bwd opt_step   ← 权重被改写，中间态，不可读
  [mb 2] fwd bwd opt_step
   ...
  [mb k] fwd bwd opt_step   ← 最后一次，这之后才是合法版本 v
  ├── 发布点：v 可读 ────────────────────────────┐
  [下一步数据准备 + mb 1 的 fwd/bwd]              │ rollout 在这段窗口内读
  [下一步 mb 1 的 opt_step]  ← 窗口关闭，v 被破坏 ┘
```

**三种延迟来源，必须分开对待：**

| 来源 | 影响 | 是否可消除 |
|---|---|---|
| ① rollout 必须等到最后一个 mini-batch 完成才能开始读 | 发布点延迟，与 optimizer step **次数**成正比 | 不可消除（算法语义决定），但可用细粒度发布缓解 |
| ② rollout 读的过程中被 optimizer 的 param 回写打断 | 需要等待或重读 | **可消除**（见下） |
| ③ trainer 被在飞读者挡住，无法进入下一步 | 训练吞吐损失 | **量化路径下不存在**，同精度路径下由写屏障收窄 |

**①是本质的，②和③才是设计问题。** 而 optimizer step「过长」主要影响的是①，
它的性质是「rollout 要多等一会儿才有新版本」，而不是「rollout 读得更慢」。
这是 staleness 增加，不是带宽损失。

**消除②和③：让被读的字节不是 optimizer 要写的字节。** 三种形态：

**统一解法：唯一的发布点是一块 wire buffer，而它只在真正需要时存在。**

与其按 dtype 组合分出三套语义，不如统一成一条规则：

> **rollout 永远读「发布缓冲」，而发布缓冲是否是一块独立内存，取决于源头是否已经
>天然提供了一份不被 optimizer 改写的字节。**

这给出两种实现，而不是三种语义，且**选择由「有无额外字节产生」决定，不由 dtype 决定**：

| 情形 | 发布缓冲是什么 | 额外显存 | 冲突 |
|---|---|---|---|
| **量化路径**（train dtype != rollout dtype，含 bf16→fp8、bf16→fp4、fp8→fp4） | 量化产物本身，它天然不是 trainer 的参数 | **0 额外**（量化必须有落点，这块落点就是发布缓冲） | 无 |
| **同精度路径**（train dtype == rollout dtype，含 bf16→bf16、fp8→fp8） | trainer 参数本身（直读） | **0** | 有，需放行（见下） |

**关键点：量化路径不存在「额外」显存。** 量化这件事本身就必须有一个输出落点，
无论是否用于发布。把这个落点注册给 NIXL 并当作发布缓冲，**成本是零**，
顺带消除了冲突。所以「`2×chunk` 常驻缓冲」不是为解决冲突付的代价，
而是量化路径本来就要付的代价，这里只是复用它。

**同精度路径不引入任何缓冲**，因为引入一份 fp8→fp8 的拷贝纯属浪费：
它消耗带宽与显存，只为了规避一个可以用放行机制解决的问题。
所以这条路径保留放行逻辑，但**把它收窄到最小**（见 §3.4.2.1）。

**两条配置期约束，属于正确性而非优化：**

1. **FP4 的量化源必须是 optimizer 的 fp32 master params，不是 fp8 model params。**
   `fp8 → fp4` 是二次量化，TE 的 fp8 参数已损失精度，再降到 fp4 会放大误差。
   由 §3.4.1 的事实二，master params 本来就在 optimizer 手里且是 fp32，
   从它量化是单次量化。这要求量化路径能访问 optimizer 状态，
   与写屏障读的是同一块结构，可行。
2. **同精度路径下「chunk 偏移与写屏障互相加剧」的系统性解法。**

   问题根源：§3.5.2.3 的偏移让任意时刻都有某个 chunk 在被读，
   而「在回写 chunk i 之前等 chunk i 的读者清零」这个 per-chunk 写屏障
   会在每个 chunk 上都等一次，最坏情况把整个 optimizer step 变成逐 chunk 的串行等待。

   系统性解法是**把等待点从「per-chunk 回写前」提到「version 发布前」**，做一次性等待：

   ```
   optimizer step 完成，准备发布 v+1 之前：
     等待所有对版本 v 的在飞读者计数归零（一次性，不分 chunk）
     -> 此时所有 chunk 均无读者，trainer 参数内存可自由写
     -> 发布 v+1，holder 更新为 trainer(v+1)
   trainer 执行下一步 backward / optimizer step 期间：
     rollout 开始拉取 v+1，与 trainer 的 backward 完全并行
     （读 v+1 的参数，而 optimizer step 写的是 fp32 master shard，两者不冲突）
   ```

   这把 `O(chunks)` 个等待点合并成一个，与偏移的有无完全解耦。
   因此**偏移在量化路径和同精度路径均可启用**，不需要按路径关闭。
   等待时间是「读者完成最慢的那个 chunk」，这与偏移是否启用无关
   （偏移改变的是哪个读者在哪个 chunk 上，不改变最慢读者的完成时间）。

   这与 §3.4.2.1 的描述是一致的——只是把写屏障从「per-chunk 回写前」
   统一提到了「version 发布前」，粒度从 chunk 级变成了 version 级。

**colocate 下同精度是完全合法的，见 §3.7.1 的时序互斥分析。**

这样做的收益是消掉了「三种形态」这个坏味道：
**配置维度从「dtype 组合」变成一个布尔量（源头是否产生新字节）**，
测试矩阵从 3 降到 2，而其中量化路径的分支几乎没有逻辑。

#### 3.4.2.1 同精度路径的放行：窄到只剩一个写屏障

同精度直读下，冲突窗口由两个已核实的事实决定，比直觉窄得多：

1. **Megatron `DistributedOptimizer`（默认开启）的 optimizer step 写的是 fp32 master
   shard，不是 bf16 model params。** 回写由 `_copy_main_params_to_model_params`
   加 param all-gather 完成（PSRL 自己依赖这个结构，见
   `megatron_optimizer.py::sync_master_params_from_model`）。
2. **optimizer offload 也只搬 master shard。** `offload_megatron_copy_params`
   （`megatron_utils.py:728`）只处理 `shard_fp32_from_float16_groups`,
   **完全不碰 model params**，所以 offload 与 RDMA 读天然无关。

于是放行只需要一个写屏障，不需要租约子系统：

```
trainer 在执行 param 回写之前：
  对即将回写的那段 chunk，检查「是否有在飞读者」
    无  -> 直接写（绝大多数情况，因为读者从发布点就开始读了）
    有  -> 等待，上限 conflict_wait_budget_ms
           超限则置该 chunk 为 INVALID 并继续写
             在飞读者收到 INVALID，放弃本轮，重拉更新版本
```

「在飞读者」是一个 **per-chunk 的计数器**，不是版本化租约表：
读者开始读时 +1、结束时 -1，由 index 维护但 trainer 可本地缓存。
**这比原方案的「版本 × chunk 租约表」简单一个量级**，
因为同精度路径下不需要回答「谁预约了哪个版本」,只需要回答「这段现在有人在读吗」。

#### 3.4.3 取舍原则：等待的上限由算法目标决定，不由 infra 方便决定

这是本节最重要的一条，也是回答「应该如何取舍」：

> **trainer 等待 rollout 读完，是用确定的吞吐换确定的 on-policy 程度。
> rollout 读旧版本，是用不确定的算法质量换吞吐。前者可测量可预算，后者不可。
> 所以默认取向是前者，但必须有界。**

具体规则：

1. **等待有硬上限**（`conflict_wait_budget_ms`，建议默认为单步时长的 5%）。
   超限则 trainer 强制推进，被打断的 rollout 收到明确的「版本已失效」而**重拉新版本**,
   绝不是「继续用旧的」。因为重拉的是**更新**的权重，这在算法上是改善而非退化。
2. **绝不为了避免等待而放宽 staleness。** 这是最容易犯的错：把 `staleness` 调大
   让 rollout 读旧版本，表面上消除了等待，实际是把 infra 问题转嫁成算法质量损失，
   且表现为 reward 曲线变差、极难归因。
3. **优先用细粒度发布换 on-policy，而不是用 staleness 换吞吐。** 见下。

#### 3.4.4 从算法角度：细粒度发布让 rollout 拿到更新的权重

延迟来源①（rollout 要等最后一个 mini-batch）是可以用**更细的发布粒度**缓解的，
而且这顺带降低 off-policy：

PSRL 已有 `fine_grain_overlap` 这个 StepStrategy（`psrl.fine_grain_overlap`），
它把训练步切成 chunk 以与 rollout 重叠。在量化路径下，可以在**每个 mini-batch 后**
都量化并发布一个中间版本，让 rollout 拿到更新的策略。

**但这必须由算法决定，infra 只提供开关。** mini-batch 中间态是否是合法采样策略，
取决于算法：GRPO/GSPO 的 importance sampling 修正项只需要知道「采样时的 logprob」，
所以中间态在数学上定义良好，但会增加策略方差。
建议做成 `publish_granularity: step | minibatch`，默认 `step`，由算法侧实验决定。

另外一条**免费的 on-policy 收益**：今天的链路要 `push(N×M)` 全部完成才登记版本，
直供下 trainer 登记后 rollout 立刻可读，省掉的正是 `N×M / B_train`。
N=32 时这不是小数。**少一跳就是少一份 staleness**，这是约束 (b) 带来的意外红利。

#### 3.4.5 内存注册有效性：留在 trainer 本地，不交给 PS

初稿要求 index 维护一个「trainer 权重可读」状态位，让 PS 感知 trainer 的内存生命周期。
**这是抽象泄漏**：PS 本该只管版本与位置，不该追踪别人何时释放内存。
重新评估后，这件事可以完全留在 trainer 本地，依据是三条已核实的事实：

1. **optimizer offload 不碰 model params。** `offload_megatron_copy_params`
   （`megatron_utils.py:728`）只搬 `shard_fp32_from_float16_groups`。
   所以 offload 根本不是风险来源，初稿把它列进去是多余的。
2. **真正会让 rkey 失效的只有 sleep / wake 路径**：`_sleep_megatron_model` 的
   `param_data.untyped_storage().resize_(0)` 与 `_sleep_fsdp_model` 的同类操作，
   之后必须 `nixl_register_after_wake_up` 重注册。
3. **这条路径今天只在 colocate 切换时触发**（`ray_trainer.py:2573` 的
   `nixl_wake_up`），而它本来就是一个**有明确起止的受控区间**，不是随时可能发生的事件。

因此正确的做法是**把窗口做成不可观测**，而不是把窗口状态发布出去：

```
trainer 侧（本地，无需 PS 参与）：
  进入 sleep 前：撤销自己作为 holder 的资格（一次 index 调用，与版本发布同一通道）
                 等待在飞读者归零（计数器已有，见 §3.4.2.1）
  resize / 重注册完成后：重新注册为 holder
```

**于是「内存注册有效」不再是一条需要持续维护的不变量，而是 holder 注册/注销的
一个自然结果**，不是 holder 就不会被指派，不会被指派就不会被读。
PS 只需要知道「谁是 holder」这一件事，而这是它本来就要维护的。

两点必须说清楚的边界：

- **这依赖「撤销 holder 资格」与「实际 resize」之间的严格顺序**。若顺序颠倒，
  窗口依然存在。所以撤销必须是**同步**调用并等待在飞读者归零，不能 fire-and-forget。
  代价是 colocate 切换多一次 round trip，而切换本来就是重量级操作，可忽略。
- **非 colocate 的稳态下这条路径根本不触发**，所以稳态零开销。

剩下的唯一不变量是**版本不被覆盖**，由 §3.4.2.1 的在飞读者计数器 + 写屏障保证。
从两条不变量减到一条，且剩下那条的实现是一个计数器而非一张表。

#### 3.4.6 不变量必须是运行时断言，不是文档要求

本文正确性大量依赖若干「必须遵守」的约定。文档里的要求无法阻止实现漂移，
所以下列四条必须落为**运行时断言**，在违反时立即失败而非静默错误：

| 不变量 | 断言点 | 违反后果（若无断言） |
|---|---|---|
| **atomic group 完整性** | 标记版本可见前，校验 group 内全部 key 已到达 | 引擎 loader 静默 no-op，权重半更新 |
| **版本相等性** | index 指派 peer 时，校验 `version(peer) == v` | 混合策略，算法侧不可见 |
| **peer 非链式** | index 指派时，校验 peer 的 `from_trainer` 位为真 | 长链延迟累积，上游覆盖导致下游读到脏数据 |
| **holder 即注册有效** | pull 准入时校验 holder 身份，sleep 前校验在飞读者归零 | RDMA 读到已释放物理页，静默数据损坏 |

前三条是纯元数据校验，开销是常数级，**应当常开**。
第四条的「在飞读者归零」在 colocate 切换路径上，频率低，也应常开。

另有两条属于配置期校验，不进热路径：
FP4 下 `is_weights_pre_processed` 必须为假（§3.8.1）、
block 对齐必须通过 `validate_fp8_block_shape`（§3.8）。

### 3.5 支柱四：全局去冗余规划，pull 语义下

`_make_comm_plan_generic` 是对每个 (target, key) 独立贪心（link priority 优先、
再按累计 volume 平衡）。结构性局限：逐 key 决策看不到全局负载，无法表达
「先去重、再在接收侧互换」这种两跳方案。而没有这个能力，trainer 直供就会撞上
**miles P2P 的单源瓶颈**：所有 rollout 都从同一批 trainer rank 读，rollout DP 一涨就线性退化
（WeightBridge 实测 16 replica 下差 34×）。

新 planner 产出 WeightBridge 的三阶段计划，建立在 PSRL 已有的 shard 代数上：

1. **receive**：两侧逻辑去重。trainer 侧多副本（DP replica）各领一份不相交切片，
   使 trainer 合计只发一份。rollout 侧多副本也各自只领一份切片。many-to-many 塌缩成 one-to-one。
2. **external-exchange**：rollout 节点间 RDMA 互换，使每节点恰好拿到它需要的全集一份。
3. **internal-exchange**：节点内走 NVLink 补齐各 worker 的副本。

目标是流量最优，对齐下界

```
T_min = max( M / B_train ,  max_v ( M_v / B_v ) )
```

`M` 为模型大小，`B_train` 为 trainer 聚合出口带宽，`M_v` / `B_v` 为 rollout 节点 `v` 的
去重后需求量与入口带宽。

**pull 语义的适配**（WeightBridge 没有覆盖的部分）：它的 plan 由 trainer 发起、
覆盖固定 worker 集合，而 PSRL 要的是每个 instance 自行拉取。做法是把 plan
**按 instance 切分、握手期编译一次**：

- planner 在握手期为每个 rollout instance 编译「它的三阶段 plan」，存在 index 里。
- instance 拉取时拿到自己那份独立执行，**不等全局 barrier**。
- external-exchange 阶段需要 peer，通过 index 查询「谁已持有 v」动态解析，
  而不是预设一个固定的交换伙伴。先到的 instance 从 trainer 读，后到的可以从 peer 读，
  **这同时就是 trainer 出口的天然减压**（SkyRL 已验证 NIXL 读单边非破坏、可被多方共享）。

这一条是 trainer 直供能否成立的关键：它把「所有 rollout 都压 trainer」变成
「trainer 只发一份，其余在 rollout 侧扩散」。

#### 3.5.1 greedy sync 与 peer 扩散的真实关系

先更正一个说法。初稿把 greedy 描述为「sync all instances ASAP，所有 instance 同时压 trainer」，
查 `greedy.py::_greedy_sync_and_migrate_loop` 后发现**它本身已带有自然的分批效果**：

- 循环每 `check_interval_in_ms`（默认 100ms）扫一轮，收集「版本落后于 PS」的 instance。
- `sync_with_ps` 一进入就把这批 instance 标记
  `instance_to_latest_stale_model_version = 当前版本`（`sync_and_migrate_mixin.py:148`）。
- 下一轮扫描时，**正在同步中的 instance 命中 `<=` 判断被跳过**（`greedy.py:48`）。
- 且 `partial_rollout.enable=False` 时还要求 `check_no_activate_tasks`，
  即只有手上没活的 instance 才会被纳入本轮。

所以真实行为是：**第一轮纳入「此刻恰好空闲且落后」的那些，后续轮次纳入陆续空闲的**。
instance 之间因负载与请求完成时间不同而天然错峰。**greedy 不等于同时发起。**

那么真正的风险是什么？不是「必然同时」，而是**「可能同时，而设计不能依赖它不同时」**：

- 版本发布是一个**同时到达**的事件（PS 推给 coordinator），所以第一轮扫描时
  所有空闲 instance 会在**同一轮**里被纳入同一个 `sync_with_ps` 调用。
- `sync_with_ps` 内部对这批是 `asyncio.gather` 并发 pull，没有串行化。
- 极端但现实的情况：训练步较长、rollout 已全部跑完在等新权重，此时发布版本
  → 全部空闲且落后 → 一轮全部纳入 → 并发压 trainer。
  **这恰好是 colocate eval 之后、以及 staleness=0 强同步配置下的常态。**

#### 3.5.2 解法：扩散由 index 强制，且 peer 必须是同版本

正确的分层是：**sync 策略决定「谁该更新」（算法语义），index 决定「从哪读」（传输语义）。
后者不能依赖前者的时序巧合。** 扩散由 index 在解析源时强制形成，无论上游是否同时发起。

**先确立一条硬约束（版本相等性）：**

> **只有当 peer 持有的版本与 trainer 当前发布版本严格相等时，它才可以作为读取源。**
> 不允许从「持有 v-1 的 peer」读，哪怕它在拓扑上更近。

理由有三层，从强到弱：

1. **算法正确性**：从 `v-1` 的 peer 读等价于主动引入一个单位的 staleness，
   而这个 staleness 是**传输层偷偷加的、算法侧不知情的**。它会污染
   importance sampling 的前提（算法以为采样策略是 v）。这与 §3.4.3 的取向一致：
   infra 绝不以降低 on-policy 为代价换吞吐。
2. **版本混杂风险**：若一个 instance 的部分 chunk 读自 v 的 trainer、部分读自 v-1 的 peer，
   得到的是**一个不存在的混合策略**，比单纯的 staleness 更糟，且无法被任何修正项补偿。
3. **实现简单性**：版本相等使 holder 位图可以**按版本整体失效**（见下），
   不需要维护「每 chunk 的版本号」这种细粒度状态。

落实到数据结构上，这条约束让实现**更简单而非更复杂**：

```
holder_bitmap 只为「当前发布版本 v」维护一份，版本推进即整体清空
  trainer 发布 v   -> 清空上一版位图，仅 trainer 为 holder
  instance 读完 v  -> 置位它的 bitmap，成为合格 peer
  trainer 发布 v+1 -> 整体清空，所有 instance 退回非 holder
```

副作用是**每个新版本的第一波必须从 trainer 读**，扩散不能跨版本累积。
这是正确性的必要代价，而 §3.5.2 的 W 机制正是为了让这第一波不至于压垮 trainer。

**选源规则（统一架构，同节点先落地）：**

```
index 解析每个 (instance, chunk) 的源，候选集 = {trainer} ∪ {version(peer) == v 的 holder}：
  1. 同节点 peer（NVLINK / PCIE）     -> 第一优先，不占 NIC
  2. trainer（IB）                    -> 在飞量 < W 时
  3. 跨节点 peer（IB）                -> 仅当 trainer 在飞量 >= W   [阶段二启用]
  4. 以上皆不可用                      -> 排队，等前面完成后重新解析
                                          （此时同节点或跨节点已出现合格 peer）
```

**阶段一只开 1、2、4，阶段二再开 3。** 架构上两者是同一条解析链路上的不同优先级,
区别仅在候选集过滤条件（`same_node(peer, requester)`）与排序权重，
**不是两套机制**。所以阶段二不需要改接口，只需放开一个开关并把节点身份
从 IP 换成 Ray node id（见 §3.5.3a）。

需要补的一处现有行为：`sync_with_ps` 目前在 pull 之前就对整批 instance
`pause_for_sync`（`sync_and_migrate_mixin.py:155`）。若扩散把一批 pull 拉长成几轮，
**先暂停的 instance 会空等更久**。所以暂停必须下移到「该 instance 真正要开始读」的时刻，
而不是整批统一暂停。这是一处实际改动，初稿没提。

#### 3.5.2.1 W 由什么决定

上一轮我把 W 说成「`W × 单读带宽 ≈ trainer 出口带宽`」，这个说法**不准确**，
因为它把 W 当成了限流阈值。想清楚之后，W 的本质是另一回事：

> **W 不是「trainer 能承受多少并发」，而是「在 trainer 只发一份数据的前提下，
> 第一波应该由几个 instance 来分摊接收」。它是去冗余的分组大小，不是限流阀门。**

这个区别很重要。NIXL 的读是**单边的**：rollout 发起 READ，trainer 侧是被动 RDMA target，
不消耗 trainer 的 CPU（`enable_prog_thread` 只影响发起方的进度推进）。
所以 trainer 并不会因为「并发读者多」而忙不过来，真正的约束是**它那张 NIC 的总带宽**。

于是 W 的取值由三个量决定，取其最小：

| 约束 | 表达式 | 含义 |
|---|---|---|
| ① NIC 不超卖 | `W ≤ B_egress / B_per_reader_useful` | 超过这个数，每个读者都变慢，总时间不变但 p99 变差 |
| ② 扩散收益存在 | `W < R` | `W ≥ R` 意味着全体直读源，扩散退化为无 |
| ③ 轮数可接受 | `W ≥ R^(1/k_max)` | 限制扩散轮数不超过 `k_max`，每轮都有同步与控制面开销 |
| ④ **QP 预算** | `W × ranks_per_reader ≤ QP_budget` | 单边读仍需 NIC 维护 QP 状态。万卡下 W=8 × 2000 rank = 1.6 万 QP，**可能超出 NIC 的 QP 缓存导致性能悬崖** |

**不要去测 QP_budget。** 不同 NIC 的 QP 缓存容量差异大，且难以静态确定。
正确做法是**让它被 W 的反馈信号自动兜住**：QP 超限表现为单读带宽的悬崖式下降，
而 W 的实测反馈（本节下文）本来就在监测单读带宽。
所以约束 ④ 不是一个要计算的公式，而是**「W 上调时必须验证带宽未劣化」这条规则的理由**。

**①是硬上限，③是软下限，实际取值在两者之间。**

更实用的是把 W 表述为「**trainer 出口的饱和点**」而非一个配置数字：

```
W_effective = max(1, floor(B_trainer_egress_available / B_per_reader_target))
  B_trainer_egress_available：trainer 节点 NIC 可用带宽
    注意要扣掉 trainer 自身的 collective（DP all-reduce / param all-gather）占用,
    这些与权重读取抢同一张 NIC
  B_per_reader_target：单个读者的目标带宽，由「可接受的单次拉取时长」反推
    = chunk_bytes / target_chunk_latency
```

三条实践建议：

1. **W 应当是自适应的，而不是静态配置。** trainer 的 NIC 余量随训练阶段变化
   （backward 的梯度 all-reduce 期间余量少、optimizer 数值计算期间余量多）。
   index 可以用「上一版本每个读者的实测完成时间」反馈调整 W：
   完成时间劣化即说明超卖，下调 W。这个反馈信号是免费的，租约释放时本来就能带上耗时。
2. **初始值取 `W = 节点内 instance 数`**，即「每个 rollout 节点派一个代表去 trainer 读」。
   这个默认值有清晰的物理含义：跨节点流量恰好是每节点一份（WeightBridge 的第二原则），
   节点内其余 instance 走 NVLink 从代表那里扩散。**这同时自动满足了阶段一只用同节点 peer 的设定。**
3. **W 的上限受传输窗口复用约束**（§3.6.1）。量化路径下 wire buffer 是流式复用的，
   `W` 个读者必须都读完 chunk i 才能装 i+1。所以 W 过大反而会让 chunk 周转变慢。
   这是另一个「W 不是越大越好」的理由。

**不应该用来决定 W 的东西**：rollout instance 总数（那是 R，不是 W）、
GPU 数量、NIXL 线程数。这些与 trainer 的出口带宽没有直接关系。

#### 3.5.2.2 W 跟随训练阶段，而非只靠实测反馈

§3.5.2.1 把 W 定为「trainer 出口的饱和点」，并提了用实测完成时间反馈调整。
但纯反馈是**滞后**的：它要等一个版本的读取劣化之后才下调。
而 trainer 的 NIC 余量其实是**可预知**的，因为它由训练阶段决定：

| 训练阶段 | trainer NIC 占用 | W 应当 |
|---|---|---|
| forward | 低（仅 TP all-reduce，多在 NVLink 内） | 放大 |
| backward | **高**（grad reduce-scatter 跨节点） | 趋近下限 |
| optimizer 数值计算 | 低（master shard 本地计算，offload 走 PCIe 不走 NIC） | 放大 |
| param 回写 + all-gather | **高** | 趋近下限，且此时写屏障生效（§3.4.2.1） |

落地方式复用已有结构，不新增机制：

- `STAGE_META`（`strategies/base.py:22`）已经把阶段分类为
  `per_sample` / `batch_coupled` / `optimizer_step`，**阶段标记的概念已存在**。
  只需让 train worker 在进入/离开各阶段时向 index 上报一个枚举值。
- **只由 DP rank 0 上报**（阶段在 DP 组内是同步的）。若每个 rank 都报，
  万卡下 2000 rank × 16 mini-batch × 2 = 6.4 万条／版本，**会突破 §3.2 的量级承诺**。
  只由 DP rank 0 报则是 `O(minibatches)`，可忽略。
- **阶段标记必须有 TTL**，超时回退 `W_base`。否则 trainer 崩在 bwd 阶段时，
  index 会永远认为 W 应该很小。
- index 侧 `W = W_base × stage_factor`，`stage_factor` 是一张三值查找表
  （如 `{fwd: 1.0, bwd: 0.25, opt: 1.0}`），**不做连续建模**。

两者是互补而非替代：**阶段标记给前馈（预知大势），实测反馈给校正（修偏差）**。
前馈避免滞后，反馈兜住「阶段分类不准」或「NIC 被其他流量占用」的情况。

**但阶段分档偏粗，更准确的信号是 NCCL 在飞状态。** `overlap_grad_reduce` 开启时
grad reduce 与 backward 交织，不存在「干净的 bwd 阶段」。所以 `stage_factor`
应当按「是否有 NCCL 集合通信在飞」判定，信号源是 Megatron 的
`grad_sync` / `param_sync` 回调，而非训练阶段枚举。
NCCL 与 NIXL 走不同流控（一个 NCCL channel、一个 UCX），**互相看不见对方，
只会简单争抢 NIC 带宽**，所以这个信号是必要的。

**一个由此派生的优化：把版本发布挂在 param all-gather 完成的回调上**（已查证可行，
见下），
而不是「optimizer step 返回后」。那一刻既是权重就位的时刻，也是 NIC 刚空闲的时刻,
**既减少 staleness，又让 rollout 的读取精确填进 NCCL 的空档**。
这比持续上报阶段更精确，成本也更低（一个回调而非周期消息）。

**Megatron-Bridge 的 `on_train_step_end` 已查证可用**
（`train.py:518`，`callbacks.py:VALID_EVENTS`）。它在 optimizer step
（包括 param all-gather）完成后触发，`CallbackContext` 携带 `optimizer` 与 `model`，
足以在此时刻触发版本发布。注册方式是向 `pretrain()` 传入自定义 `CallbackManager`，
PSRL 的 Megatron worker 已持有 `pretrain()` 的调用链，接入成本低。

一条必须注意的边界：**阶段上报不能成为同步点**。它是 fire-and-forget 的单向通知，
index 收不到就沿用上一个值。绝不能让 trainer 等待 index 确认阶段切换。

#### 3.5.2.3 chunk 起始偏移错开，让扩散在第一波内部发生

§3.5.2 的 W 机制让第一波只有 `W` 个 instance 直读 trainer，其余排队。
但若这 `W` 个 instance 都**从 chunk 0 开始按相同顺序推进**，它们会同时持有 chunk 0、
同时持有 chunk 1，排队者要等到某个 instance **整轮读完**才能找到 peer。

改进是零成本的：**让 instance `i` 从 chunk `(i × num_chunks / R) mod num_chunks` 开始，
环形推进**。于是：

- 第一波的 `W` 个 instance 在任意时刻持有的是**不同的 chunk 集合**。
- 排队者一旦被放行，立刻就能为它需要的大部分 chunk 找到合格 peer，
  **不必等任何 instance 整轮完成**，扩散在第一波内部就开始了。
- 对 trainer 出口也更友好：并发读者命中的是不同地址区间，减少同一块内存的争用。

三点约束：

1. **偏移必须与 commit group 边界对齐**（§3.6.1），不能把一个 group 切在环形起点，
   否则该 group 的可见性判定要跨环回绕，变复杂。
2. **偏移是确定性的**（由 instance 序号导出），不是随机，这样 index 可以在不通信的
   情况下预测谁会先持有哪些 chunk，用于选源。
3. 这项优化**只在 `R > W` 时有意义**，`R <= W` 时全体直读 trainer，无需扩散。

### 3.5.3 拓扑感知选源：可行性、开销与可扩展性

**结论：可行，但必须限定在「节点级、离散分级、不做实时带宽估计」这个范围内。** 逐项评估。

**(a) 拓扑数据已经有了，但精度有限。**

`network_topology.py` 已实现 `GLOBAL_TOPOLOGY`，按 `(node_ip, gpu_id)` 判定链路类型并给出
`get_link_priority()`（`LOCAL > NVLINK > IB > PCIE > IB_PCIE > ETH`），comm plan 今天就在用它。
所以**不需要新建拓扑子系统**，但必须认清两个局限：

- **带宽是硬编码常量**（`_init_default_topology` 注释写明是「8*H20 机器的默认配置」）。
  它能可靠回答「同节点还是跨节点」，**不能**可靠回答「两个跨节点 peer 哪个更快」。
- **节点身份依赖 `LOCAL_NODE_IP` 或 `gethostbyname`**，而 `LOCAL_NODE_IP` 在仓库里
  只有定义、没有任何地方设置。多网卡 / 容器网络下 `gethostbyname` 可能返回与 RDMA 实际
  走的网卡无关的地址，使同节点被误判为跨节点（后果是选源变差，不是出错）。
  **阶段一就应把节点身份换成 Ray node id**（`get_node_id()` 已有，比 IP 可靠），
  这也是阶段二开启跨节点 peer 的前置条件。

**(b) PS 侧开销：holder 位图，不按 key 展开。**

朴素的 `(v, key, shard) -> [holder]` 在 1T MoE + 数十 instance 下是千万级条目，不可接受。
正确结构是按 instance 聚合的位图，且因 §3.5.2 的版本相等性约束，**只需为当前版本维护一份**：

```
holder_bitmap : dict[instance_id, Bitmap(num_chunks)]   # 仅当前版本 v
node_index    : dict[node_id, set[instance_id]]          # 同节点 peer 查询
in_flight     : int                                      # trainer 当前在飞读者数，用于 W
```

量化：1TB 模型 / 512MB bucket ≈ 2000 bucket，位图 250 字节／instance，
64 instance 共 **16KB**。加节点索引与在飞计数，PS 的选源状态总量在 **几十 KB 量级**，
与模型大小几乎无关（只随 `M / chunk_size` 线性）。

查询代价：「同节点是否有合格 peer 持有 chunk i」= 对该节点 instance 集合做位图 OR，
`O(同节点 instance 数)` 的位运算，微秒级，**不遍历 key**。
版本推进时整体清空，`O(R)`。

**(c) 消息开销：不新增往返。**

holder 更新**搭在已有的租约释放调用上**携带 bitmap（§3.2 的按 instance 聚合）。
稳态消息量 `O(instances)` 每版本，与 chunk 数、key 数无关，仍在 §3.2 承诺的量级内。

**(d) 可扩展性评估。**

| 规模维度 | 影响 | 失效点 |
|---|---|---|
| instance 数 R | 位图 `O(R)`、查询 `O(R_同节点)`、消息 `O(R)` | R 到数百时瓶颈是 PS 单进程消息处理，与拓扑选源无关（§3.2 的控制面问题） |
| 模型大小 M | 位图随 `M/chunk_size` 线性 | 10TB 模型下 2.5KB／instance，仍可忽略 |
| chunk 粒度 | 变细则位图变大、放行更精细 | chunk < 64MB 时位图开销与收益不再划算 |
| 节点数 | 节点索引 `O(nodes)` | 无实际限制 |

**真正的可扩展性风险不是数据结构，是两个语义问题：**

1. **peer 供给会抢 rollout 自己的推理资源。** 一个 instance 为他人提供读取时，
   NIXL progress thread 占 CPU、RDMA 占它自己的 NIC，而推理是延迟敏感的。
   必须有上限：每个 peer 同时最多服务 `P` 个读者（建议 1~2），超出则不再被指派。
   **colocate 模式下直接禁用 peer 供给**（它与 eval 争同一张卡）。
2. **「已持有」不等于「可安全被读」。** 量化路径的 wire buffer 是流式复用的，
   peer 的 buffer 下一轮会被覆盖。所以 peer 作为源时**也需要租约**，否则读者会拿到
   半新半旧。这把租约从「trainer 的」扩展为「任意 holder 的」，概念上是扩展，
   实现上是同一张表。注意这与 §3.5.2 的版本相等性是两件不同的事：
   版本相等保证「内容是 v」，租约保证「读的过程中不被覆盖」。

**(e) 落地建议：两阶段，架构一致。**

- **阶段一**：候选集过滤为「同节点 + 版本等于 v」，开启 trainer 并发上限 W
  （默认取每节点一个代表，见 §3.5.2.1），节点身份换成 Ray node id。
  这已拿到两个最确定的收益：同节点读走 NVLink 不占 NIC、第一波扇出被 W 收窄。
- **阶段二**：放开跨节点 peer 作为 trainer 饱和后的兜底。
  **架构上只是去掉候选集的 `same_node` 过滤并加一档排序权重，不是新机制。**
  收益随 R 增大才显现，应在实测到 trainer 出口饱和后再开。

**不建议做的**：实时带宽探测、基于 RTT 的动态选源、全局最优流量规划。
带宽是常量，这类优化会建立在不可靠输入上，且让 PS 承担它不该有的实时性要求。

#### 3.5.3.1 rail 拓扑感知

万卡集群普遍采用多 NIC / 多轨道（rail）组网：每节点有多张 HCA，
每张接入不同的网络平面，同 rail 的跨节点流量不共享交换机，带宽更可预测。

`NetworkTopology` 目前只有 `(node_ip, gpu_id)` 两个维度，没有 rail 概念。
UCX 有 multi-rail 支持（`UCX_IB_DEVICES`），所以数据面本身可能已受益，
但选源决策看不到它。

落地方案（1-2 天工作量）：

1. `NetworkTopology.register_client` 额外接收 `rail_id`（整数），
   由 worker 启动时通过环境变量 `PSRL_RAIL_ID` 或 `ibstat` 查询上报。
2. `get_link_priority` 新增一档 `IB_SAME_RAIL`，优先级介于 `NVLINK` 与 `IB` 之间：
   同节点 > 同 rail 跨节点 > 跨 rail 跨节点。
3. 三阶段 planner 的 source 侧分配优先选**同 rail 上有副本的 trainer rank**，
   让 W 个并发读者里同 rail 流量天然占优。

收益在万卡下显现，小规模部署下无影响。

#### 3.5.4 禁止链式 peer

peer 必须是**直接从 trainer 读到的那一批**。holder 位图额外记一位
「来源是 trainer」，只有该位为真的 instance 才可被指派为源。

理由：若允许「A 读 B，而 B 自己读的是 C」，在 W 较小时会形成长链，
延迟逐跳累积，且一旦链中某个 instance 开始下一轮覆盖自己的 buffer，
下游全部失效。限制为单跳后，最坏延迟是 `trainer→peer` 加 `peer→self` 两段，
与 WeightBridge 的 external-exchange 完全一致，不会退化。

### 3.5.5 元数据平面必须先去「按 key 展开」

**这是大 MoE 与万卡场景的前置阻塞项，优先级高于本文其余所有性能优化。**

PSRL 的 canonical 命名空间是 HF 展开形式，MoE 的每个 expert 的每个投影都是独立 key。
Kimi-K2（1T，384 experts，61 层）量级：

| 形式 | 张量数 |
|---|---|
| vLLM fused（`w13_weight` / `w2_weight` per layer） | **122** |
| HF 展开（gate / up / down per expert） | **70,272** |
| HF 展开 + fp8 scale 成对 | **140,544** |

而握手期 `send_local_info`（`client.py:861`）把整个 `client_info`（含全部
`tensor_infos`）pickle 后经 `agent.send_notif` 发出，server 再按 agent 扇出
（`server.py:257`）。按每 key 约 400 字节估，**单个 client_info 就是 56MB**，
1024 client 规模下是数十 GB 的 pickle 流量与等量反序列化 CPU。
`comm_plan` 同样按 key 展开（`comm_plan.py:22`），量级相同。

**注意直供会加剧而非缓解这一点**：rollout 的对端从「本节点一个 PS worker」
变成「所有 trainer rank」，peer 读取又让 rollout 之间互为对端，
`_get_relevant_client_names_for_agent` 的过滤效果变差。

三条措施：

**(a) canonical 层面做 MoE 分组（根因治理，必做）。**
把 `experts.{i}.gate_proj.weight` 合并为 `experts.gate_proj.weight` 一个 key，
expert 维度表达为 `NIXLSharding` 的一个分片维度（`shard_mesh` 是
`OrderedDict[dim, count]`，本来就支持多维）。Kimi-K2 的 key 数从 14 万降到数百。

这更贴近引擎真实布局：vLLM 的 fused loader 本来就要 `w13_weight` 这个 3D 张量，
今天 PSRL 是主动把它拆开（`vllm_converter.py:400` 起的
`FUSED_MOE_W13_SPLIT`），拆开反而是额外工作。
风险点是 `ETP != 1` 时 expert 内还有 TP 切分，shard mesh 成为
`{0: EP, 1: ETP}` 两维，需验证 `find_finest_shard_mesh` 的 lcm 逻辑在两维上正确。

两处连带影响，都由 §3.6.1 的 chunk / commit group 分离自然吸收：

- **commit group 粒度变粗**。vLLM 的 fused loader 要求整个 `w13_weight` 就位，
  所以 group 从「一个 expert 的三个投影」变成「整个 fused 张量」。
  因为 chunk 与 commit group 已解耦，一个 group 跨多个 chunk 是常态而非特例。
- **单 key 体积变得极大**。Kimi-K2 单层 `w13_weight` 在 fp8 下约 11GB，
  所以 chunk 必须能切开单个 key，这正是 §3.6.1 给 chunk 的定义。

**(b) 描述符用 Nx5 strided 表达。** expert 维度分片通常是规则跨步的，
一条 strided desc 替代 `count` 条普通 desc。这与 §3.6 的 strided 改造是同一件事，
此处确认它还有元数据压缩收益。

**(c) 万卡规模再做：元数据走 RDMA 按需拉取。** 每个 client 把自己的
`client_info` 注册为可被 RDMA 读的内存，server 只分发「谁的 info 在哪」的指针表
（KB 级），需要时按需读。把 `O(agents²)` 的 pickle 扇出变成 `O(agents)` 指针表。

**另一处零风险改动**：`make_xfer_tag`（`client.py:42`）用 `pickle.dumps`
为每个传输生成唯一标识。14 万 key 规模下是 14 万次 pickle 的纯 CPU 开销，
换成字符串拼接或 `struct.pack` 即可。

### 3.5.6 轮数与 chunk 大小：轮级决策必须本地化

1TB 模型用 512MB chunk 得到 **2000 轮**。若每轮有任何跨进程往返
（查询源、确认完成），`2000 × RTT` 就是秒级纯开销。所以：

- **源选择在拉取开始时一次性解析完整轮次**，而非每轮查一次。
- plan 本地缓存（§3.2a）、完成通知批量上报（§3.2b），二者已规划。
- **chunk 大小应随模型规模上调**：1TB 用 4GB chunk 得到 250 轮更合理。
  约束是「`2×chunk` 常驻 wire buffer 可接受」与「轮数 × 控制面开销可忽略」的折中。

### 3.5.7 trainer 单 rank 故障：DP 副本是天然的降级目标

万卡下 trainer rank 数达数千，单 rank 故障概率不可忽略，而直供去掉了 CPU 副本，
看似没有降级目标。**但实际上有，而且是免费的。**

**关键事实：train 侧的 sharding 声明不含 DP 维度。**
`megatron_converter.py::get_sharding_for_param` 只按 TP / ETP 维度声明分片
（`shard_indices = [(self.mpu.tp_rank,)]` 或 etp_rank），
**DP 维度完全不出现在 `NIXLSharding` 里**。这意味着同一个 `(key, shard)`
在 DP 组内有 `dp_size` 个**内容完全相同**的副本，分布在不同 rank 上。

这正是 §3.5 的三阶段规划第一步「source 侧多副本各领一份不相交切片」所利用的东西：
平时它让 `dp_size` 个 rank 分摊出口（去冗余）。**故障时它就是冗余来源。**

#### 三层响应，按故障范围递进

```
① 单 rank 故障（最常见）
   index 把该 rank 从 holder 表摘除
   -> 它负责的 chunk 由同 DP 组的其他 rank 接管（内容相同，无需重算）
   -> 受影响的 pull 重新解析源，继续读
   对 rollout 不可见，对训练无影响（Megatron 自身的容错另论）

② 整个 DP 组故障（同一 chunk 的所有副本都没了）
   该 chunk 无合格 holder
   -> index 返回 CHUNK_UNAVAILABLE（明确错误，不挂起）
   -> rollout 放弃本轮，保持当前版本继续生成，等下一个版本
   这是「等下一个版本」而非「卡死」，也不是「读旧版本」:
   它用的是自己已有的旧权重，staleness 对算法可见，与计划外的混合策略不同

③ trainer 全体重启
   index 的 holder 表整体清空，版本号保留（§3.2 的版本连续性要求）
   -> 所有 pull 立即返回 CHUNK_UNAVAILABLE
   -> rollout 维持现状生成，由 coordinator 决定是否降级或暂停
   -> trainer 恢复后重新注册 holder 并发布新版本，链路自然恢复
```

#### 为什么 peer 不是主要降级手段

§3.5 的 peer 机制看似也提供冗余，但它**只覆盖「已经被读到的那个版本」**，
且 §3.5.2 的版本相等性要求在新版本发布时整体清空 holder 位图。
所以 peer 对「新版本的第一波」完全无帮助，而故障恰好可能发生在那时。
**DP 副本才是结构性的冗余，peer 只是带宽优化。** 这两者不要混为一谈。

#### 必须做对的三件事

1. **holder 摘除要能被动触发，不能只靠主动上报。** 故障 rank 不会自己说「我挂了」。
   依据是 NIXL 传输失败：rollout 的 `client_read` 报错时，**连同失败的 holder 身份
   一起上报 index**，index 据此摘除并让该 pull 重新解析。
   这把故障检测搭在已有的错误路径上，不需要心跳。
2. **`CHUNK_UNAVAILABLE` 必须是显式返回值，不能是超时。**
   初稿只说「index 需返回明确错误而非挂起」，这里落实为一个具体状态码，
   且 rollout 侧要有对应处理：放弃本轮、保持版本、不重试到死。
3. **不允许「部分 chunk 成功」就标记版本可见。**
   这是 §3.4.6 的 commit group 完整性断言的直接应用：
   少了任何一个 chunk，该 group 就不可见，整个版本对该 instance 不可见。
   否则会得到一个半更新的模型，比旧版本更糟。

#### 与现有机制的衔接

PSRL 已有 `_quarantine_failed_replicas`（`sync_and_migrate_mixin.py:74`）
处理 rollout 侧同步失败：`fail_sync` + 网关暂停 + 恢复路由循环。
本节的 ② ③ 应当复用这条路径，而不是新建一套，区别只在于触发源从
「rollout 自己失败」变成「index 报告源不可用」。

### 3.6 支柱五：执行层，plan once post many

三处改动，彼此独立可分别落地：

- **预备描述符**：握手期对每个 client 调一次 `prep_xfer_dlist`（本地 + 远端两侧），
  之后每轮用 `make_prepped_xfer(local_indices, remote_indices)` 按下标组合，
  替换每 (key, shard) 一次 `initialize_xfer`。这与 §3.5 的「plan 握手期编译」是同一件事的两面：
  plan 里的下标就是 prepped dlist 的下标。
- **strided 描述符**：用 Nx5 `(addr, len, dev_id, stride, count)` 原生表达非连续 shard，
  删掉 `max_pinned_temp_memory_slots` 整套 pinned 中转 + 槽位等待逻辑
  （`client_read` 里 `_pinned_slot_running_read_xfer` 的串行等待是真实的流水阻塞点）。
  **这条直接服务约束 (b)**：pinned 中转槽本身就是一份额外的 CPU 权重存储。
- **round-based 双 buffer**：chunk 默认 512MB，深度 2 流水。需要被重叠的是**三类**工作，
  不只是量化：
  - trainer 侧的量化 / scale 转码（量化路径）
  - RDMA 本身
  - **rollout 侧的落地格式化**（`process_weights_after_loading` 系）
  第三类绝不可忽略：miles 实测 Kimi-K2 的 `post_load_weights` 要 **884ms**，是纯 GPU 计算、
  与传输无关。按 atomic group 流水，让第 i 组的格式化与第 i+1 组的 RDMA 重叠。
  WeightBridge 实测 pack/assemble kernel 占端到端不足 3% 且被完全隐藏，前提正是这种流水。

#### 3.6.1 把 bucket 的双重身份拆开

初稿让一个 `bucket` 同时承担六个角色：plan 分桶、提交/等待、版本可见性、放行粒度、
holder 位图的位、wire buffer 复用单位。好处是省事，坏处是**任何一处要调粒度就牵动其余五处**
（§3.5.2.1 「W 过大会让 bucket 周转变慢」就是这个耦合的症状）。

本质上这里有两个独立的概念，应当分开命名：

| 概念 | 服务于 | 边界由什么决定 | 可否切开一个 key |
|---|---|---|---|
| **chunk**（传输单位） | 流水与带宽 | 固定字节数（随模型规模自适应，§3.5.6） | **可以**，大 MoE 的 11GB 单 key 必须切开 |
| **commit group**（可见性单位） | 正确性 | 引擎 loader 的原子要求（atomic group） | **不可以**，这是它的定义 |

两者的关系是 **多对多**：一个 chunk 可以含多个小 commit group，
一个大 commit group（如 MoE 的 fused `w13_weight`）可以跨多个 chunk。
于是：

- **传输、流水、W、wire buffer 复用**全部以 chunk 为单位，可以自由调大调小。
- **版本可见性与运行时格式化**以 commit group 为单位，一个 group 的全部 chunk 到齐才提交。
- **放行（在飞读者计数）以 chunk 为单位**，因为它关心的是「这段内存现在有人读吗」,
  与原子性无关。
- **holder 位图以 chunk 为单位**，因为它回答「这段我有没有」。

这一拆解消掉了初稿里最别扭的一条约束：不再需要「让一个 fused 张量跨多个 bucket
但共享一个 group 标记」这种打补丁的说法，跨 chunk 共享 commit group 本来就是
两个概念分离后的自然形态。

**atomic group 是执行层的硬不变量**（miles `atomic_groups.py` 的教训：引擎 loader 会把多个
HF 权重 cache-and-concat 成一个引擎参数，半份到达是**静默 no-op 或断言失败**）。
group 是四件事的共同单位：plan 分桶、提交与等待、版本可见性、§3.4 的租约粒度。
fp8 的 `weight` 与其 `scale` 天然同组。

#### 3.6.2 底层通信栈的落地要点

本设计不引入 NCCL 通信，数据面是纯 RDMA / NVLink。几条必须落实的细节：

**(a) GPUDirect 可用性必须探测，不能假定。**
rollout 直读 trainer 显存，走 GPUDirect RDMA 才是零 CPU 拷贝，
而这要求两侧内存已注册且 NIC 与 GPU 在同一 PCIe root complex 下。
`network_topology.py:42` 只区分 `IB` 与 `IB_PCIE`，**没有区分
「GPUDirect 可用」与「需经 host bounce」**，后者带宽可能差一个量级。
应当在握手期用 `query_memory` 探测并记入拓扑，作为选源的一档权重。

**(b) PCIe 流量归零是未被计入的收益。**
今天 train GPU → CPU PS → rollout GPU 跨 PCIe 两次（D2H + H2D），各 `N×M`。
直供后数据面 PCIe 流量**归零**（GPU 到 GPU，走 NVLink 或 NIC）。
1TB × 32 节点下这是数十 TB 的 PCIe 流量被消除。

注意 optimizer offload 仍走 PCIe（`offload_megatron_copy_params`），
与 NIC 上的权重读取**不争同一资源**，这是 §3.5.2.2 把 optimizer 阶段的 W 放大的依据。
但若 NIC 是 PCIe-attached 且与 GPU 共享 switch，两者**会**争 PCIe 带宽,
这取决于硬件拓扑，应作为可配置的保守假设。

**(c) colocate 下量化 kernel 走低优先级 stream。**
量化 kernel（`per_block_cast_to_fp8` 或 TE 的 MXFP8 quantizer）是显存带宽受限的，
colocate 时会与 rollout 的 decode kernel 争带宽。应在低优先级 stream 上提交
（`torch.cuda.Stream(priority=-1)`），让 decode 优先。一行改动，收益明确。

**(d) `num_threads` 应实测。** `nixl_agent_config` 的 `num_threads` 默认 0,
而 UCX 插件支持多线程。chunk 变大后单个传输的并行度成为瓶颈，
多线程后端可能显著提升单传输带宽。

**(e) FP4 的 swizzle 有一次全量 scale 拷贝。**
`swizzle_blockscale` 的 `permute(...).contiguous()` 是真实的重排开销。
scale 体积约为权重的 1/16，绝对量不大，但应从 FP4 的收益账里扣掉。

### 3.7 支柱六：colocate 原地交接，时序互斥消除读写冲突

#### 3.7.1 三个框架的 colocate 模型对比

查 verl / SkyRL / miles 的源码确认了 colocate 的正确模型：

| 框架 | colocate 做法 | 关键点 |
|---|---|---|
| **verl** | `sleep(level)` 释放权重显存 → `get_per_tensor_param()` 收集参数 → 直接写入 rollout 已注册显存 (`update_weights_from_ipc`) → `resume(tags=["kv_cache"])` | 训练完成后才开始写。不存在并发 |
| **SkyRL** | `skyrl_ipc`：通过 CUDA IPC 句柄共享同一 GPU 显存。同样是**训练阶段与推理阶段互斥** | IPC 句柄传递发生在 rollout 暂停后 |
| **miles** | P2P RDMA 直写。训练阶段结束后才触发 transfer | rollout pause → transfer → rollout resume |

**三者共同点**：正确性来自**时序互斥**而非内存屏障或量化产物。
trainer 持有控制权期间 rollout 是暂停的，rollout 运行期间 trainer 不写权重。
**这与精度配置完全无关**：bf16+bf16、fp8+bf16、fp8+fp8 在 colocate 下都合法，
不需要量化路径作为隔离手段。

#### 3.7.2 PSRL 的 colocate 方案（兼容所有精度组合）

PSRL 的 `switch_to_train_mode()` / `switch_to_rollout_mode()`
（`ray_trainer.py:2525-2620`）已经实现了这条互斥。本设计需要做的是：

```
train phase 结束后，进入 transfer phase：
  1. rollout 继续服务请求（streaming decoding 不强制等）
  2. index 发布新版本，holder = trainer
  3. rollout instance 按 §3.5 的同步策略异步拉取
     量化路径：量化 → 写入 wire buffer → rollout RDMA 读 wire buffer → 格式化
     同精度路径：rollout RDMA 直读 trainer 参数 → 写屏障确保读完再覆盖
  4. rollout instance 完成后标记为 holder(v)
```

**colocate 下的额外优化**：若 train 与 rollout 在同一 GPU 且精度相同，
可以省掉 RDMA 完全走 `cudaMemcpy`（zero NIC），甚至走 `update_weights_from_ipc`
（trainer 和 rollout 共享进程时直接指针赋值）。
这是 verl 的 "naive" colocate 模式（`engine_workers.py:748`），成本最低。

#### 3.7.3 colocate 与时序的关键边界

一条必须维护的不变量：**rollout 暂停**（`pause_generation`）后才允许 transfer。
这确保了读写不并发。PSRL 的 `pause_for_sync` 已经在这条路径上。

**但对于 disaggregate 模式（train 与 rollout 在不同节点），这条不变量不适用：**
rollout 在 transfer 期间仍然服务请求，这正是设计允许的。
冲突靠 §3.4.2.1 的写屏障解决，colocate 靠暂停解决。
**两种模式用不同机制，这是设计意图而非缺口。**

### 3.7a 旧 §3.7 内容（colocate 不需要 TMS）

colocate eval 的本质是**同一块物理显存在 train 与 rollout 之间按时间片交接**，
所以正确目标不是「把回灌做快」而是**让权重根本不动**。约束 (b) 和这个目标完全一致：
原地交接本来就不需要第二份副本。

做法：train 侧把（量化后的）权重直接写进 **rollout 已注册的那块参数显存**
（`convert_vllm_inplace` 注册的 `unified_state_dict` 就是它的 HF 视图）。于是切换只剩：

```
switch_to_rollout:  train 释放 optimizer state / activation（权重区不动，NIXL 注册不失效）
                    rollout 分配 KV cache
switch_to_trainer:  rollout 释放 KV cache（权重区不动）
                    train 恢复 optimizer state / activation
```

没有权重搬运，没有 dereg/rereg，没有元数据重广播。

**关于约束 (c)：这条路径不需要 TMS。** 核实了 pinned 的 vLLM
（`third_party/vllm`）后发现它已经有正好合适的原语：

- `CuMemAllocator.discard(tags)`（`cumem.py:300`）丢弃指定 tag 的映射**而不做 CPU 备份**。
- `EngineCore.release_kv_cache_memory()`（`core.py:981`）就是 `discard(("kv_cache",))`，
  且前置断言「必须先 pause 完成」「executor 内存必须 resident」，正是切换时的状态。
- KV cache 本就单独打了 `kv_cache` tag（`gpu_worker.py:842`），权重打 `weights` tag
  （`gpu_worker.py:569`），粒度天然分离。
- 反过来，`SleepModeBackend.suspend(level)`（`sleep_mode_backend.py:133`）在 level 1 会把
  `weights` **offload 到 host**，level 2 丢弃后需从 model source 重载。这两种都违反我们的目标。
  所以**不能用 `sleep()`，要用 `discard(("kv_cache",))`**。

因此 colocate 切换的 rollout 侧 = `pause` + `release_kv_cache_memory()` + 后续 `wake_up(["kv_cache"])`，
全部是 vLLM 原生 API。train 侧只需释放 optimizer/activation，这是 PSRL 自己的内存，
用现有的 storage resize helper（`_sleep_fsdp_model` / `_sleep_megatron_model` 已经在做
「resize 到 0 释放、resize 回来不拷数据」）即可，同样不需要 TMS。

**TMS 的保留范围**（符合约束 c）：仅在「train 侧 optimizer state 的释放用 resize 做不干净」
时作为可选后端。权重区和 KV cache 都不走 TMS。`tms.range` 不需要新增 weights/kv_cache 粒度，
上一版设计提的那条需求因为改用 `discard` 而消失了。

显存账：colocate 下 train bf16 权重 + rollout fp8 权重 = `1.5×M`。若要进一步压，
FP8-native train 让两侧共享同一份 fp8 权重，账面降到 `~0.5×M`，这是 FP8 与 colocate
在本设计里真正合流的地方。

### 3.9 训练后端与 vLLM 格式对齐：Megatron 与 FSDP 并列支持

PSRL 有两个训练后端，各自持有不同的权重布局，在直供路径下都需要生产
vLLM 能直接读取的 wire 格式。

#### 3.9.1 两个后端的分片结构

| 后端 | 分片维度 | DP 冗余 | PSRL 转换器 |
|---|---|---|---|
| **Megatron** | TP（行/列切分）+ EP（expert 切分）+ PP（层切分） | DP 维度不出现在 `NIXLSharding`，DP 组内所有 rank 持有相同 shard（§3.5.7 的 DP 副本接管依赖这点） | `megatron_converter.py::MegatronConverter` |
| **FSDP / FSDP2** | 参数按 `DTensor` 的 placement 切分（`Shard(dim=0)` 或 `Shard(dim=1)`）。HSDP 增加第二维 replicate | DP 同样不出现在 `NIXLSharding`（`fsdp_converter.py:325` 的 HSDP 路径只用 `mesh_dim=1` 的 rank） | `fsdp_converter.py::FSDPConverter` |

两个转换器都输出 `(state_dict, sharding_dict)` → `NIXLSharding`，
接入同一套 NIXL 传输和 index 基础设施。**后端切换不影响传输层。**

#### 3.9.2 vLLM 要求的权重格式

vLLM 的 weight loader 期望 HF 格式的 canonical key（如 `model.layers.0.self_attn.q_proj.weight`），
以及 vLLM 自己的 fused 格式（如 `model.layers.0.self_attn.qkv_proj.weight`，三矩阵合并）。

**两个后端的格式转换路径**：

| 后端 | HF canonical → vLLM fused | 关键转换点 |
|---|---|---|
| Megatron | QKV 合并、MoE expert 维度声明 | `megatron_converter.py::slice_qkv_proj_megatron`、`get_sharding_for_param` |
| FSDP | QKV 3D reshape（`maybe_reshape_qkv_to_3d`）、MoE expert 拆分（`split_fused_moe_fsdp_param`） | `fsdp_converter.py::FSDPConverter.convert_state_and_sharding_dict` |

注意 FSDP 目前的 MoE 处理是**把 fused MoE 参数拆成 per-expert key**
（`split_fused_moe_fsdp_param`），这与 §3.5.5 提出的「MoE 分组（保持 fused）」
方向相反。MoE 分组改造需要同步修改 `fsdp_converter.py` 的 split 逻辑，
使其也保持 fused 形态并改为声明多维 shard。**这是 §3.5.5 的一部分，不是额外工作。**

#### 3.9.3 FP8 训练下的 weight format 问题

Megatron 的 FP8 训练（TE `fp8_param=True`）下，model params 本身存储为 FP8，
但 `NIXLSharding` 和 key 名不变。转换器从 `fp8_param` 的真实存储 dtype 读取，
而不是名义上的 bf16。

FSDP 的 FP8 训练路径（`fp8_param` flag 在 FSDP 里不同于 Megatron，
依赖 TE 的 `fp8_autocast` 上下文管理器）需要单独核实 state dict 里的
实际 dtype。**当前设计假设 `fsdp_converter.py` 已正确处理这一点，
实现时需要加一个 dtype 断言。**

### 3.8 FP8：贯穿全链路的 wire 格式

**依赖边界先说清楚：本节所有原语都在 PSRL 已有依赖里，不 import SkyRL / miles。**

| 需要的能力 | 来源 | 符号 |
|---|---|---|
| blockwise 量化（128×128 tile） | vLLM | `vllm/utils/deep_gemm.py::per_block_cast_to_fp8` |
| per-tensor / per-group 量化 | vLLM | `_custom_ops.py::scaled_fp8_quant`、`fp8_utils.py::per_token_group_quant_fp8` |
| MXFP8 量化（1×32，E8M0） | Transformer Engine | `transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Quantizer` |
| fp8 参数 layout 与命名 | vLLM | `fp8_utils.py::create_fp8_weight_parameter` |
| 运行时格式化（UE8M0 / TMA） | vLLM | `fp8_utils.py::{process_fp8_weight_block_strategy, requant_weight_ue8m0_inplace}` |
| 分块合法性校验 | vLLM | `fp8_utils.py::{validate_fp8_block_shape, validate_fp8_block_shape_moe}` |
| FP8-native 训练参数 | Transformer Engine | `fp8_param` 路径下的常驻 fp8 参数 + scale |

SkyRL 的 `quantize.py` / `vllm_format.py` 只作为**参考读物**，说明两种 wire 的字段命名与
`quant_method` 取值该怎么填，其算法本身就是上表这些 vLLM/TE 原语的组合。PSRL 自己实现这层
组合逻辑（约数百行），放在 `psrl/utils/fp8/` 下。

**一条原则：量化发生在源头，链路不感知精度。** 两种训练形态产出**同一种** wire payload：

| train 形态 | 产出路径 | 代价 |
|---|---|---|
| BF16 train | train GPU 上 blockwise / MX cast（复用 vLLM `fp8_utils` 或 TE kernel） | 一次 amax + cast，与 chunk 流水重叠，瞬态 `O(chunk)` |
| FP8 train（TE `fp8_param`） | TE 已持有 fp8 参数 + scale，只做 **scale 格式转码**（TE blockwise ↔ `weight_scale_inv`，MXFP8 E8M0 ↔ compressed-tensors uint8） | 无重量化，不引入额外数值误差，省掉 amax，**额外显存 0** |

FP8 train 下**不要「反量化回 bf16 再量化」**：那会白付两次精度损失和一次全量 amax，
而 TE 的 scale 与 vLLM 的 scale 只差一个编码约定。注意 FP8-native 形态在约束 (b) 下
是**最优形态**，因为 trainer 持有的就是 rollout 要的字节，直供零转换零缓冲。

链路收益：传输字节减半。且因为没有 PS 存储，今天 `PSStoragePlan` 为 train/gen dtype
不等准备的第二块缓冲和 `transfer_train_to_gen` 拷贝**一并消失**（这两者本来就是约束 (b)
要删的东西）。

**运行时格式化**（转置 / TMA 对齐 / UE8M0 打包）留在 rollout 落地后做，复用 vLLM 的
`process_fp8_weight_{tensor,block}_strategy` 与 `requant_weight_ue8m0_inplace`，尽量 in-place。
train 与 inference 并行度匹配时源头可直接产出终态布局，此时 `is_weights_pre_processed()`
为真，这一步退化为 no-op。

##### 3.8.0 精度配置矩阵：哪些组合被支持

| train | rollout | 路径 | 状态 |
|---|---|---|---|
| bf16 | bf16 | 同精度 | 支持，colocate 下靠时序互斥保证正确性（§3.7.1） |
| bf16 | fp8 | 量化 | **推荐默认**，冲突归零且带宽减半 |
| bf16 | fp4 | 量化 | 支持，需 §3.8.1 的 swizzle 处理 |
| fp8 (TE `fp8_param`) | fp8 | 同精度 | 支持，只做 scale 转码 |
| fp8 (TE) | fp4 | 量化 | 支持，但**量化源必须是 fp32 master params**（§3.4.2 约束 2） |

配置校验必须拒绝的组合：`colocate && train_dtype == rollout_dtype`。

### 3.8.1 扩展到 FP4：架构可行，但两处必须显式处理

vLLM 已有 FP4 支持（`quantization/modelopt.py` NVFP4、`quantization/mxfp4.py` MXFP4、
`utils/nvfp4_utils.py`），所以「源头量化、链路不感知精度」的原则可直接延伸。
但 FP4 有两个 FP8 没有的结构性差异：

**差异一：两级 scale。** NVFP4 是 `weight` + `weight_scale`（per-block，fp8_e4m3）
+ `weight_scale_2`（per-tensor，fp32），见 `modelopt.py:854` 的
`uses_weight_scale_2_pattern()` 与 MoE 的 `w13_weight_scale_2` / `w2_weight_scale_2`。
**atomic group 要从「weight + scale」扩展为「weight + scale + scale_2」**，
这是参数化而非新机制。另有「w1 与 w3 的 `scale_2` 必须相等」的约束
（`modelopt.py:988` 断言），属量化侧正确性，由源头保证。

**差异二：scale 要 swizzle，且 swizzle 与分片不交换。**
`nvfp4_utils.py:13` 的 `swizzle_blockscale` 把 block scale 填充到
`(M→128 对齐, K→4 对齐)` 后做
`reshape(B, M/128, 4, 32, K/4, 4).permute(0,1,4,3,2,5)`。
swizzle 跨 128 行分组，而 TP 切分也在行维度，**两者会互相破坏**。

后果：下文「并行度匹配时源头直接产出终态布局、`is_weights_pre_processed()` 为真、
落地格式化退化为 no-op」这条优化**在 FP4 下不成立**。FP4 必须
**传未 swizzle 的 canonical scale，在 rollout 侧落地后再 swizzle**，
这正属于 §3.6「落地格式化与 RDMA 重叠」的那类工作。
架构上有位置，但**必须显式禁用 FP4 的 no-op 优化**，否则得到静默错误的 scale。

收益：FP4 相对 FP8 把传输字节再减半，1TB 模型降到 `M/4`，wire buffer 同样减半。

**分块对齐是断言而非风险**：在 block 对齐的 canonical 层面量化，raw block scale 形如
`(⌈N/128⌉, ⌈K/128⌉)`，对任意 block 对齐的分区都是干净子块。每个量化 key 过
`validate_fp8_block_shape` / `_moe`。不整除的 模型×TP 组合 vLLM 引擎本身也会拒绝。
miles 记录的 Kimi-K2 需把 `[128,128]` 重量化为 `[64,64]` 才能跑 `tp=32`，就是这条约束的
真实表现，应在配置校验期报出，而不是传输中途。

---

## 4. 端到端数据流

```
train step v 完成（bf16 或 TE fp8 参数）
  │
  ├─ 向 WeightIndex 登记：v 可读，holder = 本 rank（附 PP/EP 归属）
  │
  ├─ colocate？──是──▶ 量化结果原地写入 rollout 已注册参数显存（§3.7）
  │                      切换 = rollout release_kv_cache_memory / wake_up(["kv_cache"])
  │                             train 释放/恢复 optimizer state
  │                      权重零搬运，NIXL 注册不失效，无重握手，不经 host
  │
  └─ 否 ──▶ rollout instance 自行发起 pull（无全局 barrier）
             │
             │  向 index 声明 (v, chunks) 并取得租约
             │  index 返回预编译 plan + holder，按链路成本排序：
             │     同节点 peer rollout（已持有 v）> 跨节点 peer > trainer rank
             │
             ├─[①源头准备] 量化或 scale 转码（仅 dtype 不同时）
             │   Megatron 无需 gather，shard 代数直传（§3.3）
             │   瞬态 O(chunk)，深度 2 流水，读完即复用
             │
             ├─[②三阶段执行] receive -> external-exchange -> internal-exchange
             │   prepped descriptor 按 index 组合，round-based 双 buffer
             │   atomic group 为提交与等待单位
             │
             ├─[③落地格式化] vLLM fp8_utils 运行时格式化（并行度匹配时 no-op）
             │   group 完整到达才标记版本可见
             │
             └─ 释放租约
                 │
                 ▼
            trainer 在下一次 optimizer step 前等待 v-1 租约清零（通常已清零）
            超时则撤销租约并要求重拉，trainer 不被无限阻塞
```

---

## 5. 优化项清单

| # | 优化 | 收益 | 独立落地 |
|---|---|---|---|
| 1 | `prep_xfer_dlist` / `make_prepped_xfer` 替代 per-key `initialize_xfer` | 去掉 `O(#tensors)` 的 per-step 提交开销，MoE 最显著 | 是 |
| 2 | colocate 原地交接（`discard(("kv_cache",))`） | 切换成本从「全量 PCIe 往返 + 重握手」降到「KV cache 重分配」 | 是 |
| 3 | Nx5 strided 描述符 | 删掉 pinned 中转（本身是一份额外 CPU 权重存储）与槽位串行等待 | 是 |
| 4 | 源头 FP8 wire | 传输字节减半，消掉 train/gen 双缓冲，且量化产物天然就是发布缓冲 | 是 |
| 5 | round-based 三类重叠（量化 / RDMA / 落地格式化） | 隐藏 884ms 级的 `post_load_weights` | 是 |
| 6 | **canonical MoE 分组**（key 数 14万 → 数百，§3.5.5） | 元数据从 56MB/client 降到几百 KB，**大 MoE 与 1TB 的前置阻塞项** | 是 |
| 6b | **控制面下热路径**（plan 本地缓存 + 租约按 instance 聚合 + gRPC） | 消除 index 单点，**这是 7/8/9 的前置** | 是 |
| 6c | `make_xfer_tag` 去 pickle、chunk 大小自适应、轮级决策本地化 | 1TB/2000 轮下的纯 CPU 与 RTT 开销 | 是 |
| 6d | chunk / commit group 概念分离（§3.6.1） | 解开传输粒度与原子性的耦合，是 6 的前置 | 是 |
| 7b | W 跟随训练阶段（前馈）+ 实测反馈（校正） | 消除纯反馈的滞后，避免 bwd 阶段超卖 NIC | 否，依赖 7 |
| 7c | chunk 起始偏移错开 | 零成本，让扩散在第一波内部发生 | 否，依赖 7 |
| 8b | trainer 单 rank 故障：DP 副本接管 + `CHUNK_UNAVAILABLE` | 万卡下的可用性，复用已有 quarantine 路径 | 否，依赖 8 |
| 7 | 三阶段去冗余规划 + 同版本同节点 peer 优先 + 自适应 W | 消除直供带来的单源瓶颈 | 否，需新 planner |
| 8 | 版本租约 + 两条不变量（版本 / 注册有效） | 无副本下保护 trainer 权重，消除 push/pull 互斥 | 否，需与 staleness 合并 |
| 9 | trainer 直供，删除 PS worker，新建 bootstrap | 额外 CPU 存储 `N×M → 0`，train 出口 `N×M → M` | 否，需 6+7+8 |
| 9b | **holder 语义对称化 + bootstrap 走 rollout 代表读盘** | `is_rollout_mode_in_actor=True` 的**正确性前置**，无此项该场景不成立 | 否，需 index |
| 9c | **NIXL 增量注册**（`invalidateRemoteMD` / `loadRemoteMD`） | 弹性伸缩的前置阻塞，与 6 同级 | 是 |
| 10 | delta 传输（跨 DC / 慢链路） | 每步仅 1%~3% 字节变化 | 是，正交 |

落地顺序建议：**1-5 先行**（独立、立刻见效，且 3 已在削减 CPU 存储），
然后 **6 / 6b / 6c 必须先做**，6 是大 MoE 与 1TB 场景的硬阻塞（否则连握手都过不去），
6b 避免直供成功后瓶颈转移到 index，
最后 **7 → 8 → 9**（这三者必须一起才能安全地删掉 PS worker）。
**9b 与 9 同批**（holder 对称化是 `is_rollout_mode_in_actor=True` 的正确性前提，
不是优化）。**9c 在启用弹性伸缩前必须完成**。
10 / 11 是独立增量，可随时插入。
完整的第二轮审核见 [weight_transport_review2](weight_transport_review2)。

第 10 项：verl 的 `delta_sharded` 与本设计正交（它优化「发什么」，本设计优化「发给谁、怎么发」），
在跨 DC 或慢链路时才是主要收益项，不应作默认。它压缩的是**传输**而非 reload。
注意它需要一份 pinned 快照作 diff base，**在约束 (b) 下这本身就是一份额外 CPU 权重存储**，
所以只能是显式 opt-in 的例外。

---

## 5.1 待优化清单（已评估，暂不纳入）

| 项 | 为何暂不做 | 重新评估的触发条件 |
|---|---|---|
| **跳过未变化权重的量化与传输**（冻结层、未激活 expert、lr=0） | 判定成本低但需要与 optimizer 的 grad 状态耦合，且收益依赖具体训练配方 | 实测到量化或传输占端到端比例显著，且确认有大量冻结层 |
| **渐进可用（部分 chunk 到达即开始生成）** | **vLLM 层面不支持，见下** | vLLM 提供「按层增量切换」的引擎 API |
| delta 传输 | 需要一份 pinned 快照作 diff base，违反约束 (b) | 跨 DC 或链路显著慢于 GPU 时 |
| 跨节点 peer | 阶段二，收益随 R 增大才显现 | 实测到 trainer 出口饱和 |
| 元数据 RDMA 按需拉取 | MoE 分组后扇出可能已够小 | 实测握手期元数据总量仍超预算 |
| **rail 感知选源** | `NetworkTopology` 无 rail 概念，万卡多轨道下有真实带宽差。UCX 本身有 multi-rail 支持，所以数据面可能已受益，但选源决策看不到 | 万卡部署且实测到同 rail 与跨 rail 带宽差显著 |

**渐进可用为何不可行**（已查证 vLLM 源码）：

vLLM 的增量加载路径 `initialize_layerwise_reload`
（`model_executor/model_loader/reload/layerwise.py:82`）会
**先把层参数恢复到 meta device**（`restore_layer_on_meta`），
等该层全部权重到齐后才 materialize 回真实设备。
也就是说**在 reload 期间模型根本不处于可推理状态**，
「部分 chunk 到达就开始生成」在引擎层面无处落脚。

而 PSRL 走的是另一条路径：NIXL 直写 `convert_vllm_inplace` 注册的运行时张量，
**完全不经过 layerwise reload**。这条路径下参数始终在真实设备上，
但也因此没有「哪些层已更新」的引擎级概念,
生成期间若权重被部分改写，得到的是半新半旧的混合策略,
这正是 §3.4.1 明确要排除的东西。

所以渐进可用需要的是「引擎能按层原子切换新旧权重」这个能力，
vLLM 当前不提供。**这一项从方案中移除，而非保留为可选优化。**
PSRL 现有的 `pause_generation` + 直写 + `resume_after_sync` 才是正确的语义:
传输期间暂停，传完整体切换。

---

## 6. 风险与边界

- **读写冲突已收窄为一条规则。** 量化路径（train/rollout dtype 不同）用量化产物作发布缓冲，
  零额外显存、无冲突。同精度路径直读参数，用 per-chunk 在飞读者计数器加写屏障放行，
  需埋点统计等待时长 p50/p99。两条路径的选择由「源头是否产生新字节」决定，不由 dtype 枚举决定。
- **index 控制面是最可能的实际单点瓶颈。** 见 §3.2 的三条约束（plan 本地缓存、
  租约按 instance 聚合、热路径走 gRPC）。必须在直供之前完成，否则瓶颈只是转移。
- **扇出不能依赖 sync 策略的时序巧合。** greedy 本身有自然分批（§3.5.1），但版本发布是
  同时事件，全体空闲时仍会一轮并发。扩散必须由 index 用 trainer 并发上限 W 强制形成，
  且 `pause_for_sync` 要从「整批统一暂停」下移到「该 instance 真正开始读」。
- **peer 供给会抢 rollout 自己的推理资源。** 每 peer 限 `P` 个并发读者（建议 1~2），
  colocate 模式下直接禁用 peer 供给。peer 作为源时同样需要租约，
  否则它的 wire buffer 被下一轮复用会让读者拿到半新半旧。
- **只能从同版本 peer 读（§3.5.2）。** 从 `v-1` 的 peer 读等于传输层偷偷加了一个单位
  staleness，算法侧不知情。部分 chunk 读 v、部分读 v-1 更糟，那是一个不存在的混合策略。
  代价是每个新版本的第一波必须从 trainer 读，扩散不跨版本累积，这正是 W 存在的理由。
- **拓扑数据精度有限，不要在它之上做连续优化。** `network_topology.py` 的带宽是硬编码常量，
  且 `LOCAL_NODE_IP` 全仓库只有定义没有设置处。只做「同节点 / 跨节点」这种离散判定，
  并建议把节点身份从 IP 换成 Ray node id（`get_node_id()` 已有）。
- **colocate 下应禁用 peer 供给。** colocate 时 rollout 与 train 共享 GPU，
  一个 rollout 为 peer 提供读取会占用它自己的 NIC 带宽与 NIXL progress thread 的 CPU，
  直接拖慢 eval 本身。
- **失去了 PS 作为版本快照的调试能力。** 今天可以对比「PS 里的 v」与「rollout 里的 v」
  来区分传输错与训练错（`_debug_log_ps_info` 就为此存在）。直供下这个中间观测点消失，
  需补一个可选的 checksum 校验模式（miles 的 `--check-weight-update-equal` 与
  `local_weight_checksum.py` 是现成范式）。
- **`ParamSyncPlan` 的 push 成对语义必须改造。** 见 §3.3。不改造会让 rollout 读到
  被 `after_push` 还原过的值。
- **FP8 的算法层影响不是 infra 能消化的。** rollout(fp8) 与 train(bf16) 的 logprob 有系统性偏差，
  需 off-policy / importance sampling 修正。infra 的边界是保证 rollout logprob 与 fp8 权重版本
  严格对齐。SkyRL 用 rollout-vs-trainer logprob gap 作为「权重确实落对了」的判据，
  PSRL 应同样把它作为回归指标而不是只看 reward。
- **trainer 成为数据面关键路径，但 DP 副本提供了结构性冗余。** train 侧 sharding 不含
  DP 维度，同一 chunk 在 DP 组内有 `dp_size` 份相同副本，单 rank 故障可由同组接管（§3.5.7）。
  整组故障或全体重启时返回 `CHUNK_UNAVAILABLE`，rollout 保持当前版本等下一版，
  而不是读旧版本或卡死。注意 peer 只是带宽优化，不是降级手段。
- **弹性伸缩依赖 NIXL 增量注册，当前未实现。** `server.py:56/90` 的
  `TODO: Support elastic clients after initial registration` 是硬阻塞，
  需用 `invalidateRemoteMD` / `loadRemoteMD`（`nixl.h:528` 起）。
  缩容还必须主动撤销 holder 资格（§3.3.1c），不能靠被动摘除。
- **QP 数量是万卡下的隐性上限**（§3.5.2.1 约束 ④），单边读仍需 NIC 维护 QP 状态。
- **GPUDirect 不可用会退化为 host bounce**，带宽可能差一个量级，必须探测（§3.6.2a）。
- **容错未在本设计完整覆盖。** holder 抽象天然支持（失效即摘除、pull 重新解析），
  但租约回收时机、plan 重编译、instance 动态增减需单独设计。WeightBridge 同样列为 future work。
- **block 对齐约束会拒绝某些 模型×TP 组合。** 这是 vLLM 引擎自身的约束，PSRL 只是提前报出。
  不要在传输层绕过它。
- **Megatron PP/EP 下 holder 归属是启动期错误，风险可控。** converter 只遍历本 rank 的
  `named_parameters()`，天然只登记自己持有的参数，comm plan 对「target 需要但无 source 提供」
  的 key 直接 assert（`No available key {key} in source client groups`）。
  所以 PP stage 覆盖不全在启动期暴露，而非静默读到垃圾。仍需在握手期校验覆盖完整。
- **block 对齐约束会拒绝某些 模型×TP 组合。** 这是 vLLM 引擎自身的约束，PSRL 只是提前报出。
  不要在传输层绕过它。

---

## 7. 验证

1. **零额外存储**：实测稳态下无 PS worker 进程、host 侧无权重常驻，且量化瞬态显存
   上界为 `2×chunk` 而非 `O(M)`。
2. **流量最优性**：实测 trainer 出口字节 == `M`、每 rollout 节点入口 == `M_v`，并与 `T_min` 比值。
   WeightBridge 在 16~64 节点报 `1.27×`~`2.12×` ideal，可作量级参照。
3. **单源瓶颈已消除**：固定 trainer、逐步增加 rollout replica 数，AGST 应为亚线性
   （miles P2P 在此处近线性退化，是对照基线）。**同时记录控制面消息数**，
   确认它不随 replica 数平方增长（§3.2）。
4. **冲突等待分布**：同精度路径下每步写屏障等待的 p50/p99，及其随 staleness 与 replica 数的变化。
   量化路径下应恒为零，这是验证「发布缓冲确实隔离了 optimizer 写」的判据。
5. **注册有效性不变量**：在 trainer DDP buffer resize 后、重注册前注入一次 pull，
   必须被 index 拒绝，而不是读到已释放页。这条测的是静默损坏，必须有。
6. **colocate 切换**：切换耗时与 PCIe 传输量，目标是后者归零。
7. **atomic group 不变量**：故意只送达 group 的一半，必须报错而不是静默 no-op。
8. **等价性**：小模型上逐张量比对「本设计产出」与「vLLM 从原生 fp8 checkpoint 加载 +
   `process_weights_after_loading`」，在 UE8M0 / requant 容差内一致。
9. **故障注入**：杀掉一个 trainer rank，pull 必须由同 DP 组接管且 rollout 无感知。
   杀掉整个 DP 组，必须返回 `CHUNK_UNAVAILABLE` 且 rollout 保持当前版本继续生成，
   不卡死、不读到半更新模型。
10. **扩散有效性**：固定 R 与 W，记录每个 instance 的源分布（trainer / 同节点 peer），
    开启 chunk 偏移错开前后对比第一波的 peer 命中率。
11. **四条运行时断言**（§3.4.6）必须各有一个故意违反的测试，确认是报错而非静默通过。
12. **`is_rollout_mode_in_actor=True` 全流程**：启动时全 GPU 跑 rollout 能拿到 v0、
    切回 trainer 能从 rollout 侧 pull 到训练起点、再切回 rollout 仍正确。
    **这是 §3.3.1a 的回归测试，缺它则该场景的正确性无保障。**
13. **弹性伸缩**：扩容的新 instance 能冷启动拿到当前版本。缩容的 instance
    先撤销 holder 再 sleep，期间无读者失败。
14. **精度矩阵**：§3.8.0 的五个支持组合各跑一次端到端，并确认
    `colocate + 同精度` 被配置校验拒绝（正确性校验，不是性能测试）。
15. **端到端**：rollout logprob 与 bf16 参考的偏差落在预期区间，长跑中不漂移。

---

## 8. 暂不纳入：自动模型接入

按约束 (a)，本设计沿用现有显式 layout plan（`weight_layouts.py` + `model_mappings.py` +
`weight_layout_transforms.py`），不引入探测式对应关系发现。

保留两点接口余地，使将来接入不需要返工：

- **canonical 命名空间不变。** PSRL 的 unified HF state dict 正是 WeightBridge 所说的
  「作为桥梁的 checkpoint」，自动化只会替换**映射的来源**，不改变 key/shard 抽象本身。
- **plan 是数据而非代码路径。** §3.5 的三阶段 plan 在握手期编译并存入 index，
  编译输入是「两侧的 (key, shard) layout」。无论这个 layout 来自手写 builder 还是将来的
  自动推导，planner 与执行层都不需要改。

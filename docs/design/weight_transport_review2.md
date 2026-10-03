# 权重传输方案自审（第二轮）

> 对 [weight_transport](weight_transport) 的第二轮审核，覆盖实现可行性、代价与风险、
> 可扩展性（大型 MoE / FP4 / 万卡 / 1TB）、完整性与瓶颈、通信与序列化开销五项。
>
> **结论先行：主干仍然成立，但本轮发现一个会在目标规模上直接失效的问题，
> 它不在新设计里，而在 PSRL 现有的 NIXL 元数据层：握手期每个 client 要把
> 全量 `tensor_infos` pickle 后经 `send_notif` 发出，服务端再按 agent 扇出。
> 在 Kimi-K2 这类模型上这是 56MB／client、数十 GB 总量。** 这件事必须作为
> 直供之前的前置工作解决，否则万卡 / 1TB 场景连握手都过不去。

---

## 0. 本轮新发现的硬问题：元数据平面不随模型规模扩展

### 0.1 数量级

PSRL 的 canonical 命名空间是 **HF 展开形式**，MoE 的每个 expert 的每个投影都是一个独立 key。
按 Kimi-K2（1T，384 experts，61 layers）估：

| 形式 | 张量数 |
|---|---|
| vLLM fused（`w13_weight` / `w2_weight` per layer） | **122** |
| HF 展开（gate / up / down per expert） | **70,272** |
| HF 展开 + fp8 scale 成对 | **140,544** |

而 `NIXLClientInfo.tensor_infos` 是 `dict[str, NIXLTensorInfo]`，每个 `NIXLTensorInfo` 含
`desc_bytes_list` / `temp_desc_bytes_list` / `sharding`（一个带 `OrderedDict` 与
`shard_indices` 列表的对象）/ `shard_meta_infos`。保守按每 key 400 字节 pickle 计：

```
Kimi-K2 fp8 展开： 140,544 keys x ~400B  ≈  56 MB   （单个 client 的 client_info）
```

### 0.2 它是怎么被放大的

三处放大，依次叠加：

1. **`send_local_info`**（`client.py:861`）把整个 `client_info` pickle 后经
   `agent.send_notif` 发给 meta server。NIXL 的 notif 是控制通道，不是为 MB 级 payload 设计的。
2. **`notify_all_client_infos_and_comm_plan`**（`server.py:257`）为**每个 agent**
   重新 pickle 一份它「相关」的 client_infos 集合。虽然已经做了
   `_get_relevant_client_names_for_agent` 过滤和「每个 info 只序列化一次」的优化，
   但扇出仍是 `O(agents × relevant_infos)`。
3. **comm plan 本身**是
   `{client: {key: {target: [shard_indices]}}}`（`comm_plan.py:22`），
   **按 key 展开**。14 万 key × 数百 client 的 plan 对象，pickle 后同样是 GB 级。

粗略总量（1024 client 规模）：

```
client_info 扇出 ≈ 56MB × 1024  ≈  57 GB 的 pickle 流量与等量的反序列化 CPU
comm_plan        ≈ 同量级
```

**这不是优化问题，是「万卡 + 1TB」场景下的硬阻塞**：握手会在
`wait_for_server_info` 的 1200s 超时内失败，或者把 head 节点 OOM。

### 0.3 为什么新设计让它更紧迫而不是更轻

直供去掉了 PS worker，client 数量从 `train_ranks + ps_workers + rollout_ranks`
降到 `train_ranks + rollout_ranks`，看似更少。但：

- 去掉 PS 后，**rollout 的对端从「本节点一个 PS worker」变成「所有 trainer rank」**，
  `_get_relevant_client_names_for_agent` 的过滤效果变差，relevant 集合变大。
- peer 读取（§3.5）要求 rollout 之间也互为对端，**又一次扩大 relevant 集合**。

所以新设计**加剧**了这个问题。必须正面解决。

### 0.4 解法：元数据也要去「按 key 展开」

三条，按收益排序：

**(a) canonical 层面做 MoE 分组，不再逐 expert 建 key。**

这是根因治理。把 `experts.{i}.gate_proj.weight` 这类 key 合并为
`experts.gate_proj.weight` 一个 key，expert 维度表达为 `NIXLSharding` 的一个分片维度
（`shard_mesh` 本来就是 `dim -> count` 的多维结构，完全表达得了）。
于是 Kimi-K2 的 key 数从 14 万降到**数百**，元数据从 56MB 降到几百 KB。

这件事的难点不在 NIXL 侧，而在 converter：`model_mappings.py` 的
`FUSED_MOE_W13_SPLIT` / `FUSED_MOE_W2_SPLIT` 今天就是把 vLLM 的 fused 张量**拆成**
per-expert key（`vllm_converter.py:400` 起）。要反过来保持 fused 形态，
需要让两侧都以 fused 形态注册，并用 shard 代数表达 expert 切分。
**注意这与 vLLM 的 fused loader 是一致的**（它本来就要 `w13_weight` 这个 3D 张量），
所以反而更贴近引擎的真实布局。

**(b) 描述符用 Nx5 strided 表达，而不是每 shard 一条。**

`desc_bytes_list` 是 per-shard 的 list。MoE 的 expert 维度分片在内存上通常是
规则跨步的，正好是 NIXL Nx5 `(addr, len, dev_id, stride, count)` 的用例
（`nixl_bindings.cpp:148` `to_stride_dlist`）。一条 strided desc 替代 `count` 条普通 desc。
这与 §3.6 已规划的 strided 改造是同一件事，**本轮确认它还有元数据压缩的收益**，
不只是消除 pinned 中转。

**(c) 元数据走 RDMA 而非 notif，且按需拉取而非全量扇出。**

即使做了 (a)(b)，万卡下仍有 `O(agents²)` 的扇出问题。正确形态是：
每个 client 把自己的 `client_info` 注册为一块**可被 RDMA 读的内存**，
server 只分发「谁的 info 在哪」的指针表（KB 级），
需要某个对端的 info 时按需 RDMA 读。这把 `O(agents²)` 的 pickle 流量变成
`O(agents)` 的指针表 + 按需读取。

**(a) 是必须做的，(b) 顺带做，(c) 在万卡规模才需要。**

---

## 1. 实现可行性（第二轮复核）

### 1.1 结论

分档没有变化，但**新增一项前置工作**（§0.4a 的 MoE 分组），
它的工作量与「三阶段 planner」相当，且**必须先做**。

| 档位 | 项 | 变化 |
|---|---|---|
| 低风险 | prepped descriptor、描述符合并、strided、colocate `discard`、FP8 wire | 不变 |
| **新增前置** | **canonical MoE 分组（key 数 14万 → 数百）** | **本轮新增，阻塞大 MoE** |
| 中风险 | 三阶段 planner、WeightIndex+租约、同节点 peer 选源 | 不变 |
| 高风险 | `ParamSyncPlan` 语义改造、bootstrap 新建 | 不变 |

### 1.2 新增项的可行性评估

MoE 分组**可行且方向正确**，三条依据：

1. vLLM 侧本来就是 fused 的（`w13_weight` 是 `[E, 2N, K]` 的 3D 张量），
   今天 PSRL 是主动把它拆开，拆开反而是额外工作。
2. `NIXLSharding` 的 `shard_mesh` 是 `OrderedDict[dim, count]` 多维结构，
   `get_local_sharded_tensors` 支持按多维 chunk，表达 `(expert_dim, tp_dim)` 两级切分
   不需要新数据结构。
3. Megatron 侧的 expert 参数本来也是按 EP 分片的连续块，
   `get_sharding_for_param` 只需多声明一个维度。

风险是**EP 与 TP 的切分不一定对齐**（`ETP != 1` 时 expert 内还有 TP 切分），
这时 shard mesh 是 `{0: EP, 1: ETP}` 两维，需要确认 `find_finest_shard_mesh`
的 lcm 逻辑在两维上正确。这是可测的，不是未知。

---

## 2. 代价与潜在风险（第二轮）

### 2.1 本轮新增的风险

**(a) MoE 分组改变了 atomic group 的边界。** 今天 per-expert key 使
「一个 expert 的 gate+up+down」天然是三个独立传输。分组后变成
「一个 fused 张量的某些 expert 切片」，而 vLLM 的 fused loader 要求
**整个 `w13_weight` 参数就位**才能用。所以 atomic group 的粒度变粗，
等待单位变大，**可能降低流水效率**。需要在 bucket 划分时让一个 fused 张量
跨多个 bucket 但共享一个 group 标记。

**(b) 分组后单个 key 的体积变得极大。** Kimi-K2 的 `w13_weight` 单层就是
`384 experts × 2 × 7168 × 2048` 级别，fp8 下约 11GB。一个 key 一条 desc 显然不行，
必须依赖 (b) 的 strided 描述符 + bucket 切分。**这把「bucket 必须能切开单个 key」
从可选变成必须**，而今天的 bucket 概念是「若干完整 key 的集合」。这是一处设计调整。

**(c) FP4 的 swizzle 让「直写终态」不再成立。** 见 §3.2。

### 2.2 上一轮风险的复核

全部仍然有效，其中两条需要上调严重度：

- **控制面瓶颈**：上一轮评为「最可能的瓶颈」，本轮发现元数据平面的问题比租约消息
  严重一个数量级。**租约是稳态 `O(instances)`，元数据是握手期 `O(agents × keys)`**。
  两者都要治，但元数据优先。
- **trainer 可用性**：万卡下 trainer rank 数达数千，**单 rank 故障概率大幅上升**。
  peer 冗余只覆盖「已读到 v 的 bucket」，对「某个 trainer rank 独占的 shard」无效。
  这在万卡规模从「可接受」上调为「需要显式设计」：至少要能把失败的 pull
  降级为「等下一个版本」而不是整个 instance 卡死。

---

## 3. 可扩展性

### 3.1 大型 MoE

**修正前：不可行**（§0 的元数据问题）。**修正后：可行**，且有两项额外收益：

- 直供 + 三阶段去冗余让 MoE 的「expert 分片只传给需要它的 rollout」成为可能。
  这是 MoE 下 P2P 相对 broadcast 的最大收益点（WeightBridge 的数据里
  MoE 配置的加速比最高）。
- EP 布局不同的两侧（train EP=32 vs rollout EP=8）由 shard 代数自动重分片，
  不需要 per-model 代码。

遗留问题：**MoE 的 `atomic group` 与 bucket 的关系需要重新设计**（§2.1a/b）。

### 3.2 FP4 扩展性

**结论：架构可扩展，但 FP4 有两个 FP8 没有的结构性差异，必须在设计里留位。**

查 vLLM（`third_party/vllm`）确认 FP4 支持存在：
`quantization/modelopt.py`（NVFP4）、`quantization/mxfp4.py`（MXFP4）、
`utils/nvfp4_utils.py` / `utils/mxfp4_utils.py`。

**差异一：两级 scale。** NVFP4 是 `weight` + `weight_scale`（per-block，fp8_e4m3）
+ `weight_scale_2`（per-tensor，fp32）。`modelopt.py:854` 的
`uses_weight_scale_2_pattern()` 和 MoE 的 `w13_weight_scale_2` / `w2_weight_scale_2`
就是这一层。

对本设计的影响：**atomic group 要从「weight + scale」扩展为「weight + scale + scale_2」**。
这是参数化的，不是新机制，group 的定义本来就是「必须同时到达的 key 集合」。
`w13_weight_scale_2` 还有「w1 与 w3 的 scale_2 必须相等」的额外约束
（`modelopt.py:988` 的断言），这属于量化侧的正确性，由源头保证。

**差异二：scale 要 swizzle，布局不是矩阵。**
`nvfp4_utils.py:13` 的 `swizzle_blockscale` 把 block scale 填充到
`(M→128 对齐, K→4 对齐)` 后做 `reshape(B, M/128, 4, 32, K/4, 4).permute(0,1,4,3,2,5)`,
即 CUTLASS / FlashInfer 期望的交织布局。

对本设计的影响**更实质**：§3.8 里「train 与 inference 并行度匹配时可直接产出终态布局、
`is_weights_pre_processed()` 为真、落地格式化退化为 no-op」这条优化
**在 FP4 下不成立**，因为 swizzle 后的 scale 布局**与分片不交换**
（swizzle 跨 128 行分组，而 TP 切分也在行维度，两者会互相破坏）。
所以 FP4 必须：**传未 swizzle 的 canonical scale，在 rollout 侧落地后再 swizzle。**
这正好是 §3.6 里「落地格式化要与 RDMA 重叠」的那一类工作，所以架构上有位置，
但**必须把 FP4 的 no-op 优化显式标记为不可用**，否则会得到静默错误的 scale。

**差异三（收益）**：FP4 把传输字节再减半（相对 FP8），对 1TB 模型是 `M/4`。
这使形态 A 的 wire buffer 也减半。

### 3.3 万卡 + 1TB

| 维度 | 评估 |
|---|---|
| 元数据平面 | **修正前阻塞**（§0），修正后 `O(agents)` 指针表 |
| 控制面（租约/选源） | `O(instances)`，但 PS 单进程在数百 instance 时需要 §3.2 的读写分离 |
| 数据面流量 | `M` 而非 `N×M`，这是本设计的核心收益，规模越大收益越大 |
| holder 位图 | 1TB/512MB = 2000 bucket → 250B／instance，可忽略 |
| trainer 故障率 | **需要显式降级设计**（§2.2） |
| bucket 数 | 1TB 下 2000 个，每轮一个 bucket 意味着 2000 轮，**控制面每轮开销必须接近零** |

**最后一行是本轮新注意到的点**：1TB / 512MB bucket = 2000 轮。
若每轮有任何跨进程往返（查询源、确认完成），2000 × RTT 就是秒级纯开销。
所以**轮级别的决策必须全部本地化**：plan 本地缓存（已规划）、
源选择在拉取开始时一次性解析完整轮次（而非每轮查一次）、
完成通知批量上报（已规划）。这条应当写进设计。

对应地，**bucket 大小应随模型规模上调**：1TB 模型用 512MB 得到 2000 轮，
用 4GB 得到 250 轮更合理。bucket 大小的约束是
「`2×bucket` 的常驻 wire buffer 可接受」与「轮数 × 控制面开销可忽略」的折中。

---

## 4. 完整性与单点瓶颈

### 4.1 本轮新发现的瓶颈

| 候选单点 | 严重度 | 状态 |
|---|---|---|
| **NIXL 元数据扇出** | **致命**（大 MoE 下阻塞握手） | **本轮新发现，§0** |
| **轮数 × 控制面 RTT** | 高（1TB 下 2000 轮） | **本轮新发现，§3.3** |
| PS 单进程控制面 | 高 | 已规划 §3.2 读写分离 |
| trainer 出口 | 高 | 已规划 W + peer 扩散 |
| rollout 落地格式化 | 中 | 已规划流水重叠 |
| trainer 单 rank 故障 | 中（万卡下上调） | **需补降级设计** |

### 4.2 性能优化空间复核：仍然缺的两项

**(a) 没有利用「上一版本与本版本的 holder 重叠」。** 每个版本的 holder 位图整体清空
（§3.5.2 的版本相等性要求），所以每版本第一波必须从 trainer 读。但**同一个 instance
在连续两个版本里读的是同一批 bucket**，它与 trainer 的连接、prepped descriptor、
甚至 plan 都完全复用。目前设计已经复用了这些，但**没有复用「读取顺序」**：
如果让不同 instance 以**不同的 bucket 起始偏移**开始读（instance i 从 bucket
`i × num_buckets / R` 开始，环形推进），那么第一波就天然错开了热点 bucket，
**peer 扩散可以在第一波内部就发生**，而不必等第一波全部完成。
这是一个零成本的调度改进，应当纳入。

**(b) 没有考虑 trainer 侧 NIC 与自身 collective 的竞争窗口。** §3.5.2.1 提到 W 应
扣掉 trainer collective 占用，但没有利用一个结构性事实：
**Megatron 的 param all-gather 与 grad all-reduce 有明确的时间窗口**。
在 backward 的 grad reduce 期间，NIC 几乎饱和，此时 W 应趋近 0（让 rollout 读 peer）。
在 optimizer 数值计算期间 NIC 空闲，W 可放大。
把 W 做成「跟随训练阶段的函数」而非「跟随实测反馈的滞后调整」，响应更快。
这需要 trainer 向 index 上报一个粗粒度的阶段标记（fwd / bwd / opt），
成本是每 mini-batch 一条消息，可忽略。

---

## 5. 通信与序列化开销

### 5.1 跳数分析：数据面无多余跳

| 路径 | 跳数 | 说明 |
|---|---|---|
| 今天 | **2**：train GPU → CPU PS → rollout GPU | 且第一跳是 `N×M` |
| 直供（trainer 源） | **1**：train GPU → rollout GPU | RDMA 直写 vLLM 参数 |
| 同节点 peer | **1**：peer GPU → rollout GPU（NVLink/PCIE） | 不占 NIC |
| 跨节点 peer | **1 跳读取，但数据总共走了 2 段** | trainer→peer→self |
| colocate | **0** | 同一块显存（§3.7） |

**跨节点 peer 是唯一「多跳」的情况，但它不是多余的。** 数据从 trainer 到最终的
rollout 确实走了两段 IB，但第二段是**替代了一段本来也要走的 trainer→self**，
而且它让 trainer 的出口只发了一份。总跨节点字节数不变（都是 `M_v` 每节点），
改变的是**谁的 NIC 承担它**，这正是 WeightBridge 三阶段方案的本质
（external-exchange 就是这个多跳）。所以是**有意的负载再分布，不是冗余**。

一处**必须避免的真多跳**：绝不能出现「rollout A 从 rollout B 读，而 B 自己是从
rollout C 读的」这种链式。应当限制 peer 必须是**直接从 trainer 读到的那一批**
（即 holder 位图里记录「来源是 trainer」的标记），否则在 `W` 很小时会形成长链，
延迟累积。这是上一轮没有写明的约束，应当补入。

### 5.2 序列化开销

**数据面：零序列化。** 这是本设计相对今天的一个明确改进：

- 今天 `ps_mode=cpu` / `cpu_ref` 路径有 Ray 序列化（`push_model_state_dict_cpu`
  要 pickle 整个 state dict）。本设计把这两个模式删除。
- `nixl_cpu` 路径数据面本来就是 RDMA 直传、无序列化，直供保持这一点。
- `merge_and_finish_cached_xfer` 走的是 descriptor 对象，不序列化张量。

**控制面：这是 §0 的问题所在。** 复核三处：

| 位置 | 开销 | 处理 |
|---|---|---|
| `send_local_info` / `notify_all_client_infos` | **MB~GB 级 pickle**，大 MoE 下致命 | §0.4：MoE 分组 + strided + 按需 RDMA 读 |
| `comm_plan` 序列化（`comm_plan.py:33` `pickle.dumps`） | 按 key 展开，同量级 | 同上，MoE 分组后自然缩小 |
| 租约 / 选源消息 | KB 级，`O(instances)` | 已规划聚合 + gRPC |
| `make_xfer_tag`（`client.py:42`）用 **pickle 生成每个传输的 tag** | 每 (key, shard, tag) 一次 pickle | prepped descriptor 后传输数大幅减少，但仍应换成轻量拼接（f-string 或 struct） |

最后一行是本轮新注意到的小问题：`make_xfer_tag` 对每个传输做一次
`pickle.dumps` 来生成唯一标识。在 14 万 key 规模下这是 14 万次 pickle，
纯 CPU 开销。换成字符串拼接或 `struct.pack` 即可，**零风险改动**。

### 5.3 NIXL 使用方式的复核

| 能力 | 当前 | 本设计 |
|---|---|---|
| `prep_xfer_dlist` + `make_prepped_xfer` | 未用 | 已规划 |
| Nx5 strided dlist | 未用 | 已规划，本轮确认还有元数据收益 |
| `estimate_xfer_cost` | 未用 | **可用于 W 的初始标定**，不需要实时探测 |
| 单边读（非破坏） | 已用于 pull | peer 共享的正确性基础 |
| `query_memory` | 用于注册校验 | 可用于 §3.4.5 的注册有效性检查 |
| notif 通道 | **误用为 MB 级 payload 传输** | §0.4c 改为指针表 + RDMA 读 |
| telemetry（`capture_telemetry`） | 未用 | 可用于 W 的反馈信号 |

`estimate_xfer_cost` 这一项值得单列：它返回 `(duration, err_margin, method)`，
且 method 是 `ANALYTICAL_BACKEND`，即**基于后端模型的解析估计而非实测**。
这正好适合给 W 做初始标定（避免冷启动时 W 完全靠猜），
而又不会引入 §3.5.3 明确排除的「实时带宽探测」。

---

## 6. 修订建议汇总

按优先级，前两项是本轮新增且必须做：

1. **canonical MoE 分组**（§0.4a）：key 数从 14 万降到数百。
   **这是大 MoE / 1TB 场景的前置阻塞项，优先级高于一切性能优化。**
2. **元数据平面去扇出**（§0.4c）：notif 只传指针表，info 按需 RDMA 读。
   万卡规模的前置条件。
3. **bucket 大小随模型规模自适应**（§3.3），并把轮级决策全部本地化。
4. **FP4 的 swizzle 必须在 rollout 侧落地后做**，显式禁用 `is_weights_pre_processed`
   的 no-op 优化（§3.2），atomic group 扩展为含 `scale_2`。
5. **peer 必须是「直接从 trainer 读到的那批」**，禁止链式 peer（§5.1）。
6. **W 跟随训练阶段**（§4.2b），trainer 上报 fwd/bwd/opt 阶段标记。
7. **instance 的 bucket 起始偏移错开**（§4.2a），让 peer 扩散在第一波内发生。
8. **`make_xfer_tag` 换掉 pickle**（§5.2），零风险。
9. **trainer 单 rank 故障的降级路径**（§2.2），万卡下必需。

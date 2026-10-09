# 权重传输落地实现方案

> [weight_transport](weight_transport) 的实现清单。每一项给出**改哪个文件、做什么、
> 完成判据、验证方式**，以便实现期逐项打勾并确认覆盖完整。
>
> **本文不重复设计论证**，只做「设计 → 代码」的映射。设计理由一律回指主文档章节。

---

## 0. 怎么用这份文档

每个任务有四个字段，缺一不可：

| 字段 | 含义 |
|---|---|
| **落点** | 要改的 `file:line`（行号为写作时的锚点，实现时以符号名为准） |
| **做什么** | 具体改动，不是目标描述 |
| **判据** | 怎么算完成。**必须是可执行的断言或可观测的量，不是「实现了 X」** |
| **验证** | 哪一层测试覆盖（L1 单元 / L2 单节点 e2e / L3 多节点 / L4 规模实测） |

**阶段之间有硬依赖**，见 §1 的依赖图。**同一阶段内的任务可并行**。

**「不可分割块」的含义**：阶段 D（7/8/9）一旦开始就必须完成，
因为它删除 PS 数据面，中途停止会让链路不可用。
阶段 A/B/C 全部是可独立回滚的。

---

## 1. 阶段划分与依赖图

```
阶段 A  独立项，可立即做，各自可回滚
  A1  1    prepped descriptor（int32 ndarray 索引）
  A2  3    Nx5 strided descriptor
  A3  5b   post/wait 拆分 + get_new_notifs 完成检测     ← 阶段 B 的前置
  A4  6c   make_xfer_tag 去 pickle
  A5  9d-a 配置期防御五项
  A6  9e   删除 fsdp_converter 的 FSDP1 分支
  A7  9c-a join 三级判定
  A8  2    colocate CUDA IPC + stage offload
  A9  4    源头 FP8 wire

阶段 B  执行层，依赖 A3
  B1  6d   chunk / commit group 概念分离               ← 6 的前置
  B2  5    三类重叠 + slot ack bitmap
  B3  6c'  chunk 按 cost model 自适应 + 轮级决策本地化

阶段 C  元数据与控制面，依赖 B1
  C1  6    canonical MoE 分组                           ← 大 MoE 硬阻塞
  C2  6b   控制面下热路径（plan 本地缓存 + gRPC）        ← 阶段 D 的前置
  C3  7b   版本发布挂 on_train_step_end

阶段 D  不可分割块：trainer 直供（依赖 C1 + C2）
  D1  7    三阶段去冗余规划 + peer 优先 + 自适应 W
  D2  8    版本可见性 + 三条运行时断言 + holder role 维度
  D3  9    trainer 直供 + 删除 PS worker + bootstrap
  D4  9b   holder 对称化 + bootstrap 走 rollout 代表读盘
  D5  7c   chunk 起始偏移错开
  D6  8d   CHUNK_UNAVAILABLE 真能继续生成               ← 与 8b 同批
  D7  8b   DP 副本接管 + CHUNK_UNAVAILABLE
  D8  9d-b FSDP2 副本数运行时推导                        ← 与 8b 同批

阶段 E  正确性补强，依赖 D
  E1  8c   版本级等待（FSDP2 bf16 同精度必需）+ per-chunk 写屏障（可选）
  E2  9c-a2 非整除 join 重分片走版本边界 + 推 epoch
  E3  9c-b 握手改事件驱动 join / leave
  E4  9c-c partial MD 两段式 + invalidateLocalMD

阶段 F  正交可选
  F1  10   delta 传输（跨 DC），显式 opt-in
```

**关键顺序约束**（违反会导致返工或假完成）：

1. **A3 → B2**：不拆 post/wait，双 buffer 流水在实现层无处落脚（§3.6.4）
2. **C1 → B2 的收益**：MoE 分组前 `T_fmt` 主导，overlap 做了也看不到收益（§3.6.0）
3. **B1 → C1**：MoE 分组让单 key 变成 11GB，必须先有「chunk 可切开 key」
4. **C2 → D**：直供成功后瓶颈会转移到 index
5. **D6 与 D7 同批**：不改 D6 则 D7 落地即撞 assert（§3.5.7 末段）
6. **D8 与 D7 同批**：纯 FSDP2 无 DP 副本，先写死 Megatron 假设会返工
7. **E1 在启用 FSDP2 + `model_dtype=bf16` + bf16 rollout 之前必须完成**（正确性，§3.8.0a）
8. **A7 越早越好**：它定义弹性的语义边界，影响调度层设计

---

## 2. 阶段 A：独立项

### A1 — prepped descriptor（设计项 1）

**落点**：`psrl/utils/nixl/client.py`（`client_read` / `client_write` / `_cached_xfer_descs` 路径）

**做什么**：
1. 握手期对每个 `(local_client, remote_client)` 对调一次 `prep_xfer_dlist`，
   缓存 `(local_handle, remote_handle)`，与 plan 的下标对齐
2. 热路径用 `make_prepped_xfer(op, local_handle, local_idx, remote_handle, remote_idx)`
   替换 per-(key, shard) 的 `initialize_xfer`
3. **索引参数的类型必须在接口上固定为 `np.ndarray[int32]`**，不接受 list

**判据**：
- `initialize_xfer` 在热路径（版本同步期间）调用次数为 **0**，
  只在握手期出现
- 索引构造处有 `assert idx.dtype == np.int32`，
  且有一个单测故意传 list 并断言被拒绝
- 同等 descriptor 数下提交耗时相对改造前下降 **≥ 5×**（1024 条）

**为什么判据是「0 次」而非「变少」**：§3.6.3 实测传 list 会慢 15× 并抵消全部收益，
而 list 与 ndarray 都能跑通、不报错。只有「热路径零 `initialize_xfer` + dtype 断言」
这两条同时成立才能保证收益真的拿到了。

**验证**：L1（新增 `tests/nixl/test_prepped_xfer.py`：dtype 断言、索引复用、热路径计数）
+ 已有的 `bench_nixl_prepped_xfer.py` 作为性能回归基线

---

### A2 — Nx5 strided descriptor（设计项 3）

**落点**：`psrl/utils/nixl/client.py`（`max_pinned_temp_memory_slots` 整套逻辑），
`psrl/utils/nixl/nixl_spec.py`（`get_local_sharded_tensors` 的非连续分片路径）

**做什么**：
1. 对规则跨步的非连续 shard，用 Nx5 `(addr, len, dev_id, stride, count)`
   经 `prep_xfer_dlist` 表达
2. **删除** `max_pinned_temp_memory_slots` / `_pinned_slot_running_read_xfer` /
   `_temp_desc_bytes_mapping` / `_pinned_memory` 整套 pinned 中转
3. 保留一条**非规则跨步**的回退路径（strided 表达不了的情形），
   但它不应再用 pinned 中转，而是直接多条 Nx3 descriptor

**判据**：
- `_pinned_memory` 这个属性在代码中不存在（grep 零命中）
- `wait_for_server_temp_mappings` / `notify_all_client_temp_mappings` /
  `_client_temp_mappings` 全部删除，`nixl_protocol()` 从八步降为六步
- MoE 模型握手期 descriptor 数相对改造前下降 **≥ 100×**（expert 维度规则跨步）
- CPU 侧 pinned 内存占用为 **0**（约束 (b)）

**验证**：L1（`test_nixl_sharding.py` 扩展：strided 表达的分片等价性）+ L2

**注意**：`nixlXferDList` 在任何 NIXL 版本下都只接受 Nx3，
strided 只能经 `prep_xfer_dlist`（§3.6.3）。实现时不要误判为功能缺失。

---

### A3 — post/wait 拆分（设计项 5b）★ 阶段 B 的前置

**落点**：`psrl/utils/nixl/client.py:1519-1544`（`merge_and_finish_cached_xfer`），
`client.py:1572`（`wait` 的轮询循环）

**做什么**：
1. 把 `merge_and_finish_cached_xfer` 拆成两个方法：
   - `post_cached_xfer() -> list[handle]`：只提交，立即返回 handle
   - `wait_xfer(handles, timeout)`：等待这批 handle
2. 完成检测从 `time.sleep(0.001)` 轮询改为 `get_new_notifs`：
   发起方在 `make_prepped_xfer` 时带 notif 标签，完成时收到通知
3. 保留轮询作为 notif 不可用时的回退，但**默认走 notif**

**判据**：
- 存在一个测试：连续 post 两个 chunk，断言**第二个 post 的返回时刻
  早于第一个 wait 的返回时刻**。这是双 buffer 成立的充要条件
- 单轮完成检测延迟从「最多 1ms」降到 RTT 量级（< 50μs）
- chunk 下界从 256MB 降到 RTT 决定（§3.6.4 的表）

**为什么这是前置**：现有实现提交后立刻进轮询循环、不返回控制权，
`post(i) → wait(i) → post(i+1)` 严格串行，**「深度 2 双 buffer」无处落脚**。
先做 B2 会得到一个「看起来有流水、实测没有重叠」的实现。

**验证**：L1（post/wait 交错顺序断言）+ L2（端到端重叠率）

---

### A4 — `make_xfer_tag` 去 pickle（设计项 6c 的一部分）✅ 已落地

**落点**：`psrl/utils/nixl/client.py:42`

**做什么**：`pickle.dumps` 换成 UTF-8 NUL 分隔编码，单 shard 走 f-string 快路径

**判据**：140 万 key 规模下 tag 构造 < 40ms

**实际结果**：**55ms → 20ms（140k descriptor）**。
注意**主文档原来的估计是错的**：它暗示 pickle 开销是「秒级」，
实测 pickle 单次约 0.4μs，140k 次共 55ms。已回改主文档。
**收益真实但有限**——这个规模下主导开销是 NIXL 自己的 descriptor 处理。

两处实现细节值得记：
1. `target_client` 在 `wait()` 里是 `str | None`（`client.py:1553`），
   所以 `None` 可达，必须用哨兵值与空字符串区分，否则两个不同传输会碰撞
2. `(1,2)` 与 `(12,)` 必须不碰撞——无分隔符的朴素拼接会碰撞

**验证**：`tests/nixl/test_xfer_tag.py`（9 个测试全过，含单射性与 MoE 规模性能）

---

### A5 — 配置期防御五项（设计项 9d-a）

**落点**：`psrl/utils/config.py`、`psrl/trainer/ppo/ray_trainer.py:1962`、
`psrl/utils/converter/fsdp_converter.py`（`get_sharding_for_param`）

**做什么**：

| # | 改动 | 落点 |
|---|---|---|
| a | `train_model_dtype` 真实读 `engine_config.model_dtype`，不再按后端硬分流 | `ray_trainer.py:1962` |
| b | `param_offload` **与** `offload_policy` 两个 flag 分别校验并拒绝 | `config.py` |
| c | FSDP2 uneven sharding 在**声明期**断言并指向调小 `fsdp_size` | `fsdp_converter.py` 的 `get_sharding_for_param` |
| d | 握手期打印全局 finest mesh 各维大小 | `psrl/utils/nixl/server.py` 的 `make_unified_sharding` |
| e | **LoRA `merge=false` 拒绝**（§3.11） | `config.py` |

**判据**：五条各有一个故意违反的测试，断言**在配置校验期报错**而非传输期。
特别是 (a)：用 `model_dtype=bf16` 启动，断言注册给 NIXL 的字节 dtype 是 bf16 而非 fp32；
(e)：`lora_rank>0 and not lora.merge` 必须 raise，错误信息包含 `merge=true` 这条替代路径。

**为什么全是「配置期」**：这五条不做的话失败形态分别是
「传输期报错而非校验期」「按错 dtype 准备缓冲」「LoRA 下静默同步基座模型」，
**都是难归因的失败**，而 (e) 甚至完全静默（reward 不涨、无报错）。

**验证**：L1（五个 `pytest.raises`）

---

### A6 — 删除 FSDP1 分支（设计项 9e）

**落点**：`psrl/utils/converter/fsdp_converter.py`

**做什么**：删除 `fsdp_strategy == "fsdp"` 的分支与 `flat_param` 相关代码路径

**判据**：`grep -n "flat_param\|StateDictType.SHARDED_STATE_DICT" fsdp_converter.py` 零命中；
`convert_fsdp_inplace` 的 `fsdp_strategy` 参数若保留则只接受 `fsdp2` / `fsdp_hybrid`

**验证**：L1（已有 FSDP2 测试仍通过）

---

### A7 — join 三级判定（设计项 9c-a）★ 越早越好

**落点**：`psrl/utils/nixl/server.py`（`make_unified_sharding` 前插入准入判定）

**做什么**：

```python
def classify_join(current_finest, new_mesh, all_tensor_shapes):
    new_finest = {d: lcm(current_finest.get(d, 1), new_mesh.get(d, 1)) ...}
    if new_finest == current_finest:
        return "FAST_PATH"            # 整除，既有 descriptor 全部有效
    for key, shape in all_tensor_shapes.items():
        for dim, count in new_finest.items():
            if dim < len(shape) and shape[dim] % count != 0:
                return "REJECT"       # 不可表达，放行会让所有既有 client 抛 ValueError
    return "WARN_RESHARD"             # 可表达但要全局重分片
```

**判据**：三级各有一个测试
- `FAST_PATH`：hidden=4096、current=lcm(8,4)=8、新成员 TP=16 → finest=16，
  断言既有 client 的 `shard_mesh` 与 `shard_indices` **逐字段不变**
- `WARN_RESHARD`：hidden=1536、新成员 TP=3 → finest=24，1536%24==0 →
  断言告警已打印**且加入成功**，重分片后逐张量内容正确
- `REJECT`：hidden=4096、新成员 TP=3 → finest=24，4096%24!=0 →
  断言准入期被拒绝，**且既有 client 仍处于可服务状态**

**为什么第三级是正确性**：`get_local_sharded_tensors`（`nixl_spec.py:166`）
要求 `shape[dim] % count == 0`，不整除会让**所有既有 client** 在重分片时抛
`ValueError`——一次扩容变成全链路故障，而不是「变慢」。

**验证**：L1（三级判定的纯函数测试，不需要真实 NIXL）

---

### A8 — colocate CUDA IPC + stage offload（设计项 2）

**落点**：`psrl/workers/train/engine_train_worker.py`（`switch_to_*_mode`），
`psrl/workers/gen/vllm_async_server.py`（KV cache 释放与恢复）

**做什么**：
1. colocate 路径走 CUDA IPC（`reduce_tensor` / `rebuild_ipc`）pack 进**一块可复用
   uint8 buffer**，**完全不碰 NIXL 注册**
2. 切换顺序严格按 verl 的 `engine_workers.py:811-826`：
   `pause_generation` → `release_kv_cache_memory()` → transfer →
   **train 权重 offload 到 CPU** → `wake_up(["kv_cache"])`
3. 稳态切回 trainer 从 **CPU reload** 而非读 checkpoint（§3.3.1a 末段），
   判据是独立的一次性标记「trainer 本进程是否曾持有过真实权重」，
   **不能用 `is_rollout_mode_in_actor`**（它被两个 switch 函数来回翻转）

**判据**：
- colocate 路径上 `register_memory` 调用次数 **为 0**
- eval 期间 KV cache 可用量与「纯 vLLM sleep/wake」基线相当（差异 < 5%）
- 整个 transfer 期只存在**一块** IPC buffer，大小等于配置值、不随模型或张量数增长
- 连续两轮 eval，第二轮切回 trainer 后权重版本 **等于 eval 前的版本**
  （不是上次 save 的版本）——这条测「稳态切回不读盘」

**验证**：L2（新增 colocate 模式）+ L1（一次性标记的状态机）

---

### A9 — 源头 FP8 wire（设计项 4）

**落点**：新建 `psrl/utils/fp8/`，调用 vLLM 与 TE 的现成原语

**做什么**：按 §3.8 的依赖表组合 vLLM / TE 原语，不自己实现量化算法：
- blockwise：`vllm/utils/deep_gemm.py::per_block_cast_to_fp8`
- per-tensor/group：`_custom_ops.py::scaled_fp8_quant`、`fp8_utils.py::per_token_group_quant_fp8`
- 落地格式化：`fp8_utils.py::{process_fp8_weight_block_strategy, requant_weight_ue8m0_inplace}`
- 合法性校验：`fp8_utils.py::{validate_fp8_block_shape, validate_fp8_block_shape_moe}`

**判据**：
- 小模型上逐张量比对「本设计产出」与
  「vLLM 从原生 fp8 checkpoint 加载 + `process_weights_after_loading`」，
  在 UE8M0 / requant 容差内一致（§7 验证项 8）
- 传输字节相对 bf16 **减半**
- `weight` 与其 `scale` 在同一个 commit group（原子性）

**验证**：L2（精度矩阵的 fp8 行）

---

## 3. 阶段 B：执行层

### B1 — chunk / commit group 分离（设计项 6d）★ C1 的前置

**落点**：`psrl/utils/nixl/comm_plan.py`、`client.py`

**做什么**：把现在一个 `bucket` 承担的六个角色拆成两个概念（§3.6.1）：

| 概念 | 单位 | 服务于 | 可否切开一个 key |
|---|---|---|---|
| `chunk` | 固定字节数（自适应） | 流水、带宽、W、wire buffer 复用、holder 位图 | **可以** |
| `commit_group` | 引擎 loader 的原子要求 | 版本可见性、落地格式化 | **不可以** |

两者多对多：一个 chunk 含多个小 group，一个大 group（MoE fused `w13_weight`）跨多个 chunk。

**判据**：
- 存在一个测试：一个 11GB 的单 key 被切成多个 chunk 传输，
  且该 key 的 commit group 在**全部 chunk 到齐后**才标记可见
- 故意只送达 group 的一半，必须**报错**而不是静默 no-op（§7 验证项 7）
- 调整 chunk 大小不影响 commit group 边界（两个概念真的解耦了）

**验证**：L1（`test_comm_planner.py` 扩展）+ L2

---

### B2 — 三类重叠 + slot ack bitmap（设计项 5）

**落点**：`psrl/utils/nixl/client.py`（流水调度），`psrl/workers/ps/ps_manager.py`（ack 上报）

**做什么**：

1. **三类重叠**：cast / RDMA / 落地格式化按 commit group 流水，
   第 i 组的格式化与第 i+1 组的 RDMA 重叠
2. **slot ack bitmap**（**正确性，不是优化**，§3.4.2 末段）：

```
每个 wire buffer slot 持有:
    (chunk_id, version, expected_readers: Bitmap, acked: Bitmap)

读者完成一个 chunk -> 在它本来就要发的完成通知里捎带 chunk_id
trainer 收到        -> acked.set(reader_id)
acked == expected   -> slot 可复用
holder 摘除 reader  -> expected.clear(reader_id)   ← 崩溃自动收敛
超时未填满          -> 对未 ack 的读者返回 CHUNK_STALE 让其重拉
```

**判据**：
- wire buffer 常驻量 **== `2 × chunk`**，且**不随 W 增长**
- 存在一个测试：人为延迟一个读者的 ack，断言
  `conflict_wait_budget_ms` 到期后**对它返回 `CHUNK_STALE`，而不是让整条流水停住**
- 端到端时间 ≈ `max(T_rdma, T_fmt) + T_fmt/N_rounds`（流水生效）
  而非 `T_rdma + T_fmt`（完全没重叠）
- **expected 集合由 plan 派生**，不是运行期计数器：
  断言 ack 通道不引入新的控制面往返（复用 §3.2b 的完成通知）

**为什么 ack bitmap 是正确性**：`2×chunk` 循环复用下，
下一轮 cast 会覆盖慢读者正在读的字节。三个框架都用「显式归还信号 + 取不到就阻塞」
（verl 的 magic 槽、miles 的 free-queue），**没有一个靠加深 buffer**——
W 个独立 instance 的进度差没有上界，用显存赌无界量是把正确性问题伪装成容量问题。

**验证**：L2（重叠率、slot 等待分布）+ L3（慢读者注入）

---

### B3 — chunk 自适应 + 轮级决策本地化（设计项 6c 剩余部分）

**落点**：`psrl/utils/nixl/comm_plan.py`（chunk 大小计算），`client.py`（轮级决策）

**做什么**：
1. chunk 大小按 §3.6.5 的 cost model 计算，**不写死常量**：
   ```
   N_min     = ceil(max(T_fmt / T_rdma, 1) × D)        D = 2
   chunk_max = M_v / N_min
   chunk_min = 由完成检测粒度定（A3 之后是 RTT 级，A3 之前是 256MB）
   取 chunk = clamp(512MB, chunk_min, chunk_max)
   ```
2. **轮级决策本地化**：源选择在拉取开始时一次性解析完整轮次，不每轮查 index

**判据**：
- 换一个大一倍的模型，chunk 数变化而 chunk **字节数**按公式变化（不是固定值）
- 每轮传输**零跨进程往返**（trace 中无 index RPC）
- cost model 的三个输入（`T_fmt`、`T_rdma`、`BW_NIC`）可从配置或探测得到，
  不需要人工填

**验证**：L1（cost model 纯函数）+ L2（轮级 RPC 计数为 0）

---

## 4. 阶段 C：元数据与控制面

### C1 — canonical MoE 分组（设计项 6）★ 大 MoE 硬阻塞

**落点**：`psrl/utils/converter/model_mappings.py`、`vllm_converter.py`（`FUSED_MOE_W13_SPLIT` 等）

**做什么**：
1. `experts.{i}.gate_proj.weight` 合并为 `experts.gate_proj.weight` **一个 key**，
   expert 维度表达为 `NIXLSharding` 的一个分片维度
2. 删除主动拆 fused 张量的逻辑（`vllm_converter.py:400` 起）——
   vLLM 的 fused loader 本来就要 3D 的 `w13_weight`，拆开反而是额外工作
3. `ETP != 1` 时 shard mesh 变成 `{0: EP, 1: ETP}` 两维，
   **必须验证 `find_finest_shard_mesh` 的 lcm 在两维上正确**

**判据**：
- Kimi-K2 量级模型的 key 数从 **14 万降到数百**
- 单 client `client_info` 的 pickle 体积从 **56MB 降到数百 KB**
- `T_fmt` 从 846ms 降到 **~18.5ms**（§3.6.5 cost model 的预测，实测校准）
- MoE 的 `shard_dim=0`（切 expert）与 `shard_dim=1`（切 projection）
  两条路径分别端到端跑通

**为什么是硬阻塞**：不做这一项，1TB 场景连握手都过不去
（数十 GB 的 pickle 流量 + 等量反序列化 CPU）。
而且 §3.6.0 的 cost model 表明：**分组前 `T_fmt` 主导，overlap 做了也看不到收益**。

**验证**：L1（两维 lcm）+ L2（MoE 模型端到端）+ L4（1TB 规模）

---

### C2 — 控制面下热路径（设计项 6b）★ 阶段 D 的前置

**落点**：`psrl/workers/ps/ps_manager.py`、`psrl/grpc/ps_manager_service.py`

**做什么**（按「消除 > 降频 > 加速」排序，§3.2.1）：

| 措施 | 做什么 |
|---|---|
| a 消除 | plan 握手期编译、rollout **本地缓存**，稳态拉取**完全不查 PS**；只有 holder 集合变化才由 PS 主动失效通知 |
| b 降频 | 租约按 `(instance, version)` 聚合，不按 chunk。消息量 `O(instances × chunks)` → `O(instances)` |
| c 降频 | 版本发布用推送而非轮询（现有 `set_ps_model_version` 已是此形态，保持） |
| d 加速 | 准入与租约两个高频读走 gRPC + 内存快照，**不进 `@_state_locked` 临界区**；写路径仍走 Ray actor 保持强一致 |

**判据**：
- 稳态同步期间，rollout → PS 的 RPC 次数 **为 0**（plan 已本地缓存）
- 控制面消息数**不随 replica 数平方增长**（§7 验证项 3）
- `@_state_locked` 只保护写路径

**验证**：L2（RPC 计数）+ L4（64 instance 规模下的控制面消息数）

---

### C3 — 版本发布挂 `on_train_step_end`（设计项 7b）

**落点**：`psrl/workers/train/engine_train_worker.py`

**做什么**：把版本发布挂在 Megatron-Bridge 的 `on_train_step_end` 回调上
（已查证可用：`callbacks.py` 的 `VALID_EVENTS`、`train.py:518`）

**判据**：发布时刻落在「权重刚就位且 NIC 刚空闲」——
trace 中发布事件紧跟 param all-gather 完成，且不与 backward 的 NCCL 重叠

**注意**：这是**一个回调**，不是阶段上报协议（后者已移出本期，§3.5.2.2）

**验证**：L2（发布时刻的 trace）

---

## 5. 阶段 D：trainer 直供（不可分割块）

> **这一块一旦开始必须完成**，因为它删除 PS 数据面。
> 开始之前，阶段 A/B/C 必须已稳定运行——这样出问题可以二分定位到 D 内部。

### D1 — 三阶段去冗余规划（设计项 7）

**落点**：`psrl/utils/nixl/comm_plan.py`（新 planner）

**做什么**：
1. 三阶段 plan：receive / external-exchange / internal-exchange（吸收自 WeightBridge）
2. 选源优先级（§3.5.2）：
   ```
   候选集 = {trainer} ∪ {version(peer) == v 的 holder}
     1. 同节点 peer（NVLINK/PCIE）    ← 第一优先，不占 NIC
     2. trainer（IB）                 ← 在飞量 < W 时
     3. 跨节点 peer（IB）             ← 仅当 trainer 在飞量 >= W   [阶段二]
     4. 皆不可用                      ← 排队，等前面完成后重新解析
   ```
3. **硬约束：只能从同版本 peer 读**。holder 位图只为当前版本 v 维护一份，
   版本推进即整体清空
4. W 按 §3.6.5 的 cost model 给初始值，再用实测反馈收敛：
   `W_init = min(NIC_count, nodes_per_instance)`
5. **禁止链式 peer**（单跳，§3.5.4）

**判据**：
- trainer 出口字节 **== `M`**（不是 `N×M`）
- 每 rollout 节点入口 **== `M_v = M/num_nodes`**
- 与 `T_min` 的比值可报告（WeightBridge 在 16~64 节点报 `1.27×`~`2.12×`，可作参照）
- 固定 trainer、逐步增加 rollout replica，AGST 应为**亚线性**
- 断言**不存在 A 读 B 而 B 读 C** 的链式结构

**验证**：L2（流量计量）+ L3（peer 扩散）+ L4（亚线性扩展）

---

### D2 — 版本可见性 + 运行时断言 + holder role 维度（设计项 8）

**落点**：`psrl/workers/ps/ps_manager.py`（holder 表）

**做什么**：
1. holder 表加 **role 维度**：`(role, version, key, shard) -> holder`
   —— **正确性前提**，否则 actor 的 v3 与 critic 的 c3 互相覆盖（§3.10.3）
2. **四条运行时断言**（§3.4.6），必须是断言不是文档要求：

| 不变量 | 检查点 | 违反后果 |
|---|---|---|
| holder 即注册有效 | pull 准入时校验 holder 身份 | RDMA 读到已释放页，静默损坏 |
| sleep 前在飞读者归零 | colocate 切换路径 | 同上 |
| commit group 完整性 | 标记版本可见前 | 半更新模型 |
| NCCL 通信组存活 | 角色切换前后 `assert torch.distributed.is_initialized()` | 切换成本涨一个量级且不报错（防回归，§3.10.4b） |

3. 在飞读者计数：**per-chunk 整数计数器**，不是「版本×chunk 租约表」

**判据**：四条断言各有一个故意违反的测试，确认**报错而非静默通过**；
role 维度：actor 与 critic 同时发布版本（v3 / c3），断言解析到不同 holder

**验证**：L1（断言）+ L2（role 隔离）

---

### D3 — trainer 直供 + 删除 PS worker（设计项 9）

**落点**：`psrl/workers/ps/ps_storage_worker.py`（**删除**）、
`ps_worker_group.py`（**删除** `create_ps_worker_group`）、
`ps_manager.py`（只留索引功能）、
`psrl/workers/train/engine_train_worker.py`（删除 `ray_push_model` 等 push 路径）

**做什么**：
1. 删除 `PSStorageWorker` 与 `create_ps_worker_group`
2. 删除 push 路径（`ray_push_model` / `push_model_state_dict_cpu` / `cpu_ref` 等）
3. rollout 的 pull target 从 `ps_for_pull` 换成 train client
4. 删除 `ps_mode` 枚举（四个值里 `nixl_gpu` 至今 `NotImplementedError`，
   其余三个都在描述「副本放哪」，而本设计没有副本）
5. `ParamSyncPlan` 的 push 成对语义改造——不改会让 rollout 读到被
   `after_push` 还原过的值（§3.3）

**判据**：
- 稳态下**无 PS worker 进程**，host 侧**无权重常驻**
- CPU 侧权重副本 `N×M → **0**`
- cast 显存上界为 `2×chunk` 而非 `O(M)`。
  **判据是「换一个大一倍的模型，这个数字不变」**
- 布局转换产物的常驻量**单独记录**（它确实随模型增长，约 `0.15×M`），
  两者不要混在一个指标里

**验证**：L2（无 PS 进程、显存上界）+ L4（1TB 规模）

---

### D4 — holder 对称化 + bootstrap（设计项 9b）

**落点**：`psrl/workers/ps/ps_manager.py`、`psrl/workers/ps/broadcast.py`

**做什么**（§3.3.1a）：
1. holder 类型对称化：`TRAIN_GPU` / `ROLLOUT_GPU` / `SELF`，
   **trainer 不再被假定为唯一权威源**
2. `is_rollout_mode_in_actor=True` 启动：
   ```
   rollout 每节点一个代表从 checkpoint 读自己需要的 shard → 注册为 holder，版本 = v0
   其余 rollout 走同版本 peer 扩散拿到 v0（复用 broadcast.py 的树形拓扑，
                                          源头从 PS rank 0 换成 rollout 代表）
   trainer 首次被唤醒 → 从 checkpoint 加载训练起点，不从 rollout pull
   ```
3. **首次 vs 稳态的判据是独立的一次性标记**，不能用 `is_rollout_mode_in_actor`

**判据**：
- 启动时全 GPU 跑 rollout 能拿到 v0
- **首次**切回 trainer 从 checkpoint 拿到训练起点
- **稳态**切回从 CPU reload，权重版本等于 eval 前的版本（**不是上次 save 的版本**）
  —— 读盘会静默回退训练进度且不报错

**为什么首次要读盘**：cast 路径下 rollout 持有的是 fp8/fp4 权重，
**不能作为 bf16 训练的起点**（反量化有损，且 optimizer master params 需要 fp32）。

**验证**：L2（三段式全流程 + 稳态单测）

---

### D5 — chunk 起始偏移错开（设计项 7c）

**落点**：`psrl/utils/nixl/comm_plan.py`

**做什么**：instance `i` 从 `chunk (i × num_chunks / R) mod num_chunks` 开始，环形推进

**判据**：
- 偏移**与 commit group 边界对齐**（不能把一个 group 切在环形起点）
- 偏移是**确定性的**（由 instance 序号导出），不是随机——
  这样 index 可以在不通信的情况下预测谁先持有哪些 chunk
- 开启前后对比第一波的 **peer 命中率**提升
- 只在 `R > W` 时启用

**验证**：L3（源分布统计）

---

### D6 — `CHUNK_UNAVAILABLE` 真能继续生成（设计项 8d）★ 与 D7 同批

**落点**：`psrl/workers/gen/vllm_async_server.py:1036`、
`psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py:167`

**做什么**：
1. `pull_model_for_sync` 的版本 assert 改为**返回实际版本 + 标记本轮未达标**：
   ```python
   # 现状（会抛，经 _finish_gateway_sync 的 except 走到 fail_sync + /workers/pause）
   assert curr_model_version >= ps_version, "Pulled model version is stale: ..."
   # 改为：返回 (actual_version, reached_target: bool)
   ```
2. 版本发布从**乐观预发**改为 **pull 成功后发布**：
   现状是先 `_publish_weight_version_updates(ps_model_version)` 再 pull，
   失败时靠事后更正——留下「SMG 认为是 v+1、实际是 v」的秒级窗口

**判据**（§7 验证项 9）：
- (a) 该 replica 在降级后**仍在 SMG 的可用集合里**且继续服务请求——
  不是 `fail_sync` + `/workers/pause` 之后的「不可用」
- (b) 降级全程 SMG 记录的该 instance 版本**始终等于它实际持有的版本**，
  不存在「声明 v+1 而实际 v」的中间窗口。
  **必须在请求粒度上验**：降级窗口内到达的请求，其记账版本要与
  生成时实际使用的权重版本一致，否则 staleness 账本被污染且无人发现

**为什么必须与 D7 同批**：不改这一项，D7 的 DP 接管落地当天就撞 assert，
降级路径表现为「replica 被摘掉」，与「保持当前版本继续生成」正好相反。

**验证**：L3（故障注入 + 请求粒度的版本记账）

---

### D7 — DP 副本接管（设计项 8b）

**落点**：`psrl/workers/ps/ps_manager.py`、
`psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py:74`

**做什么**（三层响应，§3.5.7）：
```
① 单 rank 故障（Megatron/HSDP；纯 FSDP2 下不可用）
   index 摘除该 rank → 同 DP 组其他 rank 接管（内容相同）→ 受影响 pull 重新解析源
② 整个 DP 组故障
   返回 CHUNK_UNAVAILABLE（显式返回值，不是超时）→ rollout 保持当前版本继续生成
③ trainer 全体重启
   holder 表整体清空，版本号保留 → 所有 pull 立即 CHUNK_UNAVAILABLE
```
**holder 摘除要能被动触发**：rollout 的 `client_read` 报错时，
连同失败的 holder 身份一起上报 index——把故障检测搭在已有错误路径上，不需要心跳。

**判据**：
- 杀掉一个 trainer rank，pull 由同 DP 组接管且 rollout **无感知**
- 杀掉整个 DP 组，返回 `CHUNK_UNAVAILABLE` 且 rollout 保持当前版本继续生成，
  **不卡死、不读到半更新模型**
- 复用 `_quarantine_failed_replicas` 路径，不新建一套

**验证**：L3（故障注入）

---

### D8 — FSDP2 副本数运行时推导（设计项 9d-b）★ 与 D7 同批

**落点**：`psrl/workers/ps/ps_manager.py`（副本数推导）

**做什么**：副本数必须由 index 从 `NIXLSharding` **运行时推导**，
不能假定「DP 副本一定存在」。`replica_count == 1` 时 D7 的 ① 直接跳到 ②。

**判据**：纯 FSDP2 下杀一个 rank，必须返回 `CHUNK_UNAVAILABLE`
而**不是**尝试 DP 接管；HSDP 下同一测试必须走 DP 接管

**为什么与 D7 同批**：先写死 Megatron 假设再回头改会返工。

**验证**：L3（两种后端的同一故障场景）

---

## 6. 阶段 E：正确性补强

### E1 — 版本级等待 + per-chunk 写屏障（设计项 8c）

**落点**：`psrl/workers/ps/ps_manager.py`、`psrl/workers/train/engine_train_worker.py`

**做什么**：

```
版本级等待（FSDP2 bf16 同精度下【必需】，Megatron 同精度下是优化）：
  发布 v+1 之前，等待对 v 的在飞读者计数归零（一次性，不分 chunk）
    上限 conflict_wait_budget_ms，建议默认为单步时长的 5%
    超限则发布 v+1 并把 v 标记为 INVALID
      在飞读者收到 INVALID → 放弃本轮，重拉 v+1（更新的权重，算法上是改善）

per-chunk 写屏障（始终是优化）：
  仅 Megatron 同精度路径可用——master params 让冲突窗口窄到「param 回写那一瞬」
  FSDP2 bf16 下整个 optimizer step 都在写，降粒度没有收益
```

**判据**：
- **`FSDP2 + bf16 + bf16 + disaggregate` 是必跑项**：
  关掉版本级等待后**必须能观测到权重被破坏**（读到半新半旧）——
  这是该机制存在价值的判据
- 超时后被打断的 rollout **重拉更新的版本**，而不是「继续用旧的」
- cast 路径的「等 optimizer」等待恒为零

**启用 FSDP2 + `model_dtype=bf16` + bf16 rollout 之前必须完成这一项。**

**验证**：L2（精度矩阵的该格 + 关闭机制后的破坏观测）

---

### E2 — 非整除 join 的重分片走版本边界（设计项 9c-a2）

**落点**：`psrl/utils/nixl/server.py`、`ps_manager.py`

**做什么**：A7 判定为 `WARN_RESHARD` 时，重分片必须在版本边界进行：
```
撤销 holder → 等在飞读者归零 → 重分片 → 推进 registration_epoch → 重新广播
```
**epoch 必须推进**，尽管地址没变——「同一地址的不同切分」对远端同样是陈旧描述。

**判据**：构造「version 匹配但 epoch 陈旧」的请求，必须返回 `DESC_STALE` 而非通过；
`CHUNK_STALE` 与 `DESC_STALE` 触发**不同的恢复路径**（换源 vs 重取 descriptor）

**验证**：L3（两段校验）

---

### E3 — 握手改事件驱动 join / leave（设计项 9c-b）

**落点**：`psrl/utils/nixl/server.py:56,90`、`psrl/workers/ps/ps_manager.py:907`

**做什么**：`nixl_protocol()` 的八步直线改为事件驱动循环。
现状 `_is_all_client_shardings_recved` / `_is_all_client_infos_recved`
一旦置真就直接 return，**新 instance 没有入口**。

**判据**：运行中的集群能接受新 instance 加入（走 A7 的三级判定），
且既有 instance 不中断服务

**验证**：L3（运行中扩容）

---

### E4 — partial MD 两段式 + `invalidateLocalMD`（设计项 9c-c）

**落点**：`psrl/utils/nixl/client.py`（握手与退出路径）

**做什么**：
1. **join 分两段**：
   ```
   第一段（所有人）：conn-info-only blob，401 B
   第二段（只有真正的对端）：按 plan 解析出的那部分 descriptor
                            用 get_partial_agent_metadata 而非全量
   ```
2. **缩容四步**（现状缺第三步）：
   ```
   撤销 holder 资格 → 等在飞读者归零 → invalidateLocalMD() → sleep/释放显存
   ```

**判据**：
- 每次 join 时每个既有成员加载的 MD 字节数为 **conn-info 量级（~400B）**
  而非全量（~406B × region 数）
- 关掉 `invalidateLocalMD` 后，让远端用成员退出前的 descriptor 发起读，
  **必须失败而非读到垃圾**

**实测依据**：全量 MD 约 406 B/region 线性增长（4096 区 = 1.6MB）；
partial（4096 区中发布 1 个）= 883 B；conn-info-only = 401 B，**压缩 1884×**。

**验证**：L3（join 扇出字节数 + 退出后的陈旧 descriptor）

---

## 7. 阶段 F：正交可选

### F1 — delta 传输（设计项 10）

**落点**：新增，与主路径正交

**做什么**：只传变化的 `(position, value)`，参考 verl 的 `delta_sharded`

**判据**：跨 DC 或慢链路场景下传输字节降到 1%~3%

**注意**：它需要一份 pinned 快照作 diff base，**在约束 (b) 下这本身就是一份额外
CPU 权重存储**，所以只能是**显式 opt-in 的例外**，不能作默认。

**验证**：L4（跨 DC）

---

## 8. 测试改造（§7.0）

### 8.1 三处必须先做的结构性改动

**(a) 正确性判据从「全 1」升级为坐标函数 ★ 第一优先级**

现有 `assert_state_dict_all_ones`（`test_nixl_e2e.py:95`）**对本设计要验证的东西完全无效**：

| 错误类型 | 全 1 判据能否检出 |
|---|---|
| 数据没到 | 能 |
| **布局错误**（转置、reshape 错） | **不能**——全 1 转置后还是全 1 |
| **分片错位**（读到别的 rank 的 shard） | **不能**——全 1 的任意切片都是全 1 |
| **精度转换错误** | **不能**——全 1 cast 到 fp8 还是 1.0 |
| **MoE expert 映射错误** | **不能** |

替换为**每个元素等于其全局坐标的确定性函数**：
```python
def fill_by_global_coord(key, tensor, global_offset): ...
def assert_matches_global_coord(key, tensor, global_offset, dtype_tol): ...
    # 失败时报出第一个不匹配的坐标 —— 直接定位到是哪一维错位
```
**两个约束**：值域必须落在目标 dtype 可精确表示的范围内
（fp8 e4m3 只有 ~3.5 位尾数，**不能用大整数做哈希值**），
且 `f` 必须对坐标**单射**，否则错位换位检不出来。
实现上用**小整数的位拼接**（`layer_idx * 2^k + row * 2^j + col`），不是浮点哈希。

**(b) 删除 `create_ps_worker_group`（`:649`）**，`GenClientActor` 的 pull target
从 `ps_for_pull_name` 换成 train client。

**(c) 新增 colocate 模式**（现有测试只有 disaggregate）。
**这是改造里工作量最大的一项**，但它是 §3.7 全部结论的唯一验证手段。

### 8.2 四维参数化矩阵

| 维度 | 取值 | 现状 |
|---|---|---|
| 后端 | `megatron` / `fsdp2` | 已有（删 `fsdp1`，`fsdp_hybrid` 保留为 HSDP） |
| 精度 | `bf16+bf16` / `bf16+fp8` / `fp32+bf16` / `fp32+fp8` | 配置项已有，**cast 路径待实现** |
| 放置 | `disaggregate` / `colocate` | **需新增 colocate** |
| 并行度 | train TP/PP/EP × gen TP/EP（含不匹配） | 已有，只需扩参数化 |

**不做全笛卡尔积**，按风险选组合：
```
必跑（每次 CI）：
  megatron + bf16+fp8  + disaggregate + TP匹配      ← 推荐默认配置
  fsdp2    + fp32+bf16 + disaggregate + TP匹配      ← FSDP2 默认配置
  megatron + bf16+bf16 + colocate     + TP匹配      ← 时序互斥路径
每日跑：
  上述 × {TP 不匹配、EP 不匹配}                      ← shard 代数
  fsdp2 + bf16+bf16 + disaggregate                  ← 版本级等待必需的那一格
  megatron + fp8+fp8 + colocate                     ← FP8-native，显存最优
发版前：
  全部精度组合 × 2 放置 + MoE 模型（Qwen3-30B-A3B 级）
```

### 8.3 分层与覆盖

| 层 | 载体 | 覆盖验证项 | 条数 |
|---|---|---|---|
| **L1 单元** | `test_nixl_sharding.py` / `test_comm_planner.py` 扩展 + 新增 | shard 代数、finest mesh、FSDP2 uneven、MoE 两维 shard、A7 三级判定、cost model | — |
| **L2 e2e 单节点** | `test_nixl_e2e.py` 改造 | 7、8、11、14、15、16、17、17b、19、22、23 | **11** |
| **L3 e2e 多节点** | 新建 | 5、13、15b、18、20、21、24 | **7** |
| **L4 规模实测** | 人工，无法自动化 | 1–4、6、9、10、12、25 | **9** |

**L4 的 9 条是本方案的真实风险**——它们恰恰支撑着「万卡 / 1TB」的声明，
而只能在真实集群上验证。**这一点必须在落地计划里显式承认**，
不能把「设计上可扩展」与「已验证可扩展」混为一谈。

---

## 9. 完成度检查表

实现期逐项打勾。**判据栏必须可执行**，不接受「已实现」这种描述。

### 阶段 A
- [x] **A1 已落地并验证**：`psrl/utils/nixl/prepped_xfer.py` 的 `PreppedXferCache`
      接入 `_initialize_merged_xfer`，热路径 `initialize_xfer` 调用数 **== 0**（真实 agent 实测）；
      逐字节正确（含 rotated / subset / 重注册）；**2048 descriptor 下提交快 >2×**，
      且优势随 descriptor 数增长。`test_prepped_xfer_cache.py` 29 passed +
      `test_prepped_hotpath_gpu.py` 7 passed。

      **实现中发现三处会让收益归零的陷阱**（都已修并有回归测试）：
      1. `prep_xfer_dlist` 的 `mem_type` 要字符串，merged 路径给的是 `nixl_mem_t` 枚举 →
         裸 `KeyError`，被 fallback 吞掉只留一条 warning
      2. **每次提交重建索引数组是 O(n)**（0.085μs/desc），比 `initialize_xfer` 还慢 5 倍 →
         必须缓存索引数组
      3. **`tuple(wanted)` 做缓存 key 也是 O(n)**（0.027μs/desc）→ 需要 identity 快路径
      最终 4096 descriptor 下 cache 开销从 0.110ms 降到 0.006ms（18×）
- [x] **A2 已落地并验证**：`psrl/utils/nixl/strided_desc.py` 的 `describe_shard`
      判定哪些非连续 shard 可免中转拷贝。strided 与逐条 Nx3 **逐字节一致**。
      `test_strided_desc.py` 35 passed。

      **一处事实更正**：初稿称中转槽是「额外 CPU 权重存储」，
      读 `client.py:541` 后发现分配在 `device=self.device` 即 **GPU** 上
      （变量名 `pinned_memory` 误导）。主要收益是**消除槽位串行等待**，显存次之。

      **三处测试前提被实现推翻**（实现是对的，测试错了）：3-D 张量切中间维
      **是** strided 可表达的（尾部维连续，逐字节验证通过）；
      `torch.flip` 产生拷贝而非负 stride 视图；
      PyTorch 的 `as_strided` 根本拒绝负 stride，该分支只能直接测谓词
- [x] **A3** 「第二个 post 早于第一个 wait 返回」已有断言 — `test_post_wait_split.py` 15 passed
- [x] **A4** 140k tag 构造 20ms（预算 40ms）；单射性含 `(1,2)`vs`(12,)`、None vs ""
      — `tests/nixl/test_xfer_tag.py` 9 passed
- [x] **A5** 五条配置期防御；`model_dtype=bf16` 被真实读取
      — `tests/nixl/test_config_defenses.py` 18 passed / 4 skipped（见下）
- [x] **A6** `flat_param` / FSDP1 分支已删；e2e harness 迁到 `fully_shard`
- [x] **A7** 三级判定；**56 组穷举与真实 runtime 逐一对账**
      — `tests/nixl/test_join_admission.py` 76 passed，
      `tests/nixl/test_meta_server_sharding.py` 10 passed
- [x] **A8 CUDA IPC 机制已验证**（`test_colocate_ipc_gpu.py` 10 passed）：
      跨进程别名双向可见、一块 uint8 bucket 承载 bf16/fp32/fp8/fp16、
      fp8 weight+scale 共存、buffer 不随模型增长、**`register_memory` 调用数 == 0**。
      **KV cache 对比与稳态切回需 vLLM engine，属 L2**
- [x] **A9 fp8 量化已验证**（`test_fp8_wire_gpu.py` 19 passed）：
      blockwise round-trip 在 fp8 误差内、**wire 字节 0.45~0.55× bf16**、
      scale 网格覆盖完整、丢 scale 错 100× 以上、量化确定性。
      **与 vLLM 原生 fp8 checkpoint 逐张量对账需 engine，属 L2**

**单机验证总计：510 passed, 23 skipped**（含全部 GPU 测试）。
FSDP2 converter 在 2 卡与 4 卡各验证 27 个张量（torchrun，独立运行）。

**A5 的 4 个 skip 不是未完成**：`fsdp_converter` 导入 `verl.utils.model` →
`transformers.AutoModel`，在本环境因 numpy 2 冲突而失败，**改动前后同样失败**
（已用 `git stash` 对照确认）。该断言是纯 shape 检查，已用 AST 抽取函数体单独验证通过，
测试用 `importorskip` 跳过而非失败，以免掩盖真实回归。

### 阶段 B
- [x] **B1 已落地并验证**：`psrl/utils/nixl/chunking.py` 分离 chunk 与 commit group。
      11GB 单 key 切 22 个 chunk 且偏移正确；**group 全齐才可见，半个 group 抛错并点名缺哪些 chunk**；
      **chunk 大小改变不影响 group 成员**（4 种 chunk size 验证）。`test_chunking.py` 22 passed
- [ ] **B2 接线推迟到阶段 D**（模块已完成，见下）。
      `slot_ack.py` 保护的是「`2×chunk` 循环复用时下一轮 cast 覆盖慢读者正在读的字节」，
      而 **cast 产物作为发布缓冲是阶段 D（trainer 直供）才引入的东西**：
      当前链路里 rollout 从 PS 的 pinned 缓冲读，不存在循环复用的 wire buffer
      （grep `wire_buffer` 与 cast 路径在 `psrl/` 零命中，仅 `chunk_size.py` 的参数名）。
      依赖图写的「B2 依赖 A3」在模块层面成立，但**实际的 wire buffer 由 D 创建**，
      所以 B2 应与 D6/D7/D8 同批，而不是现在接到一个不存在的调用点上。
- [x] **B2 模块已完成并验证**：`psrl/utils/nixl/slot_ack.py` 的 `SlotAckTable`。
      **2 个 slot 服务 16 个 reader**（不随 W 增长);slot 耗尽时**抛错而非覆盖**,
      错误点名在等哪个 reader;**`forget_reader` 让死掉的 reader 自动收敛**——
      这正是选 bitmap 而非计数器的理由(计数器会永久泄漏)。`test_slot_ack.py` 20 passed
- [ ] B3 chunk 按 cost model 计算；轮级 RPC == 0

### 阶段 C
- [x] **C1 已落地并验证**：`psrl/utils/nixl/moe_grouping.py` + `BaseConverter.maybe_group_experts`。
      **真实 checkpoint 端到端实测**（经 HF converter）：Qwen3-30B-A3B 18867 → 579 key（32.6×），
      144 个 expert group，每个 key 128 shard、`shard_mesh {0: 128}`；
      `client_info` pickle **3.475MB → 0.324MB（10.7×）**；dense 模型零改动（Qwen3-1.7B 311 key，无分组）。
      **关键正确性**：expert id 必须**数值排序而非字典序**——
      字典序会把 expert 10 排在 2 前面,静默置换所有 expert。
      5 节点 e2e 通过（16 个 gen client 全部 399/399 坐标匹配）。
      `test_moe_grouping.py` 42 + `test_shard_meta_interning.py` 12 + `test_moe_grouping_gpu.py` 9 passed

      **三处设计文档未写明、由实测确定的事实**：
      1. **两侧 expert 存储结构不同**：vLLM 的 `w13_weight` 是一个 fused 3-D 张量
         （expert 轴是真实张量维度），而 Megatron/TE 每个 expert 是独立的 `torch.empty`
         （TE `grouped_linear.py` 按 GEMM 注册 `weight{i}`），彼此不共享 storage。
         所以一个 grouped key 在一侧是 1 个张量、在另一侧是 N 个——`GroupedShards` 即为此而设。
         强行 stack 成单张量会在每次同步复制整个 MoE，**比分组省下的元数据还大**。
      2. **10.7× 中只有 2.5× 来自分组本身**，其余来自 **shard metadata 的 interning**：
         128 个 expert 的 metadata 对象「相等但非同一」，而 pickle 按 identity 记忆化
         （实测 7473 B vs 488 B）。不做 interning 会把 key 数的收益大半吐回去。
      3. **§3.5.5(b) 的 strided 16384:1 压缩只适用于 vLLM 侧**。
         实测真实尺寸（768×2048）下 Megatron 相邻 expert 分配**不是均匀跨步**，
         单条 strided descriptor 覆盖不了，所以训练侧的 per-expert descriptor 不可约。
         分组后随 expert 数增长的是 **descriptor 而非 metadata**，
         判据应是「远低于线性」而不是「常数」。
- [ ] C2 稳态 rollout→PS RPC == 0；控制面消息非平方增长
- [x] **C3 已落地并验证**：`psrl/workers/train/step_boundary.py` 的 `StepBoundaryRecorder`，
      NIXL 与 Ray 两条发布路径都记录。`test_step_boundary.py` 13 passed。

      **一处设计文档的事实更正**：本文原方案要求把发布挂到 Megatron-Bridge 的
      `on_train_step_end`。该回调确实存在且时机正确（`train.py` 中在 `wrapped_train_step`
      之后触发），但**在 PSRL 中永远不会触发**——它属于 Bridge 自己的 `pretrain`/`train`
      循环，而 PSRL 只用 Bridge 做**模型转换**（`megatron_converter` 的 `AutoBridge`），
      RL 的 optimizer step 走 verl 的 `train_mini_batch`；全仓库没有任何 `pretrain()` 调用。
      挂上去会**正常导入、正常运行、永不触发**，失败形态是发布 trace 恒为空。
      因此边界定义在 PSRL 实际运行的循环上。设计想要的时序其实**已经结构性成立**
      （`update_actor` 在 optimizer step 返回后立刻调 `push_model`，NIXL 路径又在所有
      shard wait 完成后才发布），缺的只是**可观测性**——而那正是判据要求验证的东西。

### 阶段 D（不可分割）

> **当前状态**：D1/D2/D5/D6/D7/D8 的**逻辑层已落地并单测通过**；
> **直供链路已在 5 节点集群端到端跑通**（无任何 PS 进程，16 个 gen client 全部 399/399）。
> 代码层面的删除（`PSStorageWorker` 等）尚未执行 —— 先让直供跑通，删除才有可对照的基准。
>
> **一条必须先说的事实**：在 `6314bde` 之前，所有多节点 e2e 的 **Ray actor 导入的是
> `psrl_main` 这个另一个 checkout，而非本工作树**。driver 跑本树、actor 跑那棵树，
> 于是测试测的是工作树里并不存在的代码 —— 而且**会通过**。
> 此前为 C1/C2/C3/D6 声称的集群验证因此作废，已全部重跑；
> 单测与 driver 进程内的字节级测量（18867→579、interning 体积、跨节点带宽）不受影响。
> 验证方法与复现见 memory 的 `psrl-ray-actors-import-other-checkout`。

- [x] **D1 已落地（模块）**：`psrl/utils/nixl/source_resolution.py`。
      trainer 出口 == 节点数（4 节点 × 8 reader 实测为 4 次而非 32 次）；
      出口随「每节点 reader 数」增长保持不变；无链式结构（逐轮断言源不是二手 holder）；
      扩散轮数对数级（16 reader ≤ 6 轮）。`test_source_resolution.py` 22 passed

      **一个实现期发现的真实 bug**：W 个 trainer 名额若按 reader 列表顺序发放，
      4 个名额会全落在 node0，trainer 服务 16 次而非 4 次 —— 正是本项要消除的 `N×M`。
      名额必须**按节点**发放：拿到一份的节点能用 NVLink 扩散，没拿到的节点下一轮还得回 trainer。
      **判据写成「数 trainer 出口次数」而非「断言发生了扩散」，才看得见这个差别。**
- [x] **D2 已落地（模块）**：`psrl/workers/ps/holder_index.py`。
      `(role, version, chunk)` 三元键；actor 的 v3 与 critic 的 v3 互不覆盖；
      pull 准入四类拒绝（非 holder / 版本不符 / 二手 holder / 角色未发布）；
      在飞读者为 per-chunk 整数，释放前断言归零且**报错点名是哪些 chunk**。
      `test_holder_index.py` 21 passed
- [ ] **D3 部分落地：直供链路已端到端打通，删除未做**。
      5 节点实测（`nixl_e2e_5node_direct.yaml`，与 PS 版仅差一行配置）：
      **exit=0，16 个 gen client 全部 399/399，运行中零 `PSStorageWorker` 进程**，
      gen 从 train client 直接拉取。PS 路径同时复测通过（exit=0，16/16，pull 1.14s）。

      **跑通过程中暴露的两个真实阻塞点**（planner 改动本身覆盖不到）：
      1. **meta server 不给 rollout 任何 descriptor**。
         `_get_relevant_client_names_for_agent` 只会把 `PS_FOR_PULL` 发给 `PULL_SIDE`，
         无 PS 时该集合为空，首次读直接报
         `target client not found in client infos`——看起来像握手 bug，
         实际是过滤器没跟上。现改为**按注册情况判断**，与 planner 的分派方式一致。
         测试同时校验 PS 路径的扇出**没有变宽**，否则「永远加上 train client」
         也能让直供用例通过。
      2. **`client_read` 在查 plan 之前就解引用对端的 tensor info**。
         PP 下每个 train rank 只持有本 stage 的 key，向 rank 0 要 rank 12 的 key
         会抛裸 `KeyError`——读起来像权重丢失，实则是正常的 PP 分片。
         对 PS 从不触发（每个 PS worker 持有全部 key），所以这个顺序一直是
         **碰巧安全**而非设计安全。`client_write` 同步加了同样的保护。

      **性能现状要如实说**：直供 pull 2.59s vs PS 路径 1.14s。
      这是预期之内且尚未处理的——当前每个 rollout 都去读**每一个** train rank，
      正是 D1 的选源要消除的 `N×M`。接线 D1 是下一步。

      **仍待做**：删除 `PSStorageWorker` / `create_ps_worker_group` / push 路径 /
      `ps_mode` 枚举，以及 `ParamSyncPlan` 的 push 成对语义改造；
      接线 D1 选源以消除 `N×M`。
      判据（CPU 权重 == 0、cast 显存上界 `2×chunk`）需删除后实测。
- [ ] D4 首次切回读盘、稳态切回从 CPU reload 且版本 == eval 前版本
- [x] **D5 已落地**：`psrl/utils/nixl/chunk_offset.py`。
      偏移**向下**吸附到 commit group 边界（逐 instance 验证每个 group 在读序中连续），
      向下而非就近是为了保持对 instance 序号单调，index 据此无通信预测 holder；
      **收益实测**：4 instance 各读 2 个 chunk 后，错开覆盖 8 个 chunk、不错开只覆盖 2 个。
      `R <= W` 时不启用。`test_chunk_offset.py` 19 passed
- [x] **D6 已落地**：`pull_model_for_sync` 的版本 assert 已删除 ——
      它会经 `_finish_gateway_sync` 的 except 走到 `fail_sync` + `/workers/pause`，
      **把「落后一个版本」变成「摘除该 replica」**，与判据正好相反。
      发布时机从「pull 前乐观预发」改为「pull 落地后按实际版本发布」，
      消除了「SMG 记 v+1 而实例持 v」的窗口。
      连带改动：原先「仅在实际版本与目标不符时才发布」依赖预发存在，
      去掉预发后达标的 replica 将永不发布，故改为无条件发布。
      `test_degraded_sync.py` 8 passed + 5 节点 e2e exit=0
- [x] **D7 已落地（模块）**：`psrl/workers/ps/chunk_availability.py`。
      单 rank 故障由同 DP 组接管且 rollout 无感知；整组故障返回显式
      `CHUNK_UNAVAILABLE`（非超时）；trainer 全体重启清空 holder 但保留版本号。
      故障检测搭在已有错误路径上（`client_read` 失败时上报 holder 身份），不加心跳。
- [x] **D8 已落地（模块）**：副本数由 `NIXLSharding` 运行时推导。
      **决定性的一对测试**：同一个单 rank 故障，HSDP（`{0:8}`/16 rank → 2 副本）走 DP 接管，
      纯 FSDP2（`{0:16}`/16 rank → 1 副本）返回 `CHUNK_UNAVAILABLE`。
      mesh 声明的分片数超过 world size 时**报错而非返回 0 副本**（后者会让所有 chunk 静默不可用）。
      `test_chunk_availability.py` 21 passed

### 阶段 E
- [ ] E1 关掉版本级等待后能观测到 FSDP2 bf16 同精度的权重破坏
- [ ] E2 `DESC_STALE` 与 `CHUNK_STALE` 触发不同恢复路径
- [ ] E3 运行中扩容成功且既有 instance 不中断
- [ ] E4 join 扇出 ~400B；关掉 `invalidateLocalMD` 后陈旧 descriptor 必须失败

### 测试改造
- [ ] T1 判据升级为坐标函数（单射 + dtype 可表示）
- [ ] T2 删除 `create_ps_worker_group`
- [ ] T3 新增 colocate 模式
- [ ] T4 四维参数化矩阵 + 三档 CI 频率

---

## 10. 与主设计文档的对应关系

| 本文阶段 | 设计项 | 主文档章节 |
|---|---|---|
| A1 | 1 | §3.6、§3.6.3 |
| A2 | 3 | §3.6、§3.5.5(b) |
| A3 | 5b | §3.6.4 |
| A4 | 6c（部分） | §3.5.5 末段 |
| A5 | 9d-a | §3.8.0、§3.9.3c、§3.9.2.1、§3.11 |
| A6 | 9e | §3.9.1 |
| A7 | 9c-a | §3.3.1d 第二层 |
| A8 | 2 | §3.7.2.1、§3.7.3、§3.3.1a |
| A9 | 4 | §3.8 |
| B1 | 6d | §3.6.1 |
| B2 | 5 | §3.6、§3.6.0、§3.4.2 末段 |
| B3 | 6c（剩余） | §3.5.6、§3.6.5 |
| C1 | 6 | §3.5.5 |
| C2 | 6b | §3.2.1 |
| C3 | 7b | §3.5.2.2 |
| D1 | 7 | §3.5、§3.5.2、§3.5.3、§3.5.4 |
| D2 | 8 | §3.4.6、§3.10.3 |
| D3 | 9 | §3.1、§3.2、§3.3 |
| D4 | 9b | §3.3.1a |
| D5 | 7c | §3.5.2.3 |
| D6 | 8d | §3.5.7 末段 |
| D7 | 8b | §3.5.7 |
| D8 | 9d-b | §3.9.1 |
| E1 | 8c | §3.4.2.1、§3.8.0a |
| E2 | 9c-a2 | §3.3.1d、§3.7.3.0a |
| E3 | 9c-b | §3.3.1d 第一层 |
| E4 | 9c-c | §3.3.1d 第三层与末段 |
| F1 | 10 | §5 第 10 项 |
| 测试改造 | — | §7.0 |

**覆盖性核对**：设计项 0–10（含 5b/6b/6c/6d/7b/7c/8b/8c/8d/9b/9c-a/9c-a2/9c-b/9c-c/9d-a/9d-b/9e）
共 **26 项**，本文覆盖 **26 项**（0 已完成，其余 25 项分布在 A–F）。
§5.1 的「待优化清单」按设计决定**不纳入**，不在本文范围内。

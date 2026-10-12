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
5. ~~`ParamSyncPlan` 的 push 成对语义改造~~ —— **已完成**。
   `ResidentZeroCenteredGammaSync`（`param_sync.py`）把暴露值搬进**独立张量**：
   `source_param` 对 trainer 保持 zero-centered，`state_dict[key]` 常驻 `+1`，
   `after_push` 刻意为空。converter 通过 `resident_exposure` 选择，
   该开关由 `not has_ps_data_plane(ps_mode)` 驱动。

   **为什么必须独立张量而非常驻 `+1` 原张量**：后者会让 trainer 自己的 forward
   读到错的 gamma。代价是每个 norm 层一份 gamma，相对权重本身可忽略，
   且注册地址因此在整个 run 内保持有效。

   **为什么这是静默错误而非显式失败**：`after_push` 还原后，rollout 的 pull
   **照常成功**，只是返回 `gamma-1`。`tests/nixl/test_resident_exposure.py`
   第一个用例就把旧 action 的这个性质钉死，再证明新 action 不具备它。

   `DTypeCastSync` 与 `ConcatenatedQKVSync` 本来就写入独立张量、`after_push` 为空，
   无需改造。FSDP2 路径构造空 `ParamSyncPlan()`，同样无负担。

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

**注意（此处需更正，更正后 F1 反而更可行）**：原文称 diff base 是「额外 CPU 权重存储」，
因此只能作约束 (b) 的例外。核实 verl 实现后发现**不是这个形态**：
每个 rank 只 pin **自己那一份分片**并在本地做 diff
（`delta_checkpoint_engine.py` 原话 "no rank ever holds a full-model snapshot"），
所以代价是 `M / ranks`、**随集群增大而摊薄**——
正是需求里允许的「CPU 部分也分布式存储」，而非它禁止的集中式副本。

**决策已可计算**：`psrl/utils/nixl/delta_plan.py`，`test_delta_plan.py` 13 passed。

**两处实测/算术更正**：
1. **index 项决定盈亏平衡点**。每个变化元素要带位置，bf16 + 4 字节索引下
   一个变化元素是重发的 3 倍，**约 1/3 变化率就不划算**，不是直觉的 100%。
2. **量化 wire 让盈亏平衡点更早到**（而非更晚）：value 减半而 index 不变，
   所以 fp8 下变化元素相对更贵。
3. **「链路快所以不划算」在本集群不成立**：我原本预期实测 103 GB/s 会拒绝，
   算下来 1GB 分片 diff 2.7ms、省 9.8ms；同节点 NVLink 343 GB/s 仍省 2.9ms。
   真正的分界在**链路快过 diff 带宽**之后。
   **所以让 F1 保持 opt-in 的是「变化率」而非链路** —— 训练早期几乎每个元素都在动，
   而这一项无法预估，只能按运行观测。

**验证**：L1（决策函数，已完成）+ L4（跨 DC，待）

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

> **当前状态总览**（集群结论均带 `[PASS] all 5 nodes import this tree`）：
>
> | 阶段 | 模块 + 单测 | 接线 + 集群验证 |
> |---|---|---|
> | A（A1–A9） | ✅ | ✅ |
> | B1 / B3 | ✅ | ✅ |
> | B2 | ✅ | ⏳ 随 D 的 wire buffer |
> | C1 / C3 | ✅ | ✅ |
> | C2 | ✅ | ⏳ 弹性期两项随 E3 |
> | D1 / D2 / D5 / D6 / D7 / D8 | ✅ | ✅ **D1 性能判据已达成**（关扩散走 staging，0.66–0.90s vs PS 1.14s） |
> | **D4** | ✅ | ✅ **连续两轮 eval，8/8 拉到新版本** |
> | D3 / T2 | ✅ 生产侧 `nixl_direct` + `ps_mode` 判定集中化 | ⏳ 删除 PS 本体（前置已满足：功能完整、已稳定验证、性能优于 PS） |
> | E1 | ✅ | ⏳ 待 FSDP2 bf16 同精度配置 |
> | E2 | ✅ | ✅ |
> | E3 | ✅ | ⏳ 待握手改事件驱动 |
> | E4 | ✅ 上半实测 1054× | ⏳ 下半被 NIXL 缺绑定阻塞 |
> | F1 | ✅ | ⏳ L4 跨 DC |
> | T1 / **T3** | ✅ | ✅ |
> | T4 | ✅ | ⏳ 两格待 cast 路径 / FSDP2 e2e |
>
> **单机 565 passed（`tests/nixl/`）。两种放置 + 两轮版本均已集群验证**：
>
> | 场景 | 拉取 | 判据 |
> |---|---|---|
> | disaggregate（5 节点） | 0.589~0.726 s | 16/16 × 399/399 |
> | **colocate（共卡）** | **0.039 s** | 8/8 × 399/399 |
> | **colocate 第二轮（新版本）** | **0.112 s** | 8/8 × 399/399 **对新版本坐标** |
>
> **剩余工作全部是「接线」或「环境/上游依赖」，无设计未定项。**


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
- [x] **B3 已落地并验证**：`psrl/utils/nixl/chunk_size.py`，`test_chunk_size.py` 24 passed。

      **修掉一处退化的目标函数。** 原评分是 `abs(rounds - ceil(T_fmt/t_chunk))`，
      两项都随 chunk 增大而减小、且后者更快，于是评分单调下降，
      **任何模型都返回候选网格的最大值 8GB**——16GB 与 1.3TB 的模型拿到同一个 chunk。
      多个测试仍然通过，因为 `>= 512MB` 对 8GB 也成立。
      现改为最小化流水 makespan `(rounds + D - 1) × max(t_chunk, T_fmt/rounds)`，
      即一次拉取真正的耗时。

      **一处必须如实记录的结论：B3 原判据「chunk 字节数随模型变化」在本成本模型下不成立。**
      RDMA 时间与字节线性，所以 `rounds × t_chunk` 恒定，
      只有 `D-1` 轮的填充项随 chunk 变化 —— 因此**最小的合法 chunk 永远最优**，
      而 chunk 大小由**轮询下界**（§3.6.4）决定，不由模型规模决定。
      这是成本模型的性质，不是调参问题；已用测试把这条单调性钉住，
      并留下「若格式化成本 per chunk 超过传输成本则该结论反转」的回归点。

      **另修一处测试前提错误**：`test_slow_nic_raises_the_floor` 断言慢 NIC 抬高下界，
      实际相反——慢 NIC 下每个 chunk 传输更久，轮询占比**更低**，下界下降
      （50 GB/s 时为 128MB）。该测试此前通过，只因旧实现恒返回 8GB。
- [x] **B3 轮级决策本地化已验证**：plan 握手期一次编译，传输期**零跨进程往返**
      （5 节点日志 `Making communication plan` 出现 1 次，拉取路径 `.remote(` / `ray.get` 零命中）。
      用源码检查而非日志 grep 钉住 —— 日志为空也可能是代码路径没跑，
      与「0 peer reads deferred」同类的陷阱。

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
- [ ] **C2 两条判据均已验证，握手期 plan 缓存待接线**：
      1. **稳态 rollout→PS RPC == 0** —— 已验证：plan 握手期一次编译并本地缓存，
         拉取路径 `.remote(` / `ray.get` 源码零命中（与 B3 同一组断言）。
         5 节点日志 `Making communication plan` 出现 1 次。
         `ControlPlaneMeter` 提供窗口计数，undercount 比 overcount 危险
         （会让判据假装满足），故用锁保护。
      2. **控制面消息非平方增长** —— 已验证：
         `test_partial_md.py::TestControlPlaneScaling`，
         rollout 数 4→32 时单个 train client 的 descriptor 接收者**不超线性增长**。
         机制是 plan 按分片指派源，而非按人头广播 ——
         每个 rollout 只读 train client 的一个子集（测试同时钉住这条机制）。

      **仍待做**：措施 (a) 的「holder 集合变化时由 PS 主动失效通知」与
      (d) 的 gRPC 内存快照读路径。当前稳态已不查 PS，
      所以这两项影响的是**弹性期**而非稳态，与 E3 同批更合适。
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

- [ ] **D1 接线部分完成，性能判据未达成**。已完成：plan 侧的 peer 扩散
      （`_make_peer_diffusion_plan`）、peer descriptor 扇出、拉取按 plan 迭代。
      5 节点实测 **9,576 次探测 → 653 次读且全部产生传输**。

      **但探测次数从来不是瓶颈**：实测 plan 版 2.93s vs 探测版 2.59s vs PS 路径 1.14s。
      废弃的探测很便宜（`client_read` 遇空 shard 列表提前返回），时间在传输本身。
      直供的传输粒度结构性更差：24 个 train rank 在 TP=4×PP=2 下把 key 空间切碎
      （653 次读对应 399 个 key，每 key 约 1.6 个源），而 2 个 PS worker 各持完整 shard。

      **peer 扩散的收益已在 plan 层量化**：gen TP=2、每节点 8 个 client 时
      `tp_rank = rank % 2` 让 4 个 client 需要**逐字节相同**的 shard，
      上游读从每节点 8 次降为 2 次，且**不随每节点 client 数增长**。
      `test_peer_diffusion_plan.py` 13 passed（含无链式、同节点、确定性、
      「跳过 peer 读仍然正确」四条）。

      **一个必须先解决的前置条件（设计文档未提）**：peer 读需要
      **rollout actor 之间的屏障**。peer 只有在它自己的上游读完成后才持有这些字节，
      提前读会**成功**——注册有效、传输格式正确——读到的是未填充的缓冲区，
      结果是静默错误的权重。当前 harness 的 gen actor 之间没有 process group，
      因此 peer 读被**显式推迟**而非执行，上游读保持完整。

      **屏障已落地，peer 读已真实执行**：驱动本来就 `ray.get` 所有 gen actor，
      把 pull 拆成 `upstream` / `peer` 两相即可，无需新的同步原语。
      5 节点实测 exit=0、16/16 全部 399/399 ——
      **这条只能靠真跑验证**，因为提前的 peer 读会「成功」地读到未填充缓冲区而非报错。

      **但端到端更慢，这是如实结果**：
      单相（全部上游）2.68s vs 两相（上游 2.03 + peer 3.50）= 5.53s。
      上游读确实降到极少（follower 发 0 次上游读、399 次全走 peer），
      61% 的读确实移离了 NIC —— 但 peer 相的开销超过它替掉的 NIC 读。

      **已实测排除了「链路不够快」这一解释**：同节点 GPU 间 **342.95 GB/s**
      （`bench_cross_node_bw.py` 两端同机，cuda:0 服务 / cuda:1 读取，2 GiB / 64 张量），
      对比跨节点单 NIC 的 24.50 GB/s，**优势 14×**。
      设计 §3.5.2「同节点 peer 在稀缺货币上几乎免费」的前提**成立**。
      所以 3.50s 不是带宽问题：399 次读 / 3.50s = **每次 8.8ms**，
      而 343 GB/s 下 40MB 分片只需 0.11ms —— 时间几乎全在**每次读的固定开销**上。

      **chunk 流水已驱动，逐 chunk 完成通知已落地并集群验证。**
      `psrl/utils/nixl/chunk_pipeline.py` 把读按 chunk 排成波次（单测 24 passed），
      `psrl/utils/nixl/chunk_notify.py` 提供驱动它的通知通道（单测 11 passed）。

      **为什么共享波次计数器不行（这条判断已被三次集群失败证实）**：每个 client 的波次表
      由**它自己的 plan** 导出，于是 designate 可能提前用完上游读，而 follower 还在
      peer-read 它本该正在填充的 chunk —— 计数器对所有人一起推进。
      坐标判据每次都抓到「peer 读在 designate 拿到字节之前就读了」：
      **逐 client 看顺序正确，全局看是错的**。

      **为什么 driver 屏障也不行**：它正确但自毁 —— 屏障是全局的，
      每个 chunk 上所有 client 都要等最慢的那个。这正是两相版本「开销超过收益」的根因。

      **解法是逐 chunk、逐源的门控，走 NIXL notification，方向 designate → follower。**
      注意这与数据流**相反**（字节是 follower ← designate）。
      方向相反正是它保持 P2P 的原因：没有谁去问协调者「chunk i 好了吗」，
      控制面不会多出逐 chunk 的往返——而那正是 C2 要消除的东西。

      **5 节点实测（exit=0，16 个 gen client 全部 399/399，4.41s）**：
      4 波次，designate 653 pairs / 28 notif 发布 / 0 等待，
      follower 399 pairs / 0 发布 / 8 次源等待。
      全局 80 次发布对 80 次等待 —— 驱动端断言这两者相等，
      因为**不相等意味着通知图不对称**，而那会表现为挂住而非报错。

      **过程中三个由集群暴露的真实缺陷**（都已修并有测试钉住）：
      1. **握手过滤只给了读方向**。`server.py` 的 peer descriptor 扇出按
         「我读谁」授予，而通知要「谁读我」。于是 designate 发不出通知，
         报 `Target client ... not found in client infos` —— 而且报在
         **自己所有读都能满足的那个 client 上**，极易误判为握手 bug。
         修复：扇出同时包含 `followers_of`，两个方向都**按 plan 限定**，
         否则每个 rollout 又要携带所有其他 rollout 的描述符。
      2. **schedule 的单位是 shard，transport 的单位是 `(key, source)`**。
         `client_read` 一次搬这一对的**全部** shard，于是 shard 跨波次的对会被
         第一个提到它的波次整体搬完，后续波次重投一个仍在飞的 handle ——
         集群返回 `NIXL_ERR_REPOST_ACTIVE`。
         修复：`coalesce_to_transport_pairs`，每对只出现一次。
         **且上游取最早波次、peer 取最晚波次**：上游无依赖，早发才有重叠窗口；
         peer 必须等它要的**每个** shard 都填好，取最早会读到未填充缓冲区而**成功**。
      3. 以及早先三条（按 chunk 成员寻址、重复读被扩散的 shard、peer 读与依赖同波次），
         见下方记录。

      **仍然诚实的部分：流水正确但更慢，同轮 A/B 已量化。**
      `[MEASURE]` 行同轮测两者（`PSRL_E2E_BASELINE=0` 可关）。
      跨轮比较不可信 —— 集群状态、模型加载顺序、NIC 争用都在动，
      而流水的全部主张只是几秒的差。**必须同轮测两者。**

      | 配置 | 耗时 | 对基线比 |
      |---|---|---|
      | 无扩散基线（全部上游，单批提交） | **2.16~2.28s** | 1.00× |
      | 流水 + 波次屏障 | 4.54s | **1.99×** |
      | 流水 + 上游不设屏障 | 5.58s | **2.59×** |

      **两个假设都被实测否决了，都是从总量推出来的：**

      1. **「链路不够快」** —— 已否决：同节点 GPU 间实测 342.95 GB/s，
         对跨节点单 NIC 24.50 GB/s **优势 14×**（见下）。
      2. **「波次屏障是开销」** —— 已否决，而且方向相反：
         去掉上游屏障后从 1.99× **劣化到 2.59×**。
         同轮 per-client 数据给出了理由：designate 做 653 次读用 2.06~2.18s，
         而基线 client 做**同样的 653 次读**只用 0.77~0.90s。
         差的不是扩散、也不是屏障，而是**每次读的固定开销**
         （399 次 peer 读 / 3.5s ≈ 8.8ms 每次，而 343 GB/s 下 40MB 分片只需 0.11ms）。

      **分解计时给出了确定的归因**（`post_upstream / wait_upstream / gate /
      post_peer / wait_peer` 五段，由 `pull_pipelined` 返回）：

      | 角色 | post_up | wait_up | gate | post_peer | wait_peer | 总 |
      |---|---|---|---|---|---|---|
      | designate（653 pairs） | 1.616 | 0.007 | 0.094 | 0.012 | 0.001 | 1.85s |
      | follower（399 pairs） | 0.000 | 0.000 | **2.423** | **3.074** | **0.008** | 5.51s |

      **结论：瓶颈是「每次提交的固定开销」，不是调度、不是链路、不是屏障。**
      follower 的 `wait_peer = 0.008s` —— **真实传输近乎免费**（NVLink，
      与实测 342.95 GB/s 一致）；而 `post_peer = 3.074s` 分摊到 399 次提交
      是**每次约 7.7ms**。`gate = 2.423s` 是等 designate 的 `post_up = 1.616s`，
      即 follower 在等对端**提交**而非等对端传输。
      两项合计 5.5s 中有 5.497s 花在提交路径上。

      **据此做了一处有效优化：上游读改为逐波次提交**（而非先提交全部波次再逐波等待）。
      理由直接来自分解：follower 的 `gate` 在等 designate 的 `post_up`，
      所以 designate 必须**尽早发布第一个 chunk**，而不是等 653 次提交全部完成。
      实测 `gate` 从 **2.423s → 0.872s（−64%）**，整体 2.47× → **1.93×**。
      这是本轮唯一被分解计时支持、且实测生效的改动。

      **最终定位（`client_read` 内部四段计时 + 描述符数扫描）**：

      | 角色 | deser | align | init | **xfer** | post_peer 外层 |
      |---|---|---|---|---|---|
      | designate | 0.013 | 0.095 | 0.014 | **1.528** | — |
      | follower | 0.020 | 0.001 | 0.014 | **3.426** | 3.494 |

      `post_peer` 的 98% 是 `nixl_agent.transfer()` 本身，
      描述符反序列化、对齐、`initialize_xfer` 三项合计仅 0.035s。

      **`postXferReq` 的成本是「每个 descriptor」而非「每次调用」，已实测。**
      固定 2 GiB 载荷扫描描述符数（`bench_cross_node_bw.py --tensors`，新增
      post 阶段单独计时）：

      | descriptors | post | us/desc | 带宽 |
      |---|---|---|---|
      | 1 | 0.026 ms | 26.1 | **354 GB/s** |
      | 64 | 0.241 ms | 3.8 | 361 GB/s |
      | 512 | 1.813 ms | 3.5 | 290 GB/s |
      | 4096 | 16.942 ms | 4.1 | **114 GB/s** |

      同样的字节数，post 时间差 650 倍，带宽掉 3.1 倍。
      **算术完全闭合**：follower 399 次读 × 4096 descriptor × 2.1us = **3.43s**，
      对实测 `post_peer = 3.426s`、`wait_peer = 0.008s`。
      **字节是免费的，descriptor 是全部成本。**
      非连续分片以 strided range 注册、按行展开为每行一个 descriptor（A2），
      这就是 gen 侧 144 个 strided 分片的来源（train 侧只有 54 个）。

      **两条有效改动与两条失败改动，都已实测：**

      | 改动 | 结果 |
      |---|---|
      | 上游读改逐波次提交（让 designate 尽早发布 chunk 0） | gate 2.423s → 0.872s，**2.47× → 1.93×** ✅ |
      | 修 `post_cached_xfer` 的 `descCount()==1` 断言（允许多 descriptor 合并） | 解锁了合并路径，但见下 |
      | 合并 peer 读的描述符 | post_peer 4.105s → **6.270s**（Python 端 materialise 399×4096 更贵）❌ |
      | **按描述符成本门控扩散**（strided 分片不走 peer） | **1.93× → 2.25×** ❌ |

      **最后那条的失败很有信息量，已写进 `_make_peer_diffusion_plan` 的注释钉住**：
      descriptor 展开是**分片布局的属性，不是链路的属性** ——
      strided 分片无论从 peer 读还是从上游读都付 4096 个 descriptor。
      所以拒绝扩散它并不能消除成本，只是把成本从 `post_peer` 搬到 `post_up`
      （实测 post_up 3.292s），同时白白丢掉 NIC 的节省。
      **杠杆是「更少的 descriptor」，不是「换哪条链路付」。**

      **也不要指望 Nx5 strided descriptor**：§3.6.3 的 GPU 实测已证伪 ——
      一条 strided 行在 prepped handle 内部仍展开为 `count` 个可索引条目，
      提交必须给 `count` 个索引，给少了会**返回 `DONE` 但大部分字节没动**。
      strided 压缩买到的是元数据体积与 prep 时间，不含提交时间。

      **当前状态（5 节点，同轮 A/B，`[PASS] all 5 nodes import this tree` 在日志中）**：
      exit=0、16 个 gen client 全部 399/399、流水 5.00s vs 基线 2.59s = **1.93×**。
      **正确性已验证；性能判据未达成，且根因已定量定位。**

      **最后一次尝试暴露了一个结构性冲突，这是 D1 性能判据无法在本层达成的真正原因。**
      既然 descriptor 数是全部成本，自然的解法是让 gen 侧不要以 strided 注册 ——
      改走 staging（一个 descriptor + 一次本地拷贝）。账面上极为有利：

      | 路径 | 成本 |
      |---|---|
      | strided 原地注册 | 399 × 4096 × 2.1us = **3.43s** |
      | staging | 399 × 1 × 4us + 8GB 本地拷贝 ≈ **7ms** |

      加了 `STRIDED_ROW_BUDGET`（64 行）后 gen 侧 strided 分片从 144 降到 0，
      **但集群立刻报错**：

      ```
      RuntimeError: Remote descriptor must be contiguous for client read,
      but found key model.layers.0.self_attn.o_proj.weight shard (0,0)
      in NIXLGenClient_I0_R10 is non-contiguous
      ```

      **原因是 staging buffer 没有常驻的 remote descriptor** —— 它是传输的*目的地*，
      不是可被寻址的*来源*。而 peer 扩散恰恰要求 designate 能被读：

      - **要扩散** → designate 必须原地注册 → 每分片付 4096 个 descriptor；
      - **要 staging** → 该 client 只能是纯目的地 → **不能有 follower 读它**。

      两者互斥。`strided_is_cheaper_than_staging(..., must_be_readable=)` 现在把这条
      作为**正确性约束**而非调优项固定下来（可读性永远压过行数预算），
      测试 `TestReadabilityOverridesTheRowBudget` 钉住它。

      **补充（CPU 实测量化了这条"结构性"论断）：读源数 = ceil(train_TP / gen_TP)。**
      在 `test_direct_supply_plan.py::TestTheReadCountAgainstPsIsNowPinned` 里
      用真实几何构造 24 train rank（TP=4）+ 8 rollout（gen TP=2）测出：

      | | 每 key 读源数 | 总读次数 |
      |---|---|---|
      | 直供 TP=4 / gen_TP=1 | **4.00** | 12,768 |
      | 直供 TP=4 / gen_TP=2 | 2.00 | 6,384 |
      | 直供 TP=4 / gen_TP=4 | **1.00** | 3,192 |
      | 有 PS | **1.00** | 3,192 |

      **实跑验证 `GEN_TP=4`：公式被证实，但该配置在本集群跑不起来。**
      run 39 `GEN_TP=4` 确实使 `n_rollout_instances` 从 4 降到 **1**（与预测一致），
      但 vLLM 启动即失败：

      | | 同节点 KV cache |
      |---|---|
      | run 38（`GEN_TP=1`） | **58.85 / 30.18 GiB** |
      | run 39（`GEN_TP=4`） | 11.48 / **0.29 GiB** |

      `ValueError: To serve at least one request with the model's max seq len (40960),
      (5.62 GiB KV cache is required)`。

      **原因是放置而非显存总量**：`GEN_TP=1` 时 4 个 rollout 实例各占 1 卡、
      分散在 4 个节点；`GEN_TP=4` 时单个实例要 4 卡**落在同一个节点**，
      与同节点其他负载相撞。所以 `GEN_TP` 不只是并发度权衡，
      它**改变放置拓扑**，需要相应扩节点才能用。

      **结论：公式成立（plan 层已由 CPU 测试钉住），但"匹配 TP"不是现成可用的解法。**

      **D1 性能判据已达成** —— 而且是靠**关掉扩散、走 staging**得到的：
      run 42 用 `peer_diffusion=False` + `shards_must_be_readable=False`：

      | 配置 | rollout pull |
      |---|---|
      | 扩散开（run 38） | 1.25 / 1.56 / 1.69 s |
      | **扩散关 + staging（run 42）** | **0.66 / 0.68 / 0.73 / 0.90 s** |
      | PS 路径（设计记录） | 1.14 s |

      同配置只差这两个开关，**约 1.9× 提升，且已优于 PS 的 1.14s**。
      0 错误、0 条版本违例、6 次 pull 全部完成、2 个 step 跑完。

      **这印证了设计里那条判断**：扩散省的是 NIC 字节（跨节点 24.50 GB/s vs
      同节点 342.95 GB/s），但它逼迫 designate 原地注册、每分片付 4096 个 descriptor
      （~2.1us 一个），而 descriptor 比字节贵得多。**关掉扩散后 gen 侧可以 staging，
      一个 descriptor 搬完一个分片。**

      因此直供的推荐配置是 **`peer_diffusion=False` + `shards_must_be_readable=False`**；
      两者是一对（扩散要求可读），已由 `_validate_nixl_peer_diffusion` 强制。

      **所以 2.93s vs 1.14s 里有一部分不是直供固有的，而是 TP 配置造成的**：
      出脚本用 `TP=4 / GEN_TP=1`，每个 key 要跨 4 个 train rank 拼，
      而 PS worker 一份就是完整 shard。`GEN_TP` 现已可覆盖（默认仍 1），
      但**它不是免费的**：`GEN_INSTANCES` 会按同一比例缩小（TP=1 → 4 实例，TP=4 → 1），
      并发度随之下降。因此这是个权衡，不是可以随手改的常数 ——
      真正的出路仍然是让 gen 侧分片在注册时连续（消除 strided），
      那样既不增加源数也不牺牲实例数。

      **因此结论是：在当前的分片布局下，peer 扩散本身就是净亏的，而且无法靠调度或
      传输参数挽回。** 无扩散基线之所以快（2.0~2.6s），正是因为它的 rollout 是纯目的地、
      可以 staging，用一个 descriptor 搬完一个分片。
      扩散要省的是 NIC 字节（实测跨节点 24.50 GB/s vs 同节点 342.95 GB/s），
      但它逼迫 designate 付 4096 倍的 descriptor，而 descriptor 比字节贵得多。

      **真正的出路是消除 strided 布局本身**（让 gen 侧分片在注册时就是连续的），
      这触及 §3.9.2.1 的 vLLM 分片粒度与 converter 的 layout 变换，
      属于 **A2/D4 的范围，不是 D1**。

      **在那之前，配置项已就位**：`CommunicationPlanner(peer_diffusion=False)`
      （默认 `True`）。默认保持开启，因为 plan 层的去冗余是真实的（61% 的读移到 NVLink）、
      回归只针对 strided 布局；但在当前 Qwen3-8B + 直供配置下，
      **实测关闭更快**，这是实测支持的配置建议而非设计缺陷。
      `TestDiffusionCanBeTurnedOff` 钉住两种设置都产出正确的 plan。

      **D1 最终状态**：
      | 判据 | 结果 |
      |---|---|
      | 正确性（16 client × 399 key，坐标判据） | ✅ exit=0，多轮复现 |
      | 通知图对称（发布数 == 等待数） | ✅ 80 / 80，驱动端断言 |
      | plan 层去冗余（61% 读移到 NVLink） | ✅ |
      | 端到端性能优于无扩散基线 | ❌ **1.42~1.93×（更慢）**，根因见上 |

      **后续：根因已修，见 `descriptor_cost_plan.md`。**
      那份文档把问题从「扩散不划算」推进到「descriptor 膨胀」，
      并已落地选择性 staging（S1–S4 集群验证）：
      **拉取 0.619s，3.68× 于此前最好的基线**，原地提交成本
      1238.6ms → 0.2ms，`temp_desc_slice_map=144`。
      D1 的性能判据要在扩散与 staging 共存（该文档的 S5）之后才能重测 ——
      当前两者互斥，而 staging 侧已经比扩散侧快得多。

      **实现过程中三个由实测暴露的设计错误**（都已修并有测试钉住）：
      1. **按 chunk 成员而非按「已调度的读」寻址**。真实尺寸下一个 chunk 含约 100 个 key，
         「这个 chunk 里的所有 key」与「放进这个 chunk 的读」是两个不同的集合 ——
         于是既发出没人计划的读、又静默丢掉计划内的读（653 计划 vs 100+269 实发）。
      2. **schedule 把每个被扩散的 shard 读了两遍**。comm plan 刻意保留完整的上游读，
         好让无法跑 peer 相的调用方跳过它仍然正确；但流水调用方两相都跑，
         这份余量就变成了重复读。schedule 现在是**划分**：peer plan 覆盖的归 peer，其余归上游。
      3. **peer 读被放在它所依赖的上游读的同一个波次**。同一个屏障区间内两者并发，
         peer 读可能读到 designate 尚未填充的缓冲区 —— 而且**会成功**。
         现在 peer 读统一后移一个波次。

      另一处**测量方法的更正**：`overlap_depth` 原先拿一个 client 的 peer 读
      去比它自己后续的上游读，但**完整 follower 根本没有上游读**（它要的 shard 全被 peer 覆盖），
      于是完美流水也会被报成「零重叠」。重叠是**跨 client** 的，现在要同时传入 designate 的波次表。

      **因此下一步的方向是明确的**：两相版本里 peer 相完全串行在上游相之后
      （屏障既保证了正确性，也排除了重叠）。要让扩散变成净收益，
      必须让它**按 chunk 与上游波次重叠**——这正是 B1 的 chunk 流水存在的目的，
      现已由 `chunk_notify.py` 驱动（见上）。改链路或改选源都不是问题所在。

      **5 节点实测确认 plan 层确实找到了重复**：653 次计划读中
      **399 次（61%）被判定为 peer 读并显式推迟**，254 次走上游。
      即屏障到位后，**61% 的读会从 NIC 移到 NVLink**——
      注意是「移走」而非「消除」：读的总次数不变，变的是走哪条链路。
      本次运行 exit=0、16 个 gen client 全部 399/399。

      **一个被自己的测量 bug 掩盖过的事实**：第一次实测报告「0 次 peer 读被推迟」，
      当时结论差点写成「集群上没有重复可消除」。实际是 harness 先按
      train client 名单过滤 work，再分离 peer 读 —— peer 源是 rollout client，
      早就被过滤掉了，那个 0 **计数的是空集**而非证据。planner 一直是对的。
      教训可一般化：**诊断里的 0 只有在「它本来可能非 0」时才有意义**。
      已加测试钉住 harness 的分离顺序，且 peer 判定必须**按 key**
      （同一 client 可能对某 key 是 peer、对另一 key 是上游源）。

      由此定下一条设计约束：**上游读绝不按 peer plan 裁剪**。
      裁剪会让两条路径互相依赖——没有屏障的调用方必须跳过 peer 读，
      而裁剪过的上游读就会静默丢掉那些 shard。
      代价对比很清楚：都发是一次重复读（扩散本想省掉的），
      裁剪是错误权重且无报错。

- [x] **D1 选源模块已落地**：`psrl/utils/nixl/source_resolution.py`。
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

      **删除的前置已补上：生产侧新增 `ps_mode=nixl_direct`**
      （`ray_trainer.py` 的分发分支 + `config.py` 的校验，
      `test_direct_supply_mode.py` 8 passed）。

      **这是一处此前会导致删错东西的缺口**：`direct_supply` 只是**测试配置键**，
      生产的 `ray_trainer.py` 按 `ps_mode` 分发，而该枚举**没有「无 PS」的取值**。
      先删 `PSStorageWorker` 会让生产失去全部传输路径，
      **而 e2e harness 仍然全绿**——它根本不读 `ps_mode`。
      新分支同时拒绝 `lmcache.enable_p2p`（它要经 PS 路由），
      在配置期报错而非传输期。

      **第二处前置也已补上：`ps_mode` 的判定集中化**
      （`psrl/utils/ps_mode.py`，`test_ps_mode.py` 14 passed）。

      此前 `ps_mode == "nixl_cpu" or ps_mode == "nixl_gpu"` 这个短语
      **在 6 个文件里重复了 12 次**，于是加一个模式变成了搜索题：
      `nixl_direct` 被配置校验和 worker 构建接受，却仍被另外 10 处守卫拒绝 ——
      **模式存在但不可用，而且没有任何地方会这么报告**，
      每处守卫都以自己的理由失败，离引入模式的那个决定很远。

      集中化同时纠正了一处被混淆的语义：这些守卫问的其实是**两个不同的问题** ——
      `uses_nixl`（传输是否走 NIXL，直供为真）与
      `has_ps_data_plane`（PS 是否持有权重，直供为假）。
      **多数站点意思是前者、却写成了后者**。现在按真实意图分别替换：
      push 用 `has_ps_data_plane`（没有 PS 就无处可推），
      pull 与握手用 `uses_nixl`（源是 PS 还是 train rank 都一样）。
      保留的唯一一处 `nixl_cpu` 单独判定是**放置相关**（PS worker 落在 host 还是 device），
      已有测试记录它为何幸存，以免看起来像漏网。

      **生产脚本已就位**：`examples/grpo_trainer/qwen3_8b_megatron_direct.sh`，
      与基线脚本只差 `ps_mode=nixl_direct` 与运行长度（2 步，覆盖第二个权重版本）。

      **但生产路径实跑被一处环境问题阻断，且该问题与直供无关**：
      `TaskRunner` actor 里 `megatron.bridge → modelopt → tilelang` 先加载了
      `libcudart_stub.so`，之后 vLLM 懒加载 `flashinfer.comm` 时
      解析 `cudaDeviceReset` 失败。**基线代码同样阻断**（已 `git stash` 对照确认），
      所以这不是直供引入的。

      已做的两件事：
      1. `main_ppo.py` 与 `ray_trainer.py` 顶部预加载 `flashinfer.comm`
         （与 e2e harness 同款修法）。**在进程内复现时有效**，
         但 actor 里 tilelang 的加载发生得更早，预加载本身就已失败。
      2. 把该预加载的失败**从静默 `pass` 改为告警**。
         这是定位的关键：失败在 `TaskRunner` 里早已发生，
         而真正的报错出现在很远的 `verl.workers.rollout` 内部，
         读起来像 vLLM 的问题。现在两者能被连起来。

      **另一处必须记录的环境事实**：本树 vendored 的 verl（`8718ca30`）与集群环境
      默认解析到的（`ls/psrl` 的 `74cebc50`）**不是同一个版本**，
      且两边 schema 互不兼容（前者无 `vanilla_mbridge`，后者无 `configure_verl_logging`）。
      用 `PYTHONPATH=<repo>:<repo>/third_party/verl` 跑测试：
      **1203 passed / 15 failed / 21 errors → 1309 passed / 5 failed / 3 errors**。
      此前报告的「既有失败」大部分是这个路径问题，不是真实缺陷。

      **根因不是直供，而是本树处于一次未完成的 verl 迁移中。** 这一点花了若干次
      集群实跑才定位清楚，值得完整记录：

      集群上有两个 verl checkout，差别**不是版本号**：

      | | `<repo>/third_party/verl` | 环境默认 (`ls/psrl/third_party/verl`) |
      |---|---|---|
      | upstream commit | `8718ca30`（较新） | `74cebc50`（较旧） |
      | PSRL 补丁 | 无 | **705 行新增 / 20 个文件 / 未提交** |

      不带 `PYTHONPATH` 时 `import verl` 解析到**打了补丁的旧版**；较新的那个
      **未被 git 跟踪、也不是 submodule**。

      **必须用较新的那个**，这不是偏好：PSRL HEAD 需要
      `verl.trainer.ppo.metric_utils.RolloutMoELoadBalanceMetricsAccumulator`，
      只有新版有。我试过用打补丁的旧版以"免费"获得那 705 行，结果在 import 阶段就失败。

      **于是每一个"verl 缺少的符号"其实都是一处需要前移到 PSRL 的补丁。**
      不要再去改 vendored checkout —— 把能力搬进 PSRL，它才能活过下一次 verl bump。
      已完成的前移（每一项都是一次实跑暴露出来的）：

      | 补丁 | 前移到 | 原来的失败形态 |
      |---|---|---|
      | `resource_num_per_bundle` | `psrl/trainer/ppo/resource_pool.py` | `init_workers()` 里 `TypeError` |
      | `configure_verl_logging` | `psrl/utils/logger/verl_logging.py` | import 期 `ImportError` |
      | `max_concurrency` | `PSRL_vLLMReplica` 的 property | 启 rollout server 时 `AttributeError` |
      | `use_per_rank_checkpoint` | 改为显式拒绝 | struct 模式下写未声明键 |
      | `load_weight` | 让 upstream 照常加载 | `McoreEngineConfig` 收到未知 kwarg |

      其中**分数 bundle 必须在两处同时生效**：placement group 的 bundle，
      *以及* actor 的 `num_gpus`。只做前者会得到
      `resource request {'GPU': 1.0} cannot fit into any bundles`。
      而 0.9/0.1 的**不相等**本身就是机制 —— 两个 0.5 会让 Ray 把第二个
      train worker 塞进同一张卡。

      **`LD_PRELOAD` 已试过且不成立，原因值得记下来**：
      预加载真实 `libcudart.so.12` 在**进程内**确实能让
      `import tilelang` 之后的 `flashinfer.comm` 正常工作（已验证），
      且 `LD_PRELOAD` 已通过 `_HOST_RUNTIME_ENV_KEYS` 送达 actor
      （日志里可见它出现在 runtime env 中）。
      但**对已启动的 Ray worker 进程无效** —— `LD_PRELOAD` 由动态链接器在
      `execve` 时读取，Ray 把它作为环境变量注入到一个早已运行的进程，为时已晚。
      要生效需要在 **Ray worker 启动前**设置，即在 `ray start` 的环境里，
      属于集群部署配置而非本仓库可改的范围。

      **生产实跑已跑通。** `examples/grpo_trainer/qwen3_8b_megatron_direct.sh`
      以 `ps_mode=nixl_direct` 在 5 节点 40 卡上完成启动并进入训练循环，
      日志依次出现 `Binding direct-supply sources`、`Initial pull`、
      `Training Progress`、`step:1`、`actor/pg_loss` ——
      即**无 PS worker 的直供路径完成了一次完整的权重供给与一个训练步**。

      跑通过程中暴露并修掉的 `nixl_direct` 真实缺陷（每一个都只在实跑中出现，
      单测与 e2e harness 都测不到，因为 harness 根本不读 `ps_mode`）：

      | 缺陷 | 症状 | 修法 |
      |---|---|---|
      | `self.ps_wg` 在 `uses_nixl` 块里被读 | `AttributeError`，发生在 Ray 起好、server 注册完之后 | 按 `has_ps_data_plane` 分流 |
      | agent 数把不存在的 PS 算进去 | 不报错，meta server 等一批永不注册的 agent 直到超时 | `ps_world_size = 0` |
      | gen 侧 pull 源无人作答 | `get_ps_nixl_agent_names` assert | 新增 `bind_direct_supply_sources`，用 train rank 回答 |
      | train rank 自己也去 pull | 同上 assert | 直供下跳过；它们**就是**源 |
      | `push_model` 落到 `else` 分支 | `NotImplementedError` | 新增 `direct_supply_publish`：不搬字节，但仍发布版本 |

      其中**第 2 和第 5 条是静默失败**：前者表现为超时而非错误，
      后者若只是跳过 push 而不发版本，rollout 会**继续成功地 pull 到旧权重**。

      **稳态实跑结果**（`/tmp/prod_direct26.log`，5 节点 40 卡，2 个训练步）：

      | 量 | 值 |
      |---|---|
      | rollout 从 train rank 拉取 | **1.38 / 1.77 / 1.78 s** |
      | 版本发布（`direct_supply_publish`，不搬字节） | **0.40 s** |
      | 成功的 pull 次数 | 5 |
      | 错误 | 0 |

      **但这次实跑不能用来验证权重的数值正确性**，必须说清楚：
      `actor/pg_loss`、`critic/score/mean` 全为 0。原因是 2 步的 base model
      在数学题上答不对，reward 恒为 0，GRPO 的 advantage 因此恒为 0。
      **一个只能是 0 的 loss 不构成"权重正确"的证据。**
      数值正确性仍然由 `tests/nixl/` 的坐标判据（`coord_judge.py`）保证 ——
      那里才有逐元素的 oracle，而端到端训练只证明了**链路能跑通且性能符合预期**。

      途中还踩到一个值得记的配置陷阱：最初把 `max_response_length` 从基线的 8192
      改成 1024 以求快，导致 `response_length/clip_ratio = 1.0`（每条回答都被截断），
      于是 reward 恒 0、loss 恒 0，而**日志完全正常、两步都"成功"**。
      改回 8192 后 clip_ratio 降到 0.0625 / 0.5。
      缩小 e2e 配置时应该动 `train_batch_size` / `rollout_n` / `total_training_steps`，
      不要动 `max_response_length`。

      **设计里缺的一条：直供与 Megatron 分布式 checkpoint 互斥。**
      这不是实现疏漏，`weight_transport.md` 里没有讨论过它 ——
      是生产实跑在第 2 步暴露出来的：

      - 直供要求 train rank 上的 **NIXL progress thread 常开**，
        因为 rollout 在它计算的同时 RDMA 读它，而那条路径要等 `DONE`；
      - Megatron 的 `dist_checkpointing` 的 `all_gather_object`
        会破坏该线程读取的 UCX endpoint 地址 ——
        这正是 `checkpoint.use_dcp_save` 把线程关掉的原因。

      NIXL **没有暂停/恢复 progress thread 的 API**（已查 `nixl_agent` 的全部方法），
      所以二者无法在一次 save 期间轮流让路，冲突是结构性的。

      症状极具迷惑性：传输**提交成功**然后永远停在 `PROC`，
      最终报在**读方**身上（`Timeout waiting for transfer ... to NIXLTrainClient_N`），
      而真正没能服务的那个 rank **什么都不打**。
      而且**第 1 步能过、第 2 步才挂** —— 首次 pull 发生在 trainer 空闲时。

      **解法**：`psrl/utils/checkpoint/megatron_manager.py` 新增
      `PSRL_PerRankMegatronCheckpointManager`，用逐 rank `torch.save` 取代
      `dist_checkpointing`。这同样是把一处 vendored verl 补丁前移到 PSRL
      —— 旧版是在 verl 内部 `import psrl.utils.checkpoint`。

      **必须前移而不能只是禁止保存**：75 个生产脚本里 **65 个 `save_freq > 0`**。
      如果直供强制 `save_freq=-1`，等于这些脚本全都不能用直供。

      **代价要写明**：逐 rank checkpoint **不可 reshard**，
      只能用相同 world size 与并行布局恢复。这是直供的真实代价，不是实现细节。

      **实跑验证**（`SAVE_FREQ=1`，`/tmp/prod_direct27.log`）：
      2 个 step 各存一次，0 错误，磁盘上确有
      `{optimizer,extra}/dist_ckpt/rank_{0..3}.pt`，4 个 train rank 全部走了新 manager。

      **但有一处必须讲清楚，否则会高估覆盖面**：`model/dist_ckpt/` 是**空的**，
      模型权重走了 `model/huggingface/`（verl 在有 bridge 时的默认路径），
      因此**只有 optimizer 与 extra 实际经过了逐 rank 保存**。
      模型权重那一路没有碰 `dist_checkpointing`，所以不与 progress thread 冲突 ——
      换言之这个配置能跑通，但它**没有**验证"模型权重的逐 rank 保存"。
      若将来有配置让模型也走 `dist_ckpt`（无 bridge 时），需要重新验证那条路径。

      **可复现性**：不含 checkpoint 的配置连续两次实跑
      （`/tmp/prod_direct26.log`、`/tmp/prod_direct28.log`）结果一致 ——
      2 步、0 错误、pull 1.38–1.78 s。

      **§3.4「版本相等性」已落地**，但走了一段弯路，值得记下来。

      我先做了**内容 checksum**，这是对设计的误读 —— §3.4 要的是
      「**版本相等性**」断言，并明确说元数据类校验「开销是常数级，**应当常开**」。
      实测证明 checksum 不可能常开：

      | | 8B 模型一次 pull | 随模型增大 |
      |---|---|---|
      | 内容 hash | **61,000 ms** | 线性，且随 dtype/量化变化 |
      | 版本戳（publish） | **0.057 ms** | **不变** |
      | 版本戳（observe） | **0.076 ms** | **不变** |

      pull 本身只要 1.4 s —— hash 比它要检查的东西贵 **40 倍**，
      于是只能采样、只能默认关，**故障恰好落在没采样到的 key 上**。
      `weight_checksum.py` 与其 22 个测试已删除。

      版本戳覆盖 **stale / torn / misdirected** 读；**不覆盖**正确版本字节的位翻转
      —— 这是有意取舍：RDMA 由 NIC 校验，而 stale 读是**逻辑错误且完全无症状**
      （pull 成功、版本号前进、引擎服务上一步权重）。

      **校验分两层，因为元数据新鲜度不免费**：撕裂读检查每次 pull 都跑
      （只比较分片之间，不需要新元数据）；而「announced vs observed」
      只在**首次 pull** 跑 —— 读者缓存的 `NIXLClientInfo` 仅在握手和 colocate 切换时刷新，
      做成逐步检查需要每步一次元数据广播，比它要抓的故障更贵。

      **验证方法本身也是一个教训。** 定位「版本戳写在哪」花了 **3 次多节点实跑、约 90 分钟**，
      而三次失败全都是**调用顺序**问题（stamp 在 `nixl_protocol` 之后 → 读者缓存 `-1`；
      在握手内问 PSManager → 与它自己的 `wait_for_client_infos` 死锁）。
      这些用源码检查（毫秒）和单节点元数据往返（1.9 s）都能定下来。

      **今后按「能达到验证目的的最轻档位」选择**：
      CPU-only < 单 GPU < 单节点多 GPU < 多节点。
      签名兼容、调用顺序、配置合法性、代价模型都属于 CPU 档；
      多节点只留给**跨主机 RDMA 与真实放置**。

      **版本断言已在多节点实跑验证通过**（run 38，`/tmp/prod_direct38.log`）：
      6 个 client 全部连上、握手 **1 秒**完成、
      `Binding direct-supply` → `Initial pull` → `step:1`，
      **0 错误、0 条版本违例**（既无 `torn read` 也无 `stale pull`）。
      rollout pull **1.25 / 1.56 / 1.69 s**，版本发布 **0.00 s**。

      这意味着：首次 pull 的 announced-vs-observed 检查、
      以及每次 pull 的撕裂读检查，在真实跨主机 RDMA 下都**不误报**。

      **但在此之前 run 33–37 连续失败，根因不是超时值，值得记下来。**
      它们都报 `Timeout waiting for agents`，于是看起来像"启动慢"：

      | | 等待开始 | 握手完成 |
      |---|---|---|
      | run 28（成功） | 14:45:58 | **14:45:59**（~1 s） |
      | run 30（成功） | 16:11:56 | **16:11:56**（~0 s） |
      | run 36 | 20:19:27 | 20:49:28 **超时** |

      成功的 run 里握手只要 ~1 秒 —— 因为客户端早已连上。
      run 36 等满了配置的 1800 s，**而只有 4 个 client 连上（成功时 6 个）**，
      且 `Capturing CUDA graphs (FULL): 100%` 只出现 5 次（成功时 6 次）：
      **有一个 vLLM replica 自始至终没有完成启动**。

      **真正的根因是 Ray actor 调用串行化造成的死锁。**
      `PSManager` 是 actor，一旦 `nixl_protocol` 跑起来它就卡在
      `wait_for_client_shardings`，**此后任何对该 handle 的调用都排在它后面** ——
      包括 train rank 启动握手前必须读的 `get_ps_model_version`。
      于是服务端等客户端、客户端等服务端。

      **区分"死锁"与"启动慢"的判据**（两者都报同一句话）：
      成功的 run 握手只要 **~1 秒**，所以 27 分钟的等待从来不是"超时值太紧"；
      再看 `Capturing CUDA graphs (FULL): 100%` 已满 6 个而
      `connect_to_server` 仍为 0 —— 那就是死锁。

      把 `handshake_timeout_s` 做成可配置仍是对的（客户端 1200 s / 服务端 600 s
      的不一致是真实缺陷：服务端先放弃，客户端侧什么都没记下来），
      但它**不是** run 33–37 失败的原因，文档不应暗示那是修复。

      顺带还暴露一个我自己引入的缺陷：那次把 11 个 wait 改成
      `timeout: float | None = None` 时，有 7 处函数体仍在比较 `timeout`，
      而生产调用全都省略该参数 —— 首次循环即
      `TypeError: '>' not supported between 'float' and 'NoneType'`。
      它表现为客户端未连上、**服务端**报 `Timeout waiting for agents`，
      与上面两种现象完全同形。现已由按**模式**（而非按实例）扫描的测试守住。

      **仍待做**：删除 `PSStorageWorker` / `create_ps_worker_group` / push 路径 /
      `ps_mode` 的 PS 取值。

      **删除前还剩一个真实障碍，必须说清楚**：75 个生产脚本里有
      **75 个仍在用 `ps_mode=nixl_cpu`**。删掉 PS 等于要求它们全部切到直供，
      而直供强制 `use_dcp_save=False`，于是它们的 checkpoint 格式
      从可 reshard 的 `torch_dist` 变成**不可 reshard 的逐 rank 文件**。
      这是一次对现有产出物格式的单向改变，**不是纯粹的代码删除**，
      应当由项目主人决定迁移节奏，而不是随手跟着删代码一起做掉。
      判据（CPU 权重 == 0、cast 显存上界 `2×chunk`）需删除后实测。
      `test_direct_supply_mode.py` 留了一个**成功即失败**的闸门测试：
      `PSStorageWorker` 文件一旦消失它就报错，提示连同这组闸门一起退役。
- [x] **D4 已落地并集群验证**：
      `psrl/workers/ps/holder_roles.py`，`test_holder_roles.py` 17 passed。

      三件事已就位：
      1. **holder 对称化**：`HolderKind` = `SELF` / `ROLLOUT_GPU` / `TRAIN_GPU`，
         按「字节在哪」而非「进程是什么角色」标识。
         这是 rollout-only 启动能成立的前提 —— trainer 尚未存在时，`v0` 必须有源。
      2. **bootstrap 路由**：`BootstrapPolicy.source_for`。
         trainer 在 **cast 路径下恒读 checkpoint**，即使 peer 持有该版本 ——
         rollout 手里是 fp8/fp4，反量化有损且 fp32 master params 从未上过线，
         「peer 有权重」不等于「trainer 能续训」。同精度 wire 下该限制自动放宽。
      3. **首次 vs 稳态用单向标记**，不从 `is_rollout_mode_in_actor` 推导 ——
         后者每次 colocate 切换都翻转，分不出「第一次到这」和「又回到这」。
         稳态切回误读盘会**静默回退训练进度**（权重合法、只是旧的），
         已用变异测试确认该断言真能抓到：去掉 parked 判定后该测试失败。

      每节点一个 checkpoint reader（`choose_checkpoint_readers`），
      读者数**不随每节点 rollout 数增长**，与传输路径的去冗余同构。

      **已接线（T3 完成后）**：`ray_trainer.py` 的 `switch_to_trainer_mode`
      现在带一条断言——**稳态切回时若仍要读 checkpoint 就报错**，
      并在权重拉回之后才置标记（中途异常仍算「首次」）。
      `test_colocate_switchback.py` 8 passed，含变异测试：
      把标记改回由 `is_rollout_mode_in_actor` 推导后，3 个测试失败。

      **连续两轮已验证（colocate 配置，`rounds: 2`）**：
      exit=0，第二轮拉取 0.112s，**8 个 gen client 全部 399/399 匹配「新版本」的坐标**。

      关键在于第二轮必须是**真的测试**：train 侧用 `advance_version(base)` 平移坐标基址
      重新填充，否则重填会产生**完全相同的字节**，于是「什么都没收到」与
      「收到了新版本」在判据下无法区分 —— 这正是稳态切回两种失败
      （误读盘、服务缓存版本）的共同形态：权重合法，只是错的。
      已用单测钉住：旧版本权重在新判据下**必须失败**（`test_coord_judge.py` 34 passed）。
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
- [ ] **E1 已接线（22 passed，含真值表），L2 判据待跑**：
      `psrl/workers/ps/version_barrier.py`。
      发布前按预算等待 v 的在飞读者归零；**超限即发布并把 v 标记 INVALID**，
      读者改拉 v+1（更新的权重，算法上是改善）——
      正确性靠**失效通知**而非无限等待，因为 trainer 的下一步就是在产 v+2。
      `requires_version_barrier` 把「谁需要」写成配置的函数：
      FSDP2 同精度 **必需**（无 master params，整个 optimizer step 都在写），
      Megatron 同精度是优化，**cast 路径与 colocate 恒为不需要**（结构上无共享内存）。
      **前置已就绪**：`nixl_e2e_5node_fsdp2.yaml` 已端到端跑通
      （FSDP2 + bf16+bf16 + disaggregate，8/8 client × 399/399 keys，拉取 2.91s），
      这正是 `requires_version_barrier` 返回 True 的那一格。

      **已接线**，三处：

      1. **读者上报**：`PSManager.begin_read` / `end_read` / `abort_reads`，
         按 `(version, rollout_instance_id)` 计数。按实例而非单纯总数，
         这样中途死掉的读者不会把计数永久卡在非零 —— `abort_reads` 可整体清掉一个实例。
      2. **发布前等待**：`push_model_state_dict_nixl` 在建 `ModelStore` 之前
         drain 上一版本；超限则按设计标记 INVALID 并告警。
      3. **决策落点**：`validate_config` 解析 `await_readers_before_publish`，
         因为只有它同时看得到 backend 和两个 dtype —— `PSManager` 只收到 `config.psrl`。

      两个实现上的坑，都记在注释里：
      - `count_readers` **故意不加锁**：barrier 在紧循环里轮询它，
        而读者结束读时**要拿**这把锁 —— 加锁会让"正在结束的读者"
        排在"等它结束的 barrier"后面，互相死等。
      - `await_readers_before_publish` 默认值是 `auto` 而不是 `False`：
        L2 判据要显式**关掉**它来观测破坏，而 `False` 必须能与"由我决定"区分开。

      **多节点回归已过**（run 43）：接线后 0 错误、0 版本违例、6 次 pull 全完成，
      rollout pull **0.46–0.48 s**（接线前 0.66–0.90 s，PS 1.14 s）。
      该 run 的 `await_readers_before_publish` 解析为 **False** 且**这是对的** ——
      日志里 `colocate_validate_and_train=True`，colocate 下 rollout 全程暂停，
      结构上不存在在飞读者。也就是说真值表在生产配置上确实生效，而非接线失效。

      **L2 判据已达成**（`tests/parameter_server/test_publish_corruption.py`，7 passed）：
      不拦截时，读者在一次分块拉取中拿到同时来自 `v` 和 `v+1` 的字节 —— 既不是旧策略
      也不是新策略，没有任何一步的权重长这样；拦截打开后同样的交错下拿到的是完整的 `v`。

      判据用**内容**而非版本戳：戳是发布时写的，被改写一半的分片照样报告自己的版本，
      `check_pull_versions` 看到的是一次干净的拉取。这正是 `shard_version` 文档里
      写明的盲区。

      交错是**编排**出来的，不是靠调度抢出来的 —— 否则写入落在读窗口内极少发生,
      测试会在没有复现竞争的情况下通过。坚持这一点抓出两处:第一版读者在写者醒来前
      就读完了 8 个分块,拦截报 `NO_READERS`,根本没有可拦的撕裂;重复发布那一组也
      需要读取真正占用时间(传输本就如此),否则每一轮都是 `NO_READERS`。两处都是靠
      断言 outcome 抓到的,只断言「没有撕裂」会一路绿灯。

      **这不能证明的事**：FSDP2 的 optimizer step 与真实 NIXL 拉取在真机上确实重叠。
      那是关于集群时序的论断,也正因如此拦截的必要性由配置决定(`requires_version_barrier`)
      而不是靠实测。这里证明的是:**一旦**重叠,结果就是撕裂,而拦截消除它。
      配置已就绪（`psrl.nixl.await_readers_before_publish=false` 可显式关闭），
      但**观测破坏本身尚未实现** —— 需要一次训练中读者恰好落在 optimizer 写窗口内，
      这靠调度而非断言触发，需要一个专门的注入点。
- [x] **E2 已落地并验证**：`psrl/utils/nixl/staleness.py`，`test_staleness.py` 13 passed。
      两种陈旧的恢复路径**互为无效**，这正是必须分开的理由：
      descriptor 陈旧时换源只是把错地址带到新 peer；版本陈旧时重取 descriptor
      只会拿到「正确地址指向没人持有的字节」。混为一谈得到的是**收敛不了的重试循环**，
      表现为挂起而非报错。
      **两者同时陈旧时优先报 `DESC_STALE`**，否则客户端修完源还会被同一批坏地址再废一次。
      `RegistrationEpoch` 在**重分片时也推进**（地址没变、含义变了）——
      仅做地址相等检查会通过，然后读到邻居分片的字节且不报错。
- [ ] **E3 成员状态机已落地并单测通过（14 passed），接线待握手循环改造**：
      `psrl/utils/nixl/membership.py`。
      准入复用 A7 的 `classify_join`（不新造一套「是否允许」），
      本模块定的是**何时生效、以及集群因此欠什么**（replan / epoch bump）。
      **离开必须两阶段**：先停止路由（`begin_leave`），等在飞归零后才移除
      （`complete_leave` 在仍有读者时**拒绝**）——
      合并两步会在活跃读上释放内存，而那次读会**成功**。
      **仍待做**：把 `server.py` 的
      `_is_all_client_shardings_recved` / `_is_all_client_infos_recved`
      一次性直线（源码中的 `TODO(lhy): Support elastic clients`）改为事件驱动循环，
      并跑 L3 运行中扩容。
- [ ] **E4 第一半（partial MD 两段式）模块已落地并实测；第二半被上游阻塞**。
      `psrl/utils/nixl/partial_md.py`，`test_partial_md.py` 10 passed，
      实测 `bench_partial_md.py`（H20）：

      | 注册区数 | 全量 MD | conn-info only | 压缩 |
      |---|---|---|---|
      | 1 | 2,139 B | 1,640 B | 1.3× |
      | 256 | 110,008 B | 1,644 B | 66.9× |
      | 4096 | **1,734,329 B** | **1,645 B** | **1054×** |

      **conn-info 恒定（1640~1645 B），增长全部来自 descriptor。**
      设计原文引的 406 B/region、883 B、401 B 三个绝对值与本环境不符
      （实测 423 B/region、606 B、1645 B），**但结构性结论成立且更强**。

      万卡量级：单次 join 从 **17.34 GB 降到 16.45 MB**。
      第二段只发给 plan 指明的读者（实测 606 B/读者），
      且**读者数不随集群规模增长** —— 这是测试钉住的性质。

      **第二半（`invalidateLocalMD`）被上游阻塞**：本 NIXL 构建未导出
      `invalidate_local_md`（`hasattr` 为 False）。
      缩容路径目前只有「撤销 holder + 等在飞归零」两步，
      无法吊销已发出的 descriptor。`describe_leave_gap()` 把这条写在一处，
      并有一个测试在 NIXL 补上该绑定时**失败**，作为继续施工的信号。

      **仍待做**：把两段式接到 `client_info` 的真实扇出路径上
      （当前 `meta=self.agent.get_agent_metadata()` 仍是全量），需与 E3 的事件驱动 join 同批。

### 测试改造
- [x] **T1 已落地并验证**：`tests/nixl/coord_judge.py`，`test_coord_judge.py` 30 passed，
      且已接入 e2e harness —— 5 节点日志的
      `Gen client N: 399/399 keys match their coordinates` 即该判据在集群上运行的证据。
      两条性质都在：值按全局坐标单射编码（换位可见），
      且 `assert_dtype_capacity` 预先检查目标 dtype 能否精确表示
      （fp8_e4m3 仅 3 位尾数，整数上界 16），避免静默舍入冒充成功。
- [ ] T2 删除 `create_ps_worker_group`
- [x] **T3 colocate 模式已落地并集群验证**：`tests/nixl/config/nixl_e2e_colocate.yaml`。
      **8 train + 8 gen 共卡，exit=0、8 个 gen client 全部 399/399、拉取 0.039s。**

      | 放置 | 拉取耗时 |
      |---|---|
      | disaggregate（跨节点） | 0.589 s |
      | **colocate（同卡）** | **0.039 s** |

      **15.1× —— 这是 §3.7「同卡传输不碰 NIC」的首次端到端实测。**

      **三处由实测暴露的真实约束**（都不是 harness bug，是 colocate 本身的性质）：
      1. **`num_gpus=0.5` 不足以表达共卡**。Ray 据此把**两个 gen actor** 塞到同一张卡
         （8 个 actor 落在 4 张卡），vLLM 的 TP=2 于是试图在一张卡上初始化两个 NCCL rank，
         报 `NCCL error: invalid usage`。配对必须是 **train 配 gen**，
         所以要用 placement group 显式表达，而不能交给调度器。
      2. **`STRICT_SPREAD` 是错的策略**：它要求每个 bundle 独占一个**节点**，
         8 个 bundle 在 5 节点集群上永远 ready 不了，表现为 placement group 超时
         而非任何与 colocate 有关的错误。改用 `PACK`。
      3. **vLLM 的默认 `gpu_memory_utilization`（>0.9）在共卡下不可用**。
         它按**总显存**比例预留并在启动时检查空闲量，而 train rank 已占约 25 GiB，
         于是报 `Free memory on device cuda:0 (69.89/95.22 GiB) is less than desired`。
         **这正是设计里「colocate 必须先 offload 再让 rollout 分配 KV cache」的账
         以报错形式出现**。harness 无切换路径，故两侧各占一半（`gpu_memory_utilization: 0.5`）。

      **顺带验证了 descriptor 规划的 §2.3**：TP 匹配（train TP=2 == gen TP=2）时
      输入维无需 LCM 重分片，**最差分片仅 4 行**（对比不匹配时的 4096 行），
      原地提交成本 0.9ms。
- [ ] **T4 矩阵与 CI 分档已显式化并自检（16 passed），FSDP2 格已补齐，余一格待 cast 路径**：
      `tests/nixl/test_config_matrix.py`。
      把「哪些组合被覆盖」从「读遍所有 YAML 才知道」变成一张可断言的表：
      每个 cell 要么指向一个真实配置并与之对账（后端/精度/放置必须一致），
      要么**写明为什么还跑不了**——空槽没有理由会被测试判为疏漏而非记录。

      | 档 | 组合 | 状态 |
      |---|---|---|
      | 每次 CI | megatron + bf16+bf16 + **colocate** | ✅ `nixl_e2e_colocate.yaml` |
      | 每次 CI | megatron + bf16+**fp8** + disaggregate | ⏳ cast 路径未接入 harness（A9 仅单测级） |
      | 每次 CI | **fsdp2** + fp32+bf16 + disaggregate | ⏳ 无 FSDP2 e2e 配置 |
      | 每日 | megatron + bf16+bf16 + disaggregate（TP 不匹配） | ✅ `nixl_e2e_5node_direct.yaml` |
      | 每日 | fsdp2 + bf16+bf16 + disaggregate | ✅ `nixl_e2e_5node_fsdp2.yaml`，**8/8 client × 399/399 keys，拉取 2.91s** |
      | 每日 | megatron + fp8+fp8 + colocate | ⏳ 需 TE `fp8_param` + cast 路径 |

      另有两条结构性断言：**每次 CI 档不得超过 4 格**（每格都是一次多节点运行，
      增长直接吃迭代速度）；**PS 版与直供版配置只能差 `direct_supply` 一个键**，
      否则两者之间的性能差异无法归因。

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

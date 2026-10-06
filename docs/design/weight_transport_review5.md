# 权重传输方案自审（第五轮）

> 对 [weight_transport](weight_transport) 的第五轮审核，覆盖九项：实现可行性、代价与风险、
> 通信栈与万卡扩展、场景覆盖（`is_rollout_mode_in_actor` / colocate / 精度 / 弹性）、
> 与四个对照系统的差异、完整性、WeightBridge 吸收度、测试规划、以及过时约束的清理。
>
> **结论先行：方案在前四轮的基础上已经收敛，本轮未发现新的正确性缺口。
> 但发现三件事：(1) 一处标注为「正确性空白」的 NCCL 问题经查证在 PSRL 下不存在，应当关闭；
> (2) 落地项清单的依赖关系有一处矛盾（9d 依赖 index，却标为 FSDP2 前置）；
> (3) 测试规划是本方案最薄弱的一环——26 条验证项里 10 条能由改造后的
> `test_nixl_e2e.py` 覆盖、5 条需多节点 e2e，**11 条必须真实集群实测**。
>
> **复审补充**：「FSDP2 + bf16 + bf16」被初稿错误拒绝，而它是 verl
> 8 个生产 example 在用的主流配置，本轮已改为支持（§4.3）；
> 「ref 与 actor 的 PCIe 竞争」经查调用链确认不存在，已撤回（§2.2b）。**

---

## 1. 实现可行性

### 1.1 落地项的依赖图有一处矛盾

现有清单 21 项（1–5、6/6b/6c/6d、7/7b/7c、8/8b/8c、9/9b/9c/9d/9e、10）。
按「依赖什么」重新分层后，发现一处顺序矛盾：

| 项 | 标注的依赖 | 实际依赖 | 问题 |
|---|---|---|---|
| 9d（FSDP2 对齐五项） | 「否，需 index」 | 其中**三项不需要 index**：offload 双 flag 校验、uneven sharding 断言、finest mesh 可观测 | **这三项可以立刻做**，与 1-5 同批 |
| 9d 的「副本数运行时推导」 | 同上 | 需要 index 与 8b | 这一项才是真依赖 |

**建议拆分 9d**：`9d-a`（三条配置期校验与可观测，独立可落地）与 `9d-b`（副本数推导，并入 8b）。
理由是前者是**纯防御性改动**，不做的话 FSDP2 场景会以难以归因的方式失败
（传输期报错而非校验期、注册到错误地址），而它们完全不依赖直供改造。

### 1.2 真实工作量重估

按「是否改动数据面」分两类，这是比风险分档更有用的切分：

```
不改数据面（可独立验证、可随时回滚）
  1-5、6c、9e、9d-a                      约 25%
改数据面但不删 PS worker
  6、6b、6d、7b                           约 25%
改数据面且删 PS worker（必须一起上）
  7、8、9、9b、9d-b、8b                   约 45%
可选 / 正交
  7c、8c、9c、10                          约 5%
```

**第三档的 45% 是不可分割的**——这是本方案最大的实施风险，
因为它意味着「要么全部正确，要么整条链路不可用」。
缓解办法只有一个：**在第三档开始前，第一、二档必须已经稳定运行**，
这样出问题时可以二分定位到第三档内部。

### 1.3 外部依赖的确认状态

| 依赖 | 状态 | 风险 |
|---|---|---|
| `prep_xfer_dlist` / `make_prepped_xfer` | API 存在，PSRL 从未调用 | **首次使用的未知坑**，应先写最小 micro-benchmark |
| Nx5 strided dlist | 同上 | 同上 |
| `invalidateRemoteMD` / `loadRemoteMD` | API 存在，未用 | ~~9c 的前置~~ **第六轮更正**：NIXL 侧能力齐备（还多一个未发现的 `getLocalPartialMD`，实测 1884x 压缩），**弹性的真正阻塞是 finest mesh 的全局 lcm**，见主文档 §3.3.1d |
| Megatron-Bridge `on_train_step_end` | **已查证可用**（`callbacks.py` 的 `VALID_EVENTS`，`train.py:518`） | 低 |
| vLLM `discard(("kv_cache",))` | **已查证可用**（`cumem.py:300`、`core.py:981`） | 低 |
| CUDA IPC 跨进程 | **已查证**（verl `update_weights_from_ipc`、SkyRL `skyrl_ipc`） | 低 |

**前两项是唯一的实质不确定性**。建议在落地项 1 之前插入一个
「NIXL 新 API 的 micro-benchmark」步骤，用 PSRL 的真实 tensor 规模验证
`prep_xfer_dlist` 的建立开销与 `make_prepped_xfer` 的提交开销，
避免在大改造中途才发现 API 行为与预期不符。

---

## 2. 代价与潜在风险

### 2.1 本轮复核：前四轮风险的收敛状态

| 风险 | 状态 |
|---|---|
| 读写冲突 | **已降级为可选优化**（8c）。两种部署模式今天都由 `pause_for_sync` 保证正确性（§3.7.2.4） |
| 内存注册有效性 | **已根治**：colocate 不注册（§3.7.3.0），disaggregate 注册 wire buffer 而非参数（§3.7.3.0a） |
| 元数据扇出 | 有方案（MoE 分组 + strided），**但 FSDP2 的 `shard_mesh[0]=world_size` 是第二个放大源**（§3.9.2.1） |
| 控制面瓶颈 | 有方案（6b），plan 本地缓存是关键 |
| trainer 故障 | DP 副本接管，**但纯 FSDP2 下不可用**，已记录 |
| `is_rollout_mode_in_actor` | 已修复（holder 对称化 + bootstrap 读盘） |
| 地址正确性 | **本轮确认已覆盖**：wire buffer 不参与 offload，天然稳定 |
| 多角色 | §3.10 已覆盖，ref 改为备份 tag 后进一步简化 |

### 2.2 本轮新识别的两处风险

**(a) `registration_epoch` 与 `version` 是两个独立的单调量，它们的交互未定义。**

§3.7.3.0a 引入 `registration_epoch` 解决陈旧 descriptor，§3.2 已有 `{role: version}`。
两者都是单调递增，但**语义正交**：

- `version` 变化：内容变了，地址没变
- `epoch` 变化：地址变了，内容可能没变（offload/reload 后内容相同）

**一个读者可能持有「正确的 version、陈旧的 epoch」**，此时它读到的是
正确版本的权重放在错误地址上——必须拒绝。反之「陈旧 version、正确 epoch」
是正常的 staleness，应当允许（由 §3.5.2 的版本相等性处理）。

**所以校验必须是两段独立判断，不能合并成一个「代次」。** 这一点设计里没写。

**(b) ~~ref 备份与 actor offload 的 PCIe 竞争~~ —— 核实后不成立，撤回。**

初稿担心「ref restore 与 actor offload 在 eval 切换时叠加，
两个 `M` 量级搬运让切换时间翻倍」。**查调用链后发现两者在时间上不相交**：

| 操作 | 调用位置 |
|---|---|
| `switch_to_rollout_mode` / `switch_to_trainer_mode` | **只在 `_validate()` 内**（`ray_trainer.py:1103`、`:1280`） |
| `_compute_ref_log_prob` | **在训练步策略内**（`full_batch.py:67`、`fine_grain_overlap.py:133`） |

**eval 与训练步是互斥的两个阶段**——`_validate()` 跑的时候训练步不跑，
所以 actor 的 offload（eval 开始）与 ref 的 restore（训练步内）
**永远不会并发**。这个风险不存在。

**但有一个相邻的真问题留下**：同一个训练步内，
ref logprob 之后紧接 actor 的 forward/backward，
**ref restore 与 actor restore 确实会连续发生**（一进一出，各 `M`）。
这不是并发竞争而是串行开销，且 §3.10.4e 的 `_ensure_on_gpu` 懒加载
正是为它设计的——actor 的 model 在 ref logprob 期间本就该留在 GPU 上
（ref 换入的是另一块内存，不是覆盖 actor）。
**前提是 colocation group 的显存能同时放下两份**，
放不下时才退化为真正的换入换出。这一点 §3.10.3 的「colocate 显存账按角色展开」
已经列为待办，无需新增。

---

## 3. 网络通信：优化充足度与万卡 / 1TB 扩展

### 3.1 前三轮通信栈问题的落实状态

| 第三轮发现 | 本轮状态 |
|---|---|
| QP 数量上限 | **已记录为 W 的约束 ④，且明确「不测 QP_budget，靠 W 的带宽反馈兜住」**——这是正确的处理 |
| GPUDirect 可用性 | 记录为「应用 `query_memory` 探测」，**仍未给出退化行为**（见 3.3） |
| 跨节点 peer 的 incast | 阶段二才开，**推迟而非解决**，排序正确 |
| rail 拓扑 | §3.5.3.1 已有方案（`PSRL_RAIL_ID` + `IB_SAME_RAIL` 档位） |
| NCCL 与 NIXL 争 NIC | **本轮简化**：阶段上报移出本期，只保留 `on_train_step_end` 回调发布（7b） |
| `num_threads` 多线程后端 | 记录为「应实测」，未纳入清单 |

### 3.2 万卡 / 1TB 的三条阻塞项

| 维度 | 状态 | 判据 |
|---|---|---|
| 元数据平面 | **有方案但有第二个放大源**：MoE 分组治 key 数，**FSDP2 的 `world_size` shard 数需靠 HSDP 治** | §3.9.2.1 的「HSDP 优先」建议必须写进配置校验，不能只是文档建议 |
| 轮数 × 控制面 | 已解决（plan 本地缓存 + chunk 自适应；去冗余后 1TB 下每节点仅 `M_v`=15.6GB，512MB chunk 对应 31 轮，不是 250 轮，§3.5.6 已重写） | 6b + 6c |
| trainer 出口 | 已解决（三阶段去冗余 + W + peer 扩散） | 7 |

**1TB 场景的端到端估算**（32 节点，fp8 wire，512MB chunk）：

```
trainer 出口   = M = 500GB（fp8），并发 across all train ranks
每节点入口     = M_v = 500GB / 32 ≈ 16GB     ← 轮数按这个算，不是按 M
轮数           = 16GB / 512MB = 31 轮
每轮 RDMA      = 512MB / 50GB·s⁻¹ ≈ 10ms
理论下界 T_min ≈ 16GB / 50GB·s⁻¹ ≈ 320ms（每节点入口受限）
```

**注意轮数的分母是 `M_v` 而非 `M`**——去冗余之后每个节点只收它自己那份，
所以「1TB 要 125 轮」是一个把 `M` 当成每节点入口的算错（§3.5.6 已重写该规则）。

与 miles 实测 Kimi-K2 的 **7.2s** 对比，本设计的理论下界低一个量级，
**差距主要在 rollout 侧的落地格式化（884ms 级）与控制面开销**。
这说明 §3.6 的「三类重叠」是 1TB 场景的**主导优化项**，
而不是锦上添花——它的优先级应当高于清单里的位置（现在是第 5 项，与 1-4 同批，合理）。

**第六轮补充**：这个估算隐含假定「三类重叠已经成立」，
而查执行层后发现它卡在一个同步调用上（§3.6.4）——
`merge_and_finish_cached_xfer` 提交后立刻进 1ms 轮询，下一个 chunk 的
post 被上一个的 wait 挡住，**双 buffer 在现有实现下无处落脚**。
这是落地项 5b，是第 5 项的前置。

### 3.3 GPUDirect 的退化行为（本轮已补入 §3.6.2a）

**「退化行为」的含义**：RDMA 读写 GPU 显存时，若 NIC 与 GPU 不在同一
PCIe root complex 下、或驱动不支持（缺 `nvidia-peermem` / `dmabuf`），
驱动会退化为 **host bounce**——先 DMA 到 CPU 中转缓冲再发出。
**同一行 `client_read` 代码，底层走两条带宽差一个量级的路径，上层无感。**

`network_topology.py:42` 的六档 `LinkType` 没有区分这两者。

**本轮确定的处理：启动期告警，不做选源硬约束。** 三条理由：
它通常是集群配置问题应当让运维修复；
`network_topology.py` 的带宽是硬编码常量，给 host bounce 估权重只是猜测，
与 §3.5.3「不做实时带宽探测」冲突；
若所有候选源都不支持，降级也无源可选，报警比静默慢跑有用。

**不新增 `LinkType` 档位，不改选源逻辑**，只在握手期 `query_memory`
探测失败时打印含拓扑信息的 WARNING，并在验证项 2 的带宽对比里记录。

---

## 4. 场景覆盖

### 4.1 `is_rollout_mode_in_actor=True`：已覆盖，且本轮发现首次/稳态的区分已落实

§3.3.1a 的三步链路完整，且**本轮新增的「稳态切回从 CPU reload 而非读盘」**
（§3.3.1a 末段）补上了一个会静默回退训练进度的缺口。
判据「trainer 本进程是否曾持有过真实权重」是正确的——
`is_rollout_mode_in_actor` 这个 flag 被两个 switch 函数来回翻转，确实区分不了。

### 4.2 colocate：本轮有两处实质修正，现已收敛

| 初稿 | 修正后 | 依据 |
|---|---|---|
| 「权重区不动」，两份常驻 | **transfer 后 offload 到 CPU，让位给 KV cache** | verl `engine_workers.py:811-826` 的顺序与 barrier 注释 |
| colocate 下仍做 NIXL 注册 | **完全不注册，走 CUDA IPC** | miles `protocol.py:79` 的第一优先级短路 |

两处修正的共同点是：**初稿把 disaggregate 的需求（trainer 持续供给）错误地套到了 colocate 上**。

### 4.3 精度配置：「FSDP2 + bf16 + bf16」被错误拒绝，本轮已改为支持

初稿把这个组合列为「配置校验拒绝」。**这是错的，而且错得不轻**——
核实 verl 的 example 后发现 `model_dtype=bf16` 出现在 **8 个生产脚本**里
（`run_qwen3_8b_fsdp.sh:82`、`run_qwen3_next_80b_a3b_fsdp.sh:84`、
`run_gpt_oss_20b_fsdp.sh:98` 等），**是主流配置而非边缘情形**——
大模型下 fp32 参数占两倍显存，改 bf16 是标准做法。

**真实约束比初稿写的窄**：FSDP2 确实无 master params，
所以 optimizer step 原地改写参数内存，**但 §3.4.2 约束 2 的版本级一次性等待
本来就覆盖这种情况**。差别只在必要性：

| 后端 + 精度 | 冲突窗口 | 版本级等待 |
|---|---|---|
| Megatron bf16+bf16 | param 回写那一瞬 | 优化 |
| **FSDP2 bf16+bf16** | **整个 optimizer step** | **必需** |
| FSDP2 fp32+任意 | 无 | 不适用 |

本轮已改为支持，并新增 §3.8.0a 说明其代价与两条退路
（改 `model_dtype=fp32` 或改 fp8 rollout）。
连带把落地项 8c 拆成两半：**版本级等待在该配置下是正确性必需**，
per-chunk 写屏障始终是优化。

### 4.4 弹性扩缩容：档位 A/B 的划分是本方案最诚实的一处

§3.10.4a 承认了「约束 (b) 与 role 缩容到 0 互斥」，并给出两档而不是辩解。
这是正确的处理。三点确认：

- **档位 A 的边界写清楚了**：每个 role 必须常驻至少一个副本
- **档位 B 的代价量化了**：`num_models × M` 的 CPU，换 1.5–2× 吞吐
- **两档共用 holder 抽象**：`STORE_CPU` 加回来是一行枚举，数据面不改

**一处遗漏（本轮已补入 §3.10.4a）**：档位 B 下 `STORE_CPU` holder 的一致性维护。
若 CPU holder 内容陈旧而选源按链路成本把它排在前面，就会读到旧版本。

**根因不是机制缺失，而是「永不失效」这个表述诱导实现者给 CPU holder 开后门**：
§3.5.2 的版本相等性本来就会让陈旧 holder 在版本推进时整体退出候选集，
**只要不豁免 CPU holder，缺口自动关闭**。
已修正表述为「物理介质不会消失，但版本有效性与 GPU holder 完全同等」，
并把 GPS 的角色限定为「bootstrap 与 shift-up 的源」而非稳态选源候选——
这把 Write 频率从「每步」降到「每次 role 归零前」。

---

## 5. 与 verl / miles / SkyRL / PSRL 现状的差异

### 5.1 四个维度的对照

| 维度 | verl | miles | SkyRL | PSRL 现状 | 本设计 |
|---|---|---|---|---|---|
| **拓扑** | 串行中继链（`build_topology:288`） | 并行 P2P + EP 去冗余 | pull + 预烘焙 plan | train → N×CPU PS → rollout | **pull + 三阶段去冗余** |
| **中转介质** | GPU bucket 4GB | **CPU 全模型 `M`** | GPU ring | **CPU `N×M`** | GPU chunk 1~4GB |
| **重分片** | 自己实现 | **借 sglang CPU replica 自动完成** | shard 代数 | **shard 代数（已成熟）** | 复用现有 shard 代数 |
| **colocate** | `mode="naive"`，不建 NIXL | `cuda_ipc` 短路 | `skyrl_ipc` | 全量 host 往返 + 重握手 | **CUDA IPC + stage offload** |
| **多角色切换** | `param_offload` 自动上下文 | `TensorBackuper` tag | **`_ensure_on_gpu` 二维懒加载** | 无（critic/ref 各自读盘） | 照搬 SkyRL |
| **ref 形态** | 独立进程 | **actor 的 CPU 备份 tag** | 独立进程 | 独立进程 | 照搬 miles |

### 5.2 本设计的三处结构性优势

1. **唯一同时具备「并行去冗余拓扑」与「零 PCIe 中转」的方案**。
   miles 有前者无后者（CPU 中转），verl 有后者无前者（串行链）。
   这是 §3.7.3.0a 末段那个「各取一半」的来源。
2. **唯一覆盖 colocate + disaggregate + 两种后端 + 八种精度组合的方案**。
   SkyRL 的 `sharded_rdt` 与 FP8 wire 互斥且不支持 colocate；
   miles 无 critic；verl 的链式在大 MoE 下不可用。
3. **PS 从数据面彻底移出，CPU 占用 `N×M → 0`**。
   这是相对 PSRL 现状的主要收益，也是约束 (b) 的直接产物。

### 5.3 三处有意识的取舍

| 取舍 | 放弃了什么 | 换到了什么 |
|---|---|---|
| GPU wire buffer 而非 CPU 全模型 | 地址天然稳定、自动重分片、零 GPU 占用 | 零 PCIe、最低延迟。**成立前提是 PSRL 已有 shard 代数** |
| 不做自动模型接入（约束 a） | WeightBridge 的探测式对应关系发现 | 实现复杂度，且 canonical 命名空间已能支撑将来接入（§8） |
| 写屏障降级为可选（8c） | 理论上更短的 pause 窗口 | 实现确定性——直供在现有 `pause_for_sync` 下就正确 |

---

## 6. 完整性

### 6.1 本轮关闭一处「正确性空白」

§3.10.4b 把「NCCL 通信组在角色切换时的处理」标为**正确性空白**，
理由是 PivotRL 把它归为「须保留在 GPU」且重建代价高。

**核实 PSRL 后发现这个空白在 PSRL 下不存在**：

```
grep -rn "destroy_process_group|destroy_model_parallel" --include=*.py psrl/
→ 零命中
```

**PSRL 从不销毁进程组**。`switch_to_*_mode` 只做 NIXL dereg + 显存 resize，
`init_process_group` 在 worker 构造时调一次（`engine_train_worker.py:78`）后不再变动。
所以通信组天然「保留在 GPU」，与 PivotRL 的结论一致，**只是 PSRL 本来就是对的**。

**建议**：把这一项从「正确性空白」改为「**已满足的不变量，需加断言防回归**」——
即在角色切换路径上断言 `torch.distributed.is_initialized()` 恒为真，
防止将来有人为了省显存去销毁通信组。

### 6.2 仍然缺的三项

| 项 | 性质 | 说明 |
|---|---|---|
| **`epoch` 与 `version` 的交互** | 正确性 | 见 §2.2a，两段独立校验，设计里没写 |
| **档位 B 下 GPS 的一致性维护** | 正确性（档位 B） | 见 §4.4，CPU holder 内容陈旧会被选中 |
| **PCIe 带宽作为互斥资源** | 性能 | 见 §2.2b，ref restore 与 actor offload 可能叠加 |

### 6.3 一处范围说明而非缺口

**本设计不覆盖「训练中途改变并行度」**（弹性只沿 DP 维，§3.10.4c）。
这是与 PivotRL 一致的工程限制，理由是改并行度会动 finest mesh，
牵动所有 client 的 descriptor。应当在 §3.3.1c 显式写成约束而非留白。

---

## 7. 与 WeightBridge 的对照

### 7.1 已吸收的四项

| WeightBridge 机制 | 本设计的位置 | 吸收形式 |
|---|---|---|
| **三阶段路由**（receive / external-exchange / internal-exchange） | §3.5 | **完整吸收**，是去冗余规划的骨架 |
| **`T_min` 流量下界** | §3.5、§7 验证项 2 | 作为优化目标与验收判据 |
| **round-based 批处理** | §3.6 | chunk + 双 buffer 深度 2 |
| **pack/assemble 与传输重叠** | §3.6 的三类重叠 | 扩展为「cast / RDMA / 落地格式化」三类 |

### 7.2 未吸收的三项及原因

| 未吸收 | 原因 |
|---|---|
| **探测 `load_weights` 自动推导对应关系** | **约束 (a) 明确排除**。PSRL 有成熟的显式 layout plan（`model_mappings.py`），自动推导是另一个独立项目。§8 保留了接口余地：canonical 命名空间与 plan-as-data 两点不变，将来接入不需返工 |
| **trainer 发起的推送语义** | 与 PSRL 的 pull 语义冲突。WeightBridge 的 plan 由 trainer 编译并覆盖固定 worker 集合，而 PSRL 要支持 rollout 自行择时拉取与弹性伸缩。**本设计把 plan 按 instance 切分、握手期编译**，这是对 pull 语义的适配（§3.5 开头已说明） |
| **固定 worker 集合的假设** | 同上。弹性伸缩要求 holder 集合可变，这也是 §3.3.1c 与 9c 存在的理由 |

### 7.3 一处超出 WeightBridge 的设计

**holder 的对称性**（§3.1）是 WeightBridge 没有的概念。
它假定 trainer 是唯一的权威源，而 PSRL 的三种部署形态
（colocate 切换、全 rollout 启动、弹性伸缩）都会让源转移。
这不是优化而是正确性要求——`is_rollout_mode_in_actor=True` 下
trainer 以 `meta` 模式注册、没有真实显存。

---

## 8. 测试验证规划：这是本方案最薄弱的一环

### 8.1 现状盘点

`tests/nixl/test_nixl_e2e.py`（891 行）的结构：

| 组件 | 内容 |
|---|---|
| `MetaServerActor` | NIXL meta server |
| `TrainClientActor` | 支持 `fsdp` / `fsdp_hybrid` / `megatron` 三种引擎 |
| `GenClientActor` | vLLM 侧，TP/PP/EP/DP 可配 |
| `create_ps_worker_group` | **PS worker，本设计要删除的东西** |
| 正确性判据 | `assert_state_dict_all_ones`——**把权重设成全 1 再验证全 1** |

配置（`config/nixl_e2e.yaml`）已参数化：`train_engine_type`、Megatron 的 TP/PP/EP/ETP、
gen 的 TP/PP/EP/DP、`train_dtype` / `gen_dtype`。

### 8.2 能否改造：可以，但要先做三处结构性改动

**(a) 删除 `create_ps_worker_group`，改为 trainer 直供。**
这是必然的——PS worker 本身要被删除。改造后 `GenClientActor` 的 pull target
从 `ps_for_pull_name` 换成 train client，这与设计 §3.3 的改动一一对应。
**工作量不大，因为测试的骨架（meta server + train/gen client）正是直供需要的两端。**

**(b) 正确性判据必须从「全 1」升级。**
`assert_state_dict_all_ones` 只能验证「数据到了」，**验不出**：

- 布局是否正确（全 1 张量转置后还是全 1）
- 分片是否错位（全 1 的任意切片都是全 1）
- 精度转换是否正确（全 1 cast 到 fp8 还是全 1）

**必须改为「每个元素是其全局坐标的函数」**，例如
`value = hash(key) * 1e6 + flat_index`。这样任何错位、转置、
分片错误都会被检出。**这是改造的第一优先级**，
因为现有判据对本设计要验证的东西（§3.9.2 的格式转换、§3.8 的精度矩阵）**完全无效**。

**(c) 参数化维度从 1 维扩到 4 维。**
现在是单一配置跑一次，需要改成 pytest 参数化矩阵。

### 8.3 改造后的覆盖矩阵

| 维度 | 取值 | 现有支持 |
|---|---|---|
| 后端 | megatron / fsdp2 | **已有**（删掉 fsdp1） |
| 精度 | bf16+bf16 / bf16+fp8 / fp32+bf16 / fp32+fp8 | 配置项已有，**cast 路径未实现** |
| 放置 | disaggregate / colocate | **无**，需新增 colocate 模式 |
| 并行度 | train TP/PP/EP × gen TP/EP | **已有**，只需扩参数化 |

**23 条验证项的可覆盖性**：

| 可由改造后的 e2e 覆盖 | 需要独立测试 | 需要集群实测 |
|---|---|---|
| 7（atomic group）、8（等价性）、11（四条断言）、14（精度矩阵）、16（后端对称性）、17（FSDP2 五条）、19（TP 不匹配）、22（role 隔离） | 5（注册有效性三条）、18（布局产物时效）、20（buffer 上界）、21（多角色切换） | 1–4、6、9、10、12、13、15、23 |
| **8 条** | **4 条** | **11 条** |

**只有三分之一能由 e2e 覆盖，这是方案的真实风险**：
11 条需要真实集群（多节点、故障注入、长跑），
而这些恰恰是「万卡 / 1TB」声明所依赖的那些。

### 8.4 建议的测试分层

```
L1 单元（现有 test_nixl_sharding.py / test_comm_planner.py 扩展）
    shard 代数、finest mesh、comm plan 的去冗余正确性
    新增：FSDP2 的 uneven sharding 断言、MoE 分组后的多维 shard
L2 e2e 单节点（test_nixl_e2e.py 改造）
    4 维参数化矩阵，判据改为坐标函数
    覆盖 8 条验证项
L3 e2e 多节点（新建）
    peer 扩散、W 限流、故障注入、colocate 切换
    覆盖 4 条
L4 规模实测（无法自动化，需人工跑）
    1TB / 万卡、带宽对齐 T_min、长跑漂移
    覆盖 11 条
```

**L2 的改造是性价比最高的一步**，因为它的骨架已经存在，
而「判据升级 + 4 维参数化」能一次性覆盖本设计最容易出错的部分
（格式转换、精度矩阵、后端差异）。

---

## 9. 过时的限制与可简化的冗余设计

### 9.1 本轮发现可关闭的两项

| 项 | 处理 |
|---|---|
| **NCCL 通信组的「正确性空白」** | **关闭**。PSRL 从不销毁进程组（grep 零命中），这个不变量天然满足。改为「加断言防回归」（§6.1） |
| **「FSDP2 + 同精度」一律拒绝** | **改为支持**：disaggregate 下版本级等待是正确性必需，colocate 下靠时序互斥天然安全（§4.3） |

### 9.2 前四轮已清理，本轮确认无残留

逐项核实，以下在前几轮被推翻的说法已无残留：

- 「colocate 必须走量化路径」→ 已改为时序互斥（grep 无残留）
- 「权重区不动」→ 已改为 stage 边界 offload（grep 无残留）
- 「阶段上报 + TTL + stage_factor」→ 已移出本期，只留回调发布
- 「per-key IPC 判定」→ 已删除，改为无条件 pack
- 「ref 与 actor 共享 v0 的精度判定」→ 已随 ref 改为备份 tag 而删除
- `from_trainer` → 已统一为 `is_origin`
- 「弹性 = 时间片调度」→ 已更正为「GPU 数量调度可行，档位 A/B」

### 9.3 一处仍可简化的冗余

**§3.4.3 的 `conflict_wait_budget_ms` 与 INVALID 机制。**

它服务的是 8c（写屏障），而 8c 已降级为「应最后做，且可以不做」。
但 §3.4.3 的篇幅与位置（在主干 §3.4 内）仍然像是核心机制。

**建议**：把 §3.4.3 的机制部分（硬上限、INVALID 状态码、重拉协议）
**移入 8c 的落地说明**，主干只保留那条取舍原则
（「等待的上限由算法目标决定，不由 infra 方便决定」），
因为**那条原则独立于是否实现写屏障**——它约束的是「遇到传输压力时能不能调大 staleness」，
这个诱惑在任何实现下都存在。

---

## 10. 修订建议汇总

按优先级：

### 已落地（本轮直接改入主文档）

| # | 项 | 落点 |
|---|---|---|
| 1 | `epoch` 与 `version` 两段独立校验，区分 `CHUNK_STALE` / `DESC_STALE` | §3.7.3.0a 新增小节 + 验证项 23 |
| 2 | **「FSDP2 + bf16 + bf16」从拒绝改为支持** | §3.8.0 矩阵 + 新增 §3.8.0a，并把 8c 拆成「版本级等待（必需）/ per-chunk 写屏障（优化）」 |
| 3 | 档位 B 的 GPS 一致性 | §3.10.4a 新增三条修复，采纳 (1)+(3) |
| 4 | **测试分层与改造方案** | §7.0 新增：判据升级为坐标函数、删 PS worker、新增 colocate、四维参数化矩阵、L1–L4 分层 |
| 5 | `conflict_wait_budget_ms` 机制移入 8c | §3.4.3 只留取舍原则，机制细节入 §5 的 8c 说明 |
| 6 | GPUDirect 降级为启动期告警 | §3.6.2a 重写 |
| 7 | **关闭 NCCL 空白，改为防回归断言** | §3.10.4b 重写（附 grep 证据）+ 待办表改性质 + 验证项 17b |
| 8 | **拆分 9d 为 9d-a / 9d-b** | §5 清单拆分，9d-a 与 1-5 同批；验证项 17(f) 新增 `model_dtype` 真实读取 |
| 9 | **NIXL 新 API 的 micro-benchmark** | §5 清单新增**落地项 0**，排在 1 之前 |
| 10 | **完整的三维支持矩阵**（后端 × 放置 × 精度） | §3.9.0 新增，十一格逐格给出支持状态与所依赖机制 |

### 交付前仍必须完成（本轮末追加）

**结论：方案不能现在交付。** 两件事卡着：

| # | 项 | 性质 |
|---|---|---|
| 11 | **chunk 大小规则算错了**（§3.5.6 已重写） | **正确性级的设计错误**，见下 |
| 12 | ~~落地项 0 的 micro-benchmark~~ | **已完成**，见下 |

#### 第 12 项已完成（NIXL 1.5.0 源码构建）

从 `third_party/nixl`（v1.5.0）源码编译安装后重测
（`tests/nixl/bench_nixl_prepped_xfer.py`，H20 + UCX 1.20）：

| descriptors | prep | `make_prepped_xfer` | `initialize_xfer` | 加速 |
|---|---|---|---|---|
| 1024 | 0.007ms | 0.004ms | 0.024ms | **5.4x** |
| 4096 | 0.019ms | 0.006ms | 0.096ms | **16.8x** |
| 16384 | 0.085ms | 0.014ms | 0.424ms | **30.1x** |

**落地项 1 的收益成立且随规模放大**，但有两个前提：

1. **索引必须是 int32 ndarray**。传 Python list 慢 15 倍
   （4096 条下 0.088ms vs 0.006ms）——pybind11 逐元素转换，
   **恰好把 prepped 要消除的 per-descriptor 开销加了回来，且不报错**。
   **本 benchmark 的早期版本就踩了这个坑**，因而一度测出「prepped 更慢」的错误结论。
2. **Nx5 strided 需要 NIXL ≥ 1.5.0**。`scripts/install_nixl.sh` 原本 pin v1.2.0，
   其 `prepXferDlist` 只有两个重载（都是 Nx3）。1.5.0 的第三个重载走
   `to_stride_dlist`，实测 **16384:1 压缩比**，prep 0.002ms / submit 0.001ms，
   **两者都与 count 无关**。已把安装脚本升到 v1.5.0。

**一处澄清**：16384 条 descriptor 的 `initialize_xfer` 也只要 0.42ms，
相对 312ms RDMA 是噪声。所以 WeightBridge 的「590ms 控制面开销」
**是 per-tensor 格式转换而非提交 API**，§1.2 原本混为一谈。

Q4 确认 prepped handle 接受任意索引顺序与子集（identity/rotated/subset/reversed 全过），
支撑 chunk 偏移错开与 commit group 子集提交。

修正已落入 §1.2、§3.5.5、§3.6.3（重写）、落地清单第 0/1 项，
以及 `scripts/install_nixl.sh`。

### 本轮撤回

| 项 | 原因 |
|---|---|
| ~~PCIe 带宽纳入 `_ensure_on_gpu` 的互斥资源~~ | eval 与训练步互斥，ref restore 与 actor offload 不会并发（§2.2b） |

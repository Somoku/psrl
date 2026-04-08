# PSRL Elastic RM 模块深度研究报告

> 生成日期：2026-04-07
> 分析范围：`psrl/utils/elastic_rm/`（全部文件）及 `psrl/trainer/ppo/ray_trainer.py`（`init_elastic_rm_runtime`、`init_workers` 相关部分）

---

## 目录

1. [模块背景与核心目的](#1-模块背景与核心目的)
2. [架构总览](#2-架构总览)
3. [文件与类逐一说明](#3-文件与类逐一说明)
4. [函数级详解](#4-函数级详解)
5. [运行时工作流](#5-运行时工作流)
6. [Scaling 策略深度解析](#6-scaling-策略深度解析)
7. [调用链分析](#7-调用链分析)
8. [关键设计决策](#8-关键设计决策)
9. [已知问题与边界条件](#9-已知问题与边界条件)

---

## 1. 模块背景与核心目的

### 1.1 问题背景

在 PSRL 的异步强化学习训练流程中，集群 GPU 资源同时承载两类推理负载：

- **Rollout（生成推理）**：大语言模型自回归地生成对话轮次，供训练采样使用。
- **Reward Model（奖励模型推理）**：对生成的结果打分，反馈给 PPO 算法。

这两类负载**不需要在同一时刻以相同强度同时运行**。生成高峰期奖励模型可能处于低负载状态，反之亦然。如果让两者各自独占一部分 GPU，则实际上任何时刻都会有大量 GPU 空闲浪费。

### 1.2 Elastic RM 的核心目标

**在同一批物理 GPU 上，以时间复用（Time-Multiplexing）方式动态分配 Rollout 和 RewardModel 实例**，使总 GPU 利用率最大化，同时保证训练吞吐不下降。

实现机制依赖 `torch_memory_saver`（TMS）：vLLM 实例可以在运行时将权重 + KV 缓存从 GPU 显存卸载到 CPU，从而"让出"GPU，让另一个实例在同一张 GPU 上加载并运行。

---

## 2. 架构总览

```
psrl/trainer/ppo/ray_trainer.py
  └─ PSRL_RayPPOTrainer
       ├─ init_workers()                      ← 构建所有 Worker Group（含共享 placement group）
       └─ init_elastic_rm_runtime()           ← 弹性调度运行时初始化入口
            ├─ 全部实例 SLEEP（初始状态）
            ├─ select_initial_awake_ids()     ← 挑选非冲突的初始唤醒实例
            ├─ WAKE_UP 初始实例
            └─ ElasticExecutor.remote()       ← Ray 远程 Actor（调度中枢）
                 ├─ _monitor_loop()           ← 异步任务：周期性拉取状态 → 决策
                 ├─ _scale_up_handler_loop()  ← 异步任务：消费 scale_up 任务
                 └─ _scale_down_handler_loop()← 异步任务：消费 scale_down 任务

psrl/utils/elastic_rm/
  ├─ elastic_executor.py   ElasticExecutor  调度中枢（Ray remote Actor）
  ├─ cluster_topology.py   ClusterTopology  GPU 占用拓扑（双向索引）
  ├─ scaling_policy.py     ScalingPolicy    弹性调度决策策略（纯函数）
  │                        ThroughputProfileLoader  吞吐量模型加载器
  ├─ diagnostics.py        日志诊断工具函数
  └─ __init__.py           （空）
```

### 数据流（每个监控周期）

```
Coordinators                ElasticExecutor                 ScalingPolicy
    │                            │                               │
    │←─ get_engine_status ───────│                               │
    │←─ get_router_backlog ──────│                               │
    │                            │                               │
    │     AgentLoopManager       │                               │
    │←─ get_trainer_waiting ─────│                               │
    │                            │                               │
    │                            │─── build InstanceSignal[] ───▶│
    │                            │                    decide()   │
    │                            │◀── ScalingDecision ──────────│
    │                            │                               │
    │◀── exec_command(SLEEP) ────│   (由 handler 循环执行)        │
    │◀── exec_command(WAKE_UP) ──│                               │
    │◀── exec_command(ABORT) ────│                               │
```

---

## 3. 文件与类逐一说明

### 3.1 `cluster_topology.py` — GPU 拓扑管理

#### `GPUSlot`（frozen dataclass）

唯一标识集群中的一张 GPU：
```python
@dataclass(frozen=True, order=True)
class GPUSlot:
    node_id: str | None   # Ray 节点 ID
    gpu_id: int           # 该节点上的 GPU 编号（物理/逻辑）
```

#### `InstanceIdentifier`（frozen dataclass）

跨角色、跨模型唯一标识一个实例：
```python
@dataclass(frozen=True)
class InstanceIdentifier:
    role: PSRL_Role
    model_name: str
    instance_id: RolloutInstanceId  # tuple[str, int] = (replica_id, dp_rank)
```

#### `InstanceStatus`（Enum）

实例当前的 GPU 占用状态：
- `ASLEEP`：权重/KV 缓存已从 GPU 卸载，GPU 空闲。
- `AWAKEN`：实例正常运行，占据 GPU。

#### `InstanceInfo`（dataclass）

单个实例的全部拓扑状态：`key`、`gpu_slots`（frozenset[GPUSlot]）、`status`。

#### `ClusterTopology`

**集中式 GPU-实例双向索引，提供 O(1) 冲突检测。**

内部维护两个索引：
- `instances: dict[InstanceIdentifier, InstanceInfo]` — 正向索引（实例 → GPU 集合）
- `gpu_to_instances: dict[GPUSlot, set[InstanceIdentifier]]` — 反向索引（GPU → 实例集合）

两个索引在每次修改时同步更新，调用方无需自行维护反向映射。

---

### 3.2 `scaling_policy.py` — 弹性调度决策

#### `InstanceSignal`（dataclass）

每个实例的一份运行时快照，作为决策输入：

| 字段 | 含义 |
|------|------|
| `role_name` | 角色（Rollout / RewardModel） |
| `model_name` | 模型名称 |
| `instance_id` | 实例 ID |
| `is_awaken` | 当前是否唤醒 |
| `kv_cache_utilization` | KV 缓存占用率 [0, 1] |
| `running_queue_num` | 引擎正在处理的请求数 |
| `waiting_queue_num` | 队列中等待的请求数 |
| `generation_throughput` | 实时生成吞吐量（token/s） |
| `total_token_num` | 当前实例处理的总 token 数 |
| `snapshot_timestamp` | 快照时间戳（用于陈旧检测） |
| `gpu_keys` | frozenset[(node_id, gpu_id)]，来自 ClusterTopology |

#### `ScalingAction`（dataclass）

一个调度动作：
- `action_type`：`"scale_up"` 或 `"scale_down"`
- `role_name` / `model_name`：目标角色和模型
- `preferred_instance_ids`：策略倾向唤醒/休眠的实例 ID（Executor 可降级到其他实例）
- `reason`：日志用的决策原因字符串

#### `ScalingDecision`（dataclass）

一次完整的调度决策：`actions` 列表、`reason`、`estimated_lambda`（估计请求到达率）、`role_to_total_mu`（各角色总吞吐能力）。

#### `ThroughputProfileLoader`

离线拟合的吞吐量模型加载器，支持三种数据源（优先级递降）：

1. **公式拟合文件** (`throughput_model_dir/<model>_token.json`)：
   公式：`mu(x) = A * (1 - (B*x + 1)^(-k))`，其中 x = running_queue_num。
2. **新式 profile JSON**（`profile_paths[model_name]`）：
   支持 `default_mu`、`mu_by_running_queue`（运行队列 → 吞吐率的查找表）、`mu_table`（多条件阈值表）。
3. **运行时实测值** `generation_throughput`：最终 fallback。

#### `ScalingPolicy`

**纯决策对象（无副作用）**，接受 `InstanceSignal` 列表，输出 `ScalingDecision`。

---

### 3.3 `elastic_executor.py` — 调度执行中枢

#### `ElasticExecutor`（Ray remote Actor）

**整个弹性调度系统的核心，以 Ray 远程 Actor 形式运行，内部跑三条并发 asyncio 任务循环。**

主要状态：

| 属性 | 类型 | 含义 |
|------|------|------|
| `coordinators` | `dict[PSRL_Role, dict[str, ActorHandle]]` | 各角色/模型对应的 Coordinator |
| `instances_status_flags` | 三层嵌套 dict | 每个实例当前的 ASLEEP/AWAKEN 状态 |
| `instances_engine_stats` | 三层嵌套 dict | 每个实例的最新引擎快照 |
| `topology` | `ClusterTopology` | GPU 拓扑，用于冲突检测 |
| `scaling_policy` | `ScalingPolicy` | 决策对象 |
| `scale_up_task_queue` | `asyncio.Queue` | 待执行的扩容任务 |
| `scale_down_task_queue` | `asyncio.Queue` | 待执行的缩容任务 |
| `_decision_execution_in_progress` | `bool` | 当前是否有决策在执行（防并发） |

---

### 3.4 `diagnostics.py` — 诊断日志

提供一个可通过环境变量 `PSRL_ELASTIC_RM_BACKLOG_DIAG=1` 开启的诊断日志函数 `log_elastic_rm_backlog_diag`，用于追踪 `get_router_backlog_size` RPC 调用链的超时问题。

---

## 4. 函数级详解

### 4.1 `ClusterTopology` 函数

#### `register(role, model_name, instance_id, gpu_slots, status)`
注册一个实例及其 GPU 占用槽。若已存在则先清除旧的反向索引再重建，支持幂等重注册。

#### `update_gpu_slots(role, model_name, instance_id, gpu_slots)`
更新已注册实例的 GPU 布局（例如动态迁移场景），同步维护反向索引。

#### `set_status / get_status`
原子性读写单个实例的 ASLEEP/AWAKEN 状态，修改 `InstanceInfo` 对象（frozen → 重建）。

#### `has_other_role_awaken_on_shared_gpu(role, model_name, instance_id)`
**冲突检测核心**：遍历该实例的所有 GPU 槽，查反向索引找出同 GPU 上的其他实例，若其中有任何来自不同 role 且状态为 AWAKEN，则返回 True。
时间复杂度：O(GPU 数量 × 每 GPU 上的实例数)，通常为 O(1)~O(小常数)。

#### `get_awaken_gpu_slots()`
遍历所有实例，收集所有 AWAKEN 实例占用的 GPUSlot 集合。用于无冲突实例选择。

#### `select_non_conflicting_awake_ids(role, model_name, instance_ids, target_awake_num, min_awake_num)`
**贪心选取非冲突实例**：按 instance_id 排序后依次尝试，累积已选实例的 GPU 槽，跳过与已选或已占用 GPU 冲突的候选。若最终选中数量 < `min_awake_num` 则抛出 RuntimeError。

#### `collect_gpu_slots_from_worker_group(worker_group)` (static)
通过 Ray 调用 worker group 中每个 worker 的 `get_runtime_gpu_ids` 和 `get_node_id`，构建 frozenset[GPUSlot]。假设每个 worker 使用一张 GPU（取第一个 accelerator id）。

---

### 4.2 `ThroughputProfileLoader` 函数

#### `estimate_mu_by_running_queue(model_name, running_queue_num, fallback_mu)`
按三级优先级估计某模型在指定 running_queue 长度下的吞吐量（token/s）：
① 公式参数文件 → `A*(1-(B*x+1)^(-k))`
② fallback_mu 参数
③ 返回 None

#### `estimate_instance_mu(signal)`
基于单个 `InstanceSignal` 估计实例吞吐量，完整的三级优先链（公式 → profile JSON → 运行时实测值）。

---

### 4.3 `ScalingPolicy` 函数

#### `decide(signals, execution_in_progress, router_backlog_by_role, trainer_waiting_hint)`
**主决策入口**，含以下前置门控（任意一个触发则提前返回 no-action）：

1. `enable` 为 False → `policy_disabled`
2. `signals` 为空 → `empty_signals`
3. `execution_in_progress` 为 True → `decision_execution_in_progress`
4. 冷却期未过（`cooldown_ms`）→ `cooldown`
5. **Force Wake**：若某 role 零唤醒实例但 router backlog > 0，强制唤醒一个实例（优先级高于所有信号陈旧检查）。
6. 所有信号均陈旧（timestamp 超过 5 秒）→ `all_signals_stale`

通过门控后调用 `_make_stepwise_decision`，返回 `ScalingDecision`。

#### `_make_stepwise_decision(grouped, instance_mu, role_total_mu, trainer_waiting_hint)`
五优先级决策逻辑（详见第 6 节）。

#### `_estimate_lambda(signals, role_to_total_mu)`
用 EWMA（指数加权移动平均）估算当前请求到达率 λ：
`raw_λ = total_mu + ΔQueue / Δt`
`λ_ewma = α * raw_λ + (1-α) * λ_ewma_prev`

#### `_build_mu_maps(signals)`
为每个实例计算吞吐能力 μ（通过 profile loader），汇总每个 role 的总 μ。

#### `_pick_scale_up_candidate(role_signals, instance_mu)`
从 ASLEEP 实例中选出 μ 最高（最有价值）的候选唤醒实例。

#### `_pick_scale_down_candidate(role_signals, instance_mu)`
从 AWAKEN 实例中选出 KV 缓存利用率 ≤ `theta_low`（空闲）且 μ 最低的候选休眠实例。用于自发缩容（P3）和 P1 转让。

#### `_pick_scale_down_candidate_for_bottleneck_transfer(role_signals, instance_mu)`
从 AWAKEN 实例中选出 μ 最低的候选休眠实例。**不要求 KV ≤ theta_low**（双满载时不存在低负载实例），用于 P2 瓶颈转让。

#### `_pick_scale_up_candidate_on_free_gpu(full_role_signals, other_role_signals, instance_mu)`
在满载 role 的 ASLEEP 实例中，找出 GPU 与对方 role 所有 AWAKEN 实例完全不重叠的候选。此类实例可直接唤醒无需 sleep 对方，是最优扩容路径（P1 Step A）。

#### `_estimate_role_total_mu_with_rebalance(role_signals, scale_up_signal, scale_down_signal)`
**模拟扩/缩容后的预期总吞吐量**，用于 P2 决策前的增益预估：
- 扩容模拟：把原有队列平均分摊到 N+1 个实例，重新用 profile 估算每实例 μ 后求和。
- 缩容模拟：把被移除实例的 running 队列均摊到剩余 N-1 个实例，同样重算。

---

### 4.4 `ElasticExecutor` 函数

#### `register_role(role_name, model_name, instance_ids, gpu_slots_per_instance)`
批量注册一个角色的所有实例，初始状态 ASLEEP，引擎统计初始化为默认空快照。

#### `select_initial_awake_ids(role_name, model_name, target_awake_num, min_awake_num)`
初始化时选出非冲突实例并在本地状态中标记为 AWAKEN。
**注意**：此方法会修改本地 status 字典，连续调用不同 role 时自动避开已占用 GPU。

#### `start_busy_loop() / stop_busy_loop()`
启动/停止三条异步任务：monitor、scale_up_handler、scale_down_handler。

#### `_monitor_loop()`
**主监控循环**，每 `monitor_interval_ms` 毫秒执行一次：

1. `_sync_engine_status_from_coordinators()` — 拉取各 Coordinator 的引擎状态快照
2. `_sync_router_backlog_from_coordinators()` — 拉取路由器积压队列大小
3. `_sync_trainer_waiting_hint()` — 拉取 AgentLoopManager 的 trainer 等待提示
4. `_build_instance_signals()` — 组装 InstanceSignal 列表
5. `scaling_policy.decide(...)` — 调用决策策略
6. 将 action 分发到 `scale_up_task_queue` / `scale_down_task_queue`

若当前有决策在执行中（`_decision_execution_in_progress=True`），记录 stall tick 计数，超过阈值则强制放弃该决策状态（`_abandon_in_flight_decision`）。

#### `_scale_up_handler_loop()`
消费 `scale_up_task_queue`，每次处理一个扩容任务：

1. `_find_instances_to_scaled_down_for_other_roles()` — 检查是否需要先 sleep 其他 role 的实例以释放 GPU
2. 并行执行所有前置 sleep（`_scale_down_instance`）
3. `_find_instances_to_scaled_up()` — 找出实际要唤醒的实例（GPU 冲突过滤）
4. 并行执行所有 wake_up（`_scale_up_instance`）
5. `_interrupt_waiting_after_scale_up()` — ABORT 等待队列中的旧请求，加速重新分发

#### `_scale_down_handler_loop()`
消费 `scale_down_task_queue`，每次处理一个缩容任务：
1. `_find_instances_to_scaled_down_in_role()` — 按 KV 缓存利用率排序，选出目标实例
2. 并行执行所有 sleep（`_scale_down_instance`）

#### `_scale_up_instance(instance_to_scaled_up)`
发送 `CommandType.WAKE_UP` 到对应 Coordinator，成功后更新本地状态为 AWAKEN，并同步更新 ClusterTopology。

#### `_scale_down_instance(instance_to_scaled_down)`
安全检查（`min_awake_per_role` 限制、最后一个唤醒实例仍有任务时不休眠），通过后发送 `CommandType.SLEEP` 到对应 Coordinator，成功后更新本地状态为 ASLEEP。

#### `_interrupt_waiting_after_scale_up(role_name, model_name)`
扩容后对已唤醒的所有实例发送 `CommandType.ABORT`，中断 `post_scale_up_abort_waiting_ratio` 比例的等待请求。被 abort 的请求会重新进入路由器，此时可以被刚唤醒的新实例处理，实现负载再均衡。

#### `_find_instances_to_scaled_up(role_need_to_scale_up)`
从 ASLEEP 实例中：①优先使用策略推荐的 `preferred_instance_ids`；②若优先实例均被 GPU 冲突过滤，降级到全部 ASLEEP 实例中挑选非冲突的；③返回前 `num_instances` 个。

#### `_find_instances_to_scaled_down_for_other_roles(role_need_to_scale_up)`
在"隐式转让"（P1 Step B）场景中，选出其他 role 中 KV 缓存利用率最低的 AWAKEN 实例来 sleep，为目标 role 腾出 GPU 空间。受 `min_awake_per_role` 和 `removable_budget` 限制。

#### `_find_instances_to_scaled_down_in_role(role_need_to_scale_down)`
在同 role 内，选出 KV 缓存利用率最低的 AWAKEN 实例来 sleep，受 `min_awake_per_role` 约束。

#### `_sync_engine_status_from_coordinators()`
并行调用所有 Coordinator 的 `get_instance_engine_status_snapshot` RPC，使用 `_await_coordinator_refs_with_per_ref_timeout` 进行每 ref 独立超时控制，将结果写入 `instances_engine_stats`。

#### `_sync_router_backlog_from_coordinators()`
并行调用所有 Coordinator 的 `get_router_backlog_size` RPC，按 role 汇总积压量，写入 `router_backlog_by_role`。

#### `_sync_trainer_waiting_hint()`
调用 `AgentLoopManager.get_trainer_waiting_hint` 获取 trainer 当前是否在等待 rollout/reward，以及正在等待的 pending 数量。失败时降级为 `trainer_busy=True`（保守默认）。

#### `_build_instance_signals()`
遍历所有实例状态，从 `instances_engine_stats` 中读取 scheduler 统计，从 `topology` 中读取 gpu_keys，组装 `InstanceSignal` 列表。

#### `_await_coordinator_refs_with_per_ref_timeout(refs, task_keys, op_label)`
**每个 RPC 独立设置超时**，防止一个慢 Coordinator 阻塞整个监控周期。超时的 ref 返回 `TimeoutError` 对象而非抛异常，调用方可以跳过该 tick 的对应数据。

#### `_await_elastic_coordinator_command(coordinator, command, stage)`
带结构化追踪日志的 coordinator `exec_command` 封装，记录 START/END/EXCEPTION 及耗时，便于排查 SLEEP/WAKE_UP/ABORT 卡死问题。

#### `_mark_decision_action_finished(decision_id)`
追踪决策执行进度：每个 action 完成时调用，所有 action 均完成时清除 `_decision_execution_in_progress`，允许策略接受下一个决策。

#### `_abandon_in_flight_decision(reason, stall_ticks)`
强制清除决策执行锁，使策略不再被阻塞。handler 任务仍可能在后台完成 RPC，此时 `_mark_decision_action_finished` 成为 no-op。

---

## 5. 运行时工作流

### 5.1 初始化阶段（`init_workers` + `init_elastic_rm_runtime`）

```
1. init_workers():
   a. 创建 elastic_shared_pool（单一共享 placement group，覆盖所有 GPU）
   b. 为每个 rollout 实例用 _build_elastic_sub_resource_pool() 创建 SubRayResourcePool
      - SubRayResourcePool 从共享 PG 中切取连续 bundle（循环错位分配，减少 GPU 冲突）
   c. 为每个 reward_model 实例同样创建 SubRayResourcePool
   d. 创建 train/reward_model/gen worker groups（多线程并行初始化）

2. init_elastic_rm_runtime():
   a. 启动 rollout_coordinator 的 busy loop（用于接受 elastic SLEEP/WAKE_UP 命令）
   b. 创建 ElasticExecutor Ray Actor
   c. 向各 Coordinator 发送 SLEEP 命令，让所有实例进入休眠
   d. 调用 ElasticExecutor.register_role() 注册 Rollout 实例（含 GPU 槽信息）
   e. 对每个 reward model 同样 SLEEP + register_role
   f. 调用 select_initial_awake_ids() 选出非冲突的初始唤醒 RM 实例 → WAKE_UP
   g. 选出初始唤醒 Rollout 实例 → WAKE_UP
   h. 调用 ElasticExecutor.start_busy_loop() 启动三条监控/处理循环
```

**关键点**：所有实例先全部 SLEEP，然后 ElasticExecutor 通过 `select_initial_awake_ids` 保证：
- RM 实例先被选取，占用其 GPU 槽
- Rollout 实例再从剩余空闲 GPU 中选取
- 从一开始就不存在 GPU 冲突

### 5.2 稳态监控阶段（监控循环）

```
每 monitor_interval_ms（默认 1000ms）执行一次：

[tick N]
├─ 并行 RPC：拉取所有 Coordinator 的 engine_status_snapshot（每 ref 独立超时）
├─ 并行 RPC：拉取所有 Coordinator 的 router_backlog_size
├─ RPC：拉取 AgentLoopManager.get_trainer_waiting_hint
├─ _build_instance_signals() → List[InstanceSignal]
├─ ScalingPolicy.decide() → ScalingDecision
│    └─ 返回 0~2 个 ScalingAction（通常为 0 或 1 个）
└─ 若 decision.actions 不为空：
     ├─ 分配 decision_id
     ├─ 设置 _decision_execution_in_progress = True
     └─ 将 action 放入 scale_up_task_queue 或 scale_down_task_queue
```

### 5.3 扩容执行阶段（scale_up_handler）

```
收到 scale_up task：

1. _find_instances_to_scaled_down_for_other_roles()
   └─ 若目标 GPU 被其他 role 占用：选出其他 role 的低 KV 实例

2. 并行 SLEEP 其他 role 实例（若有）：
   └─ coordinator.exec_command(SLEEP, instance_ids=[...])

3. _find_instances_to_scaled_up()
   ├─ 优先使用策略推荐的 instance_id
   ├─ GPU 冲突过滤（cross-role 检查）
   └─ 降级到任意非冲突 ASLEEP 实例

4. 并行 WAKE_UP 目标实例：
   └─ coordinator.exec_command(WAKE_UP, instance_ids=[...])

5. _interrupt_waiting_after_scale_up()
   └─ coordinator.exec_command(ABORT, instance_to_uids={...})
   └─ 中断比例由 post_scale_up_abort_waiting_ratio 控制（默认 1.0）

6. _mark_decision_action_finished(decision_id)
   └─ 若所有 action 完成 → _decision_execution_in_progress = False
```

### 5.4 缩容执行阶段（scale_down_handler）

```
收到 scale_down task：

1. _find_instances_to_scaled_down_in_role()
   ├─ 优先 preferred_instance_ids
   ├─ 按 KV 缓存利用率升序排列
   └─ 受 min_awake_per_role 和 max_scalable_down 限制

2. 并行 SLEEP 目标实例：
   ├─ 单实例安全检查：若 min_awake_per_role=0 且是最后一个唤醒实例且仍有任务，跳过
   └─ coordinator.exec_command(SLEEP, instance_ids=[...])

3. _mark_decision_action_finished(decision_id)
```

---

## 6. Scaling 策略深度解析

### 6.1 策略参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `theta_low` | 0.3 | KV 缓存低水位线，低于此值认为实例空闲（可缩容/可转让） |
| `theta_max` | 0.85 | KV 缓存高水位线，高于此值认为实例满载（需扩容） |
| `cooldown_ms` | 3000 | 两次决策之间的最小间隔（毫秒），防止频繁震荡 |
| `hysteresis` | 0.05 | P2 转让的最小增益阈值（必须超过此值才执行） |
| `min_awake_per_role` | 0 | 每个 role 最少保持唤醒的实例数 |
| `full_load_mode` | `"any"` | 满载判断：`"any"` = 任意一个实例 ≥ theta_max；`"all"` = 所有实例均 ≥ theta_max |
| `max_waiting_queue_for_scale_down` | 0 | P3 缩容时，等待队列超过此值则跳过缩容 |

### 6.2 五优先级决策树

```
decide()
│
├─ 前置门控（policy_disabled / empty_signals / execution_in_progress / cooldown）
│
├─ Force Wake（优先级最高，不受信号陈旧影响）
│   条件：某 role awaken_count=0 AND router_backlog > 0
│   动作：强制 scale_up 该 role 的一个实例
│
├─ 全信号陈旧检查（all_signals_stale → 跳过）
│
└─ _make_stepwise_decision()
     │
     ├─ Priority -1：Trainer 空闲优先
     │   条件：trainer_busy=False AND pending_total>0 AND waiting_on ∈ {rollout, reward}
     │   动作：直接 scale_up 等待侧的实例，忽略后续所有优先级
     │   目的：让 trainer 不因为等待 rollout/reward 而空闲
     │
     ├─ Priority 1：单侧满载扩容
     │   子条件 P1a：仅 RM 满载（Rollout 未满）
     │     Step A：在空闲 GPU 上扩容 RM（_pick_scale_up_candidate_on_free_gpu）→ 无需 sleep
     │     Step B：找不到空闲 GPU 时，转让一个低负载 Rollout 实例（隐式 sleep 由 handler 执行）
     │   子条件 P1b：仅 Rollout 满载（RM 未满）
     │     Step A：在空闲 GPU 上扩容 Rollout
     │     Step B：转让一个低负载 RM 实例
     │
     ├─ Priority 2：双侧满载瓶颈优化
     │   条件：Rollout 和 RM 均满载
     │   计算：分别模拟"sleep Rollout 一个 + wake RM 一个"和"sleep RM 一个 + wake Rollout 一个"的增益
     │   评估指标：min(rollout_mu_after, rm_mu_after) - min(rollout_mu_before, rm_mu_before)
     │   选取：增益最大且 > hysteresis 的转让方向
     │   目的：最大化系统瓶颈吞吐量（木桶短板）
     │
     ├─ Priority 3：单侧自发缩容
     │   条件 P3a：Rollout 低负载（有实例 KV ≤ theta_low）AND RM 未满 AND 等待队列 ≤ 阈值
     │   条件 P3b：RM 低负载 AND Rollout 未满 AND 等待队列 ≤ 阈值
     │   动作：直接 scale_down 该 role 最空闲的实例，释放 GPU 供其他 role 使用
     │
     └─ 无动作（记录详细原因到 no_action 日志）
```

### 6.3 P1 转让的隐式 scale_down

P1b/P1a 转让场景中，策略只发出一个 `scale_up` action（不发出 `scale_down`）。
**真正的 sleep 操作发生在 `_scale_up_handler_loop` 中**：

```python
# scale_up_handler_loop:
instances_to_scaled_down = _find_instances_to_scaled_down_for_other_roles(task)
# 先 sleep 其他 role 的低 KV 实例
await asyncio.gather(*[_scale_down_instance(i) for i in instances_to_scaled_down])
# 再 wake 目标实例
instances_to_scaled_up = _find_instances_to_scaled_up(task)
await asyncio.gather(*[_scale_up_instance(i) for i in instances_to_scaled_up])
```

这是一个**决策与执行层面的语义分离**：策略层只表达"扩容目标 role"的意图，执行层负责判断是否需要先让位。

### 6.4 吞吐量估算公式

当存在拟合公式文件时，使用：
```
mu(x) = A * (1 - (B*x + 1)^(-k))
```
其中 x = running_queue_num，该公式是对生产环境中不同并发请求数下 vLLM 实测吞吐量的参数化拟合，能比实时测量更稳定地反映模型的吞吐能力曲线。

### 6.5 请求到达率 λ 估算

用简单的 EWMA 平滑请求到达率：
```
raw_λ = 当前总 μ + ΔQueue / Δt
λ_ewma = α * raw_λ + (1-α) * λ_ewma_prev
```
`total_mu + ΔQueue/Δt` 的直觉：稳态下 λ ≈ μ；如果队列在增长（ΔQueue > 0），说明 λ > μ，需要纳入这部分超出量。

---

## 7. 调用链分析

### 7.1 初始化调用链

```
TaskRunner.run()
  └─ PSRL_RayPPOTrainer.init_workers()
       └─ 创建 elastic_shared_pool（placement group）
       └─ _build_elastic_sub_resource_pool(rollout/RM 各实例)
       └─ _run_worker_group_tasks()（多线程并行）
       └─ init_rollout_coordinator()
       └─ init_elastic_rm_runtime()
            ├─ start_rollout_coordinator()（启动 coordinator busy loop）
            ├─ 收集各 WorkerGroup 的 gpu_slots（ClusterTopology.collect_gpu_slots_from_worker_group）
            ├─ 所有实例：rollout_coordinator.exec_command(SLEEP)
            ├─ ElasticExecutor.register_role(Rollout, rollout_all_ids, rollout_gpu_slots)
            ├─ 每个 RM：reward_model_coordinator.exec_command(SLEEP)
            ├─ ElasticExecutor.register_role(RewardModel, rm_all_ids, rm_gpu_slots)
            ├─ 选 RM 初始唤醒：ElasticExecutor.select_initial_awake_ids → WAKE_UP
            ├─ 选 Rollout 初始唤醒：ElasticExecutor.select_initial_awake_ids → WAKE_UP
            └─ ElasticExecutor.start_busy_loop()
                 ├─ asyncio.create_task(_monitor_loop)
                 ├─ asyncio.create_task(_scale_up_handler_loop)
                 └─ asyncio.create_task(_scale_down_handler_loop)
```

### 7.2 监控-决策-执行调用链

```
_monitor_loop() [每 monitor_interval_ms 一次]
  ├─ _sync_engine_status_from_coordinators()
  │    └─ coordinator.get_instance_engine_status_snapshot.remote() [并行，每 ref 独立超时]
  ├─ _sync_router_backlog_from_coordinators()
  │    └─ coordinator.get_router_backlog_size.remote() [并行，每 ref 独立超时]
  ├─ _sync_trainer_waiting_hint()
  │    └─ agent_loop_manager.get_trainer_waiting_hint.remote()
  ├─ _build_instance_signals()
  │    └─ 遍历 instances_status_flags + instances_engine_stats + topology → List[InstanceSignal]
  └─ ScalingPolicy.decide(signals, ...)
       ├─ 门控检查（disabled / empty / in_progress / cooldown）
       ├─ Force Wake 检查（backlog > 0 AND awaken = 0）
       ├─ 全信号陈旧检查
       ├─ _group_by_role() / _build_mu_maps() / _estimate_lambda()
       └─ _make_stepwise_decision()
            ├─ P-1: trainer_idle 分支
            ├─ P1: 单侧满载分支（_pick_scale_up_candidate_on_free_gpu / _pick_scale_down_candidate）
            ├─ P2: 双侧满载分支（_estimate_role_total_mu_with_rebalance + 增益比较）
            └─ P3: 单侧自发缩容分支

       → ScalingDecision.actions → scale_up_task_queue 或 scale_down_task_queue

_scale_up_handler_loop() [从 scale_up_task_queue 取任务]
  ├─ _find_instances_to_scaled_down_for_other_roles() [找出需前置 sleep 的其他 role 实例]
  ├─ asyncio.gather(_scale_down_instance × N) [并行 sleep]
  │    └─ _await_elastic_coordinator_command(SLEEP) → coordinator.exec_command.remote
  ├─ _find_instances_to_scaled_up() [找出目标实例，GPU 冲突过滤]
  ├─ asyncio.gather(_scale_up_instance × M) [并行 wake]
  │    └─ _await_elastic_coordinator_command(WAKE_UP) → coordinator.exec_command.remote
  ├─ _interrupt_waiting_after_scale_up()
  │    └─ _await_elastic_coordinator_command(ABORT) → coordinator.exec_command.remote
  └─ _mark_decision_action_finished(decision_id)

_scale_down_handler_loop() [从 scale_down_task_queue 取任务]
  ├─ _find_instances_to_scaled_down_in_role() [按 KV 排序选出目标实例]
  ├─ asyncio.gather(_scale_down_instance × N) [并行 sleep]
  │    └─ _await_elastic_coordinator_command(SLEEP) → coordinator.exec_command.remote
  └─ _mark_decision_action_finished(decision_id)
```

---

## 8. 关键设计决策

### 8.1 决策与执行分离（ScalingPolicy vs ElasticExecutor）

ScalingPolicy 是一个**纯计算对象**（无 Ray、无 I/O、无副作用），只接受 signal 列表，返回 decision 对象。这使得策略可以被独立单元测试，也可以被替换为不同的决策算法而无需修改 Executor。

### 8.2 三循环异步架构

Monitor、scale_up_handler、scale_down_handler 三条任务通过 `asyncio.Queue` 通信，使得：
- 监控不会因等待 RPC 完成而积压（独立循环）
- 扩容和缩容操作不会互相阻塞

### 8.3 每 RPC 独立超时

`_await_coordinator_refs_with_per_ref_timeout` 为每个 Coordinator RPC 独立设置超时，而不是对整批 RPC 共享一个超时。这确保一个慢/卡死的 Coordinator 不会影响其他 Coordinator 的状态同步，避免全量数据丢失。

### 8.4 Force Wake 绕过信号陈旧检查

Force Wake 逻辑故意放在 `all_signals_stale` 检查之前：休眠实例没有引擎在运行，其快照天然是陈旧的，若让陈旧检查先运行，Force Wake 永远无法触发，会导致某个 role 长时间零唤醒、router backlog 积压。

### 8.5 ClusterTopology 双向索引

将 GPU → 实例的反向索引集中维护在 ClusterTopology 中，而非分散在各函数内手动计算，避免了多份不同步的状态。

### 8.6 post-scale-up ABORT

唤醒新实例后，用 ABORT 中断排队中的请求，让它们重新进入路由器。由于路由器此时感知到新实例可用，会将请求分发给新实例，实现扩容后的快速负载均衡，而非等待已有实例自然完成旧请求。

### 8.7 SubRayResourcePool 交错分配

`_build_elastic_sub_resource_pool` 中，`start_bundle_index` 采用 `(group_idx * subgroup_world_size) % (max_start + 1)` 计算，实现循环错位：不同实例的 placement group bundle 起始位置不重叠（当 world_size 整除时），减少了多个实例竞争同一组 GPU 的概率。

---

## 9. 已知问题与边界条件

### 9.1 Sleep Level 不对称导致的潜在 OOM

| Role | Sleep Level | 权重卸载 | KV 缓存卸载 |
|------|-------------|---------|------------|
| Rollout | 2 | ✅ 卸载到 CPU | ✅ 卸载到 CPU |
| Reward Model | 1 | ❌ 留在 GPU | ✅ 释放 |

当 RM 进入 SLEEP 后，其权重仍占据 GPU 显存。若此时在同一 GPU 上唤醒 Rollout，两份权重 + Rollout 的 KV 缓存可能超出 GPU 显存预算。代码中没有内存预算检查，GPU 冲突检测只基于 AWAKEN 状态，无法感知 SLEEP 后仍驻留的 level-1 权重。

### 9.2 P1 转让语义分散

P1 的 `scale_up` action 在 handler 中会触发隐式的前置 `scale_down`，但策略的 action 列表里只有 `scale_up`。这使得：决策时 `rollout_down is not None`（KV ≤ theta_low）被用作转让可行性判断，但 handler 执行时按实时 KV 重新选实例。若两次调用之间状态发生变化，可能出现多 wake 了一个实例但没有对应 sleep 的情况。

### 9.3 P3 等待队列检查粒度

`max_waiting_queue_for_scale_down` 参数注释说明是"per-role total waiting queues"，但实际代码仅检查 `rollout_down.waiting_queue_num`（单个候选实例的等待队列），不是该 role 所有唤醒实例的等待队列总和，语义与注释不符。

### 9.4 `_scale_down_instance` 日志格式错误

```python
psrl_logger.info(
    "Skip scale down for role=%s model=%s instance=%d: ...",
    instance_role, instance_model_name, instance_id,  # ← instance_id 是 tuple，%d 会抛 TypeError
    min_awake_per_role,
)
```

`instance_id` 类型为 `RolloutInstanceId = tuple[str, int]`，使用 `%d` 格式化会在运行时抛出 `TypeError`。应改为 `%s`。

### 9.5 角色名称硬编码

`_make_stepwise_decision` 中硬编码了：
```python
rollout_role = "Rollout"
rm_role = "RewardModel"
```
无法扩展到三个或更多角色（如同时存在多个不同大小的 RewardModel）。

### 9.6 无批量调度

每次决策最多只产生一个 scale_up 或 scale_down action（通过提前 return 实现）。若需要同时唤醒多个实例（例如 trainer 完全空闲且有多个空闲 GPU），当前策略无法在一次决策中表达，需要多个决策周期才能完成。

---

## 附录：关键配置参数汇总

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `enable` / `enable_policy` | False | 是否启用弹性调度 |
| `monitor_interval_ms` | 1000 | 监控循环间隔（毫秒） |
| `theta_low` | 0.3 | KV 低水位线 |
| `theta_max` | 0.85 | KV 高水位线 |
| `cooldown_ms` | 3000 | 决策冷却期（毫秒） |
| `hysteresis` | 0.05 | P2 转让最小增益 |
| `min_awake_per_role` | 0 | 每 role 最少唤醒实例数 |
| `full_load_mode` | `"any"` | 满载判断模式（any/all） |
| `max_waiting_queue_for_scale_down` | 0 | P3 缩容的等待队列上限 |
| `post_scale_up_abort_waiting_ratio` | 1.0 | 扩容后 ABORT 等待请求的比例 |
| `coordinator_sync_timeout_s` | 60.0 | 状态同步 RPC 超时（秒） |
| `coordinator_command_timeout_s` | None | SLEEP/WAKE_UP/ABORT 命令超时 |
| `decision_execution_abandon_stall_ticks` | 0 | 决策卡死放弃阈值（0=禁用） |
| `monitor_instance_log_interval_ms` | 5000 | 实例状态日志间隔（毫秒） |
| `throughput_model_dir` | None | 吞吐量公式文件目录 |
| `throughput_model_output_len` | 1024 | 吞吐量模型的目标输出长度 |
| `lambda_ewma_alpha` | 0.2 | λ EWMA 平滑系数 |
| `min_awake_per_role`（init） | 0 | 初始化时每 role 的最小唤醒数 |

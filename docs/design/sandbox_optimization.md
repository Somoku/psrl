# Sandbox 分阶段优化设计

本文面向 PSRL 的 p3b Docker backend，定义资源管理、弹性分配、环境复用和按需启动的工程路线。状态为设计提案，文中的新增接口、控制器、指标和实验工具均需实现。阅读前置为 {doc}`sandbox_service`、{doc}`sandbox_architecture` 和 {doc}`sandbox_snapshot_store`。

优化的目标是提高成功 rollout 与完整 GRPO group 的产出率，降低每个有效样本的资源成本，约束是隔离、奖励非劣与基础设施失败率不上升。容器数量、准入数量和首个进程启动时间只作为解释指标，不能单独证明训练效率提高。

## 1. 目标与场景先验

p3b 服务于长程 Agentic RL。这个场景的成本结构与短 FaaS 不同，优化必须先对齐它，否则会得到一份在错误维度上密度更高的系统。

| 场景特征 | 对资源管理的含义 |
| --- | --- |
| bursty creation | 容器创建集中在 step 边界和 group 内同步工具调用，创建延迟随并发劣化 |
| agent 状态长期维护 | 单条 episode 是分钟到半小时量级，sandbox 在两次工具调用之间长期驻留 |
| LLM call 时空闲 | CPU 在等待模型时几乎为零，内存在等待期间只被占用、不被使用 |
| GRPO group 内 prompt 与 sandbox 相同 | 同组请求镜像、规格、准备内容一致，是天然的高命中缓存与合并机会 |
| grading 与 rollout 关联 | grader 与 rollout 共享部分内容，但必须保持独立环境与评分语义 |

### 1.1 首要成本

**首要成本是 LLM call 间隙的驻留空转，不是冷启动。** 一个 agent sandbox 在整条轨迹里大部分时间在等模型返回，期间它占用内存、不消耗 CPU，也不推进任务。冷启动是每条 episode 一次性成本，对整个 episode 的 wall time 占比小。因此本设计把「让空闲 agent 真正让出内存」放在冷启动优化之前。

冷启动仍然是 step 边界并发爬升的限制项，但它的收益杠杆按代价排序是：拉取合并、按完整规格的预热池、真实的镜像库存与局部性放置，最后才是按需加载。

### 1.2 首要指标

主指标为成功 rollout/s、完整可训练 group/s、端到端训练 step 时间和每个有效样本的资源成本。所有提交任务、重试和基础设施失败保留在分母和日志中，单独报告失败率与任务分布偏移。

在这个场景下必须额外记录两类量，否则无法判断内存弹性与复用是否真的有效：

- **LLM 间隙驻留 byte-seconds 与 resident-idle 占比。** 只有它能说明「让空闲 agent 让出内存」的机制是否生效。共享内存节省使用节点或资源域物理观测，不能累加各进程 RSS。
- **冷启动分项时延。** 排队、准备、启动、首个有效工具调用分开记录，避免把启动等待转移到首次工具调用却被读成改善。

其余时延分项为执行、grading 和释放，包含足够样本的 p95/p99。同时报告 GPU 等待环境时间、策略版本滞后、CPU 秒、物理内存 byte-seconds、磁盘空间峰值、读写字节、copy-up 和源站网络字节。

### 1.3 不变量

以下约束贯穿所有阶段：

- 节点是本机物理资源准入的最终执行者，控制面负责公平性、配额和放置。Python SDK 不再维护另一套节点容量账本。
- 分配终止时，只有确认资源已销毁，才能归还剩余物理预算。租约失效意味着进入回收流程，不意味着容器已经消失。
- 共享不可变基础内容。每条轨迹保留独立的可写文件系统、进程、执行身份、随机状态和结果目录。
- grader 从角色匹配的干净基线创建独立环境，只应用被评估的 patch，不得继承 rollout 的可写层或隐藏状态。
- 资源压力、镜像读取故障和执行超时必须有独立的基础设施失败分类，不能静默变成策略负奖励或删掉困难样本。
- 预热、准备、预取、缓存、压缩、清理和控制器自身均纳入节点成本。
- 节点保留执行与数据平面，控制面不代理命令输出、文件内容或镜像块。

## 2. 与既有架构决策的关系

本设计的范围与既有架构决策相关，其中 D2 与 S8 已为支持本节结论修订。下面记录修订内容，以及与 group 语义决策的划界。

| 决策 | 内容 | 与本设计的关系 |
| --- | --- | --- |
| `sandbox_refactor` D2 | 内部 Docker backend 是默认部署 backend 并保持契约完整，需要无外部服务，但不追求 provider 级密度 | 已修订：P4 的能力门控 snapshotter 属于允许范围 |
| `sandbox_snapshot_store` S8 | DSec 式内核级按需效率属于外部 provider，不属于内部 Docker backend | 已按此划界：内核级路径仍排除，能力门控的节点内 snapshotter 单列 |
| `sandbox_refactor` D15 | group 逐成员准入，任何成员不得在等待兄弟时占容量，原子组预留被拒绝 | P3 的 group 局部性只能是软放置偏好，不得引入组级容量预留 |
| `sandbox_refactor` D16 | group 是 N 个只差身份的规格 | 支持 P3 的三类身份拆分，身份与制品键分离 |

两件与决策相关的事实：

- **部署里实际运行的只有 Docker backend。** `p3b/deploy.json`、`deploy_direct.json`、`deploy_node_250.json`、`deploy_control.json`、`deploy_node.json` 以及各 `example*.json` 全部声明 `type: docker`，没有任何 provider 配置。这是修订 D2 的依据。
- **按需加载对同类容器路径不可用。** `image_on_demand` 与 `image_block_delivery` 只在各 backend 的 `Capabilities()` 里出现，仅被 capability 匹配消费。OpenSandbox 的核实结论见 8.1：Docker runtime 完整拉取镜像，按需加载只存在于 fast-sandbox 路径。

**范围决定（已修订）。** D2 与 S8 已按下述结论修订：Docker backend 是默认部署 backend，同时保持契约完整，需要无外部服务，但不追求 provider 级密度。允许一个**能力门控的节点内 snapshotter** 作为可选项，理由是它属于节点部署成分而非外部服务，且是唯一能在不依赖外部平台的前提下端到端验证 lazy 的地方。DSec 式内核级页共享与 microVM 路径仍排除在 Docker backend 之外。

## 3. 当前实现与缺口对照

以下路径用于定位改造入口。结论来自源码检查，部署拓扑相关风险仍需在目标 Linux 节点验证。标注为**已修**的行说明其描述的是更早的 revision，不应再作为实现依据。

| 入口 | 当前行为 | 设计要求 |
| --- | --- | --- |
| `p3b/internal/node/lifecycle.go` | 显式 Release 已按先销毁再归还处理（`Release` 在 `backend.Release` 失败时保留 grant） | 保留该顺序，消除旁路 |
| `p3b/internal/node/lifecycle.go` | **已修**：独立 TTL 路径已删除，`Sweep` 不再无条件 `admission.Expire()`，预算只在销毁确认后归还 | 已收敛到同一状态机 |
| `p3b/internal/node/lifecycle.go` | idle pause 与 idle reap 已移除，停滞判定只余 owner 存活与 lifetime。pause 只走控制面，无 generation 协调 | 生命周期动作按 generation 串行确认 |
| `p3b/internal/node/admission.go` | **已修**：`Renew` 经 `control.RenewOwner` 到 `Node.RenewOwner`。`SetDraining` 现由 `cleanup_failed` 置位与清除，有生产调用链 | 保留 |
| `p3b/internal/node/admission.go` | owner 存活检查**已修**：`main.go` 在生产路径安装 `gate.OwnerAlive` | 保留，补齐死亡检测与重启对账 |
| `p3b/internal/pressure/` | 读的是外层部署 cgroup 而非 sandbox 子树（5.2 有实测）。读失败在 `Read()` 里静默返回零，而 `effectiveEnvelopeLocked` 把零当「未超售」 | 采样域与 sandbox 子域对齐，失效值不得当作空闲 |
| `p3b/internal/server/nodeserver.go` | `view()` 重复采样（两次 `pressure()`），`ImageDigests`/`ImageReferences` 写成空 map，`localityScore` 永远为 0 | 单一采样器发布带时间戳快照，补齐镜像与准备制品库存 |
| `p3b/internal/backend/dockerbackend/docker.go` | 创建设置静态内存和 CPU 限额，未设置 `CgroupParent`。镜像准备发生在 create 并发控制之前 | 补齐资源域，分开限制拉取、创建、执行和清理 |
| `p3b/internal/backend/dockerbackend/docker.go` | `Backend.Sweep` **已修**：改为可选接口 `backend.Reconciler`，节点启动时按标签重建账本。仍缺拉取请求合并，`callRaw` 仍整块缓冲响应 | 请求合并与有界输出仍待做 |
| `p3b/internal/backend/dockerbackend/warmpool.go` | 预热匹配只比 image、memory、cpu，不含 env、workdir、runtime、network。接管不更新实际限额。补充仅受数量约束，销毁失败被吞 | 规格正确、实际资源可核对、压力感知的预热池 |
| `p3b/internal/placement/` | 已有镜像局部性评分，但库存输入为空。控制面重试不排除已拒绝节点 | 可解释的库存评分、拒绝节点排除和退避 |
| `p3b/internal/server/convert.go` 与 `p3b/internal/quota/` | fleet 与节点预算均由同一静态规格转换，等待依赖轮询 | 拆分租户公平性与物理准入，支持有界事件唤醒 |
| Python p3b adapter 与 SDK | **已修**：`_to_service_spec` 现转发 workdir、exec_mode、required_features、required_resume_level 与 backend pin，并把全部 PSRL 字段登记为转发、拒绝或明确不转发（新增字段会被用例挡住）。拒绝集不变 | 保留：新增字段必须在此边界做决定，禁止静默忽略 |
| `p3b/sdk/sandboxd/` | 连接池默认大小 32，池槽等待在请求超时之外 | 总截止时间覆盖连接池排队 |
| 所有 backend 的 `image_on_demand` | 能力字符串，仅用于路由匹配。OpenSandbox 已核实：Docker runtime 完整拉取，无按需路径 | opensandbox backend 不声明该能力，其交付为块级。AgentEnv 与 CubeSandbox 待核实后再定 |

现有 overcommit 主要扩展基于观测值的内存准入包络，不构成 CPU 与内存联合弹性控制器。CPU 请求总和仍可先成为瓶颈。不得把一个更大的 overcommit 系数当成 P2。

Docker pause 保留内存，Docker commit 主要保存文件系统。两者均不能直接提供进程匿名内存的跨容器共享或完整进程状态恢复。

## 4. 阶段划分与顺序

阶段按「先正确、再让出内存、再复用、最后按需读取」排序。P2 的 CPU 弹性降为可选项，P4 拆为可早做与延后两段。

| 阶段 | 解决的问题 | 交付边界 | 后续依赖 |
| --- | --- | --- | --- |
| P1 资源管理完整性 | 容器与账本失配、观测缺口、预热与清理失控 | 静态策略下可靠的生命周期、预算和观测 | 所有优化的正确性基线 |
| P2 内存弹性 | 最高限额预留造成闲置，空闲 agent 内存不回流 | 单个内存控制器，CPU 弹性可选 | P1 的可信观测和原子记账 |
| P3 组内准备复用 | 同组、跨请求、rollout 与 grader 重复准备 | 干净冻结点与共享基础内容 | P1 的制品引用与预算 |
| P4 就绪语义与按需读取 | 首个工具可用等待、块读取放大 | 就绪语义与合并，能力门控的按需层 | P3 的内容标识，P1 的故障语义 |

每阶段分别以 B1 静态基线、内存弹性消融、复用子机制消融、就绪语义与 eager/lazy 消融验收。

## 5. P1：生命周期、观测与并发

### 5.1 分配事务与回收闭环

统一状态为 reserved → creating → resident → destroying → destroyed。结果未知的操作进入 reconciling，销毁失败保留占用并进入 cleanup_failed。TTL、取消、显式 Release 和 orphan sweep 都驱动同一状态机。

具体到当前代码，要消除的是 `lifecycle.Start` 里 `Sweep` 与 `admission.Expire()` 之间的旁路：销毁失败时容器仍占用内存，而 `Expire` 已把它按 owner 静默释放。收敛方式是让 budget 归还只发生在销毁确认之后，而不是由独立的 owner 静默计时器触发。

grant 必须绑定 owner、generation 和 spec，并且只能消费一次。创建请求响应丢失时，通过 allocation 标签和运行时查询恢复结果，不能直接重复创建。取消创建后使用独立、有限时长的清理 context，避免用户请求的取消同时取消清理。

采用成熟事务存储保存 allocation journal，提交资源预留后再操作 Docker，通过幂等状态转换处理非原子的外部副作用。启动时先对账日志与带 p3b 标签的实际容器，再恢复准入。当前 `Backend.Sweep` 是死代码，重启后没有任何路径把存量容器重建到账本，这是必须补的第一块。

owner heartbeat、租约和节点 drain 接入真实调用链。epoch 用于拒绝旧 owner 的请求，节点 agent 还需本机单执行者机制与 Docker 访问权控制。仅增加 epoch 字段不能阻止旧进程继续操作 Docker socket。`SetDraining` 目前无生产调用链，需补齐由 `cleanup_failed` 触发的 drain 路径。

清理采用有界并发和退避，保留独立于创建洪峰的执行容量。每次删除必须确认运行时对象已不存在，失败记录不能从账本和预热池中直接遗忘。

### 5.2 可解释的资源域

为 sandbox 建立明确的 cgroup 子树，Docker 创建时设置对应 `CgroupParent`。节点服务放在独立叶节点，避免 cgroup v2 的内部进程约束。节点预算同时受 sandbox 子树、祖先限制、cpuset 与宿主机安全余量约束。

只运行一个采样器。CPU 分母使用实际有效容量，内存观测区分 sandbox 子树与宿主机。采样错误、过期和不可用必须显式表示，不能把零当成空闲。观测不可用时停止弹性扩容，保留可解释的保守准入模式。

已在两个部署节点上按实际 Docker 与 cgroup 配置实测（`go test -tags integration ./internal/pressure/`，用例让容器申请固定内存并要求该域读数同步上升）。结论与源码推断不同：cgroup v1 部署上进程自身的 cgroup 路径在挂载下不存在，采样器回退到挂载根，读到的是**外层部署 cgroup**，而**容器确实在该域内**。实测容器自身计费 4.31 GB 时该域读数上升 4.29 GB，因此「容器不在该子树下」在本部署不成立。

真正的偏差是域过粗。该域 usage 1.14 TB、limit 2.36 TB，其中 `docker` 子树（全部容器）仅约 106 GB，其余是该部署内与 sandbox 无关的内存。`LocalMemCeiling` 与 `UtilizationTarget` 因而表示的是部署整体比例，分母也是外层 cgroup 的 limit 而不是节点声明的包络：一个因无关原因接近 85% 的部署会让节点开始拒绝 sandbox。

需要的是把 sandbox 子树与采样域对齐，即由 `CgroupParent` 建立子树并让采样器指向它，使这两个阈值作用于 sandbox 域。读数失效的显式表示仍待补，本部署未触发它，因为回退总能读到可读文件。

### 5.3 避免重复工作与控制面拥塞

| 工作项 | 实现要求 |
| --- | --- |
| 镜像请求合并 | 按平台与解析后 digest 合并在途拉取。解析、拉取和 create 各有有界并发。单个等待者取消不取消其他等待者需要的工作 |
| 预热池 | 使用完整兼容规格匹配，包含 image、env、workdir、runtime、网络及能力。实际限额需符合接管规格，否则销毁重建 |
| 预热预算 | 预热容器、补充任务和销毁失败均记账。压力升高时先停止补充和回收闲置预热，不回收活跃轨迹 |
| 库存与放置 | 发布真实镜像 digest 与准备制品库存，节点拒绝带原因和观测版本，单次重试排除已拒绝节点 |
| 输出和文件 | 流式读取或限定缓冲。截断必须显式返回，不能把不完整测试日志当成完整结果 |
| 取消 | 区分排队取消、未启动取消、运行中终止和结果未知。清理执行进程树，不自动重试结果未知的非幂等命令 |
| pause/resume | 按 generation 核验动作，统一 paused 状态。新执行与暂停决策串行化 |
| SDK 规格 | 补齐 workdir、exec mode 和能力传递，无法支持的字段明确拒绝。总截止时间覆盖连接池排队 |

控制面的等待策略不在本设计范围内，不要改动。class 满时是队列而不是判决：份额随 sandbox 释放而清空，请求要的是**将会出现**的容量，因此服务端不设等待上限，也不主动拒绝，等待只由调用方自己的截止时间结束。唯一的即时拒绝是请求大于该 class 的全部分额，那说明任何释放都无法让它被满足，属于请求本身的性质而非负载判断，这条保留。调用方的截止时间必须覆盖连接池排队。

这里的预热是创建尚未执行用户任务的干净容器，不能扩展为复用执行过任意任务的脏容器。

### 5.4 测试、消融与出口

| 测试 | 必须成立的断言 |
| --- | --- |
| 并发创建、取消与丢响应 | 同一分配最多一个活跃运行时对象，未知结果能够被对账识别 |
| TTL 到期与删除失败 | 进入回收但不提前释放预算，重试成功后只归还一次 |
| owner/agent 重启 | 存量容器被完整重建到账本，旧 epoch 请求不能修改新分配 |
| pause 与 exec 竞争 | 有效执行不被陈旧暂停动作冻结，Resume 后状态一致 |
| 预热规格与 refill 洪峰 | 不错误复用 env/workdir/限额，闲置池和失败清理不超预算 |
| cgroup 拓扑与读数失效 | 容器占用计入目标域（现有 integration 用例覆盖这一半），过期读数不触发扩容 |
| 大输出和文件 | 内存有界，完整性或截断状态明确，取消后无遗留进程 |

P1 内部分开测量请求合并、预热策略与库存放置。正确性修复作为统一基线，不提供生产开关重新启用账本泄漏等错误行为。

## 6. P2：内存弹性

agent sandbox 的内存压力来自空闲 agent 的驻留，而 CPU 在等待模型时近乎为零。因此本阶段以内存为唯一核心控制器，CPU 弹性作为可选项而不是交付物。

### 6.1 拆分保证量、上限与需求

硬上限是保护边界，不是每个驻留 sandbox 永久占据的准入预留。内存保证量代表可兑现的承诺，上限代表允许申请的最大范围，软目标由控制器调节。所有保证量必须有物理预算支持，不能同时承诺又超售。

`memory.high` 主要触发回收与节流，`memory.max` 是硬保护，两者不可互换。降低限额不会立即创造等量空闲，主动回收也可能不足或超过请求量。控制器必须根据实际释放确认容量。[cgroup v2 文档](https://docs.kernel.org/admin-guide/cgroup-v2.html)

### 6.2 准入与控制闭环

以内存物理预算 `B` 为界，准入检查采用以下保守结构：

```text
observed_usage + pending_growth + running_headroom
    + candidate_headroom + safety_reserve <= B
```

`pending_growth` 表示已授权但观测尚未吸收的增长，`running_headroom` 覆盖运行中任务的保证缺口与预测突发，`candidate_headroom` 包含新请求启动及未来增长。三者必须有无重叠定义，不能把已反映在 observed_usage 中的增长再次计入。预留和比较在节点内原子完成。

不能假设同组轨迹峰值独立。编译与 grader 可同步启动，需按 group 与阶段估计相关突发余量，并以保守包络和真实失败率校准预测。共享文件页的节省在观测证实前不作为确定的空闲容量。

控制流程为观测 → 估计需求 → 原子申请增长预算 → 更新 runtime → 读回确认 → 发布结果。更新失败时仅撤销尚未实际生效的预算，结果未知先对账。缩容需等待实际用量下降后归还，OOM 已终止的执行不能靠事后扩容恢复。

内存实现快速增长、缓慢收缩、滞回与最短驻留时间。使用 PSI、内存事件及执行阶段提示作为输入，不能只根据 RPC 是否在途判断忙闲，容器内可能有后台进程。针对长程 agent 的关键点是把「等待模型返回」识别为可回收窗口：此时进程树处于阻塞，工作集可被压到低水位，恢复时以 refault 成本衡量是否值得。

节点压力升高时依次停止预热和预取、回收可回收缓存、停止新准入，再按既定公平策略限制非关键突发。不得无条件收缩活跃任务到其真实工作集以下。

fleet quota 管理逻辑公平性与租户上限，节点管理物理可行性。若控制面仍按每个请求最大限额累加，即使节点支持弹性也无法提高准入。两层的资源语义必须一起更新，且只允许一个控制器写同一 cgroup 参数。

为完成中的 group 和 grading 保留受预算约束的服务能力，防止所有容量被新 rollout 占满。现有 mini-SWE 路径会先提取 patch 并释放 rollout，再启动 grading，应保留这一顺序。跨请求的 rollout 与 grading 重叠仍需计入联合峰值。

WeEnv 的快速增长与缓慢回收、VenusRL 的实际用量及增长余量准入为此提供思路。它们的默认阈值、后端和实验规模不作为 PSRL 配置值。[WeEnv](https://arxiv.org/html/2609.30766v1)、[VenusRL](https://arxiv.org/html/2610.03286v1)

### 6.3 CPU 弹性：为什么降级为可选项

CPU 在 agent 场景稀疏且在 LLM call 时完全空闲，而 Linux 调度器已把 CPU 当作可压缩资源处理。为一个稀疏、可压缩的维度引入权重、突发和滞回控制器，收益低，却新增节流尾延迟风险。`admission.go` 的既有注释也承认 CPU 已被调度器有效超售。

因此 CPU 弹性不作为 P2 交付物。若某类负载实测显示 CPU 请求总和先于内存成为瓶颈，再以独立实验评估 CPU 权重与突发上限，并把节流后的 p99 工具时延纳入出口条件。

### 6.4 测试、消融与出口

- 状态机测试覆盖并发扩容、更新超时、部分生效、旧观测、预测误差、收缩延迟和预算只归还一次。
- Linux 集成测试覆盖硬上限、memory.high 压力、OOM 事件、匿名页与文件页不同回收行为。
- 回放覆盖同组同步内存突发、长推理后同时恢复、编译与 grader 争用、观测故障和训练进程共置。
- 消融为静态 P1、内存弹性开关、以及可选的 CPU 弹性开关，对控制周期、滞回和安全余量做敏感性分析。
- 出口要求吞吐或成本达到预设收益，同时奖励非劣、有效 group 比例不下降、尾延迟和 OOM 不越界，并且控制过程无持续振荡。首要验收量是 LLM 间隙驻留 byte-seconds 的下降。

## 7. P3：面向 GRPO 与 grading 的组内准备复用

这是本场景独有的、价值最高的一节。同组请求镜像、规格、准备内容一致，而当前每条 attempt 独立准备，冗余最大。

### 7.1 分离三类身份

PSRL 在 `psrl/utils/dataset/data_processor.py` 根据 parent 扩展 rollout，`psrl/workers/agent_loop/manager.py` 的 `get_dispatch_plan` 把同组分配给同一 Python worker。这不保证请求落到同一 p3b 节点，需显式传播 group 提示。

| 身份 | 建议组成 | 用途 |
| --- | --- | --- |
| 调度身份 | run、train/validation、parent、group attempt | group 放置、完成优先级与公平性 |
| 执行身份 | run namespace、child uid、attempt generation、rollout/grading phase | 所有权、幂等、清理和故障追踪 |
| 制品身份 | dataset problem/version/input、base digest、recipe/env/input digest、平台、基线角色、schema、信任域 | 跨请求复用不可变内容 |

重试可能沿用 uid，因此 uid 不能单独作为执行代次。结果提交端必须校验当前 attempt，旧 grader 的延迟结果不能写入新 attempt 的 group，旧清理操作不能释放新 attempt 的引用。`mini_swe_env.py` 中用于防清理冲突的运行时 task UUID 也不能替代稳定制品键。反过来，不能为提高缓存命中而删除执行身份中的随机或代次成分。

### 7.2 共享层级

| 层级 | 共享方式 | 必须私有的内容 | 实现判断 |
| --- | --- | --- | --- |
| 镜像只读层 | 按 digest 复用 Docker 存储与宿主机页缓存 | 每容器 upperdir | 已有基础，先测量实际命中 |
| 依赖与工具 | 内容寻址的只读组件或准备镜像 | 安装临时目录与任务输出 | 需制品 manifest 和发布流程 |
| 任务准备结果 | recipe 与输入完全匹配时共享干净快照 | agent 修改、session、凭据和随机状态 | P3 核心 |
| GRPO group | 同一准备制品创建多个私有分支，软局部性放置 | 轨迹进程、可写状态和奖励结果 | 不复用脏容器 |
| rollout 与 grader | 共享经验证的共同基础层或兼容准备组件 | 角色专属准备和 grader 工作目录 | 需要显式基线谱系 |
| 匿名内存 | 运行时 fork、快照或显式页共享 | 每实例写时复制和运行状态 | 不属于普通 Docker 镜像复用，列为未来方向 |

OverlayFS 已允许相同文件共享页缓存，首次写入则可能发生 copy-up。P3 必须区分已有共享、减少重复准备带来的收益和新增复用机制，不能把已有 page cache 作为新贡献。[Docker OverlayFS 文档](https://docs.docker.com/engine/storage/drivers/overlayfs-driver/)

group 局部性是软约束，节点压力或排队成本过高时允许拆组。这与 `sandbox_refactor` D15 一致：不得引入组级容量预留，成员仍逐个准入。

### 7.3 组内一次准备与干净冻结点

**这是首先实现的子机制。** 共享冻结点必须在注入 session URL、凭据、attempt 信息及 agent 写入之前。当前该边界已存在但未接入：`harness_agent_loop` 里逐 attempt 调用 `prepare`，而 p3b 路径的 `prepare` 是 no-op，session URL 注入与 `harness.prepare` 都在 acquire 之后逐 attempt 执行。

生产接入需改造 `psrl/workers/agent_loop/loops/harness_agent_loop.py` 的普通 acquire 路径，将 `mini_swe_harness_agent_loop.py` 中的准备拆成可共享构建与每 attempt 初始化。共享冻结点位于 session URL 注入和 `harness.prepare` 之前，角色专属构建计划仍需分别验证。

准备过程若依赖时间、网络可变输入或未记录的环境变量，必须固定输入或声明不可缓存。manifest 记录基线来源和验证结果，而非仅记录一个镜像名。

### 7.4 制品管理与引用（延后）

控制面 artifact catalog 按完整制品键分配唯一的有效准备租约，节点 artifact manager 执行构建、验证、上传及本地请求合并。租约 generation 防止失联生产者迟到发布，目录区分可跨节点获取的已发布制品与仅本地可用制品。协调仅传递 manifest 和状态，数据通过制品存储直接流转。

这一层长期正确，但相对当前失败率偏重。现有 `TaskSnapshotCache` 的本地 JSON 与 `acquire_prepared` 提供了接口参考，但 `acquire_prepared` 当前只有测试调用，本地缓存也不构成多 worker 的分布式协调器。在 7.3 落地并测出收益之前，不引入完整的全局目录与 GC。准备租约、发布、引用和 GC 到时统一归属 p3b 的制品管理，并移除被替代的独立权威路径。

活动 allocation 持有制品 pin，节点重启重建引用。GC 仅删除无活动引用且超过策略预算的制品。容量管理同时考虑镜像、准备结果、文件块、构建临时空间和保留时间。不要以无限缓存换取表面吞吐。

`examples/mini_swe/runner.py` 的 baked 引用不能只由镜像引用字符串决定，应包括解析后的 digest 与准备 recipe。rollout 可能使用清理过的派生镜像，grader 则使用独立的问题镜像，不能默认最终镜像相同。

尤其 SWE-smith grader 可能依赖 git 历史，而 rollout 清理可能去掉 git 元数据。`FILESYSTEM` 类型只能描述恢复形式，不能证明适合某个 grader。必须验证角色、基准 commit、目录布局和准备谱系，再决定共享哪部分。

### 7.5 测试、消融与出口

- 同组一个轨迹修改文件、权限、环境变量或后台进程，其他轨迹与 grader 的初始可观察状态不变。
- 使用已知正确及错误 patch 检验奖励一致性，覆盖依赖 git 历史的 grader，防止错误基线改变评分。
- 验证 tag 更新、recipe/env/input 变化、平台和信任域变化均正确失效。资源 quota 不影响准备内容时不纳入制品键。
- 验证准备生产者死亡、部分发布、所有等待者取消、节点重启、引用重建和 GC 与恢复竞争。
- 验证取消后重试、旧 grader 延迟返回和旧清理请求，不得污染新 attempt 的结果或引用。
- 分别启用准备合并、本地跳表缓存、group 局部性、grader 基础层复用。独立统计准备次数、物理文件页、匿名页和 copy-up。
- 做 P2 开关与 P3 开关的二因素实验，识别共享降低实际用量后带来的额外准入收益，避免重复计算两阶段贡献。

出口要求隔离与评分测试全部通过，在相同缓存预算下减少重复准备或物理成本，并提升有效 group 产出。随机状态和轨迹独立性属于验收项，不能以更相似的轨迹换取吞吐。

## 8. P4：就绪语义与按需读取

### 8.1 与 provider 的关系

已按下列范围完成验证，结论是 OpenSandbox 现有的按需加载能力不覆盖与 p3b Docker backend 同类的容器路径，因此 P4 不是重复工作。验证对象为 vendored 的 OpenSandbox 与本集群部署的 OpenSandbox 服务。使用 microVM 的 AgentEnv 的 overlaybd 与 ublk 路径不在比较范围，因为它与 Docker 路径不可移植。

| 路径 | 按需加载 | 依据 |
| --- | --- | --- |
| OpenSandbox Docker runtime | 无，创建时完整拉取镜像 | 镜像缺失或平台不匹配即重新 pull，没有 lazy、OverlayBD 或快照器，配置中也没有对应开关 |
| OpenSandbox Kubernetes runtime | 无按需镜像加载。on-demand Pool 是预热 Pod 池 | 池移除 Pod 启动，不改变镜像物化方式 |
| OpenSandbox fast-sandbox runtime | 有。`SandboxTemplate` 把 OCI 镜像转为可启动 rootfs 加完整快照，可选 OverlayBD 打包，制品按摘要发布到兼容 S3 的对象存储，运行时按需拉取并恢复 | 需要 Kubernetes、fast-sandbox CRD、Fastlet、S3 与 Firecracker，不是容器 runtime |
| p3b opensandbox direct mode | 无。p3b 自行驱动容器 runtime，完整物化镜像 | 没有 lazy 路径，该模式不声明此能力 |

本集群部署的 OpenSandbox 使用 Docker runtime，因此其 provider 模式声明的 `image_on_demand` 在此部署下不成立，direct 模式下的同一声明属于过度声明。

由此得到两条要求：

1. `opensandbox` backend 不声明 `image_on_demand`。provider 模式的镜像交付是块级，不是惰性层加载，声明它会让需要真正惰性加载的 spec 误选该 backend。只有部署为 fast-sandbox runtime 且 preflight 能核实时，才重新评估该声明。direct 模式自行驱动容器 runtime，完整物化镜像，同样不声明。
2. P4 纳入 p3b Docker backend 的范围，按 8.2 与 8.3 推进，不需要与 provider 去重。节点内 snapshotter 已由修订后的 D2 允许，S8 对内核级路径的排除仍然适用。

OpenSandbox 升级后需要重新验证本表，尤其是 Docker runtime 是否引入快照器集成。

### 8.2 就绪语义与合并（可早做）

这一段不依赖 lazy snapshotter，本身是 P1 邻域的纯收益，也是后续按需读取的前置可观测性。

区分 metadata_ready、rootfs_mounted、process_ready 和 tool_ready。服务对外提供的可执行就绪不能只由 `sleep infinity` 成功判断，应覆盖真实入口、shell、解释器或任务首次需要的工具。

块与命令输出在向调用方暴露前验证完整性，范围读取、并发下载、解压缓冲和本地缓存全部有界。同一内容块的并发需求合并，前台缺页读取优先于后台预取。截断显式返回。

### 8.3 能力门控的按需层（延后）

优先验证 Nydus 或 stargz snapshotter 与目标 Docker/containerd 版本的受支持集成，不自行实现完整远端文件系统。Docker 使用何种 image store 和 snapshotter 决定可用能力，不能仅修改 Python SDK 或把 pull 放到后台就宣称支持 lazy loading。[Nydus](https://github.com/dragonflyoss/nydus)、[stargz snapshotter](https://github.com/containerd/stargz-snapshotter)

接入方式与既有能力门控同构：节点启动时探测 snapshotter，探测到才声明 `image_on_demand`，显式请求 lazy 而能力不满足时拒绝配置，不做无声降级。这与 `dockerbackend` 对 `isolation_runtime` 的处理一致，即在配置了 runtime 时才声明，并在 preflight 阶段拒绝 daemon 缺失的 runtime。

eager 模式作为明确选择的策略与实验基线保留。如果必须引入另一 runtime，则通过独立 backend capability 接入既有 allocation 生命周期，不再复制整套调度和回收逻辑。

根 manifest 与元数据固定到 digest。以 image、recipe、role 和 phase 记录启动工作集，按预算预取高概率块。rollout 与 grader 的工具链可能不同，不能共用未经验证的工作集。工作集学习只记录制品内容访问，不把请求秘密写入公共缓存。

不默认后台下载完整制品。将完整填充作为单独策略评估，因为它可能只移动等待时间而没有减少传输。冷启动优化必须同时报告首次工具可用、完整 group 完成和总传输成本。

远端读取超时采用有界重试。源站故障导致执行环境失败时，按基础设施故障分类，不能透明重放可能已经修改状态的命令。缓存 pin、驱逐和运行中挂载的引用语义需由 snapshotter 保证并纳入集成测试。

### 8.4 测试、消融与出口

| 场景 | 验证点 |
| --- | --- |
| 完全冷启动 | 制品未完整下载即可执行真实工具，入口依赖读取正确 |
| 元数据热与块热 | 分离元数据、挂载和数据加载成本 |
| 同组启动洪峰 | 请求合并有效，源站与本地解压不因并发失控 |
| 随机读取与大文件写入 | 报告缺页尾延迟及 copy-up 放大，不只测试小入口文件 |
| 缺块、损坏和断连 | 内容校验有效，错误分类正确，不产生静默错误评分 |
| 缓存驱逐与服务重启 | 活跃挂载内容可用，引用和预算可恢复 |

消融为 eager、仅 lazy、lazy 加工作集预取，分别在冷缓存和预算受限缓存下测试。8.2 的就绪语义与合并独立消融，可在 lazy 之前单独验收。出口要求完整 group 的收益达到门槛，首次访问尾延迟、源站字节、额外 CPU 与后台 drain 均可接受。

## 9. 公共验证与度量协议

验证分为状态机单测、Linux 集成测试、固定任务轨迹回放和在线 RL 四层。fake backend 只能验证决策和故障处理，不能证明 cgroup、Docker、页缓存或 lazy loading 的实际收益。

| 维度 | 必须覆盖的组合 |
| --- | --- |
| 请求组织 | 单请求、真实 GRPO group size、同组同步突发、跨组交错、训练与验证竞争 |
| 环境复用 | 同镜像同准备、同镜像不同准备、不同镜像共享基础层、完全异构、tag 指向变化 |
| 负载形态 | 长推理等待、CPU 密集测试、编译、内存尖峰、后台线程、同步 grader 波峰 |
| 缓存状态 | 完全冷、元数据热、文件块热、准备制品热、缓存容量受限 |
| 故障 | RPC 取消、响应丢失、owner 死亡、节点重启、Docker 不可用、制品损坏和源站断连 |

固定硬件、内核、runtime、CPU 拓扑、镜像 digest、任务集合和随机种子。冷缓存实验使用独立测试节点或隔离的缓存目录，不在训练节点全局 drop caches。预热和后台填充都计入实验时间窗口，并报告工作结束后的 drain 成本。

先用 pilot 估计方差，再预先确定重复次数、最小有效收益和质量非劣界限。用配对任务及随机化实验顺序减少噪声，对 task 与 group 聚类计算置信区间。在线 RL 不要求 token 逐个相同，但必须保持奖励定义、评测覆盖、训练预算和样本选择规则一致。

每阶段上线门槛为正确性测试全部通过、质量满足非劣界限、基础设施失败无不可接受增长，并在目标负载中达到预先声明的收益。某类负载无收益时应限制策略适用范围，不能只挑最佳案例报告。

需要新增统一 benchmark harness，输出机器可读 manifest 与事件轨迹。manifest 记录源码和 runtime 版本、镜像 digest、任务集、种子、策略、预算、缓存初态及故障注入计划。事件贯穿 allocation、trajectory、group 和 artifact 身份。现有 `tests/sandbox/bench_p3b_concurrency.py` 与 `tests/sandbox/bench_p3b_v2.py` 的 create/echo/release 测试可用于控制路径基线，但不能替代长期驻留、推理等待、同步突发和 grading 的 RL 回放。实验结果写入与复现脚本对应的 `FINDINGS.md`，包含负结果和适用范围。

## 10. 实验矩阵

| 变体 | 包含内容 | 回答的问题 |
| --- | --- | --- |
| B0 | 固定的原始实现 revision，仅隔离环境运行 | P1 修复与优化的整体代价和收益 |
| B1 | P1 正确性基线与静态资源策略 | 所有后续策略的共同对照 |
| B2 | B1 加内存弹性 | 空闲 agent 让出内存的收益与风险 |
| B2C | B1 加可选 CPU 弹性 | 仅在本场景证实 CPU 先成瓶颈时评估 |
| B3 各子变体 | B1 加组内准备合并、本地跳表缓存、group 局部性或角色复用 | 每类去冗余的直接收益 |
| B2+B3 | 内存弹性与准备复用联合 | 交互收益和相关突发风险 |
| B4S | 固定前置策略，加就绪语义与块合并 | 首个工具可用与块读取放大的改善 |
| B4E / B4L | 在 B4S 上分别 eager、lazy | 按需读取的增量贡献 |

策略开关仅控制有效优化，不保留已修复的泄漏、错误评分或失效隔离路径。B0 通过独立 revision 比较，不在生产代码维护旧实现兼容层。

## 11. 工程拆分与复杂度

复杂度包含协议改动、异常状态、测试和部署约束。它不是性能收益预测，也不是未经团队容量校准的工期承诺。

| 工作包 | 主要落点 | 复杂度 | 关键成本或风险 |
| --- | --- | --- | --- |
| allocation 状态机、日志与对账 | node、backend、owner 协议 | 高 | 外部副作用非事务性，重启和未知结果必须闭环 |
| cgroup 域与统一采样 | pressure、Docker 创建、节点部署 | 中高 | systemd 与 cgroupfs、祖先约束和共置拓扑 |
| 拉取合并、库存与排队 | dockerbackend、placement、quota | 中 | 取消传播、缓存失效和公平性 |
| 预热规格与预算 | node warmpool、lifecycle | 中 | 接管正确性和失败销毁计费 |
| 有界 I/O、取消和 SDK 契约 | backend、协议、Python adapter | 中高 | 输出完整性、流控和进程树终止 |
| 内存弹性与两层配额 | admission、controller、quota 协议 | 高 | 相关峰值、OOM、部分更新和公平性 |
| 组内准备复用与干净冻结点 | artifact manager、SDK、mini-SWE workflow | 高 | 干净冻结点、评分语义与失败隔离 |
| 制品目录与 GC | artifact manager | 中高 | 延后，待准备复用测出收益 |
| group 局部性与 grader 提示 | workflow、placement、库存协议 | 中 | 避免热点和无限 pin，保持逐个准入 |
| 就绪语义与块合并 | backend、snapshotter 接口 | 中 | 就绪判定与输出完整性 |
| 能力门控的 snapshotter | Docker/containerd 部署、backend capabilities | 高 | 制品格式、挂载服务故障与运维 |
| 内核或 microVM 页共享 | 独立 runtime/backend | 很高 | 超出当前 Docker 控制层，需额外隔离审查与故障测试 |

实施顺序为 P1 生命周期与观测 → P1 并发和预热 → P2 内存弹性 → P3 组内准备复用 → P3 局部性与角色复用 → P4 就绪语义与合并 → P4 能力门控 snapshotter。P3 的制品设计可在 P2 实验期间开发，但联合上线前必须完成交互消融。P4 的 snapshotter 段已按 8.1 核实不与 provider 重复，可直接进入实现。

## 12. 未来方向与证据边界

本节只以 2024 至 2026 年正式系统会议论文作为会议证据。每项区分论文机制与 PSRL 工程推断，按实际瓶颈选择研究分支，不要求全部实现。

### F1. 从固定安全余量走向性能约束下的内存回收

**依据：**[MDK，OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/patel) 将目标设为性能约束下最大化内存节省，提供离线策略评估与 Memory Performance Curves。

- **想做什么：**在 P2 的可靠闭环上，按 image、recipe、任务阶段学习「回收多少内存会带来多少性能损失」，决定收缩对象、幅度及维持时间。
- **前后区别：**只根据用量、压力和固定余量收缩 → 将 refault、恢复成本和实际延迟约束纳入决策。
- **落点与复杂度：**扩展节点 resource controller 和 benchmark，复杂度中高。离线建模与在线策略分离，不能使用评测集的未来访问作为在线输入。
- **验证：**按时间切分训练和测试轨迹，与 P2 阈值策略比较物理 byte-seconds、完整 group/s、refault、p99 工具时延和 OOM。

### F2. 从固定并发上限走向多瓶颈反馈控制

**依据：**[Svalinn，OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/pardeshi) 分离准入与各瓶颈的时延控制。[Dirigent，SOSP 2024](https://anakli.inf.ethz.ch/papers/dirigent_sosp24.pdf) 说明高频 sandbox 生命周期可能让控制路径成为瓶颈。

- **想做什么：**在 P1 已分开的 pull、解压、create、exec 和 cleanup 队列上，按局部排队、服务时间与内存带宽压力调节并发。
- **迁移边界是状态语义。** Svalinn 中的丢弃或重传不能直接用于已执行的有副作用命令，Dirigent 放宽精确状态重建的做法不能替代 P1 的所有权日志。

### F3. 让环境与评分弹性服务于完整 RL 流水线

**依据：**[RollArt，OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/gao) 将生成、环境交互和 reward 按轨迹解耦，并将无状态 reward 交给独立弹性资源。

- **想做什么：**结合 group 剩余轨迹、grader backlog、GPU 等待和策略滞后，在新 rollout、完成中的 group 与 grading 之间动态分配预算。
- **验证：**固定 CPU 与 GPU 总量或统一成本预算，比较完整 group/s、time-to-score、GPU 空闲、评分积压和策略滞后。独立执行的 SWE grader 必须声明完备输入并从干净基线启动，任意 agent 命令不能视为无状态函数。

### 暂不进入 p3b 的分支

以下方向依赖内核、宿主机或 microVM 能力，普通 Docker API 无法独立实现，短 FaaS 的冷启动收益也不自动适用于长期驻留 RL。

| 方向 | 依据 | 排除理由 |
| --- | --- | --- |
| 干净进程初始化复用与快照恢复 | [AFaaS，OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/chai-xiaohu)、[Spice，OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/holmes) | 需要 guest 或内核原语，不是 stock Docker 配置 |
| lazy 文件系统的元数据与文件页快路径 | [FlacIO，FAST 2025](https://www.usenix.org/conference/fast25/presentation/liu-yubo)、[CoFS，FAST 2026](https://www.usenix.org/conference/fast26/presentation/wang-li) | 需要 OverlayFS 或 FUSE 内核改造 |
| 带宽感知的预测预取 | [ThinkAhead，FAST 2026](https://www.usenix.org/conference/fast26/presentation/chen) | 原场景最终需全量驻留，不能据此承诺总流量下降 |
| 分层内存与可扩展换页 | [Colloid，SOSP 2024](https://www.cs.cornell.edu/~midhul/papers/colloid.pdf)、[ScaleSwap，FAST 2026](https://www.usenix.org/conference/fast26/presentation/ahn) | 需要多层内存或内核 swap 改造 |

下列 2026 年 agent 与 RL 场景工作作为补充证据，其 microVM 或 DAX 路径不直接等于 Docker 能力，需验证实际存储与 runtime 接口后再评估。[DSec](https://arxiv.org/html/2609.22978v1)、[WeEnv](https://arxiv.org/html/2609.30766v1)、[VenusRL](https://arxiv.org/html/2610.03286v1)

所有论文报告的收益不作为 PSRL 的预期收益，未在目标 Linux 节点验证的能力均需工程预检。共同启示是分离逻辑环境与物理驻留、识别真正不可变的共享内容、把数据传输和执行启动解耦，并将控制开销及故障语义纳入端到端验证。

:::{seealso}
- {doc}`sandbox_service`: 控制面与节点面的边界
- {doc}`sandbox_refactor`: D2、D8、D15、D16 与多节点设计
- {doc}`sandbox_snapshot_store`: 快照存储与 S8 的范围界定
:::

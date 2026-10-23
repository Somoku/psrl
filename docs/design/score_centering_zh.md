# Score Centering（分数居中）

分数居中用于消除「训练-推理不一致」（training-inference mismatch）给策略梯度带来的偏差。rollout
采样器（vLLM）与训练器（FSDP 或 Megatron）用同一份权重做前向，但走的是不同的 kernel、dtype 和 batch
形状，因此智能体实际采样所服从的分布 $q$ 与训练器评分用的分布 $p_\theta$ 并不相同。分数居中的做法是
稀疏地记录采样器的分布，再从策略梯度项里减去「训练器自身的期望得分」，从而在不改变期望目标的前提下
抵消这种漂移。算法来自 arXiv 2609.20807，且只作用在训练侧：采样器只负责上报它的 head 分布。

本文面向熟悉 PPO 的读者，说明分数居中在 PPO 之上补了什么、PSRL 支持的两种 head 模型、一次捕获如何
一路流到损失，以及它的性能代价。捕获侧的 wire 契约见
[sampler_distribution.md](sampler_distribution.md)，本文的英文版见
[score_centering.md](score_centering.md)。

## 1. PPO 在 off-policy 下错在哪

PPO 的裁剪目标建立在「行为策略就是产生 rollout 的策略」这一假设上。设重要性比

$$
r_t(\theta) = \frac{p_\theta(a_t\mid s_t)}{p_{\text{old}}(a_t\mid s_t)},
$$

其裁剪目标为

$$
\mathcal{L}^{\text{clip}}(\theta) = -\mathbb{E}_t\left[\min\!\big(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\big)\right].
$$

在异步 RL 循环里，这个假设会在两个层面失效：

- **陈旧性（staleness）**：$p_{\text{old}}$ 落后若干步。这本来就是比值要吸收的偏差。
- **采样不一致（sampling mismatch）**：即便陈旧性为零，采样器记录的 log-prob（vLLM 在
  `logprobs_mode=processed_logprobs` 下给出的值）与训练器前向算出的 log-prob 也来自不同的数值路径，于是行为分布 $q$ 与训练分布 $p_\theta$ 之间会有一个小而系统性的漂移。

分数居中针对的是第二个。关键在于：这个漂移不会自己平均掉。取常数奖励时策略仍会持续移动，原因是
「被采样 token 的得分」在采样器分布下不满足零均值。

当不一致较大时，PSRL 可以用 **bypass 模式**，即带重要性权重的 REINFORCE 目标

$$
\mathcal{L}(\theta) = -\mathbb{E}_{a\sim q}\!\left[\,w(a)\,\log p_\theta(a)\,A\,\right],
\qquad
w(a) = f\!\left(\frac{p_\theta(a)}{q(a)}\right),
$$

其中 $f$ 是可选的 IS 规则（第 4 节）。它在 $p_\theta = q$ 时无偏，但梯度中仍残留漂移项
$\mathbb{E}_{a\sim q}[\,w(a)\nabla_\theta\log p_\theta(a)\,]$。分数居中加进来的校正项正是用来抵消它。

## 2. 分数居中项

### 2.1 记号

对某个 response 位置，记

- $a$：采样到的 token，$a\sim q$。
- $H$：候选 token 集合，也就是采样器的 head（第 3 节给出它的两种取法）。
- $q_v,\ p_\theta(v)$：采样器与训练器在候选 $v$ 处的概率。
- $A$：该位置的 advantage。
- $f$：IS 规则。

### 2.2 期望得分恒等式

对一个归一化分布 $p_\theta$，其得分函数的期望为零：

$$
\mathbb{E}_{v\sim p_\theta}\!\left[\nabla_\theta\log p_\theta(v)\right]
= \sum_v p_\theta(v)\,\nabla_\theta\log p_\theta(v)
= \nabla_\theta\sum_v p_\theta(v)
= \nabla_\theta\,1
= 0 .
$$

这就是「居中」的来源。它提供了一个均值为零的控制变量，把它从策略梯度中减去，只改变方差、不改变期望。

### 2.3 残差与控制变量

bypass 目标的主项期望梯度为

$$
\nabla_\theta\,\mathbb{E}_{a\sim q}\!\left[-A\,w(a)\log p_\theta(a)\right]
= -A\sum_v q_v\,w_v\,\nabla_\theta\log p_\theta(v),
\qquad
w_v = f\!\left(\frac{p_\theta(v)}{q_v}\right).
$$

分数居中引入如下残差，并把它投影到训练器的得分上：

$$
q_{\text{mass}} = \sum_{v\in H} q_v,
\qquad
p^{\text{mass}}_\theta = \sum_{v\in H} p_\theta(v),
$$

$$
\rho = \frac{1-q_{\text{mass}}}{\max(1-p^{\text{mass}}_\theta,\ \varepsilon)},
\qquad
\alpha = \rho\, f\!\left(\frac{1}{\rho}\right),
$$

$$
h_v = q_v\,w_v,
\qquad
r_v = h_v - \alpha\, p_\theta(v),
\qquad
C = \sum_{v\in H}\mathrm{sg}[r_v]\,\log p_\theta(v).
$$

其中 $\mathrm{sg}[\cdot]$ 表示 stop-gradient。每个位置的训练损失为

$$
\mathcal{L} = -A\,w(a)\log p_\theta(a) + A\,C .
$$

### 2.4 为什么它能抵消漂移

对 $C$ 求梯度时残差 $r_v$ 被冻结，只有 $\log p_\theta(v)$ 带梯度：

$$
\nabla_\theta \mathcal{L}
= -A\,w(a)\nabla_\theta\log p_\theta(a)
+ A\sum_{v\in H} r_v\,\nabla_\theta\log p_\theta(v).
$$

对 $a\sim q$ 取期望：

$$
\mathbb{E}_{a\sim q}\!\left[\nabla_\theta \mathcal{L}\right]
= -A\sum_{v} q_v w_v \nabla_\theta\log p_\theta(v)
+ A\sum_{v\in H}\big(q_v w_v - \alpha\, p_\theta(v)\big)\nabla_\theta\log p_\theta(v).
$$

**support 模式是精确的。** 此时 $H$ 就是 $q$ 的完整支撑，于是 $q_{\text{mass}} = 1$，从而
$\rho = 0$、$\alpha = 0$，上式变成

$$
\mathbb{E}_{a\sim q}\!\left[\nabla_\theta \mathcal{L}\right]
= -A\sum_{v\in H} q_v w_v \nabla_\theta\log p_\theta(v)
+ A\sum_{v\in H} q_v w_v \nabla_\theta\log p_\theta(v)
= 0,
$$

主项里头部携带的漂移被精确抵消，不需要任何尾部建模。

**top-k 模式用 $\alpha$ 建模尾部。** 此时 $H$ 只是 $q$ 的前 $k$ 个 token，主项还有一部分质量落在
$H$ 之外：

$$
\mathbb{E}_{a\sim q}\!\left[\nabla_\theta \mathcal{L}\right]
= -A\sum_{v\notin H} q_v w_v \nabla_\theta\log p_\theta(v)
- A\,\alpha\sum_{v\in H} p_\theta(v)\nabla_\theta\log p_\theta(v).
$$

$\rho$ 是采样器与训练器的尾部质量比（$1-q_{\text{mass}}$ 对 $1-p^{\text{mass}}_\theta$），$\alpha$ 用它
把训练侧尾部按比例缩放，使残差在整个词表上近似于采样器的残差，抵消在尾部模型的精度内成立。

### 2.5 实现要点

- 残差在 `torch.no_grad()` 下计算，只有 $\log p_\theta(v)$ 带梯度。去掉这个 detach，校正项就变成了
  另一个目标函数，而不再是控制变量。
- padding 槽位携带哨兵 id `-1` 与 `-inf` log-prob，head mask 会在它贡献损失之前将其排除。
- 支持模式下 $\rho$ 的分子分母都会趋近于零，代码直接短路为「无尾部」（`alpha = 0`），而不是去评估一个
  在浮点噪声上求出的 $\rho$。

## 3. 两种 head 模型

$H$ 是 rollout 阶段采样器记录下来的 head。PSRL 支持两种建模方式，由采样参数决定：

| | top-k（$top\_p = 1$） | support（$top\_p < 1$） |
|---|---|---|
| head | 采样器概率最高的 $k$ 个 token | 采样器的完整 nucleus |
| 训练器 $p$ | 全词表归一化 | 在记录的支撑上重归一化 |
| 尾部 | 用 $\alpha$ 建模，$\rho$ 为尾部质量比 | 无尾部，$\alpha = 0$ |
| 校正 | $\sum_{H}(q_v w_v - \alpha p_v)\log p_v$ | $\sum_{H} q_v w_v \log p_v$ |
| 捕获宽度 | $sampler\_max\_tokens$ | $sampler\_max\_tokens$ 加一个哨兵槽 |

**top-k** 是采样未截断时的默认模式。head 精确，尾部被建模为「训练器尾部按采样器尾部质量缩放」，因此
校正项不需要看到完整词表，$\rho$ 与 $\alpha$ 承载了这个尾部近似。

**support** 是截断采样（$top\_p < 1$ 或 $top\_k > 0$）下的精确模式。head 就是采样器的完整支撑，由一个
哨兵条目证明其完整性，于是没有尾部、$\alpha = 0$。这里训练器分布取「支撑条件分布」：训练器把 $p_\theta$
在记录的支撑上重归一化，同时 REINFORCE 项所用的「被采样 token 的策略 log-prob」（先减去 `sc_log_support_mass`）
也做同样的重归一化，于是策略项与校正项共享同一个分布与同一个梯度，而全词表 log-prob 保留给 KL 项。这
正是 support 模式精确、而非尾部近似的根本原因。

## 4. 重要性权重规则

$f$ 由 rollout-correction 配置选择，同时作用于 $h_v$ 与 $\alpha$。所有计算都在概率或对数比值上进行，
因此不会对无界的比值取指数，也不会出现 $q_v = 0$ 时的 $0\cdot\infty$。

| `rollout_is` / 阈值 | $f(r)$ | 效果 |
|---|---|---|
| `None` | $1$ | 无 IS 权重，$h_v = q_v$ |
| 单个浮点 $c$（TIS） | $\min(r,c)$ | 上截断，head 变为 $\min(p_v, c\,q_v)$ |
| `"low_high"`（IcePop） | 区间内为 $1$，区间外为 $0$ | 把比值落在 $[low, high]$ 之外的项置零 |

被采样 token 自身的权重 $w(a) = f(p_\theta(a)/q(a))$ 由 REINFORCE 项通过 `rollout_is_weights` 施加。

## 5. 完整 workflow

```
rollout (vLLM)                       SMG gateway + TITO              PSRL agent loop
  sampler_top_logprobs width  --->   per-turn sampler_distribution   --->  SamplerDistribution
                                     (offsets, ids, log-probs,             (validate, trim)
                                      complete)                            |
                                                                           v
training (FSDP / Megatron)           veRL SC logits processor        sampler_adapter
  sc_correction  <--- centering  <---  p at head ids, q at head ids  <---  dense head or
  policy log-prob                     (chunked, vocab-parallel)            compact support
```

1. **捕获。** SessionRouter 在 chat、Responses、Anthropic Messages 与 `/generate` 四条路径上都向 SMG 请求
   `sampler_top_logprobs`。chat 与 Responses 在 SMG 内部下沉为统一的 chat 训练 IR。TITO 为每一轮保存一份
   紧凑的 `SamplerDistribution`，并在 partial rollout 中保留它。
2. **组装。** `psrl.utils.tito.training_data` 遍历各轮，用贪心匹配对齐、裁剪边界 token，并拼接每一轮的
   分布。`psrl.utils.sampler_adapter` 把 head 铺到完整的 prompt+response 序列上，response token $r$ 落在
   预测行 $\text{prompt\_length} - 1 + r$。
3. **打分。** engine 的 logits processor 从同一份 logits 上同时算出训练器的 head log-prob 与居中项，
   actor 损失再加上 $A\cdot C$。
4. **记录。** 每个 token 上报 `sc_correction`、`sc_sampler_head_mass`、`sc_train_head_mass` 与
   `sc_tail_ratio`，前缀为 `actor/sc_*`。

最小配置（rollout 上开捕获，损失上开开关）：

```yaml
algorithm:
  rollout_correction:
    score_centering: true
    bypass_mode: true          # required
    loss_type: reinforce       # required
    rollout_is: token          # None | token
    rollout_is_threshold: 2.0  # TIS float, or a "low_high" IcePop band
gen_actor_rollout_ref:
  rollout:
    sampler_distribution: auto # disabled | topk | support | auto
    sampler_max_tokens: 32
    sampler_compact_support: true
    logprobs_mode: processed_logprobs
    temperature: 1.0
    top_p: 1.0                 # topk needs 1.0, support allows < 1
```

`auto` 在 $top\_p < 1$ 或 $top\_k > 0$ 时解析为 `support`，否则为 `topk`。该开关会在 worker 选定损失
函数之前被镜像到 actor 的静态 `policy_loss` 上，原因是 worker 在 `init_model` 里只挑一次 `loss_fn`，
驱动侧在训练时的运行期改动会太晚。分数居中要求 logits 被物化，拒绝蒸馏、投机采样、`mtp` 训练与动态
context parallel，且验证会话会关闭捕获。

## 6. 性能优化

分数居中不是免费的。它需要训练器在任意 token id 上的 logits，因此 fused 的 logits/loss kernel 必须关闭，
完整的 $(\text{rows}, \text{vocab})$ logits 会被物化。在 cross-entropy 之外，它还要多算一次 head 归一化
器和一次反向重算，大约相当于两遍 $(\text{rows}, \text{vocab})$。PSRL 与 veRL 补丁把这些代价限制住，并让
通信与主机内存保持很小。

| 优化 | 机制 | 效果 |
|---|---|---|
| 按元素预算分块 | actor engine 配置项 `score_centering_chunk_elems`（默认 6400 万元素，约 256 MiB）限制 fp32 $(\text{rows}, \text{vocab})$ 工作区，每个 chunk 的行数由本 rank 的词表分片推出 | 词表增大时显存保持平坦 |
| 反向重算 | 归一化 primitive 只保存 logits、ids 与每行两个标量，反向时按 chunk 重算 softmax | 不为反向保存 $(\text{rows}, \text{vocab})$ 的 softmax |
| 词表并行归约 | 张量并行下只有被选中的 logits 与每行归一化标量跨 rank，绝不传完整词表 | 单 rank 显存随 $\text{vocab}/\text{TP}$ 缩放 |
| support 路径 | support 模式在每行记录的支撑上归一化，复杂度 $O(\text{rows}\cdot k)$，只做一次归约，而不是词表规模的一次 | support 模式的代价与词表基本无关 |
| 紧凑传输 | `sampler_compact_support` 传每行长度加扁平的 ids 与 log-probs，而不是稠密的 $[T,k]$ head，由训练器在设备上重建。它是默认值，且支持 context parallel | rollout 到训练器的载荷大幅缩小 |
| FSDP 批量 gather | 五个逐 token 字段用一次集合通信一起 gather | 每个 micro-batch 从五次通信降为一次 |
| 主机同步 | 紧凑字段在一次 device-to-host 同步里读取行长度的 min、max、sum | 每个字段一次同步，而不是每次校验一次 |
| 校验门控 | `sampler_strict_validation` 门控逐行的重复、质量与一致性检查，结构性检查始终执行 | 长响应在训练时跳过 $O(\text{rows})$ 的 Python 遍历 |
| TITO 组装向量化 | 逐轮的 token 拆分与边界匹配改用 numpy | 不再对每个 response token 做 Python 循环 |

需要计入预算（而非 bug）的成本：

- **物化 logits。** 长序列下峰值由 $(\text{rows}, \text{vocab})$ 的 logits 及其梯度主导。用张量并行或
  context parallel 把单 rank 的 $\text{rows}$ 压下来。
- **Megatron 的 CE clone。** Megatron 的 fused 词表并行 cross-entropy 会原地改写它保存的 logits，而
  分数居中的反向还要复用同一份 logits，因此 engine 会在 CE 之前克隆一份。若同时开启 entropy，峰值会达到
  三份 $(\text{rows}, \text{vocab})$。
- **逐 chunk 通信。** 词表并行路径每个行 chunk 发出三次 all-reduce，因此通信次数按
  $\text{rows}\cdot\text{vocab}/(\text{chunk\_elems}\cdot\text{TP})$ 增长。张量并行与更大的预算都能把它
  压下来。
- **采样侧成本。** vLLM 要为每个位置计算 top-k log-prob 来填满捕获宽度，网关再逐位置组装 head。捕获宽度
  是主要旋钮：`sampler_max_tokens` 只按采样实际所需开大即可。

## 7. 代码索引

| 路径 | 作用 |
|---|---|
| `psrl/utils/sampler_distribution.py` | `SamplerCapture`、`SamplerDistribution`、捕获宽度定义与校验 |
| `psrl/utils/sampler_adapter.py` | prompt+response 序列上的稠密与紧凑生产者 |
| `psrl/utils/tito/training_data.py` | 多轮对齐、裁剪、逐轮分布组装 |
| `psrl/workers/gen/session_router.py` | 各协议的捕获请求契约 |
| `psrl/workers/agent_loop/loops/` | 逐轮捕获、TITO 会话处理、`/generate` 解码 |
| `third_party/verl/verl/trainer/ppo/score_centering.py` | head log-prob primitive、居中项、logits processor、损失入口 |
| `third_party/verl/verl/workers/utils/losses.py` | 把 `sc_correction` 加到策略损失；采样 log-prob 的支撑重归一化已在 logits processor 内完成 |
| `third_party/verl/verl/workers/engine/{fsdp,megatron}/transformer_impl.py` | engine 接线、chunk 预算、CP 切分与 Megatron CE clone |

veRL 侧改动以 `patch/verl/dc709af.patch` 的形式发布，应用到固定的 veRL 版本上。

## 验证

在 PSRL 仓库根目录、并把打过补丁的 veRL 放到路径上后运行聚焦的 CPU 套件：

```bash
PYTHONPATH=third_party/verl python -m pytest -q \
  tests/utils/test_sampler_distribution.py \
  tests/utils/test_sampler_adapter.py \
  tests/tito/test_session_router.py \
  tests/tito/test_session_agent_loop.py \
  third_party/verl/tests/trainer/ppo/test_score_centering_on_cpu.py
```

这套用例覆盖捕获契约、两种 head 模型、损失集成与多轮对齐，但不替代加速器上的验证：请按你实际训练的
FSDP/Ulysses 与 Megatron TP/CP/PP 组合验证，并在所选捕获宽度下测量训练峰值显存与捕获带宽。


## 验证结果

CPU 套件锁定了估计器本身。加速器路径在单机 8 卡上用 GSM8K 做了端到端验证：4 张训练卡、1 个 rollout
实例和 1 个 validate 实例。FSDP2 使用 Qwen2.5-0.5B-Instruct；top-k 32 取
`temperature=1.0, top_p=1.0`，support 捕获取 `top_p=0.9`，回复上限 1024 token。奖励使用仅用于验证的
GSM8K 打分器，同时接受 `####` 与 `\boxed{}`；否则 instruct 模型不接受格式时正确答案也会得 0 分。

| Head | IS 规则 | `sc_support_aligned` | sampler head mass | trainer head mass | `sc_correction` |
|---|---|---|---|---|---|
| top-k 32 | 无 | 0 | 0.988 | 0.987 | -4.5e-4 |
| support | 无 | 1.000 | 1.000 | 0.976 | -8.2e-4 |
| support | IcePop 0.9-1.1 | 1.000 | 1.000 | 0.976 | +1.9e-2 |
| top-k 32 | TIS 2.0 | 0 | 0.994 | 0.994 | -5.4e-5 |

这些结果说明：

- support 行的 `sc_sampler_head_mass = 1`、`sc_sampler_tail_mass ≈ 0`、`sc_support_aligned = 1`，
  即核采样支撑被完整捕获，训练端在该支撑上重归一化。
- top-k 行上 trainer 与 sampler 的 head mass 一致到约 1e-5，采样 token 的 log-prob 一致到约 1e-3
  （`rollout_corr/log_ppl_diff`），同策略下中心化项接近 0。
- IcePop 的区间拒绝了 2.2% 的 token 比率（`rollout_is_oob_ratio`）且 support 对齐仍精确；TIS 的
  `rollout_is_mean ≈ 1`、最大比率 1.19、有效样本率 0.999。
- 支撑预算不足会 fail closed：`sampler_max_tokens=128` 时 GSM8K 上的核超过预算，agent loop 抛出
  `Sampler support exceeds the capture budget`。本数据分布用 `512` 足够；应按实际核大小而不是
  head mass 配置预算。

Megatron 引擎走同一路径，在单机上用张量并行 2 分别跑 Qwen2.5-0.5B 与 Qwen3-8B（2 张训练卡、4 张
rollout 卡）。vocab-parallel normalizer 的 head mass 在 0.5B 上与 sampler 一致到 1e-4（0.99515 对
0.99517），在 8B base 上一致到 1e-6（0.9999987 对 0.9999988，采样 log-prob 差 7.4e-4），与 FSDP
同级；head、complete receipt 与 sampled label 的 THD 切分路径每步都会执行，张量并行 4 走同一条
代码路径，只是词表分片更宽。

### 训练开销

上面的 8B Megatron TP2 两次运行只有 score centering 不同。被优化器状态主导的峰值显存不变（分配峰值
72.44 GB，差异小于 0.001 GB）；chunk 化的 fp32 normalizer 工作区受 256 MiB 元素预算约束，不随词表
增长。按训练 token 计，开启中心化时 actor update 为 404 tok/s，关闭时为 417 tok/s（约 3% 开销），
MFU 从 0.344 降到 0.298。在 FSDP2 上，processor 同样只是在已物化 logits 之上增加有界工作区，
而不是再开一份词表大小的缓冲。

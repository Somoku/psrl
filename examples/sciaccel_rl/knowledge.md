# SciAccel-RL：Thinking / TITO 前缀哈希 知识笔记

本文记录 `runner.py` 里 `separate_reasoning=False` 这行配置背后的完整推理链，以及
psrl 与 SkyRL 在多轮训练数据组装上的架构级差异。这些内容不在代码或 git history 里，
但直接决定了「动 chat template / 动 reasoning flag / 换模型」的风险边界。

## 引用锚点

文中所有 `file:line` 均基于以下版本。`third_party/smg` 与 SkyRL 会随上游更新，
行号对不上时先核对下面的 commit，再判断是文档过期还是看错了文件。

| 组件 | 版本 | 备注 |
|---|---|---|
| `third_party/smg` | `b2143480` | 不是 git submodule，是独立 checkout |
| SkyRL | `59d4daed` | `/apdcephfs_zwfy10/share_303541817/lhy/SkyRL`（全机唯一副本） |
| harbor | `0.22.0` | pip 包，`site-packages/harbor/` |

**路径别名**：`/apdcephfs_zwfy10/...` 与 `/apdcephfs_zwfy10_303541817/...` 是同一挂载的
两个别名（实测同 inode），不是两份拷贝。另外 base conda 环境残留着指向
`share_303541817/ls/psrl` 的 editable 安装，但 `source lhy/env/psrl.sh` 后
`sys.path` 里不会出现 `ls/`，全部解析到 `lhy/psrl/third_party/`。
（`ls/` 那份 smg 的 protocols/tito/reasoning_parser 关键文件与本份逐字节相同。）

## 1. `separate_reasoning` 在做什么

SMG（gateway）的请求级开关，**默认 true**：

`third_party/smg/crates/protocols/src/chat.rs:303-305`
```rust
/// Separate reasoning content from final answer (O1-style models)
#[serde(default = "default_true")]
pub separate_reasoning: bool,
```

打开时，gateway 在把 backend 输出转成 OpenAI 响应前会跑一遍 reasoning parser
（`model_gateway/src/routers/grpc/regular/processor.rs:106-140`），把 `<think>…</think>`
切掉，CoT 进 `reasoning_content`，`</think>` 之后的进 `content`：

```
模型原始输出:  <think>我先看 setup.py …</think>\n\n我要运行测试
separate_reasoning=true  → content="\n\n我要运行测试",  reasoning_content="我先看 setup.py …"
separate_reasoning=false → content="<think>我先看 setup.py …</think>\n\n我要运行测试", reasoning_content=None
```

## 2. 为什么打开会炸掉 TITO

TITO 靠**消息前缀哈希**把多轮请求串成一条 trajectory，而哈希把 `reasoning_content`
也算进去了：

`third_party/smg/crates/tito/src/normalizer.rs:187-191`
```rust
// reasoning_content
let reasoning = reasoning_content.as_deref().unwrap_or("");
hasher.update(reasoning.as_bytes());
```

但 terminus-2 侧写回历史时**只写 `{role, content}`**：

`harbor/llms/chat.py:113-116`
```python
assistant_message = {"role": "assistant", "content": llm_response.content}
if self._interleaved_thinking and llm_response.reasoning_content:
    assistant_message["reasoning_content"] = llm_response.reasoning_content
```

`interleaved_thinking` 默认 False（`harbor/agents/terminus_2/terminus_2.py:179`），
runner.py 也没开。于是：

| 字段 | 第 N 轮 server 存的 leaf | 第 N+1 轮 client 回放的 prefix |
|---|---|---|
| content | `"\n\n我要运行测试"` | `"\n\n我要运行测试"` ✓ |
| reasoning_content | `"我先看 setup.py …"` | `None` ✗ |

哈希不同 → `find_prefix` 找不到父节点 → `resolve_trajectory_id` 里
`leaf_hash == parent_hash` 匹配失败（`crates/tito/src/store.rs:380-391`），
落到 else 分支分配**全新 trajectory id**。结果：32 轮 rollout 变成 32 条各只有一轮的
trajectory，per-turn record 链断掉。

关掉后 parser 整段 no-op，`<think>` 留在 `content` 里，两边字节一致，auto 匹配恢复。

### 为什么不用 `interleaved_thinking=True`

哈希层面也能修好，但会丢 CoT：`config/qwen3_acc_thinking.jinja2` 的 assistant 分支
只渲染 `content`，完全不看 `reasoning_content`。CoT 被拆出去就永远进不了下一轮 prompt，
正好抵消这个自定义模板的唯一目的。留在 `content` 里是唯一同时满足
「哈希一致」+「CoT 进 prompt」的方案。

附带鲁棒性收益：若 CoT 被 max_tokens 截断、没吐出 `</think>`，parser 会把**全部**文本
判成 reasoning（`crates/reasoning_parser/src/parsers/base.rs:62-65`），`content` 变空 →
client 回放到一条空 assistant 消息。关掉后不会丢内容。

## 3. SkyRL 的做法：不是配 flag，而是根本不开 parser

`SkyRL/examples/train_integrations/harbor/run_harbor_gen.sh:44` 传给 vLLM 的只有
`engine_init_kwargs.chat_template`，**没有 `reasoning_parser`**。而 vLLM 默认值是空串：

- `third_party/vllm/vllm/config/reasoning.py:22` → `reasoning_parser: str = ""`
- `third_party/vllm/vllm/parser/parser_manager.py:230-232`：
  ```python
  if not reasoning_parser_name:
      return None      # ← 不给就是 None，压根不 split
  ```

所以 **SkyRL 不需要任何 flag：vLLM 默认就不拆**，`<think>` 天然留在 `content`。
我们需要 `separate_reasoning=False` 纯粹因为 **SMG 的默认值反了**。
同一个目标，SkyRL 靠默认值达成，我们得显式关掉。

SkyRL 把这个契约写进了 README（`SkyRL/skyrl/train/utils/templates/README.md`）：

> we do not read `reasoning_content` at all ... Therefore, you **should not parse the
> thinking content and pass in all generated text as part of `content`**.
> The motivation is to not strip thinking tokens and keep the chat history in
> multi-turn training **strictly appending**, making the training on-policy without
> performing step-wise training.

## 4. thinking 会一路累积，一条 traj 里有多个 `<think>` —— 这是刻意设计

对比两个模板的 assistant 分支：

```jinja
{# skyrl-agent 版 & 官方 Qwen3：靠 loop.index0 > ns.last_query_index 门控，只留最后一轮 #}
{%- if reasoning_content %}
    {{- '...\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}

{# skyrl/train 版（= 我们的 config/qwen3_acc_thinking.jinja2，实测字节相同）#}
{{- '<|im_start|>' + message.role + '\n' + content }}   ← 就这一行，无条件
```

因为 `content` 本来就带 `<think>…</think>`，无条件拼接 = 历史里每轮 CoT 全部保留。
32 轮就是 32 个 `<think>` 块。

**为什么必须这样**：token-level on-policy 的硬性要求。若第 N 轮生成时 prompt 里有 CoT，
而第 N+1 轮渲染时剥掉了，第 N 轮的 response tokens 在第 N+1 轮序列里就不再是同一段 token
—— 训练时算的 logprob 和 rollout 时对不上，梯度是错的。README 的 "strictly appending"
就是指这个：历史只能追加，不能改写。

代价是 context 涨得快，也是 `max_model_len=32768` / `max_turns=32` 这套配置的由来。

## 5. psrl vs SkyRL：训练数据组装方式完全不同

| | SkyRL | psrl |
|---|---|---|
| 数据来源 | Harbor `rollout_details`（client 侧攒的 per-turn token ids） | SMG TITO `accumulated_token_ids`（server 侧一条连续 token 流） |
| 样本粒度 | **step-wise**：每轮一条独立样本，各带 `prompt_token_ids[t]` | **一条 trajectory**：`prompt_ids` + 拼接的 `response_ids` + `response_mask` |
| 靠什么串起来 | 不用串，`trajectory_ids` 标记归属即可 | **前缀哈希匹配**（`crates/tito/src/normalizer.rs`） |

- SkyRL：`harbor_generator.py:140-166` 是 `for t in range(n_turns): prompt_token_ids.append(p_ids)`，每轮独立成样本。
- psrl：`psrl/utils/tito/training_data.py:85-175`，按 `prompt_token_count` 切片，
  env token 打 mask=0，assistant token 打 mask=1，**拼成一条连续序列**。

**psrl 对「历史严格追加」的依赖比 SkyRL 更强，失败模式也不同**：

- **SkyRL**：即使 CoT 被剥，step-wise 每条样本仍自洽（prompt 和 completion 是同一次请求
  配对的），只是变成 off-policy —— 训练照跑，指标慢慢变差。**静默退化。**
- **psrl**：`training_data.py:135` 的 greedy match 会发现 record 的 output_ids 与
  `accumulated_token_ids` 对不上，`trim_count` 超限直接 `raise ValueError`
  （`:161-167`）；或哈希不匹配导致 trajectory 分裂成 32 条单轮，
  `get_primary_training_data` 的 `len(training_data) != 1` 断言抛错。**会炸，不会静默。**

### 结论

当前配置（`separate_reasoning=False` + acc_thinking 模板）自洽且正确，语义上等价于
SkyRL 的默认行为。但它靠**两个必须同时成立的条件**撑着：

1. reasoning parser 关掉（`separate_reasoning=False`）
2. 模板无条件拼 `content`（不做 reasoning_content 门控）

任何一边被改动（换模型时顺手换成官方 Qwen3 模板、升级 SMG 后 flag 改名等），
psrl 会以 ValueError 或 trajectory 数量断言的形式炸出来。

## 6. 这不是 Qwen3 特有的问题

三个成因没有一个跟 Qwen3 绑定：parser 默认开、client 丢 `reasoning_content`、
TITO 哈希算它。**任何 thinking 模型走这条链路都会中招**：deepseek-r1、glm4.5/4.7/5、
kimi-k2-thinking、minimax-m2、nemotron…

Qwen3.5 也中招，且被同一个 parser 接管。pattern 表按顺序首次匹配 + 子串匹配
（`crates/reasoning_parser/src/factory.rs:188-191`）：

```rust
registry.register_pattern("qwen3-thinking", "qwen3_thinking");
registry.register_pattern("qwen-thinking",  "qwen3_thinking");
registry.register_pattern("qwen3",          "qwen3");   // ← "qwen3.5-xxx" 命中这条
registry.register_pattern("qwen",           "qwen3");
```

`"Qwen3.5-30B".lowercase()` 含 `"qwen3"`，照样拿到 `Qwen3Parser`。

**不受影响的**：匹配不到任何 pattern 的模型（fallback 到 `PassthroughParser`，
`factory.rs:227-233`，原样透传），或压根不产 `<think>` 的非思考模型
（`reasoning_text` 为空 → `reasoning_content=None` → 两边哈希本来就一致）。

## 7. 迁到 Qwen3.5：需要新写一个 acc 模板

同样的 flag 依然需要，但周边假设变了。**两个失败模式已用 Qwen3.5-9B 实测确认**
（下方渲染结果均为 `apply_chat_template` 实际输出，非推断）。

先说 SkyRL 那边的情况：**SkyRL 至今没有 3.5 版 acc 模板**。
`skyrl/train/utils/templates/` 下只有 `qwen3_acc_thinking.jinja2` 一个文件；
SkyRL 有 9 个 qwen3.5 训练脚本，但全部是 `env_class=gsm8k` / `aime`（单轮数学），
没有一个设 `chat_template`。而所有设 `chat_template` 的脚本
（`examples/train_integrations/harbor/run_{harbor_gen,codecontest,codecontest_fully_async}.sh`）
都是 Qwen3-8B + 那一个 qwen3 模板。

**结论：这块没有上游参考实现可抄，多轮 + 3.5 + thinking 累积的组合 SkyRL 没做过。**

### 前提：3.5 官方模板 ≈ psrl 的 `qwen3.5_fixed.jinja`

实测两者只差一处：非首位 system message，官方 `raise_exception`，
psrl 版降级渲染成 user。所以下面用官方模板的实测结论对 `qwen3.5_fixed.jinja` 同样成立。

### (a) 3.5 在 prompt 里 prefill `<think>\n`，CoT 内联会导致标签不成对

```jinja
{# config/qwen3_acc_thinking.jinja2 —— 只在关闭 thinking 时注入空 think #}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}

{# qwen3.5_fixed.jinja / 3.5 官方 —— 开启时主动 prefill 开标签 #}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}     ← 这里
{%- endif %}
```

Qwen3 是模型自己吐 `<think>`，标签成对，塞回 `content` 没问题。Qwen3.5 的开标签在
prompt 里，所以模型输出是 `推理…</think>\n\n答案` —— **`content` 里只有孤立的 `</think>`，
没有开标签**。

用 `separate_reasoning=False` 下 terminus-2 的真实历史形态
（`content='COT1</think>\n\nA1'`）实测渲染：

```
# ACC-THINKING（当前 qwen3 用的模板）—— 无条件拼 content
'...<|im_start|>assistant\nCOT1</think>\n\nA1<|im_end|>\n...<|im_start|>assistant\n'
                          ^^^^^^^^^^^^^ 孤立 </think>，且末尾没有 prefill <think>

# OFFICIAL 3.5
'...<|im_start|>assistant\nA1<|im_end|>\n...<|im_start|>assistant\n<think>\n'
                          ^^ CoT 被 split 掉了       末尾有 prefill
```

两边都不对：
- acc 模板：渲染出**不成对的 `</think>`**，且句尾缺 prefill `<think>\n`，与 3.5 训练分布不符。
- 官方模板：`qwen3.5_fixed.jinja:90-98` 的 `content.split('</think>')` 会把 CoT
  反解进 `reasoning_content` 再受门控（见 (b)）—— 结果是 CoT 被丢掉。

### (b) 3.5 只保留最后一轮 thinking —— 实测确认

`qwen3.5_fixed.jinja:100-104` 的 `loop.index0 > ns.last_query_index` 门控。
两轮 assistant 实测（官方模板）：

```
'<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n
 <|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n<think>\nCOT2\n</think>\n\nA2<|im_end|>\n'

COT1 还在吗: False   ← 被剥掉
COT2 还在吗: True    ← 只留最后一轮
```

这正是第 4 节说的「破坏 strictly appending」：第 1 轮生成时 prompt 里有 COT1，
第 2 轮渲染时 COT1 消失 → 同一段 response token 在两次序列里不一致 → logprob 对不上。
在 psrl 的 TITO 单序列路径下这会直接炸（第 5 节），不是静默退化。

### 所以 3.5 版 acc 模板要满足

1. **无条件保留每轮 CoT**（去掉 `loop.index0 > ns.last_query_index` 门控），语义对齐
   qwen3_acc_thinking 的「strictly appending」。
2. **处理好 prefill 与内联 CoT 的配对**。这是 3.5 独有的新问题，qwen3 版没有。
   两条路线，需要实测定夺：
   - **A**：历史 assistant 渲染成 `<think>\n` + CoT + `</think>\n\n` + answer
     （即补上开标签，让历史里标签成对），末尾保留 prefill `<think>\n`。
     与 3.5 训练分布一致，但要求从 `content` 里可靠地 split 出 CoT。
   - **B**：不 prefill，让模型自己吐 `<think>`，退化成 qwen3 的形态。改动小，
     但偏离 3.5 的训练分布，需要验证模型是否稳定自发输出开标签。

倾向 A（对齐训练分布优先），但**必须先跑短 session 确认 `content` 里 `</think>`
的实际形态**（是否恒定只有一个、是否可能出现在 answer 正文里）再定。

### 附带

3.5 的 tool call 是 XML 风格（`<tool_call><function=name><parameter=k>`）而非 Qwen3 的
JSON 风格，terminus-2 的 parser 选择也要跟着换。

## 附：thinking 开关的判定链

供排查时参考。SMG 通过扫模板字符串推断 thinking 默认值：

`crates/tokenizer/src/chat_template.rs:80-112` `detect_thinking_toggle()`
- 模板不含 `enable_thinking` / `thinking ` 相关变量 → `ThinkingToggle::None`
- 含 `set thinking = false` / `set enable_thinking = false` → `DefaultOff`（DeepSeek V3.1）
- 其余 → `DefaultOn`（Qwen3、Qwen3.5、Nemotron、GLM-4.6/5、Kimi-K2.5）

然后 `should_mark_reasoning_started()`（`model_gateway/src/routers/grpc/utils/parsers.rs:20-28`）
决定 parser 是否以 `in_reasoning=true` 起步（对应模板已 prefill `<think>` 的情况）。
注意这条链只在 `separate_reasoning=true` 时才生效 —— 我们关掉后整段是 no-op。

## 8. `model_info.max_output_tokens` 不限制生成 —— 别当成 max_tokens

排查 Qwen3.5-9B eval 时踩过的坑，结论已在代码里核对：

**litellm chat 路径上，`model_info.max_output_tokens` 是纯元数据。** 全仓 grep
`harbor/llms/` 找不到任何地方把它塞进 chat-completions 请求。它只被两处读：

- `get_model_output_limit()` → 只走 **Responses API** 分支（`lite_llm.py:706`）
- cost accounting

所以 sciaccel 这条路（terminus-2 自己发 litellm 请求，不经
`get_session_sampling_params`）**没有 per-request `max_tokens`**，生成长度只由
服务端 `--max-model-len` 决定。

推论一：`lite_llm.py:432` 那个 `finish_reason == "length"` → `OutputLengthExceededError`，
撞的是**上下文窗口**，不是客户端 cap。实测 Qwen3.5-9B 每轮输出 219–549 tokens，
标称 cap 4096 从未生效 —— 把 `OutputLengthExceededError` 单独归一类会误导成
"调大 max_output_tokens"，实际毫无作用。

推论二：`OutputLengthExceededError` **不会终止 episode**。
`terminus_2.py:1113-1150` 的处理是：把截断文本作为 assistant message 存进
history，再追加一条 "ERROR!! NONE of the actions you just requested were
performed because you exceeded N tokens... break it into chunks"，然后
**递归重试** `_query_llm`。所以行为就是"截断 → 告知 → 继续推理"。

两个副作用：
- 截断文本 + 那条斥责都留在 transcript 里，每次截断都永久抬高上下文，
  把 episode 推向真正的 32768 墙。
- 那条递归**没有深度保护**（grep 无 retry counter），一直超长就一直递归
  直到上下文真的炸。
- `salvage_truncated_response` 只在 XML parser 有，但实测 44/44 条截断都断在
  推理中段（连命令块都没开），换 XML 也全部救不回来 —— 见下。

真正会终止 episode 的是 `ContextLengthExceededError`（`terminus_2.py:1014`）：
`enable_summarize=False` 时直接 re-raise，Harbor 跳过 `_run_verifier`，
`rewards` 为空。

**但 PSRL 侧不会丢掉它**（`agent_loop.py:147-166`）：判定为
`TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED`，**partial trajectory 照常训练**
（对齐 SkyRL harbor_generator）。而且只在 verifier 完全没跑出分时才回落 0 —
verifier 真跑出分数就保留，因为 ladder 对"交付了前缀"有部分分。

所以对 RL 的影响比想象的小：不是丢样本，而是"训练一条没走到交付的
partial trajectory + reward=0"。这个 label 本身是**正确的**（确实没交付），
属于真实信号，不是噪声。

**`parser_name="xml"` 救不了。** 已核对 `salvage_truncated_response`
（`terminus_xml_plain_parser.py:528`）：它要求截断文本里同时存在 `</commands>`
和 `</response>`。实测 44 条 overflow trial 的最后一条 agent message
**全部**断在推理中段（例：`...then search for all usages of these arrays.`），
0 条含完整命令块 —— 换 XML parser 后 salvage 依然全部返回 None。

真正的成因就是输入涨过窗口，没有别的：overflow 组与正常组的每轮输入
token 中位数是 18,231 vs 17,993，统计上无差别。最后一轮有的越过 32768、
有的没越过，纯看落点。所以能动的杠杆只有三个：更大的窗口、更少的轮数、
或每轮更少的输入（observation 已被 harbor 截到 10 KB，这条已经到底了）。

# Road To RL Infra

这是一个围绕 RL Infra / LLM Infra 的学习笔记仓库，主要记录推理调度、并行策略、以及 RL 训练链路相关的阅读与理解。

整体更偏向个人 study notes，而不是可直接运行的软件项目。适合用来快速回顾某个主题的核心概念、源码路径和关键词。

## 仓库结构

- `Inference-Study-Record/`：推理侧笔记，重点是调度、长上下文、并行与推理系统实现。
- `Train-Study-Record/`：训练 / RL 侧笔记，重点是 GRPO、rollout、loss 与参数更新链路。
- `skills/`：给 Claude Code 用的内部技能文档，这里目前主要是 GitHub Markdown / 数学公式写作规范。

## 笔记导航

### Inference

- [`Inference-Study-Record/Kimi Rollout DCP+MTP.md`](./Inference-Study-Record/Kimi%20Rollout%20DCP%2BMTP.md)
  讲 Kimi 相关推理笔记里 DCP（Decode Context Parallel）的核心直觉，重点解释长上下文下 KV cache 如何按序列维切分，以及 decode step 中多卡 attention 的张量形状与执行方式。
  `关键词`：DCP、Context Parallel、KV Cache、Decode、Attention、长上下文、多卡并行

- [`Inference-Study-Record/SGlang Schedule.md`](./Inference-Study-Record/SGlang%20Schedule.md)
  梳理 SGLang runtime 的调度主链路，从 Engine / TokenizerManager / Scheduler / DetokenizerManager 的关系，一直到 waiting queue、running batch、prefix cache、prefill/decode 切换与 event loop。
  `关键词`：SGLang、Scheduler、Prefill、Decode、KV Cache、Prefix Cache、Batch 调度、ZMQ IPC

### Training / RL

- [`Train-Study-Record/Megatron-GRPO-Study-Record.md`](./Train-Study-Record/Megatron-GRPO-Study-Record.md)
  总结 Megatron-LM 中 GRPO 训练的核心流程，包括 rollout 后的数据流、`ref_state_dict` 的作用、old/ref logprob、GRPO loss 组成以及 advantage 的组内标准化。
  `关键词`：Megatron-LM、GRPO、Rollout、ref_state_dict、KL、Entropy、Advantage、Optimizer Step

### Internal skill doc

- [`skills/github-md-math-writing/SKILL.md`](./skills/github-md-math-writing/SKILL.md)
  一个内部 Markdown 写作技能文档，主要约束 GitHub 上公式与说明文字如何安全、清晰地混排，适合写包含数学符号的 README 或学习笔记时参考。
  `关键词`：GitHub Markdown、Math、KaTeX、README、公式渲染、写作规范

## 从哪里开始

如果你主要关注：

- **推理系统 / 调度**：先看 `SGlang Schedule.md`
- **长上下文并行 / KV cache 压力**：先看 `Kimi Rollout DCP+MTP.md`
- **RL 训练 / GRPO 更新链路**：先看 `Megatron-GRPO-Study-Record.md`
- **GitHub 数学公式 Markdown 写法**：看 `skills/github-md-math-writing/SKILL.md`

# Road To RL Infra

围绕 RL Infra / LLM Infra 的学习笔记仓库，持续记录推理调度、并行策略、训练链路与工程工具的理解过程。

本仓库是 study notes，不是可直接运行的完整工程。目标是方便快速回顾一个主题的核心概念、源码入口和关键词。

## 仓库结构

- `Inference-Study-Record/`：推理侧笔记，关注调度、长上下文、并行与推理系统实现。
- `Train-Study-Record/`：训练 / RL 侧笔记，关注 GRPO、rollout、loss 与参数更新链路。
- `RL-Framework/`：框架相关笔记（当前已有 `slime.md` 占位，后续补充）。
- `Tools/`：工程工具学习记录（如 Nsight Systems 分析）。
- `skills/`：内部写作/技能文档（当前为 GitHub Markdown 数学公式写作规范）。

## 内容导航

### Inference

- [`Inference-Study-Record/SGlang Schedule.md`](./Inference-Study-Record/SGlang%20Schedule.md)
  - SGLang runtime 调度主链路：Engine / TokenizerManager / Scheduler / DetokenizerManager 的协作关系。
  - 覆盖 waiting queue、running batch、prefix cache、prefill/decode 切换、event loop。
  - 关键词：`SGLang` `Scheduler` `Prefill` `Decode` `KV Cache` `Batch 调度`

- [`Inference-Study-Record/Kimi Rollout DCP+MTP.md`](./Inference-Study-Record/Kimi%20Rollout%20DCP%2BMTP.md)
  - Kimi 推理相关笔记，重点是 DCP（Decode Context Parallel）的直觉与张量形状。
  - 解释长上下文下 KV cache 的序列维切分、多卡 decode attention 的执行路径。
  - 关键词：`DCP` `Context Parallel` `KV Cache` `Decode` `Attention`

### Training / RL

- [`Train-Study-Record/Megatron-GRPO-Study-Record.md`](./Train-Study-Record/Megatron-GRPO-Study-Record.md)
  - Megatron-LM 中 GRPO 训练流程拆解：rollout -> logprob 计算 -> loss 组成 -> optimizer step。
  - 包含 `ref_state_dict`、old/ref logprob、advantage 组内标准化等关键点。
  - 关键词：`Megatron-LM` `GRPO` `Rollout` `KL` `Entropy` `Advantage`

### Tools / Framework

- [`Tools/Nsys.md`](./Tools/Nsys.md)
  - NVIDIA Nsight Systems（NSYS）基础使用与可视化阅读笔记。

- [`RL-Framework/slime.md`](./RL-Framework/slime.md)
  - 预留框架学习文档（当前为空，后续补充）。

### Internal docs

- [`skills/github-md-math-writing/SKILL.md`](./skills/github-md-math-writing/SKILL.md)
  - GitHub Markdown 数学公式写作规范，适合写带公式的 README / 技术笔记时参考。

## 推荐阅读顺序

1. 推理调度入门：`SGlang Schedule.md`
2. 长上下文并行：`Kimi Rollout DCP+MTP.md`
3. 训练更新链路：`Megatron-GRPO-Study-Record.md`
4. 工具实践：`Tools/Nsys.md`

## 维护说明

- 文档以中文为主，英文术语保留原名。
- 每篇笔记尽量保持：背景 -> 主链路 -> 关键词 -> 参考入口 的结构，便于后续回看和补充。

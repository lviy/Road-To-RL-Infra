# Megatron RL（GRPO）学习记录

## 1. 主题与目标
本记录总结了在 `Megatron-LM` 中，RL 训练（GRPO）从 rollout 到权重更新的核心机制，重点包括：
- 权重更新主链路
- `ref_state_dict` 的作用
- GRPO 各项公式含义（比率/截断、KL、熵、IS 修正）
- 一轮 rollout 后的数据流
- 组内标准化优势（advantage）的原因

---

## 2. 权重更新主流程（从一轮 iteration 看）

1. 准备参考策略 $`\pi_{ref}`$
- RL 训练会构建固定参考权重 `ref_state_dict`（通常来自预训练/SFT权重），用于 KL 正则约束。

2. 采样 rollout（得到 $`\pi_{old}`$ 数据）
- 根据 `grpo_iterations` 与缓存策略，决定本轮是否重新采样。
- 若使用独立 inference model，会先把训练模型权重同步/交换到推理模型再采样。

3. 预处理用于更新的数据
- 将 rollout 转成训练张量（tokens、mask、position_ids 等）。
- 计算两套关键 logprob：
  - `old_logprobs`：采样策略侧（$`\pi_{old}`$）
  - `ref_logprobs`：参考策略侧（$`\pi_{ref}`$）

4. 前向与损失
- 计算当前策略 $`\pi_\theta`$ 的 logprob。
- 组成 GRPO 单 token loss（policy clip + KL + entropy + 可选 IS 修正）。

5. 反向与参数更新
- 标准 Megatron 训练步：`zero_grad -> forward/backward -> optimizer.step -> lr_scheduler.step`。
- `optimizer.step()` 内部（混精场景）一般是：
  - copy model grads -> main grads
  - unscale & inf check
  - clip grad
  - inner optimizer step
  - main params copy 回 model params

---

## 3. 为什么需要 `ref_state_dict`

问题：更新权重不是只看当前状态吗？

结论：在 GRPO/PPO 类目标里，更新不仅依赖当前策略 $`\pi_\theta`$，还依赖：
- $`\pi_{old}`$：用于离策略比率与 clipping（保证步子不过猛）
- $`\pi_{ref}`$：用于 KL 正则（防止策略漂移过远）

没有 $`\pi_{ref}`$（或 KL 系数过低）时，模型容易过度追 reward，出现能力退化、风格漂移或 reward hacking。

---

## 4. GRPO 公式与直觉

### 4.1 比率与截断（policy clip）

```math
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
```

```math
\tilde r_t(\theta)=clip(r_t(\theta), 1-\epsilon_{low}, 1+\epsilon_{up})
```

- $`r_t`$ 衡量“当前策略相对采样策略”的概率变化。
- clip 的作用：限制单步更新幅度，防止个别样本造成过大梯度冲击。

#### 说白了，$`r_t`$ 就是在现在状态与之前的状态在当前 token 做当前动作的概率变化之比。为什么要 clip 呢？因为要让这个比值拥有一个上下界，不要让它一步更新太猛。
### 4.2 参考 KL 项

设
```math
\Delta_t=\log \pi_{ref}(a_t|s_t)-\log \pi_\theta(a_t|s_t)
```

Megatron 实现使用：
```math
KL_t=\exp(\Delta_t)-\Delta_t-1
```

性质：
- 始终非负
- 当 $`\pi_\theta`$ 与 $`\pi_{ref}`$ 一致时为 0
- 偏离越大惩罚越强

### 4.3 熵项

```math
H_t=-\pi_\theta(a_t|s_t)\log\pi_\theta(a_t|s_t)
```

在 loss 中通常是 $`-\alpha H_t`$，最小化 loss 等价于鼓励更高熵（防止策略过早塌缩到极尖分布）。

#### 因为 $`-x \log x`$ 是一个在 $`(0,1]`$ 上大于 0 的函数，趋于 0 和 1 时都是 0，这样可以防止它选择太极端的概率（0 或 1）。

### 4.4 可选 IS 修正（inference correction）

```math
w_t=\exp(\log \pi_{old}(a_t|s_t)-\log \pi_{inf}(a_t|s_t))
```

可选截断：
```math
w_t = \min(w_t, c)
```

作用：不是单纯“变小”，而是将由 $`\pi_{inf}`$ 采样得到的样本重新加权到更接近 $`\pi_{old}`$ 目标分布。
#### 因为更新是从 $`\pi_{ref}`$ 做更新的，但是推理出的 reward 是从 $`\pi_{inf}`$ 的状态推出来的，两者拥有偏差，通过对 $`\pi_{inf}`$ 加一些修正让其更加接近 $`\pi_{ref}`$ 的分布。
### 4.5 单 token 损失与 batch 聚合

```math
\mathcal L_t(\theta)=
-w_t \min(r_t(\theta)A_t, \tilde r_t(\theta)A_t)
+\beta KL_t
-\alpha H_t
```

```math
\mathcal L(\theta)=
\frac{\sum_t m_t \mathcal L_t(\theta)}{\sum_t m_t}
```

其中 $`m_t`$ 是 loss mask。

---

## 5. 符号解释

- $`\pi_\theta(a_t|s_t)`$：当前策略在状态 $`s_t`$ 下选择动作/token $`a_t`$ 的条件概率。
- $`s_t`$：当前上下文状态（prompt + 已生成前缀）。
- $`a_t`$：当前步生成的 token。
- $`\pi_\theta(a_t|s_t)`$ 取值区间：理论上 $`(0,1]`$。

当策略塌缩为尖锐分布时：
- 少数 token 概率接近 1
- 大多数 token 概率接近 0
- 熵趋低，探索能力下降

---

## 6. 一轮 rollout 后的数据流（示例）

假设：
- `grpo_prompts_per_step=64`
- `grpo_group_size=16`
- `global_batch_size=64`

则：
```math
N_{rollout}=64 \times 16=1024
```

每次 collection 可支持全局更新步数：
```math
N_{updatesPerCollection}=\frac{64 \times 16}{64}=16
```

若 `grpo_iterations=\mu`，则该批 rollout 可复用：
```math
N_{updatesTotal}=\mu \cdot \frac{64 \times 16}{64}
```

数据流摘要：
1. 采样得到 64 组，每组 16 条 trajectory
2. 组内 reward 标准化得到 advantage
3. 按 DP rank 切分样本
4. 构建训练输入张量
5. 预计算 `old_logprobs` 与 `ref_logprobs`
6. 进入训练循环，按 microbatch 进行 forward/backward
7. 调用 `optimizer.step()` 完成参数更新

---

## 7. Advantage（$`A_t`$）怎么来

Megatron 这个 GRPO 路径中，$`A_t`$ 不是来自 value network，而是来自组内标准化 reward：

```math
A_{g,i}=\frac{r_{g,i}-\mu_g}{\sigma_g+10^{-4}}
```

多轮对话时，会按 turn 展开/重复映射到训练样本。

---

## 8. 为什么做组内标准化

不仅仅是“防梯度爆炸”，核心是：

1. 降低方差，稳定更新
- 控制 advantage 尺度，减少极端样本主导。

2. 消除跨 prompt 的绝对奖励偏置
- 不同 prompt 难度不同，组内标准化后模型学“同 prompt 下谁更优”。

3. 在无 critic 设定下提供 baseline 效果
- 等价于以组均值为基线做中心化，和 PPO advantage 去中心化思想一致。

---

## 9. 一句话总结

Megatron 的 GRPO 更新不是“只看当前策略”单点优化，而是通过 $`\pi_{old}`$（稳定步长）+ $`\pi_{ref}`$（防漂移）+ advantage 标准化（降方差）共同约束，再进入 Megatron 标准反传与优化器更新链路，完成稳定的 RL 微调。

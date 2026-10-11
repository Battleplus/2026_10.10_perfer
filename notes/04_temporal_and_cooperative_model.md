# 单元 4：两个依赖维度怎样进入 MAPT

## 先看数据

一个偏好 batch 的核心张量是：

```text
observations: [B, T, N, O]
actions:      [B, T, N, 1]     # 离散动作编号，进入模型后 one-hot
timesteps:    [B, T]
attn_mask:    [B, T, N]        # 真实历史为 1，填充位置为 0
```

`B` 是比较样本数，`T` 是轨迹片段长度，`N` 是智能体数，`O` 是观测宽度。

## 论文图 1 / 图 3 的思路

1. 对每个时刻，把所有单位的观测放在一起做协作上下文编码；动作也映射到嵌入空间。
2. CPA（cooperative-aware preference attention）沿智能体维度对每个时刻建模依赖，为各智能体产生增强奖励，再聚合。
3. TPA（temporal-aware preference attention）沿时间维度建模哪些时刻重要。
4. 将两个片段的加权奖励送进二选一偏好概率，计算交叉熵。

核心意图是同时看“队友之间如何配合”和“整段经历中何时发生关键行为”。

论文 Eq. (4) 可以读成在同一时刻让单位 `i` 汇总所有单位的 value：

```text
α_{i,j,t} = softmax_j(q_{i,t}ᵀ k_{j,t})
r_hat_{i,t} = Σ_j α_{i,j,t} v_{j,t}
```

Eq. (5) 让每个时间 query `k` 汇总各时刻奖励，`x_k = Σ_t β_{k,t} r_hat_t`；Eq. (6) 再对所有 query 的权重取平均，`w_t = (1/T) Σ_k β_{k,t}`，从而把片段分数写成 `Σ_t w_t r_hat_t`。注意 `α` 在单位轴上归一化，`β` 在时间轴上归一化。

## 作者实现的数据流

`MultiTransRewardDivideModel.forward` 中可以跟到这条路径：

```text
[B,T,N,O] 与 [B,T,N,1]
          ↓ 折叠 B、T
[B*T,N,O] 与 [B*T,N,1]
          ↓ AgentLayer：观察编码 + 动作编码/融合
[B*T,N,D]
          ↓ 改排为每个单位一条时间序列
[B*N,T,D] → TimeLayer（默认因果掩码） → [B,T,N,D]
          ├─ agent_weight_head：同一时刻跨 N 注意力
          └─ time_weight_head：同一单位跨 T 注意力
                           ↓
每步每单位奖励 [B,T,N,1]
```

默认公开 Transformer 分支设 `use_lstm=False`。这里的开关指奖励模型时间层：默认配置见 `code/MAPT/mat/algorithms/reward_model/models/MultiPrefTransformer.py:9-26`，分支选择见 `code/MAPT/mat/algorithms/reward_model/models/encoder_decoder_divide.py:284-319`；它不会关闭 MAPT 奖励模型，也不是 MAT 策略网络的开关。

论文 Fig. 1/3 与 Eq. (4)-(6) 描述合作维与时间维的依赖及加权聚合，但公开 commit 不应被说成与图逐层等同。实现先由 `AgentLayer` 编码观测/动作，再把每个 agent 排成时间序列交给 `TimeLayer`，之后分别运行加权 agent head 和 time head，最后将该加权时间输出加到线性奖励投影。可核查位置：`encoder_decoder_divide.py:162-207`（AgentLayer）、`:284-319`（TimeLayer）、`:321-432`（组合与两个加权头）。偏好训练时各 agent 和 time 的序列分数聚合与交叉熵在 `MultiPrefTransformer.py:73-110`。

要区分两种时间注意力 mask。`SelfAttention.forward` 在 `encoder_decoder_divide.py:41-55` 用 `torch.tril` 构造因果掩码；但同文件 `:418-431` 的 weighted time head 用全 1 的 `torch.ones` 矩阵。`ops.py:129-166` 对 True 位置保留 attention logits，因此窗口中每个 query 都能读该窗口全部位置。偏好训练会对整段轨迹计算这些 query 的加权分数；策略推理则给模型最近的滑动历史窗口并取最后时刻的预测（`mat/utils/pref_reward_assistant.py:185-205`），所以当前输出只依赖截至当前步可见的窗口。两处 mask 语义不同，不能简化成“整套时间建模都因果”。

## 代码定位

- `code/MAPT/mat/algorithms/reward_model/models/encoder_decoder_divide.py:41-55`：`SelfAttention.forward` 的因果 mask；`:162-207`：AgentLayer；`:284-319`：TimeLayer；`:321-432`：模型组合、加权 agent/time heads，其中 `:418-431` 是窗口内全位置可见的 weighted time head。`ops.py:129-166` 展示布尔 mask 为 True 时保留对应 attention logits。
- `code/MAPT/mat/algorithms/reward_model/models/MultiPrefTransformer.py:9-26`：默认奖励模型配置（含 `use_lstm=False`）；`:73-110`：训练偏好分数与交叉熵。
- `code/MAPT/mat/utils/pref_reward_assistant.py:185-205`：策略阶段组装历史窗口、调用奖励模型并取当前时刻输出。

## 小实验

先用随机初始化的小模型对一个真实环境 rollout 切出的 batch 做 forward，确认输入输出形状、有限数值和梯度存在。之后在实际模型调用处记录 `agent_weight_head` 与 `time_weight_head` 的权重归一化情况；比较真实权重和单独把一个注意力模块替换成均匀权重后的奖励预测差。

均匀权重检查只诊断已训练模型的单次预测影响，不重新训练网络，因此不能写成论文里的完整移除模块消融。

## 理解检查

1. `[B,T,N,O]` 的 `N` 和 `T` 分别是哪两个关系轴？
2. 为什么要把 `[B,T,N,D]` 改排成 `[B*N,T,D]` 才送入按单位处理的时间层？
3. 若统一时间平均，短片段中的关键动作可能被怎样稀释？

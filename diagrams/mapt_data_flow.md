# MAPT 数据流与张量形状

```mermaid
flowchart LR
  A[真实轨迹对<br/>obs [B,T,N,O]<br/>act [B,T,N,1]] --> B[AgentLayer<br/>观测编码 + 动作编码<br/>按每个时刻跨 N 建模]
  B --> C[时间序列整理<br/>[B*T,N,D] → [B*N,T,D]]
  C --> D[TimeLayer<br/>时间位置 + 因果编码<br/>[B*N,T,D]]
  D --> E[逐步逐智能体特征<br/>[B,T,N,D]]
  E --> F[Agent 权重头<br/>每个 t 沿 N 注意力]
  E --> G[Time 权重头<br/>每个 agent 沿 T 注意力]
  F --> H[每步奖励投影与融合]
  G --> H
  H --> I[奖励分数<br/>[B,T,N,1]]
  I --> J[偏好训练：沿 N、T 聚合<br/>每段 [B,1]]
  J --> K[两段 logits [B,2]<br/>CE 对比标签]
  I --> L[策略 rollout 中取最近一步奖励<br/>滚动历史窗口]
  L --> M[MAT / PPO 更新策略]
```

符号：`B` batch，`T` 片段/历史长度，`N` 单位数，`O` 观测宽度，`D` 隐藏维度。策略推理历史的 `T` 由 checkpoint 序列长度和配置决定。

这张图概括论文意图和公开代码接口；作者 commit 的实际 attention heads 位于 `MultiTransRewardDivideModel.forward`，代码还把两个偏好权重头的输出加到线性逐步奖励投影上。需要做结构性结论时应以代码为准并说明此差异。

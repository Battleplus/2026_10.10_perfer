# 复现补丁记录

## 基线

- 官方仓库：`https://github.com/catezi/MAPT`
- 本次固定 commit：`0b4ef2712995681febca7631f9a27e1b0dccccbf`
- 官方基线源码本地归档：`code/MAPT`
- 服务器执行副本（含本项目复现补丁）：`/home/user1/projects/amb-mapt-study/MAPT`
- 模型目标：公开 Transformer 分支 `use_lstm=False`。

## 已观察到的复现约束

| 项目 | 源码行为 | 本项目处理 |
| --- | --- | --- |
| 依赖文件 | README 写 `requirements.txt`，实际文件名 `requirments.txt` | 独立环境安装实际所需的核心包；安装命令和版本会记录 |
| 示例配置 | 示例奖励/策略脚本设 `use_lstm=True` | 教学 MAPT Transformer 配置设 `False`，结果单列 |
| 路径/GPU | policy shell 含作者本机目录并设 `CUDA_VISIBLE_DEVICES=0` | 使用项目绝对路径；仅映射授权 GPU1 |
| checkpoint | 仅保存权重与 `seq_len` | 训练记录完整配置，加载时显式核对维度与模型开关 |
| episode 历史 | `clear_done` 曾为每个完成环境取同一 `start_obs[-1]`；首轮修复保存了 `his_next_obs` 的视图，随后清零也清掉待恢复值 | 按环境索引恢复并先复制起始观测；`tutorials/check_history_reset.py` 的不同 sentinel 回归检查在本地和服务器均通过 |
| checkpoint 配置 | 原保存仅含权重和序列长度 | 已加入配置与维度元数据，加载时核对关键配置；GPU 保存/重载预测一致性检查通过，最大差为 `0.0` |
| 时间偏好头 | 普通时间编码器用因果 mask；weighted time head mask 为全 True | 作为实现细节记录，诊断图分开观察 |
| Transformer 与 LSTM 分支 | 奖励模型支持 Transformer 和 LSTM 时间编码；官方 3m 示例脚本启用 `use_lstm=True` | 本项目选用 `use_lstm=False` 的 Transformer 奖励模型分支；默认和分支选择位置分别见 `code/MAPT/mat/algorithms/reward_model/models/MultiPrefTransformer.py:9-26`、`encoder_decoder_divide.py:284-319`。该开关不控制 MAT 策略网络 |
| 论文图与公开实现 | Fig. 1/3、Eq. (4)-(6) 展示合作/时间偏好注意力的建模意图 | 公开实现的具体数据流是先 AgentLayer、再逐 agent 的 TimeLayer、最后两个加权头和线性投影；见 `encoder_decoder_divide.py:162-207,284-319,321-432`。不将其描述为与论文示意逐层等同 |
| 时间 attention mask | `SelfAttention.forward` 有因果下三角 mask；weighted time head 单独调用全 1 mask | 因果层位置为 `encoder_decoder_divide.py:41-55`；weighted time head 的全窗口可见调用为 `:418-431`，`ops.py:129-166` 定义 True mask 保留 logits。偏好训练按整段轨迹评分，策略推理则由 `pref_reward_assistant.py:185-205` 输入最近窗口并取末时刻输出；两种 mask 语义分开说明 |
| PyTorch 注意力 mask | PyTorch 1.10 不接受上游 `torch.where(mask, tensor, -10000.0)` 的混合参数写法 | 改为 `masked_fill`；GPU 奖励模型 smoke test 已通过 |
| 偏好数据 | README 的分享链接已过期；官方 loader 按样本顺序切分 | 采集脚本已上传服务器并从真实 SMAC rollout 重采；当前 3m 数据为 20 局、22 段、30 对，按 episode 隔离为 24 训练对和 6 验证对 |
| Python 依赖 | PySC2 3.0 与 protobuf 5.29 不兼容 | 仅在独立 `mapt_study` 环境将 protobuf 固定为 3.20.3 |

修复记录只在完成可复现验证后追加，不把计划中的检查写成已修复。

## 已完成验证

- 服务器 `3m` reset/step 通过；seed 1 的随机合法动作运行 5 步，观测、共享状态和动作掩码形状分别为 `(3, 64)`、`(3, 81)`、`(3, 9)`。
- GPU1 CUDA 梯度计算通过，检测到 RTX 2080 Ti；项目没有占用 GPU0。
- MAPT 奖励模型输出 `(2, 4, 3, 1)`，反向梯度范数 `2.16865`，保存/重载后预测最大差 `0.0`。
- `tutorials/02_collect_script_preferences.py` 已上传并重新生成当前 3m 偏好数据；数据按 episode 隔离，24 对用于训练、6 对用于验证。标签由项目脚本教师依据累计 SMAC 任务回报产生，不是人工反馈或论文作者数据。
- 3m seed 1 的 GT、MR、MAPT 策略队列已分别完成 100k 与 1M 配置；1M 队列在 GPU1 串行运行并以 `QUEUE_COMPLETE` 结束。预算、实际 transitions、最终 eval 和 wall time 见 [实验日志](experiment_log.md) 与生成报告；每组只有一个 seed，不支持论文级统计结论。

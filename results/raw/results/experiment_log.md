# 实验日志

## 环境与输入

| 项目 | 状态 |
| --- | --- |
| 论文 | AAAI 2024, MAPT, 9 页 |
| 官方代码 | `catezi/MAPT`, commit `0b4ef2712995681febca7631f9a27e1b0dccccbf` |
| 服务器 | Ubuntu 22.04；GPU1 授权用于本项目，GPU0 保留给现有任务 |
| 偏好数据 | 官方 AnyShare 链接返回 `This address has expired`；当前 3m 数据由项目脚本教师从 20 局真实 rollout 生成，共 22 段、30 对，按 episode 隔离为 24 训练对和 6 验证对；标签计数 18 / 12 / 0 |
| SMAC / SC2 | 服务器已配置 SC2 4.10 和 `3m`、`3s5z` 地图；`3m` reset/step 和 5 步随机合法动作验证通过 |
| Python | 独立 Conda env `mapt_study`，Python 3.8 |

## 训练结果

| 方法 | 地图 | Seed | 配置预算 / 实际 transitions | 最后 eval step | eval return / win rate | wall time | 偏好 train/eval loss | stdout / 标量目录 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| GT | 3m | 1 | 100,000 / 99,840 | 96,480 | 20.000 / 1.00 | 209.25s | 不适用 | `results/raw/results/policy_100k/gt_mat_seed1_100k.stdout.log`; `results/raw/gt_3m_seed1_100k_eval_returns.csv` |
| MR | 3m | 1 | 100,000 / 99,840 | 96,480 | 0.000 / 0.00 | 224.28s | 不适用 | `results/raw/results/policy_mr_100k/mr_seed1_100k.stdout.log`; `results/raw/mr_3m_seed1_100k_eval_returns.csv` |
| MAPT | 3m | 1 | 100,000 / 99,840 | 96,480 | 6.225 / 0.00 | 240.35s | 不适用 | `results/raw/results/policy_mapt_100k/mapt_seed1_100k.stdout.log`; `results/raw/mapt_3m_seed1_100k_eval_returns.csv` |
| GT | 3m | 1 | 1,000,000 / 999,840 | 984,480 | 17.6658 / 0.80 | 1,895.44s | 不适用 | `results/raw/results/policy_gt_1m/gt_seed1_1m.stdout.log`; `results/raw/results/policy_gt_1m/results/StarCraft2/3m/mat/gt_seed1_1m/run1/logs/summary.json` |
| MR | 3m | 1 | 1,000,000 / 999,840 | 984,480 | 4.7671 / 0.00 | 2,114.13s | 不适用 | `results/raw/results/policy_mr_1m/mr_seed1_1m.stdout.log`; `results/raw/results/policy_mr_1m/results/StarCraft2/3m/mat/mr_seed1_1m/run1/logs/summary.json` |
| MAPT | 3m | 1 | 1,000,000 / 999,840 | 984,480 | 5.9178 / 0.00 | 2,333.58s | 不适用 | `results/raw/results/policy_mapt_1m/mapt_seed1_1m.stdout.log`; `results/raw/results/policy_mapt_1m/results/StarCraft2/3m/mat/mapt_seed1_1m/run1/logs/summary.json` |

GT、MR、MAPT 的 1M 运行均正常完成 2083 次完整更新，每次 8 个 rollout、每个 60 步，因此各有 `2083 × 8 × 60 = 999,840` transitions。三者 stdout 最后一次周期进度样本均为 998,880（update 2080/2083）；最后一次 TensorBoard eval 均为 step 984,480。配置预算、完整更新实际步数、stdout 采样与 eval step 是不同口径。

## 奖励模型短训

| 模型 | 数据 | Epoch 0 train/eval | Epoch 1 train/eval | Epoch 2 train/eval | 策略 checkpoint |
| --- | --- | --- | --- | --- | --- |
| MR | 3m 脚本教师偏好，24/6 对 | 0.7165 / 0.6457 | 0.6242 / 0.6091 | 0.5549 / 0.5747 | `results/raw/results/reward_short/smac/3m/MR/mr_seed1_short/models/reward_model_2.pt` |
| MAPT | 3m 脚本教师偏好，24/6 对 | 1.9820 / 0.0668 | 0.2142 / 0.4653 | 0.2154 / 1.2251 | `results/raw/results/reward_short/smac/3m/MultiPrefTransformerDivide/mapt_seed1_short/models/reward_model_0.pt` |

## 3s5z 注意力诊断

由 `tutorials/03_diagnose_3s5z_attention.py` 在 CPU 上完成 12 局真实 rollout、33 段和 60 个脚本教师偏好对（48 train / 12 validation，标签 30 / 30 / 0）。输入为 8 agents、252 维观测、14 个离散动作和 16 步窗口。最佳 MAPT 为 epoch 1（train/eval loss `1.3765/1.4124`）。同一段轨迹的累计预测为 baseline `-2.7622`、协作权重均匀化 `-1.6646`、时间权重均匀化 `8.6342`。该单 seed 小样本结果只作探索性敏感度诊断，不是论文消融或论文结果。数据和模型工件在 `results/raw/results/diagnostics/3s5z_attention/`。

## 1M 队列状态

1M 队列使用 GPU1 串行运行 GT、MR、MAPT，三组均以 exit code 0 完成。MAPT 运行于 13:01:45–13:40:39 +08，wall time 为 `2333.58s`；最终 `summary.json` 的 step `984,480` eval return 为 `5.917808219178082`，win rate 为 `0.0`。完整 stdout、模型、summary 和 TensorBoard 事件已同步到 `results/raw/results/policy_mapt_1m/`。队列日志：`/home/user1/projects/amb-mapt-study/results/policy_1m_queue/queue.log`。三组均是单 seed 教学运行，不代表论文的多 seed 结论。

## 操作记录

- 2026-10-11：检查论文 PDF 并渲染第 3、4 页，公式与 MAPT 结构可读。
- 2026-10-11：本地固定官方 commit，源码归档 SCP 到独立服务器项目。
- 2026-10-11：用户级 Conda 清华镜像访问 conda-forge 返回 403；通过命令级 `--override-channels -c https://repo.anaconda.com/pkgs/main` 创建独立 Python 3.8 环境，没有改全局配置。
- 2026-10-11：在 `mapt_study` 中安装论文代码核心依赖，包括 PyTorch 1.10.2+cu113、torchvision 0.11.3+cu113、NumPy 1.23.5、Gym 0.12.4、Transformers 4.18.0 和 PySC2 3.0.0；PySC2 与 protobuf 5.29 不兼容，将独立环境 protobuf 固定到 3.20.3。
- 2026-10-11：SC2 4.10 解压并安装 `3m`、`3s5z` 地图；`3m` reset 后观测为 `(3, 64)`、共享状态为 `(3, 81)`、动作掩码为 `(3, 9)`。seed 1 随机合法动作完成 5 步。
- 2026-10-11：设置 `CUDA_VISIBLE_DEVICES=1` 后，PyTorch 1.10.2+cu113 识别到单张 RTX 2080 Ti 并完成 CUDA 梯度计算；GPU0 未用于本项目。
- 2026-10-11：历史重置哨兵检查在本地和服务器均通过。首轮修复曾保存 `his_next_obs` 的数组视图，清零时会同时清掉待恢复数据；最终改为复制起始观测并按环境索引恢复。
- 2026-10-11：MAPT 奖励模型 GPU smoke test 通过：前向输出形状 `(2, 4, 3, 1)`，梯度范数 `2.16865`，checkpoint 保存再加载后预测最大差为 `0.0`。这是模型/序列化检查，不是训练结果。
- 2026-10-11：较早的 228 对 3m 脚本教师数据没有 episode 隔离，不用于当前奖励模型；当前使用上面列出的 30 对隔离数据。
- 2026-10-11：GT/MR/MAPT 100k 策略比较、MR/MAPT 三 epoch 奖励短训及 3s5z CPU 诊断已完成；GT/MR/MAPT 1M 队列均以 exit code 0 完成，实际步数、最后 eval 和 wall time 已核验并同步。

此表只记录真实训练完成后的策略指标；不能用启动中的进程、模型 smoke test 或语法检查填成绩。奖励偏好标签由本项目脚本教师按两段轨迹的累计 SMAC 任务回报比较产生，不是人工反馈或作者数据。以上结果均是单 seed 教学复现，不代表论文的多 seed 结论。

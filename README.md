# MAPT 论文复现与学习记录

论文：Zhu et al., “Decoding Global Preferences: Temporal and Cooperative Dependency Modeling in Multi-Agent Preference-Based Reinforcement Learning”, AAAI 2024。

本仓库包含论文学习笔记、实验入口、结果和复现补丁。MAPT 上游仓库没有根目录许可证，因此这里不重新发布其完整源码；运行代码时从上游固定 commit 检出，再应用本项目的小补丁。上游地址、commit 和改动说明见[补丁记录](results/patch_log.md)及[第三方声明](THIRD_PARTY_NOTICES.md)。

## 学习路径

1. [SMAC3m 环境、观测、动作、奖励和轨迹](notes/01_smac3m_environment.md)
2. [为什么多智能体偏好学习需要奖励模型](notes/02_why_preferences.md)
3. [论文偏好概率与交叉熵](notes/03_preference_probability.md)
4. [MAPT 的协作维与时间维张量、注意力和数据流](notes/04_temporal_and_cooperative_model.md)
5. [偏好奖励模型训练与 MAT 策略训练](notes/05_reward_and_policy_training.md)
6. [小规模对照、真实环境奖励曲线和注意力诊断](notes/06_experiments_and_diagnostics.md)

每单元给出直觉、公式、代码位置、小实验和理解检查。练习答案留给学习者；日志中的结果只代表实际运行过的部分。

## 学习状态

目前没有单元的理解检查得到用户作答并核验，因此没有单元标记为已掌握。下一次学习互动从单元 1 开始；每单元先回答理解检查并确认理解，再进入下一单元。文件、图表和实验运行记录不代表用户已经掌握对应内容。

## 开始

先读第一单元。若要运行环境教程，需要 Linux、Python、SMAC、StarCraft II 4.10 和 `3m` 地图。先在仓库根目录检出补丁对应的上游版本：

```bash
git clone https://github.com/catezi/MAPT.git MAPT
cd MAPT
git checkout 0b4ef2712995681febca7631f9a27e1b0dccccbf
git apply ../patches/MAPT_local_changes.patch
cd ..
```

设置本机 Python 与 StarCraft II 路径后运行：

```bash
export MAPT_ROOT="$PWD/MAPT"
export PYTHON_BIN="$(command -v python)"
export SC2PATH="/path/to/StarCraftII"
"$PYTHON_BIN" tutorials/01_smac3m_env.py --steps 5
```

入口读取 `MAPT_ROOT` 指向的源码并启动原始 3m 环境。StarCraft II 地图或依赖未就绪时会由环境报出具体错误，不会退回到合成观测。

检查并行环境 episode 历史重置时，从项目根目录运行 `"$PYTHON_BIN" tutorials/check_history_reset.py`。这是带不同 sentinel 的缓冲区索引检查，不是 SMAC 轨迹。

## 实验命令

在已安装 PyTorch、SMAC 和 SC2 4.10 的 Linux 环境中，从项目根目录运行小样本 3s5z 诊断。脚本会隐藏 CUDA 并把 PyTorch 线程限制为 1；为保留现有诊断产物，示例使用一个新的输出目录。

```bash
cd /path/to/this-repository
MAPT_ROOT="$PWD/MAPT" SC2PATH="/path/to/StarCraftII" \
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PYTHON_BIN" tutorials/03_diagnose_3s5z_attention.py \
  --episodes 12 --window 16 --pair-limit 60 --epochs 3 --batch-size 8 \
  --torch-threads 1 --seed 1 --output-dir results/diagnostics/3s5z_attention_rerun
```

汇总报告和曲线：[实验报告](results/study_report/report.md)、[策略评估曲线](results/study_report/policy_eval.png)、[奖励损失曲线](results/study_report/reward_loss.png)。GT、MR、MAPT 三组 1M 策略运行均已完成，原始 stdout、summary 和 TensorBoard 事件已同步到 `results/raw/results/`；三组比较见报告。结果是 seed 1 的教学运行，不代表论文的多 seed 结论。

策略基准脚本可按需重跑；必须提供与 MAPT 配套的环境、依赖及 MR/MAPT 奖励模型 checkpoint。该 checkpoint 没有包含在本仓库中。可通过 `MR_REWARD_MODEL`、`MAPT_REWARD_MODEL` 提供路径；默认使用仓库下的 `results/reward_short/...` 约定位置。脚本支持 `MAPT_ROOT`、`PYTHON_BIN`、`SC2PATH` 和 `CUDA_VISIBLE_DEVICES` 环境变量。

```bash
cd /path/to/this-repository
MAPT_ROOT="$PWD/MAPT" PYTHON_BIN="/path/to/python" SC2PATH="/path/to/StarCraftII" \
bash scripts/run_policy_queue.sh
```

配置预算 `1,000,000` 不等于训练实际步数。队列每次完整更新有 8 个 rollout、每个 60 步，2083 次更新实际为 `999,840` transitions；每次评估为 5 局，stdout 周期采样步数与 TensorBoard 最后评估步数需分别报告。当前 GT、MR、MAPT 比较各只有 seed 1，不支持论文级统计结论。

## 目录

- `notes/`：逐单元学习笔记
- `tutorials/`：可运行的小实验
- `diagrams/`：模型数据流与张量形状图
- `results/`：配置、实验日志和图表
- `patches/`：针对固定上游 commit 的本地改动
- `results/raw/`：已同步的原始日志、指标、部分偏好/诊断文件及策略 checkpoint；不包含完整训练环境、地图或短训奖励模型 checkpoint

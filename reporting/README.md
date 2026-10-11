# 本地训练日志报告器

只读解析同步到本机的原始训练输出或 TensorBoard 标量导出，不启动训练。输入清单相对路径以清单文件所在目录解析，输出默认为 `results/study_report/`。

## 运行

```powershell
python reporting\generate_report.py --self-check
python reporting\generate_report.py reporting\manifest.json
```

清单样例在 `manifest.example.json`。复制为 `manifest.json` 并填写真实路径、方法、seed、预算、实际运行步数和教师数据来源。每个 run 可包含奖励训练 stdout、策略 stdout 和一份标量 CSV/JSON 导出；多个 100k/1m 预算或多个 seed 各写一条 run。标量导出格式为 CSV 列 `tag,step,value`，或 JSON 数组 `[ {"tag":"eval_win_rate/eval_win_rate", "step":1000, "value":0.25} ]`。策略 eval 标量 tag 支持 MAPT 常见的 `eval_average_episode_rewards/eval_average_episode_rewards` 和 `eval_win_rate/eval_win_rate`。TensorBoard event 文件请先导出成上述 CSV/JSON。

奖励 stdout 示例识别官方训练输出 `---------------------- {epoch} 3`，随后按模型读取 `reward/trans_loss` 与 `reward/eval_trans_loss`，或 MR 的 `reward/rf_loss` 与 `reward/eval_rf_loss`。策略 stdout 通过 `total num timesteps N/M` 关联后续 `eval win rate is ...` 和显式 `eval_average_episode_rewards is ...`。训练期 `some episodes done, average rewards` 不会作为评估回报。

工具生成机器可读 JSON/CSV、中文 Markdown 报告和有数据时的 PNG 曲线。默认拒绝覆盖同名结果；要更新报告请显式传 `--force`。没有实际指标时只写明数据缺失，不生成空图。含图数据时需 Python 环境安装 matplotlib。

#!/usr/bin/env python3
"""Parse MAPT teaching-run logs and build a small, source-traceable report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(r"-+\s*\{?epoch\}?\s*[,=:]?\s*(\d+)", re.I)
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
STEP_RE = re.compile(r"total num timesteps\s+(\d+)\s*/\s*(\d+)", re.I)
WIN_RE = re.compile(r"eval win rate is\s*" + NUMBER, re.I)
RETURN_RE = re.compile(r"eval(?:uate)?(?:_average_episode_rewards| average episode rewards(?: of agent)?)\s*(?:is|:)\s*" + NUMBER, re.I)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_reward_stdout(path: Path, model: str) -> list[dict[str, Any]]:
    """Parse MAPT reward training output_res-style epoch logs."""
    train_tag = "reward/rf_loss" if model.lower() == "mr" else "reward/trans_loss"
    eval_tag = "reward/eval_rf_loss" if model.lower() == "mr" else "reward/eval_trans_loss"
    rows: list[dict[str, Any]] = []
    current_epoch: int | None = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            if current_epoch is None:
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tag = parts[0]
            if tag not in (train_tag, eval_tag):
                continue
            value = _float(parts[-1])
            if value is not None:
                rows.append({"epoch": current_epoch, "metric": "train_loss" if tag == train_tag else "eval_loss", "value": value, "tag": tag})
    return rows


def _scalar_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            source = list(csv.DictReader(stream))
    else:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            source = payload.get("scalars", payload.get("data", payload))
        else:
            source = payload
        if isinstance(source, dict):
            expanded = []
            for tag, values in source.items():
                if isinstance(values, list):
                    for point in values:
                        if isinstance(point, (list, tuple)) and len(point) >= 3:
                            expanded.append({"tag": tag, "step": point[-2], "value": point[-1]})
                        elif isinstance(point, dict):
                            expanded.append({"tag": tag, **point})
                elif isinstance(values, dict):
                    expanded.append({"tag": tag, **values})
            source = expanded
    out = []
    for item in source:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", item.get("metric", item.get("name", ""))))
        step = item.get("step", item.get("env_step", item.get("epoch")))
        value = item.get("value", item.get("scalar", item.get("y")))
        if _float(step) is None or _float(value) is None:
            continue
        out.append({"tag": tag, "step": int(float(step)), "value": float(value)})
    return out


def _metric_from_tag(tag: str) -> str | None:
    low = tag.lower()
    if "eval_win_rate" in low:
        return "win_rate"
    if "eval_average_episode_rewards" in low or "eval_avg_episode_reward" in low:
        return "eval_return"
    if "eval_trans_loss" in low or "eval_rf_loss" in low:
        return "eval_loss"
    if low.endswith("trans_loss") or low.endswith("rf_loss"):
        return "train_loss"
    return None


def parse_scalar_file(path: Path, category: str) -> list[dict[str, Any]]:
    rows = []
    for item in _scalar_rows(path):
        metric = _metric_from_tag(item["tag"])
        if metric and (category != "policy" or metric in ("eval_return", "win_rate")):
            rows.append({"step": item["step"], "metric": metric, "value": item["value"], "tag": item["tag"]})
    return rows


def parse_policy_stdout(path: Path) -> list[dict[str, Any]]:
    """Pair evaluation values with the closest preceding MAPT timestep line."""
    rows: list[dict[str, Any]] = []
    current_step: int | None = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            step_match = STEP_RE.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
            if current_step is None:
                continue
            win_match = WIN_RE.search(line)
            if win_match:
                rows.append({"step": current_step, "metric": "win_rate", "value": float(win_match.group(1)), "tag": "stdout:eval win rate"})
                continue
            return_match = RETURN_RE.search(line)
            if return_match:
                rows.append({"step": current_step, "metric": "eval_return", "value": float(return_match.group(1)), "tag": "stdout:eval average episode rewards"})
    return rows


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_runs(manifest_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("manifest 的 runs 必须是数组")
    reward_all: list[dict[str, Any]] = []
    policy_all: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    issues: list[str] = []
    for i, config in enumerate(runs):
        if not isinstance(config, dict):
            issues.append(f"runs[{i}] 不是对象，已跳过")
            continue
        run_id = str(config.get("id", f"run-{i + 1}"))
        meta = {k: config.get(k) for k in ("id", "method", "task", "seed", "budget_steps", "actual_env_steps", "runtime_seconds", "data_source", "notes", "reward_log", "policy_stdout", "scalars")}
        meta["id"] = run_id
        reward_rows: list[dict[str, Any]] = []
        reward_path = resolve(manifest_path.parent, config.get("reward_log"))
        if reward_path:
            if reward_path.is_file():
                reward_rows.extend(parse_reward_stdout(reward_path, str(config.get("reward_model", config.get("method", "MAPT")))))
            else:
                issues.append(f"{run_id}: reward_log 不存在: {reward_path}")
        policy_rows: list[dict[str, Any]] = []
        stdout_steps: list[int] = []
        policy_path = resolve(manifest_path.parent, config.get("policy_stdout"))
        if policy_path:
            if policy_path.is_file():
                policy_rows.extend(parse_policy_stdout(policy_path))
                with policy_path.open(encoding="utf-8", errors="replace") as stream:
                    stdout_steps.extend(int(match.group(1)) for line in stream if (match := STEP_RE.search(line)))
            else:
                issues.append(f"{run_id}: policy_stdout 不存在: {policy_path}")
        scalar_path = resolve(manifest_path.parent, config.get("scalars"))
        if scalar_path:
            if scalar_path.is_file():
                scalar_rows = parse_scalar_file(scalar_path, "both")
                for row in scalar_rows:
                    if row["metric"] in ("train_loss", "eval_loss"):
                        reward_rows.append({**row, "epoch": row["step"]})
                    else:
                        policy_rows.append(row)
            else:
                issues.append(f"{run_id}: scalars 文件不存在: {scalar_path}")
        policy_by_point = {(row["metric"], row["step"]): row for row in policy_rows}
        policy_rows = [policy_by_point[key] for key in sorted(policy_by_point)]
        meta["reward_points"] = len(reward_rows)
        meta["policy_points"] = len(policy_rows)
        observed_steps = stdout_steps + [row["step"] for row in policy_rows if row.get("step") is not None]
        if observed_steps:
            meta["observed_env_steps"] = max(observed_steps)
        if stdout_steps:
            meta["stdout_last_step"] = max(stdout_steps)
        eval_steps = [row["step"] for row in policy_rows if row["metric"] in ("eval_return", "win_rate")]
        if eval_steps:
            meta["last_eval_step"] = max(eval_steps)
        for row in reward_rows:
            reward_all.append({"run_id": run_id, "method": meta.get("method"), "seed": meta.get("seed"), **row})
        for row in policy_rows:
            policy_all.append({"run_id": run_id, "method": meta.get("method"), "seed": meta.get("seed"), **row})
        run_meta.append(meta)
    return run_meta, reward_all, policy_all, issues


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_reports(outdir: Path, reward_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> list[str]:
    made = []
    if not reward_rows and not policy_rows:
        return made
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("有实际曲线数据，需要 matplotlib；请在报告 Python 环境安装 matplotlib 后重试") from exc
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK SC", "DengXian"):
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    disclaimer = "小数据单种子教学实验；不能声称复现论文表1"
    groups = sorted({row["run_id"] for row in reward_rows})
    if reward_rows:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        for group in groups:
            for metric, label in (("train_loss", "训练 loss"), ("eval_loss", "评估 loss")):
                points = sorted((r["epoch"], r["value"]) for r in reward_rows if r["run_id"] == group and r["metric"] == metric)
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=f"{group} / {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"MAPT / MR 奖励模型训练曲线\n{disclaimer}")
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8)
        target = outdir / "reward_loss.png"
        fig.savefig(target, dpi=160, facecolor="white")
        plt.close(fig)
        made.append(target.name)
    if policy_rows:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        for group in sorted({row["run_id"] for row in policy_rows}):
            for metric, ax, label in (("eval_return", axes[0], "Eval 平均累计回报"), ("win_rate", axes[1], "Eval 胜率")):
                points = sorted((r["step"], r["value"]) for r in policy_rows if r["run_id"] == group and r["metric"] == metric)
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=group)
                ax.set_ylabel(label)
                ax.grid(True, alpha=.25)
                if ax.lines:
                    ax.legend(fontsize=8)
        axes[-1].set_xlabel("环境步数")
        fig.suptitle(f"GT / MR / MAPT 策略评估\n{disclaimer}")
        target = outdir / "policy_eval.png"
        fig.savefig(target, dpi=160, facecolor="white")
        plt.close(fig)
        made.append(target.name)
    return made


def self_check() -> None:
    reward_sample = """---------------------- {epoch} 0\nreward/trans_loss 0.73\nreward/eval_trans_loss 0.61\n---------------------- {epoch} 1\nreward/trans_loss 0.52\nreward/eval_trans_loss 0.48\n"""
    policy_sample = """Map 3m Algo mat Exp demo updates 0/10 episodes, total num timesteps 1000/10000, FPS 21.\nsome episodes done, average rewards: 4.0\neval win rate is 0.25.\nMap 3m Algo mat Exp demo updates 5/10 episodes, total num timesteps 6000/10000, FPS 19.\neval win rate is 0.75.\n"""
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        reward_file = root / "reward.log"
        policy_file = root / "policy.log"
        reward_file.write_text(reward_sample, encoding="utf-8")
        policy_file.write_text(policy_sample, encoding="utf-8")
        summary_file = root / "summary.json"
        summary_file.write_text(json.dumps({
            "eval_average_episode_rewards/eval_average_episode_rewards": [[1.0, 120, 4.5]],
            "eval_win_rate/eval_win_rate": [[1.0, 120, 0.6]],
            "incre_win_rate/incre_win_rate": [[1.0, 120, 0.2]],
        }), encoding="utf-8")
        reward_rows = parse_reward_stdout(reward_file, "MAPT")
        policy_rows = parse_policy_stdout(policy_file)
        summary_rows = parse_scalar_file(summary_file, "policy")
        assert [(row["epoch"], row["metric"], row["value"]) for row in reward_rows] == [(0, "train_loss", .73), (0, "eval_loss", .61), (1, "train_loss", .52), (1, "eval_loss", .48)]
        assert policy_rows == [{"step": 1000, "metric": "win_rate", "value": .25, "tag": "stdout:eval win rate"}, {"step": 6000, "metric": "win_rate", "value": .75, "tag": "stdout:eval win rate"}]
        assert [(row["step"], row["metric"], row["value"]) for row in summary_rows] == [(120, "eval_return", 4.5), (120, "win_rate", .6)]
    print("parser self-check passed: reward epochs/losses and policy timesteps/eval win rate; train reward excluded")


def main() -> int:
    parser = argparse.ArgumentParser(description="解析 MAPT/MR 真实日志并生成教学实验报告")
    parser.add_argument("manifest", nargs="?", type=Path, help="JSON 清单路径，见 reporting/README.md")
    parser.add_argument("--outdir", type=Path, default=Path("results/study_report"), help="报告输出目录")
    parser.add_argument("--force", action="store_true", help="覆盖本工具生成的同名报告文件")
    parser.add_argument("--self-check", action="store_true", help="用少量独立日志样例检查 parser")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.manifest:
        parser.error("需要提供 manifest，或指定 --self-check")
    manifest_path = args.manifest.resolve()
    outdir = args.outdir.resolve()
    run_meta, reward_rows, policy_rows, issues = load_runs(manifest_path)
    outdir.mkdir(parents=True, exist_ok=True)
    filenames = ["report.json", "reward_metrics.csv", "policy_metrics.csv", "report.md"]
    existing = [name for name in filenames if (outdir / name).exists()]
    if existing and not args.force:
        raise FileExistsError(f"输出已存在，为避免覆盖请另选 --outdir 或明确使用 --force: {', '.join(existing)}")
    images = plot_reports(outdir, reward_rows, policy_rows)
    payload = {"manifest": str(manifest_path), "runs": run_meta, "reward_metrics": reward_rows, "policy_metrics": policy_rows, "images": images, "issues": issues}
    (outdir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(outdir / "reward_metrics.csv", reward_rows, ["run_id", "method", "seed", "epoch", "step", "metric", "value", "tag"])
    write_csv(outdir / "policy_metrics.csv", policy_rows, ["run_id", "method", "seed", "step", "metric", "value", "tag"])
    md = ["# MAPT 教学实验日志报告", "", "> 这是 3m 小数据、单 seed 的教学实验，不能声称复现论文表1。每次 eval 为 5 局，只反映小样本点估计。报告仅汇总清单中实际存在的日志和标量。", "", "## 数据来源与运行元数据", "", "| ID | 方法 | seed | 预算步数 | 实际训练步数 | stdout末次采样 | 末次评估步数 | 用时（秒） | reward点 | policy点 | 教学数据来源 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for run in run_meta:
        run_policy = [row for row in policy_rows if row["run_id"] == run["id"]]
        last_eval_step = max((row["step"] for row in run_policy), default=None)
        actual_steps = run.get("actual_env_steps") or run.get("observed_env_steps")
        stdout_last_step = run.get("stdout_last_step")
        runtime = run.get("runtime_seconds")
        md.append(f"| {run['id']} | {run.get('method') or '-'} | {run.get('seed') if run.get('seed') is not None else '-'} | {run.get('budget_steps') or '-'} | {actual_steps or '-'} | {stdout_last_step or '-'} | {last_eval_step or '-'} | {runtime if runtime is not None else '-'} | {run['reward_points']} | {run['policy_points']} | {run.get('data_source') or '-'} |")
    md.extend(["", "## 日志来源", ""])
    for run in run_meta:
        sources = []
        if run.get("reward_log"):
            sources.append(f"reward stdout `{run['reward_log']}`")
        if run.get("policy_stdout"):
            sources.append(f"policy stdout `{run['policy_stdout']}`")
        if run.get("scalars"):
            sources.append(f"标量数据 `{run['scalars']}`")
        if run.get("notes"):
            sources.append(run["notes"])
        if sources:
            md.append(f"- `{run['id']}`：" + "；".join(sources))
    md.extend(["", "## 指标", "", f"奖励模型 loss 记录：{len(reward_rows)} 点。策略评估记录：{len(policy_rows)} 点。训练 episode reward 不作为 eval 累计回报。"])
    policy_runs = sorted({row["run_id"] for row in policy_rows})
    policy_summary = []
    for run_id in policy_runs:
        rows = [row for row in policy_rows if row["run_id"] == run_id]
        eval_rows = [row for row in rows if row["metric"] in ("eval_return", "win_rate")]
        if not eval_rows:
            continue
        last_step = max(row["step"] for row in eval_rows)
        at_last = {row["metric"]: row["value"] for row in eval_rows if row["step"] == last_step}
        meta = next((run for run in run_meta if run["id"] == run_id), {})
        eval_return = f"{at_last['eval_return']:.4f}" if "eval_return" in at_last else "-"
        win_rate = f"{at_last['win_rate']:.3f}" if "win_rate" in at_last else "-"
        budget = f"{meta['budget_steps'] // 1000}k" if meta.get("budget_steps") and meta["budget_steps"] < 1000000 else ("1M" if meta.get("budget_steps") else "-")
        policy_summary.append(f"| {run_id} | {meta.get('method') or '-'} | {budget} | {last_step} | {eval_return} | {win_rate} |")
    if policy_summary:
        md.extend(["", "### 策略末次评估", "", "| ID | 方法 | 预算 | Eval 步数 | 平均累计回报 | 胜率 |", "| --- | --- | ---: | ---: | ---: | ---: |", *policy_summary])
    reward_summary = []
    for run_id in sorted({row["run_id"] for row in reward_rows}):
        rows = [row for row in reward_rows if row["run_id"] == run_id]
        epoch = max(row["epoch"] for row in rows)
        at_last = {row["metric"]: row["value"] for row in rows if row["epoch"] == epoch}
        meta = next((run for run in run_meta if run["id"] == run_id), {})
        train_loss = f"{at_last['train_loss']:.5f}" if "train_loss" in at_last else "-"
        eval_loss = f"{at_last['eval_loss']:.5f}" if "eval_loss" in at_last else "-"
        reward_summary.append(f"| {run_id} | {meta.get('method') or '-'} | {epoch} | {train_loss} | {eval_loss} |")
    if reward_summary:
        md.extend(["", "### 奖励模型末次 loss", "", "| ID | 方法 | epoch | train loss | eval loss |", "| --- | --- | ---: | ---: | ---: |", *reward_summary])
    if {"mapt_3m_seed1_100k", "mapt_3m_seed1_1m"}.issubset(policy_runs):
        md.extend(["", "### 解读限制", "", "100k 与 1M 的末次 eval 都只有单 seed、每次 5 局；这些点估计不能用于判断稳定的性能升降。当前小数据教学实验没有观察到 MAPT 的稳定优势，不能外推为论文结论。"])
    for image in images:
        md.extend(["", f"![{image}]({image})"])
    if not reward_rows and not policy_rows:
        md.extend(["", "当前清单没有可绘制的训练或策略评估指标；不生成空白曲线图。"])
    if issues:
        md.extend(["", "## 输入问题", ""])
        md.extend(f"- {issue}" for issue in issues)
    md.append("")
    (outdir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"报告已生成: {outdir}")
    print(f"reward={len(reward_rows)} policy_eval={len(policy_rows)} images={len(images)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

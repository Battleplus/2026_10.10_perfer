#!/usr/bin/env python3
"""Collect a small SMAC 3m preference set using a scripted reward teacher."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def choose_actions(avail, rng, policy):
    actions = np.empty(avail.shape[0], dtype=np.int64)
    for agent_id, mask in enumerate(avail):
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("An agent has no legal action")
        if policy == "attack":
            attacks = legal[legal >= 6]
            moves = legal[(legal >= 2) & (legal <= 5)]
            choices = attacks if attacks.size else moves
            if not choices.size:
                choices = legal
        else:
            choices = legal
        actions[agent_id] = int(rng.choice(choices))
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--pair-limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episodes < 2 or args.window < 1 or args.pair_limit < 1:
        parser.error("episodes must be >= 2; window and pair-limit must be positive")

    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root / "MAPT"
    if not repo_root.exists():
        repo_root = project_root / "code" / "MAPT"
    if not repo_root.exists():
        raise FileNotFoundError("Expected MAPT beside this script or under code/MAPT")

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    from mat.config import get_config
    from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from mat.scripts.train_policy.train_smac import parse_args as parse_policy_args

    env_args = parse_policy_args(
        ["--env_name", "StarCraft2", "--map_name", "3m", "--seed", str(args.seed)],
        get_config(),
    )
    env = StarCraft2Env(env_args)
    env.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    segments = []
    episode_summaries = []

    try:
        for episode_id in range(args.episodes):
            policy = "attack" if episode_id % 2 == 0 else "random"
            obs, _, avail = env.reset()
            observations = []
            actions = []
            episode_return = 0.0

            for _ in range(env.episode_limit):
                observations.append(np.asarray(obs, dtype=np.float32))
                action = choose_actions(np.asarray(avail), rng, policy)
                actions.append(action.reshape(-1, 1))
                obs, _, rewards, dones, _, avail = env.step(action.tolist())
                episode_return += float(np.asarray(rewards).reshape(-1)[0])
                if np.all(dones):
                    break

            observations = np.asarray(observations, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.int64)
            segment_count = 0
            for start in range(0, len(actions) - args.window + 1, args.window):
                end = start + args.window
                segments.append({
                    "episode_id": episode_id,
                    "policy": policy,
                    "episode_return": episode_return,
                    "obs": observations[start:end].copy(),
                    "actions": actions[start:end].copy(),
                })
                segment_count += 1
            episode_summaries.append({
                "episode_id": episode_id,
                "policy": policy,
                "episode_return": episode_return,
                "steps": len(actions),
                "segments": segment_count,
            })
            print("episode={} policy={} steps={} return={:.3f} segments={}".format(
                episode_id, policy, len(actions), episode_return, segment_count
            ))
    finally:
        env.close()

    train_episode_count = int(0.8 * args.episodes)
    train_episode_ids = set(range(train_episode_count))
    val_episode_ids = set(range(train_episode_count, args.episodes))

    def make_candidates(episode_ids):
        indices = [
            i for i, segment in enumerate(segments)
            if segment["episode_id"] in episode_ids
        ]
        return [
            (i, j)
            for offset, i in enumerate(indices)
            for j in indices[offset + 1:]
            if segments[i]["episode_id"] != segments[j]["episode_id"]
        ]

    train_candidates = make_candidates(train_episode_ids)
    val_candidates = make_candidates(val_episode_ids)
    rng.shuffle(train_candidates)
    rng.shuffle(val_candidates)
    val_pair_count = min(len(val_candidates), args.pair_limit // 5)
    train_pair_count = min(len(train_candidates), 4 * val_pair_count)
    if train_pair_count < 4 * val_pair_count:
        val_pair_count = train_pair_count // 4
        train_pair_count = 4 * val_pair_count
    if val_pair_count == 0:
        raise RuntimeError("Not enough held-out episode pairs for an 80/20 split")
    candidates = (
        train_candidates[:train_pair_count]
        + val_candidates[:val_pair_count]
    )
    dataset = []
    label_counts = {"traj0_preferred": 0, "traj1_preferred": 0, "tie": 0}
    for i, j in candidates:
        segment0, segment1 = segments[i], segments[j]
        difference = segment0["episode_return"] - segment1["episode_return"]
        label = 1 if difference > 0 else (-1 if difference < 0 else 0)
        label_counts[
            "traj0_preferred" if label == 1 else
            "traj1_preferred" if label == -1 else "tie"
        ] += 1
        dataset.append({
            "traj0": {"obs": segment0["obs"], "actions": segment0["actions"]},
            "traj1": {"obs": segment1["obs"], "actions": segment1["actions"]},
            "label": label,
        })

    if len(dataset) < 5:
        raise RuntimeError(
            "Only {} preference pairs were produced; at least five are required "
            "by the bundled train/validation loader".format(len(dataset))
        )

    output = args.output or (
        project_root / "results" / "script_teacher_3m_seed{}.pkl".format(args.seed)
    )
    output = output if output.is_absolute() else project_root / output
    metadata_path = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError("Output already exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "source": "project script teacher; labels compare cumulative SMAC episode reward",
        "map": "3m",
        "seed": args.seed,
        "episodes": args.episodes,
        "window_length": args.window,
        "pair_limit": args.pair_limit,
        "trajectory_segments": len(segments),
        "preference_pairs": len(dataset),
        "training_pairs": train_pair_count,
        "validation_pairs": val_pair_count,
        "training_episode_ids": sorted(train_episode_ids),
        "validation_episode_ids": sorted(val_episode_ids),
        "label_counts": label_counts,
        "episode_summaries": episode_summaries,
        "pickle_file": output.name,
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print("saved={} pairs={} labels={}".format(output, len(dataset), label_counts))


if __name__ == "__main__":
    main()

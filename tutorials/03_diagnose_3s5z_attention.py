#!/usr/bin/env python3
"""Train a small CPU MAPT model and inspect 3s5z attention-weight sensitivity."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch


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


def collect_preferences(env, episodes, window, pair_limit, seed):
    rng = np.random.default_rng(seed)
    segments = []
    episode_summaries = []

    try:
        for episode_id in range(episodes):
            policy = "attack" if episode_id % 2 == 0 else "random"
            obs, _, avail = env.reset()
            observations = []
            actions = []
            episode_return = 0.0

            for _ in range(env.episode_limit):
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                action = choose_actions(np.asarray(avail), rng, policy)
                actions.append(action.reshape(-1, 1))
                obs, _, rewards, dones, _, avail = env.step(action.tolist())
                episode_return += float(np.asarray(rewards).reshape(-1)[0])
                if np.all(dones):
                    break

            observations = np.asarray(observations, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.int64)
            segment_count = 0
            for start in range(0, len(actions) - window + 1, window):
                end = start + window
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
            ), flush=True)
    finally:
        env.close()

    train_episode_count = int(0.8 * episodes)
    train_episode_ids = set(range(train_episode_count))
    val_episode_ids = set(range(train_episode_count, episodes))

    def candidates_for(episode_ids):
        indices = [
            index for index, segment in enumerate(segments)
            if segment["episode_id"] in episode_ids
        ]
        return [
            (left, right)
            for offset, left in enumerate(indices)
            for right in indices[offset + 1:]
            if segments[left]["episode_id"] != segments[right]["episode_id"]
        ]

    train_candidates = candidates_for(train_episode_ids)
    val_candidates = candidates_for(val_episode_ids)
    rng.shuffle(train_candidates)
    rng.shuffle(val_candidates)
    val_pair_count = min(len(val_candidates), pair_limit // 5)
    train_pair_count = min(len(train_candidates), 4 * val_pair_count)
    if train_pair_count < 4 * val_pair_count:
        val_pair_count = train_pair_count // 4
        train_pair_count = 4 * val_pair_count
    if val_pair_count == 0:
        raise RuntimeError("Not enough held-out episode pairs for an 80/20 split")

    pairs = []
    label_counts = {"traj0_preferred": 0, "traj1_preferred": 0, "tie": 0}
    selected = (
        train_candidates[:train_pair_count]
        + val_candidates[:val_pair_count]
    )
    for left, right in selected:
        segment0, segment1 = segments[left], segments[right]
        difference = segment0["episode_return"] - segment1["episode_return"]
        label = 1 if difference > 0 else (-1 if difference < 0 else 0)
        count_key = (
            "traj0_preferred" if label == 1 else
            "traj1_preferred" if label == -1 else "tie"
        )
        label_counts[count_key] += 1
        pairs.append({
            "traj0": {"obs": segment0["obs"], "actions": segment0["actions"]},
            "traj1": {"obs": segment1["obs"], "actions": segment1["actions"]},
            "label": label,
        })

    if len(pairs) < 5:
        raise RuntimeError("At least five preference pairs are required")
    metadata = {
        "source": "project script teacher; labels compare cumulative SMAC episode reward",
        "map": "3s5z",
        "seed": seed,
        "episodes": episodes,
        "window_length": window,
        "pair_limit": pair_limit,
        "trajectory_segments": len(segments),
        "preference_pairs": len(pairs),
        "training_pairs": train_pair_count,
        "validation_pairs": val_pair_count,
        "training_episode_ids": sorted(train_episode_ids),
        "validation_episode_ids": sorted(val_episode_ids),
        "label_counts": label_counts,
        "episode_summaries": episode_summaries,
    }
    return pairs, metadata


def train_model(pairs, output_dir, epochs, batch_size, seed, torch_threads):
    import transformers
    from mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer
    from mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel
    from mat.algorithms.reward_model.models.torch_utils import batch_to_torch, index_batch
    from mat.algorithms.reward_model.utils.dataloader import load_dataset

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = output_dir / "script_teacher_3s5z_seed{}.pkl".format(seed)
    with dataset_path.open("wb") as stream:
        pickle.dump(pairs, stream, protocol=pickle.HIGHEST_PROTOCOL)

    train_data, eval_data, env_info = load_dataset(
        "smac", "3s5z", str(dataset_path), "Discrete"
    )
    device = torch.device("cpu")
    model_config = MultiPrefTransformer.get_default_config()
    config = transformers.GPT2Config(**model_config)
    trans = MultiTransRewardDivideModel(
        config=config,
        observation_dim=env_info["observation_dim"],
        action_dim=env_info["action_dim"],
        n_agent=env_info["n_agent"],
        action_type="Discrete",
        max_episode_steps=env_info["max_len"],
        device=device,
    )
    reward_model = MultiPrefTransformer(config, trans, device)
    train_size = train_data["observations0"].shape[0]
    eval_size = eval_data["observations0"].shape[0]
    best_eval = float("inf")
    best_epoch = None
    history = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        permutation = np.random.permutation(train_size)
        train_losses = []
        for start in range(0, train_size, batch_size):
            batch = batch_to_torch(
                index_batch(train_data, permutation[start:start + batch_size]), device
            )
            train_losses.append(float(reward_model.train(batch)["trans_loss"]))

        eval_losses = []
        with torch.no_grad():
            for start in range(0, eval_size, batch_size):
                batch = batch_to_torch(
                    index_batch(eval_data, range(start, min(start + batch_size, eval_size))),
                    device,
                )
                eval_losses.append(float(reward_model.evaluation(batch)["eval_trans_loss"]))
        train_loss = float(np.mean(train_losses))
        eval_loss = float(np.mean(eval_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss})
        print("epoch={} train_loss={:.6f} eval_loss={:.6f}".format(
            epoch, train_loss, eval_loss
        ), flush=True)
        if eval_loss < best_eval:
            best_eval = eval_loss
            best_epoch = epoch
            reward_model.save_model(str(model_dir) + os.sep, epoch)

    if best_epoch is None:
        raise RuntimeError("No finite validation loss; no checkpoint was saved")
    checkpoint = model_dir / "reward_model_{}.pt".format(best_epoch)
    reward_model.load_model(str(checkpoint))
    reward_model.trans.eval()
    return reward_model, checkpoint, dataset_path, history, env_info


def predict_reward(trans, ops, states, actions, timesteps, mode, n_agent):
    original_attention = ops.attention
    target_length = {"cooperation_uniform": n_agent, "time_uniform": states.shape[1]}.get(mode)
    replacements = 0

    def attention_with_uniform_override(
        query, key, value, casual_mask, masked_bias, dropout,
        scale_attn_weights, training, attn_mask=None, head_mask=None,
    ):
        nonlocal replacements
        output, weights = original_attention(
            query, key, value, casual_mask, masked_bias, dropout,
            scale_attn_weights, training, attn_mask=attn_mask, head_mask=head_mask,
        )
        if target_length is None or query.shape[-2] != target_length:
            return output, weights

        allowed = casual_mask.to(dtype=torch.bool).expand_as(weights)
        if attn_mask is not None:
            allowed = allowed & (attn_mask > -1.0).expand_as(weights)
        uniform = allowed.to(dtype=value.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = dropout(uniform)
        if head_mask is not None:
            uniform = uniform * head_mask
        replacements += 1
        return torch.matmul(uniform, value), uniform

    try:
        if target_length is not None:
            ops.attention = attention_with_uniform_override
        with torch.no_grad():
            prediction = trans(states, actions, timesteps)
    finally:
        ops.attention = original_attention

    if target_length is not None and replacements != 1:
        raise RuntimeError(
            "Expected one {} attention replacement, got {}".format(mode, replacements)
        )
    return float(prediction.sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--pair-limit", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.episodes < 5 or args.window < 1 or args.pair_limit < 5:
        parser.error("episodes must be >= 5; window and pair-limit must be >= 1")
    if args.epochs < 1 or args.batch_size < 1 or args.torch_threads < 1:
        parser.error("epochs, batch-size, and torch-threads must be positive")
    if torch.cuda.is_available():
        raise RuntimeError("This diagnostic is CPU-only; CUDA_VISIBLE_DEVICES must be empty")

    project_root = Path(__file__).resolve().parents[1]
    map_root = project_root / "MAPT"
    if not map_root.exists():
        map_root = project_root / "code" / "MAPT"
    if not map_root.exists():
        raise FileNotFoundError("Expected MAPT beside this script or under code/MAPT")
    output_dir = args.output_dir or project_root / "results" / "diagnostics" / "3s5z_attention"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(map_root)
    sys.path.insert(0, str(map_root))
    from mat.config import get_config
    from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from mat.scripts.train_policy.train_smac import parse_args as parse_policy_args
    from mat.algorithms.reward_model.models import ops

    env_args = parse_policy_args(
        ["--env_name", "StarCraft2", "--map_name", "3s5z", "--seed", str(args.seed)],
        get_config(),
    )
    env = StarCraft2Env(env_args)
    env.seed(args.seed)
    pairs, metadata = collect_preferences(
        env, args.episodes, args.window, args.pair_limit, args.seed
    )
    reward_model, checkpoint, dataset_path, history, env_info = train_model(
        pairs, output_dir, args.epochs, args.batch_size, args.seed,
        args.torch_threads,
    )

    sample = pairs[0]["traj0"]
    states = torch.as_tensor(sample["obs"], dtype=torch.float32).unsqueeze(0)
    actions = torch.as_tensor(sample["actions"], dtype=torch.int64).unsqueeze(0)
    timesteps = torch.arange(states.shape[1], dtype=torch.long).unsqueeze(0)
    predictions = {
        "baseline": predict_reward(
            reward_model.trans, ops, states, actions, timesteps, "baseline", env_info["n_agent"]
        ),
        "cooperation_uniform": predict_reward(
            reward_model.trans, ops, states, actions, timesteps,
            "cooperation_uniform", env_info["n_agent"],
        ),
        "time_uniform": predict_reward(
            reward_model.trans, ops, states, actions, timesteps,
            "time_uniform", env_info["n_agent"],
        ),
    }
    baseline = predictions["baseline"]
    diagnostic = {
        "map": "3s5z",
        "seed": args.seed,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_threads": torch.get_num_threads(),
        "model_checkpoint": str(checkpoint),
        "preference_dataset": str(dataset_path),
        "model_input": env_info,
        "sample_shape": {
            "observations": list(states.shape),
            "actions": list(actions.shape),
            "timesteps": list(timesteps.shape),
        },
        "predicted_segment_return": predictions,
        "delta_from_baseline": {
            name: value - baseline
            for name, value in predictions.items()
            if name != "baseline"
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "history": history,
        },
        "data": metadata,
        "interpretation": (
            "Exploratory sensitivity check on one scripted-teacher trajectory and a small "
            "map-specific model; uniform weights are substituted one component at a time. "
            "This is not a retrained ablation or a paper result."
        ),
    }
    result_path = output_dir / "attention_diagnostic.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(diagnostic, stream, indent=2)
        stream.write("\n")
    data_meta_path = output_dir / "script_teacher_3s5z_seed{}.json".format(args.seed)
    with data_meta_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print("diagnostic={} predictions={}".format(result_path, predictions), flush=True)


if __name__ == "__main__":
    main()

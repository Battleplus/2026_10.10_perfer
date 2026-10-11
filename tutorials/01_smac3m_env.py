#!/usr/bin/env python3
"""Inspect a few real SMAC 3m transitions using the official MAPT env."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args_cli = parser.parse_args()

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
    from mat.scripts.train_policy.train_smac import parse_args

    env_args = parse_args(
        ["--env_name", "StarCraft2", "--map_name", "3m", "--seed", str(args_cli.seed)],
        get_config(),
    )
    env = StarCraft2Env(env_args)
    env.seed(args_cli.seed)
    rng = np.random.default_rng(args_cli.seed)

    try:
        obs, shared_state, avail = env.reset()
        obs = np.asarray(obs)
        shared_state = np.asarray(shared_state)
        avail = np.asarray(avail)
        print("map=3m agents={} actions_per_agent={}".format(env.n_agents, env.get_total_actions()))
        print("obs.shape={} shared_state.shape={} avail.shape={}".format(
            obs.shape, shared_state.shape, avail.shape
        ))

        if obs.shape[0] != env.n_agents or avail.shape != (env.n_agents, env.get_total_actions()):
            raise RuntimeError("Unexpected environment output shapes")

        for t in range(args_cli.steps):
            legal = [np.flatnonzero(mask) for mask in avail]
            if any(indices.size == 0 for indices in legal):
                raise RuntimeError("An agent has no legal action")
            actions = [int(rng.choice(indices)) for indices in legal]
            obs, shared_state, rewards, dones, infos, avail = env.step(actions)
            obs = np.asarray(obs)
            shared_state = np.asarray(shared_state)
            rewards = np.asarray(rewards)
            dones = np.asarray(dones)
            avail = np.asarray(avail)
            print("t={} actions={} reward={:.4f} done={}".format(
                t, actions, float(rewards[0, 0]), bool(np.all(dones))
            ))
            if np.all(dones) and t + 1 < args_cli.steps:
                obs, shared_state, avail = env.reset()
                obs = np.asarray(obs)
                shared_state = np.asarray(shared_state)
                avail = np.asarray(avail)
    finally:
        env.close()


if __name__ == "__main__":
    main()

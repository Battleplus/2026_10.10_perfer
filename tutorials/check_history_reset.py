#!/usr/bin/env python3
"""Check that done rollout threads keep their own reset observation."""

from pathlib import Path
import sys

import numpy as np

project_root = Path(__file__).resolve().parents[1]
repo_root = project_root / "MAPT"
if not repo_root.exists():
    repo_root = project_root / "code" / "MAPT"
sys.path.insert(0, str(repo_root))

from mat.utils.pref_reward_assistant import PrefRewardAssistant


assistant = PrefRewardAssistant.__new__(PrefRewardAssistant)
assistant.last_dones = np.array([1.0, 0.0, 1.0], dtype=np.float32)
assistant.his_obs = np.ones((3, 4, 2, 3), dtype=np.float32)
assistant.his_act = np.ones((3, 4, 2, 1), dtype=np.float32)
assistant.his_timesteps = np.tile(np.arange(4), (3, 1)).astype(np.int64)
assistant.his_attn_masks = np.ones((3, 4, 2), dtype=np.float32)
assistant.his_next_obs = np.empty((3, 4, 2, 3), dtype=np.float32)
for env_id, start_value in enumerate((10.0, 20.0, 30.0)):
    assistant.his_next_obs[env_id] = start_value

assistant.clear_done()

for env_id, start_value in ((0, 10.0), (2, 30.0)):
    assert np.all(assistant.his_obs[env_id] == 0)
    assert np.all(assistant.his_act[env_id] == 0)
    assert np.all(assistant.his_timesteps[env_id] == 0)
    assert np.all(assistant.his_attn_masks[env_id] == 0)
    assert np.all(assistant.his_next_obs[env_id, :-1] == 0)
    assert np.all(assistant.his_next_obs[env_id, -1] == start_value)

assert np.all(assistant.his_obs[1] == 1)
assert np.all(assistant.his_act[1] == 1)
assert np.array_equal(assistant.his_timesteps[1], np.arange(4))
assert np.all(assistant.his_attn_masks[1] == 1)
assert np.all(assistant.his_next_obs[1] == 20.0)
print("history reset check passed: done threads retained their own reset observations")

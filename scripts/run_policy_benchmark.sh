#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {gt|mr|mapt} <budget_steps>" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
model="$1"
budget_steps="$2"
[[ "$budget_steps" =~ ^[1-9][0-9]*$ ]] || usage
case "$model" in
    gt|mr|mapt) ;;
    *) usage ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
mapt_root="${MAPT_ROOT:-$project_root/MAPT}"
python_bin="${PYTHON_BIN:-python}"
case "$budget_steps" in
    100000) budget_label=100k ;;
    1000000) budget_label=1m ;;
    *) budget_label="$budget_steps" ;;
esac
result_dir="${POLICY_RESULT_ROOT:-$project_root/results/policy_${model}_${budget_label}}"
experiment_name="${model}_seed1_${budget_label}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SC2PATH="${SC2PATH:-$project_root/third_party/SC2/StarCraftII}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

args=(
    --env_name StarCraft2
    --algorithm_name mat
    --experiment_name "$experiment_name"
    --map_name 3m
    --seed 1
    --n_training_threads 1
    --n_rollout_threads 8
    --episode_length 60
    --num_env_steps "$budget_steps"
    --use_eval
    --eval_episodes 5
    --eval_interval 50
    --save_interval 50
    --log_interval 10
    --use_value_active_masks
    --config_dir
    --log_dir "$result_dir"
)

case "$model" in
    mr)
        mr_reward_model="${MR_REWARD_MODEL:-$project_root/results/reward_short/smac/3m/MR/mr_seed1_short/models/reward_model_2.pt}"
        args+=(
            --use_preference_reward
            --preference_model_type MR
            --preference_model_dir "$mr_reward_model"
        )
        ;;
    mapt)
        mapt_reward_model="${MAPT_REWARD_MODEL:-$project_root/results/reward_short/smac/3m/MultiPrefTransformerDivide/mapt_seed1_short/models/reward_model_0.pt}"
        args+=(
            --use_preference_reward
            --preference_model_type MultiPrefTransformerDivide
            --preference_model_dir "$mapt_reward_model"
            --preference_medium_process_type cat
        )
        ;;
esac

if [[ ! -f "$mapt_root/mat/scripts/train_policy/train_smac.py" ]]; then
    echo "MAPT source not found; set MAPT_ROOT to the checked-out upstream repository" >&2
    exit 4
fi

cd "$mapt_root/mat/scripts"
exec /usr/bin/time -p "$python_bin" train_policy/train_smac.py "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
queue_dir="$project_root/results/policy_1m_queue"
budget_steps=1000000
export SC2PATH="${SC2PATH:-$project_root/third_party/SC2/StarCraftII}"

mkdir -p "$queue_dir"
exec 9>"$queue_dir/queue.lock"
if ! flock -n 9; then
    echo "Another policy queue is already running" >&2
    exit 3
fi

queue_log="$queue_dir/queue.log"

run_model() {
    local model="$1"
    local marker="$queue_dir/${model}_1000000.status"
    local result_dir="$project_root/results/policy_${model}_1m"
    local stdout_log="$result_dir/${model}_seed1_1m.stdout.log"

    if [[ -f "$marker" ]] && grep -qx 'exit_code=0' "$marker"; then
        printf 'SKIP model=%s budget_steps=%s reason=completed_marker\n' "$model" "$budget_steps" | tee -a "$queue_log"
        return 0
    fi

    mkdir -p "$result_dir"
    local started_at
    started_at="$(date --iso-8601=seconds)"
    printf 'START model=%s budget_steps=%s started_at=%s stdout=%s\n' "$model" "$budget_steps" "$started_at" "$stdout_log" | tee -a "$queue_log"

    local exit_code
    if env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" SC2PATH="$SC2PATH" POLICY_RESULT_ROOT="$result_dir" \
        "$script_dir/run_policy_benchmark.sh" "$model" "$budget_steps" >"$stdout_log" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    local finished_at
    finished_at="$(date --iso-8601=seconds)"
    printf 'FINISH model=%s budget_steps=%s started_at=%s finished_at=%s exit_code=%s\n' \
        "$model" "$budget_steps" "$started_at" "$finished_at" "$exit_code" | tee -a "$queue_log"
    printf 'model=%s\nbudget_steps=%s\nstarted_at=%s\nfinished_at=%s\nexit_code=%s\n' \
        "$model" "$budget_steps" "$started_at" "$finished_at" "$exit_code" >"$marker.tmp"
    mv "$marker.tmp" "$marker"

    return "$exit_code"
}

for model in gt mr mapt; do
    run_model "$model" || {
        exit_code=$?
        printf 'STOP model=%s budget_steps=%s exit_code=%s\n' "$model" "$budget_steps" "$exit_code" | tee -a "$queue_log"
        exit "$exit_code"
    }
done

printf 'QUEUE_COMPLETE budget_steps=%s finished_at=%s\n' "$budget_steps" "$(date --iso-8601=seconds)" | tee -a "$queue_log"

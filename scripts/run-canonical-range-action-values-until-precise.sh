#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 POLICY VALUE_NETWORK OUTPUT_DIR THREADS [SEED] [MILESTONES_CSV] [SHARD_WORKERS]" >&2
  exit 2
fi

policy=$1
value_network=$2
output_dir=$3
threads=$4
seed=${5:-10401}
milestones_csv=${6:-878,1500,1650,1725,1755}
shard_workers=${7:-1}
if ! [[ "$shard_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "shard workers must be a positive integer" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "$0")/.." && pwd)
runner="$repo_root/scripts/run-canonical-range-shards.sh"
finalizer="$repo_root/scripts/finalize-canonical-range-action-values.sh"
shard_dir="$output_dir/served-full-leaves"

IFS=',' read -r -a milestones <<<"$milestones_csv"
previous=0
for milestone in "${milestones[@]}"; do
  if ! [[ "$milestone" =~ ^[0-9]+$ ]] || ((milestone <= previous || milestone > 1755)); then
    echo "milestones must be strictly increasing integers no greater than 1755" >&2
    exit 2
  fi
  range_size=$((milestone - previous))
  worker_count=$shard_workers
  if ((worker_count > range_size)); then
    worker_count=$range_size
  fi
  runner_pids=()
  for ((worker = 0; worker < worker_count; worker++)); do
    worker_start=$((previous + range_size * worker / worker_count))
    worker_end=$((previous + range_size * (worker + 1) / worker_count))
    echo "canonical shard worker $((worker + 1))/$worker_count covers [$worker_start, $worker_end)"
    "$runner" \
      "$worker_start" "$worker_end" "$threads" \
      "$policy" "$value_network" "$shard_dir" "$seed" &
    runner_pids+=("$!")
  done
  runner_status=0
  for runner_pid in "${runner_pids[@]}"; do
    if ! wait "$runner_pid"; then
      runner_status=1
    fi
  done
  if ((runner_status != 0)); then
    echo "canonical shard worker failed before checkpoint $milestone" >&2
    exit 1
  fi

  checkpoint_dir="$output_dir/checkpoint-$milestone"
  merged_cache="$checkpoint_dir/merged.json.gz"
  action_values="$checkpoint_dir/action-values.json.gz"
  mkdir -p "$checkpoint_dir"
  set +e
  "$finalizer" \
    "$shard_dir" "$policy" "$merged_cache" "$action_values" \
    "$checkpoint_dir" "$milestone"
  status=$?
  set -e
  if ((status == 0)); then
    echo "canonical action-EV precision gate passed at $milestone sampled orbits"
    exit 0
  fi
  if ((status != 3)); then
    echo "canonical action-EV checkpoint $milestone failed structurally" >&2
    exit "$status"
  fi
  previous=$milestone
done

echo "canonical action-EV precision gate did not pass at any configured milestone" >&2
exit 3

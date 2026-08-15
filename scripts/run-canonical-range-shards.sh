#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 START_OFFSET END_OFFSET THREADS POLICY VALUE_NETWORK OUTPUT_DIR [SEED]" >&2
  exit 2
fi

start_offset=$1
end_offset=$2
threads=$3
policy=$4
value_network=$5
output_dir=$6
seed=${7:-10301}

repo_root=$(cd "$(dirname "$0")/.." && pwd)
solver_dir="$repo_root/preflop-solver"
binary="$solver_dir/target/release/preflop-solver"
mkdir -p "$output_dir"

if [[ ! -x "$binary" ]]; then
  cargo build --release --manifest-path "$solver_dir/Cargo.toml"
fi

for ((offset = start_offset; offset < end_offset; offset++)); do
  printf -v shard "%s/orbit-%04d.json.gz" "$output_dir" "$offset"
  if [[ -s "$shard" ]]; then
    echo "skip complete shard $shard"
    continue
  fi
  "$binary" preflop-range-cache \
    --range-policy "$policy" \
    --value-network "$value_network" \
    --seed "$seed" \
    --orbit-offset "$offset" \
    --maximum-flop-orbits 1 \
    --maximum-leaves 49 \
    --board-workers 1 \
    --resolver-iterations 2 \
    --resolver-averaging-delay 0 \
    --threads "$threads" \
    --output "$shard"
done

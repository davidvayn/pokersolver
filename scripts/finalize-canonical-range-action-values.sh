#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SHARD_DIR POLICY MERGED_CACHE ACTION_VALUES OUTPUT_DIR" >&2
  exit 2
fi

shard_dir=$1
policy=$2
merged_cache=$3
action_values=$4
output_dir=$5
repo_root=$(cd "$(dirname "$0")/.." && pwd)
binary="$repo_root/preflop-solver/target/release/preflop-solver"
batch_dir="$output_dir/range-cache-merge-batches"
mkdir -p "$batch_dir"

shards=()
for ((offset = 0; offset < 1755; offset++)); do
  printf -v shard "%s/orbit-%04d.json.gz" "$shard_dir" "$offset"
  if [[ ! -s "$shard" ]]; then
    echo "missing canonical range shard $shard" >&2
    exit 1
  fi
  shards+=("$shard")
done

batches=()
for ((start = 0, batch = 0; start < 1755; start += 50, batch++)); do
  printf -v batch_cache "%s/batch-%02d.json.gz" "$batch_dir" "$batch"
  batches+=("$batch_cache")
  if [[ -s "$batch_cache" ]]; then
    echo "skip complete merge batch $batch_cache"
    continue
  fi
  merge_args=()
  end=$((start + 50))
  if ((end > 1755)); then
    end=1755
  fi
  for ((index = start; index < end; index++)); do
    merge_args+=(--cache "${shards[$index]}")
  done
  "$binary" preflop-range-cache-merge "${merge_args[@]}" --output "$batch_cache"
done

if [[ ! -s "$merged_cache" ]]; then
  merge_args=()
  for batch_cache in "${batches[@]}"; do
    merge_args+=(--cache "$batch_cache")
  done
  "$binary" preflop-range-cache-merge "${merge_args[@]}" --output "$merged_cache"
fi

"$binary" preflop-range-action-values \
  --cache "$merged_cache" \
  --policy "$policy" \
  --output "$action_values"
gzip -t "$merged_cache"
gzip -t "$action_values"
policy_sha256=$(shasum -a 256 "$policy" | awk '{print $1}')
source_policy_sha256=$(jq -er '.source_policy_sha256' "$policy")
action_policy_sha256=$(gzip -dc "$action_values" | jq -er '.policy_artifact_sha256')
action_source_policy_sha256=$(gzip -dc "$action_values" | jq -er '.source_policy_sha256')
if [[ "$action_policy_sha256" != "$policy_sha256" ]]; then
  echo "canonical action values target policy $action_policy_sha256, expected $policy_sha256" >&2
  exit 1
fi
if [[ "$action_source_policy_sha256" != "$source_policy_sha256" ]]; then
  echo "canonical action values target source policy $action_source_policy_sha256, expected $source_policy_sha256" >&2
  exit 1
fi
gzip -dc "$action_values" | jq -e '
  .schema == "hu-preflop-canonical-range-action-values-v1" and
  .corpus_deals == 22100 and
  .policy_lookup_coverage >= 0.9999 and
  .action_ev_standard_error_coverage >= 0.95 and
  .evaluated_information_sets == [8450, 8450]
' >/dev/null
shasum -a 256 "$merged_cache" "$action_values"

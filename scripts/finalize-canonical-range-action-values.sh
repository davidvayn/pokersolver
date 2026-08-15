#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 SHARD_DIR POLICY MERGED_CACHE ACTION_VALUES OUTPUT_DIR [EXPECTED_ORBITS]" >&2
  exit 2
fi

shard_dir=$1
policy=$2
merged_cache=$3
action_values=$4
output_dir=$5
expected_orbits=${6:-1755}
if ((expected_orbits < 2 || expected_orbits > 1755)); then
  echo "expected orbit count must be between 2 and 1755" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "$0")/.." && pwd)
binary="$repo_root/preflop-solver/target/release/preflop-solver"
batch_dir="$output_dir/range-cache-merge-batches-$expected_orbits"
mkdir -p "$batch_dir"

shards=()
for ((offset = 0; offset < expected_orbits; offset++)); do
  printf -v shard "%s/orbit-%04d.json.gz" "$shard_dir" "$offset"
  if [[ ! -s "$shard" ]]; then
    echo "missing canonical range shard $shard" >&2
    exit 1
  fi
  shards+=("$shard")
done

batches=()
for ((start = 0, batch = 0; start < expected_orbits; start += 50, batch++)); do
  printf -v batch_cache "%s/batch-%02d.json.gz" "$batch_dir" "$batch"
  batches+=("$batch_cache")
  if [[ -s "$batch_cache" ]]; then
    echo "skip complete merge batch $batch_cache"
    continue
  fi
  merge_args=()
  end=$((start + 50))
  if ((end > expected_orbits)); then
    end=$expected_orbits
  fi
  for ((index = start; index < end; index++)); do
    merge_args+=(--cache "${shards[$index]}")
  done
  "$binary" preflop-range-cache-merge "${merge_args[@]}" --output "$batch_cache"
done

if [[ ! -s "$merged_cache" ]]; then
  if ((${#batches[@]} == 1)); then
    cp "${batches[0]}" "$merged_cache"
  else
    merge_args=()
    for batch_cache in "${batches[@]}"; do
      merge_args+=(--cache "$batch_cache")
    done
    "$binary" preflop-range-cache-merge "${merge_args[@]}" --output "$merged_cache"
  fi
fi

"$binary" preflop-range-action-values \
  --cache "$merged_cache" \
  --policy "$policy" \
  --output "$action_values"
gzip -t "$merged_cache"
gzip -t "$action_values"
merged_orbits=$(gzip -dc "$merged_cache" | jq -er '.boards | length')
covered_raw_flops=$(gzip -dc "$merged_cache" | jq -er '.covered_raw_flops')
complete_enumeration=$(gzip -dc "$merged_cache" | jq -r '.complete_canonical_flop_enumeration')
if ((merged_orbits != expected_orbits)); then
  echo "merged cache covers $merged_orbits canonical orbits, expected $expected_orbits" >&2
  exit 1
fi
if ((expected_orbits == 1755)); then
  if [[ "$complete_enumeration" != "true" || "$covered_raw_flops" != "22100" ]]; then
    echo "complete canonical cache does not cover all 22,100 raw flops" >&2
    exit 1
  fi
elif [[ "$complete_enumeration" != "false" ]]; then
  echo "partial canonical cache incorrectly claims complete enumeration" >&2
  exit 1
fi
policy_sha256=$(shasum -a 256 "$policy" | awk '{print $1}')
source_policy_sha256=$(jq -er '.source_policy_sha256' "$policy")
action_policy_sha256=$(gzip -dc "$action_values" | jq -er '.policy_artifact_sha256')
action_source_policy_sha256=$(gzip -dc "$action_values" | jq -er '.source_policy_sha256')
action_corpus_deals=$(gzip -dc "$action_values" | jq -er '.corpus_deals')
if [[ "$action_policy_sha256" != "$policy_sha256" ]]; then
  echo "canonical action values target policy $action_policy_sha256, expected $policy_sha256" >&2
  exit 1
fi
if [[ "$action_source_policy_sha256" != "$source_policy_sha256" ]]; then
  echo "canonical action values target source policy $action_source_policy_sha256, expected $source_policy_sha256" >&2
  exit 1
fi
if [[ "$action_corpus_deals" != "$covered_raw_flops" ]]; then
  echo "canonical action values cover $action_corpus_deals raw flops, expected $covered_raw_flops" >&2
  exit 1
fi
gzip -dc "$action_values" | jq -e '
  .schema == "hu-preflop-canonical-range-action-values-v1" and
  .policy_lookup_coverage >= 0.9999 and
  (.action_ev_standard_error_coverage | type == "number") and
  .evaluated_information_sets == [8450, 8450]
' >/dev/null
if ! gzip -dc "$action_values" | jq -e '.action_ev_standard_error_coverage >= 0.95' >/dev/null; then
  coverage=$(gzip -dc "$action_values" | jq -er '.action_ev_standard_error_coverage')
  echo "canonical action-EV precision coverage $coverage is below 0.95" >&2
  exit 3
fi
shasum -a 256 "$merged_cache" "$action_values"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 OUTPUT_DIR POLICY MODEL_VERSION TARGET_FILE" >&2
  exit 2
fi

output_dir=$1
policy=$2
model_version=$3
target_file=$4
repo_root=$(cd "$(dirname "$0")/.." && pwd)
binary="$repo_root/preflop-solver/target/release/preflop-solver"
model_dir="$repo_root/preflop-solver/models/practice"
validation_model_dir="$repo_root/preflop-solver/models/validation"
holdout_dir="$repo_root/preflop-solver/neural/runs/20bb-v82-residual-policy/s3000-b16-cos3e4-3e5"

action_values=
for milestone in 878 1500 1650 1725 1755; do
  candidate="$output_dir/checkpoint-$milestone/action-values.json.gz"
  if [[ -s "$candidate" ]] && gzip -dc "$candidate" | jq -e '
    .schema == "hu-preflop-canonical-range-action-values-v1" and
    .policy_lookup_coverage >= 0.9999 and
    .action_ev_standard_error_coverage >= 0.95 and
    .evaluated_information_sets == [8450, 8450]
  ' >/dev/null; then
    action_values=$candidate
    break
  fi
done
if [[ -z "$action_values" ]]; then
  echo "no canonical action-value checkpoint has passed the serving gate" >&2
  exit 3
fi

cd "$repo_root"
npm run practice:promote-action-values -- \
  --artifact "$action_values" \
  --policy "$policy" \
  --model-version "$model_version" \
  --target-file "$target_file"
npm run practice:verify-resolver-artifacts -- --model-version "$model_version"

comparison=$(mktemp)
cleanup() {
  rm -f "$comparison"
}
trap cleanup EXIT
"$binary" range-policy-compare \
  --network-a "$model_dir/v102-seed20931-range-policy.json.gz" \
  --network-b "$validation_model_dir/v102-seed20932-range-policy.json.gz" \
  --dataset "$holdout_dir/student-0-own-heldout.jsonl.gz" \
  --dataset "$holdout_dir/student-1-own-heldout.jsonl.gz" \
  --output "$comparison"
jq -e '
  .networkSha256s == [
    "7296e5a54cd0c310f5fd7dc126937b41131c54d00b0bc2c6807d7791c14772f0",
    "6a97155c767a4bb4beab5ff4e792965e58780be4e904601cc37c7c6555b1e1f1"
  ] and
  .datasetSha256s == [
    "d981edfe0435bc00870612673396871d54347c4e10456327ef74a25911444933",
    "f84269d93e973d23954a52ef9196dc0654991d24988825672dc68950ecbb00c7"
  ] and
  ([.gates[]] | all) and
  .validation.status == "accepted_for_full_game_pilot"
' "$comparison" >/dev/null

cargo test --release --manifest-path preflop-solver/Cargo.toml
npm run test:practice-tools
npm test
npm run build
npm run test:practice-resolver-integration

echo "canonical action-EV release candidate is promoted and verified but remains inactive"

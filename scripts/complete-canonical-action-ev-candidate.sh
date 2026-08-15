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
preflop_validation="$repo_root/preflop-solver/neural/v27-preflop-sequence.json"

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

# The exact comparison is the evidence for the manifest's cross-seed fields.
# Reject rounded/stale metadata that no longer describes the pinned policies,
# even when both the report and the manifest happen to pass independently.
jq -e \
  --arg version "$model_version" \
  --slurpfile comparison "$comparison" \
  --slurpfile preflop "$preflop_validation" '
  [.[] | select(.version == $version)] as $models |
  ($models | length) == 1 and
  ($models[0].validation as $validation |
    $preflop[0].schema == "hu-v27-preflop-sequence-v1" and
    $preflop[0].depthBb == 20 and
    $preflop[0].tabularDcfr.seeds == [7601, 7602] and
    $preflop[0].distillation.studentSha256 == [
      "151fc1d90a5c03f105e543c54d02d097df51c66a422d01e696731b9c83b409dd",
      "fcb3c62aad24a86bd04b4bdd1be8a1cbe052a6f6552f830a4a3101965e00a9e"
    ] and
    ($preflop[0].routedFullHandValidation.authentic as $authentic |
      $preflop[0].routedFullHandValidation.forcedDeviation as $forced |
      ((($validation.crossSeedFrequencyMae - ([
        $comparison[0].actionFrequencyMae,
        $authentic.actionFrequencyMae,
        $forced.actionFrequencyMae
      ] | max)) | abs) <= 1e-9) and
      ((($validation.primaryActionAgreement - ([
        $comparison[0].primaryActionAgreement,
        $authentic.primaryActionAgreement,
        $forced.primaryActionAgreement
      ] | min)) | abs) <= 1e-9) and
      ((($validation.maximumAggregateActionDelta - ([
        $comparison[0].maximumAggregateActionDelta,
        $authentic.maximumAggregateActionDelta,
        $forced.maximumAggregateActionDelta
      ] | max)) | abs) <= 1e-9) and
      ((($validation.policyCoverage - ([
        $comparison[0].lookupCoverage,
        $authentic.lookupCoverage,
        $forced.lookupCoverage
      ] | min)) | abs) <= 1e-12)) and
    $validation.crossSeedFrequencyMae <= 0.05 and
    $validation.primaryActionAgreement >= 0.85 and
    $validation.maximumAggregateActionDelta <= 0.03 and
    $validation.policyCoverage >= 0.9999 and
    $validation.actionEvStandardErrorCoverage >= 0.95 and
    $validation.projectedStorageBytes <= (20 * 1024 * 1024 * 1024) and
    $validation.rawProbabilitySumsValid == true and
    $validation.quantizedProbabilitySumsValid == true and
    $validation.independentSeedCount == 2 and
    ($validation.trainingHoursPerSeed | length) == 2 and
    ($validation.trainingHoursPerSeed | all(. >= 8 and . <= 12)) and
    $comparison[0].maximumProbabilitySumError <= 1e-6)
' data/practice/full-hand-manifests.json >/dev/null

cargo test --release --manifest-path preflop-solver/Cargo.toml
npm run test:practice-tools
npm test
npm run build
npm run test:practice-resolver-integration

jq -e --arg version "$model_version" '
  [.[] | select(.version == $version)] as $models |
  ($models | length) == 1 and
  $models[0].active == false and
  $models[0].validation.status != "accepted"
' data/practice/full-hand-manifests.json >/dev/null

echo "canonical action-EV release candidate is promoted and verified but remains inactive"

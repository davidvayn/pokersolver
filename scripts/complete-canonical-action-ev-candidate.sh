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
candidate_freeze="$repo_root/preflop-solver/neural/20bb-v50-full-hand-candidate-freeze.json"
action_ev_handoff="$repo_root/preflop-solver/neural/20bb-v104-serving-action-ev.json"
manifest_registry="$repo_root/data/practice/full-hand-manifests.json"

require_sha256() {
  local file=$1
  local expected=$2
  local actual
  actual=$(shasum -a 256 "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$file SHA-256 $actual differs from $expected" >&2
    exit 1
  fi
}

seed_5101_narrow="$repo_root/preflop-solver/neural/runs/20bb-long-v1-narrow-seed5101/metrics.jsonl"
seed_5101_wide="$repo_root/preflop-solver/neural/runs/20bb-long-v1-wide-seed5101/metrics.jsonl"
seed_5102_narrow="$repo_root/preflop-solver/neural/runs/20bb-long-v1-narrow-seed5102/metrics.jsonl"
seed_5102_wide="$repo_root/preflop-solver/neural/runs/20bb-long-v1-wide-seed5102/metrics.jsonl"
require_sha256 "$seed_5101_narrow" 6a9d2c7faf5774f43e9a9eab62a6aa0515872b50036d429bad3a261f69117874
require_sha256 "$seed_5101_wide" d6922d764eba1b9f81168dd1357f9a2a161c37404dbdd9933fb00a3c5c375f43
require_sha256 "$seed_5102_narrow" 37579b2827251656dc471284818ada3dde38aad23980b6da1a07d6c776a48d84
require_sha256 "$seed_5102_wide" 2c4ce9c53e01c162d90a47aa62d8e9206b0d0b18ca29dfc14e25b67b3dcf48a0

training_hours_7601=$(jq -s -e '
  if length == 518 and (map(.elapsed_seconds) | all(type == "number" and . > 0))
  then (map(.elapsed_seconds) | add) / 3600
  else error("seed 7601 training metrics are incomplete")
  end
' "$seed_5101_narrow" "$seed_5101_wide")
training_hours_7602=$(jq -s -e '
  if length == 515 and (map(.elapsed_seconds) | all(type == "number" and . > 0))
  then (map(.elapsed_seconds) | add) / 3600
  else error("seed 7602 training metrics are incomplete")
  end
' "$seed_5102_narrow" "$seed_5102_wide")
training_hours=$(jq -cn \
  --argjson first "$training_hours_7601" \
  --argjson second "$training_hours_7602" \
  '[$first, $second]')
jq -en --argjson hours "$training_hours" '
  ($hours | length) == 2 and ($hours | all(. >= 8 and . <= 12))
' >/dev/null
jq -e '
  .schema == "hu-routed-full-hand-candidate-freeze-v1" and
  .depthBb == 20 and
  .reproducibility.status == "byte-identical" and
  [.candidates[].candidateSeed] == [7601, 7602] and
  [.candidates[].preflop.run] == [
    "neural/runs/20bb-long-v1-narrow-seed5101",
    "neural/runs/20bb-long-v1-narrow-seed5102"
  ] and
  [.candidates[].postflop.run] == [
    "neural/runs/20bb-long-v1-wide-seed5101",
    "neural/runs/20bb-long-v1-wide-seed5102"
  ]
' "$candidate_freeze" >/dev/null
jq -e '
  .schema == "hu-20bb-serving-action-ev-handoff-v1" and
  .servedNetwork.decodedSha256 ==
    "310b9d1a39a3ecd6beff4ac99533a8ce5847dba05d9627b650a446c36e26b7c3" and
  .servingPolicyExport.sha256 ==
    "1b14fa8987663f37cd0f0f2889fa2574aaf7d31455d6f66d0fca6fe9ceec1114" and
  .varianceGate.binaryRebuildReplay.compressedBytesIdentical == true and
  .varianceGate.binaryRebuildReplay.originalCompressedSha256 ==
    .varianceGate.binaryRebuildReplay.replayedCompressedSha256 and
  .varianceGate.binaryRebuildReplay.replayedCompressedSha256 ==
    "0faec9712120bea301d893854f738e31ec55329179abf2105e4f960849594740"
' "$action_ev_handoff" >/dev/null

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
activation_backup=
activation_complete=false
cleanup() {
  rm -f "$comparison"
  if [[ -n "$activation_backup" && -f "$activation_backup" ]]; then
    if [[ "$activation_complete" == true ]]; then
      rm -f "$activation_backup"
    else
      mv "$activation_backup" "$manifest_registry"
      echo "restored inactive resolver manifest after failed activation checks" >&2
    fi
  fi
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
  --argjson training_hours "$training_hours" \
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
      "fcb3c62aad24a86bd04b4bdd1be8a1cbe0522a6f6552f830a4a3101965e00a9e"
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
    $models[0].runtime.resolver == {
      "flopIterations": 2,
      "flopResolvedActor": 1,
      "turnIterations": 2,
      "turnResolvedActor": 1,
      "riverIterations": 2,
      "riverResolvedActor": 1,
      "deterministic": true
    } and
    ($validation.trainingHoursPerSeed | length) == 2 and
    ((($validation.trainingHoursPerSeed[0] - $training_hours[0]) | abs) <= 1e-9) and
    ((($validation.trainingHoursPerSeed[1] - $training_hours[1]) | abs) <= 1e-9) and
    $comparison[0].maximumProbabilitySumError <= 1e-6)
' data/practice/full-hand-manifests.json >/dev/null

cargo test --release --manifest-path preflop-solver/Cargo.toml
npm run test:practice-tools
npm test

activation_backup=$(mktemp "$manifest_registry.activation-backup.XXXXXX")
cp "$manifest_registry" "$activation_backup"
npm run practice:activate-experimental -- --model-version "$model_version"
npm run practice:verify-resolver-artifacts -- --model-version "$model_version"
npm test -- \
  lib/practice-models.test.ts \
  app/api/practice/practice-api.test.ts \
  app/api/practice/resolve/route-post.test.ts
npm run build
npm run practice:verify-resolver-build
PRACTICE_RESOLVER_THREADS=1 npm run test:practice-resolver-integration

jq -e --arg version "$model_version" '
  [.[] | select(.version == $version)] as $models |
  ($models | length) == 1 and
  $models[0].label == "Experimental self-play" and
  $models[0].active == true and
  $models[0].validation.status == "accepted" and
  $models[0].validation.exploitabilityGateDeferred == true
' data/practice/full-hand-manifests.json >/dev/null

activation_complete=true
echo "canonical action-EV release candidate is active as Experimental self-play"

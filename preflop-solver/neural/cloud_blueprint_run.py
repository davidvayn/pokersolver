#!/usr/bin/env python3
"""Run independent schema-v3 full-game DCFR seeds on a memory-heavy cloud CPU.

The trainer itself is intentionally single-worker and deterministic. This
orchestrator uses cloud cores only for independent seeds; it never merges
incompatible regret tables or changes the sampled game. Internal resumable
checkpoints use schema 5; frozen policy artifacts remain model schema v3.
New checkpoints use lossless named MessagePack inside streaming gzip. Both
MessagePack and JSON-gzip codecs are readable for schema 5; older training
schemas require their original binary and cannot be resumed here. Final
artifacts use streaming JSON-gzip.
Completed checkpoints can also be evaluated again at the same iteration count
without retraining or copying the frozen tables.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from worker_resources import WorkerResourceGuard, process_memory_bytes


RANKS = "23456789TJQKA"
CANONICAL_HAND_CLASSES = frozenset(
    [rank * 2 for rank in RANKS]
    + [
        f"{RANKS[high]}{RANKS[low]}{suffix}"
        for high in range(len(RANKS))
        for low in range(high)
        for suffix in ("s", "o")
    ]
)
EXPECTED_ROOT_COMBOS = 1_326
PROBABILITY_TOLERANCE = 1e-6
MAX_U64 = 2**64 - 1
POLICY_STABILITY_THRESHOLDS = {
    "maximumAggregateActionFrequencyDelta": 0.03,
    "maximumComboWeightedPerActionMae": 0.05,
    "minimumPrimaryActionAgreement": 0.85,
    "maximumProbabilitySumError": PROBABILITY_TOLERANCE,
}
MAX_HS_DCFR_HORIZON = 10_000_000
CHECKPOINT_SUFFIX = ".checkpoint.msgpack.gz"
LEGACY_CHECKPOINT_SUFFIX = ".checkpoint.json.gz"


def selected_dcfr_schedule(args: argparse.Namespace) -> tuple[str, int, str | None]:
    """Return the immutable schedule name, horizon, and solver CLI flag."""
    if args.hs_dcfr_30_horizon > 0:
        return "hs30", args.hs_dcfr_30_horizon, "--hs-dcfr-30-horizon"
    return "fixed", 0, None


@dataclass(frozen=True)
class SeedRun:
    seed: int
    held_out_seed: int
    root_deviation_seed: int
    action_value_seed: int
    output: Path
    checkpoint: Path
    summary: Path
    log: Path
    resume_source: Path | None = None


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seeds must contain integers") from error
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "--seeds requires at least two distinct integers"
        )
    if any(seed < 0 or seed > MAX_U64 - 300_000 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "--seeds must be non-negative and leave room for derived evaluation seeds"
        )
    return seeds


def available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        measured = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(measured.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stream_gzip_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    decompressed = 0
    with gzip.open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            decompressed += len(chunk)
            digest.update(chunk)
    return decompressed, digest.hexdigest()


def artifact_prefix_identity(path: Path) -> dict[str, object]:
    """Read immutable top-level identity without materializing the policy."""
    with gzip.open(path, "rb") as source:
        prefix = source.read(256 * 1024)
    string_fields = (
        "solver_version",
        "artifact_id",
        "config_hash",
        "training_config_hash",
        "model",
    )
    identity: dict[str, object] = {}
    for field in string_fields:
        match = re.search(
            rb'"' + field.encode() + rb'"\s*:\s*("(?:\\.|[^"\\])*")',
            prefix,
        )
        if match is None:
            raise ValueError(f"blueprint artifact prefix is missing {field}")
        try:
            identity[field] = json.loads(match.group(1))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"blueprint artifact prefix has an invalid {field}"
            ) from error
    schema = re.search(rb'"schema_version"\s*:\s*([0-9]+)', prefix)
    approximate = re.search(rb'"approximate"\s*:\s*(true|false)', prefix)
    if schema is None or approximate is None:
        raise ValueError("blueprint artifact prefix has incomplete schema identity")
    identity["schema_version"] = int(schema.group(1))
    identity["approximate"] = approximate.group(1) == b"true"
    return identity


def validate_artifact_summary_identity(
    identity: dict[str, object], summary: dict[str, object]
) -> None:
    expected = {
        # The frozen artifact envelope remains v1; the model identifier carries
        # the trajectory-policy schema v3 version.
        "schema_version": 1,
        "solver_version": summary["solverVersion"],
        "artifact_id": summary["artifactId"],
        "config_hash": summary["configHash"],
        "training_config_hash": summary["trainingConfigHash"],
        "model": summary["model"],
        "approximate": True,
    }
    if identity != expected:
        raise ValueError(
            f"blueprint artifact identity does not match its summary: "
            f"expected {expected!r}, found {identity!r}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reader_upgrade_digest(actual: str, parent_record: dict[str, object]) -> None:
    expected = parent_record.get("canonicalJsonSha256")
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
        raise ValueError("checkpoint reader upgrade changed canonical policy/evaluation output")


def missing_run_outputs(run: SeedRun) -> list[Path]:
    return [
        path for path in (run.output, run.checkpoint, run.summary) if not path.is_file()
    ]


def stale_generated_outputs(run: SeedRun, launched_at_unix_ns: int) -> list[Path]:
    """Return outputs that were not replaced by the current child attempt.

    A completed checkpoint may intentionally remain immutable during a
    same-iteration evaluation retry, but the frozen artifact and compact
    summary must always be regenerated and revalidated.
    """
    return [
        path
        for path in (run.output, run.summary)
        if path.is_file() and path.stat().st_mtime_ns < launched_at_unix_ns
    ]


def combo_weight(hand: str) -> int:
    if len(hand) == 2:
        return 6
    return 4 if hand.endswith("s") else 12


def validated_root_strategy_map(
    rows: object,
) -> tuple[dict[str, dict[str, object]], tuple[str, ...], float]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("blueprint summary has missing compact root strategies")
    if len(rows) > len(CANONICAL_HAND_CLASSES):
        raise ValueError("blueprint summary has too many compact root strategies")
    strategies: dict[str, dict[str, object]] = {}
    expected_actions: tuple[str, ...] | None = None
    maximum_probability_sum_error = 0.0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("blueprint summary has an invalid root strategy row")
        hand = row.get("hand")
        if not isinstance(hand, str) or hand not in CANONICAL_HAND_CLASSES:
            raise ValueError("blueprint summary has an invalid root hand class")
        if hand in strategies:
            raise ValueError(f"blueprint summary has duplicate root hand {hand}")
        for field in ("regretUpdates", "averageVisits"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"blueprint summary root {hand} has an invalid {field}"
                )
        if row.get("averageVisits", 0) < 1 or row.get("trainedAverage") is not True:
            raise ValueError(
                f"blueprint summary root {hand} does not have a trained average"
            )
        actions = row.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"blueprint summary root {hand} has no actions")
        probabilities: dict[str, float] = {}
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(f"blueprint summary root {hand} has an invalid action")
            label = action.get("action")
            probability = action.get("probability")
            if (
                not isinstance(label, str)
                or label in probabilities
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    f"blueprint summary root {hand} has an invalid action distribution"
                )
            probabilities[label] = float(probability)
        action_labels = tuple(sorted(probabilities))
        if expected_actions is None:
            expected_actions = action_labels
        elif action_labels != expected_actions:
            raise ValueError("blueprint summary root action sets differ by hand")
        probability_sum_error = abs(sum(probabilities.values()) - 1.0)
        maximum_probability_sum_error = max(
            maximum_probability_sum_error, probability_sum_error
        )
        if probability_sum_error > PROBABILITY_TOLERANCE:
            raise ValueError(
                f"blueprint summary root {hand} probabilities sum to "
                f"{sum(probabilities.values())}"
            )
        primary_action = max(
            action_labels,
            key=lambda label: probabilities[label],
        )
        strategies[hand] = {
            "comboWeight": combo_weight(hand),
            "probabilities": probabilities,
            "primaryAction": primary_action,
        }
    return strategies, expected_actions or (), maximum_probability_sum_error


def policy_stability_summary(summaries: list[dict[str, object]]) -> dict[str, object]:
    parsed = [
        validated_root_strategy_map(summary.get("rootStrategies"))
        for summary in summaries
    ]
    hand_counts = [len(strategies) for strategies, _, _ in parsed]
    complete_coverage = all(
        set(strategies) == CANONICAL_HAND_CLASSES for strategies, _, _ in parsed
    )
    compatible_actions = all(actions == parsed[0][1] for _, actions, _ in parsed)
    maximum_probability_sum_error = max(error for _, _, error in parsed)
    valid_probability_sums = (
        maximum_probability_sum_error
        <= POLICY_STABILITY_THRESHOLDS["maximumProbabilitySumError"]
    )
    available = complete_coverage and compatible_actions
    maximum_aggregate_delta: float | None = None
    maximum_action_mae: float | None = None
    minimum_primary_agreement: float | None = None
    minimum_combo_primary_agreement: float | None = None

    if available:
        maximum_aggregate_delta = 0.0
        maximum_action_mae = 0.0
        minimum_primary_agreement = 1.0
        minimum_combo_primary_agreement = 1.0
        actions = parsed[0][1]
        for left_index, (left, _, _) in enumerate(parsed):
            left_aggregates = {
                action: sum(
                    int(row["comboWeight"]) * float(row["probabilities"][action])
                    for row in left.values()
                )
                / EXPECTED_ROOT_COMBOS
                for action in actions
            }
            for right, _, _ in parsed[left_index + 1 :]:
                right_aggregates = {
                    action: sum(
                        int(row["comboWeight"]) * float(row["probabilities"][action])
                        for row in right.values()
                    )
                    / EXPECTED_ROOT_COMBOS
                    for action in actions
                }
                maximum_aggregate_delta = max(
                    maximum_aggregate_delta,
                    *(
                        abs(left_aggregates[action] - right_aggregates[action])
                        for action in actions
                    ),
                )
                maximum_action_mae = max(
                    maximum_action_mae,
                    *(
                        sum(
                            int(left[hand]["comboWeight"])
                            * abs(
                                float(left[hand]["probabilities"][action])
                                - float(right[hand]["probabilities"][action])
                            )
                            for hand in CANONICAL_HAND_CLASSES
                        )
                        / EXPECTED_ROOT_COMBOS
                        for action in actions
                    ),
                )
                class_matches = sum(
                    left[hand]["primaryAction"] == right[hand]["primaryAction"]
                    for hand in CANONICAL_HAND_CLASSES
                )
                combo_matches = sum(
                    int(left[hand]["comboWeight"])
                    for hand in CANONICAL_HAND_CLASSES
                    if left[hand]["primaryAction"] == right[hand]["primaryAction"]
                )
                minimum_primary_agreement = min(
                    minimum_primary_agreement,
                    class_matches / len(CANONICAL_HAND_CLASSES),
                )
                minimum_combo_primary_agreement = min(
                    minimum_combo_primary_agreement,
                    combo_matches / EXPECTED_ROOT_COMBOS,
                )

    gates = {
        "completeRootCoverage": complete_coverage,
        "compatibleActionSets": compatible_actions,
        "validProbabilitySums": valid_probability_sums,
        "aggregateActionFrequencyDelta": available
        and maximum_aggregate_delta
        <= POLICY_STABILITY_THRESHOLDS["maximumAggregateActionFrequencyDelta"],
        "comboWeightedPerActionMae": available
        and maximum_action_mae
        <= POLICY_STABILITY_THRESHOLDS["maximumComboWeightedPerActionMae"],
        "primaryActionAgreement": available
        and minimum_primary_agreement
        >= POLICY_STABILITY_THRESHOLDS["minimumPrimaryActionAgreement"],
    }
    return {
        "available": available,
        "rootHandClassesPerSeed": hand_counts,
        "maximumProbabilitySumError": maximum_probability_sum_error,
        "maximumAggregateActionFrequencyDelta": maximum_aggregate_delta,
        "maximumComboWeightedPerActionMae": maximum_action_mae,
        "minimumPrimaryActionAgreement": minimum_primary_agreement,
        "minimumComboWeightedPrimaryActionAgreement": (minimum_combo_primary_agreement),
        "thresholds": dict(POLICY_STABILITY_THRESHOLDS),
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate_summary(
    summary: object, args: argparse.Namespace, seed: int
) -> dict[str, object]:
    if not isinstance(summary, dict):
        raise ValueError("blueprint summary must be a JSON object")
    schedule_name, schedule_horizon, _ = selected_dcfr_schedule(args)
    expected = {
        "schema": "hu-blueprint-run-summary-v1",
        "model": "hu-abstracted-external-sampling-dcfr-trajectory-v3",
        "seed": seed,
        "traversal": (
            "public_chance_sampling"
            if args.public_chance_sampling
            else "external_sampling"
        ),
        "averagingDelay": args.averaging_delay,
        "dcfr": {
            "positive_regret_exponent": args.dcfr_alpha,
            "negative_regret_exponent": args.dcfr_beta,
            "strategy_exponent": args.dcfr_gamma,
        },
        "dcfrSchedule": schedule_name,
        "dcfrScheduleHorizon": schedule_horizon,
        "requestedIterations": args.iterations,
        "trainingIterations": args.iterations,
        "heldOutDeals": args.held_out_deals,
        "rootLocalDeviationSamplesPerClass": args.root_deviation_samples,
        "actionValueDeals": args.action_value_deals,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(
                f"blueprint summary {field} does not match the run: "
                f"expected {value!r}, found {summary.get(field)!r}"
            )
    if summary.get("potentialBins", 3) != args.potential_bins:
        raise ValueError("blueprint summary potentialBins does not match the run")
    for field, enabled in (
        ("canonicalSuitBuckets", args.canonical_suit_buckets),
        ("integrateTerminalActions", args.integrate_terminal_actions),
        ("opponentCheckdownBaseline", args.opponent_checkdown_baseline),
    ):
        if summary.get(field, False) is not enabled:
            raise ValueError(f"blueprint summary {field} does not match the run")
    if summary.get("depthBb") != args.depth:
        raise ValueError("blueprint summary depth does not match the run")
    if summary.get("stoppedEarly") is not False:
        raise ValueError(
            "training stopped before the requested iteration count: "
            f"{summary.get('stopReason')}"
        )
    for field in ("configHash", "trainingConfigHash"):
        value = summary.get(field)
        if not isinstance(value, str) or len(value) != 16:
            raise ValueError(f"blueprint summary has an invalid {field}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"blueprint summary has an invalid {field}") from error
    for field in ("artifactId", "solverVersion"):
        value = summary.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"blueprint summary has an invalid {field}")
    numeric_fields = (
        "heldOutButtonMeanNetBb",
        "heldOutButtonNetStandardErrorBb",
        "heldOutUnknownInformationSetFraction",
        "heldOutUntrainedInformationSetFraction",
        "rootLocalDeviationGainBb",
        "rootLocalDeviationStandardErrorBb",
        "rootLocalDeviation99PctLowerBoundBb",
        "rootTrainedComboFraction",
        "rootContinuationUnknownInformationSetFraction",
        "rootContinuationUntrainedInformationSetFraction",
        "actionValueExportedInformationSetCoverage",
        "actionValueStandardErrorCoverage",
    )
    for field in numeric_fields:
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"blueprint summary has a missing or invalid {field}")
        if not math.isfinite(value):
            raise ValueError(f"blueprint summary has a non-finite {field}")
    for field in (
        "heldOutButtonNetStandardErrorBb",
        "rootLocalDeviationStandardErrorBb",
    ):
        if summary[field] < 0:
            raise ValueError(f"blueprint summary has a negative {field}")
    for field in (
        "rootTrainedComboFraction",
        "heldOutUnknownInformationSetFraction",
        "heldOutUntrainedInformationSetFraction",
        "rootContinuationUnknownInformationSetFraction",
        "rootContinuationUntrainedInformationSetFraction",
        "actionValueExportedInformationSetCoverage",
        "actionValueStandardErrorCoverage",
    ):
        if not 0.0 <= summary[field] <= 1.0:
            raise ValueError(f"blueprint summary has an out-of-range {field}")
    for field in (
        "informationSets",
        "preflopInformationSets",
        "postflopInformationSets",
        "trainedInformationSets",
        "exportedInformationSets",
        "actionValueEvaluatedInformationSets",
    ):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"blueprint summary has a missing or invalid {field}")
    if summary["trainedInformationSets"] > summary["informationSets"]:
        raise ValueError("blueprint summary trained information sets exceed total")
    if (
        summary["preflopInformationSets"] + summary["postflopInformationSets"]
        != summary["informationSets"]
    ):
        raise ValueError("blueprint summary street counts do not equal total")
    if summary["exportedInformationSets"] > summary["trainedInformationSets"]:
        raise ValueError("blueprint summary exported information sets exceed trained")
    if summary["actionValueEvaluatedInformationSets"] > summary["informationSets"]:
        raise ValueError("blueprint summary action-value count exceeds total")
    if (
        args.export_postflop_strategies
        and summary["exportedInformationSets"] != summary["trainedInformationSets"]
    ):
        raise ValueError(
            "blueprint summary full postflop export omitted trained information sets"
        )
    if (
        not args.export_postflop_strategies
        and summary["exportedInformationSets"] > summary["preflopInformationSets"]
    ):
        raise ValueError("blueprint summary preflop-only export contains excess rows")
    if summary.get("validationStatus") != "advisory_only":
        raise ValueError("blueprint summary has an invalid validation status")
    validation_reasons = summary.get("validationReasons")
    if (
        not isinstance(validation_reasons, list)
        or not validation_reasons
        or not all(isinstance(reason, str) and reason for reason in validation_reasons)
    ):
        raise ValueError("blueprint summary has invalid validation reasons")
    validated_root_strategy_map(summary.get("rootStrategies"))
    return summary


def aggregate_validated_summaries(
    seed_records: dict[str, object],
) -> dict[str, object] | None:
    summaries = []
    for record in seed_records.values():
        if not isinstance(record, dict) or record.get("status") != "complete":
            return None
        summary = record.get("blueprintSummary")
        if not isinstance(summary, dict):
            return None
        summaries.append(summary)
    if len(summaries) < 2:
        return None
    gains = [float(summary["rootLocalDeviationGainBb"]) for summary in summaries]
    return {
        "seedCount": len(summaries),
        "rootLocalDeviationGainMeanBb": sum(gains) / len(gains),
        "rootLocalDeviationGainSpreadBb": max(gains) - min(gains),
        "minimumActionValueStandardErrorCoverage": min(
            float(summary["actionValueStandardErrorCoverage"]) for summary in summaries
        ),
        "maximumHeldOutUnknownInformationSetFraction": max(
            float(summary["heldOutUnknownInformationSetFraction"])
            for summary in summaries
        ),
        "maximumHeldOutUntrainedInformationSetFraction": max(
            float(summary["heldOutUntrainedInformationSetFraction"])
            for summary in summaries
        ),
        "maximumRootContinuationUnknownInformationSetFraction": max(
            float(summary["rootContinuationUnknownInformationSetFraction"])
            for summary in summaries
        ),
        "maximumRootContinuationUntrainedInformationSetFraction": max(
            float(summary["rootContinuationUntrainedInformationSetFraction"])
            for summary in summaries
        ),
        "maximumInformationSets": max(
            int(summary["informationSets"]) for summary in summaries
        ),
        "crossSeedRootPolicyStability": policy_stability_summary(summaries),
    }


def run_fingerprint(
    args: argparse.Namespace,
    binary_sha256: str,
    parent_run_fingerprint: str | None = None,
) -> str:
    schedule_name, schedule_horizon, _ = selected_dcfr_schedule(args)
    settings = {
        "binarySha256": binary_sha256,
        "depthBb": args.depth,
        "iterations": args.iterations,
        "seeds": args.seeds,
        "maxInformationSets": args.max_information_sets,
        "averagingDelay": args.averaging_delay,
        "checkpointEvery": args.checkpoint_every,
        "heldOutDeals": args.held_out_deals,
        "rootDeviationSamples": args.root_deviation_samples,
        "actionValueDeals": args.action_value_deals,
        "dcfr": [args.dcfr_alpha, args.dcfr_beta, args.dcfr_gamma],
        "dcfrSchedule": schedule_name,
        "dcfrScheduleHorizon": schedule_horizon,
        "compactServingGrid": args.compact_serving_grid,
        "publicChanceSampling": args.public_chance_sampling,
        "integrateTerminalActions": args.integrate_terminal_actions,
        "opponentCheckdownBaseline": args.opponent_checkdown_baseline,
        "exportPostflopStrategies": args.export_postflop_strategies,
        "evaluationOnly": args.evaluation_only,
        "verifyCheckpointReaderUpgrade": args.verify_checkpoint_reader_upgrade,
        "parentRunFingerprint": parent_run_fingerprint,
    }
    if args.canonical_suit_buckets:
        settings["canonicalSuitBuckets"] = True
    if args.potential_bins != 3:
        settings["potentialBins"] = args.potential_bins
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_command(args: argparse.Namespace, run: SeedRun) -> list[str]:
    command = [
        str(args.binary),
        "blueprint",
        "--effective-stack-bb",
        str(args.depth),
        "--iterations",
        str(args.iterations),
        "--max-information-sets",
        str(args.max_information_sets),
        "--seed",
        str(run.seed),
        "--averaging-delay",
        str(args.averaging_delay),
        "--held-out-deals",
        str(args.held_out_deals),
        "--held-out-seed",
        str(run.held_out_seed),
        "--root-deviation-samples",
        str(args.root_deviation_samples),
        "--root-deviation-seed",
        str(run.root_deviation_seed),
        "--action-value-deals",
        str(args.action_value_deals),
        "--action-value-seed",
        str(run.action_value_seed),
        "--dcfr-alpha",
        str(args.dcfr_alpha),
        "--dcfr-beta",
        str(args.dcfr_beta),
        "--dcfr-gamma",
        str(args.dcfr_gamma),
        "--checkpoint",
        str(run.checkpoint),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--output",
        str(run.output),
        "--summary",
        str(run.summary),
    ]
    resume_source = run.checkpoint if run.checkpoint.exists() else run.resume_source
    if resume_source is not None:
        command.extend(["--resume", str(resume_source)])
    if args.compact_serving_grid:
        command.append("--compact-serving-grid")
    if args.public_chance_sampling:
        command.append("--public-chance-sampling")
    if args.canonical_suit_buckets:
        command.append("--canonical-suit-buckets")
    if args.potential_bins != 3:
        command.extend(["--potential-bins", str(args.potential_bins)])
    if args.integrate_terminal_actions:
        command.append("--integrate-terminal-actions")
    if args.opponent_checkdown_baseline:
        command.append("--opponent-checkdown-baseline")
    _, schedule_horizon, schedule_flag = selected_dcfr_schedule(args)
    if schedule_flag is not None:
        command.extend([schedule_flag, str(schedule_horizon)])
    if args.export_postflop_strategies:
        command.append("--export-postflop-strategies")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-from-dir",
        type=Path,
        help="extend or re-evaluate a completed compatible stage in a new directory",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help=(
            "reuse a completed parent checkpoint at the same iteration count "
            "with new evaluation/export controls"
        ),
    )
    parser.add_argument(
        "--verify-checkpoint-reader-upgrade", action="store_true",
        help="evaluation-only reader migration; requires identical canonical output to parent",
    )
    parser.add_argument("--depth", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("26001,26002"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--max-information-sets", type=int, default=5_000_000)
    parser.add_argument("--averaging-delay", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--held-out-deals", type=int, default=10_000)
    parser.add_argument("--root-deviation-samples", type=int, default=100)
    parser.add_argument("--action-value-deals", type=int, default=10_000)
    parser.add_argument("--dcfr-alpha", type=float, default=1.5)
    parser.add_argument("--dcfr-beta", type=float, default=0.0)
    parser.add_argument("--dcfr-gamma", type=float, default=2.0)
    parser.add_argument(
        "--hs-dcfr-30-horizon",
        type=int,
        default=0,
        help="opt-in immutable HS-DCFR(30) horizon; zero keeps fixed DCFR",
    )
    parser.add_argument("--bytes-per-information-set", type=int, default=2_300)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=20.0)
    parser.add_argument(
        "--max-worker-memory-gib", type=float, default=0.0,
        help="sampled live worker memory stop; 0 disables",
    )
    parser.add_argument(
        "--max-worker-minutes", type=float, default=0.0,
        help="wall-clock limit per seed including reload/export; 0 disables",
    )
    parser.add_argument("--compact-serving-grid", action="store_true")
    parser.add_argument("--potential-bins", type=int, default=3, choices=range(1, 256),
                        help="new card abstraction; non-default values require fresh training")
    parser.add_argument(
        "--canonical-suit-buckets", action="store_true",
        help="new suit-invariant bucket identity; incompatible with legacy checkpoints",
    )
    parser.add_argument(
        "--public-chance-sampling",
        action="store_true",
        help=(
            "research-only compatible-range traversal; use only after matched "
            "local pilots justify paid compute"
        ),
    )
    parser.add_argument(
        "--export-postflop-strategies",
        dest="export_postflop_strategies",
        action="store_true",
        help="export the complete trained profile (default)",
    )
    parser.add_argument(
        "--no-export-postflop-strategies",
        dest="export_postflop_strategies",
        action="store_false",
        help="diagnostic preflop-only artifact; never use for a full-hand model",
    )
    parser.set_defaults(export_postflop_strategies=True)
    variance_mode = parser.add_mutually_exclusive_group()
    variance_mode.add_argument("--integrate-terminal-actions", action="store_true")
    variance_mode.add_argument("--opponent-checkdown-baseline", action="store_true")
    parser.add_argument("--allow-resource-oversubscription", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_numeric_options(args: argparse.Namespace) -> None:
    if args.verify_checkpoint_reader_upgrade and not args.evaluation_only:
        raise SystemExit("checkpoint reader upgrade requires evaluation-only")
    if args.integrate_terminal_actions and args.opponent_checkdown_baseline:
        raise SystemExit("choose only one opponent variance-reduction mode")
    if (
        args.integrate_terminal_actions or args.opponent_checkdown_baseline
    ) and not args.public_chance_sampling:
        raise SystemExit("opponent variance reduction requires public-chance sampling")
    if not math.isfinite(args.depth) or args.depth <= 1.0:
        raise SystemExit("depth must be finite and exceed the 1bb big blind")
    if not 2 <= args.iterations <= MAX_U64:
        raise SystemExit("iterations must fit Rust's positive u64 domain")
    if not 1 <= args.max_information_sets <= MAX_U64:
        raise SystemExit(
            "max information sets must fit the cloud host's 64-bit usize domain"
        )
    if args.averaging_delay < 0 or args.averaging_delay >= args.iterations:
        raise SystemExit("averaging delay must be non-negative and below iterations")
    if not 1 <= args.checkpoint_every <= MAX_U64:
        raise SystemExit("checkpoint cadence must fit Rust's positive u64 domain")
    if (
        min(
            args.held_out_deals,
            args.root_deviation_samples,
            args.action_value_deals,
            args.bytes_per_information_set,
        )
        < 1
    ):
        raise SystemExit(
            "evaluation counts and bytes per information set must be positive"
        )
    if any(
        value > MAX_U64
        for value in (
            args.held_out_deals,
            args.root_deviation_samples,
            args.action_value_deals,
        )
    ):
        raise SystemExit("evaluation counts must fit Rust's positive u64 domain")
    if not math.isfinite(args.minimum_free_disk_gb) or args.minimum_free_disk_gb < 0:
        raise SystemExit("minimum free disk must be finite and non-negative")
    for name in ("max_worker_memory_gib", "max_worker_minutes"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise SystemExit(f"{name} must be finite and non-negative")
    if not all(
        math.isfinite(value)
        for value in (args.dcfr_alpha, args.dcfr_beta, args.dcfr_gamma)
    ):
        raise SystemExit("DCFR exponents must be finite")
    if any(value < 0 for value in (args.dcfr_alpha, args.dcfr_beta, args.dcfr_gamma)):
        raise SystemExit("DCFR exponents must be non-negative")
    if args.hs_dcfr_30_horizon < 0 or (0 < args.hs_dcfr_30_horizon < args.iterations):
        raise SystemExit(
            "HS-DCFR horizon must be zero or at least the requested iterations"
        )
    if args.hs_dcfr_30_horizon > MAX_HS_DCFR_HORIZON:
        raise SystemExit(
            f"HS-DCFR horizon may not exceed {MAX_HS_DCFR_HORIZON} iterations"
        )
    if args.hs_dcfr_30_horizon > 0 and (
        args.dcfr_alpha,
        args.dcfr_beta,
        args.dcfr_gamma,
    ) != (1.5, 0.0, 2.0):
        raise SystemExit("HS-DCFR(30) requires the default base DCFR exponents")
    if args.max_concurrent < 1 or args.max_concurrent > len(args.seeds):
        raise SystemExit("max concurrency must be between one and the seed count")


def validate(args: argparse.Namespace) -> None:
    validate_numeric_options(args)
    if args.max_worker_memory_gib or args.max_worker_minutes:
        try:
            measured, _ = process_memory_bytes(os.getpid())
            if measured <= 0:
                raise OSError("empty process memory reading")
        except OSError as error:
            raise SystemExit(f"resource guard unavailable: {error}") from error
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise SystemExit(f"solver binary is missing or not executable: {args.binary}")
    if args.resume_from_dir is not None:
        if not args.resume_from_dir.is_dir():
            raise SystemExit(
                f"resume source is not a directory: {args.resume_from_dir}"
            )
        if args.resume_from_dir.resolve() == args.output_dir.resolve():
            raise SystemExit("--resume-from-dir must differ from --output-dir")
    elif args.evaluation_only:
        raise SystemExit("--evaluation-only requires --resume-from-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(args.output_dir).free
    if free_disk < args.minimum_free_disk_gb * 1024**3:
        raise SystemExit(
            f"only {free_disk / 1024**3:.1f}GiB free; "
            f"{args.minimum_free_disk_gb:.1f}GiB required"
        )
    available = available_memory_bytes()
    projected = (
        args.max_information_sets
        * args.bytes_per_information_set
        * args.max_concurrent
        * 1.20
    )
    if (
        available is not None
        and projected > available * 0.85
        and not args.allow_resource_oversubscription
    ):
        raise SystemExit(
            f"projected peak {projected / 1024**3:.1f}GiB exceeds 85% of "
            f"available memory {available / 1024**3:.1f}GiB; lower concurrency/"
            "max-information-sets or pass --allow-resource-oversubscription"
        )


def command_value(command: object, flag: str) -> str | None:
    if not isinstance(command, list) or flag not in command:
        return None
    index = command.index(flag)
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def load_parent_stage(
    args: argparse.Namespace, binary_sha256: str
) -> tuple[str | None, dict[int, Path]]:
    if args.verify_checkpoint_reader_upgrade and not args.evaluation_only:
        raise SystemExit("checkpoint reader upgrade requires evaluation-only")
    if args.resume_from_dir is None:
        return None, {}
    manifest_path = args.resume_from_dir / "run-manifest.json"
    try:
        parent = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"resume source manifest is unreadable: {error}") from error
    if (
        parent.get("schema") != "hu-schema-v3-cloud-blueprint-run-v1"
        or parent.get("status") != "complete"
    ):
        raise SystemExit("resume source is not a completed schema-v3 cloud stage")
    if (parent.get("binarySha256") != binary_sha256
            and not args.verify_checkpoint_reader_upgrade):
        raise SystemExit("resume source was produced by a different solver binary")
    if parent.get("depthBb") != args.depth:
        raise SystemExit("resume source depth differs from the requested extension")
    parent_iterations = parent.get("iterations")
    if isinstance(parent_iterations, bool) or not isinstance(parent_iterations, int):
        raise SystemExit("resume source has an invalid iteration count")
    if args.evaluation_only:
        if parent_iterations != args.iterations:
            raise SystemExit(
                "evaluation-only target must equal the completed parent iterations"
            )
    elif parent_iterations >= args.iterations:
        raise SystemExit("extension target must exceed the completed parent iterations")
    parent_fingerprint = parent.get("runFingerprint")
    if not isinstance(parent_fingerprint, str) or len(parent_fingerprint) != 64:
        raise SystemExit("resume source has an invalid run fingerprint")
    try:
        int(parent_fingerprint, 16)
    except ValueError as error:
        raise SystemExit("resume source has an invalid run fingerprint") from error
    _, _, schedule_flag = selected_dcfr_schedule(args)
    expected_immutable = {
        "--effective-stack-bb": str(args.depth),
        "--averaging-delay": str(args.averaging_delay),
        "--dcfr-alpha": str(args.dcfr_alpha),
        "--dcfr-beta": str(args.dcfr_beta),
        "--dcfr-gamma": str(args.dcfr_gamma),
        "--hs-dcfr-30-horizon": (
            str(args.hs_dcfr_30_horizon)
            if schedule_flag == "--hs-dcfr-30-horizon"
            else None
        ),
    }
    if args.verify_checkpoint_reader_upgrade:
        # Only the reader binary may change. This mode cannot train, change
        # export coverage, or quietly change the evaluator's sample budget.
        expected_immutable.update({
            "--max-information-sets": str(args.max_information_sets),
            "--held-out-deals": str(args.held_out_deals),
            "--root-deviation-samples": str(args.root_deviation_samples),
            "--action-value-deals": str(args.action_value_deals),
        })
    parent_commands = parent.get("commands")
    parent_seeds = parent.get("seeds")
    if not isinstance(parent_commands, dict) or not isinstance(parent_seeds, dict):
        raise SystemExit("resume source manifest is missing commands or seed records")
    if set(parent_seeds) != {str(seed) for seed in args.seeds}:
        raise SystemExit("resume source seed set differs from the requested extension")
    checkpoints: dict[int, Path] = {}
    for seed in args.seeds:
        command = parent_commands.get(str(seed))
        if args.verify_checkpoint_reader_upgrade:
            if not isinstance(command, list):
                raise SystemExit("checkpoint reader upgrade requires parent command records")
            if ("--export-postflop-strategies" in command) != args.export_postflop_strategies:
                raise SystemExit("checkpoint reader upgrade changes export coverage")
            for flag, expected in (
                ("--held-out-seed", seed + 100_000),
                ("--root-deviation-seed", seed + 200_000),
                ("--action-value-seed", seed + 300_000),
            ):
                if command_value(command, flag) != str(expected):
                    raise SystemExit(f"checkpoint reader upgrade changes {flag}")
        if command_value(command, "--seed") != str(seed):
            raise SystemExit(f"resume source command seed differs for seed {seed}")
        for flag, expected in expected_immutable.items():
            if command_value(command, flag) != expected:
                raise SystemExit(f"resume source changes immutable setting {flag}")
        if command_value(command, "--potential-bins") != (None if args.potential_bins == 3 else str(args.potential_bins)):
            raise SystemExit("resume source changes immutable setting --potential-bins")
        if bool(
            isinstance(command, list) and "--compact-serving-grid" in command
        ) != bool(args.compact_serving_grid):
            raise SystemExit("resume source changes the serving action grid")
        if bool(
            isinstance(command, list) and "--public-chance-sampling" in command
        ) != bool(args.public_chance_sampling):
            raise SystemExit("resume source changes the blueprint traversal")
        for flag, enabled in (
            ("--canonical-suit-buckets", args.canonical_suit_buckets),
            ("--integrate-terminal-actions", args.integrate_terminal_actions),
            ("--opponent-checkdown-baseline", args.opponent_checkdown_baseline),
        ):
            if (isinstance(command, list) and flag in command) != enabled:
                raise SystemExit(f"resume source changes immutable setting {flag}")
        record = parent_seeds[str(seed)]
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise SystemExit(f"resume source seed {seed} is not complete")
        prefix = f"hu-{args.depth:g}bb-schema-v3-seed{seed}"
        candidates = [
            args.resume_from_dir / f"{prefix}{CHECKPOINT_SUFFIX}",
            args.resume_from_dir / f"{prefix}{LEGACY_CHECKPOINT_SUFFIX}",
        ]
        # Evaluation-only stages deliberately reuse their parent's immutable
        # checkpoint instead of duplicating gigabytes. Resolve that recorded
        # source, still requiring the exact size and SHA-256 below.
        recorded_source = record.get("checkpoint")
        if isinstance(recorded_source, str):
            source = Path(recorded_source)
            if source.is_absolute() and source.name in {path.name for path in candidates}:
                candidates.append(source)
        expected_size = record.get("checkpointCompressedBytes")
        expected_hash = record.get("checkpointSha256")
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise SystemExit(f"resume source checkpoint is missing: {candidates[0]}")
        size_matches = [
            path for path in existing if path.stat().st_size == expected_size
        ]
        if not size_matches:
            raise SystemExit(f"resume source checkpoint size changed for seed {seed}")
        if not isinstance(expected_hash, str):
            raise SystemExit(f"resume source checkpoint hash changed for seed {seed}")
        checkpoint = next(
            (path for path in size_matches if file_sha256(path) == expected_hash),
            None,
        )
        if checkpoint is None:
            raise SystemExit(f"resume source checkpoint hash changed for seed {seed}")
        checkpoints[seed] = checkpoint
    return parent_fingerprint, checkpoints


def main() -> int:
    args = parse_args()
    validate(args)
    binary_sha256 = file_sha256(args.binary)
    parent_fingerprint, parent_checkpoints = load_parent_stage(args, binary_sha256)
    parent_records = (
        json.loads((args.resume_from_dir / "run-manifest.json").read_text())["seeds"]
        if args.verify_checkpoint_reader_upgrade else {}
    )
    runs = []
    for seed in args.seeds:
        prefix = f"hu-{args.depth:g}bb-schema-v3-seed{seed}"
        runs.append(
            SeedRun(
                seed=seed,
                held_out_seed=seed + 100_000,
                root_deviation_seed=seed + 200_000,
                action_value_seed=seed + 300_000,
                output=args.output_dir / f"{prefix}.artifact.json.gz",
                checkpoint=args.output_dir / f"{prefix}{CHECKPOINT_SUFFIX}",
                summary=args.output_dir / f"{prefix}.summary.json",
                log=args.output_dir / f"{prefix}.log",
                resume_source=parent_checkpoints.get(seed),
            )
        )
    commands = {run.seed: build_command(args, run) for run in runs}
    manifest_path = args.output_dir / "run-manifest.json"
    fingerprint = run_fingerprint(args, binary_sha256, parent_fingerprint)
    first_started_at = int(time.time())
    attempt = 1
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"existing run manifest is unreadable: {error}") from error
        if (
            existing.get("schema") != "hu-schema-v3-cloud-blueprint-run-v1"
            or existing.get("runFingerprint") != fingerprint
        ):
            raise SystemExit(
                "existing output directory belongs to a different binary or run "
                "configuration; choose a new directory"
            )
        first_started_at = int(existing.get("firstStartedAtUnix", first_started_at))
        attempt = int(existing.get("attempt", 0)) + 1
    state: dict[str, object] = {
        "schema": "hu-schema-v3-cloud-blueprint-run-v1",
        "runFingerprint": fingerprint,
        "parentRunFingerprint": parent_fingerprint,
        "resumeFromDirectory": (
            str(args.resume_from_dir) if args.resume_from_dir is not None else None
        ),
        "evaluationOnly": args.evaluation_only,
        "verifyCheckpointReaderUpgrade": args.verify_checkpoint_reader_upgrade,
        "binarySha256": binary_sha256,
        "firstStartedAtUnix": first_started_at,
        "startedAtUnix": int(time.time()),
        "attempt": attempt,
        "status": "dry_run" if args.dry_run else "running",
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "availableMemoryBytes": available_memory_bytes(),
            "freeDiskBytesAtStart": shutil.disk_usage(args.output_dir).free,
        },
        "depthBb": args.depth,
        "iterations": args.iterations,
        "maxInformationSets": args.max_information_sets,
        "maxConcurrent": args.max_concurrent,
        "resourceLimits": {
            "maxWorkerMemoryGiB": args.max_worker_memory_gib,
            "maxWorkerMinutes": args.max_worker_minutes,
            "minimumFreeDiskGiB": args.minimum_free_disk_gb,
            "sampleIntervalSeconds": 0.25,
            "note": "Sampled stop, not a kernel memory cap; macOS preflight reports installed RAM.",
        },
        "commands": {str(seed): command for seed, command in commands.items()},
        "seeds": {},
    }
    atomic_json(manifest_path, state)
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return 0

    pending = list(runs)
    active: dict[
        int, tuple[SeedRun, subprocess.Popen[bytes], object, Path | None, int]
    ] = {}
    failures = 0
    validation_stopped = False
    interrupted = False
    resource_stop = threading.Event()
    guards: dict[int, WorkerResourceGuard] = {}

    def stop_children(_signal: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        resource_stop.set()
        for guard in guards.values():
            guard.request_stop("operator interrupt")

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)

    while pending or active:
        while (
            pending and len(active) < args.max_concurrent
            and not interrupted and not resource_stop.is_set()
        ):
            run = pending.pop(0)
            resume_source = (
                run.checkpoint if run.checkpoint.exists() else run.resume_source
            )
            resumed = resume_source is not None
            log_handle = run.log.open("ab")
            launched_at_unix_ns = time.time_ns()
            process = subprocess.Popen(
                commands[run.seed],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            guards[run.seed] = WorkerResourceGuard(
                process, args.output_dir,
                max_memory_bytes=int(args.max_worker_memory_gib * 1024**3),
                max_seconds=args.max_worker_minutes * 60,
                minimum_free_disk_bytes=int(args.minimum_free_disk_gb * 1024**3),
                stop_event=resource_stop,
            ).start()
            active[run.seed] = (
                run,
                process,
                log_handle,
                resume_source,
                launched_at_unix_ns,
            )
            state["seeds"][str(run.seed)] = {
                "status": "running",
                "pid": process.pid,
                "resumed": resumed,
                "resumeSource": (
                    str(resume_source) if resume_source is not None else None
                ),
                "startedAtUnix": int(time.time()),
                "startedAtUnixNs": launched_at_unix_ns,
            }
            atomic_json(manifest_path, state)

        time.sleep(1.0)
        for seed, (
            run,
            process,
            log_handle,
            resume_source,
            launched_at_unix_ns,
        ) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            checkpoint_for_record = (
                resume_source
                if args.evaluation_only and resume_source is not None
                else run.checkpoint
            )
            record: dict[str, object] = {
                **guards.pop(seed).finish(),
                "status": "complete" if return_code == 0 else "failed",
                "returnCode": return_code,
                "finishedAtUnix": int(time.time()),
                "checkpoint": str(checkpoint_for_record),
                "output": str(run.output),
                "summary": str(run.summary),
                "log": str(run.log),
                "resumed": resume_source is not None,
                "resumeSource": (
                    str(resume_source) if resume_source is not None else None
                ),
            }
            missing_outputs = [
                path
                for path in (run.output, checkpoint_for_record, run.summary)
                if not path.is_file()
            ]
            stale_outputs = stale_generated_outputs(run, launched_at_unix_ns)
            if return_code == 0 and not missing_outputs and not stale_outputs:
                try:
                    decompressed_bytes, canonical_sha256 = stream_gzip_and_sha256(
                        run.output
                    )
                    if args.verify_checkpoint_reader_upgrade:
                        try:
                            validate_reader_upgrade_digest(canonical_sha256, parent_records[str(seed)])
                        except ValueError:
                            validation_stopped = True
                            resource_stop.set()
                            raise
                    summary = validate_summary(
                        json.loads(run.summary.read_text()), args, seed
                    )
                    identity = artifact_prefix_identity(run.output)
                    validate_artifact_summary_identity(identity, summary)
                    record.update(
                        {
                            "compressedBytes": run.output.stat().st_size,
                            "decompressedBytes": decompressed_bytes,
                            "canonicalJsonSha256": canonical_sha256,
                            "artifactIdentity": identity,
                            "checkpointCompressedBytes": (
                                checkpoint_for_record.stat().st_size
                            ),
                            "checkpointSha256": file_sha256(checkpoint_for_record),
                            "blueprintSummary": summary,
                        }
                    )
                except (OSError, EOFError, ValueError, json.JSONDecodeError) as error:
                    failures += 1
                    record["status"] = "failed"
                    record["integrityError"] = str(error)
            else:
                failures += 1
                record["status"] = "failed"
                if return_code == 0:
                    problems = []
                    if missing_outputs:
                        problems.append(
                            "missing required outputs: "
                            + ", ".join(str(path) for path in missing_outputs)
                        )
                    if stale_outputs:
                        problems.append(
                            "outputs were not regenerated by this attempt: "
                            + ", ".join(str(path) for path in stale_outputs)
                        )
                    record["integrityError"] = "; ".join(problems)
            if record["resourceStopReason"] is not None:
                record["status"] = "interrupted" if interrupted else "resource_stopped"
            state["seeds"][str(seed)] = record
            del active[seed]
            atomic_json(manifest_path, state)
        if (interrupted or resource_stop.is_set()) and not active:
            break

    state["finishedAtUnix"] = int(time.time())
    state["status"] = (
        "interrupted" if interrupted else (
            "validation_stopped" if validation_stopped else (
                "resource_stopped" if resource_stop.is_set() else ("failed" if failures else "complete")
            )
        )
    )
    for run in pending:
        state["seeds"][str(run.seed)] = {"status": "not_started"}
    if state["status"] == "complete":
        state["aggregateSummary"] = aggregate_validated_summaries(state["seeds"])
    atomic_json(manifest_path, state)
    return 130 if interrupted else (1 if failures or resource_stop.is_set() else 0)


if __name__ == "__main__":
    sys.exit(main())

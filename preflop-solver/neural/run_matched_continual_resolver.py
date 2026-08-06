#!/usr/bin/env python3
"""Plan or execute the frozen cross-seed continual-resolver release gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import freeze_resolver_reach_release as release_freeze
import run_resolver_reach_crossfit as crossfit
import run_resolver_reach_release as release_runner


PLAN_SCHEMA = "hu-matched-continual-resolver-execution-plan-v1"
VALUE_VALIDATION_SCHEMA = "hu-resolver-reach-value-release-validation-v1"
SOLUTION_SCHEMA = "hu-depth-limited-flop-public-belief-solution-v2"
SOLUTION_METHOD = (
    "frozen_average_resolver_strategy_scored_by_independent_turn_cfv_network_"
    "with_exact_turn_chance_and_exact_flop_all_in_runouts"
)
RANKS = "23456789TJQKA"
SUITS = "cdhs"
COMBO_COUNT = 1326


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_freeze", type=Path)
    parser.add_argument("--value-validation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-plan", type=Path)
    return parser.parse_args()


def resolved(repository_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def card_index(card: str) -> int:
    if len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS:
        raise ValueError(f"invalid card in matched resolver root: {card}")
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def board_indices(board: str) -> list[int]:
    cards = [card_index(card) for card in board.split(",")]
    if len(cards) != 3 or len(set(cards)) != 3:
        raise ValueError(f"invalid matched resolver flop: {board}")
    return cards


def validate_value_report(
    value_validation_path: Path,
    release_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    report = json.loads(value_validation_path.read_text())
    expected_release_sha = release_freeze.sha256_file(release_path)
    reference = report.get("releaseFreeze", {})
    referenced_path = resolved(repository_root, reference.get("path", ""))
    if (
        report.get("schema") != VALUE_VALIDATION_SCHEMA
        or report.get("status")
        != "accepted-awaiting-strategy-and-full-game-gates"
        or report.get("activationAllowed") is not False
        or reference.get("sha256") != expected_release_sha
        or referenced_path.resolve() != release_path.resolve()
        or not report.get("gates")
        or not all(value is True for value in report["gates"].values())
    ):
        raise ValueError(
            "matched resolver remains sealed until every frozen value-release gate passes"
        )
    return report


def validate_release_contract(
    release: dict[str, Any], repository_root: Path
) -> None:
    protocol_path = resolved(repository_root, release["protocol"]["path"])
    protocol, experiment, corpus = release_freeze.validate_protocol(
        protocol_path, repository_root, require_unopened=False
    )
    if (
        release.get("matchedContinualResolver")
        != protocol.get("matchedContinualResolver")
        or release.get("reservedResolverEvaluation")
        != corpus.get("reservedEvaluationShards")
        or release.get("trainer", {}).get("trainingSeeds")
        != experiment.get("postSelection", {}).get("releaseTrainingSeeds")
    ):
        raise ValueError("release freeze diverges from its pinned evaluation protocol")


def release_models(
    release: dict[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in release["trainer"]["trainingSeeds"]]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("matched resolver requires exactly two independent release seeds")
    output_directory = release["trainer"]["outputDirectory"]
    models = []
    for seed in seeds:
        relative_path = f"{output_directory}/turn-value-range-seed{seed}.json"
        path = resolved(repository_root, relative_path)
        payload = json.loads(path.read_text())
        if (
            int(payload.get("seed", -1)) != seed
            or payload.get("usesExactRanges") is not True
            or payload.get("sourceValidationStatus") != "accepted"
            or not isinstance(payload.get("sourceDatasetSha256"), str)
            or len(payload["sourceDatasetSha256"]) != 64
            or not isinstance(payload.get("sourcePolicySha256"), str)
            or len(payload["sourcePolicySha256"]) != 64
        ):
            raise ValueError(f"release value network is not eligible: {relative_path}")
        models.append(
            {
                "seed": seed,
                "path": relative_path,
                "sha256": release_freeze.sha256_file(path),
                "sourceDatasetSha256": payload.get("sourceDatasetSha256"),
                "sourcePolicySha256": payload.get("sourcePolicySha256"),
            }
        )
    return models


def resolver_command(
    controls: dict[str, Any],
    board: str,
    strategy_model: dict[str, Any],
    evaluation_model: dict[str, Any],
    output: str,
) -> list[str]:
    return [
        "target/release/preflop-solver",
        "flop-pbs-resolve",
        "--effective-stack-bb",
        str(controls["effectiveStackBb"]),
        "--board",
        board,
        "--pot-bb",
        str(controls["rootPotBb"]),
        "--actor",
        str(controls["rootActor"]),
        "--iterations",
        str(controls["iterations"]),
        "--averaging-delay",
        str(controls["averagingDelay"]),
        "--threads",
        str(controls["threads"]),
        "--value-network",
        strategy_model["path"],
        "--evaluation-value-network",
        evaluation_model["path"],
        "--output",
        output,
    ]


def build_plan(
    release: dict[str, Any],
    value_report: dict[str, Any],
    release_path: Path,
    value_validation_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    controls = release["matchedContinualResolver"]
    if not controls.get("runOnlyAfterValueReleaseGatesPass") or not controls.get(
        "crossEvaluateBothReleaseSeeds"
    ):
        raise ValueError("release freeze does not require matched cross-evaluation")
    models = release_models(release, repository_root)
    boards = [
        board
        for shard in release["reservedResolverEvaluation"]
        for board in shard["boards"]
    ]
    for board in boards:
        board_indices(board)
    if not boards or len(boards) != len(set(boards)):
        raise ValueError("reserved matched-resolver roots must be nonempty and unique")
    output_directory = (
        f"{release['trainer']['outputDirectory']}/matched-continual-resolver"
    )
    jobs = []
    for board in boards:
        board_label = board.replace(",", "")
        for strategy_index, evaluation_index in ((0, 1), (1, 0)):
            strategy = models[strategy_index]
            evaluation = models[evaluation_index]
            output = (
                f"{output_directory}/flop-{board_label}-strategy{strategy['seed']}"
                f"-evaluation{evaluation['seed']}.json"
            )
            jobs.append(
                {
                    "name": (
                        f"flop-{board_label}-strategy{strategy['seed']}"
                        f"-evaluation{evaluation['seed']}"
                    ),
                    "board": board,
                    "boardIndices": board_indices(board),
                    "strategyModel": strategy,
                    "evaluationModel": evaluation,
                    "output": output,
                    "command": resolver_command(
                        controls, board, strategy, evaluation, output
                    ),
                }
            )
    return {
        "schema": PLAN_SCHEMA,
        "status": "frozen-for-matched-continual-resolver-evaluation",
        "activationAllowed": False,
        "releaseFreeze": {
            "path": str(release_path),
            "sha256": release_freeze.sha256_file(release_path),
        },
        "valueReleaseValidation": {
            "path": str(value_validation_path),
            "sha256": release_freeze.sha256_file(value_validation_path),
            "status": value_report["status"],
        },
        "controls": controls,
        "models": models,
        "rootCount": len(boards),
        "jobs": jobs,
    }


def legal_combo(combo: int, board: set[int]) -> bool:
    high = 1
    while high * (high - 1) // 2 <= combo:
        high += 1
    high -= 1
    low = combo - high * (high - 1) // 2
    return high not in board and low not in board


def validate_strategy_probabilities(
    strategies: Any, board: list[int]
) -> tuple[bool, float]:
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("matched resolver output has no frozen strategies")
    maximum_sum_error = 0.0
    blocked = set(board)
    for strategy in strategies:
        labels = strategy.get("action_labels")
        probabilities = strategy.get("probabilities")
        if (
            not isinstance(labels, list)
            or not labels
            or len(labels) != len(set(labels))
            or not isinstance(probabilities, list)
            or len(probabilities) != COMBO_COUNT * len(labels)
        ):
            raise ValueError("matched resolver strategy shape is invalid")
        action_count = len(labels)
        for combo in range(COMBO_COUNT):
            row = probabilities[combo * action_count : (combo + 1) * action_count]
            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and -1e-7 <= value <= 1.0 + 1e-7
                for value in row
            ):
                raise ValueError("matched resolver strategy has an invalid probability")
            expected = 1.0 if legal_combo(combo, blocked) else 0.0
            error = abs(sum(float(value) for value in row) - expected)
            maximum_sum_error = max(maximum_sum_error, error)
    return maximum_sum_error <= 1e-5, maximum_sum_error


def finite_vector(value: Any, expected: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    )


def valid_uniform_root_ranges(value: Any, board: list[int]) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    blocked = set(board)
    legal_count = sum(
        1 for combo in range(COMBO_COUNT) if legal_combo(combo, blocked)
    )
    expected = 1.0 / legal_count
    for player_range in value:
        if not finite_vector(player_range, COMBO_COUNT):
            return False
        for combo, weight in enumerate(player_range):
            target = expected if legal_combo(combo, blocked) else 0.0
            if float(weight) < 0.0 or abs(float(weight) - target) > 1e-12:
                return False
    return True


def inspect_solution(job: dict[str, Any], controls: dict[str, Any], path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    strategy = job["strategyModel"]
    evaluation = job["evaluationModel"]
    state = payload.get("state", {})
    structural = (
        payload.get("schema") == SOLUTION_SCHEMA
        and payload.get("method") == SOLUTION_METHOD
        and payload.get("approximate") is True
        and float(payload.get("effective_stack_bb", float("nan")))
        == float(controls["effectiveStackBb"])
        and int(payload.get("value_network_seed", -1)) == strategy["seed"]
        and payload.get("value_network_sha256") == strategy["sha256"]
        and payload.get("value_network_source_dataset_sha256")
        == strategy["sourceDatasetSha256"]
        and payload.get("value_network_source_policy_sha256")
        == strategy["sourcePolicySha256"]
        and int(payload.get("evaluation_value_network_seed", -1))
        == evaluation["seed"]
        and payload.get("evaluation_value_network_sha256") == evaluation["sha256"]
        and payload.get("evaluation_value_network_source_dataset_sha256")
        == evaluation["sourceDatasetSha256"]
        and payload.get("evaluation_value_network_source_policy_sha256")
        == evaluation["sourcePolicySha256"]
        and payload.get("uses_exact_ranges") is True
        and state.get("street") == "flop"
        and state.get("board") == job["boardIndices"]
        and int(state.get("actor", -1)) == int(controls["rootActor"])
        and state.get("invested_bb")
        == [float(controls["rootPotBb"]) / 2.0] * 2
        and state.get("street_invested_bb") == [0.0, 0.0]
        and state.get("public_history") == ["public_belief:flop_start"]
        and int(state.get("aggressions", -1)) == 0
        and int(state.get("checks", -1)) == 0
        and state.get("raise_reopened") is True
        and valid_uniform_root_ranges(state.get("ranges"), job["boardIndices"])
        and int(payload.get("iterations", -1)) == int(controls["iterations"])
        and int(payload.get("averaging_delay", -1))
        == int(controls["averagingDelay"])
        and int(payload.get("threads", -1)) == int(controls["threads"])
        and all(
            isinstance(payload.get(field), list) and len(payload[field]) == 2
            for field in ("counterfactual_values_bb", "opponent_compatible_mass")
        )
    )
    # Both vector fields contain one vector per player.
    structural = structural and all(
        finite_vector(vector, COMBO_COUNT)
        for field in ("counterfactual_values_bb", "opponent_compatible_mass")
        for vector in payload.get(field, [])
    )
    structural = structural and all(
        float(value) >= 0.0
        for vector in payload.get("opponent_compatible_mass", [])
        for value in vector
    )
    if not structural:
        raise ValueError(f"matched resolver output provenance/state is invalid: {path}")
    probabilities_valid, maximum_sum_error = validate_strategy_probabilities(
        payload["strategies"], job["boardIndices"]
    )
    metrics = payload.get("metrics", {})
    required_metrics = (
        "depth_limited_exploitability_bb_per_hand",
        "resolver_relative_exploitability_improvement",
        "zero_sum_residual_after_projection_bb",
    )
    if not all(
        isinstance(metrics.get(name), (int, float))
        and math.isfinite(float(metrics[name]))
        for name in required_metrics
    ):
        raise ValueError(f"matched resolver output has invalid metrics: {path}")
    exploitability = float(metrics["depth_limited_exploitability_bb_per_hand"])
    improvement = float(metrics["resolver_relative_exploitability_improvement"])
    zero_sum = abs(float(metrics["zero_sum_residual_after_projection_bb"]))
    gates = {
        "artifactProvenance": True,
        "acceptedSolverValidation": payload.get("validation", {}).get("status")
        == "accepted",
        "probabilitySums": probabilities_valid,
        "maximumLocalExploitability": 0.0
        <= exploitability
        <= float(controls["maximumExploitabilityBbPerHand"]),
        "positiveResolverImprovement": improvement > 0.0,
        "zeroSumResidual": zero_sum <= 1e-6,
    }
    return {
        "board": job["board"],
        "strategySeed": strategy["seed"],
        "evaluationSeed": evaluation["seed"],
        "solution": str(path),
        "solutionSha256": release_freeze.sha256_file(path),
        "depthLimitedExploitabilityBbPerHand": exploitability,
        "resolverRelativeExploitabilityImprovement": improvement,
        "zeroSumResidualBb": zero_sum,
        "maximumProbabilitySumError": maximum_sum_error,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def run_job(
    job: dict[str, Any],
    controls: dict[str, Any],
    repository_root: Path,
    resume: bool,
) -> None:
    output = resolved(repository_root, job["output"])
    if output.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite matched resolver output: {output}")
        evidence = inspect_solution(job, controls, output)
        if not evidence["accepted"]:
            raise ValueError(f"reused matched resolver output failed its gate: {output}")
        print(json.dumps({"event": "matched-resolver-job-reused", "name": job["name"]}), flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "matched-resolver-job-start", "name": job["name"]}), flush=True)
    with output.with_suffix(".log").open("w") as sink:
        subprocess.run(
            job["command"],
            cwd=repository_root,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not output.is_file():
        raise ValueError(f"matched resolver job created no output: {output}")
    evidence = inspect_solution(job, controls, output)
    if not evidence["accepted"]:
        raise ValueError(
            f"matched resolver gate failed at {job['name']}: "
            f"{evidence['depthLimitedExploitabilityBbPerHand']:.6f}bb/hand"
        )
    print(json.dumps({"event": "matched-resolver-job-complete", "name": job["name"]}), flush=True)


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.release_freeze.resolve().parent.parent
    release = release_runner.validate_release_freeze(args.release_freeze, repository_root)
    validate_release_contract(release, repository_root)
    value_report = validate_value_report(
        args.value_validation, args.release_freeze, repository_root
    )
    plan = build_plan(
        release,
        value_report,
        args.release_freeze,
        args.value_validation,
        repository_root,
    )
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_plan.with_suffix(args.output_plan.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output_plan)
    if not args.execute:
        print(encoded, end="")
        return
    for job in plan["jobs"]:
        run_job(job, plan["controls"], repository_root, args.resume)
    print(json.dumps({"status": "matched-resolver-complete-activation-still-disabled"}))


if __name__ == "__main__":
    main()

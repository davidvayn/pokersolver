#!/usr/bin/env python3
"""Plan or execute the frozen V49 range-consistent response gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any

import freeze_range_response_release as response_freeze
import freeze_resolver_reach_release as release_freeze
import run_matched_continual_resolver as matched_v1


PLAN_SCHEMA = "hu-range-response-release-execution-plan-v1"
CONVERGENCE_SCHEMA = "hu-flop-resolver-convergence-diagnostic-v3"
CONVERGENCE_METHOD = (
    "single_paired_alternating_dcfr_trajectory_with_frozen_average_checkpoints_"
    "cross_scored_by_independent_turn_cfv_network"
)
RESPONSE_SCHEMA = "hu-flop-range-response-diagnostic-v1"
RESPONSE_METHOD = (
    "one_player_depth_limited_dcfr_with_frozen_opponent_and_response_conditioned_"
    "public_ranges_cross_scored_by_independent_turn_cfv_network"
)
DEFAULT_DCFR = {
    "positive_regret_exponent": 1.5,
    "negative_regret_exponent": 0.0,
    "strategy_exponent": 2.0,
}


def dcfr_controls(controls: dict[str, Any], field: str) -> dict[str, float]:
    raw = controls.get(field, DEFAULT_DCFR)
    return {key: float(raw[key]) for key in DEFAULT_DCFR}


def append_dcfr_args(command: list[str], dcfr: dict[str, float]) -> None:
    command.extend(
        [
            "--dcfr-alpha",
            str(dcfr["positive_regret_exponent"]),
            "--dcfr-beta",
            str(dcfr["negative_regret_exponent"]),
            "--dcfr-gamma",
            str(dcfr["strategy_exponent"]),
        ]
    )


def convergence_method(controls: dict[str, Any]) -> str:
    method = CONVERGENCE_METHOD
    if bool(controls.get("strategyRegretMatchingPlus", False)):
        method += "_regret_matching_plus"
    return method


def solution_method(controls: dict[str, Any]) -> str:
    method = matched_v1.SOLUTION_METHOD
    if bool(controls.get("strategyRegretMatchingPlus", False)):
        method += "_regret_matching_plus"
    return method


def response_method(controls: dict[str, Any]) -> str:
    method = RESPONSE_METHOD
    if bool(controls.get("responseRegretMatchingPlus", False)):
        method += "_regret_matching_plus"
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-rejection", action="store_true")
    parser.add_argument("--output-plan", type=Path)
    return parser.parse_args()


def resolved(repository_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def model_fields(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(model["seed"]),
        "path": model["path"],
        "sha256": model["sha256"],
        "sourceDatasetSha256": model["sourceDatasetSha256"],
        "sourcePolicySha256": model["sourcePolicySha256"],
    }


def convergence_command(
    controls: dict[str, Any],
    root: dict[str, Any],
    strategy: dict[str, Any],
    evaluation: dict[str, Any],
    output: str,
) -> list[str]:
    command = [
        "target/release/preflop-solver",
        "flop-pbs-convergence",
        "--effective-stack-bb",
        str(controls["effectiveStackBb"]),
        "--board",
        root["board"],
        "--pot-bb",
        str(controls["rootPotBb"]),
        "--actor",
        str(controls["rootActor"]),
        "--checkpoints",
        ",".join(str(value) for value in controls["strategyCheckpoints"]),
        "--averaging-delay",
        str(controls["strategyAveragingDelay"]),
        "--threads",
        str(controls["threads"]),
        "--value-network",
        strategy["path"],
        "--evaluation-value-network",
        evaluation["path"],
        "--output",
        output,
    ]
    if bool(controls.get("strategyRegretMatchingPlus", False)):
        command.append("--regret-matching-plus")
    append_dcfr_args(command, dcfr_controls(controls, "strategyDcfr"))
    return command


def response_command(
    controls: dict[str, Any],
    convergence_output: str,
    evaluation: dict[str, Any],
    output: str,
) -> list[str]:
    command = [
        "target/release/preflop-solver",
        "flop-pbs-range-response",
        "--effective-stack-bb",
        str(controls["effectiveStackBb"]),
        "--convergence-report",
        convergence_output,
        "--strategy-iterations",
        str(controls["strategyIterations"]),
        "--evaluation-value-network",
        evaluation["path"],
        "--checkpoints",
        ",".join(str(value) for value in controls["responseCheckpoints"]),
        "--averaging-delay",
        str(controls["responseAveragingDelay"]),
        "--threads",
        str(controls["threads"]),
        "--output",
        output,
    ]
    if bool(controls.get("responseRegretMatchingPlus", False)):
        command.append("--regret-matching-plus")
    append_dcfr_args(command, dcfr_controls(controls, "responseDcfr"))
    return command


def build_plan(
    freeze: dict[str, Any], freeze_path: Path
) -> dict[str, Any]:
    controls = freeze["controls"]
    models = [model_fields(model) for model in freeze["models"]]
    roots = freeze["rootSelection"]["roots"]
    output_directory = controls["outputDirectory"]
    jobs: list[dict[str, Any]] = []
    for root in roots:
        root_label = root["board"].replace(",", "")
        for strategy_index, evaluation_index in ((0, 1), (1, 0)):
            strategy = models[strategy_index]
            evaluation = models[evaluation_index]
            stem = (
                f"flop-{root_label}-strategy{strategy['seed']}"
                f"-evaluation{evaluation['seed']}"
            )
            convergence_output = f"{output_directory}/{stem}-convergence.json"
            response_output = f"{output_directory}/{stem}-range-response.json"
            jobs.append(
                {
                    "name": stem,
                    "root": root,
                    "strategyModel": strategy,
                    "evaluationModel": evaluation,
                    "convergenceOutput": convergence_output,
                    "responseOutput": response_output,
                    "convergenceCommand": convergence_command(
                        controls,
                        root,
                        strategy,
                        evaluation,
                        convergence_output,
                    ),
                    "responseCommand": response_command(
                        controls,
                        convergence_output,
                        evaluation,
                        response_output,
                    ),
                }
            )
    return {
        "schema": PLAN_SCHEMA,
        "status": "frozen-for-fresh-range-response-execution",
        "activationAllowed": False,
        "interpretation": (
            "finite learned-response rejection evidence; not an exploitability upper bound"
        ),
        "freeze": {
            "path": str(freeze_path),
            "sha256": release_freeze.sha256_file(freeze_path),
        },
        "controls": controls,
        "gates": freeze["gates"],
        "models": models,
        "rootCount": len(roots),
        "jobs": jobs,
    }


def _hash_u64(digest: Any, value: int) -> None:
    digest.update(struct.pack("<Q", value))


def _hash_string(digest: Any, value: str) -> None:
    encoded = value.encode()
    _hash_u64(digest, len(encoded))
    digest.update(encoded)


def strategy_sha256(strategies: Any) -> str:
    if not isinstance(strategies, list):
        raise ValueError("frozen flop strategies must be a list")
    digest = hashlib.sha256()
    digest.update(b"hu-flop-strategy-v1\0")
    _hash_u64(digest, len(strategies))
    for strategy in strategies:
        history = strategy.get("public_history")
        labels = strategy.get("action_labels")
        probabilities = strategy.get("probabilities")
        actor = strategy.get("actor")
        if (
            not isinstance(history, list)
            or not all(isinstance(value, str) for value in history)
            or not isinstance(labels, list)
            or not all(isinstance(value, str) for value in labels)
            or not isinstance(actor, int)
            or not isinstance(probabilities, list)
        ):
            raise ValueError("frozen flop strategy cannot be canonically hashed")
        _hash_u64(digest, len(history))
        for value in history:
            _hash_string(digest, value)
        _hash_u64(digest, actor)
        _hash_u64(digest, len(labels))
        for value in labels:
            _hash_string(digest, value)
        _hash_u64(digest, len(probabilities))
        for value in probabilities:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("frozen flop strategy has a non-finite probability")
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def valid_root_state(
    state: Any, root: dict[str, Any], controls: dict[str, Any]
) -> bool:
    board = root["boardIndices"]
    return (
        isinstance(state, dict)
        and state.get("street") == "flop"
        and state.get("board") == board
        and int(state.get("actor", -1)) == int(controls["rootActor"])
        and state.get("invested_bb")
        == [float(controls["rootPotBb"]) / 2.0] * 2
        and state.get("street_invested_bb") == [0.0, 0.0]
        and state.get("public_history") == ["public_belief:flop_start"]
        and int(state.get("aggressions", -1)) == 0
        and int(state.get("checks", -1)) == 0
        and state.get("raise_reopened") is True
        and matched_v1.valid_uniform_root_ranges(state.get("ranges"), board)
    )


def validate_response_diagnostics(
    payload: dict[str, Any], board: list[int]
) -> tuple[bool, float]:
    strategies = payload.get("final_response_strategies")
    attribution = payload.get("information_set_attribution")
    if (
        not isinstance(strategies, list)
        or len(strategies) != 2
        or not isinstance(attribution, list)
        or len(attribution) != 2
    ):
        return False, math.inf
    maximum_probability_error = 0.0
    for responder in range(2):
        if (
            not isinstance(strategies[responder], list)
            or not strategies[responder]
            or any(
                not isinstance(strategy, dict)
                or strategy.get("actor") != responder
                for strategy in strategies[responder]
            )
            or not isinstance(attribution[responder], list)
            or not attribution[responder]
        ):
            return False, math.inf
        try:
            valid, error = matched_v1.validate_strategy_probabilities(
                strategies[responder], board
            )
        except ValueError:
            return False, math.inf
        maximum_probability_error = max(maximum_probability_error, error)
        if not valid:
            return False, maximum_probability_error
        for node in attribution[responder]:
            labels = node.get("action_labels") if isinstance(node, dict) else None
            frozen = node.get("frozen_action_frequencies") if isinstance(node, dict) else None
            response = (
                node.get("response_action_frequencies")
                if isinstance(node, dict)
                else None
            )
            deltas = node.get("action_frequency_deltas") if isinstance(node, dict) else None
            action_evs = node.get("conditional_action_ev_bb") if isinstance(node, dict) else None
            numeric = [
                node.get("node_reach_probability"),
                node.get("reach_weighted_combo_policy_total_variation"),
                node.get("reach_weighted_primary_action_agreement"),
                node.get("maximum_combo_total_variation"),
            ] if isinstance(node, dict) else []
            ev_numeric = [
                node.get("conditional_frozen_strategy_ev_bb"),
                node.get("conditional_response_strategy_ev_bb"),
                node.get("conditional_best_action_ev_bb"),
                node.get("conditional_frozen_strategy_ev_loss_bb"),
                node.get("conditional_response_strategy_ev_loss_bb"),
            ] if isinstance(node, dict) else []
            if (
                not isinstance(node, dict)
                or node.get("actor") != responder
                or not isinstance(node.get("public_history"), list)
                or not isinstance(labels, list)
                or not labels
                or not all(isinstance(label, str) for label in labels)
                or not all(
                    isinstance(vector, list)
                    and len(vector) == len(labels)
                    and all(
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in vector
                    )
                    for vector in (frozen, response, deltas, action_evs)
                )
                or len(numeric) != 4
                or not all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in numeric
                )
                or not 0.0 < float(numeric[0]) <= 1.0 + 1e-9
                or not all(0.0 <= float(value) <= 1.0 + 1e-9 for value in numeric[1:])
                or abs(sum(float(value) for value in frozen) - 1.0) > 1e-5
                or abs(sum(float(value) for value in response) - 1.0) > 1e-5
                or any(
                    abs(float(response[index]) - float(frozen[index]) - float(deltas[index]))
                    > 1e-6
                    for index in range(len(labels))
                )
                or len(ev_numeric) != 5
                or not all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in ev_numeric
                )
                or float(ev_numeric[3]) < 0.0
                or float(ev_numeric[4]) < 0.0
                or abs(
                    max(float(ev_numeric[2]) - float(ev_numeric[0]), 0.0)
                    - float(ev_numeric[3])
                )
                > 1e-8
                or abs(
                    max(float(ev_numeric[2]) - float(ev_numeric[1]), 0.0)
                    - float(ev_numeric[4])
                )
                > 1e-8
                or any(
                    float(ev_numeric[2]) + 1e-8 < float(value)
                    for value in action_evs
                )
            ):
                return False, maximum_probability_error
            top_combos = node.get("top_combo_deviations")
            if not isinstance(top_combos, list) or len(top_combos) > 20:
                return False, maximum_probability_error
            previous_combo_score = math.inf
            for combo in top_combos:
                combo_frozen = combo.get("frozen_probabilities") if isinstance(combo, dict) else None
                combo_response = combo.get("response_probabilities") if isinstance(combo, dict) else None
                combo_action_evs = combo.get("action_ev_bb") if isinstance(combo, dict) else None
                cards = combo.get("cards") if isinstance(combo, dict) else None
                card_names = combo.get("card_names") if isinstance(combo, dict) else None
                combo_numeric = [
                    combo.get("reach_probability"),
                    combo.get("total_variation"),
                    combo.get("frozen_strategy_ev_bb"),
                    combo.get("response_strategy_ev_bb"),
                    combo.get("best_action_ev_bb"),
                    combo.get("frozen_ev_loss_bb"),
                    combo.get("response_ev_loss_bb"),
                ] if isinstance(combo, dict) else []
                combo_score = (
                    float(combo_numeric[0]) * float(combo_numeric[1])
                    if len(combo_numeric) == 7
                    and all(
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in combo_numeric[:2]
                    )
                    else math.inf
                )
                if (
                    not isinstance(combo, dict)
                    or isinstance(combo.get("combo_key"), bool)
                    or not isinstance(combo.get("combo_key"), int)
                    or int(combo["combo_key"]) < 0
                    or not isinstance(cards, list)
                    or len(cards) != 2
                    or any(
                        isinstance(card, bool)
                        or not isinstance(card, int)
                        or not 0 <= int(card) < 52
                        for card in cards
                    )
                    or len(set(int(card) for card in cards)) != 2
                    or any(int(card) in board for card in cards)
                    or not isinstance(card_names, list)
                    or len(card_names) != 2
                    or not all(isinstance(name, str) and name for name in card_names)
                    or not isinstance(combo.get("hand_class"), str)
                    or not combo["hand_class"]
                    or combo.get("frozen_primary_action") not in labels
                    or combo.get("response_primary_action") not in labels
                    or not all(
                        isinstance(vector, list)
                        and len(vector) == len(labels)
                        and all(
                            not isinstance(value, bool)
                            and isinstance(value, (int, float))
                            and math.isfinite(float(value))
                            for value in vector
                        )
                        for vector in (combo_frozen, combo_response, combo_action_evs)
                    )
                    or len(combo_numeric) != 7
                    or not all(
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in combo_numeric
                    )
                    or not 0.0 < float(combo_numeric[0]) <= 1.0 + 1e-9
                    or not 0.0 < float(combo_numeric[1]) <= 1.0 + 1e-9
                    or combo_score > previous_combo_score + 1e-15
                    or abs(sum(float(value) for value in combo_frozen) - 1.0) > 1e-5
                    or abs(sum(float(value) for value in combo_response) - 1.0) > 1e-5
                    or abs(max(float(value) for value in combo_action_evs) - float(combo_numeric[4]))
                    > 1e-8
                    or abs(
                        sum(
                            float(probability) * float(value)
                            for probability, value in zip(combo_frozen, combo_action_evs)
                        )
                        - float(combo_numeric[2])
                    )
                    > 1e-6
                    or abs(
                        sum(
                            float(probability) * float(value)
                            for probability, value in zip(combo_response, combo_action_evs)
                        )
                        - float(combo_numeric[3])
                    )
                    > 1e-6
                    or abs(
                        max(float(combo_numeric[4]) - float(combo_numeric[2]), 0.0)
                        - float(combo_numeric[5])
                    )
                    > 1e-8
                    or abs(
                        max(float(combo_numeric[4]) - float(combo_numeric[3]), 0.0)
                        - float(combo_numeric[6])
                    )
                    > 1e-8
                ):
                    return False, maximum_probability_error
                previous_combo_score = combo_score
    return True, maximum_probability_error


def inspect_convergence(
    job: dict[str, Any], controls: dict[str, Any], gates: dict[str, Any], path: Path
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    strategy = job["strategyModel"]
    evaluation = job["evaluationModel"]
    final = payload.get("final_solution", {})
    checkpoints = payload.get("checkpoints", [])
    checkpoint_solutions = payload.get("checkpoint_solutions", [])
    expected_checkpoints = [int(value) for value in controls["strategyCheckpoints"]]
    checkpoint_iterations = [int(value.get("iterations", -1)) for value in checkpoints]
    solution_iterations = [
        int(value.get("iterations", -1))
        for value in checkpoint_solutions
        if isinstance(value, dict)
    ]
    structural = (
        payload.get("schema") == CONVERGENCE_SCHEMA
        and payload.get("method") == convergence_method(controls)
        and payload.get("approximate") is True
        and bool(payload.get("regret_matching_plus", False))
        == bool(controls.get("strategyRegretMatchingPlus", False))
        and payload.get("dcfr", DEFAULT_DCFR)
        == dcfr_controls(controls, "strategyDcfr")
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
        and payload.get("evaluation_has_distinct_training_identity") is True
        and valid_root_state(payload.get("state"), job["root"], controls)
        and int(payload.get("averaging_delay", -1))
        == int(controls["strategyAveragingDelay"])
        and int(payload.get("threads", -1)) == int(controls["threads"])
        and checkpoint_iterations == expected_checkpoints
        and solution_iterations == expected_checkpoints
        and checkpoint_solutions[-1] == final
        and final.get("schema") == matched_v1.SOLUTION_SCHEMA
        and final.get("method") == solution_method(controls)
        and final.get("approximate") is True
        and bool(final.get("regret_matching_plus", False))
        == bool(controls.get("strategyRegretMatchingPlus", False))
        and final.get("dcfr", DEFAULT_DCFR)
        == dcfr_controls(controls, "strategyDcfr")
        and float(final.get("effective_stack_bb", float("nan")))
        == float(controls["effectiveStackBb"])
        and int(final.get("value_network_seed", -1)) == strategy["seed"]
        and final.get("value_network_sha256") == strategy["sha256"]
        and final.get("value_network_source_dataset_sha256")
        == strategy["sourceDatasetSha256"]
        and final.get("value_network_source_policy_sha256")
        == strategy["sourcePolicySha256"]
        and int(final.get("evaluation_value_network_seed", -1))
        == evaluation["seed"]
        and final.get("evaluation_value_network_sha256") == evaluation["sha256"]
        and final.get("evaluation_value_network_source_dataset_sha256")
        == evaluation["sourceDatasetSha256"]
        and final.get("evaluation_value_network_source_policy_sha256")
        == evaluation["sourcePolicySha256"]
        and final.get("uses_exact_ranges") is True
        and int(final.get("iterations", -1)) == int(controls["strategyIterations"])
        and int(final.get("averaging_delay", -1))
        == int(controls["strategyAveragingDelay"])
        and int(final.get("threads", -1)) == int(controls["threads"])
        and final.get("strategies")
        and final.get("state") == payload.get("state")
    )
    if not structural:
        raise ValueError(f"range-response convergence provenance/state is invalid: {path}")
    if not all(
        isinstance(checkpoint.get("metrics"), dict)
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in checkpoint["metrics"].values()
        )
        for checkpoint in checkpoints
    ):
        raise ValueError(f"range-response convergence metrics are invalid: {path}")
    solution_structural = all(
        solution.get("schema") == matched_v1.SOLUTION_SCHEMA
        and solution.get("method") == solution_method(controls)
        and solution.get("approximate") is True
        and bool(solution.get("regret_matching_plus", False))
        == bool(controls.get("strategyRegretMatchingPlus", False))
        and solution.get("dcfr", DEFAULT_DCFR)
        == dcfr_controls(controls, "strategyDcfr")
        and float(solution.get("effective_stack_bb", float("nan")))
        == float(controls["effectiveStackBb"])
        and int(solution.get("value_network_seed", -1)) == strategy["seed"]
        and solution.get("value_network_sha256") == strategy["sha256"]
        and solution.get("value_network_source_dataset_sha256")
        == strategy["sourceDatasetSha256"]
        and solution.get("value_network_source_policy_sha256")
        == strategy["sourcePolicySha256"]
        and int(solution.get("evaluation_value_network_seed", -1))
        == evaluation["seed"]
        and solution.get("evaluation_value_network_sha256") == evaluation["sha256"]
        and solution.get("evaluation_value_network_source_dataset_sha256")
        == evaluation["sourceDatasetSha256"]
        and solution.get("evaluation_value_network_source_policy_sha256")
        == evaluation["sourcePolicySha256"]
        and solution.get("evaluation_has_distinct_training_identity") is True
        and solution.get("uses_exact_ranges") is True
        and solution.get("state") == payload.get("state")
        and int(solution.get("averaging_delay", -1))
        == int(controls["strategyAveragingDelay"])
        and int(solution.get("threads", -1)) == int(controls["threads"])
        for solution in checkpoint_solutions
    )
    if not solution_structural:
        raise ValueError(f"range-response checkpoint strategy is invalid: {path}")
    probability_checks = [
        matched_v1.validate_strategy_probabilities(
            solution["strategies"], job["root"]["boardIndices"]
        )
        for solution in checkpoint_solutions
    ]
    probabilities_valid = all(valid for valid, _ in probability_checks)
    maximum_sum_error = max(error for _, error in probability_checks)
    computed_hash = strategy_sha256(final["strategies"])
    if computed_hash != payload.get("final_strategy_sha256"):
        raise ValueError(f"range-response convergence strategy hash is invalid: {path}")
    accepted = probabilities_valid and maximum_sum_error <= float(
        gates["maximumProbabilitySumError"]
    )
    return {
        "path": str(path),
        "sha256": release_freeze.sha256_file(path),
        "strategySha256": computed_hash,
        "maximumProbabilitySumError": maximum_sum_error,
        "accepted": accepted,
    }


def inspect_response(
    job: dict[str, Any],
    controls: dict[str, Any],
    gates: dict[str, Any],
    convergence: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    strategy = job["strategyModel"]
    evaluation = job["evaluationModel"]
    checkpoints = payload.get("checkpoints", [])
    expected_iterations = [int(value) for value in controls["responseCheckpoints"]]
    iterations = [int(value.get("iterations", -1)) for value in checkpoints]
    diagnostics_valid, response_probability_error = validate_response_diagnostics(
        payload, job["root"]["boardIndices"]
    )
    structural = (
        payload.get("schema") == RESPONSE_SCHEMA
        and payload.get("method") == response_method(controls)
        and payload.get("approximate") is True
        and bool(payload.get("response_regret_matching_plus", False))
        == bool(controls.get("responseRegretMatchingPlus", False))
        and payload.get("response_dcfr", DEFAULT_DCFR)
        == dcfr_controls(controls, "responseDcfr")
        and "not an exploitability upper bound" in payload.get("interpretation", "")
        and payload.get("frozen_strategy_sha256") == convergence["strategySha256"]
        and int(payload.get("frozen_strategy_iterations", -1))
        == int(controls["strategyIterations"])
        and int(payload.get("strategy_value_network_seed", -1)) == strategy["seed"]
        and payload.get("strategy_value_network_sha256") == strategy["sha256"]
        and payload.get("strategy_value_network_source_dataset_sha256")
        == strategy["sourceDatasetSha256"]
        and payload.get("strategy_value_network_source_policy_sha256")
        == strategy["sourcePolicySha256"]
        and int(payload.get("evaluation_value_network_seed", -1))
        == evaluation["seed"]
        and payload.get("evaluation_value_network_sha256") == evaluation["sha256"]
        and payload.get("evaluation_value_network_source_dataset_sha256")
        == evaluation["sourceDatasetSha256"]
        and payload.get("evaluation_value_network_source_policy_sha256")
        == evaluation["sourcePolicySha256"]
        and payload.get("evaluation_has_distinct_training_identity") is True
        and valid_root_state(payload.get("state"), job["root"], controls)
        and int(payload.get("response_averaging_delay", -1))
        == int(controls["responseAveragingDelay"])
        and int(payload.get("threads", -1)) == int(controls["threads"])
        and iterations == expected_iterations
        and diagnostics_valid
        and payload.get("validation", {}).get("status") == "diagnostic_only"
    )
    if not structural:
        raise ValueError(f"range-response evidence provenance/state is invalid: {path}")
    baseline = [
        payload.get("baseline_profile_value_p0_bb"),
        payload.get("baseline_profile_value_p1_bb"),
    ]
    numeric_fields = (
        "response_value_p0_bb",
        "response_value_p1_bb",
        "response_gain_p0_bb",
        "response_gain_p1_bb",
        "range_consistent_response_gain_bb_per_hand",
        "maximum_zero_sum_residual_bb",
    )
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in baseline
    ) or not all(
        all(
            isinstance(checkpoint.get(field), (int, float))
            and math.isfinite(float(checkpoint[field]))
            for field in numeric_fields
        )
        for checkpoint in checkpoints
    ):
        raise ValueError(f"range-response evidence has invalid numeric values: {path}")
    consistency_error = max(
        max(
            abs(
                float(checkpoint["range_consistent_response_gain_bb_per_hand"])
                - (
                    float(checkpoint["response_gain_p0_bb"])
                    + float(checkpoint["response_gain_p1_bb"])
                )
                / 2.0
            ),
            abs(
                float(checkpoint["response_gain_p0_bb"])
                - max(
                    0.0,
                    float(checkpoint["response_value_p0_bb"])
                    - float(baseline[0]),
                )
            ),
            abs(
                float(checkpoint["response_gain_p1_bb"])
                - max(
                    0.0,
                    float(checkpoint["response_value_p1_bb"])
                    - float(baseline[1]),
                )
            ),
        )
        for checkpoint in checkpoints
    )
    gains = [
        float(checkpoint["range_consistent_response_gain_bb_per_hand"])
        for checkpoint in checkpoints
    ]
    maximum_gain = max(gains)
    final_increase = max(0.0, gains[-1] - gains[-2])
    maximum_residual = max(
        abs(float(checkpoint["maximum_zero_sum_residual_bb"]))
        for checkpoint in checkpoints
    )
    baseline_residual = abs(float(baseline[0]) + float(baseline[1]))
    checks = {
        "artifactProvenance": True,
        "strategyArtifactLink": True,
        "responseDiagnostics": diagnostics_valid,
        "responseStrategyProbabilitySums": response_probability_error
        <= float(gates["maximumProbabilitySumError"]),
        "gainArithmetic": consistency_error <= 1e-12
        and all(gain >= 0.0 for gain in gains),
        "maximumResponseGain": maximum_gain
        <= float(gates["maximumRangeConsistentResponseGainBbPerHand"]),
        "finalCheckpointIncrease": final_increase
        <= float(gates["maximumFinalCheckpointIncreaseBbPerHand"]),
        "zeroSumResidual": max(maximum_residual, baseline_residual)
        <= float(gates["maximumZeroSumResidualBb"]),
    }
    return {
        "path": str(path),
        "sha256": release_freeze.sha256_file(path),
        "checkpointGainsBbPerHand": gains,
        "maximumResponseGainBbPerHand": maximum_gain,
        "finalCheckpointIncreaseBbPerHand": final_increase,
        "maximumZeroSumResidualBb": max(maximum_residual, baseline_residual),
        "gainArithmeticErrorBb": consistency_error,
        "maximumResponseStrategyProbabilitySumError": response_probability_error,
        "gates": checks,
        "accepted": all(checks.values()),
    }


def inspect_job(
    job: dict[str, Any],
    controls: dict[str, Any],
    gates: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    convergence = inspect_convergence(
        job,
        controls,
        gates,
        resolved(repository_root, job["convergenceOutput"]),
    )
    response = inspect_response(
        job,
        controls,
        gates,
        convergence,
        resolved(repository_root, job["responseOutput"]),
    )
    return {
        "name": job["name"],
        "board": job["root"]["board"],
        "texture": job["root"]["texture"],
        "strategySeed": job["strategyModel"]["seed"],
        "evaluationSeed": job["evaluationModel"]["seed"],
        "convergence": convergence,
        "response": response,
        "accepted": convergence["accepted"] and response["accepted"],
    }


def run_command(command: list[str], repository_root: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as sink:
        subprocess.run(
            command,
            cwd=repository_root,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_job(
    job: dict[str, Any],
    controls: dict[str, Any],
    gates: dict[str, Any],
    repository_root: Path,
    resume: bool,
) -> dict[str, Any]:
    convergence_path = resolved(repository_root, job["convergenceOutput"])
    response_path = resolved(repository_root, job["responseOutput"])
    if not convergence_path.exists():
        convergence_path.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "range-response-strategy-start", "name": job["name"]}), flush=True)
        run_command(
            job["convergenceCommand"],
            repository_root,
            convergence_path.with_suffix(".log"),
        )
    elif not resume:
        raise ValueError(f"refusing to overwrite convergence artifact: {convergence_path}")
    convergence = inspect_convergence(job, controls, gates, convergence_path)
    if not convergence["accepted"]:
        raise ValueError(f"frozen strategy evidence failed: {convergence_path}")

    if not response_path.exists():
        print(json.dumps({"event": "range-response-search-start", "name": job["name"]}), flush=True)
        run_command(
            job["responseCommand"],
            repository_root,
            response_path.with_suffix(".log"),
        )
    elif not resume:
        raise ValueError(f"refusing to overwrite response artifact: {response_path}")
    response = inspect_response(job, controls, gates, convergence, response_path)
    evidence = {
        "name": job["name"],
        "board": job["root"]["board"],
        "strategySeed": job["strategyModel"]["seed"],
        "evaluationSeed": job["evaluationModel"]["seed"],
        "convergence": convergence,
        "response": response,
        "accepted": convergence["accepted"] and response["accepted"],
    }
    print(
        json.dumps(
            {
                "event": "range-response-job-complete",
                "name": job["name"],
                "accepted": evidence["accepted"],
                "maximumGainBbPerHand": response["maximumResponseGainBbPerHand"],
                "finalIncreaseBbPerHand": response["finalCheckpointIncreaseBbPerHand"],
            }
        ),
        flush=True,
    )
    return evidence


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.freeze.resolve().parent.parent
    freeze = response_freeze.validate_freeze(args.freeze, repository_root)
    plan = build_plan(freeze, args.freeze)
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_plan.with_suffix(args.output_plan.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output_plan)
    if not args.execute:
        print(encoded, end="")
        return
    subprocess.run(
        ["cargo", "build", "--release", "--locked"],
        cwd=repository_root,
        check=True,
    )
    rejections = []
    for job in plan["jobs"]:
        evidence = run_job(
            job,
            plan["controls"],
            plan["gates"],
            repository_root,
            args.resume,
        )
        if not evidence["accepted"]:
            rejections.append(job["name"])
            if not args.continue_on_rejection:
                raise ValueError(f"fresh range-response gate rejected {job['name']}")
    print(
        json.dumps(
            {
                "status": "range-response-execution-complete",
                "activationAllowed": False,
                "rejections": rejections,
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rules-pinned pilot evidence audit. This cannot activate or certify a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from cash_profiles import CASH_NETWORK_SCHEMA, CASH_STATE_FEATURE_COUNT, CASH_STATE_FEATURE_SCHEMA, PAYOFF_CONTRACT, profile_rules, rules_digest


def finite(value, low=0., high=float("inf")) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def load_run(directory: Path) -> dict:
    state = json.loads((directory / "state.json").read_text())
    config = state["config"]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rules = profile_rules(config.get("cash_profile"))
    if (rules is None or config.get("cash_rules_sha256") != rules_digest(rules) or state["config_hash"] != digest
            or config.get("cash_network_schema") != CASH_NETWORK_SCHEMA
            or config.get("cash_state_feature_schema") != CASH_STATE_FEATURE_SCHEMA
            or config.get("cash_terminal_action_integration") != "own-expectation-v1"):
        raise ValueError("cash run config/rules digest is invalid")
    rounds = state["completed_rounds"]
    metrics = state["metrics"]
    if (type(rounds) is not int or rounds < 1 or [row["round"] for row in metrics] != list(range(1, rounds + 1))
            or any(not finite(row.get("elapsed_seconds"), 0.000001) or row.get("records_truncated") is not False for row in metrics)
            or state["completed_traversals"] != rounds * config["traversals_per_round"]):
        raise ValueError("cash run lacks complete measured, untruncated round evidence")
    artifact = directory / "artifacts" / f"round-{rounds:06d}"
    source = json.loads((artifact / "model-source.json").read_text())
    weights = (artifact / "average-policy.json").read_bytes()
    weight_hash = hashlib.sha256(weights).hexdigest()
    model = json.loads(weights)
    if (source.get("validation_status") != "training_not_activated" or source.get("config") != config
            or source.get("round") != rounds or source.get("payoff_contract") != PAYOFF_CONTRACT
            or source.get("rules_sha256") != rules_digest(rules) or source.get("cash_rules") != rules
            or source.get("native_average_policy_sha256") != weight_hash
            or model.get("schema") != CASH_NETWORK_SCHEMA or model.get("strategy_transform") != "softmax"
            or model.get("input_size") != CASH_STATE_FEATURE_COUNT + 9
            or model.get("cash_rules") != rules or model.get("cash_depth_bb") != config["depth_bb"]):
        raise ValueError("frozen cash average weights differ from their measured training source")
    return dict(config=config, rules=rules, weights_sha256=weight_hash, weights_bytes=len(weights),
                action_abstraction=model["cash_action_abstraction"], rounds=rounds,
                training_hours=sum(row["elapsed_seconds"] for row in metrics) / 3600)


def validate_evaluation(report: dict, run: dict, schema: str) -> None:
    game = report["game"]
    if (report.get("schema") != schema or report.get("validation_status") != "research_only"
            or report.get("rules_sha256") != rules_digest(run["rules"])
            or report.get("policy_sha256") != run["weights_sha256"]
            or game.get("cash_rules") != run["rules"] or game.get("effective_stack_bb") != run["config"]["depth_bb"]
            or game.get("action_abstraction") != run["action_abstraction"]
            or game.get("small_blind_bb") != run["rules"]["blindsUnits"][0] / run["rules"]["unitsPerBb"]
            or game.get("big_blind_bb") != 1 or report.get("confidence") != .99
            or type(report.get("deals")) is not int or report["deals"] < 2):
        raise ValueError("cash evaluation crossed rules, weights, depth or abstraction")


def audit(runs: list[dict], full_hands: list[dict], responses: list[dict]) -> dict:
    if len(runs) != 2 or len(full_hands) != 2 or len(responses) != 2:
        raise ValueError("cash audit requires a paired run and paired evaluations")
    comparable = [{k: v for k, v in run["config"].items() if k != "seed"} for run in runs]
    if (comparable[0] != comparable[1] or runs[0]["config"]["seed"] == runs[1]["config"]["seed"]
            or runs[0]["rounds"] != runs[1]["rounds"] or runs[0]["action_abstraction"] != runs[1]["action_abstraction"]):
        raise ValueError("cash seeds are not independent matched configurations")
    for index, (run, full, response) in enumerate(zip(runs, full_hands, responses)):
        validate_evaluation(full, run, "hu-cash-full-hand-restricted-evaluation-v1")
        validate_evaluation(response, run, "hu-cash-causal-sample-game-nash-conv-v1")
        if full.get("comparison_sha256") != runs[1-index]["weights_sha256"]:
            raise ValueError("cross-seed comparison does not use the other frozen weights")
        for baseline in (full["baseline"], full["comparison_baseline"]):
            values, house = baseline["mean_own_payoff_bb"], baseline["mean_house_rake_bb"]
            cap = run["rules"]["rake"]["capUnits"] / run["rules"]["unitsPerBb"]
            depth = run["config"]["depth_bb"]
            if (len(values) != 2 or any(not finite(v, -depth, depth) for v in values) or not finite(house, 0, cap)
                    or abs(sum(values) + house) > 1e-8 or not finite(baseline["maximum_accounting_residual_bb"], 0, 1e-8)):
                raise ValueError("cash evaluation own payoffs do not reconcile independent house rake")
        if (not finite(full.get("comparison_action_frequency_mae"), 0, 1)
                or not finite(full.get("comparison_primary_agreement"), 0, 1)
                or not full.get("comparison_aggregate_action_deltas")
                or any(not finite(delta, -1, 1) for delta in full["comparison_aggregate_action_deltas"].values())
                or full["diagnostics"]["comparison_rows"] != full["diagnostics"]["valid_probability_rows"]
                or full["diagnostics"]["comparison_rows"] <= 0):
            raise ValueError("cash comparison has invalid frequency/coverage diagnostics")
        depth = run["config"]["depth_bb"]
        for deviation in full["restricted_responses"]:
            if (not finite(deviation["mean_gain_bb"], -2*depth, 2*depth)
                    or not finite(deviation["maximum_accounting_residual_bb"], 0, 1e-8)):
                raise ValueError("cash fixed-response accounting/gains are invalid")
        mean, upper = response["sample_mean_total_nash_conv_bb"], response["abstract_game_optimistic_upper_bound_bb"]
        if not finite(mean) or not finite(upper, mean) or len(response["samples"]) != response["deals"]:
            raise ValueError("cash response evidence has invalid means/bounds/sample counts")
        totals = []
        for sample in response["samples"]:
            if len(sample) != 2:
                raise ValueError("cash response omits a seat")
            for seat, row in enumerate(sample):
                for ledger in (row["baseline"], row["response"]):
                    if (len(ledger["own_payoffs_bb"]) != 2
                            or any(not finite(v, -depth, depth) for v in ledger["own_payoffs_bb"])
                            or not finite(ledger["house_rake_bb"], 0, cap)
                            or abs(sum(ledger["own_payoffs_bb"]) + ledger["house_rake_bb"]) > 1e-8
                            or not finite(ledger["maximum_accounting_residual_bb"], 0, 1e-8)):
                        raise ValueError("cash response own utilities do not reconcile house rake")
                gain = max(0., row["response"]["own_payoffs_bb"][seat] - row["baseline"]["own_payoffs_bb"][seat])
                if not finite(row["own_gain_bb"], 0, 2*depth) or abs(gain-row["own_gain_bb"]) > 1e-8:
                    raise ValueError("cash response gain differs from its own-player values")
            totals.append(sum(row["own_gain_bb"] for row in sample))
        margin = 4 * depth * math.sqrt(math.log(100) / (2*response["deals"]))
        if (abs(sum(totals)/len(totals) - mean) > 1e-8
                or abs(response["one_sided_hoeffding_margin_bb"]-margin) > 1e-8
                or abs(upper-min(4*depth, mean+margin)) > 1e-8):
            raise ValueError("cash response means/bounds disagree with measured samples")
    for reports in (full_hands, responses):
        if any(reports[0][field] != reports[1][field] for field in ("seed", "deals")):
            raise ValueError("paired cash evaluations must use matched chance budgets")

    def gate(value, target, minimum=False):
        passed = value >= target if minimum else value <= target
        return {"status": "passed" if passed else "failed", "measured": value,
                "minimum" if minimum else "maximum": target}

    mae = max(p["comparison_action_frequency_mae"] for p in full_hands)
    agreement = min(p["comparison_primary_agreement"] for p in full_hands)
    delta = max(abs(d) for p in full_hands for d in p["comparison_aggregate_action_deltas"].values())
    hours = [run["training_hours"] for run in runs]
    upper = max(p["abstract_game_optimistic_upper_bound_bb"] for p in responses)
    gates = dict(cross_seed_frequency_mae=gate(mae, .05), primary_action_agreement=gate(agreement, .85, True),
                 maximum_aggregate_action_delta=gate(delta, .03),
                 measured_training_duration=dict(status="passed" if all(8 <= h <= 12 for h in hours) else "failed", hours_per_seed=hours, required_hours=[8, 12]),
                 raw_probability_sums=dict(status="passed", scope="checked native inference rows only"),
                 cash_chip_conservation=dict(status="passed", scope="reported complete-hand baselines and deviations"),
                 total_deviation_bound=dict(status="failed" if upper > .5 else "unqualified", measured=upper, maximum=.5,
                     scope="optimistic finite sample-game research bound for the pinned sizing abstraction; not an unrestricted certificate"))
    for name, reason in {
        "policy_lookup_coverage": "Valid inference rows do not measure training support or held-out authentic/forced-deviation serving coverage.",
        "quantized_probability_sums": "No quantized cash serving artifact was evaluated.",
        "reach_weighted_action_ev_precision": "Fixed CLI snapshots are not 95% reach-weighted serving precision evidence.",
        "projected_total_hosted_storage": "Measured pilot weight bytes do not project all accepted depths and serving artifacts.",
        "full_routed_serving": "No accepted cash continuation route or production manifest exists.",
        "browser_acceptance": "No complete cash browser-mode acceptance evidence was supplied.",
    }.items():
        gates[name] = dict(status="unmeasured", reason=reason)
    return dict(schema="hu-cash-paired-pilot-qualification-v1", status="failed" if any(g["status"] == "failed" for g in gates.values()) else "incomplete",
                validation_status="research_only", release_eligible=False,
                profile_id=runs[0]["rules"]["id"], rules_sha256=rules_digest(runs[0]["rules"]),
                depth_bb=runs[0]["config"]["depth_bb"], seeds=[r["config"]["seed"] for r in runs],
                weights_sha256=[r["weights_sha256"] for r in runs], measured_weight_bytes=[r["weights_bytes"] for r in runs],
                gates=gates, limitations=["This audit cannot activate a manifest or certify Approximate GTO.",
                    "Cross-seed stability and restricted response gains are not equilibrium proof.",
                    "Serving/model bias is distinct from Monte Carlo sampling error."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs=2, required=True)
    parser.add_argument("--full-hand-report", type=Path, nargs=2, required=True)
    parser.add_argument("--response-report", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit([load_run(p) for p in args.run_dir],
                   [json.loads(p.read_text()) for p in args.full_hand_report],
                   [json.loads(p.read_text()) for p in args.response_report])
    report["source_capture_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (*args.full_hand_report, *args.response_report)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

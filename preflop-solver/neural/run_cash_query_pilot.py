#!/usr/bin/env python3
"""Bounded end-to-end native cash queries. Never activates a practice model."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from cash_profiles import CASH_NETWORK_SCHEMA, rules_digest


def validate_result(result: dict, request: dict, samples: int) -> None:
    query = request["query"]
    exact = result.get("evEvaluation") == "exact-river-policy-expectation"
    mixed = result.get("evEvaluation") == "paired-monte-carlo-with-exact-all-ins"
    if (result.get("schema") != "hu-cash-average-policy-result-v1"
            or result.get("validationStatus") != "research_only"
            or result.get("rulesSha256") != request["rulesSha256"]
            or result.get("networkSha256") != request["networkSha256"]
            or any(result.get(key) != query[key] for key in ("requestId", "modelVersion", "depthBb", "stateHash"))
            or result.get("evEvaluation") not in ("exact-river-policy-expectation","paired-full-hand-monte-carlo","paired-monte-carlo-with-exact-all-ins")
            or result.get("rolloutSamplesPerAction") != (0 if exact else samples)
            or (exact and (query.get("street") != "river" or not 0 < result.get("exactPolicyTreeNodes",0) <= 2_000_000))
            or (not exact and result.get("exactPolicyTreeNodes") != 0)
            or type(result.get("exactAllInChanceOutcomes")) is not int
            or not 0 <= result["exactAllInChanceOutcomes"] <= 2_000_000
            or (not mixed and result["exactAllInChanceOutcomes"] != 0)
            or (mixed and query.get("street") not in ("flop","turn"))
            or not 0 <= result.get("maximumAccountingResidualBb", float("inf")) <= 1e-8
            or not 0 <= result.get("maximumProbabilitySumError", float("inf")) <= 1e-6):
        raise ValueError("cash query crossed pinned identity or accounting")
    actions = result.get("actions", [])
    methods = result.get("actionEvEvaluations",[])
    if len(methods) != len(actions) or any(m not in ("paired-full-hand-monte-carlo","exact-river-policy-expectation","exact-closed-all-in-expectation") for m in methods):
        raise ValueError("cash query lacks per-action exact/sampled provenance")
    if ((exact and any(m != "exact-river-policy-expectation" for m in methods))
            or (not exact and "exact-river-policy-expectation" in methods)
            or (mixed != ("exact-closed-all-in-expectation" in methods))):
        raise ValueError("cash query exact action methods disagree with its evaluation route")
    if not actions or any(a.get("confidence") != "low"
            or any(not isinstance(a.get(key), (int,float)) or isinstance(a[key], bool) or not math.isfinite(a[key])
                   for key in ("probability", "evBb", "standardErrorBb"))
            or not 0 <= a["probability"] <= 1 or a["standardErrorBb"] < 0
            or abs(a["evBb"]) > query["depthBb"] + 1e-8 for a in actions):
        raise ValueError("cash query returned invalid own-value/probability rows")
    if abs(sum(a["probability"] for a in actions)-1) > 1e-6:
        raise ValueError("cash query mix is not normalized")
    if exact and any(a["standardErrorBb"] != 0 for a in actions):
        raise ValueError("enumerated river cannot report Monte Carlo sampling error")
    if any(m != "paired-full-hand-monte-carlo" and a["standardErrorBb"] != 0 for a,m in zip(actions,methods)):
        raise ValueError("enumerated all-in cannot report Monte Carlo sampling error")


def run(binary: Path, weights: Path, output: Path, samples: int, seed: int = 923) -> dict:
    if not 16 <= samples <= 4096:
        raise ValueError("cash query pilot supports 16..4096 samples per action")
    raw = weights.read_bytes(); model = json.loads(raw)
    if model.get("schema") != CASH_NETWORK_SCHEMA or model.get("strategy_transform") != "softmax" or model.get("cash_depth_bb") != 20:
        raise ValueError("cash query pilot requires frozen 20bb own-payoff average weights")
    rules = model["cash_rules"]
    if rules["id"] != "pokerstars-usd-regular-hu-nl25-2026-10-06":
        raise ValueError("query fixtures belong to the frozen NL25 game")
    fixture = Path(__file__).resolve().parents[2] / "data/practice/cash-query-fixtures.json"
    cases = json.loads(fixture.read_text())["cases"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        query = dict(case["query"], stateHash=case["stateHash"])
        request = {"schema":"hu-cash-average-policy-query-v1", "rulesSha256":rules_digest(rules),
                   "networkSha256":hashlib.sha256(raw).hexdigest(), "query":query}
        budget = {"request":request, "rollout_samples":samples,"seed":seed,
                  "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}
        digest = hashlib.sha256(json.dumps(budget,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        input_path, result_path = output / f"query-{digest}.json", output / f"result-{digest}.json"
        input_path.write_text(json.dumps(request,separators=(",", ":")) + "\n")
        cached = result_path.is_file(); started = time.monotonic()
        if not cached:
            result = subprocess.run([str(binary.resolve()),"cash-policy-query","--cash-profile","nl25",
                "--model-version",query["modelVersion"],"--compact-serving-grid","--networks",str(weights.resolve()),
                "--input",str(input_path),"--rollout-samples",str(samples),"--seed",str(seed),"--output",str(result_path)],
                capture_output=True,text=True,timeout=120)
            if result.returncode:
                raise RuntimeError(f"cash query {case['name']} failed: {result.stderr.strip()}")
        result = json.loads(result_path.read_text()); validate_result(result,request,samples)
        rows.append({"name":case["name"],"elapsed_seconds":time.monotonic()-started,"cached":cached,
            "ev_evaluation":result["evEvaluation"],"exact_policy_tree_nodes":result["exactPolicyTreeNodes"],
            "exact_all_in_chance_outcomes":result["exactAllInChanceOutcomes"],"action_ev_evaluations":result["actionEvEvaluations"],
            "result_file":str(result_path),"maximum_standard_error_bb":max(a["standardErrorBb"] for a in result["actions"]),
            "action_count":len(result["actions"]),"actions_with_se_at_most_002bb":sum(a["standardErrorBb"] <= .02 for a in result["actions"]),
            "maximum_accounting_residual_bb":result["maximumAccountingResidualBb"]})
    report = {"schema":"hu-cash-native-query-pilot-v1","status":"research_only","active":False,
        "rules_sha256":rules_digest(rules),"weights_sha256":hashlib.sha256(raw).hexdigest(),"rollout_samples_per_action":samples,
        "seed":seed,"queries":rows,"limitations":["fixed parity snapshots, not authentic/forced-deviation coverage",
        "river enumeration has zero sampling error, not zero policy/posterior-model error",
        "action counts are unweighted diagnostics, not the 95% reach-weighted EV precision gate",
        "CLI process-start latency, not website worker/browser latency; no website activation"]}
    (output / "report.json").write_text(json.dumps(report,indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--native-binary",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=128); parser.add_argument("--seed",type=int,default=923)
    args = parser.parse_args()
    print(json.dumps(run(args.native_binary,args.weights,args.output,args.samples,args.seed),indent=2))

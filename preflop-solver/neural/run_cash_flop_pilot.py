#!/usr/bin/env python3
"""Bounded, frozen-weight flop continuation-sensitivity pilot; no activation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from cash_turn_roots import validate_roots
from native_value_dataset import compatible_masses


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_solution(result: dict, config: dict, network_sha: str, rules_sha: str) -> None:
    if (result.get("schema") != "hu-cash-depth-limited-flop-pilot-v1"
            or result.get("validation", {}).get("status") != "research_only"
            or result.get("full_game_exploitability") != "unmeasured"
            or result.get("expected_house_rake_bb") is not None
            or result.get("continuation_model_confidence") != "unqualified-research-only"
            or result.get("value_network_sha256") != network_sha
            or result.get("rules_sha256") != rules_sha
            or result.get("iterations") != config["iterations"]
            or result.get("threads") != config["threads"]
            or result.get("game") != config["game"]
            or result.get("action_value_method") != "exact-public-chance-and-terminals-with-learned-own-payoff-turn-leaves-v1"):
        raise ValueError("cash flop result crossed pinned identity, budget, or research status")
    state = result.get("state", {})
    for key, value in config["state"].items():
        if key == "ranges":
            actual, expected = np.asarray(state.get(key)), np.asarray(value)
            if actual.shape != (2,1326) or not np.allclose(actual,expected,atol=1e-12,rtol=0):
                raise ValueError("cash flop result changed the public ranges")
        elif state.get(key) != value:
            raise ValueError("cash flop result changed the public state")
    net = np.asarray(result.get("profile_net_bb"))
    if (net.shape != (2,) or not np.isfinite(net).all()
            or (np.abs(net) > config["game"]["effective_stack_bb"]+1e-6).any()
            or abs(net.sum()-result.get("predicted_profile_payoff_sum_bb",float("inf"))) > 1e-9):
        raise ValueError("cash flop own-payoff summary is invalid")
    root = result.get("root", {})
    if root.get("public_history") != config["state"]["public_history"] or root.get("actor") != config["state"]["actor"]:
        raise ValueError("cash flop root does not match its public input")
    strategies = result.get("strategies", [])
    if not strategies or root not in strategies:
        raise ValueError("cash flop result lost its frozen root policy")
    for row in strategies:
        actor = row.get("actor")
        labels = row.get("action_labels", [])
        if actor not in (0,1) or not labels or len(set(labels)) != len(labels):
            raise ValueError("cash flop strategy has invalid actions")
        probabilities, values = np.asarray(row.get("probabilities")), np.asarray(row.get("action_values_bb"))
        if (probabilities.shape != (1326*len(labels),) or values.shape != probabilities.shape
                or not np.isfinite(probabilities).all() or not np.isfinite(values).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or (np.abs(values) > config["game"]["effective_stack_bb"]+1e-5).any()):
            raise ValueError("cash flop strategy has invalid probabilities/action estimates")
        legal = np.asarray(config["state"]["ranges"])[actor] > 0
        probabilities = probabilities.reshape(1326,-1)
        if not np.allclose(probabilities[legal].sum(axis=1),1.,atol=1e-6,rtol=0) or (probabilities[~legal] != 0).any():
            raise ValueError("cash flop strategy probabilities do not respect legal support")


def generate(binary: Path, output: Path, root: dict, network: Path, rules_sha: str):
    request = dict(schema="cash-flop-frozen-pilot-request-v2", root_sha256=root["root_sha256"],
                   input=root["solve_input"], value_network_sha256=sha(network), native_binary_sha256=sha(binary))
    digest = hashlib.sha256(json.dumps(request,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    path, capture = output/f"solution-{digest}.json", output/f"capture-{digest}.json"
    cached = path.exists()
    start = time.monotonic()
    if not cached:
        input_path = output/f"input-{digest}.json"
        input_path.write_text(json.dumps(request["input"]) + "\n")
        subprocess.run([str(binary.resolve()),"cash-flop-pilot","--input",str(input_path),
                        "--value-network",str(network.resolve()),"--output",str(path)],check=True,capture_output=True,text=True,timeout=1800)
        capture.write_text(json.dumps(dict(request=request,solution_sha256=sha(path)),indent=2) + "\n")
    if not capture.exists() or json.loads(capture.read_text()) != dict(request=request,solution_sha256=sha(path)):
        raise ValueError("cash flop cached solution lacks its immutable capture")
    result = json.loads(path.read_text())
    validate_solution(result,request["input"],request["value_network_sha256"],rules_sha)
    return dict(path=str(path),root_sha256=root["root_sha256"],value_network_sha256=request["value_network_sha256"],
                native_binary_sha256=request["native_binary_sha256"],cache_hit=cached,elapsed_seconds=time.monotonic()-start)


def run(binary: Path, roots_path: Path, policy: Path, networks: list[Path], output: Path, profile="nl25", workers=2):
    if workers not in (1,2) or len(networks) != 2:
        raise ValueError("cash flop pilot permits exactly two frozen value weights and 1..2 processes")
    source = json.loads(roots_path.read_text())
    validate_roots(source,json.loads(policy.read_text()),sha(policy),profile,street="flop")
    if len(source["roots"]) > 4:
        raise ValueError("initial cash flop pilot is capped at four authentic roots")
    output.mkdir(parents=True,exist_ok=True)
    tasks = [(root,network) for root in source["roots"] for network in networks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda task: generate(binary,output,*task,source["rules_sha256"]),tasks))
    comparisons = []
    for index,root in enumerate(source["roots"]):
        left,right = [json.loads(Path(row["path"]).read_text()) for row in rows[index*2:index*2+2]]
        if left["root"]["action_labels"] != right["root"]["action_labels"]:
            raise ValueError("paired cash flop legal actions differ")
        probabilities = [np.asarray(r["root"]["probabilities"]).reshape(1326,-1) for r in (left,right)]
        values = [np.asarray(r["root"]["action_values_bb"]).reshape(1326,-1) for r in (left,right)]
        ranges = np.asarray(root["solve_input"]["state"]["ranges"])
        actor = left["root"]["actor"]
        reach = (ranges*compatible_masses(ranges))[actor]; reach /= reach.sum()
        comparisons.append(dict(root_sha256=root["root_sha256"],board=root["solve_input"]["state"]["board"],
            gross_pot_bb=sum(root["solve_input"]["state"]["invested_bb"]),action_labels=left["root"]["action_labels"],
            root_probability_mae=float(np.sum(reach[:,None]*np.abs(probabilities[0]-probabilities[1]))/probabilities[0].shape[1]),
            root_primary_agreement=float(np.clip(np.sum(reach*(probabilities[0].argmax(axis=1)==probabilities[1].argmax(axis=1))),0.,1.)),
            weighted_action_ev_mae_bb=float(np.sum(reach[:,None]*np.abs(values[0]-values[1]))/values[0].shape[1]),
            aggregate_action_frequencies=[(reach @ p).tolist() for p in probabilities],
            profile_net_bb=[left["profile_net_bb"],right["profile_net_bb"]]))
    report = dict(schema="cash-flop-paired-oracle-sensitivity-pilot-v1",status="research_only",active=False,
                  roots_sha256=sha(roots_path),source_policy_sha256=sha(policy),value_network_sha256s=[sha(n) for n in networks],
                  native_binary_sha256=sha(binary),captures=rows,comparisons=comparisons,full_game_exploitability="unmeasured",
                  limitations=["Same deterministic DCFR solve under two value hypotheses; not independent full-hand policy seeds.",
                               "These conditional root comparisons do not satisfy release coverage, confidence, or equilibrium gates."])
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser()
    for key in ("native-binary","roots","policy","output"): parser.add_argument(f"--{key}",type=Path,required=True)
    parser.add_argument("--value-network",type=Path,required=True,action="append")
    parser.add_argument("--workers",type=int,default=2); parser.add_argument("--profile",default="nl25")
    args=parser.parse_args()
    print(json.dumps(run(args.native_binary,args.roots,args.policy,args.value_network,args.output,args.profile,args.workers),indent=2))


if __name__ == "__main__": main()

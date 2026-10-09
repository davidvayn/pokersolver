#!/usr/bin/env python3
"""Exact fresh-turn reference probes on a pinned flop policy's actual leaves.

Only the check/check branch and two public turn proposals per root are tested.
This is a conditional distribution-shift diagnostic, not full-hand validation.
Both students see identical beliefs under one frozen flop policy per root.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from cash_turn_roots import validate_roots
from cash_flop_leaves import leaf_input
from cash_value_dataset import validate_label
from native_value_dataset import compatible_masses
from run_cash_flop_pilot import sha, validate_solution


def check_check_input(root: dict, solution: dict, turn: int, iterations: int = 64) -> dict:
    return leaf_input(root, solution, turn, ["check", "check"], iterations)


def reference(binary: Path, output: Path, config: dict, provenance: dict):
    # Reference solving depends on the pinned leaf and binary, not which new
    # students are compared afterward. Reuse exact labels across paired fits.
    request=dict(schema="cash-frozen-flop-leaf-probe-request-v2",input=config,provenance=provenance,
                 native_binary_sha256=sha(binary))
    digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    input_path=output/f"input-{digest}.json"; label_path=output/f"reference-{digest}.json"
    capture_path=output/f"capture-{digest}.json"
    if not label_path.exists():
        input_path.write_text(json.dumps(config)+"\n")
        subprocess.run([str(binary.resolve()),"turn-river-pbs-solve","--input",str(input_path),"--output",str(label_path)],
                       capture_output=True,text=True,check=True,timeout=1800)
        capture_path.write_text(json.dumps(dict(request=request,reference_sha256=sha(label_path)))+"\n")
    if not input_path.exists() or json.loads(input_path.read_text()) != config:
        raise ValueError("turn probe input differs from its pinned leaf")
    if not capture_path.exists() or json.loads(capture_path.read_text()) != dict(request=request,reference_sha256=sha(label_path)):
        raise ValueError("turn probe cache crossed its frozen input/reference")
    label=json.loads(label_path.read_text()); validate_label(label)
    if label["input"] != config: raise ValueError("turn reference changed its requested game/state")
    return label_path, label


def compare(binary: Path, config: dict, label: dict, networks: list[Path], input_path: Path):
    ranges=np.asarray(config["state"]["ranges"]); joint_weights=ranges*compatible_masses(ranges)
    joint=joint_weights.sum(axis=1)
    teacher=np.asarray(label["counterfactual_values_bb"])
    cash=label["metrics"]["cash"]
    comparisons=[]
    for network in networks:
        result=subprocess.run([str(binary.resolve()),"cash-turn-value-predict","--input",str(input_path),"--value-network",str(network.resolve())],
                              capture_output=True,text=True,check=True,timeout=120)
        prediction=json.loads(result.stdout)
        if prediction["rules_sha256"] != cash["rules_sha256"] or not prediction["research_only"]:
            raise ValueError("student query crossed the reference's own-payoff rules")
        values=np.asarray(prediction["counterfactual_values_bb"])
        if values.shape != (2,1326) or not np.isfinite(values).all(): raise ValueError("invalid student values")
        means=np.sum(joint_weights*values,axis=1)/joint
        comparisons.append(dict(network_sha256=sha(network),
            reach_weighted_cfv_rmse_bb=float(np.sqrt(np.sum(joint_weights*(values-teacher)**2)/joint_weights.sum())),
            own_profile_value_bb=means.tolist(),
            signed_house_accounting_bias_bb=float(means.sum()+cash["expected_house_rake_bb"])))
    return comparisons


def evaluate(binary: Path, output: Path, config: dict, networks: list[Path], provenance: dict):
    path, label = reference(binary, output, config, provenance)
    input_path = path.with_name(path.name.replace("reference-", "input-"))
    return dict(board=config["state"]["board"], gross_pot_bb=sum(config["state"]["invested_bb"]),
                provenance=provenance, reference_sha256=sha(path), reference_path=str(path),
                reference_cash_metrics=label["metrics"]["cash"],
                comparisons=compare(binary, config, label, networks, input_path))


def run(binary: Path, roots_path: Path, policy: Path, pair_report: Path, networks: list[Path], output: Path, seed=967):
    if len(networks) != 2: raise ValueError("leaf probe requires two frozen value hypotheses")
    roots=json.loads(roots_path.read_text()); pair=json.loads(pair_report.read_text())
    validate_roots(roots,json.loads(policy.read_text()),sha(policy),"nl25",street="flop")
    if (len(roots["roots"]) > 4 or pair["roots_sha256"] != sha(roots_path)
            or pair["native_binary_sha256"] != sha(binary) or pair["status"] != "research_only"
            or pair["value_network_sha256s"] != [sha(n) for n in networks]):
        raise ValueError("leaf probe crossed its frozen flop comparison")
    output.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(seed); tasks=[]
    for index,root in enumerate(roots["roots"]):
        path=Path(pair["captures"][index*2]["path"]); solution=json.loads(path.read_text())
        validate_solution(solution,root["solve_input"],sha(networks[0]),roots["rules_sha256"])
        deck=[c for c in range(52) if c not in root["solve_input"]["state"]["board"]]
        for turn in rng.choice(deck,size=2,replace=False):
            config=check_check_input(root,solution,int(turn))
            provenance=dict(root_sha256=root["root_sha256"],flop_solution_sha256=sha(path),
                            branch="check/check",public_turn_proposal=int(turn),sampling_seed=seed)
            tasks.append((config,provenance))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda task:evaluate(binary,output,task[0],networks,task[1]),tasks))
    report=dict(schema="cash-frozen-flop-turn-leaf-probe-v1",status="research_only",active=False,
                source_pair_report_sha256=sha(pair_report),native_binary_sha256=sha(binary),rows=rows,
                full_game_exploitability="unmeasured",
                limitations=["Only check/check and two public turn proposals per root; not full chance coverage.",
                             "Fresh 64-update turn/river references are not a pinned complete served continuation.",
                             "Continuation errors are diagnostics, not additional release gates."])
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def reevaluate(binary: Path, original: Path, networks: list[Path], output: Path):
    report = json.loads(original.read_text())
    if (report.get("schema") != "cash-frozen-flop-turn-leaf-probe-v1"
            or report.get("status") != "research_only" or report.get("active") is not False
            or report.get("native_binary_sha256") != sha(binary)
            or not 1 <= len(report.get("rows", [])) <= 8 or len(networks) != 2):
        raise ValueError("student comparison requires a pinned bounded leaf reference report")
    rows = []
    for row in report["rows"]:
        path = Path(row["reference_path"])
        if sha(path) != row["reference_sha256"]:
            raise ValueError("leaf reference changed after its diagnostic capture")
        label = json.loads(path.read_text()); validate_label(label)
        config = label["input"]
        checked_path, checked_label = reference(binary, path.parent, config, row["provenance"])
        if checked_path != path or checked_label != label:
            raise ValueError("leaf comparison crossed its immutable capture")
        updated = dict(row)
        updated["comparisons"] = compare(binary, config, label, networks,
                                          path.with_name(path.name.replace("reference-", "input-")))
        rows.append(updated)
    result = dict(report, rows=rows, source_reference_report_sha256=sha(original),
                  compared_network_sha256s=[sha(n) for n in networks])
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for key in ("binary", "output"): parser.add_argument(f"--{key}",required=True,type=Path)
    for key in ("roots", "policy", "pair-report", "reference-report"): parser.add_argument(f"--{key}",type=Path)
    parser.add_argument("--value-network",action="append",required=True,type=Path)
    args=parser.parse_args()
    if args.reference_report:
        if any((args.roots, args.policy, args.pair_report)):
            parser.error("reference comparison cannot replace its captured roots or flop policy")
        result = reevaluate(args.binary, args.reference_report, args.value_network, args.output)
    else:
        if not all((args.roots, args.policy, args.pair_report)):
            parser.error("initial probe requires roots, policy, and pair-report")
        result = run(args.binary,args.roots,args.policy,args.pair_report,args.value_network,args.output)
    print(json.dumps(result,indent=2))

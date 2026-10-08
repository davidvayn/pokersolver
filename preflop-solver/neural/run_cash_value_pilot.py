#!/usr/bin/env python3
"""Bounded exact-label preflight. No cloud calls, model activation, or promises."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from cash_profiles import profile_rules, rules_digest
from cash_value_dataset import build_dataset, validate_label, validate_dataset
from cash_turn_roots import load_roots
from native_value_dataset import identity_hash

BOARDS = ("4c,5d,7h,9s", "Ac,Kd,8h,2s", "Tc,9d,6h,3s", "Qc,Jd,4h,8s", "2c,3d,5h,Ks", "7c,7d,Jh,As")


def generate(binary: Path, directory: Path, profile: str, board: str, iterations: int, pot: float):
    request = {"schema":"cash-turn-label-request-v1", "rules_sha256":rules_digest(profile_rules(profile)),
               "native_binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
               "board":board, "depth_bb":20,"pot_bb":pot,"iterations":iterations,
               "averaging_delay":0,"action_abstraction":"compact-serving-candidate"}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = directory / f"label-{digest}.json"
    started = time.monotonic()
    cache_hit = path.exists()
    if not cache_hit:
        result = subprocess.run([str(binary.resolve()), "turn-river-pbs-solve", "--cash-profile", profile,
                        "--board", board, "--effective-stack-bb", "20", "--pot-bb", str(pot),
                        "--iterations", str(iterations), "--averaging-delay", "0", "--compact-serving-grid", "--output", str(path)],
                       text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"native cash labels failed for {board}: {result.stderr.strip()}")
    label = json.loads(path.read_text()); validate_label(label)
    if (label["metrics"]["cash"]["rules_sha256"] != request["rules_sha256"]
            or label["joint_iterations"] != iterations or label["input"]["state"]["invested_bb"] != [pot/2,pot/2]):
        raise ValueError("cached cash label differs from requested frozen game/budget")
    return path, {"input_sha256":digest,"elapsed_seconds":time.monotonic()-started,"cache_hit":cache_hit,
                  "nash_conv_bb_per_hand":label["metrics"]["cash"]["nash_conv_bb_per_hand"],
                  "expected_house_rake_bb":label["metrics"]["cash"]["expected_house_rake_bb"],
                  "accounting_residual_bb":label["metrics"]["cash"]["conservation_residual_bb"]}


def generate_root(binary: Path, directory: Path, profile: str, root: dict, iterations: int, lineage: dict):
    config = copy.deepcopy(root["solve_input"]); config["iterations"] = iterations
    request = {"schema":"cash-turn-label-request-v2", "rules_sha256":rules_digest(profile_rules(profile)),
               "native_binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
               "solve_input":config, "public_reach_source":dict(lineage,root_sha256=root["root_sha256"])}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path, input_path = directory / f"label-{digest}.json", directory / f"input-{digest}.json"
    started = time.monotonic()
    input_path.write_text(json.dumps(config,separators=(",", ":")) + "\n")
    (directory / f"request-{digest}.json").write_text(json.dumps(request,separators=(",", ":")) + "\n")
    cache_hit = path.exists()
    if not cache_hit:
        result = subprocess.run([str(binary.resolve()), "turn-river-pbs-solve", "--cash-profile", profile,
                        "--input",str(input_path),"--output",str(path)], text=True,capture_output=True,timeout=180)
        if result.returncode:
            raise RuntimeError(f"native authentic cash labels failed: {result.stderr.strip()}")
    label = json.loads(path.read_text()); validate_label(label)
    if label["input"] != config:
        raise ValueError("cached authentic labels differ from pinned public ranges/state/game/budget")
    return path, {"input_sha256":digest,"source_root_sha256":root["root_sha256"],
                  "elapsed_seconds":time.monotonic()-started,"cache_hit":cache_hit,
                  "nash_conv_bb_per_hand":label["metrics"]["cash"]["nash_conv_bb_per_hand"],
                  "expected_house_rake_bb":label["metrics"]["cash"]["expected_house_rake_bb"],
                  "accounting_residual_bb":label["metrics"]["cash"]["conservation_residual_bb"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, default=Path(__file__).resolve().parents[1]/"target/release/preflop-solver")
    parser.add_argument("--cash-profile", choices=("nl25","nl25-rake-off-control"), default="nl25")
    parser.add_argument("--iterations", type=int, default=8); parser.add_argument("--workers", type=int, choices=(1,2), default=2)
    parser.add_argument("--pot-bb", type=float)
    parser.add_argument("--roots-file", type=Path, help="Native authentic public-turn capture, not hand-conditioned ranges")
    parser.add_argument("--source-weights", type=Path, help="Required frozen average policy for authentic root provenance")
    args = parser.parse_args()
    if (not 2 <= args.iterations <= 128 or bool(args.roots_file) != bool(args.source_weights)
            or (args.roots_file and args.pot_bb is not None)):
        raise ValueError("authentic labels require paired roots/weights and cannot override the captured pot")
    pot = 38. if args.pot_bb is None else args.pot_bb
    if (not args.roots_file and (not math.isfinite(pot) or not 2 <= pot < 40
            or abs(pot*25/2-round(pot*25/2)) > 1e-8)):
        raise ValueError("cash preflight is capped at 128 updates, 20bb stacks and cent-aligned 2..40bb gross pots")
    source = lineage = None
    if args.roots_file:
        source,capture = load_roots(args.roots_file,args.source_weights,args.cash_profile)
        lineage = {"schema":"hu-cash-public-reach-lineage-v1","kind":"authentic-frozen-average-turn-roots",
                   "rules_sha256":source["rules_sha256"],"policy_sha256":source["policy_sha256"],"root_corpus_sha256":capture}
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = ([pool.submit(generate_root,args.native_binary,args.output,args.cash_profile,root,args.iterations,lineage) for root in source["roots"]]
                   if source else [pool.submit(generate,args.native_binary,args.output,args.cash_profile,board,args.iterations,pot) for board in BOARDS])
        results = [f.result() for f in futures]
    corpus = build_dataset([path for path,_ in results])
    if lineage:
        corpus["reach_provenance"] = dict(lineage,root_sha256=[root["root_sha256"] for root in source["roots"]],
                                         label_input_sha256=[identity_hash(label["input"]) for label in corpus["labels"]])
        validate_dataset(corpus)
    (args.output / "dataset.json").write_text(json.dumps(corpus,separators=(",", ":")) + "\n")
    report = {"schema":"hu-cash-value-label-preflight-v1","status":"research_only", "profile":args.cash_profile,
              "iterations":args.iterations,"pot_bb":None if source else pot,"workers":args.workers,"roots":[r for _,r in results],
              "reach_source":lineage or {"kind":"uniform-fresh-turn-roots"},
              "native_binary_sha256":hashlib.sha256(args.native_binary.read_bytes()).hexdigest(),
              "dataset_sha256":hashlib.sha256((args.output/"dataset.json").read_bytes()).hexdigest(),
              "limitations":["conditional authentic turn roots under one research policy" if source else "uniform-range fresh turn roots with the stated gross pot only",
                  "finite-budget reference, not an equilibrium oracle", "not complete serving coverage or unrestricted exploitability"]}
    (args.output / "label-report.json").write_text(json.dumps(report,indent=2) + "\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()

"""Paired frozen-belief action-value diagnosis using saved native packets; no fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::continuation::saved_frozen_prediction_probe"


def select_student(profile, students, alternative=False):
    # Preserve seed pairing across architectures; never pick whichever model
    # happens to look best on this development root.
    matches = [p for p in students if (p["seed"] == profile["seed"] - 89500
               if alternative else p["modelSha256"] == profile["modelSha256"])]
    if len(matches) != 1 or matches[0]["maximumParityErrorBb"] > 1e-4:
        raise ValueError("missing, ambiguous or unverified paired value model")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alternative-values", action="store_true",
                        help="diagnose a different paired value model on unchanged source-policy beliefs; not its policy response")
    for key in ("binary", "comparison", "students", "equity", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
        if key != "output": parser.add_argument("--"+key+"-sha256", required=True)
    args = parser.parse_args()
    pinned = {}
    for key in ("binary", "comparison", "students", "equity", "output"):
        path = getattr(args, key).resolve()
        setattr(args, key, path)
        if key != "output":
            if sha256(path) != getattr(args, key+"_sha256"):
                raise ValueError("pinned input changed: "+key)
            pinned[str(path)] = sha256(path)
    comparison, students = (json.loads(p.read_text()) for p in (args.comparison,args.students))
    if (comparison["status"] != "complete" or students["status"] != "complete"
            or len(comparison["profiles"]) != 2 or len(students["predictions"]) != 2):
        raise ValueError("completed paired comparison and students required")
    if args.output.exists(): raise ValueError("refusing to overwrite a diagnostic")
    args.output.mkdir()
    scripts = Path(__file__).resolve().parent
    for name in ("audit_native_flop_response.mjs", "native_action_diagnostics.mjs",
                 "native_action_value_probe.mjs", "diagnose_native_action_values.mjs"):
        pinned[str(scripts/name)] = sha256(scripts/name)
    pinned[str(args.equity.with_suffix(".f32le"))] = sha256(args.equity.with_suffix(".f32le"))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: stop.set())
    record = dict(schema="native-frozen-action-value-probe-v1", status="running", releaseAccepted=False,
                  pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)), profiles=[],
                  alternativeValues=args.alternative_values,
                  interpretation="Unchanged source-policy final beliefs/actions/native response; alternative values, when requested, are ranking diagnostics only, never the alternative model's policy response.")
    started = time.monotonic()
    atomic_json(args.output/"manifest.json", record)
    try:
        for profile in comparison["profiles"]:
            student = select_student(profile, students["predictions"], args.alternative_values)
            model, candidate = Path(student["model"]), Path(profile["candidate"])
            response, packets = candidate.parent/"response.json", candidate.parent/"packets"
            expected = {str(model):student["modelSha256"], str(candidate):profile["candidateSha256"],
                        str(response):profile["comparison"]["responseSha256"]}
            if len(profile["packets"]) != 49: raise ValueError("incomplete native reference packets")
            for packet in profile["packets"]:
                expected[str(packets/("turn-%d.json"%packet["turn"]))] = packet["sha256"]
            if any(sha256(Path(p)) != h for p,h in expected.items()): raise ValueError("reference artifact changed")
            pinned.update(expected)
            stage = args.output/str(profile["seed"])
            stage.mkdir()
            predicted, diagnosed = stage/"predictions.json", stage/"action-values.json"
            timing = guarded(test_command(args.binary, TEST), {
                "POKER_FROZEN_PREDICT_CANDIDATE": str(candidate),
                "POKER_FROZEN_PREDICT_CANDIDATE_SHA": profile["candidateSha256"],
                "POKER_FROZEN_PREDICT_MODEL": str(model), "POKER_FROZEN_PREDICT_MODEL_SHA": student["modelSha256"],
                "POKER_FROZEN_PREDICT_OUTPUT": str(predicted),
                "POKER_FROZEN_PREDICT_ALTERNATIVE": "1" if args.alternative_values else "",
            }, stage/"prediction-worker", seconds=180, stop=stop)
            guarded(["node",str(scripts/"diagnose_native_action_values.mjs"),str(candidate),str(packets),
                     str(args.equity),str(response),str(predicted),str(diagnosed)]
                    + ([student["modelSha256"]] if args.alternative_values else []),
                    {}, stage/"diagnostic-worker",seconds=60,stop=stop)
            row = dict(seed=profile["seed"], predictionsSha256=sha256(predicted),
                       sourcePolicyModelSha256=profile["modelSha256"], valueModelSha256=student["modelSha256"],
                       diagnosticSha256=sha256(diagnosed), predictionWorker=timing)
            record["profiles"].append(row)
            atomic_json(args.output/"manifest.json",record)
            print(json.dumps(dict(event="action-value-probe-complete",seed=profile["seed"])),flush=True)
        if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
            raise ValueError("stopped stage or pinned input changed")
        record["status"] = "complete"
    except (OSError,ValueError,KeyError,StopIteration) as error:
        stop.set()
        record["status"],record["failure"] = "failed",str(error)
    finally:
        record["elapsedSeconds"] = time.monotonic()-started
        atomic_json(args.output/"manifest.json",record)
    print(json.dumps(dict(status=record["status"],failure=record.get("failure"))),flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()

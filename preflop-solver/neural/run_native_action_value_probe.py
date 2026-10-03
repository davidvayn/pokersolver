"""Paired frozen-belief action-value diagnosis using saved native packets; no fitting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::continuation::saved_frozen_prediction_probe"
BENCHMARK_SPOTS = ("limped-paired", "single-raised-high-rainbow",
                   "three-bet-monotone", "three-bet-high-rainbow")


def select_student(profile, students, alternative=False):
    # Preserve seed pairing across architectures; never pick whichever model
    # happens to look best on this development root.
    matches = [p for p in students if (p["seed"] == profile["seed"] - 89500
               if alternative else p["modelSha256"] == profile["modelSha256"])]
    if len(matches) != 1 or matches[0]["maximumParityErrorBb"] > 1e-4:
        raise ValueError("missing, ambiguous or unverified paired value model")
    return matches[0]


def benchmark_cases(protocol, completed, spot_ids):
    """Select complete, unmodified cases; never silently replace a missing seed."""
    if (protocol.get("schema") != "postflop-benchmark-protocol-v1"
            or completed.get("schema") != "postflop-benchmark-results-v1"
            or completed.get("status") != "complete"
            or completed.get("protocolSha256") != protocol.get("sha256")):
        raise ValueError("incomplete or mismatched benchmark")
    spots = {spot["id"]: spot for spot in protocol["spots"]}
    models = {model["seed"]: model for model in protocol["models"]}
    if len(models) != 2 or set(models) != {100101, 100102}:
        raise ValueError("benchmark needs both retained model seeds")
    if not spot_ids or len(set(spot_ids)) != len(spot_ids) or set(spot_ids) - spots.keys():
        raise ValueError("duplicate or unknown benchmark spot")
    rows = {(row["spot"], row["seed"]): row for row in completed["rows"]}
    if len(rows) != len(completed["rows"]):
        raise ValueError("duplicate benchmark result row")
    selected = []
    for spot_id in spot_ids:
        for seed in sorted(models):
            row = rows.get((spot_id, seed))
            if (not row or not row.get("all49Turns") or not row.get("flopAccountingAuditPassed")
                    or row["board"] != spots[spot_id]["board"]):
                raise ValueError("missing, incomplete or mismatched benchmark case")
            selected.append((spots[spot_id], models[seed], row))
    return selected


def benchmark_main(argv):
    """Probe the completed fresh-board benchmark using its saved native leaves."""
    parser = argparse.ArgumentParser(description="Frozen benchmark action-value diagnosis")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark-manifest-sha256", required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--students-sha256", required=True)
    parser.add_argument("--spots", default=",".join(BENCHMARK_SPOTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.benchmark_root.resolve(), args.output.resolve()
    manifest_path, protocol_path = root / "complete/manifest.json", root / "protocol.json"
    students_path = args.students.resolve()
    if (sha256(manifest_path) != args.benchmark_manifest_sha256
            or sha256(students_path) != args.students_sha256):
        raise ValueError("completed benchmark or student manifest changed")
    protocol, completed, students = (json.loads(p.read_text())
                                      for p in (protocol_path, manifest_path, students_path))
    protocol["sha256"] = sha256(protocol_path)
    if students.get("status") != "complete":
        raise ValueError("retained students are incomplete")
    spots = args.spots.split(",")
    cases = benchmark_cases(protocol, completed, spots)
    models = {student["seed"]: student for student in students["predictions"]}
    if set(models) != {10601, 10602}:
        raise ValueError("both retained value models are required")
    binary = Path(protocol["binary"])
    if sha256(binary) != protocol["pinnedInputs"][str(binary)]:
        raise ValueError("benchmark binary changed")
    scripts = Path(__file__).resolve().parent
    sources = {str(path): sha256(path) for path in
               (Path(__file__), scripts / "audit_native_flop_response.mjs",
                scripts / "native_action_diagnostics.mjs", scripts / "native_action_value_probe.mjs",
                scripts / "diagnose_native_action_values.mjs")}
    pinned = {str(manifest_path): args.benchmark_manifest_sha256,
              str(protocol_path): protocol["sha256"], str(students_path): args.students_sha256,
              str(binary): sha256(binary), **sources}
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("output exists without a resumable manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    identity = dict(schema="benchmark-action-value-probe-v1", benchmarkSha256=args.benchmark_manifest_sha256,
                    protocolSha256=protocol["sha256"], studentsSha256=args.students_sha256,
                    spots=spots, sourceHashes=sources, releaseAccepted=False)
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("resumed benchmark probe identity changed")
    record = {**identity, "status": "running", "cases": []}
    atomic_json(manifest, record)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    started = time.monotonic()
    try:
        for spot, source_model, source_row in cases:
            for model_seed in (10601, 10602):
                if stop.is_set():
                    raise ValueError("diagnostic interrupted")
                source_model_path = Path(source_model["path"])
                student = models[model_seed]
                model = Path(student["model"])
                if (sha256(source_model_path) != source_model["sha256"]
                        or sha256(model) != student["modelSha256"]
                        or student["maximumParityErrorBb"] > 1e-4):
                    raise ValueError("model identity or inference parity changed")
                case = root / "jobs" / spot["id"] / str(source_model["seed"])
                candidate, response, equity = (case / name for name in
                                               ("candidate.json", "response.json", "equity.json"))
                packets = case / "packets"
                expected = {str(candidate): source_row["candidateSha256"],
                            str(response): source_row["responseSha256"],
                            str(equity): sha256(equity),
                            str(equity.with_suffix(".f32le")): sha256(equity.with_suffix(".f32le"))}
                for turn in range(52):
                    if turn in spot["board"]:
                        continue
                    packet = packets / f"turn-{turn}.json"
                    receipt = json.loads((case / f"turn-{turn}" / "completed.json").read_text())
                    expected[str(packet)] = receipt["outputs"][str(packet)]
                if any(sha256(Path(path)) != digest for path, digest in expected.items()):
                    raise ValueError("saved native benchmark packet changed")
                pinned.update(expected)
                crossed = student["modelSha256"] != source_model["sha256"]
                stage = output / spot["id"] / str(source_model["seed"]) / str(model_seed)
                stage.mkdir(parents=True, exist_ok=True)
                receipt_path = stage / "completed.json"
                case_identity = dict(spot=spot["id"], solverSeed=source_model["seed"],
                                     valueModelSeed=model_seed, crossedValueModel=crossed,
                                     candidateSha256=source_row["candidateSha256"],
                                     responseSha256=source_row["responseSha256"],
                                     modelSha256=student["modelSha256"])
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if (receipt.get("identity") != case_identity
                            or any(sha256(Path(path)) != digest
                                   for path, digest in receipt["outputs"].items())):
                        raise ValueError("completed diagnostic changed")
                else:
                    attempt = 0
                    while (stage / f"prediction-worker-{attempt}").exists():
                        attempt += 1
                    predicted = stage / f"predictions-{attempt}.json"
                    diagnosed = stage / f"action-values-{attempt}.json"
                    worker = guarded(test_command(binary, TEST), {
                        "POKER_FROZEN_PREDICT_CANDIDATE": str(candidate),
                        "POKER_FROZEN_PREDICT_CANDIDATE_SHA": source_row["candidateSha256"],
                        "POKER_FROZEN_PREDICT_MODEL": str(model),
                        "POKER_FROZEN_PREDICT_MODEL_SHA": student["modelSha256"],
                        "POKER_FROZEN_PREDICT_OUTPUT": str(predicted),
                        "POKER_FROZEN_PREDICT_ALTERNATIVE": "1" if crossed else "",
                    }, stage / f"prediction-worker-{attempt}", seconds=180, stop=stop)
                    guarded(["node", str(scripts / "diagnose_native_action_values.mjs"),
                             str(candidate), str(packets), str(equity), str(response),
                             str(predicted), str(diagnosed)]
                            + ([student["modelSha256"]] if crossed else [])
                            + ["--action-diagnostics"], {},
                            stage / f"diagnostic-worker-{attempt}", seconds=180, stop=stop)
                    result = json.loads(diagnosed.read_text())
                    if (not math.isclose(result["half_summed_gain_bb"], source_row["halfSummedGainBb"],
                                         rel_tol=0, abs_tol=1e-9)
                            or not result["actionDiagnostics"] or not result["actionValueDiagnostics"]):
                        raise ValueError("benchmark diagnosis disagrees with frozen response")
                    receipt = dict(identity=case_identity, worker=worker,
                                   outputs={str(predicted): sha256(predicted),
                                            str(diagnosed): sha256(diagnosed)})
                    atomic_json(receipt_path, receipt)
                diagnostic = json.loads(next(Path(p).read_text() for p in receipt["outputs"]
                                             if Path(p).name.startswith("action-values-")))
                native_rows = sorted(diagnostic["actionDiagnostics"],
                                     key=lambda row: row["rootWeightedContributionBb"], reverse=True)
                ranking_rows = sorted(diagnostic["actionValueDiagnostics"],
                                      key=lambda row: row["rootWeightedRankingLossBb"], reverse=True)
                record["cases"].append({**case_identity,
                    "nativeResponseGainBb": diagnostic["half_summed_gain_bb"],
                    "largestLocalMistakes": [{"history": row["history"],
                        "rootWeightedContributionBb": row["rootWeightedContributionBb"],
                        "mix": row["reachWeightedMix"], "actions": row["actionLabels"]}
                        for row in native_rows[:3]],
                    "largestRankingErrors": [{"history": row["history"],
                        "rootWeightedRankingLossBb": row["rootWeightedRankingLossBb"],
                        "nativeBestAgreement": row["nativeBestAgreement"],
                        "nativeLossFromPredictedBestBb": row["nativeLossFromPredictedBestBb"]}
                        for row in ranking_rows[:3]],
                    "receipt": str(receipt_path)})
                atomic_json(manifest, record)
                print(json.dumps({"event": "benchmark-action-values", **case_identity}), flush=True)
        if any(sha256(Path(path)) != digest for path, digest in pinned.items()):
            raise ValueError("pinned input changed during diagnosis")
        record["status"] = "complete"
    except (OSError, ValueError, KeyError, StopIteration) as error:
        record["status"], record["failure"] = "failed", str(error)
    finally:
        record["elapsedSeconds"] = time.monotonic() - started
        atomic_json(manifest, record)
    print(json.dumps({"status": record["status"], "failure": record.get("failure")}), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_main(sys.argv[2:])
    else:
        main()

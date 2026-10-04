"""Paired 20bb frozen-response pilot for new value students at fixed 128 updates.

Only the model weights change from the frozen September benchmark. The same
development roots, chance seeds, action tree, native64 played continuations,
49 legal turns, and independent JS flop audit are retained. This cannot certify
full-game exploitability or publication readiness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import signal
import threading
import time

from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import (HERE, MEMORY, PREFIX, RESPONSE,
                                    STAGE_SECONDS, WORKER_LIMIT, PilotMemoryGuard,
                                    controller_lock, run_job)
from run_flop_update_pilot import (BASELINE_SHA, DEFAULT_BASELINE, SPOTS,
                                   check_baseline, solve_case)


def select_students(path: Path, expected_sha: str):
    if sha256(path) != expected_sha:
        raise ValueError("pinned student pair changed")
    report = json.loads(path.read_text())
    rows = report.get("predictions", [])
    if (report.get("status") != "complete" or len(rows) != 2
            or [row.get("seed") for row in rows] != [10601, 10602]):
        raise ValueError("complete independent student pair required")
    models = {}
    for solver_seed, row in zip((100101, 100102), rows, strict=True):
        model = Path(row["model"])
        if (sha256(model) != row["modelSha256"]
                or row["maximumParityErrorBb"] > 1e-4):
            raise ValueError("student weights or native inference parity changed")
        models[solver_seed] = dict(seed=solver_seed, path=str(model), sha256=row["modelSha256"])
    return report, models


def summarize(cases, scored):
    if len(scored) != len(cases):
        raise ValueError("incomplete student policy cannot be scored")
    selected = {(row["spot"], row["seed"]): row for row in scored}
    if len(selected) != len(scored):
        raise ValueError("duplicate student candidate")
    pairs = []
    for spot, model, old in cases:
        row = selected.get((spot["id"], model["seed"]))
        if row is None or not math.isfinite(row["gainBb"]) or row["gainBb"] < -1e-8:
            raise ValueError("missing or invalid student response gain")
        improvement = old["halfSummedGainBb"] - row["gainBb"]
        pairs.append(dict(spot=spot["id"], seed=model["seed"],
                          oldGainBb=old["halfSummedGainBb"], newGainBb=row["gainBb"],
                          improvementBb=improvement,
                          improvementPercentPot=100 * improvement / spot["startingPotBb"]))
    return dict(pairs=pairs,
                equalCaseMeanImprovementBb=math.fsum(x["improvementBb"] for x in pairs) / len(pairs),
                equalCaseMeanImprovementPercentPot=math.fsum(x["improvementPercentPot"] for x in pairs) / len(pairs),
                bothSeedsImproveByRoot={spot_id: all(x["improvementBb"] > 0 for x in pairs if x["spot"] == spot_id)
                                        for spot_id in {x["spot"] for x in pairs}})


def run(args):
    if not 1 <= args.packet_workers <= WORKER_LIMIT or not 0 < args.maximum_seconds <= STAGE_SECONDS:
        raise ValueError("student pilot worker/time limits are outside measured bounds")
    root, output, binary = args.baseline_root.resolve(), args.output.resolve(), args.binary.resolve()
    students_path, preflight_path = args.students.resolve(), args.preflight_manifest.resolve()
    if (not output.parent.is_dir() or sha256(binary) != args.binary_sha256
            or sha256(preflight_path) != args.preflight_sha256):
        raise ValueError("pinned binary, parity preflight, or output parent changed")
    protocol_sha, cases = check_baseline(root, args.baseline_sha256, args.spots.split(","))
    students, models = select_students(students_path, args.students_sha256)
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("schema") != "flop-update-pilot-v1"
            or preflight.get("phase") != "preflight" or preflight.get("status") != "complete"
            or preflight.get("baselineSha256") != args.baseline_sha256
            or preflight.get("binarySha256") != args.binary_sha256
            or preflight.get("cases", [{}])[0].get("parityPassed") is not True):
        raise ValueError("same-binary frozen-policy parity preflight required")
    sources = {str(path): sha256(path) for path in
               (Path(__file__), HERE / "run_flop_update_pilot.py",
                HERE / "run_postflop_gap_pilot.py", HERE / "run_native_value_pilot.py",
                HERE / "worker_resources.py", HERE / "audit_native_flop_response.mjs")}
    identity = dict(schema="postflop-student-value-pilot-v1",
                    baselineSha256=args.baseline_sha256, protocolSha256=protocol_sha,
                    studentManifestSha256=args.students_sha256,
                    studentCorpusSha256=students["corpusSha256"],
                    preflightSha256=args.preflight_sha256,
                    binarySha256=args.binary_sha256, sourceHashes=sources,
                    spots=args.spots.split(","), flopIterations=128,
                    playedTurnIterations=64, packetWorkers=args.packet_workers,
                    maximumSeconds=args.maximum_seconds,
                    releaseAccepted=False)
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("output exists without student pilot manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("student pilot identity changed on resume")
    record = {**identity, "status": "running", "cases": []}
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    timer = threading.Timer(args.maximum_seconds, stop.set)
    timer.daemon = True
    started = time.monotonic()
    with controller_lock(output):
        atomic_json(manifest, record)
        timer.start()
        guard = None
        try:
            guard = PilotMemoryGuard(stop, output / "system-memory.json").start()
            for spot, original_model, old in cases:
                if stop.is_set():
                    raise ValueError("student pilot stopped")
                seed = original_model["seed"]
                model = models[seed]
                work, candidate, candidate_sha, receipt = solve_case(
                    binary, root, spot, model, old, output, stop, 128)
                row = dict(spot=spot["id"], seed=seed,
                           modelSha256=model["sha256"], candidateSha256=candidate_sha,
                           solveSeconds=receipt["worker"]["workerElapsedSeconds"],
                           peakMemoryBytes=receipt["worker"]["sampledPeakMemoryBytes"])
                common = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),
                              POKER_NATIVE_FLOP_CANDIDATE_SHA=candidate_sha)
                packets = work / "packets"
                packets.mkdir(exist_ok=True)

                def packet(turn):
                    path = packets / f"turn-{turn}.json"
                    run_job(binary, RESPONSE + "saved_native_flop_turn_packet",
                            {**common, "POKER_NATIVE_FLOP_TURN": str(turn),
                             "POKER_NATIVE_FLOP_OUTPUT": str(path)},
                            work / f"turn-{turn}", [path], stop, seconds=900, memory=MEMORY)
                    data = json.loads(path.read_text())
                    if (data["candidate_sha256"] != candidate_sha
                            or data["turn"] != turn or data["turn_iterations"] != 64):
                        raise ValueError("student turn packet identity changed")
                    return turn

                turns = [card for card in range(52) if card not in spot["board"]]
                with ThreadPoolExecutor(max_workers=args.packet_workers) as pool:
                    futures = [pool.submit(packet, turn) for turn in turns]
                    try:
                        for number, future in enumerate(as_completed(futures), 1):
                            future.result()
                            if number % 7 == 0 or number == len(turns):
                                print(json.dumps(dict(event="student-value-turns", spot=spot["id"],
                                                      seed=seed, completed=number, total=len(turns))),
                                      flush=True)
                    except BaseException:
                        stop.set()
                        raise
                response = work / "response.json"
                run_job(binary, RESPONSE + "saved_native_flop_response_aggregate",
                        {**common, "POKER_NATIVE_FLOP_PACKET_DIRECTORY": str(packets),
                         "POKER_NATIVE_FLOP_OUTPUT": str(response)},
                        work / "aggregate", [response], stop, seconds=180)
                equity = root / "jobs" / spot["id"] / str(seed) / "equity.json"
                run_job(binary, None, {}, work / "audit", [], stop, seconds=180,
                        command=["node", str(HERE / "audit_native_flop_response.mjs"),
                                 str(candidate), str(packets), str(equity), str(response)])
                row.update(responseSha256=sha256(response),
                           gainBb=json.loads(response.read_text())["half_summed_gain_bb"],
                           oldGainBb=old["halfSummedGainBb"])
                record["cases"].append(row)
                atomic_json(manifest, record)
                print(json.dumps(dict(event="student-value-case", **row)), flush=True)
            if (stop.is_set() or guard.reason
                    or sha256(root / "complete/manifest.json") != args.baseline_sha256
                    or sha256(root / "protocol.json") != protocol_sha
                    or sha256(binary) != args.binary_sha256
                    or sha256(students_path) != args.students_sha256
                    or sha256(preflight_path) != args.preflight_sha256
                    or any(sha256(Path(path)) != digest for path, digest in sources.items())):
                raise ValueError("student pilot stopped or a pinned input changed")
            record["pairedSummary"] = summarize(cases, record["cases"])
            record["status"] = "complete"
        except (OSError, ValueError, KeyError, AssertionError) as error:
            stop.set()
            record["status"], record["failure"] = "failed", str(error)
        finally:
            timer.cancel()
            record["elapsedSeconds"] = time.monotonic() - started
            if guard:
                record["systemMemoryGuard"] = guard.finish()
            atomic_json(manifest, record)
    print(json.dumps(dict(status=record["status"], failure=record.get("failure"),
                          cases=record["cases"], pairedSummary=record.get("pairedSummary"))), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--students-sha256", required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--spots", default=",".join(SPOTS))
    parser.add_argument("--packet-workers", type=int, default=WORKER_LIMIT)
    parser.add_argument("--maximum-seconds", type=int, default=STAGE_SECONDS)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

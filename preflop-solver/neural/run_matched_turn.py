"""Evaluate two matched changes to the live turn/river policy, locally only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time

from run_live_nash_benchmark import ROOT, save, sha
from worker_resources import WorkerResourceGuard

TEST = "blueprint::neural::live_nash_benchmark::matched_turn::run_matched_turn_task"
ORIGINAL_SHA = "a7769aef694a24f03c38fed2de26a8ee489612c7f59d6ac276c3c5120c3f9a2b"
VARIANTS = ("live-2", "routed-32", "joint-2")
# These textures and cards are locked before any candidate score is observed.
FRESH = (
    ("fresh-limped-connected", [8, 14, 16], 41),  # 4c 5h 6c -> Qd
    ("fresh-raised-monotone", [48, 24, 4], 31),   # Ac 8c 3c -> 9s
    ("fresh-three-bet-paired", [39, 36, 9], 22),  # Js Jc 4d -> 7h
)


def read(path):
    return json.loads(path.read_text())


def prepare(path, original, binary, workers):
    old_path = original / "protocol.json"
    if sha(old_path) != ORIGINAL_SHA:
        raise ValueError("expected the original completed turn benchmark protocol")
    old = read(old_path)
    manifest = old["manifest"]
    if manifest not in read(ROOT / "data/practice/full-hand-manifests.json") or not manifest["active"]:
        raise ValueError("live manifest changed since the original benchmark")
    sources = [binary, Path(__file__), ROOT / "data/practice/full-hand-manifests.json", old_path,
               Path(__file__).with_name("worker_resources.py"),
               Path(__file__).with_name("run_live_nash_benchmark.py")]
    blueprint = ROOT / "preflop-solver/src/blueprint"
    sources.extend(blueprint / name for name in ["neural.rs", "public_belief.rs",
        "neural/live_nash_benchmark.rs", "neural/matched_turn.rs", "public_belief/live_policy_response.rs"])
    for artifact in manifest["runtime"]["artifactFiles"].values():
        model = ROOT / "preflop-solver/models/practice" / artifact
        sources.append(model)
        if Path(f"{model}.bin").is_file():
            sources.append(Path(f"{model}.bin"))
    cases = []
    for case, (name, board, turn) in zip(old["cases"], FRESH, strict=True):
        base = original / case["id"] / "base.json"
        direct = original / case["id"] / "direct-turn-result.json"
        for result, job in [(base, "base"), (direct, f"turn-{case['turn']:02}")]:
            receipt = original / "jobs" / f"{case['id']}-{job}" / "completed.json"
            info = read(receipt)
            if (sha(result) != info["resultSha256"] or info["exitCode"] != 0
                    or info["identity"]["protocolSha256"] != ORIGINAL_SHA):
                raise ValueError(f"unverified prior result: {result}")
            sources.extend([receipt, result])
        cases.append(dict(case, cohort="matched", baseSha256=sha(base)))
        cases.append(dict(case, id=name, board=board, turn=turn, cohort="fresh"))
    save(path, dict(schema="matched-live-turn-v1", manifest=manifest, binary=str(binary),
        original=str(original), originalProtocolSha256=ORIGINAL_SHA,
        pinnedInputs={str(p): sha(p) for p in sources}, cases=cases,
        threadsPerWorker=old["threadsPerWorker"], maximumWorkers=workers,
        maximumWorkerSeconds=600, maximumWorkerBytes=2 * 1024**3,
        minimumFreeDiskBytes=20 * 1024**3,
        variants={"live-2": "unmodified live policy",
                  "routed-32": "increase only the turn solver from 2 to 32 updates",
                  "joint-2": "same 2-update budget; deploy both seats and retain all joint river rows"},
        selection="lowest mean conditional NashConv, and only if at least 1e-6 bb below control",
        scope="20bb rake-free, exact within the modeled action tree, conditional on flop check/check"))


def summarize(cases, data, variants):
    rows = {case["id"]: {variant: data[case["id"], variant] for variant in variants} for case in cases}
    averages = {}
    for variant in variants:
        scores = [values[variant]["evaluation"]["metrics"]["nashConvBb"] for values in rows.values()]
        turn = [values[variant]["timing"]["coldTurnSeconds"] for values in rows.values()]
        river = [values[variant]["timing"]["coldRiverSeconds"] for values in rows.values()]
        averages[variant] = dict(meanNashConvBb=statistics.mean(scores),
            meanExploitabilityBb=statistics.mean(scores) / 2, medianColdTurnSeconds=statistics.median(turn),
            medianRiverQuerySeconds=statistics.median(river))
    return dict(cases=rows, aggregates=averages)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, output)


def run_locked(args, output):
    protocol_path = output / "protocol.json"
    if not protocol_path.exists():
        prepare(protocol_path, args.original.resolve(), args.binary.resolve(), args.workers)
    protocol = read(protocol_path)
    for name, digest in protocol["pinnedInputs"].items():
        if sha(name) != digest:
            raise ValueError(f"pinned input changed: {name}")
    original = Path(protocol["original"])
    if args.binary.resolve() != Path(protocol["binary"]) or args.original.resolve() != original:
        raise ValueError("resume inputs must match the original invocation")
    protocol_sha = sha(protocol_path)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())

    def job(case, task, variant="prefix"):
        if stop.is_set():
            raise RuntimeError("benchmark stopped")
        case_id = case["id"]
        directory = output / case_id / f"{task}-{variant}"
        directory.mkdir(parents=True, exist_ok=True)
        prefix = (original / case_id / "base.json" if case["cohort"] == "matched" else
                  output / case_id / "prefix-prefix" / "result.json")
        pins = dict(protocolSha256=protocol_sha, task=task, case=case_id, variant=variant,
                    prefixSha256=None if task == "prefix" else sha(prefix))
        receipt = directory / "completed.json"
        result = directory / "result.json"
        if receipt.exists():
            record = read(receipt)
            if record["pins"] != pins or record["resultSha256"] != sha(result):
                raise ValueError(f"checkpoint integrity mismatch: {directory}")
            return read(result)
        # Each attempted subprocess writes to its own directory. Promote only
        # results from an observed successful exit; abandoned files are evidence.
        attempt = directory / f"attempt-{time.time_ns()}"
        attempt.mkdir()
        trial = attempt / "result.json"
        env = dict(POKER_LIVE_NASH_ROOT=str(ROOT), POKER_MATCHED_PROTOCOL=str(protocol_path),
                   POKER_LIVE_NASH_CASE=case_id, POKER_MATCHED_RESULT=str(trial),
                   POKER_MATCHED_TASK=task, POKER_MATCHED_VARIANT=variant, POKER_MATCHED_PREFIX=str(prefix))
        cmd = [protocol["binary"], "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
        started = time.monotonic()
        with (attempt / "stdout.log").open("x") as stdout, (attempt / "stderr.log").open("x") as stderr:
            process = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            guard = WorkerResourceGuard(process, attempt,
                max_memory_bytes=protocol["maximumWorkerBytes"], max_seconds=protocol["maximumWorkerSeconds"],
                minimum_free_disk_bytes=protocol["minimumFreeDiskBytes"], stop_event=stop).start()
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    guard.request_stop("benchmark controller interrupted")
                    process.wait()
                worker = guard.finish()
        evidence = dict(pins=pins, command=cmd, exitCode=code, worker=worker, seconds=time.monotonic()-started)
        try:
            if code != 0 or worker["resourceStopReason"] or not trial.exists():
                raise ValueError(f"failed worker: {attempt}")
            payload = read(trial)
            if task != "prefix":
                if any(payload[key] != pins[key] for key in ("protocolSha256", "prefixSha256", "case", "variant")):
                    raise ValueError("result identity mismatch")
                if task == "evaluate" and (not math.isfinite(payload["metrics"]["nashConvBb"])
                        or payload["metrics"]["nashConvBb"] < -1e-7):
                    raise ValueError("invalid NashConv")
            # Preserve any unreceipted prior output from a controller crash.
            if result.exists():
                result.rename(attempt / "unreceipted-prior-result.json")
            trial.rename(result)
            save(receipt, dict(evidence, resultSha256=sha(result)))
        except Exception:
            save(attempt / "failed.json", evidence)
            stop.set()
            raise
        print(json.dumps(dict(case=case_id, task=task, variant=variant, seconds=evidence["seconds"],
                             nashConvBb=payload.get("metrics", {}).get("nashConvBb"),
                             coldTurnSeconds=payload.get("coldTurnSeconds"),
                             peakBytes=worker["sampledPeakMemoryBytes"])), flush=True)
        return payload

    def cohort(cases, variants):
        if cases[0]["cohort"] == "fresh":
            with ThreadPoolExecutor(max_workers=protocol["maximumWorkers"]) as pool:
                list(pool.map(lambda case: job(case, "prefix"), cases))
        data = {}
        # All timing commands run alone, rotating variant order across cases.
        for i, case in enumerate(cases):
            for variant in variants[i % len(variants):] + variants[:i % len(variants)]:
                data[case["id"], variant] = {"timing": job(case, "timing", variant)}
        pairs = [(case, variant) for case in cases for variant in variants]
        with ThreadPoolExecutor(max_workers=protocol["maximumWorkers"]) as pool:
            for (case, variant), evaluation in zip(pairs, pool.map(lambda pair: job(pair[0], "evaluate", pair[1]), pairs), strict=True):
                if case["cohort"] == "matched" and variant == "live-2":
                    previous = read(original / case["id"] / "direct-turn-result.json")
                    for key in ("profileBb", "bestResponseBb", "gainBb"):
                        if any(abs(a-b) > 1e-10 for a,b in zip(previous["metrics"][key], evaluation["metrics"][key], strict=True)):
                            stop.set()
                            raise ValueError("new control must reproduce original turn result")
                data[case["id"], variant]["evaluation"] = evaluation
        return summarize(cases, data, variants)

    matched = cohort([c for c in protocol["cases"] if c["cohort"] == "matched"], VARIANTS)
    scores = matched["aggregates"]
    winner = min(VARIANTS[1:], key=lambda variant: scores[variant]["meanNashConvBb"])
    if scores[winner]["meanNashConvBb"] >= scores["live-2"]["meanNashConvBb"] - 1e-6:
        winner = None
    selection = dict(protocolSha256=protocol_sha, winner=winner, matched=matched)
    if not (output / "matched.json").exists():
        save(output / "matched.json", selection)
    elif read(output / "matched.json") != selection:
        raise ValueError("matched selection changed on resume")
    print(json.dumps(dict(winner=winner, matched=scores)), flush=True)
    fresh = cohort([c for c in protocol["cases"] if c["cohort"] == "fresh"], ("live-2", winner)) if winner else None
    summary = dict(selection, fresh=fresh, scope=protocol["scope"], status="complete", fullFlopBenchmark=False)
    if not (output / "summary.json").exists():
        save(output / "summary.json", summary)
    elif read(output / "summary.json") != summary:
        raise ValueError("final summary changed on resume")
    print(json.dumps(dict(winner=winner, fresh=fresh["aggregates"] if fresh else None)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    run(parser.parse_args())

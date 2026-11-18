"""Read-only exact-card conditional response benchmark of the website profile.

Uses the opt-in native test adapter, never fits or promotes a model. Three
prespecified board/line cases; averages are descriptive, not population bounds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from worker_resources import WorkerResourceGuard

ROOT = Path(__file__).resolve().parents[2]
TEST = "blueprint::neural::live_nash_benchmark::run_live_nash_task"
CASES = [
    dict(id="limped-paired", board=[51, 49, 22], turn=4, potBb=2,
         preflop=["limp", "check"]),
    dict(id="single-raised-rainbow", board=[43, 29, 0], turn=18, potBb=5,
         preflop=["raise_to_2.500bb", "call"]),
    dict(id="three-bet-monotone", board=[28, 32, 36], turn=49, potBb=15,
         preflop=["raise_to_2.500bb", "raise_to_7.500bb", "call"]),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    if path.exists():
        raise ValueError(f"refuse to overwrite {path}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol_path = output / "protocol.json"
    if not protocol_path.exists():
        if args.stage != "preflight":
            raise ValueError("run preflight first")
        output.mkdir(parents=True, exist_ok=False)
        manifest_path = ROOT / "data/practice/full-hand-manifests.json"
        manifest = next(m for m in json.loads(manifest_path.read_text()) if m["active"])
        sources = [Path(__file__), ROOT / "preflop-solver/src/blueprint/neural.rs",
                   ROOT / "preflop-solver/src/blueprint/neural/live_nash_benchmark.rs",
                   ROOT / "preflop-solver/src/blueprint/public_belief.rs",
                   ROOT / "preflop-solver/src/blueprint/public_belief/live_policy_response.rs",
                   ROOT / "lib/server/practice-solver-process.ts", manifest_path,
                   Path(__file__).with_name("worker_resources.py"), args.binary.resolve()]
        save(protocol_path, dict(schema="live-solver-nash-protocol-v1", manifest=manifest,
             binary=str(args.binary.resolve()), pinnedInputs={str(p): sha(p) for p in sources},
             cases=CASES, maximumWorkers=4, threadsPerWorker=2,
             maximumWorkerSeconds=600, maximumWorkerBytes=2 * 1024**3,
             minimumFreeDiskBytes=20 * 1024**3,
             metric="sum_i [V_i(BR_i, policy_-i) - V_i(policy)]",
             exploitability="NashConv / 2", turnCondition="flop check/check",
             weighting="equal weight per prespecified root, not reach weighted across the whole game",
             limitations=["three illustrative stratified roots, not a population guarantee",
                          "20bb rake-free modeled action tree only", "no full-hand Nash certificate",
                          "all card-legal private combos and all future public cards retained",
                          "actual f64 served probabilities; no learned values used as terminal truth"]))
    protocol = json.loads(protocol_path.read_text())
    stop = threading.Event()
    for name, expected in protocol["pinnedInputs"].items():
        if sha(name) != expected:
            raise ValueError(f"pinned input changed: {name}")
    if str(args.binary.resolve()) != protocol["binary"]:
        raise ValueError("binary changed")

    def job(case, task, turn=None):
        if stop.is_set():
            raise ValueError("benchmark stopped after a failed worker")
        name = f"{case['id']}-{task}" + (f"-{turn:02}" if turn is not None else "")
        directory = output / "jobs" / name
        receipt = directory / "completed.json"
        command = [protocol["binary"], "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
        environment = dict(POKER_LIVE_NASH_ROOT=str(ROOT), POKER_LIVE_NASH_OUTPUT=str(output),
                           POKER_LIVE_NASH_CASE=case["id"], POKER_LIVE_NASH_TASK=task)
        if turn is not None:
            environment["POKER_LIVE_NASH_TURN"] = str(turn)
        result = output / case["id"] / {"base": "base.json", "packet": f"turn-{turn:02}.json" if turn is not None else "",
                 "turn": "direct-turn-result.json", "aggregate": "flop-result.json"}[task]
        identity = dict(command=command, environment=environment, protocolSha256=sha(protocol_path))
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["identity"] != identity or sha(result) != record["resultSha256"]:
                raise ValueError("completed receipt identity changed")
            return record
        directory.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **environment},
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            guard = WorkerResourceGuard(process, directory,
                max_memory_bytes=protocol["maximumWorkerBytes"],
                max_seconds=protocol["maximumWorkerSeconds"],
                minimum_free_disk_bytes=protocol["minimumFreeDiskBytes"], stop_event=stop).start()
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    guard.request_stop("controller interrupted")
                    process.wait()
                worker = guard.finish()
        record = dict(identity=identity, seconds=time.monotonic()-started, worker=worker, exitCode=code)
        if code or worker["resourceStopReason"] or not result.exists():
            stop.set()
            save(directory / "failed.json", record)
            raise ValueError(f"failed/incomplete job {name}; inspect {directory}")
        record["resultSha256"] = sha(result)
        save(receipt, record)
        print(json.dumps(dict(job=name, seconds=record["seconds"], peakBytes=worker["sampledPeakMemoryBytes"])), flush=True)
        return record

    if args.stage == "preflight":
        case = CASES[0]
        job(case, "base")
        direct = job(case, "turn", case["turn"])
        packet = job(case, "packet", case["turn"])
        estimate = packet["seconds"] * sum(52-len(c["board"]) for c in CASES) / protocol["maximumWorkers"]
        record = dict(status="cost-preflight-complete", directTurnSeconds=direct["seconds"],
                      packetSeconds=packet["seconds"], projectedPacketSeconds=estimate,
                      caveat="one board/card cost only; other pots/textures and contention may differ")
        save(output / "preflight.json", record)
        print(json.dumps(record), flush=True)
        return
    if not (output / "preflight.json").exists():
        raise ValueError("completed cost preflight required")
    with ThreadPoolExecutor(max_workers=protocol["maximumWorkers"]) as executor:
        for future in as_completed([executor.submit(job, c, "base") for c in CASES]):
            future.result()
        tasks = [executor.submit(job, c, "turn", c["turn"]) for c in CASES]
        tasks += [executor.submit(job, c, "packet", t) for c in CASES for t in range(52) if t not in c["board"]]
        for future in as_completed(tasks):
            future.result()
    for case in CASES:
        job(case, "aggregate")
    rows = []
    for case in CASES:
        for street, filename in [("flop", "flop-result.json"), ("turn", "direct-turn-result.json")]:
            result = json.loads((output / case["id"] / filename).read_text())
            rows.append(dict(case=case["id"], street=street, potBb=case["potBb"], **result["metrics"]))
    summaries = {}
    for street in ("flop", "turn"):
        selected = [r for r in rows if r["street"] == street]
        summaries[street] = dict(meanNashConvBb=sum(r["nashConvBb"] for r in selected)/len(selected),
                                meanExploitabilityBb=sum(r["exploitabilityBb"] for r in selected)/len(selected),
                                meanExploitabilityPercentPot=sum(100*r["exploitabilityBb"]/r["potBb"] for r in selected)/len(selected))
    save(output / "summary.json", dict(schema="live-solver-nash-result-v1", status="complete",
         protocolSha256=sha(protocol_path), rows=rows, summaries=summaries,
         limitations=protocol["limitations"]))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()

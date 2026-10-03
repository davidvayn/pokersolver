"""Matched native-versus-learned flop update pilot on frozen benchmark roots.

Only the value source changes. Both arms use the same 32 flop updates, seeded
chance stream, 64-update played turn/river policy and frozen response evaluator.
Completed per-turn receipts permit an interrupted run to resume without scoring
partial work. This is conditional research, not a release qualification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import platform
import re
import signal
import subprocess
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

HERE = Path(__file__).resolve().parent
PREFIX = "blueprint::public_belief::counterfactual_turn::flop_pilot::"
RESPONSE = PREFIX + "frozen_response::tests::"
SPOTS = ("limped-paired", "single-raised-high-rainbow")
WORKER_LIMIT = 4
STAGE_SECONDS = 12 * 3600
MEMORY = 1536 * 1024**2
SERIAL_SOLVE_MEMORY = 2560 * 1024**2


def same_public_state(left, right):
    """Reject root drift but allow native range renormalization roundoff."""
    if {k: v for k, v in left.items() if k != "ranges"} != {
            k: v for k, v in right.items() if k != "ranges"}:
        return False
    a, b = left.get("ranges", []), right.get("ranges", [])
    if len(a) != 2 or len(b) != 2:
        return False
    for x, y in zip(a, b):
        if len(x) != len(y) or not x:
            return False
        if any(not math.isfinite(p) or not math.isfinite(q) or p < 0 or q < 0
               or (p == 0) != (q == 0) or abs(p - q) > 1e-14 for p, q in zip(x, y)):
            return False
        if abs(math.fsum(x) - 1) > 1e-12 or abs(math.fsum(y) - 1) > 1e-12:
            return False
    return True


@contextmanager
def controller_lock(output):
    with (output / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another gap pilot controls this directory") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def effective_memory_pressure(system=None, command=subprocess.run):
    """Treat macOS warning as hazardous only with low reported free memory.

    Critical pressure always stops. A warning with >=25% free memory is
    advisory; each worker still has an independent 1.5 GiB hard RSS limit.
    Missing telemetry fails closed in the guard after three readings.
    """
    system = system or platform.system()
    if system == "Darwin":
        pressure = command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                           check=True, capture_output=True, text=True, timeout=3)
        level = int(pressure.stdout.strip())
        if level not in (1, 2, 4):
            raise ValueError("unknown macOS memory pressure level")
        if level != 2:
            return level
        sample = command(["memory_pressure", "-Q"], check=True,
                         capture_output=True, text=True, timeout=3)
        match = re.search(r"System-wide memory free percentage:\s*(\d+)%", sample.stdout)
        if not match or not 0 <= int(match.group(1)) <= 100:
            raise ValueError("missing macOS free-memory telemetry")
        return 2 if int(match.group(1)) < 25 else 1
    if system == "Linux":
        values = {line.split(":")[0]: int(line.split()[1])
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemAvailable:", "MemTotal:"))}
        fraction = values["MemAvailable"] / values["MemTotal"]
        return 4 if fraction < .05 else 2 if fraction < .15 else 1
    raise ValueError("memory telemetry requires macOS or Linux")


class PilotMemoryGuard:
    def __init__(self, stop, output, reader=effective_memory_pressure, interval=2):
        self.stop, self.output, self.reader, self.interval = stop, output, reader, interval
        self.finished = threading.Event()
        self.warning_samples = self.failures = self.samples = self.maximum_level = 0
        self.reason = None
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def sample(self):
        try:
            level = self.reader()
            self.samples += 1
            self.maximum_level = max(self.maximum_level, level)
            self.warning_samples = self.warning_samples + 1 if level == 2 else 0
            self.failures = 0
            if level == 4 or self.warning_samples >= 60:
                self.reason = "system memory pressure exceeded pilot headroom"
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            self.failures += 1
            if self.failures >= 3:
                self.reason = "system memory telemetry unavailable: " + str(error)
        if self.reason:
            self.stop.set()

    def report(self):
        return dict(samples=self.samples, maximumPressureLevel=self.maximum_level,
                    consecutiveWarnings=self.warning_samples, stopReason=self.reason,
                    sampledAtUnix=time.time())

    def start(self):
        temporary = self.output.with_suffix(".json.tmp")
        if temporary.exists():
            temporary.rename(self.output.parent / f"memory.interrupted-{time.time_ns()}.tmp")
        self.sample()
        if self.reason:
            raise ValueError(self.reason)
        self.thread.start()
        return self

    def watch(self):
        while not self.finished.is_set():
            self.sample()
            try:
                atomic_json(self.output, self.report())
            except OSError as error:
                self.reason = "cannot record memory telemetry: " + str(error)
                self.stop.set()
            if self.stop.is_set() or self.finished.wait(self.interval):
                break

    def finish(self):
        self.finished.set()
        self.thread.join(timeout=4)
        return self.report()


def run_job(binary, test, env, stage, outputs, stop, seconds=900, command=None,
            memory=MEMORY):
    """Resume only complete, identity-matching jobs; preserve interrupted output."""
    stage.mkdir(exist_ok=True)
    receipt = stage / "completed.json"
    identity = dict(command=command or test_command(binary, test), environment=env,
                    maximumMemoryBytes=memory, maximumSeconds=seconds)
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["identity"] != identity:
            raise ValueError("recovered pilot job identity changed")
        if any(sha256(Path(path)) != digest for path, digest in previous["outputs"].items()):
            raise ValueError("recovered pilot output changed")
        return previous
    attempt = 0
    while (stage / f"attempt-{attempt}").exists():
        attempt += 1
    temporary = receipt.with_suffix(".json.tmp")
    if temporary.exists():
        temporary.rename(stage / f"completed.interrupted-{attempt}.tmp")
    for path in outputs:
        if path.exists():
            path.rename(stage / f"{path.name}.interrupted-{attempt}")
    worker = guarded(identity["command"], env, stage / f"attempt-{attempt}",
                     seconds, memory, stop)
    record = dict(identity=identity, worker=worker,
                  outputs={str(path): sha256(path) for path in outputs})
    atomic_json(receipt, record)
    return record


def select_cases(protocol, baseline, requested):
    """Require two original seed results at each requested root."""
    if (protocol.get("schema") != "postflop-benchmark-protocol-v1"
            or baseline.get("schema") != "postflop-benchmark-results-v1"
            or baseline.get("status") != "complete"
            or baseline.get("protocolSha256") != protocol.get("sha256")):
        raise ValueError("incomplete or mismatched baseline")
    spots = {s["id"]: s for s in protocol["spots"]}
    models = {m["seed"]: m for m in protocol["models"]}
    rows = {(r["spot"], r["seed"]): r for r in baseline["rows"]}
    if (len(rows) != len(baseline["rows"]) or set(models) != {100101, 100102}
            or not requested or len(set(requested)) != len(requested)
            or set(requested) - spots.keys()):
        raise ValueError("missing, duplicate or unknown matched case")
    result = []
    for spot_id in requested:
        for seed in sorted(models):
            row = rows.get((spot_id, seed))
            if (not row or row["board"] != spots[spot_id]["board"]
                    or not row.get("all49Turns") or not row.get("flopAccountingAuditPassed")):
                raise ValueError("original result lacks complete paired evidence")
            result.append((spots[spot_id], models[seed], row))
    return result


def estimate_work_seconds(root, cases, workers=WORKER_LIMIT):
    """Estimate a full two-arm response stage from measured packet receipts."""
    packet_seconds = []
    for spot, model, _ in cases:
        work = root / "jobs" / spot["id"] / str(model["seed"])
        measured = [json.loads(path.read_text())["worker"]["workerElapsedSeconds"]
                    for path in work.glob("turn-*/completed.json")]
        if len(measured) != 49:
            raise ValueError("baseline has incomplete turn timings")
        packet_seconds.extend(measured)
    # Two arms per seed/root; one hour is only an allowance until a native
    # preflight is measured (see measured_native_projection below).
    return dict(observedPackets=len(packet_seconds),
                estimatedPacketSeconds=2 * math.fsum(packet_seconds) / workers,
                estimatedSeconds=2 * math.fsum(packet_seconds) / workers + 3600,
                workers=workers,
                interpretation="Projection from old candidate policies at the same roots; not a time guarantee")


def measured_native_projection(estimate, preflight, case_count):
    if (preflight.get("schema") != "postflop-gap-matched-leaf-pilot-v1"
            or preflight.get("phase") != "preflight"
            or preflight.get("status") != "complete"
            or preflight.get("iterations") != 8
            or len(preflight.get("cases", [])) != 1
            or preflight["cases"][0].get("arm") != "native"
            or preflight["cases"][0].get("solveSeconds", 0) <= 0):
        raise ValueError("completed native eight-update preflight required")
    # Fourfold iterations for every native arm plus a 50% allowance for
    # runtime variation, the measured response packets, and one hour for
    # learned solves, serial work and tail. Not a guaranteed completion time.
    native = preflight["cases"][0]["solveSeconds"] * 4 * case_count * 1.5
    return {**estimate, "measuredNativeSolveSeconds": native,
            "estimatedSeconds": estimate["estimatedPacketSeconds"] + native + 3600,
            "preflightPeakMemoryBytes": preflight["cases"][0]["peakMemoryBytes"]}


def run(args):
    root, output = args.baseline_root.resolve(), args.output.resolve()
    baseline_path, protocol_path = root / "complete/manifest.json", root / "protocol.json"
    if sha256(baseline_path) != args.baseline_sha256:
        raise ValueError("frozen benchmark manifest changed")
    protocol, baseline = json.loads(protocol_path.read_text()), json.loads(baseline_path.read_text())
    protocol["sha256"] = sha256(protocol_path)
    cases = select_cases(protocol, baseline, args.spots.split(","))
    binary = Path(protocol["binary"])
    if sha256(binary) != protocol["pinnedInputs"][str(binary)]:
        raise ValueError("original benchmark binary changed")
    estimate = estimate_work_seconds(root, cases)
    preflight_hash = None
    if args.phase == "compare":
        if args.preflight_manifest is None:
            raise ValueError("matched comparison requires a completed native preflight")
        preflight_hash = sha256(args.preflight_manifest.resolve())
        preflight = json.loads(args.preflight_manifest.read_text())
        if (preflight.get("baselineSha256") != args.baseline_sha256
                or preflight.get("binarySha256") != sha256(binary)
                or not preflight.get("spots")
                or preflight["spots"][0] != cases[0][0]["id"]):
            raise ValueError("preflight does not match frozen comparison")
        estimate = measured_native_projection(estimate, preflight, len(cases))
        if estimate["preflightPeakMemoryBytes"] >= SERIAL_SOLVE_MEMORY:
            raise ValueError("native preflight exceeded serial memory headroom")
    if args.phase == "compare" and estimate["estimatedSeconds"] >= STAGE_SECONDS:
        raise ValueError("projected native pilot exceeds 12-hour stage budget")
    sources = {str(path): sha256(path) for path in (Path(__file__),
               HERE / "run_native_value_pilot.py",
               HERE / "worker_resources.py",
               HERE / "audit_native_flop_response.mjs", HERE / "native_action_diagnostics.mjs")}
    identity = dict(schema="postflop-gap-matched-leaf-pilot-v1", phase=args.phase,
                    baselineSha256=args.baseline_sha256, protocolSha256=protocol["sha256"],
                    binarySha256=sha256(binary), sourceHashes=sources,
                    spots=args.spots.split(","), iterations=8 if args.phase == "preflight" else 32,
                    nativeWorkers=WORKER_LIMIT, serialSolveMemoryBytes=SERIAL_SOLVE_MEMORY,
                    packetWorkerMemoryBytes=MEMORY, preflightManifestSha256=preflight_hash,
                    releaseAccepted=False)
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("output exists without a pilot manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("pilot identity changed on resume")
    record = {**identity, "status": "running", "costProjection": estimate, "cases": []}
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    timer = threading.Timer(STAGE_SECONDS, stop.set)
    timer.daemon = True
    started = time.monotonic()
    with controller_lock(output):
        atomic_json(manifest, record)
        timer.start()
        memory_guard = None
        try:
            memory_guard = PilotMemoryGuard(stop, output / "system-memory.json").start()
            selected = cases[:1] if args.phase == "preflight" else cases
            for spot, model, old in selected:
                arms = ("native",) if args.phase == "preflight" else ("native", "learned")
                root_input = root / "inputs" / (spot["id"] + ".json")
                source_inputs = json.loads((root / "inputs.json").read_text())
                if sha256(root_input) != source_inputs["roots"][str(root_input)]:
                    raise ValueError("original public root changed")
                reference_equity = root / "jobs" / spot["id"] / str(model["seed"]) / "equity.json"
                equity_metadata = json.loads(reference_equity.read_text())
                equity_bytes = reference_equity.with_suffix(".f32le")
                if sha256(equity_bytes) != equity_metadata["sha256"]:
                    raise ValueError("original exact equity input changed")
                for arm in arms:
                    if stop.is_set():
                        raise ValueError("pilot stopped")
                    work = output / spot["id"] / str(model["seed"]) / arm
                    work.mkdir(parents=True, exist_ok=True)
                    candidate = work / "candidate.json"
                    env = dict(POKER_NATIVE_FLOP_INPUT=str(root_input),
                               POKER_NATIVE_FLOP_INPUT_SHA=sha256(root_input),
                               POKER_NATIVE_FLOP_SEED=str(model["seed"]),
                               POKER_NATIVE_FLOP_ITERATIONS=str(identity["iterations"]),
                               POKER_NATIVE_FLOP_TURN_ITERATIONS="64",
                               POKER_NATIVE_FLOP_CHANCE_BASELINE="none",
                               POKER_NATIVE_FLOP_TURN_SAMPLES="1",
                               POKER_NATIVE_FLOP_LEAF_WORKERS="1",
                               POKER_NATIVE_FLOP_OUTPUT=str(candidate))
                    if arm == "learned":
                        if sha256(Path(model["path"])) != model["sha256"]:
                            raise ValueError("retained value model changed")
                        env.update(POKER_NATIVE_FLOP_VALUE_MODEL=model["path"],
                                   POKER_NATIVE_FLOP_VALUE_MODEL_SHA=model["sha256"])
                    trained = run_job(binary, PREFIX + "saved_20bb_native_flop_pilot", env,
                                      work / "solve", [candidate], stop,
                                      seconds=900 if args.phase == "preflight" else 3600,
                                      memory=SERIAL_SOLVE_MEMORY)
                    policy = json.loads(candidate.read_text())
                    input_value = json.loads(root_input.read_text())
                    if (not same_public_state(policy["state"], input_value["public"])
                            or policy["game"] != input_value["game"]
                            or policy["iterations"] != identity["iterations"]
                            or policy["turn_iterations"] != 64
                            or policy.get("learned_leaf_model_sha256")
                            != (model["sha256"] if arm == "learned" else None)):
                        raise ValueError("matched pilot changed game, root or budget")
                    row = dict(spot=spot["id"], seed=model["seed"], arm=arm,
                               candidateSha256=sha256(candidate),
                               solveSeconds=trained["worker"]["workerElapsedSeconds"],
                               peakMemoryBytes=trained["worker"]["sampledPeakMemoryBytes"],
                               baselineGainBb=old["halfSummedGainBb"])
                    if args.phase == "preflight":
                        record["cases"].append(row)
                        atomic_json(manifest, record)
                        continue
                    common = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),
                                  POKER_NATIVE_FLOP_CANDIDATE_SHA=row["candidateSha256"])
                    packets = work / "packets"
                    packets.mkdir(exist_ok=True)

                    def packet(turn):
                        path = packets / f"turn-{turn}.json"
                        run_job(binary, RESPONSE + "saved_native_flop_turn_packet",
                                {**common, "POKER_NATIVE_FLOP_TURN":str(turn),
                                 "POKER_NATIVE_FLOP_OUTPUT":str(path)},
                                work / f"turn-{turn}", [path], stop, seconds=900)
                        value = json.loads(path.read_text())
                        if (value["candidate_sha256"] != row["candidateSha256"]
                                or value["turn"] != turn or value["turn_iterations"] != 64):
                            raise ValueError("continuation packet identity changed")
                        return turn

                    turns = [card for card in range(52) if card not in spot["board"]]
                    with ThreadPoolExecutor(max_workers=WORKER_LIMIT) as pool:
                        futures = [pool.submit(packet, card) for card in turns]
                        try:
                            for number, future in enumerate(as_completed(futures), 1):
                                future.result()
                                if number % 7 == 0 or number == len(turns):
                                    print(json.dumps(dict(event="gap-pilot-turns", spot=spot["id"],
                                                          seed=model["seed"], arm=arm,
                                                          completed=number, total=49)), flush=True)
                        except BaseException:
                            stop.set()
                            raise
                    response = work / "response.json"
                    run_job(binary, RESPONSE + "saved_native_flop_response_aggregate",
                            {**common, "POKER_NATIVE_FLOP_PACKET_DIRECTORY":str(packets),
                             "POKER_NATIVE_FLOP_OUTPUT":str(response)},
                            work / "aggregate", [response], stop, seconds=180)
                    audit = HERE / "audit_native_flop_response.mjs"
                    run_job(binary, None, {}, work / "audit", [], stop, seconds=180,
                            command=["node", str(audit), str(candidate), str(packets),
                                     str(reference_equity), str(response)])
                    row.update(responseSha256=sha256(response),
                               gainBb=json.loads(response.read_text())["half_summed_gain_bb"])
                    record["cases"].append(row)
                    atomic_json(manifest, record)
                    print(json.dumps(dict(event="gap-pilot-case", **row)), flush=True)
            if (stop.is_set() or memory_guard.reason
                    or sha256(baseline_path) != args.baseline_sha256
                    or sha256(protocol_path) != protocol["sha256"]
                    or (preflight_hash is not None
                        and sha256(args.preflight_manifest.resolve()) != preflight_hash)
                    or any(sha256(Path(path)) != digest for path, digest in sources.items())):
                raise ValueError("pilot stopped or pinned inputs changed")
            record["status"] = "complete"
        except (OSError, ValueError, KeyError, AssertionError) as error:
            stop.set()
            record["status"], record["failure"] = "failed", str(error)
        finally:
            timer.cancel()
            record["elapsedSeconds"] = time.monotonic() - started
            if memory_guard:
                record["systemMemoryGuard"] = memory_guard.finish()
            atomic_json(manifest, record)
    print(json.dumps(dict(status=record["status"], failure=record.get("failure"),
                          cases=record["cases"], costProjection=estimate)), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "compare"), required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--spots", default=",".join(SPOTS))
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

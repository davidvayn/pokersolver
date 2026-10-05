"""One matched tail-average control: cheap learned warmup, native64 finish.

This is a compute-allocation experiment, not a label-generator or release gate.
Both arms preserve the original root, seed, model and 32 update chance stream.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from run_native_inner_budget_pilot import controls
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import (HERE, MEMORY, PREFIX, RESPONSE, SPOTS,
    PilotMemoryGuard, controller_lock, run_job, same_public_state, select_cases)


def tail_environment(root_input, input_sha, model, output, tail, environ=None):
    inherited = os.environ if environ is None else environ
    if tail not in (0, 8): raise ValueError("only the predeclared zero/eight native-tail comparison is allowed")
    if any(name in inherited for name in ("POKER_NATIVE_FLOP_NATIVE_TAIL_ITERATIONS", "POKER_NATIVE_FLOP_AVERAGING_START")):
        raise ValueError("cannot inherit a different averaging/leaf schedule")
    return dict(POKER_NATIVE_FLOP_INPUT=str(root_input), POKER_NATIVE_FLOP_INPUT_SHA=input_sha,
        POKER_NATIVE_FLOP_SEED="100101", POKER_NATIVE_FLOP_ITERATIONS="32", POKER_NATIVE_FLOP_TURN_ITERATIONS="64",
        POKER_NATIVE_FLOP_CHANCE_BASELINE="none", POKER_NATIVE_FLOP_TURN_SAMPLES="1", POKER_NATIVE_FLOP_LEAF_WORKERS="1",
        POKER_NATIVE_FLOP_VALUE_MODEL=model["path"], POKER_NATIVE_FLOP_VALUE_MODEL_SHA=model["sha256"],
        POKER_NATIVE_FLOP_NATIVE_TAIL_ITERATIONS=str(tail), POKER_NATIVE_FLOP_AVERAGING_START="25",
        POKER_NATIVE_FLOP_OUTPUT=str(output))


def verdict(rows, native, learned):
    if len(rows) != 2 or {row.get("arm") for row in rows} != {"learned-tail-average", "native8-tail-average"}:
        raise ValueError("exactly two complete audited arms required")
    selected = {row["arm"]: row for row in rows}
    if (any(not math.isfinite(row["gainBb"]) or row["gainBb"] < -1e-8
            or not math.isfinite(row["solveSeconds"]) or row["solveSeconds"] <= 0 for row in [*rows, native, learned])
            or native["gainBb"] >= learned["gainBb"]):
        raise ValueError("finite completed policy/cost references and native benefit required")
    control, hybrid = selected["learned-tail-average"], selected["native8-tail-average"]
    improvement = control["gainBb"] - hybrid["gainBb"]
    benefit_retained = (learned["gainBb"] - hybrid["gainBb"]) / (learned["gainBb"] - native["gainBb"])
    cost_fraction = hybrid["solveSeconds"] / native["solveSeconds"]
    promising = improvement >= .05 and benefit_retained >= .5 and cost_fraction <= .4
    return dict(status="promising" if promising else "not_promising", improvementVsMatchedTailBb=improvement,
        originalNativeBenefitRetained=benefit_retained, constructionCostFraction=cost_fraction,
        gainVsOriginalLearnedBb=learned["gainBb"]-hybrid["gainBb"], gainAboveNative64Bb=hybrid["gainBb"]-native["gainBb"],
        releaseAccepted=False, generatorAccepted=False,
        interpretation="One consumed limped root/seed, native64 all49 continuations; no paired-seed or full-game guarantee")


def measured_packet_work(timing):
    rows = timing.get("cases", [])
    if (timing.get("schema") != "native-inner-budget-first-quality-screen-v1"
            or timing.get("status") != "complete" or len(rows) != 1
            or (rows[0].get("spot"), rows[0].get("seed")) != ("limped-paired", 100101)):
        raise ValueError("completed matching all49 native64 timing receipt required")
    packets = rows[0].get("packets", {})
    if len(packets) != 49 or any(not math.isfinite(p["seconds"]) or p["seconds"] <= 0 for p in packets.values()):
        raise ValueError("missing or invalid measured packet timing")
    if (len({p["turn"] for p in packets.values()}) != 49
            or any(type(p["turn"]) is not int or not 0 <= p["turn"] < 52 or key != str(p["turn"])
                   for key, p in packets.items())):
        raise ValueError("duplicated timing turn")
    return math.fsum(p["seconds"] for p in packets.values())


def run(args):
    for name in ("baseline_root", "reference", "binary", "preflight", "timing_reference", "output"):
        setattr(args, name, getattr(args, name).resolve())
    root, output, binary = args.baseline_root, args.output, args.binary
    if (sha256(args.reference) != args.reference_sha256 or sha256(binary) != args.binary_sha256
            or sha256(args.preflight) != args.preflight_sha256 or sha256(args.timing_reference) != args.timing_reference_sha256):
        raise ValueError("frozen reference/binary/preflight changed")
    baseline = root / "complete/manifest.json"; protocol_path = root / "protocol.json"
    if sha256(baseline) != args.baseline_sha256: raise ValueError("original benchmark changed")
    protocol = json.loads(protocol_path.read_text()); protocol["sha256"] = sha256(protocol_path)
    cases = select_cases(protocol, json.loads(baseline.read_text()), list(SPOTS))
    reference = json.loads(args.reference.read_text()); old = controls(reference, cases)
    packet_work = measured_packet_work(json.loads(args.timing_reference.read_text()))
    if reference["baselineSha256"] != args.baseline_sha256 or reference["protocolSha256"] != protocol["sha256"]:
        raise ValueError("matched32 references belong to another benchmark")
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get("schema") != "flop-update-pilot-v1" or preflight.get("phase") != "preflight"
            or preflight.get("status") != "complete" or preflight.get("binarySha256") != args.binary_sha256
            or preflight.get("baselineSha256") != args.baseline_sha256
            or preflight.get("cases", [{}])[0].get("parityPassed") is not True):
        raise ValueError("rebuilt default candidate must reproduce the frozen baseline exactly")
    spot, model, _ = cases[0]
    if (spot["id"], model["seed"]) != ("limped-paired", 100101): raise ValueError("predeclared first case order changed")
    timed_turns = {p["turn"] for p in json.loads(args.timing_reference.read_text())["cases"][0]["packets"].values()}
    if timed_turns != set(range(52)) - set(spot["board"]): raise ValueError("timing reference has different board removal")
    root_input = root / "inputs/limped-paired.json"
    inputs = json.loads((root / "inputs.json").read_text())
    equity = root / "jobs/limped-paired/100101/equity.json"
    pinned = {str(args.reference): args.reference_sha256, str(binary): args.binary_sha256,
        str(args.preflight): args.preflight_sha256, str(baseline): args.baseline_sha256,
        str(args.timing_reference): args.timing_reference_sha256,
        str(protocol_path): protocol["sha256"], str(root / "inputs.json"): sha256(root / "inputs.json"),
        str(root_input): inputs["roots"][str(root_input)], model["path"]: model["sha256"],
        str(equity): sha256(equity), str(equity.with_suffix(".f32le")): json.loads(equity.read_text())["sha256"]}
    for arm in ("native", "learned"):
        row = old[(spot["id"], model["seed"], arm)]
        work = args.reference.parent / spot["id"] / str(model["seed"]) / arm
        pinned[str(work / "candidate.json")] = row["candidateSha256"]
        pinned[str(work / "response.json")] = row["responseSha256"]
        policy = json.loads((work / "candidate.json").read_text())
        response = json.loads((work / "response.json").read_text())
        if (policy["seed"] != 100101 or policy["iterations"] != 32 or policy["turn_iterations"] != 64
                or policy.get("response_turn_iterations", 64) != 64 or policy.get("leaf_schedule") is not None
                or policy.get("learned_leaf_model_sha256") != (model["sha256"] if arm == "learned" else None)
                or response["candidate_sha256"] != row["candidateSha256"] or response["public_turns"] != 49):
            raise ValueError("original reference model, budget, schedule or response identity differs")
        audit = work / "audit/completed.json"; receipt = json.loads(audit.read_text())
        if (receipt["worker"]["status"] != "complete" or receipt["worker"]["exitCode"] != 0
                or receipt["worker"].get("resourceStopReason")):
            raise ValueError("original reference lacks successful independent audit")
        if abs(json.loads((work / "response.json").read_text())["half_summed_gain_bb"] - row["gainBb"]) > 1e-12:
            raise ValueError("reference response differs from its manifest")
        pinned[str(audit)] = sha256(audit)
    for path in sorted((HERE.parent / "src").rglob("*.rs")):
        pinned[str(path)] = sha256(path)
    for name in (Path(__file__).name, "run_native_inner_budget_pilot.py", "run_postflop_gap_pilot.py",
                 "run_native_value_preflight.py", "run_native_value_pilot.py", "worker_resources.py",
                 "audit_native_flop_response.mjs", "native_action_diagnostics.mjs", "native_action_value_probe.mjs"):
        path = HERE / name; pinned[str(path)] = sha256(path)
    if any(sha256(Path(p)) != h for p, h in pinned.items()): raise ValueError("input or source changed")
    if output.exists(): raise ValueError("never overwrite the first matched tail experiment")
    output.mkdir(); stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set); timer.daemon = True
    record = dict(schema="late-native-tail-first-pilot-v1", status="running", cases=[], pinnedInputs=pinned,
        flopIterations=32, turnIterations=64, averagingStartIteration=25, packetWorkers=4,
        maximumSeconds=7200, constructionMemoryBytes=4*1024**3, releaseAccepted=False)
    atomic_json(output / "manifest.json", record)
    native = old[(spot["id"], model["seed"], "native")]; learned = old[(spot["id"], model["seed"], "learned")]
    with controller_lock(output):
        timer.start(); pressure = PilotMemoryGuard(stop, output / "system-memory.json").start()
        try:
            built = []
            for arm, tail in (("learned-tail-average", 0), ("native8-tail-average", 8)):
                work = output / arm; work.mkdir(); candidate = work / "candidate.json"
                receipt = run_job(binary, PREFIX + "saved_20bb_native_flop_pilot",
                    tail_environment(root_input, pinned[str(root_input)], model, candidate, tail),
                    work / "solve", [candidate], stop, seconds=1800, memory=4*1024**3)
                value = json.loads(candidate.read_text()); original = json.loads(root_input.read_text())
                if (not same_public_state(value["state"], original["public"]) or value["game"] != original["game"]
                        or value["seed"] != 100101 or value["iterations"] != 32 or value["turn_iterations"] != 64
                        or value.get("response_turn_iterations") is not None
                        or value.get("learned_leaf_model_sha256") != model["sha256"]
                        or value.get("leaf_schedule") != dict(native_tail_iterations=tail, averaging_start_iteration=25)):
                    raise ValueError("scheduled construction changed frozen inputs or played budget")
                row = dict(arm=arm, spot=spot["id"], seed=100101, modelSha256=model["sha256"],
                    candidateSha256=sha256(candidate), solveSeconds=receipt["worker"]["workerElapsedSeconds"],
                    peakMemoryBytes=receipt["worker"]["sampledPeakMemoryBytes"])
                built.append((work, candidate, row))
                print(json.dumps(dict(event="tail-policy-built", **row)), flush=True)
            # Reserve two complete native64 arms from the pinned recent all49
            # timing receipt, four workers, and 50% margin. Never drop turns.
            projected = 1.5 * (time.monotonic()-started + 2*packet_work/4) + 180
            record["projectedTotalSeconds"] = projected
            if built[1][2]["solveSeconds"] > .4*native["solveSeconds"] or projected > 7200:
                record.update(status="cost_rejected", constructions=[row for _, _, row in built],
                    reason="native tail cost target or full two-arm projection failed; response unmeasured")
            else:
                for work, candidate, row in built:
                    common = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate), POKER_NATIVE_FLOP_CANDIDATE_SHA=row["candidateSha256"])
                    packets = work / "packets"; packets.mkdir()
                    def packet(turn):
                        path = packets / f"turn-{turn}.json"
                        receipt = run_job(binary, RESPONSE + "saved_native_flop_turn_packet",
                            {**common, "POKER_NATIVE_FLOP_TURN":str(turn), "POKER_NATIVE_FLOP_OUTPUT":str(path)},
                            work / f"turn-{turn}", [path], stop, seconds=900, memory=MEMORY)
                        value = json.loads(path.read_text())
                        if value["turn"] != turn or value["turn_iterations"] != 64 or value["candidate_sha256"] != row["candidateSha256"]:
                            raise ValueError("played turn packet changed schedule identity/budget")
                        return dict(turn=turn, sha256=sha256(path), seconds=receipt["worker"]["workerElapsedSeconds"])
                    turns = [c for c in range(52) if c not in spot["board"]]; completed = {}
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        futures = [pool.submit(packet, turn) for turn in turns]
                        try:
                            for index, future in enumerate(as_completed(futures), 1):
                                item = future.result(); completed[str(item["turn"])] = item
                                if index % 7 == 0 or index == 49:
                                    print(json.dumps(dict(event="tail-response-turns", arm=row["arm"], completed=index, total=49)), flush=True)
                        except BaseException:
                            stop.set()
                            for future in futures: future.cancel()
                            raise
                    response = work / "response.json"
                    run_job(binary, RESPONSE + "saved_native_flop_response_aggregate", {**common,
                        "POKER_NATIVE_FLOP_PACKET_DIRECTORY":str(packets), "POKER_NATIVE_FLOP_OUTPUT":str(response)},
                        work / "aggregate", [response], stop, seconds=180)
                    run_job(binary, None, {}, work / "audit", [], stop, seconds=180,
                        command=["node", str(HERE / "audit_native_flop_response.mjs"), str(candidate), str(packets), str(equity), str(response)])
                    row.update(gainBb=json.loads(response.read_text())["half_summed_gain_bb"], responseSha256=sha256(response), packets=completed)
                    record["cases"].append(row); atomic_json(output / "manifest.json", record)
                record.update(status="complete", result=verdict(record["cases"], native, learned))
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()): raise ValueError("stopped or pinned inputs changed")
        except BaseException as error:
            stop.set(); record.update(status="failed", failure=str(error)); raise
        finally:
            timer.cancel(); record.update(elapsedSeconds=time.monotonic()-started, systemMemoryGuard=pressure.finish())
            atomic_json(output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "reference", "binary", "preflight", "timing-reference"):
        p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--"+name.replace("-root", "")+"-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__": main()

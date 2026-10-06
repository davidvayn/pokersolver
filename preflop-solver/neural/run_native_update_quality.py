"""One predeclared all-native32 versus native64 flop-update quality screen."""
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


def first_verdict(gain, control):
    if any(not math.isfinite(v) or v < -1e-8 for v in (gain, control)):
        raise ValueError("finite complete conditional response gains required")
    improvement = control-gain
    return dict(promising=improvement >= .02, improvementBb=improvement,
        regressionBb=max(0, -improvement), releaseAccepted=False,
        interpretation="One consumed root/seed under native64 continuations; not full-game exploitability")


def paired_verdict(gain, control, first_gain, first_control, spot="limped-paired"):
    if not first_verdict(first_gain, first_control)["promising"]:
        raise ValueError("first complete quality screen must pass before pairing")
    second = first_verdict(gain, control)
    mean = ((first_control-first_gain) + second["improvementBb"])/2
    return dict(promising=mean >= .02 and second["regressionBb"] <= .01,
        meanImprovementBb=mean, secondImprovementBb=second["improvementBb"],
        secondRegressionBb=second["regressionBb"], releaseAccepted=False,
        interpretation=("Two consumed limped-root seeds under native64 continuations; other roots and full-game exploitability unmeasured"
            if spot == "limped-paired" else "Two consumed single-raised-root seeds under native64 continuations; full-game exploitability unmeasured"))


def progression(spot, seed, has_first, has_pair):
    if spot not in SPOTS or seed not in (100101,100102):
        raise ValueError("only the original matched roots/seeds are allowed")
    if has_first != (seed == 100102):
        raise ValueError("second seed requires its own root's first screen; first seed cannot confirm itself")
    if has_pair != (spot != "limped-paired"):
        raise ValueError("single-raised expansion requires the completed limped pair")


def pin_screen(path, record, pinned, binary, root_input, control):
    """Reuse complete audited policies, never partial or differently routed work."""
    if (record.get("schema") != "native64-flop-update-first-screen-v1" or record.get("status") != "complete"
            or record.get("seed", 100101) != control["seed"] or record.get("spot", "limped-paired") != control["spot"]
            or record.get("flopIterations") != 64 or record.get("turnIterations") != 64
            or record.get("controlGainBb") != control["gainBb"]
            or record["pinnedInputs"].get(str(binary)) != pinned[str(binary)]
            or record["pinnedInputs"].get(str(root_input)) != sha256(root_input)):
        raise ValueError("completed screen identity/root/budgets differ")
    for filename, digest in (("candidate.json",record["candidateSha256"]),("response.json",record["responseSha256"])):
        pinned[str(path.parent / filename)] = digest
    candidate = json.loads((path.parent / "candidate.json").read_text())
    response = json.loads((path.parent / "response.json").read_text())
    public = json.loads(root_input.read_text()); audit = path.parent / "audit/completed.json"
    worker = json.loads(audit.read_text())["worker"]
    if (not same_public_state(candidate["state"],public["public"]) or candidate["game"] != public["game"]
            or candidate["seed"] != control["seed"] or candidate["iterations"] != 64 or candidate["turn_iterations"] != 64
            or candidate.get("response_turn_iterations") is not None or candidate.get("learned_leaf_model_sha256") is not None
            or candidate.get("leaf_schedule") is not None or response["public_turns"] != 49
            or response["candidate_sha256"] != record["candidateSha256"]
            or abs(response["half_summed_gain_bb"]-record["gainBb"]) > 1e-12
            or worker.get("status") != "complete" or worker.get("exitCode") != 0 or worker.get("resourceStopReason")):
        raise ValueError("completed screen policy/response/audit differs")
    pinned[str(audit)] = sha256(audit)
    turns = set(range(52))-set(candidate["state"]["board"])
    if set(record.get("packets",{})) != {str(t) for t in turns} or len(turns) != 49:
        raise ValueError("completed screen omits exact public chance")
    for turn in turns:
        row = record["packets"][str(turn)]
        if row["turn"] != turn: raise ValueError("packet registry turn changed")
        pinned[str(path.parent / "packets" / f"turn-{turn}.json")] = row["sha256"]


def native_environment(root, digest, output, environ=None, seed=100101):
    inherited = os.environ if environ is None else environ
    if any(k.startswith("POKER_NATIVE_FLOP_") for k in inherited):
        raise ValueError("cannot inherit different flop/model/averaging settings")
    if seed not in (100101, 100102):
        raise ValueError("only the original matched seeds are allowed")
    return dict(POKER_NATIVE_FLOP_INPUT=str(root), POKER_NATIVE_FLOP_INPUT_SHA=digest,
        POKER_NATIVE_FLOP_SEED=str(seed), POKER_NATIVE_FLOP_ITERATIONS="64",
        POKER_NATIVE_FLOP_TURN_ITERATIONS="64", POKER_NATIVE_FLOP_TURN_SAMPLES="1",
        POKER_NATIVE_FLOP_CHANCE_BASELINE="none", POKER_NATIVE_FLOP_LEAF_WORKERS="4",
        POKER_NATIVE_FLOP_OUTPUT=str(output))


def verify_cost(cost, control):
    rows = cost.get("cases", [])
    if (cost.get("schema") != "native-parallel-construction-cost-v1"
            or cost.get("status") != "complete" or cost.get("leafWorkers") != 4
            or [r.get("iterations") for r in rows] != [8, 32]
            or not all(r.get("parityPassed") is True for r in rows)
            or rows[1].get("candidateSha256") != control["candidateSha256"]
            or not math.isfinite(rows[1].get("solveSeconds", float("nan")))
            or not 0 < rows[1]["solveSeconds"] < 1800):
        raise ValueError("complete exact parallel native32 cost/parity preflight required")
    return rows[1]["solveSeconds"]


def run(args):
    root, binary, output = args.baseline_root.resolve(), args.binary.resolve(), args.output.resolve()
    pinned = {}
    for name in ("reference", "cost", "build", "binary", "default_parity"):
        path = getattr(args, name).resolve(); digest = getattr(args, name+"_sha256")
        if sha256(path) != digest: raise ValueError(name+" changed")
        pinned[str(path)] = digest
    baseline = root / "complete/manifest.json"; protocol_path = root / "protocol.json"
    if sha256(baseline) != args.baseline_sha256: raise ValueError("baseline changed")
    pinned[str(baseline)] = args.baseline_sha256; pinned[str(protocol_path)] = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text()); protocol["sha256"] = pinned[str(protocol_path)]
    cases = select_cases(protocol, json.loads(baseline.read_text()), list(SPOTS))
    reference = json.loads(args.reference.read_text()); old = controls(reference, cases)
    progression(args.spot,args.seed,args.first_screen is not None,args.paired_screen is not None)
    control = old[(args.spot, args.seed, "native")]
    seconds32 = verify_cost(json.loads(args.cost.read_text()), old[("limped-paired", 100101, "native")])
    root_input = root / "inputs" / (args.spot+".json")
    first = None
    if (args.first_screen is None) != (args.first_screen_sha256 is None):
        raise ValueError("first screen and hash must be supplied together")
    if (args.paired_screen is None) != (args.paired_screen_sha256 is None):
        raise ValueError("paired screen and hash must be supplied together")
    if args.paired_screen is not None:
        path = args.paired_screen.resolve(); pair = json.loads(path.read_text())
        pinned[str(path)] = args.paired_screen_sha256
        first_paths = [Path(p) for p,h in pair["pinnedInputs"].items() if h == pair.get("firstScreenSha256")]
        if len(first_paths) != 1: raise ValueError("limped pair omits its first-screen identity")
        first_path = first_paths[0]; initial = json.loads(first_path.read_text())
        pinned[str(first_path)] = pair["firstScreenSha256"]
        limp = root / "inputs/limped-paired.json"
        a,b = (old[("limped-paired",seed,"native")] for seed in (100101,100102))
        pin_screen(first_path,initial,pinned,binary,limp,a)
        pin_screen(path,pair,pinned,binary,limp,b)
        expected = paired_verdict(pair["gainBb"],b["gainBb"],initial["gainBb"],a["gainBb"])
        if pair.get("result") != expected or not expected["promising"]:
            raise ValueError("limped pair did not pass the original mean/nonregression criteria")
    if args.seed == 100102:
        if args.first_screen is None: raise ValueError("paired seed requires the completed promising first screen")
        path = args.first_screen.resolve(); first = json.loads(path.read_text())
        pinned[str(path)] = args.first_screen_sha256
        first_control = old[(args.spot, 100101, "native")]
        pin_screen(path,first,pinned,binary,root_input,first_control)
        if (first.get("result") != first_verdict(first["gainBb"],first_control["gainBb"])
                or not first["result"]["promising"]):
            raise ValueError("first screen is incomplete, mismatched or rejected")
    original = args.reference.resolve().parent / args.spot / str(args.seed) / "native"
    for filename, digest in (("candidate.json", control["candidateSha256"]),
                             ("response.json", control["responseSha256"])):
        pinned[str(original / filename)] = digest
    response = json.loads((original / "response.json").read_text())
    audit = original / "audit/completed.json"; worker = json.loads(audit.read_text())["worker"]
    if (response["public_turns"] != 49 or response["candidate_sha256"] != control["candidateSha256"]
            or abs(response["half_summed_gain_bb"]-control["gainBb"]) > 1e-12
            or worker.get("status") != "complete" or worker.get("exitCode") != 0
            or worker.get("resourceStopReason")):
        raise ValueError("native32 reference lacks complete response and independent audit")
    pinned[str(audit)] = sha256(audit)
    inputs = root / "inputs.json"
    pinned[str(inputs)] = sha256(inputs)
    pinned[str(root_input)] = json.loads(inputs.read_text())["roots"][str(root_input)]
    equity = root / "jobs" / args.spot / str(args.seed) / "equity.json"
    pinned[str(equity)] = sha256(equity)
    pinned[str(equity.with_suffix(".f32le"))] = json.loads(equity.read_text())["sha256"]
    build = json.loads(args.build.read_text())
    if build.get("binary") != str(binary) or build.get("binarySha256") != args.binary_sha256:
        raise ValueError("wrong build snapshot")
    default = json.loads(args.default_parity.read_text())
    default_worker = default.get("worker", {})
    if (default.get("status") != "complete" or default.get("binarySha256") != args.binary_sha256
            or default.get("candidateSha256") != "f1faa095fc4b905786d446f32bf9927db645fb8008258187e49b33d812e1fe4d"
            or default.get("parityPassed") is not True or default_worker.get("status") != "complete"
            or default_worker.get("exitCode") != 0 or default_worker.get("resourceStopReason")):
        raise ValueError("rebuilt default must reproduce the original learned32 policy")
    pinned[str(args.default_parity.resolve().parent / "candidate.json")] = default["candidateSha256"]
    if not build["rustSources"]:
        raise ValueError("empty source snapshot")
    pinned.update(build["rustSources"])
    for name in (Path(__file__).name, "run_native_inner_budget_pilot.py", "run_postflop_gap_pilot.py",
                 "run_native_value_preflight.py", "run_native_value_pilot.py", "worker_resources.py",
                 "audit_native_flop_response.mjs", "native_action_diagnostics.mjs", "native_action_value_probe.mjs"):
        path = HERE / name; pinned[str(path)] = sha256(path)
    # Measured complete native64 packets from the retained control, not labels
    # from a different continuation budget or partial sample.
    packet_seconds = []
    for turn in range(52):
        if turn in json.loads(root_input.read_text())["public"]["board"]: continue
        receipt = original / f"turn-{turn}/completed.json"
        packet_seconds.append(json.loads(receipt.read_text())["worker"]["workerElapsedSeconds"])
        pinned[str(receipt)] = sha256(receipt)
    if len(packet_seconds) != 49 or any(not math.isfinite(s) or s <= 0 for s in packet_seconds):
        raise ValueError("complete native64 packet cost reference required")
    projection = 1.25*2*seconds32 + 1.5*math.fsum(packet_seconds)/4 + 180
    if projection > 7200: raise ValueError("projected first quality screen exceeds two hours")
    if any(sha256(Path(p)) != h for p, h in pinned.items()): raise ValueError("input/source changed")
    if output.exists(): raise ValueError("never overwrite a native update screen")
    output.mkdir(); stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set); timer.daemon = True
    record = dict(schema="native64-flop-update-first-screen-v1", status="running", pinnedInputs=pinned,
        maximumSeconds=7200, projectedSeconds=projection, packetWorkers=4, releaseAccepted=False,
        controlGainBb=control["gainBb"], flopIterations=64, turnIterations=64,
        spot=args.spot, seed=args.seed, firstScreenSha256=args.first_screen_sha256,
        pairedScreenSha256=args.paired_screen_sha256)
    with controller_lock(output):
        pressure = None
        try:
            atomic_json(output / "manifest.json", record); timer.start()
            pressure = PilotMemoryGuard(stop, output / "system-memory.json").start()
            candidate = output / "candidate.json"
            solve = run_job(binary, PREFIX+"saved_20bb_native_flop_pilot",
                native_environment(root_input, pinned[str(root_input)], candidate, seed=args.seed),
                output / "solve", [candidate], stop, seconds=3600, memory=4*1024**3)
            value = json.loads(candidate.read_text()); original_input = json.loads(root_input.read_text())
            if (not same_public_state(value["state"], original_input["public"]) or value["game"] != original_input["game"]
                    or value["seed"] != args.seed or value["iterations"] != 64 or value["turn_iterations"] != 64
                    or value.get("response_turn_iterations") is not None
                    or value.get("learned_leaf_model_sha256") is not None or value.get("leaf_schedule") is not None):
                raise ValueError("candidate changed model, budgets, averaging or frozen game")
            record.update(candidateSha256=sha256(candidate), solveSeconds=solve["worker"]["workerElapsedSeconds"],
                peakMemoryBytes=solve["worker"]["sampledPeakMemoryBytes"])
            atomic_json(output / "manifest.json", record)
            print(json.dumps(dict(event="native64-built", **{k:record[k] for k in ("candidateSha256", "solveSeconds", "peakMemoryBytes")})), flush=True)
            packets = output / "packets"; packets.mkdir()
            common = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate), POKER_NATIVE_FLOP_CANDIDATE_SHA=record["candidateSha256"])
            def packet(turn):
                path = packets / f"turn-{turn}.json"
                receipt = run_job(binary, RESPONSE+"saved_native_flop_turn_packet",
                    {**common, "POKER_NATIVE_FLOP_TURN":str(turn), "POKER_NATIVE_FLOP_OUTPUT":str(path)},
                    output / f"turn-{turn}", [path], stop, seconds=900, memory=MEMORY)
                data = json.loads(path.read_text())
                if data["candidate_sha256"] != record["candidateSha256"] or data["turn"] != turn or data["turn_iterations"] != 64:
                    raise ValueError("packet identity or played budget changed")
                return dict(turn=turn, sha256=sha256(path), seconds=receipt["worker"]["workerElapsedSeconds"])
            turns = [t for t in range(52) if t not in value["state"]["board"]]; completed = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(packet, turn) for turn in turns]
                try:
                    for number, future in enumerate(as_completed(futures), 1):
                        item = future.result(); completed[str(item["turn"])] = item
                        if number % 7 == 0 or number == 49:
                            print(json.dumps(dict(event="native64-response-turns", completed=number, total=49)), flush=True)
                except BaseException:
                    stop.set()
                    for future in futures: future.cancel()
                    raise
            response_path = output / "response.json"
            run_job(binary, RESPONSE+"saved_native_flop_response_aggregate", {**common,
                "POKER_NATIVE_FLOP_PACKET_DIRECTORY":str(packets), "POKER_NATIVE_FLOP_OUTPUT":str(response_path)},
                output / "aggregate", [response_path], stop, seconds=180)
            run_job(binary, None, {}, output / "audit", [], stop, seconds=180,
                command=["node", str(HERE/"audit_native_flop_response.mjs"), str(candidate), str(packets), str(equity), str(response_path)])
            gain = json.loads(response_path.read_text())["half_summed_gain_bb"]
            record.update(status="complete", gainBb=gain, responseSha256=sha256(response_path), packets=completed,
                result=(first_verdict(gain, control["gainBb"]) if first is None else
                        paired_verdict(gain, control["gainBb"], first["gainBb"], first["controlGainBb"],spot=args.spot)))
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
                raise ValueError("stopped or pinned source changed")
            print(json.dumps(dict(event="native64-quality", gainBb=gain, result=record["result"])), flush=True)
        except BaseException as error:
            stop.set(); record.update(status="failed", failure=str(error)); raise
        finally:
            timer.cancel(); record["elapsedSeconds"] = time.monotonic()-started
            if pressure is not None: record["systemMemoryGuard"] = pressure.finish()
            atomic_json(output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, required=True); p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--seed", type=int, choices=(100101,100102), default=100101)
    p.add_argument("--spot", choices=SPOTS, default="limped-paired")
    p.add_argument("--first-screen", type=Path); p.add_argument("--first-screen-sha256")
    p.add_argument("--paired-screen", type=Path); p.add_argument("--paired-screen-sha256")
    for name in ("reference", "cost", "build", "binary", "default_parity"):
        option = name.replace("_", "-")
        p.add_argument("--"+option, type=Path, required=True); p.add_argument("--"+option+"-sha256", required=True)
    p.add_argument("--output", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()

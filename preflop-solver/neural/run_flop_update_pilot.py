"""Bounded, paired 128-versus-256 learned-leaf flop-update experiment.

First prove that the rebuilt binary reproduces a frozen 128-update candidate
byte-for-byte. Only then reuse its 49-turn response as the matched control.
All 256-update policies receive the same played native-64 continuation and
frozen-response audit. These selected roots are development cases, not an
unseen or full-game exploitability evaluation.
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

from run_native_value_preflight import atomic_json, sha256, RESERVE_BYTES
from run_postflop_gap_pilot import (HERE, MEMORY, PREFIX, RESPONSE,
                                    SERIAL_SOLVE_MEMORY, STAGE_SECONDS,
                                    WORKER_LIMIT, PilotMemoryGuard,
                                    controller_lock, run_job, same_public_state,
                                    select_cases)

DEFAULT_BASELINE = HERE / "runs/local-postflop-benchmark-20260921-c"
BASELINE_SHA = "9a7f178b88fb595235c22e83f8fb173c9413d8692da34f6990a8a0540faf0ad8"
SPOTS = ("limped-paired", "single-raised-high-rainbow")


def check_baseline(root, expected_sha, requested):
    manifest_path, protocol_path = root / "complete/manifest.json", root / "protocol.json"
    if sha256(manifest_path) != expected_sha:
        raise ValueError("frozen benchmark manifest changed")
    protocol = json.loads(protocol_path.read_text())
    baseline = json.loads(manifest_path.read_text())
    protocol_sha = sha256(protocol_path)
    protocol["sha256"] = protocol_sha
    if protocol["flopIterations"] != 128 or protocol["turnIterations"] != 64:
        raise ValueError("baseline budgets changed")
    prior_binary = Path(protocol["binary"])
    if sha256(prior_binary) != protocol["pinnedInputs"][str(prior_binary)]:
        raise ValueError("frozen benchmark binary changed")
    cases = select_cases(protocol, baseline, requested)
    input_pins = json.loads((root / "inputs.json").read_text())["roots"]
    for spot, model, row in cases:
        root_input = root / "inputs" / f"{spot['id']}.json"
        if sha256(root_input) != input_pins[str(root_input)]:
            raise ValueError("frozen public root changed")
        if sha256(Path(model["path"])) != model["sha256"]:
            raise ValueError("retained value model changed")
        old_work = root / "jobs" / spot["id"] / str(model["seed"])
        if (sha256(old_work / "candidate.json") != row["candidateSha256"]
                or sha256(old_work / "response.json") != row["responseSha256"]):
            raise ValueError("frozen candidate or response changed")
        equity = old_work / "equity.json"
        if sha256(equity.with_suffix(".f32le")) != json.loads(equity.read_text())["sha256"]:
            raise ValueError("frozen equity changed")
    return protocol_sha, cases


def paired_summary(cases, new_rows):
    if len(new_rows) != len(cases):
        raise ValueError("cannot score incomplete paired comparison")
    lookup = {(row["spot"], row["seed"]): row for row in new_rows}
    if len(lookup) != len(new_rows):
        raise ValueError("duplicate update-comparison case")
    differences = []
    for spot, model, old in cases:
        row = lookup.get((spot["id"], model["seed"]))
        if row is None or not math.isfinite(row["gainBb"]) or row["gainBb"] < -1e-8:
            raise ValueError("missing or invalid update-comparison gain")
        delta = old["halfSummedGainBb"] - row["gainBb"]
        differences.append(dict(spot=spot["id"], seed=model["seed"],
                                startingPotBb=spot["startingPotBb"],
                                learned128GainBb=old["halfSummedGainBb"],
                                learned256GainBb=row["gainBb"],
                                improvementBb=delta,
                                improvementPercentPot=100 * delta / spot["startingPotBb"]))
    return dict(cases=differences,
                equalCaseMeanImprovementBb=math.fsum(x["improvementBb"] for x in differences) / len(differences),
                equalCaseMeanImprovementPercentPot=math.fsum(x["improvementPercentPot"] for x in differences) / len(differences),
                bothSeedsImproveByRoot={s: all(x["improvementBb"] > 0 for x in differences if x["spot"] == s)
                                        for s in {x["spot"] for x in differences}})


def candidate_environment(root_input, model, seed, iterations, output):
    return dict(POKER_NATIVE_FLOP_INPUT=str(root_input),
                POKER_NATIVE_FLOP_INPUT_SHA=sha256(root_input),
                POKER_NATIVE_FLOP_SEED=str(seed),
                POKER_NATIVE_FLOP_ITERATIONS=str(iterations),
                POKER_NATIVE_FLOP_TURN_ITERATIONS="64",
                POKER_NATIVE_FLOP_CHANCE_BASELINE="none",
                POKER_NATIVE_FLOP_TURN_SAMPLES="1",
                POKER_NATIVE_FLOP_LEAF_WORKERS="1",
                POKER_NATIVE_FLOP_VALUE_MODEL=model["path"],
                POKER_NATIVE_FLOP_VALUE_MODEL_SHA=model["sha256"],
                POKER_NATIVE_FLOP_OUTPUT=str(output))


def solve_case(binary, root, spot, model, old, output, stop, iterations):
    work = output / spot["id"] / str(model["seed"])
    work.mkdir(parents=True, exist_ok=True)
    root_input = root / "inputs" / f"{spot['id']}.json"
    candidate = work / "candidate.json"
    env = candidate_environment(root_input, model, model["seed"], iterations, candidate)
    receipt = run_job(binary, PREFIX + "saved_20bb_native_flop_pilot", env,
                      work / "solve", [candidate], stop, seconds=3600,
                      memory=SERIAL_SOLVE_MEMORY)
    policy = json.loads(candidate.read_text())
    source = json.loads(root_input.read_text())
    if (not same_public_state(policy["state"], source["public"])
            or policy["game"] != source["game"]
            or policy["iterations"] != iterations
            or policy["turn_iterations"] != 64
            or policy.get("learned_leaf_model_sha256") != model["sha256"]):
        raise ValueError("update pilot changed game, root, value model, or budget")
    candidate_sha = sha256(candidate)
    return work, candidate, candidate_sha, receipt


def run(args):
    root, output, binary = args.baseline_root.resolve(), args.output.resolve(), args.binary.resolve()
    if not binary.is_file() or not output.parent.is_dir():
        raise ValueError("pinned binary or output parent missing")
    requested = args.spots.split(",")
    protocol_sha, cases = check_baseline(root, args.baseline_sha256, requested)
    sources = {str(path): sha256(path) for path in
               (Path(__file__), HERE / "run_postflop_gap_pilot.py",
                HERE / "run_native_value_pilot.py", HERE / "worker_resources.py",
                HERE / "audit_native_flop_response.mjs")}
    binary_sha = sha256(binary)
    identity = dict(schema="flop-update-pilot-v1", phase=args.phase,
                    baselineSha256=args.baseline_sha256, protocolSha256=protocol_sha,
                    binarySha256=binary_sha, sourceHashes=sources,
                    spots=requested, controlIterations=128, candidateIterations=256,
                    playedTurnIterations=64, packetWorkers=WORKER_LIMIT,
                    serialMemoryBytes=SERIAL_SOLVE_MEMORY, packetMemoryBytes=MEMORY,
                    minimumFreeDiskBytes=RESERVE_BYTES, releaseAccepted=False)
    if args.phase == "compare":
        if args.preflight_manifest is None:
            raise ValueError("comparison needs a completed parity preflight")
        preflight_path = args.preflight_manifest.resolve()
        preflight = json.loads(preflight_path.read_text())
        if (preflight.get("schema") != identity["schema"]
                or preflight.get("phase") != "preflight"
                or preflight.get("status") != "complete"
                or preflight.get("baselineSha256") != args.baseline_sha256
                or preflight.get("binarySha256") != binary_sha
                or preflight.get("cases", [{}])[0].get("parityPassed") is not True):
            raise ValueError("parity preflight mismatches this comparison")
        identity["preflightSha256"] = sha256(preflight_path)
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("output exists without pilot manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("pilot identity changed on resume")
    record = {**identity, "status": "running", "cases": []}
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
            for spot, model, old in cases[:1] if args.phase == "preflight" else cases:
                if stop.is_set():
                    raise ValueError("pilot stopped")
                iterations = 128 if args.phase == "preflight" else 256
                work, candidate, candidate_sha, receipt = solve_case(
                    binary, root, spot, model, old, output, stop, iterations)
                row = dict(spot=spot["id"], seed=model["seed"],
                           candidateSha256=candidate_sha,
                           solveSeconds=receipt["worker"]["workerElapsedSeconds"],
                           peakMemoryBytes=receipt["worker"]["sampledPeakMemoryBytes"])
                if args.phase == "preflight":
                    row["parityPassed"] = candidate_sha == old["candidateSha256"]
                    record["cases"].append(row)
                    atomic_json(manifest, record)
                    if not row["parityPassed"]:
                        raise ValueError("rebuilt binary does not reproduce frozen 128-update candidate")
                    break
                common = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),
                              POKER_NATIVE_FLOP_CANDIDATE_SHA=candidate_sha)
                packets = work / "packets"
                packets.mkdir(exist_ok=True)

                def packet(turn):
                    path = packets / f"turn-{turn}.json"
                    run_job(binary, RESPONSE + "saved_native_flop_turn_packet",
                            {**common, "POKER_NATIVE_FLOP_TURN": str(turn),
                             "POKER_NATIVE_FLOP_OUTPUT": str(path)},
                            work / f"turn-{turn}", [path], stop, seconds=900)
                    value = json.loads(path.read_text())
                    if (value["candidate_sha256"] != candidate_sha
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
                                print(json.dumps(dict(event="update-pilot-turns", spot=spot["id"],
                                                      seed=model["seed"], completed=number,
                                                      total=len(turns))), flush=True)
                    except BaseException:
                        stop.set()
                        raise
                response = work / "response.json"
                run_job(binary, RESPONSE + "saved_native_flop_response_aggregate",
                        {**common, "POKER_NATIVE_FLOP_PACKET_DIRECTORY": str(packets),
                         "POKER_NATIVE_FLOP_OUTPUT": str(response)},
                        work / "aggregate", [response], stop, seconds=180)
                reference_equity = root / "jobs" / spot["id"] / str(model["seed"]) / "equity.json"
                run_job(binary, None, {}, work / "audit", [], stop, seconds=180,
                        command=["node", str(HERE / "audit_native_flop_response.mjs"),
                                 str(candidate), str(packets), str(reference_equity), str(response)])
                row.update(responseSha256=sha256(response),
                           gainBb=json.loads(response.read_text())["half_summed_gain_bb"],
                           baselineGainBb=old["halfSummedGainBb"])
                record["cases"].append(row)
                atomic_json(manifest, record)
                print(json.dumps(dict(event="update-pilot-case", **row)), flush=True)
            if (stop.is_set() or memory_guard.reason
                    or sha256(root / "complete/manifest.json") != args.baseline_sha256
                    or sha256(root / "protocol.json") != protocol_sha
                    or sha256(binary) != binary_sha
                    or any(sha256(Path(path)) != digest for path, digest in sources.items())):
                raise ValueError("pilot stopped or a pinned input changed")
            if args.phase == "compare":
                record["pairedSummary"] = paired_summary(cases, record["cases"])
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
                          cases=record["cases"], pairedSummary=record.get("pairedSummary"))), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "compare"), required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--spots", default=",".join(SPOTS))
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

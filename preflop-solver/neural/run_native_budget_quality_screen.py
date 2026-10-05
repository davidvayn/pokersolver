"""Complete one staged native4/native64 control; never qualify a partial pair."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import signal
import threading
import time

from run_native_inner_budget_pilot import case_rejection, evaluation_candidate
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import HERE, MEMORY, RESPONSE, PilotMemoryGuard, controller_lock, run_job


def verify_screen_inputs(construction, preflight):
    row = construction.get("costRejection", {})
    measured = (row.get("solveSeconds", float("nan")), row.get("native64SolveSeconds", float("nan")),
                construction.get("projectedTotalSeconds", float("nan")), construction.get("maximumSeconds", float("nan")))
    if (construction.get("schema") != "native-inner-budget-pilot-v1"
            or construction.get("status") != "rejected" or construction.get("cases") != []
            or (row.get("spot"), row.get("seed")) != ("limped-paired", 100101)
            or construction.get("constructionTurnIterations") != 4
            or construction.get("playedTurnIterations") != 64
            or any(not math.isfinite(value) or value <= 0 for value in measured)
            or row.get("solveSeconds", float("inf")) > .2 * row.get("native64SolveSeconds", 0)
            or construction.get("projectedTotalSeconds", 0) <= construction.get("maximumSeconds", float("inf"))):
        raise ValueError("only the completed first construction stopped for full-pair projected cost can be screened")
    if (preflight.get("schema") != "native-streamed-policy-hash-preflight-v1"
            or preflight.get("status") != "complete" or preflight.get("exactPacketParity") is not True
            or [row.get("arm") for row in preflight.get("packets", [])] != ["buffered", "streamed"]
            or len({row.get("sha256") for row in preflight["packets"]}) != 1):
        raise ValueError("completed exact native packet parity required")
    return row


def run(args):
    source, preflight_path, binary, output = [p.resolve() for p in
        (args.construction, args.hash_preflight, args.binary, args.output)]
    if sha256(source) != args.construction_sha256 or sha256(preflight_path) != args.hash_preflight_sha256:
        raise ValueError("frozen construction/hash preflight changed")
    construction = json.loads(source.read_text()); preflight = json.loads(preflight_path.read_text())
    row = dict(verify_screen_inputs(construction, preflight))
    if preflight["pinnedInputs"].get(str(binary)) != args.binary_sha256:
        raise ValueError("evaluation binary was not verified in the exact hash preflight")
    streamed_receipt = preflight_path.parent / "streamed/completed.json"
    if json.loads(streamed_receipt.read_text())["identity"]["command"][0] != str(binary):
        raise ValueError("quality screen must use the measured streamed evaluation binary")
    historical_sources = {path: digest for path, digest in construction["pinnedInputs"].items()
                          if Path(path).suffix in (".py", ".mjs", ".rs")}
    # Completed construction pins its original runner version in the immutable
    # manifest; active evaluation code can evolve without rewriting that history.
    pinned = {**{path: digest for path, digest in construction["pinnedInputs"].items() if path not in historical_sources},
              **preflight["pinnedInputs"],
              str(source): args.construction_sha256, str(preflight_path): args.hash_preflight_sha256}
    pinned[str(streamed_receipt)] = sha256(streamed_receipt)
    work = source.parent / row["spot"] / str(row["seed"])
    raw, source_candidate, receipt_path = work / "training-candidate.json", work / "candidate.json", work / "solve/completed.json"
    receipt = json.loads(receipt_path.read_text()); worker = receipt["worker"]
    if (worker.get("status") != "complete" or worker.get("exitCode") != 0 or worker.get("resourceStopReason")
            or receipt["outputs"].get(str(raw)) != row["trainingCandidateSha256"]
            or worker["workerElapsedSeconds"] != row["solveSeconds"]
            or evaluation_candidate(json.loads(raw.read_text())) != json.loads(source_candidate.read_text())):
        raise ValueError("raw construction did not complete unchanged before stronger evaluation export")
    pinned.update({str(raw): row["trainingCandidateSha256"], str(source_candidate): row["candidateSha256"],
                   str(receipt_path): sha256(receipt_path)})
    for name in (Path(__file__).name, "run_native_inner_budget_pilot.py", "audit_native_flop_response.mjs",
                 "native_action_diagnostics.mjs"):
        path = HERE / name; pinned[str(path)] = sha256(path)
    if any(sha256(Path(path)) != digest for path, digest in pinned.items()):
        raise ValueError("quality-screen input changed")
    original_input = Path(receipt["identity"]["environment"]["POKER_NATIVE_FLOP_INPUT"])
    board = json.loads(original_input.read_text())["public"]["board"]
    baseline_root = original_input.parent.parent
    equity = baseline_root / "jobs" / row["spot"] / str(row["seed"]) / "equity.json"
    identity = dict(schema="native-inner-budget-first-quality-screen-v1", pinnedInputs=pinned,
                    historicalConstructionSourceHashes=historical_sources, packetWorkers=2, maximumSeconds=7200,
                    releaseAccepted=False, generatorPilotPassed=False)
    if output.exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("status") == "complete" or any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("completed quality screen or changed resume identity")
    output.mkdir(exist_ok=True)
    record = dict(**identity, status="running", cases=[])
    stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set); timer.daemon = True
    with controller_lock(output):
        atomic_json(output / "manifest.json", record); timer.start()
        pressure = PilotMemoryGuard(stop, output / "system-memory.json").start()
        try:
            candidate = output / "candidate.json"
            run_job(binary, RESPONSE + "saved_native_flop_response_candidate",
                dict(POKER_NATIVE_FLOP_CANDIDATE=str(raw), POKER_NATIVE_FLOP_CANDIDATE_SHA=row["trainingCandidateSha256"],
                     POKER_NATIVE_FLOP_RESPONSE_ITERATIONS="64", POKER_NATIVE_FLOP_OUTPUT=str(candidate)),
                output / "canonical-export", [candidate], stop, seconds=30, memory=MEMORY)
            if json.loads(candidate.read_text()) != evaluation_candidate(json.loads(raw.read_text())):
                raise ValueError("canonical export changed the trained policy or response budget")
            row["sourceCandidateSha256"] = row["candidateSha256"]; row["candidateSha256"] = sha256(candidate)
            packets = output / "packets"; packets.mkdir(exist_ok=True)
            common = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate), POKER_NATIVE_FLOP_CANDIDATE_SHA=row["candidateSha256"])
            def packet(turn):
                path = packets / f"turn-{turn}.json"
                receipt = run_job(binary, RESPONSE + "saved_native_flop_turn_packet",
                    {**common, "POKER_NATIVE_FLOP_TURN": str(turn), "POKER_NATIVE_FLOP_OUTPUT": str(path)},
                    output / f"turn-{turn}", [path], stop, seconds=900, memory=MEMORY)
                value = json.loads(path.read_text())
                if (value["candidate_sha256"] != row["candidateSha256"] or value["turn"] != turn
                        or value["turn_iterations"] != 64): raise ValueError("stronger played budget or packet identity changed")
                return dict(turn=turn, sha256=sha256(path), seconds=receipt["worker"]["workerElapsedSeconds"])
            turns = [turn for turn in range(52) if turn not in board]; completed = {}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(packet, turn) for turn in turns]
                try:
                    for index, future in enumerate(as_completed(futures), 1):
                        result = future.result(); completed[str(result["turn"])] = result
                        if index % 7 == 0 or index == 49:
                            print(json.dumps(dict(event="native4-quality-turns", completed=index, total=49)), flush=True)
                except BaseException:
                    stop.set()
                    for future in futures: future.cancel()
                    raise
            response = output / "response.json"
            run_job(binary, RESPONSE + "saved_native_flop_response_aggregate",
                {**common, "POKER_NATIVE_FLOP_PACKET_DIRECTORY": str(packets), "POKER_NATIVE_FLOP_OUTPUT": str(response)},
                output / "aggregate", [response], stop, seconds=180)
            run_job(binary, None, {}, output / "audit", [], stop, seconds=180,
                command=["node", str(HERE / "audit_native_flop_response.mjs"), str(candidate), str(packets), str(equity), str(response)])
            row.update(gainBb=json.loads(response.read_text())["half_summed_gain_bb"],
                       responseSha256=sha256(response), packets=completed)
            rejection = case_rejection(row)
            record.update(status="complete", cases=[row], firstControlPassed=rejection is None, rejectionReason=rejection,
                          interpretation="One complete all49 native64 response; other roots/seeds and full-game exploitability unmeasured")
            if stop.is_set() or any(sha256(Path(path)) != digest for path, digest in pinned.items()):
                raise ValueError("quality screen stopped or inputs changed")
            print(json.dumps(dict(event="native4-quality-result", gainBb=row["gainBb"], firstControlPassed=rejection is None)), flush=True)
        except Exception as error:
            record.update(status="failed", failure=str(error)); stop.set(); raise
        finally:
            timer.cancel(); record["elapsedSeconds"] = time.monotonic() - started
            record["systemMemoryGuard"] = pressure.finish(); atomic_json(output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("construction", "hash-preflight", "binary"):
        p.add_argument("--" + name, type=Path, required=True); p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--output", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()

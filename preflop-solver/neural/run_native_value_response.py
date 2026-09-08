"""Paired policy test: learned flop leaves, unchanged native played turn/river policy.

Reuses and re-audits existing matched 32/64 native controls. Evaluates all public
turns before maximizing a flop response. This is NOT full-game exploitability.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

PREFIX = "blueprint::public_belief::counterfactual_turn::flop_pilot::"
RESPONSE = PREFIX + "frozen_response::tests::"


def select_control_profiles(control, context):
    rows=control.get("profiles",[])
    if context is not None:
        rows=[p for p in rows if p.get("context")==context and p.get("iterations")==32]
    if control.get("status")!="complete" or len(rows)!=2 or {p["seed"] for p in rows}!={100101,100102}:
        raise ValueError("select exactly two completed native32 controls")
    return rows


def validate_search_budget(iterations, prior, students, transfer=None, updated_values=False):
    if iterations not in (32,128): raise ValueError("unpreflighted flop update budget")
    if updated_values and (iterations!=128 or prior is None or transfer is not None):
        raise ValueError("updated values require a same-budget 128-update control, not a transfer reference")
    reference_iterations=128 if updated_values else 32
    if transfer is not None:
        if prior is not None or iterations!=128 or transfer.get("bothSeedsRetainOrImproveResponse") is not True:
            raise ValueError("transfer needs a successful fixed 128-update pair and no competing learned32 reference")
        prior,reference_iterations=transfer,128
    if prior is None:
        if iterations != 32: raise ValueError("extra flop updates require the same-weight 32-update reference")
        return {}
    if (prior.get("status") != "complete" or prior.get("flopIterations") != reference_iterations
            or prior.get("trainingTurnIterations") != 64 or prior.get("playedTurnIterations") != 64
            or len(prior.get("profiles",[])) != 2):
        raise ValueError("complete 32/64/64 learned reference required")
    rows = {p["seed"]:p for p in prior["profiles"]}
    if set(rows) != {100101,100102}: raise ValueError("unmatched reference seeds")
    for index,student in enumerate(students["predictions"]):
        row=rows[100101+index]
        if (not updated_values and row["modelSha256"] != student["modelSha256"]) or "comparison" not in row:
            raise ValueError("reference weights changed or evaluation incomplete")
    return {} if transfer is not None else rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "students", "control", "input", "equity", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("binary", "students", "control", "input", "equity"):
        parser.add_argument("--" + key + "-sha256", required=True)
    parser.add_argument("--flop-iterations",type=int,choices=(32,128),default=32)
    parser.add_argument("--learned-control",type=Path)
    parser.add_argument("--learned-control-sha256")
    parser.add_argument("--updated-values",action="store_true",
                        help="compare new weights to a pinned learned128 control at identical search budgets")
    parser.add_argument("--transfer-from",type=Path)
    parser.add_argument("--transfer-from-sha256")
    parser.add_argument("--control-context")
    args = parser.parse_args()
    pinned = {}
    for name in ("binary", "students", "control", "input", "equity", "output"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        if name != "output":
            expected = getattr(args, name + "_sha256")
            if sha256(path) != expected: raise ValueError("pinned input mismatch: " + name)
            pinned[str(path)] = expected
    students = json.loads(args.students.read_text())
    control = json.loads(args.control.read_text())
    if students["status"] != "complete" or len(students["predictions"]) != 2:
        raise ValueError("completed paired students and controls required")
    control_profiles=select_control_profiles(control,args.control_context)
    if bool(args.learned_control) != bool(args.learned_control_sha256):
        raise ValueError("learned reference path and hash must be provided together")
    prior=None
    if args.learned_control:
        args.learned_control=args.learned_control.resolve()
        if sha256(args.learned_control) != args.learned_control_sha256: raise ValueError("learned reference changed")
        pinned[str(args.learned_control)]=args.learned_control_sha256
        prior=json.loads(args.learned_control.read_text())
    if bool(args.transfer_from) != bool(args.transfer_from_sha256):
        raise ValueError("transfer reference path and hash must be provided together")
    transfer=None
    if args.transfer_from:
        args.transfer_from=args.transfer_from.resolve()
        if sha256(args.transfer_from)!=args.transfer_from_sha256: raise ValueError("transfer reference changed")
        pinned[str(args.transfer_from)]=args.transfer_from_sha256
        transfer=json.loads(args.transfer_from.read_text())
    prior_rows=validate_search_budget(args.flop_iterations,prior,students,transfer,args.updated_values)
    if args.output.exists(): raise ValueError("refusing to overwrite a comparison")
    args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(5400, stop.set)
    timer.daemon = True
    timer.start()
    audit = Path(__file__).resolve().parent / "audit_native_flop_response.mjs"
    pinned[str(audit)] = sha256(audit)
    audit_helper = audit.with_name("native_action_diagnostics.mjs")
    pinned[str(audit_helper)] = sha256(audit_helper)
    value_probe_helper = audit.with_name("native_action_value_probe.mjs")
    pinned[str(value_probe_helper)] = sha256(value_probe_helper)
    pinned[str(args.equity.with_suffix(".f32le"))] = sha256(args.equity.with_suffix(".f32le"))
    record = dict(schema="native-value-paired-policy-response-v1", status="running", releaseAccepted=False,
                  pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)), profiles=[],
                  flopIterations=args.flop_iterations, trainingTurnIterations=64, playedTurnIterations=64,
                  nativeControlFlopIterations=32, nativeControlMatchedFlopUpdates=args.flop_iterations==32,
                  comparisonKind=("updated_values_fixed_128_updates_vs_learned128_and_native32" if args.updated_values else
                                  "fixed_128_update_transfer_vs_native32" if transfer else
                                  "matched_updates" if args.flop_iterations==32 else "same_weights_more_flop_updates_vs_learned32_and_native32"),
                  controlContext=args.control_context,
                  learnedReferenceFlopIterations=prior["flopIterations"] if prior_rows else None,
                  maximumStageSeconds=5400, maximumPacketWorkers=4, maximumWorkerMemoryBytes=2*1024**3,
                  interpretation="One reused development flop root; paired conditional response gains, not full-game exploitability or release qualification. At 128 updates the native32 comparison is a compute-allocation comparison, not equal-update evidence.")
    atomic_json(args.output / "manifest.json", record)
    started = time.monotonic()
    try:
        for index, student in enumerate(students["predictions"]):
            seed = 100101 + index
            stage = args.output / str(seed)
            stage.mkdir()
            baseline = next(p for p in control_profiles if p["seed"] == seed)
            baseline_path = Path(baseline["path"])
            if sha256(baseline_path) != baseline["sha256"]: raise ValueError("control policy changed")
            baseline_policy = json.loads(baseline_path.read_text())
            if baseline_policy["iterations"] != 32 or baseline_policy["turn_iterations"] != 64 or baseline_policy.get("response_turn_iterations",64) != 64:
                raise ValueError("unmatched control budget")
            baseline_env = {"POKER_NATIVE_FLOP_CANDIDATE": str(baseline_path), "POKER_NATIVE_FLOP_CANDIDATE_SHA": baseline["sha256"],
                            "POKER_NATIVE_FLOP_PACKET_DIRECTORY": baseline["packets"], "POKER_NATIVE_FLOP_OUTPUT": str(stage / "control-response.json")}
            guarded(test_command(args.binary, RESPONSE + "saved_native_flop_response_aggregate"), baseline_env, stage / "control-replay", 180, stop=stop)
            expected = next(a["outputSha256"] for a in control["aggregates"] if a["name"] == baseline["name"] + "/aggregate")
            if sha256(stage / "control-response.json") != expected: raise ValueError("control response replay changed")
            guarded(["node", str(audit), str(baseline_path), baseline["packets"], str(args.equity), str(stage / "control-response.json")], {}, stage / "control-audit", 180, stop=stop)
            model = Path(student["model"])
            if sha256(model) != student["modelSha256"]: raise ValueError("student model changed")
            previous_gain=None
            if seed in prior_rows:
                previous=prior_rows[seed]
                previous_candidate=Path(previous["candidate"])
                previous_response=previous_candidate.parent/"response.json"
                for path,digest in ((previous_candidate,previous["candidateSha256"]),
                                    (previous_response,previous["comparison"]["responseSha256"])):
                    if sha256(path)!=digest: raise ValueError("learned control artifact changed")
                    pinned[str(path)]=digest
                old=json.loads(previous_candidate.read_text())
                if old["state"]!=baseline_policy["state"] or old["game"]!=baseline_policy["game"]:
                    raise ValueError("learned control root/game changed")
                previous_gain=json.loads(previous_response.read_text())["half_summed_gain_bb"]
                if previous_gain!=previous["comparison"]["learnedGainBb"]: raise ValueError("learned reference gain changed")
            candidate = stage / "candidate.json"
            env = {"POKER_NATIVE_FLOP_INPUT": str(args.input), "POKER_NATIVE_FLOP_INPUT_SHA": args.input_sha256,
                   "POKER_NATIVE_FLOP_SEED": str(seed), "POKER_NATIVE_FLOP_ITERATIONS": str(args.flop_iterations), "POKER_NATIVE_FLOP_TURN_ITERATIONS": "64",
                   "POKER_NATIVE_FLOP_CHANCE_BASELINE": "none", "POKER_NATIVE_FLOP_TURN_SAMPLES": "1", "POKER_NATIVE_FLOP_LEAF_WORKERS": "4",
                   "POKER_NATIVE_FLOP_VALUE_MODEL": str(model), "POKER_NATIVE_FLOP_VALUE_MODEL_SHA": student["modelSha256"],
                   "POKER_NATIVE_FLOP_OUTPUT": str(candidate)}
            training = guarded(test_command(args.binary, PREFIX + "saved_20bb_native_flop_pilot"), env, stage / "learned-training", 900, stop=stop)
            candidate_sha = sha256(candidate)
            derived = json.loads(candidate.read_text())
            if (derived["state"] != baseline_policy["state"] or derived["game"] != baseline_policy["game"]
                    or derived["learned_leaf_model_sha256"] != student["modelSha256"]
                    or derived["iterations"] != args.flop_iterations
                    or derived["maximum_conditional_turn_response_gain_bb"] is not None):
                raise ValueError("learned candidate root/game/diagnostic mismatch")
            packets = stage / "packets"
            packets.mkdir()
            common = {"POKER_NATIVE_FLOP_CANDIDATE": str(candidate), "POKER_NATIVE_FLOP_CANDIDATE_SHA": candidate_sha}
            profile = dict(seed=seed, modelSha256=student["modelSha256"], candidateSha256=candidate_sha,
                           candidate=str(candidate), training=training, packets=[])
            record["profiles"].append(profile)
            def packet(turn):
                if stop.is_set(): raise ValueError("response stage stopped")
                path = packets / ("turn-%d.json" % turn)
                worker = guarded(test_command(args.binary, RESPONSE + "saved_native_flop_turn_packet"),
                                 {**common, "POKER_NATIVE_FLOP_TURN": str(turn), "POKER_NATIVE_FLOP_OUTPUT": str(path)},
                                 stage / ("turn-%d" % turn), 420, stop=stop)
                return dict(turn=turn, sha256=sha256(path), worker=worker)
            turns = [c for c in range(52) if c not in derived["state"]["board"]]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(packet, turn) for turn in turns]
                try:
                    for future in as_completed(futures):
                        profile["packets"].append(future.result())
                        atomic_json(args.output / "manifest.json", record)
                        print(json.dumps(dict(event="native-response-progress", seed=seed, turns=len(profile["packets"]), total=49)), flush=True)
                except BaseException:
                    stop.set()
                    raise
            response = stage / "response.json"
            guarded(test_command(args.binary, RESPONSE + "saved_native_flop_response_aggregate"),
                    {**common,"POKER_NATIVE_FLOP_PACKET_DIRECTORY":str(packets),"POKER_NATIVE_FLOP_OUTPUT":str(response)}, stage / "aggregate",180,stop=stop)
            guarded(["node",str(audit),str(candidate),str(packets),str(args.equity),str(response)],{},stage / "independent-audit",180,stop=stop)
            measured = json.loads(response.read_text())["half_summed_gain_bb"]
            baseline_gain = json.loads((stage / "control-response.json").read_text())["half_summed_gain_bb"]
            profile["comparison"] = dict(controlGainBb=baseline_gain, learnedGainBb=measured, learnedMinusControlBb=measured-baseline_gain,
                                         responseSha256=sha256(response), controlResponseSha256=expected)
            if previous_gain is not None:
                profile["comparison"].update(learnedReferenceGainBb=previous_gain,learnedMinusReferenceBb=measured-previous_gain)
                if not args.updated_values:
                    profile["comparison"].update(learned32GainBb=previous_gain,learnedMinusLearned32Bb=measured-previous_gain)
            if sha256(model) != student["modelSha256"] or sha256(candidate) != candidate_sha: raise ValueError("candidate inputs changed")
            print(json.dumps(dict(event="native-policy-comparison", seed=seed, **profile["comparison"])),flush=True)
            atomic_json(args.output / "manifest.json", record)
        if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()): raise ValueError("stopped stage or pinned input changed")
        record["status"] = "complete"
        record["bothSeedsRetainOrImproveResponse"] = all(p["comparison"]["learnedMinusControlBb"] <= 0 for p in record["profiles"])
        if prior_rows:
            record["bothSeedsImproveLearnedReference"] = all(p["comparison"]["learnedMinusReferenceBb"] < 0 for p in record["profiles"])
            if not args.updated_values: record["bothSeedsImproveLearned32"] = record["bothSeedsImproveLearnedReference"]
    except (OSError, ValueError, KeyError) as error:
        stop.set()
        record["status"], record["failure"] = "failed", str(error)
    finally:
        timer.cancel()
        record["elapsedSeconds"] = time.monotonic() - started
        atomic_json(args.output / "manifest.json", record)
    print(json.dumps(dict(status=record["status"], failure=record.get("failure"), comparisons=[p.get("comparison") for p in record["profiles"]])), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()

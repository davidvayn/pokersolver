"""Sixteen explicitly intervened public-belief labels on four frozen proposers.

Diagnostic only: neither source policies nor training/held-out corpora change.
The native policies solved at intervened ranges are not the source policy's
played continuation, so this is never a source-policy response evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

import native_value_dataset as native
from run_native_value_pilot import guarded, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from train_public_value_network import weighted_metrics

TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::forced_beliefs::saved_forced_action_native_capture"


def diagnostic_metrics(source):
    native.validate_dataset(source)
    if (source.get("capture_selection") != "explicit_forced_root_actions_diagnostic_only"
            or source.get("source_policy_unchanged") is not True
            or source.get("releaseAccepted") is not False
            or len(source["targets"]) != 4 or len(source["interventions"]) != 4):
        raise ValueError("invalid forced-belief provenance")
    for row in source["interventions"]:
        if (row["forced_action_probability"] != 1 or not 0 <= row["source_action_frequency"] <= 1
                or not native._sha(row["intervened_proposer_sha256"])
                or not native._sha(row["public_belief_sha256"])):
            raise ValueError("invalid explicit intervention")
    truth = np.asarray([t["counterfactual_values_bb"] for t in source["targets"]])
    values = np.asarray(source["predictions"])
    weights = np.asarray([np.asarray(t["ranges"])*np.asarray(t["opponent_compatible_mass"])
                          for t in source["targets"]])
    if values.shape != (4,2,1326) or not np.isfinite(values).all() or np.max(np.abs(values)) > 20+1e-8:
        raise ValueError("invalid native predictions")
    if any((values[i,:,~native.legal_combos(t["board"])] != 0).any() for i,t in enumerate(source["targets"])):
        raise ValueError("board-blocked predicted value")
    if any(t["state_distribution"] != "explicit_forced_root_action_public_belief_native_label" for t in source["targets"]):
        raise ValueError("missing forced-distribution tag")
    return weighted_metrics(truth.reshape(4,-1),values.reshape(4,-1),weights.reshape(4,-1),np.ones(4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary","students"):
        parser.add_argument("--"+name,type=Path,required=True)
        parser.add_argument("--"+name+"-sha256",required=True)
    parser.add_argument("--comparison",type=Path,action="append",required=True)
    parser.add_argument("--comparison-sha256",action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if len(args.comparison)!=2 or len(args.comparison_sha256)!=2 or len({p.resolve() for p in args.comparison})!=2:
        raise ValueError("exactly two completed paired contexts required")
    args.binary,args.students,args.output = [p.resolve() for p in (args.binary,args.students,args.output)]
    pinned = {}
    for path,digest in [(args.binary,args.binary_sha256),(args.students,args.students_sha256),
                        *zip([p.resolve() for p in args.comparison],args.comparison_sha256)]:
        if sha256(path)!=digest: raise ValueError("pinned input changed")
        pinned[str(path)] = digest
    students = json.loads(args.students.read_text())
    comparisons = [json.loads(p.read_text()) for p in args.comparison]
    if students.get("status")!="complete" or len(students["predictions"])!=2:
        raise ValueError("complete paired students required")
    for m in comparisons:
        if (m.get("status")!="complete" or len(m["profiles"])!=2
                or m["pinnedInputs"].get(str(args.students))!=args.students_sha256):
            raise ValueError("completed source-policy comparisons must use these students")
    for name in ("native_value_dataset.py","run_native_value_pilot.py","run_native_value_preflight.py","worker_resources.py","train_public_value_network.py"):
        path=Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output.mkdir(exist_ok=False)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    started=time.monotonic()
    record=dict(schema="native-forced-belief-diagnostic-v1",status="running",releaseAccepted=False,
                pinnedInputs=pinned,runnerSha256=sha256(Path(__file__)),profiles=[],states=16,
                interpretation=__doc__)
    atomic_json(args.output/"manifest.json",record)
    try:
        for context,m in enumerate(comparisons):
            for profile in m["profiles"]:
                if stop.is_set() or any(sha256(Path(p))!=h for p,h in pinned.items()):
                    raise ValueError("stopped stage or changed input")
                student=next(p for p in students["predictions"] if p["modelSha256"]==profile["modelSha256"])
                candidate,model=Path(profile["candidate"]),Path(student["model"])
                for path,digest in ((candidate,profile["candidateSha256"]),(model,student["modelSha256"])):
                    if sha256(path)!=digest: raise ValueError("source/model artifact changed")
                    pinned[str(path)]=digest
                name="context-%d-seed%d"%(context,profile["seed"])
                output=args.output/(name+".json.gz")
                worker=guarded(test_command(args.binary,TEST),{
                    "POKER_FORCED_CANDIDATE":str(candidate),"POKER_FORCED_CANDIDATE_SHA":profile["candidateSha256"],
                    "POKER_FORCED_MODEL":str(model),"POKER_FORCED_MODEL_SHA":student["modelSha256"],
                    "POKER_FORCED_SAMPLE_SEED":str(90801+context),"POKER_FORCED_OUTPUT":str(output)
                },args.output/(name+"-worker"),300,2*1024**3,stop)
                source=read_capture(output)
                if source["source_policy_sha256"]!=profile["candidateSha256"] or source["proposal_model_sha256"]!=student["modelSha256"]:
                    raise ValueError("capture source identity changed")
                metrics=diagnostic_metrics(source)
                row=dict(context=context,seed=profile["seed"],path=str(output),sha256=sha256(output),
                         intervenedJointReachMetrics=metrics,worker=worker)
                record["profiles"].append(row)
                if len(record["profiles"])==1: record["preflightPassed"]=True
                atomic_json(args.output/"manifest.json",record)
                print(json.dumps(dict(event="forced-beliefs-complete",context=context,seed=profile["seed"],rmseBb=metrics["weightedRmseBb"])),flush=True)
        if stop.is_set() or any(sha256(Path(p))!=h for p,h in pinned.items()):
            raise ValueError("stopped stage or changed input")
        record["status"]="complete"
    except (OSError,ValueError,KeyError,StopIteration) as error:
        stop.set()
        record["status"],record["failure"]="failed",str(error)
    record["elapsedSeconds"]=time.monotonic()-started
    atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","failure")}),flush=True)
    if record["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()

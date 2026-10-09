"""Own-policy value diagnosis for one completed, audited control; no new solve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

from run_native_action_value_probe import TEST
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard


def completed_case(response, students, students_sha256, spot, seed):
    if (response.get("schema") != "postflop-student-value-pilot-v1"
            or response.get("status") not in ("complete", "rejected")
            or response.get("studentManifestSha256") != students_sha256
            or students.get("status") != "complete"):
        raise ValueError("completed or rejected audited response and matching students required")
    rows = [r for r in response.get("cases", []) if (r["spot"], r["seed"]) == (spot, seed)]
    if len(rows) != 1:
        raise ValueError("exactly one completed control required; never use a partial case")
    row = rows[0]
    expected_seed = {100101: 10601, 100102: 10602}[seed]
    models = [r for r in students["predictions"] if r["seed"] == expected_seed]
    if (len(models) != 1 or models[0]["modelSha256"] != row["modelSha256"]
            or not 0 <= models[0]["maximumParityErrorBb"] <= .0001):
        raise ValueError("completed control model/parity pairing differs")
    return row, models[0]


def chance_turns(candidate):
    if candidate.get("schema") != "hu-native-counterfactual-turn-flop-pilot-v1":
        raise ValueError("native flop candidate required")
    state = candidate.get("state", {})
    board = state.get("board", [])
    if (state.get("street") != "flop" or len(board) != 3 or len(set(board)) != 3
            or any(type(card) is not int or not 0 <= card < 52 for card in board)):
        raise ValueError("exact unique flop board required")
    return [turn for turn in range(52) if turn not in board]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","response-manifest","students","equity"):
        p.add_argument("--"+name,type=Path,required=True)
        p.add_argument("--"+name+"-sha256",required=True)
    p.add_argument("--spot",required=True)
    p.add_argument("--seed",type=int,choices=(100101,100102),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); pinned={}
    for name in ("binary","response_manifest","students","equity"):
        path=getattr(a,name).resolve(); setattr(a,name,path)
        digest=getattr(a,name+"_sha256")
        if sha256(path)!=digest: raise ValueError("completed control identity changed")
        pinned[str(path)]=digest
    response=json.loads(a.response_manifest.read_text()); students=json.loads(a.students.read_text())
    row, model_row = completed_case(response, students, a.students_sha256, a.spot, a.seed)
    model=Path(model_row["model"]).resolve()
    case=a.response_manifest.parent/a.spot/str(a.seed)
    candidate, response_file, packets=case/"candidate.json",case/"response.json",case/"packets"
    pinned[str(model)]=row["modelSha256"]
    pinned[str(candidate)]=row["candidateSha256"]; pinned[str(response_file)]=row["responseSha256"]
    turns=chance_turns(json.loads(candidate.read_text()))
    # Audit and prediction independently validate histories and complete private
    # vectors. Pin every chance packet, not just the aggregate response file.
    for turn in turns:
        path=packets/f"turn-{turn}.json"; pinned[str(path)]=sha256(path)
    blob=a.equity.with_suffix(".f32le"); pinned[str(blob)]=sha256(blob)
    for name in (Path(__file__).name,"run_native_value_pilot.py","worker_resources.py",
                 "audit_native_flop_response.mjs","native_action_diagnostics.mjs",
                 "native_action_value_probe.mjs","diagnose_native_action_values.mjs"):
        path=Path(__file__).with_name(name); pinned[str(path)]=sha256(path)
    if any(sha256(Path(path))!=digest for path,digest in pinned.items()):
        raise ValueError("cached diagnostic source changed")
    a.output=a.output.resolve()
    if a.output.exists(): raise ValueError("never overwrite a completed-control diagnosis")
    a.output.mkdir(); stop=threading.Event(); started=time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    pressure=PilotMemoryGuard(stop,a.output/"system-memory.json").start()
    record=dict(schema="completed-control-own-policy-diagnosis-v1",status="running",releaseAccepted=False,
        pinnedInputs=pinned,spot=a.spot,seed=a.seed,policyGainBb=row["gainBb"],newNativeSolves=0)
    atomic_json(a.output/"manifest.json",record)
    try:
        predicted,diagnosed=a.output/"predictions.json",a.output/"action-values.json"
        record["predictionWorker"]=guarded(test_command(a.binary,TEST),dict(
            POKER_FROZEN_PREDICT_CANDIDATE=str(candidate),POKER_FROZEN_PREDICT_CANDIDATE_SHA=row["candidateSha256"],
            POKER_FROZEN_PREDICT_MODEL=str(model),POKER_FROZEN_PREDICT_MODEL_SHA=row["modelSha256"],
            POKER_FROZEN_PREDICT_OUTPUT=str(predicted),POKER_FROZEN_PREDICT_ALTERNATIVE=""),
            a.output/"prediction-worker",300,2*1024**3,stop)
        record["diagnosticWorker"]=guarded(["node",str(Path(__file__).with_name("diagnose_native_action_values.mjs")),
            str(candidate),str(packets),str(a.equity),str(response_file),str(predicted),str(diagnosed),"--action-diagnostics"],
            {},a.output/"diagnostic-worker",120,2*1024**3,stop)
        result=json.loads(diagnosed.read_text())
        if abs(result["half_summed_gain_bb"]-row["gainBb"])>1e-8:
            raise ValueError("own-policy diagnosis changed the audited policy gain")
        if stop.is_set() or any(sha256(Path(path))!=digest for path,digest in pinned.items()):
            raise ValueError("diagnosis stopped or source changed")
        record["predictionsSha256"]=sha256(predicted); record["diagnosticSha256"]=sha256(diagnosed)
        record["status"]="complete"
    except Exception as error:
        record["status"]="failed"; record["failure"]=str(error); stop.set(); raise
    finally:
        record["elapsedSeconds"]=time.monotonic()-started; record["systemMemoryGuard"]=pressure.finish()
        atomic_json(a.output/"manifest.json",record)
    print(json.dumps(dict(status=record["status"],newNativeSolves=0,seconds=record["elapsedSeconds"])),flush=True)


if __name__=="__main__":main()

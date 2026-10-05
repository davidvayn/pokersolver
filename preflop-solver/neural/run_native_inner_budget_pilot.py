"""Matched native4 construction / native64 played-response cost pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import (HERE, MEMORY, PREFIX, RESPONSE, SERIAL_SOLVE_MEMORY,
    SPOTS, PilotMemoryGuard, controller_lock, estimate_work_seconds, run_job, same_public_state, select_cases)


def evaluation_candidate(candidate):
    if (candidate.get("schema")!="hu-native-counterfactual-turn-flop-pilot-v1"
            or candidate.get("iterations")!=32 or candidate.get("turn_iterations")!=4
            or candidate.get("response_turn_iterations") is not None
            or candidate.get("learned_leaf_model_sha256") is not None
            or not candidate.get("strategies")):
        raise ValueError("unmodified native4/32 construction required")
    result=copy.deepcopy(candidate);result["response_turn_iterations"]=64
    if ({k:v for k,v in result.items() if k!="response_turn_iterations"}
            !={k:v for k,v in candidate.items() if k!="response_turn_iterations"}):
        raise ValueError("response override changed the trained policy")
    return result


def case_rejection(row):
    names=("gainBb","native64GainBb","learned32GainBb","solveSeconds","native64SolveSeconds")
    if (any(not math.isfinite(row[name]) or row[name]<0 for name in names)
            or row["native64SolveSeconds"]<=0 or row["learned32GainBb"]<=row["native64GainBb"]):
        raise ValueError("completed finite policy/cost measurements required")
    if row["gainBb"]>row["native64GainBb"]+.05:
        return "native4 response regresses native64 by more than 0.05bb"
    if row["gainBb"]>(row["native64GainBb"]+row["learned32GainBb"])/2:
        return "native4 retains less than half the measured native leaf benefit"
    if row["solveSeconds"]>row["native64SolveSeconds"]*.2:
        return "native4 construction fails the fivefold speed target"
    return None


def controls(reference,cases):
    if (reference.get("schema")!="postflop-gap-matched-leaf-pilot-v1"
            or reference.get("status")!="complete" or reference.get("phase")!="compare"
            or reference.get("iterations")!=32 or reference.get("spots")!=list(SPOTS)):
        raise ValueError("complete original matched32 native/learned reference required")
    rows={(r["spot"],r["seed"],r["arm"]):r for r in reference["cases"]}
    required={(spot["id"],model["seed"],arm) for spot,model,_ in cases for arm in ("native","learned")}
    if len(rows)!=8 or len(reference["cases"])!=8 or set(rows)!=required:
        raise ValueError("exact four paired native/learned controls required")
    for row in rows.values():
        if any(not math.isfinite(row[key]) or row[key]<=0 for key in ("gainBb","solveSeconds")):
            raise ValueError("invalid reference cost or response gain")
    return rows


def construction_environment(root_input,input_sha,seed,output,environ=None):
    inherited=os.environ if environ is None else environ
    if any(name in inherited for name in ("POKER_NATIVE_FLOP_VALUE_MODEL", "POKER_NATIVE_FLOP_VALUE_MODEL_SHA")):
        raise ValueError("native-only construction cannot inherit neural model environment")
    # Absence, not an empty path, selects the Rust native evaluator.
    return dict(POKER_NATIVE_FLOP_INPUT=str(root_input),POKER_NATIVE_FLOP_INPUT_SHA=input_sha,
        POKER_NATIVE_FLOP_SEED=str(seed),POKER_NATIVE_FLOP_ITERATIONS="32",POKER_NATIVE_FLOP_TURN_ITERATIONS="4",
        POKER_NATIVE_FLOP_CHANCE_BASELINE="none",POKER_NATIVE_FLOP_TURN_SAMPLES="1",POKER_NATIVE_FLOP_LEAF_WORKERS="1",
        POKER_NATIVE_FLOP_OUTPUT=str(output))


def run(args):
    args.baseline_root=args.baseline_root.resolve();args.reference=args.reference.resolve();args.output=args.output.resolve()
    export_binary=args.export_binary.resolve()
    if sha256(export_binary)!=args.export_binary_sha256:raise ValueError("canonical export binary changed")
    if sha256(args.reference)!=args.reference_sha256:raise ValueError("native64 reference changed")
    root=args.baseline_root;baseline=root/"complete/manifest.json";protocol_path=root/"protocol.json"
    if sha256(baseline)!=args.baseline_sha256:raise ValueError("frozen baseline changed")
    protocol=json.loads(protocol_path.read_text());protocol["sha256"]=sha256(protocol_path)
    cases=select_cases(protocol,json.loads(baseline.read_text()),list(SPOTS))
    reference=json.loads(args.reference.read_text());old=controls(reference,cases)
    binary=Path(protocol["binary"]);binary_sha=protocol["pinnedInputs"][str(binary)]
    if (sha256(binary)!=binary_sha or reference["binarySha256"]!=binary_sha
            or reference["baselineSha256"]!=args.baseline_sha256 or reference["protocolSha256"]!=protocol["sha256"]):
        raise ValueError("matched original binary/baseline/protocol required")
    pinned={str(args.reference):args.reference_sha256,str(binary):binary_sha,
        str(baseline):args.baseline_sha256,str(protocol_path):protocol["sha256"],str(export_binary):args.export_binary_sha256}
    inputs=json.loads((root/"inputs.json").read_text());pinned[str(root/"inputs.json")]=sha256(root/"inputs.json")
    for spot,model,_ in cases:
        input_path=root/"inputs"/(spot["id"]+".json");pinned[str(input_path)]=inputs["roots"][str(input_path)]
        equity=root/"jobs"/spot["id"]/str(model["seed"])/"equity.json"
        metadata=json.loads(equity.read_text());pinned[str(equity)]=sha256(equity)
        pinned[str(equity.with_suffix(".f32le"))]=metadata["sha256"]
        for arm in ("native","learned"):
            receipt=old[(spot["id"],model["seed"],arm)];work=args.reference.parent/spot["id"]/str(model["seed"])/arm
            candidate=work/"candidate.json";response=work/"response.json";audit=work/"audit/completed.json"
            pinned[str(candidate)]=receipt["candidateSha256"];pinned[str(response)]=receipt["responseSha256"]
            pinned[str(audit)]=sha256(audit)
            payload=json.loads(candidate.read_text());worker=json.loads(audit.read_text())["worker"]
            gain=json.loads(response.read_text())["half_summed_gain_bb"]
            if (payload["seed"]!=model["seed"] or payload["iterations"]!=32 or payload["turn_iterations"]!=64
                    or worker.get("status")!="complete" or worker.get("exitCode")!=0 or worker.get("resourceStopReason")
                    or abs(gain-receipt["gainBb"])>1e-8):raise ValueError("reference lacks completed unchanged policy/audit")
            for turn in range(52):
                if turn not in spot["board"]:
                    packet=work/"packets"/f"turn-{turn}.json";pinned[str(packet)]=sha256(packet)
    for name in (Path(__file__).name,"run_postflop_gap_pilot.py","run_native_value_pilot.py","worker_resources.py",
            "run_native_value_preflight.py","audit_native_flop_response.mjs","native_action_diagnostics.mjs"):
        path=HERE/name;pinned[str(path)]=sha256(path)
    if any(sha256(Path(path))!=digest for path,digest in pinned.items()):raise ValueError("pilot input changed")
    projection=estimate_work_seconds(root,cases,workers=2)["estimatedPacketSeconds"]/2
    identity=dict(schema="native-inner-budget-pilot-v1",pinnedInputs=pinned,maximumSeconds=7200,
        flopIterations=32,constructionTurnIterations=4,playedTurnIterations=64,packetWorkers=2,
        constructionMemoryBytes=int(args.construction_memory_gib*1024**3),
        releaseAccepted=False)
    if args.output.exists():
        existing=json.loads((args.output/"manifest.json").read_text())
        if any(existing.get(k)!=v for k,v in identity.items()):raise ValueError("pilot identity changed on resume")
        if existing.get("status") in ("complete","rejected"):raise ValueError("completed/rejected pilot cannot be rerun")
    else:args.output.mkdir()
    with controller_lock(args.output):
        stop=threading.Event();started=time.monotonic()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
        timer=threading.Timer(7200,stop.set);timer.daemon=True;timer.start()
        pressure=PilotMemoryGuard(stop,args.output/"system-memory.json").start()
        record={**identity,"status":"running","cases":[]};atomic_json(args.output/"manifest.json",record)
        try:
            for index,(spot,model,_) in enumerate(cases):
                if stop.is_set():raise ValueError("native budget pilot stopped")
                seed=model["seed"];control=old[(spot["id"],seed,"native")];learned=old[(spot["id"],seed,"learned")]
                work=args.output/spot["id"]/str(seed);work.mkdir(parents=True,exist_ok=True)
                raw=work/"training-candidate.json";candidate=work/"candidate.json"
                root_input=root/"inputs"/(spot["id"]+".json")
                solve=run_job(binary,PREFIX+"saved_20bb_native_flop_pilot",
                    construction_environment(root_input,pinned[str(root_input)],seed,raw),
                    work/"solve",[raw],stop,seconds=900,memory=identity["constructionMemoryBytes"])
                policy=json.loads(raw.read_text());expected=evaluation_candidate(policy)
                input_value=json.loads(root_input.read_text())
                if (not same_public_state(policy["state"],input_value["public"]) or policy["game"]!=input_value["game"]
                        or policy["seed"]!=seed):raise ValueError("construction changed root/game/seed")
                run_job(export_binary,RESPONSE+"saved_native_flop_response_candidate",
                    dict(POKER_NATIVE_FLOP_CANDIDATE=str(raw),POKER_NATIVE_FLOP_CANDIDATE_SHA=sha256(raw),
                         POKER_NATIVE_FLOP_RESPONSE_ITERATIONS="64",POKER_NATIVE_FLOP_OUTPUT=str(candidate)),
                    work/"canonical-export",[candidate],stop,seconds=30,memory=MEMORY)
                if json.loads(candidate.read_text())!=expected:raise ValueError("canonical evaluation export changed trained policy")
                seconds=solve["worker"]["workerElapsedSeconds"]
                row=dict(spot=spot["id"],seed=seed,trainingCandidateSha256=sha256(raw),candidateSha256=sha256(candidate),
                    solveSeconds=seconds,native64SolveSeconds=control["solveSeconds"],native64GainBb=control["gainBb"],
                    learned32GainBb=learned["gainBb"],peakMemoryBytes=solve["worker"]["sampledPeakMemoryBytes"])
                if index==0:
                    projected=1.5*(seconds*len(cases)+projection)+180
                    record["projectedTotalSeconds"]=projected
                    if seconds>.2*control["solveSeconds"] or projected>7200:
                        record.update(status="rejected",costRejection=row,reason="cost preflight failed; policy response unmeasured")
                        break
                print(json.dumps(dict(event="native4-construction",**row)),flush=True)
                common=dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),POKER_NATIVE_FLOP_CANDIDATE_SHA=row["candidateSha256"])
                packets=work/"packets";packets.mkdir(exist_ok=True)
                def packet(turn):
                    path=packets/f"turn-{turn}.json"
                    run_job(binary,RESPONSE+"saved_native_flop_turn_packet",{**common,"POKER_NATIVE_FLOP_TURN":str(turn),
                        "POKER_NATIVE_FLOP_OUTPUT":str(path)},work/f"turn-{turn}",[path],stop,seconds=900,memory=MEMORY)
                    data=json.loads(path.read_text())
                    if data["candidate_sha256"]!=row["candidateSha256"] or data["turn"]!=turn or data["turn_iterations"]!=64:
                        raise ValueError("played/evaluation continuation budget changed")
                    return sha256(path)
                turns=[turn for turn in range(52) if turn not in spot["board"]]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures={pool.submit(packet,turn):turn for turn in turns};digests={}
                    try:
                        for number,future in enumerate(as_completed(futures),1):
                            digests[str(futures[future])]=future.result()
                            if number%7==0 or number==49:print(json.dumps(dict(event="native4-response-turns",spot=spot["id"],seed=seed,completed=number,total=49)),flush=True)
                    except BaseException:stop.set();raise
                response=work/"response.json"
                run_job(binary,RESPONSE+"saved_native_flop_response_aggregate",{**common,
                    "POKER_NATIVE_FLOP_PACKET_DIRECTORY":str(packets),"POKER_NATIVE_FLOP_OUTPUT":str(response)},
                    work/"aggregate",[response],stop,seconds=180)
                equity=root/"jobs"/spot["id"]/str(seed)/"equity.json"
                run_job(binary,None,{},work/"audit",[],stop,seconds=180,command=["node",str(HERE/"audit_native_flop_response.mjs"),
                    str(candidate),str(packets),str(equity),str(response)])
                row.update(gainBb=json.loads(response.read_text())["half_summed_gain_bb"],responseSha256=sha256(response),packetSha256=digests)
                record["cases"].append(row);atomic_json(args.output/"manifest.json",record)
                print(json.dumps(dict(event="native4-audited-case",**{k:v for k,v in row.items() if k!="packetSha256"})),flush=True)
                rejection=case_rejection(row)
                if rejection:record.update(status="rejected",reason=rejection);break
            if stop.is_set() or any(sha256(Path(path))!=digest for path,digest in pinned.items()):raise ValueError("pilot stopped or pinned source changed")
            if record["status"]!="rejected":
                if len(record["cases"])!=4:raise ValueError("cannot score an incomplete pair")
                record["status"]="complete";record["generatorPilotPassed"]=True
        except Exception as error:
            record["status"]="failed";record["failure"]=str(error);stop.set();raise
        finally:
            timer.cancel();record["elapsedSeconds"]=time.monotonic()-started;record["systemMemoryGuard"]=pressure.finish()
            atomic_json(args.output/"manifest.json",record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root",type=Path,required=True);p.add_argument("--baseline-sha256",required=True)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--reference-sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--export-binary",type=Path,required=True);p.add_argument("--export-binary-sha256",required=True)
    p.add_argument("--construction-memory-gib",type=float,choices=(2.5,4),default=SERIAL_SOLVE_MEMORY/1024**3)
    run(p.parse_args())


if __name__=="__main__":main()

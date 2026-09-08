"""Fixed frozen preflop/history continuation-noise diagnostic; never a release gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::preflop_continuation::fixed_noise::frozen_open_call_continuation_noise"


def summarize(outputs):
    if not outputs or len({r["boardIndex"] for r in outputs}) != len(outputs):
        raise ValueError("independent board clusters required")
    first = outputs[0]
    for r in outputs:
        if (r["schema"] != "frozen-preflop-continuation-noise-v1" or r["releaseAccepted"] is not False
                or any(r[k] != first[k] for k in ("preflopSha256", "modelSha256", "kernelSha256",
                    "history", "classes", "classReachWeights", "turnSampleCount"))
                or r["turnSampleCount"] != len(r["records"])
                or len({s["turn"] for s in r["records"]}) != len(r["records"])
                or len(set(r["board"])) != 3
                or any(s["turn"] in r["board"] for s in r["records"])):
            raise ValueError("mixed identity, invalid chance or incomplete fixed-noise data")
    values = np.asarray([[r["conditionalStrategicResidualBb"] for r in o["records"]] for o in outputs],dtype=float)
    weights = np.asarray(first["classReachWeights"],dtype=float)
    b,t,_,_ = values.shape
    if (t not in (2,4) or values.shape != (b,t,2,169) or weights.shape != (2,169)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights<0).any() or weights.sum()<=0):
        raise ValueError("invalid fixed-noise vectors")
    weights = weights/weights.sum()
    within = float((values.var(axis=1,ddof=1).mean(axis=0)*weights).sum())
    within_mean = within*(1-t/49)/t  # simple random sample WITHOUT replacement
    between = float((values.mean(axis=1).var(axis=0,ddof=1)*weights).sum()) if b>1 else None
    return dict(independentBoardClusters=b, turnsPerBoard=t,
        reachWeightedWithinTurnVarianceBb2=within,
        withinTurnContributionToBoardMeanVarianceBb2=within_mean,
        observedBoardMeanVarianceBb2=between,
        estimatedFlopVarianceComponentBb2=None if between is None else between-within_mean,
        classMeanErrorRmsBb=None if between is None else (between/b)**0.5,
        releaseAccepted=False, fullHandExploitability=None,
        interpretation="Small nested chance diagnostic at fixed frozen 2bb-open/call ranges. Board clusters, not private hands, are independent. The estimated component may be negative from sampling noise; do not clamp or treat this as an action-EV precision/release gate.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","preflop","model","kernel"):
        p.add_argument("--"+name,type=Path,required=True)
        p.add_argument("--"+name+"-sha256",required=True)
    p.add_argument("--boards",type=int,choices=[1,4],default=1)
    p.add_argument("--turns",type=int,choices=[2,4],default=2)
    p.add_argument("--preflight",type=Path)
    p.add_argument("--preflight-sha256")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    pinned={}
    for name in ("binary","preflop","model","kernel"):
        path=getattr(args,name).resolve(); digest=getattr(args,name+"_sha256")
        if sha256(path)!=digest: raise ValueError("fixed-noise input changed")
        setattr(args,name,path); pinned[str(path)]=digest
    if (args.boards,args.turns)!=(1,2):
        if not args.preflight or sha256(args.preflight)!=args.preflight_sha256:
            raise ValueError("larger diagnostic requires pinned one-board/two-turn preflight")
        prior=json.loads(args.preflight.read_text())
        if (prior["status"]!="complete" or (prior["boards"],prior["turns"])!=(1,2)
                or any(prior["pinnedInputs"].get(k)!=v for k,v in pinned.items())):
            raise ValueError("incompatible fixed-noise cost preflight")
        if prior["elapsedSeconds"]*args.boards*args.turns/2*1.5>1800:
            raise ValueError("projected diagnostic exceeds thirty-minute aggregate budget")
        pinned[str(args.preflight.resolve())]=args.preflight_sha256
    for name in ("run_native_value_pilot.py","run_native_value_preflight.py","worker_resources.py"):
        path=Path(__file__).with_name(name).resolve(); pinned[str(path)]=sha256(path)
    args.output=args.output.resolve(); args.output.mkdir(exist_ok=False)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    started=time.monotonic()
    record=dict(schema="fixed-continuation-noise-controller-v1",status="running",boards=args.boards,turns=args.turns,
        pinnedInputs=pinned,runnerSha256=sha256(Path(__file__)),maximumWorkerSeconds=600,
        maximumWorkerMemoryBytes=2*1024**3,maximumConcurrentWorkers=min(2,args.boards),jobs=[],releaseAccepted=False)
    atomic_json(args.output/"manifest.json",record)
    def job(index):
        if stop.is_set() or any(sha256(Path(k))!=v for k,v in pinned.items()): raise ValueError("stopped or changed input")
        output=args.output/f"board{index}.json"
        worker=guarded(test_command(args.binary,TEST),dict(
            POKER_NOISE_PREFLOP=str(args.preflop),POKER_NOISE_PREFLOP_SHA=args.preflop_sha256,
            POKER_COMPACT_MODEL=str(args.model),POKER_COMPACT_MODEL_SHA=args.model_sha256,
            POKER_COMPACT_CHECKDOWN=str(args.kernel),POKER_COMPACT_CHECKDOWN_SHA=args.kernel_sha256,
            POKER_NOISE_BOARD_INDEX=str(index),POKER_NOISE_TURNS=str(args.turns),POKER_COMPACT_OUTPUT=str(output)),
            args.output/f"board{index}-worker",600,2*1024**3,stop)
        value=json.loads(output.read_text())
        if (value["boardIndex"]!=index or value["turnSampleCount"]!=args.turns
                or value["preflopSha256"]!=args.preflop_sha256 or value["modelSha256"]!=args.model_sha256
                or value["kernelSha256"]!=args.kernel_sha256): raise ValueError("wrong fixed-noise result")
        return value,dict(index=index,output=str(output),outputSha256=sha256(output),worker=worker)
    try:
        values=[]
        with ThreadPoolExecutor(max_workers=min(2,args.boards)) as pool:
            futures=[pool.submit(job,i) for i in range(args.boards)]
            try:
                for f in as_completed(futures):
                    value,report=f.result(); values.append(value); record["jobs"].append(report)
                    record["jobs"].sort(key=lambda v:v["index"]); atomic_json(args.output/"manifest.json",record)
                    print(json.dumps(dict(event="fixed-noise-board-complete",index=value["boardIndex"],seconds=value["seconds"])),flush=True)
            except Exception:
                stop.set(); raise
        if stop.is_set() or any(sha256(Path(k))!=v for k,v in pinned.items()): raise ValueError("stopped or changed inputs")
        record.update(status="complete",summary=summarize(sorted(values,key=lambda v:v["boardIndex"])))
    except (OSError,ValueError,KeyError) as e: record.update(status="failed",failure=str(e))
    record["elapsedSeconds"]=time.monotonic()-started
    atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","summary","failure")}),flush=True)
    if record["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()

"""All-turn learned baseline at an already measured frozen native policy. No training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
from run_compact_continuation_noise import TEST, summarize
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256


def corrected_observations(native, prediction):
    if (prediction["schema"]!="fixed-continuation-turn-predictions-v1" or prediction["releaseAccepted"] is not False
            or any(prediction[k]!=native[k] for k in ("boardIndex","board","policySha256",
                "preflopSha256","modelSha256","kernelSha256","classes","classReachWeights"))):
        raise ValueError("prediction/native frozen identity mismatch")
    rows=prediction["predictions"]
    if len(rows)!=49 or {r["turn"] for r in rows}!=set(range(52))-set(native["board"]):
        raise ValueError("exact prediction chance mean requires ALL 49 unique public turns")
    mass=np.asarray(prediction["classOpponentMass"],dtype=float)
    values=np.asarray([r["predictedRawCfvBb"] for r in rows],dtype=float)
    if (mass.shape!=(2,169) or values.shape!=(49,2,169) or (mass<0).any()
            or not np.isfinite(mass).all() or not np.isfinite(values).all()):
        raise ValueError("invalid prediction CFVs or opponent masses")
    mean=values.mean(axis=0)
    lookup={r["turn"]:values[i] for i,r in enumerate(rows)}
    corrected=copy.deepcopy(native)
    for r in corrected["records"]:
        raw=np.asarray(r["rawCfvBb"],dtype=float)
        baseline=np.asarray(r["sampledCheckdownCfvBb"],dtype=float)
        value=mean+raw-lookup[r["turn"]]-baseline
        if not np.isfinite(value).all() or (np.abs(value[mass==0])>1e-10).any():
            raise ValueError("invalid corrected CFV without opponent support")
        residual=np.divide(value,mass,out=np.zeros_like(value),where=mass>0)
        r["conditionalStrategicResidualBb"]=residual.tolist()
    return corrected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","preflop","model","kernel","native_manifest"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
        p.add_argument("--"+name.replace("_","-")+"-sha256",required=True)
    p.add_argument("--preflight",type=Path)
    p.add_argument("--preflight-sha256")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); pinned={}
    for name in ("binary","preflop","model","kernel","native_manifest"):
        path=getattr(args,name).resolve(); digest=getattr(args,name+"_sha256")
        if sha256(path)!=digest: raise ValueError("prediction diagnostic input changed")
        setattr(args,name,path); pinned[str(path)]=digest
    source=json.loads(args.native_manifest.read_text())
    if source["status"]!="complete" or len(source["jobs"])!=source["boards"]:
        raise ValueError("complete native diagnostic required")
    for name in ("preflop","model","kernel"):
        if source["pinnedInputs"].get(str(getattr(args,name)))!=getattr(args,name+"_sha256"):
            raise ValueError("native input provenance differs")
    if source["boards"]>1:
        if not args.preflight or sha256(args.preflight)!=args.preflight_sha256:
            raise ValueError("multiple boards require the pinned one-board prediction cost preflight")
        prior=json.loads(args.preflight.read_text())
        if (prior["status"]!="complete" or prior["boards"]!=1 or prior["elapsedSeconds"]*source["boards"]*1.5>900
                or any(prior["pinnedInputs"].get(str(getattr(args,n)))!=getattr(args,n+"_sha256")
                    for n in ("binary","preflop","model","kernel"))):
            raise ValueError("incompatible or unaffordable prediction preflight")
        pinned[str(args.preflight.resolve())]=args.preflight_sha256
    for job in source["jobs"]:
        if sha256(Path(job["output"]))!=job["outputSha256"]: raise ValueError("native observation changed")
        pinned[job["output"]]=job["outputSha256"]
    for name in ("run_compact_continuation_noise.py","run_native_value_pilot.py","run_native_value_preflight.py","worker_resources.py"):
        path=Path(__file__).with_name(name).resolve(); pinned[str(path)]=sha256(path)
    args.output=args.output.resolve(); args.output.mkdir(exist_ok=False)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    started=time.monotonic()
    record=dict(schema="fixed-continuation-prediction-controller-v1",status="running",boards=source["boards"],
        pinnedInputs=pinned,runnerSha256=sha256(Path(__file__)),jobs=[],maximumConcurrentWorkers=min(2,source["boards"]),
        maximumWorkerSeconds=180,maximumWorkerMemoryBytes=2*1024**3,releaseAccepted=False)
    atomic_json(args.output/"manifest.json",record)
    def job(item):
        index=item["index"]
        if stop.is_set() or any(sha256(Path(k))!=v for k,v in pinned.items()): raise ValueError("stopped or changed input")
        output=args.output/f"board{index}.json"
        worker=guarded(test_command(args.binary,TEST),dict(POKER_NOISE_PREFLOP=str(args.preflop),
            POKER_NOISE_PREFLOP_SHA=args.preflop_sha256,POKER_COMPACT_MODEL=str(args.model),
            POKER_COMPACT_MODEL_SHA=args.model_sha256,POKER_COMPACT_CHECKDOWN=str(args.kernel),
            POKER_COMPACT_CHECKDOWN_SHA=args.kernel_sha256,POKER_NOISE_BOARD_INDEX=str(index),
            POKER_NOISE_TURNS=str(source["turns"]),POKER_COMPACT_OUTPUT=str(output),
            POKER_NOISE_NATIVE_SOURCE=item["output"],POKER_NOISE_NATIVE_SOURCE_SHA=item["outputSha256"]),
            args.output/f"board{index}-worker",180,2*1024**3,stop)
        native=json.loads(Path(item["output"]).read_text()); prediction=json.loads(output.read_text())
        if prediction["nativeSourceSha256"]!=item["outputSha256"]: raise ValueError("wrong native observation identity")
        derived=corrected_observations(native,prediction)
        return native,derived,dict(index=index,output=str(output),outputSha256=sha256(output),
            predictionSeconds=prediction["predictionSeconds"],worker=worker)
    try:
        originals=[]; corrected=[]
        with ThreadPoolExecutor(max_workers=min(2,source["boards"])) as pool:
            futures=[pool.submit(job,item) for item in source["jobs"]]
            try:
                for f in as_completed(futures):
                    native,derived,report=f.result(); originals.append(native); corrected.append(derived); record["jobs"].append(report)
                    record["jobs"].sort(key=lambda r:r["index"]); atomic_json(args.output/"manifest.json",record)
                    print(json.dumps(dict(event="fixed-prediction-complete",**report)),flush=True)
            except Exception: stop.set(); raise
        if stop.is_set() or any(sha256(Path(k))!=v for k,v in pinned.items()): raise ValueError("stopped or changed inputs")
        record.update(status="complete",native=summarize(originals),corrected=summarize(corrected),
            interpretation="Same frozen flop policy and observed native values. Corrected estimator uses the complete 49-turn prediction mean plus native-minus-predicted sampled residual. Not replacement native labels or release qualification.")
    except (OSError,ValueError,KeyError) as e: record.update(status="failed",failure=str(e))
    record["elapsedSeconds"]=time.monotonic()-started; atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","native","corrected","failure")}),flush=True)
    if record["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()

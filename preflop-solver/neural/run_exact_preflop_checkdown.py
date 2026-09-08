"""Bounded exact class-payoff kernel: 16-orbit preflight, then all 1,755 orbits.

No neural fitting or policy is produced. Integer reduction is deterministic;
timings/provenance remain in a separate manifest. Completed chunks are resumable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import math
from pathlib import Path
import shutil
import signal
import threading
import time

from cloud_blueprint_run import CANONICAL_HAND_CLASSES, combo_weight
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256, RESERVE_BYTES

TEST="blueprint::preflop_continuation::exact_checkdown::exact_preflop_checkdown_chunk"
PREFLIGHT=[round(i*1754/15) for i in range(16)]


def read_chunk(path):
    if path.stat().st_size>4*1024**2: raise ValueError("oversized exact chunk")
    with gzip.open(path,"rb") as stream: raw=stream.read(4*1024**2+1)
    if len(raw)>4*1024**2: raise ValueError("oversized decoded exact chunk")
    v=json.loads(raw)
    if (v["schema"]!="exact-preflop-checkdown-orbit-chunk-v1" or v["complete"] is not False
            or v["releaseAccepted"] is not False or v["classes"]!=sorted(CANONICAL_HAND_CLASSES)
            or v["classMultiplicities"]!=[combo_weight(c) for c in v["classes"]]
            or not 1<=len(v["orbits"])<=16):
        raise ValueError("invalid exact chunk contract")
    indices=[r["index"] for r in v["orbits"]]
    if indices!=sorted(set(indices)) or not all(type(i) is int and 0<=i<1755 for i in indices):
        raise ValueError("invalid orbit indices")
    for name in ("compatiblePairs","weightedWinUnits","weightedFlopPairs"):
        if len(v[name])!=169**2 or any(type(x) is not int or x<0 for x in v[name]):
            raise ValueError("invalid integer class counts")
    for a in range(169):
        for b in range(169):
            k,j=a*169+b,b*169+a
            if (v["weightedFlopPairs"][k]!=v["weightedFlopPairs"][j]
                    or v["weightedWinUnits"][k]+v["weightedWinUnits"][j]!=1980*v["weightedFlopPairs"][k]):
                raise ValueError("chunk lost zero-sum integer accounting")
    for r in v["orbits"]:
        if (len(set(r["board"]))!=3 or not all(type(c) is int and 0<=c<52 for c in r["board"])
                or r["orbitSize"] not in (4,12,24) or not math.isfinite(r["seconds"]) or r["seconds"]<=0):
            raise ValueError("invalid orbit metadata")
    return v


def merge(chunks):
    if not chunks: raise ValueError("no exact chunks")
    first=chunks[0]
    wins,pairs=[0]*169**2,[0]*169**2
    indices=set()
    raw_flops=0
    for chunk in chunks:
        for name in ("classes","classMultiplicities","compatiblePairs"):
            if chunk[name]!=first[name]: raise ValueError("incompatible exact chunks")
        for r in chunk["orbits"]:
            if r["index"] in indices: raise ValueError("duplicate exact orbit")
            indices.add(r["index"])
            raw_flops+=r["orbitSize"]
        for i in range(169**2):
            wins[i]+=chunk["weightedWinUnits"][i]
            pairs[i]+=chunk["weightedFlopPairs"][i]
    if indices!=set(range(1755)) or raw_flops!=22100:
        raise ValueError("partial chance coverage is not an exact payoff kernel")
    for a in range(169):
        for b in range(169):
            k,j=a*169+b,b*169+a
            if pairs[k]!=first["compatiblePairs"][k]*17296 or wins[k]+wins[j]!=1980*pairs[k]:
                raise ValueError("complete kernel card-removal or symmetry failure")
    return dict(schema="exact-preflop-checkdown-class-kernel-v1", complete=True,
        classes=first["classes"],classMultiplicities=first["classMultiplicities"],
        compatiblePairs=first["compatiblePairs"],weightedFlopPairs=pairs,weightedWinUnits=wins,
        canonicalFlops=1755,rawFlops=22100)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--binary-sha256",required=True)
    parser.add_argument("--mode",choices=["preflight","complete"],default="preflight")
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,choices=[1,2],default=2)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    args.binary,args.output=args.binary.resolve(),args.output.resolve()
    if sha256(args.binary)!=args.binary_sha256: raise ValueError("binary changed")
    pinned={str(args.binary):args.binary_sha256}
    for name in ("run_native_value_pilot.py","run_native_value_preflight.py","worker_resources.py","cloud_blueprint_run.py"):
        path=Path(__file__).with_name(name).resolve()
        pinned[str(path)]=sha256(path)
    inherited=[]
    if args.mode=="complete":
        if not args.preflight or sha256(args.preflight)!=args.preflight_sha256:
            raise ValueError("complete kernel requires pinned cost preflight")
        prior=json.loads(args.preflight.read_text())
        if (prior["status"]!="complete" or prior["mode"]!="preflight"
                or prior["binarySha256"]!=args.binary_sha256 or len(prior["chunks"])!=1):
            raise ValueError("incompatible cost preflight")
        inherited=prior["chunks"]
        pinned[str(args.preflight.resolve())]=args.preflight_sha256
        projected=prior["elapsedSeconds"]*1755/16/args.workers*1.5
        if projected>3600: raise ValueError("projected complete pass exceeds one-hour guard")
        budget=Path(inherited[0]["path"]).stat().st_size*math.ceil(1755/16)*3
        if shutil.disk_usage(args.output.parent).free<RESERVE_BYTES+budget:
            raise ValueError("insufficient projected disk headroom")
    if args.resume:
        record=json.loads((args.output/"manifest.json").read_text())
        if (record["pinnedInputs"]!=pinned or record["runnerSha256"]!=sha256(Path(__file__))
                or record["mode"]!=args.mode or record["status"]=="complete"):
            raise ValueError("cannot resume changed or completed exact pass")
        record["status"]="running"
        record.pop("failure",None)
    else:
        args.output.mkdir(exist_ok=False)
        record=dict(schema="exact-preflop-checkdown-controller-v1",status="running",mode=args.mode,
            binarySha256=args.binary_sha256,pinnedInputs=pinned,runnerSha256=sha256(Path(__file__)),
            chunks=inherited,maximumWorkers=args.workers,maximumSeconds=3600,
            releaseAccepted=False,interpretation=__doc__)
    done=set()
    for row in record["chunks"]:
        path=Path(row["path"])
        if sha256(path)!=row["sha256"]: raise ValueError("completed exact chunk changed")
        for r in read_chunk(path)["orbits"]:
            if r["index"] in done: raise ValueError("overlapping retained chunks")
            done.add(r["index"])
    wanted=PREFLIGHT if args.mode=="preflight" else list(range(1755))
    remaining=[i for i in wanted if i not in done]
    jobs=[remaining[i:i+16] for i in range(0,len(remaining),16)]
    started=time.monotonic()
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    timer=threading.Timer(3600,stop.set); timer.daemon=True; timer.start()
    atomic_json(args.output/"manifest.json",record)
    def job(indices):
        if stop.is_set(): raise ValueError("exact pass stopped")
        name="orbits-%04d"%indices[0]
        attempt=1
        while (args.output/(name+"-attempt%d"%attempt)).exists(): attempt+=1
        stem=name+"-attempt%d"%attempt
        path=args.output/(stem+".json.gz")
        worker=guarded(test_command(args.binary,TEST),{"POKER_EXACT_ORBITS":json.dumps(indices),
            "POKER_EXACT_OUTPUT":str(path)},args.output/stem,90,2*1024**3,stop)
        value=read_chunk(path)
        if [r["index"] for r in value["orbits"]]!=indices: raise ValueError("wrong orbit chunk")
        return dict(path=str(path),sha256=sha256(path),indices=indices,worker=worker)
    try:
        with ThreadPoolExecutor(max_workers=1 if args.mode=="preflight" else args.workers) as pool:
            futures=[pool.submit(job,indices) for indices in jobs]
            for future in as_completed(futures):
                try:
                    row=future.result()
                except Exception:
                    stop.set()
                    for pending in futures: pending.cancel()
                    raise
                record["chunks"].append(row)
                if any(sha256(Path(p))!=h for p,h in pinned.items()):
                    stop.set(); raise ValueError("pinned exact input changed")
                atomic_json(args.output/"manifest.json",record)
                print(json.dumps(dict(event="exact-orbits-complete",orbits=sum(len(r["indices"]) for r in record["chunks"]))),flush=True)
        if stop.is_set(): raise ValueError("exact pass stopped")
        if args.mode=="complete":
            kernel=merge([read_chunk(Path(r["path"])) for r in record["chunks"]])
            path=args.output/"kernel.json.gz"
            encoded=gzip.compress(json.dumps(kernel,separators=(",",":"),sort_keys=True).encode(),mtime=0)
            with path.open("xb") as stream: stream.write(encoded)
            record.update(kernel=str(path),kernelSha256=sha256(path),kernelBytes=len(encoded))
        record["status"]="complete"
    except (OSError,ValueError,KeyError) as error:
        stop.set(); record.update(status="failed",failure=str(error))
    finally:
        timer.cancel()
    record["elapsedSeconds"]=time.monotonic()-started
    atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","kernelSha256","kernelBytes","failure")}),flush=True)
    if record["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()

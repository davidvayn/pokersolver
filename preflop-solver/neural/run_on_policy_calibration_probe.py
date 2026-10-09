"""Small TRAIN-only induced-belief comparison; no fitting or activation."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from run_native_value_pilot import PREDICT_TEST, guarded, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from run_search_distribution_pilot import TEST
from run_student_value_pilot import select_students
from training_coverage import extra_calibration, extension_families


def validate_capture(source, family, model_sha, policy_sha):
    if (source.get("source_public_input_sha256") != family["rootSha256"]
            or source.get("source_policy_sha256") != policy_sha
            or source.get("proposal_model_sha256") != model_sha
            or source.get("seed") != family["solverSeed"]
            or source.get("sampling_seed") != family["sampleSeed"]
            or source.get("proposal_turn_iterations") != 64
            or source.get("native_label_queries") != 16
            or source.get("policy_observation_parity_checked") is not True):
        raise ValueError("induced capture root/proposal/seed/parity changed")
    extra_calibration([], [dict(board=family["family"])], source, policy_sha)


def shifted_enough(results):
    if len(results) != 3 or sorted(r["root"] for r in results) != [100,101,102]:
        raise ValueError("all three predeclared TRAIN family comparisons required")
    for row in results:
        values = [row[key]["authenticRmseBb"] for key in ("old", "current")]
        if not np.isfinite(values).all() or min(values) < 0: raise ValueError("invalid shift metrics")
    return sum(r["current"]["authenticRmseBb"] >= 1.1*r["old"]["authenticRmseBb"]
        and r["current"]["authenticRmseBb"] > r["old"]["authenticRmseBb"] for r in results) >= 2


def measure_values(binary, capture, model, work, cache, stop):
    # Analysis uses NumPy plus the actual native predictor, not a surrogate
    # framework forward. Import feature preparation only after labeling.
    import train_public_value_network as training
    from validate_public_value_parity import python_prediction
    data = training.load_dataset(capture, 1, "payoff-exposure")
    contexts, queries, _ = training.feature_dataset_cached(data, training.FEATURE_SCHEMA_EXACT_RUNOUT, 1, cache)
    predictions = work/"predictions.json"
    worker = guarded(test_command(binary, PREDICT_TEST), dict(
        POKER_NATIVE_PREDICT_DATASET=str(capture), POKER_NATIVE_PREDICT_DATASET_SHA=sha256(capture),
        POKER_NATIVE_PREDICT_MODEL=model["path"], POKER_NATIVE_PREDICT_MODEL_SHA=model["sha256"],
        POKER_NATIVE_PREDICT_OUTPUT=str(predictions)), work/"prediction-worker",300,2*1024**3,stop)
    values = np.asarray(json.loads(predictions.read_text())["predictions"])
    if values.shape != (16,2,1326) or not np.isfinite(values).all(): raise ValueError("invalid native predictions")
    payload = json.loads(Path(model["path"]).read_text())
    parity = max(float(abs(python_prediction(data,payload,i,(contexts[i],queries[i]))-values[i]).max()) for i in range(16))
    if parity > .0001 or not np.isfinite(parity): raise ValueError("new-state NumPy/native parity failed")
    truth = data.targets.reshape((16,2,1326))*data.target_scales[:,None,None]
    weights = data.projection_weights
    total = weights.sum(axis=(1,2)); reached = total > 1e-9
    if not reached.any(): raise ValueError("capture has no authentic joint support")
    mse = np.divide((weights*(truth-values)**2).sum(axis=(1,2)),total,
        out=np.zeros(16),where=reached)
    bands = [r["state_distribution"] for r in data.source["targets"]]
    native_gaps = [v for row in data.source["targets"] for v in (row.get("conditional_response_gain_bb") or [])]
    return dict(authenticRmseBb=float(np.sqrt(mse[reached].mean())),reachedStates=int(reached.sum()),
        bandRmseBb={band:float(np.sqrt(mse[[i for i in range(16) if reached[i] and bands[i]==band]].mean()))
            for band in sorted(set(bands)) if any(reached[i] and bands[i]==band for i in range(16))},
        meanNativeLabelResponseGainBb=float(np.mean(native_gaps)) if native_gaps else None,
        maximumNativeLabelResponseGainBb=max(native_gaps) if native_gaps else None,
        maximumParityErrorBb=parity,predictionsSha256=sha256(predictions),worker=worker)


def measure(binary, capture, model, work, cache, stop):
    # Feature preparation is the largest analysis allocation. Keep it in an
    # independently guarded child rather than in the unbounded controller.
    request=work/"request.json"
    result=work/"result.json"
    atomic_json(request,dict(binary=str(binary),binarySha256=sha256(binary),
        capture=str(capture),captureSha256=sha256(capture),model=model,
        work=str(work),cache=str(cache),result=str(result)))
    worker=guarded([sys.executable,str(Path(__file__).resolve()),"--measure-request",str(request)],
        {},work/"analysis-worker",600,6*1024**3,stop)
    values=json.loads(result.read_text())
    return {**values,"analysisWorker":worker}


def measurement_worker():
    p=argparse.ArgumentParser(description="Guarded induced-belief analysis worker")
    p.add_argument("--measure-request",type=Path,required=True)
    request=json.loads(p.parse_args().measure_request.read_text())
    for name in ("binary","capture"):
        if sha256(Path(request[name]))!=request[name+"Sha256"]:
            raise ValueError("analysis input changed")
    if sha256(Path(request["model"]["path"]))!=request["model"]["sha256"]:
        raise ValueError("analysis model changed")
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    values=measure_values(Path(request["binary"]),Path(request["capture"]),request["model"],
        Path(request["work"]),Path(request["cache"]),stop)
    if stop.is_set(): raise ValueError("analysis stopped")
    atomic_json(Path(request["result"]),values)


def main():
    if "--measure-request" in sys.argv[1:]:
        measurement_worker()
        return
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","bundles","students"):
        p.add_argument("--"+name,type=Path,required=True); p.add_argument("--"+name+"-sha256",required=True)
    p.add_argument("--feature-cache",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); pinned={}
    for name in ("binary","bundles","students"):
        path=getattr(a,name).resolve(); setattr(a,name,path)
        digest=getattr(a,name+"_sha256")
        if sha256(path)!=digest: raise ValueError("pinned induced-probe input changed")
        pinned[str(path)]=digest
    bundles=json.loads(a.bundles.read_text()); _,models=select_students(a.students,a.students_sha256)
    if bundles.get("status")!="complete" or sorted(f["root"] for f in bundles["families"])!=[2,3,4,100,101,102]:
        raise ValueError("frozen six-family TRAIN bundle set required")
    registry=bundles["trainingExtension"]; plan=json.loads(Path(registry["path"]).read_text())
    old={tuple(f["family"]) for f in bundles["families"] if f["root"]<100}
    forbidden={tuple(f) for f in plan["excludedFamilies"]}-old
    extension_families(registry,plan["corpusSha256"],plan["splitReferenceSha256"],old,forbidden)
    pinned[registry["path"]]=registry["sha256"]; pinned.update(plan["pinnedInputs"])
    families=[f for f in bundles["families"] if f["root"]>=100]
    registered={f["root"]:f for f in plan["families"]}
    for family in families:
        rule=registered[family["root"]]
        if any(family.get(k)!=rule.get(k) for k in ("family","rootSha256","solverSeed","sampleSeed")):
            raise ValueError("family differs from frozen TRAIN registry")
        for path,digest in ((rule["rootPath"],rule["rootSha256"]),
                (family["candidatePath"],family["candidateSha256"]),
                (family["extraCapture"]["path"],family["extraCapture"]["sha256"])):
            pinned[path]=digest
        source=read_capture(Path(family["extraCapture"]["path"]))
        # The old collection only checked observation parity on its first
        # family; all complete source identities/budgets are still verified.
        if (source["source_public_input_sha256"]!=rule["rootSha256"]
                or source["source_policy_sha256"]!=family["candidateSha256"]
                or source["proposal_model_sha256"]!=rule["modelSha256"]
                or source["seed"]!=rule["solverSeed"] or source["sampling_seed"]!=rule["sampleSeed"]):
            raise ValueError("old matched capture provenance changed")
        extra_calibration([], [dict(board=family["family"])],source,family["candidateSha256"])
        model=models[rule["solverSeed"]]; pinned[model["path"]]=model["sha256"]
    for name in (Path(__file__).name,"training_coverage.py","run_native_value_pilot.py","worker_resources.py",
            "run_native_value_preflight.py","run_postflop_gap_pilot.py","run_search_distribution_pilot.py",
            "run_student_value_pilot.py","action_contrast_dataset.py","native_replay.py",
            "train_public_value_network.py","validate_public_value_parity.py","native_value_dataset.py"):
        path=Path(__file__).with_name(name); pinned[str(path)]=sha256(path)
    if any(sha256(Path(path))!=digest for path,digest in pinned.items()): raise ValueError("probe dependency changed")
    a.output=a.output.resolve(); a.feature_cache=a.feature_cache.resolve()
    if a.output.exists(): raise ValueError("never overwrite induced-belief probe")
    a.output.mkdir(); stop=threading.Event(); started=time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    timer=threading.Timer(1800,stop.set); timer.daemon=True; timer.start()
    pressure=PilotMemoryGuard(stop,a.output/"system-memory.json").start()
    record=dict(schema="on-policy-calibration-shift-probe-v1",status="running",pinnedInputs=pinned,
        releaseAccepted=False,maximumSeconds=1800,families=[],newNativeStates=48,maximumNativeWorkers=2)
    atomic_json(a.output/"manifest.json",record)
    try:
        for index,family in enumerate(families):
            rule=registered[family["root"]]; model=models[rule["solverSeed"]]
            work=a.output/f"root-{family['root']}"; work.mkdir()
            capture,candidate=work/"current.json.gz",work/"current-proposer.json"
            worker=guarded(test_command(a.binary,TEST),dict(
                POKER_SEARCH_ROOT=rule["rootPath"],POKER_SEARCH_ROOT_SHA=rule["rootSha256"],
                POKER_SEARCH_MODEL=model["path"],POKER_SEARCH_MODEL_SHA=model["sha256"],
                POKER_SEARCH_OUTPUT=str(capture),POKER_SEARCH_POLICY_OUTPUT=str(candidate),
                POKER_SEARCH_TRUNK_SEED=str(rule["solverSeed"]),POKER_SEARCH_SAMPLE_SEED=str(rule["sampleSeed"]),
                POKER_SEARCH_LABEL_COUNT="16",POKER_SEARCH_WORKERS="2",POKER_SEARCH_FLOP_ITERATIONS="128",
                POKER_SEARCH_LABEL_TURN_ITERATIONS="64",POKER_SEARCH_VERIFY_POLICY_PARITY="1"),
                work/"capture-worker",600,2*1024**3,stop)
            validate_capture(read_capture(capture),rule,model["sha256"],sha256(candidate))
            row=dict(root=family["root"],solverSeed=rule["solverSeed"],modelSha256=model["sha256"],
                capture=dict(path=str(capture),sha256=sha256(capture)),candidateSha256=sha256(candidate),worker=worker)
            for name,path in (("old",Path(family["extraCapture"]["path"])),("current",capture)):
                destination=work/name; destination.mkdir()
                row[name]=measure(a.binary,path,model,destination,a.feature_cache,stop)
            record["families"].append(row); atomic_json(a.output/"manifest.json",record)
            print(json.dumps(dict(event="induced-belief-measured",root=family["root"],
                oldRmseBb=row["old"]["authenticRmseBb"],currentRmseBb=row["current"]["authenticRmseBb"])),flush=True)
            if index==0 and 1.5*(time.monotonic()-started)*2 > 1800-(time.monotonic()-started):
                raise ValueError("first-family projection exceeds remaining stage cap")
        if stop.is_set() or any(sha256(Path(path))!=digest for path,digest in pinned.items()):
            raise ValueError("probe stopped or input changed")
        record["datasetAggregationPilotSupported"]=shifted_enough(record["families"])
        record["status"]="complete"
    except Exception as error:
        record["status"]="failed"; record["failure"]=str(error); stop.set(); raise
    finally:
        timer.cancel(); record["elapsedSeconds"]=time.monotonic()-started
        record["systemMemoryGuard"]=pressure.finish(); atomic_json(a.output/"manifest.json",record)


if __name__=="__main__": main()

"""Inspect frozen TRAIN contrast allocation; no fitting or new native labels."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import run_action_contrast_students as base
import train_public_value_network as training
import native_value_dataset as native
from action_contrast_dataset import contrast_loss_and_q_gradient, leaf_gradient
from action_contrast_loss import gradient_add
from decision_gradient_allocation import contrast_allocation, margin_loss_and_q_gradient, ranking_pilot_supported
from retained_initialization import import_retained_weights
from run_native_value_pilot import guarded
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from serving_value_projection import BoundedValueProjection
from training_gradient_conflict import gradient_relationship


def parameter_vjp(model,bundle,derivative):
    total=None
    for start in range(0,len(derivative),4):
        end=min(start+4,len(derivative))
        inputs=base.BundleObjective.inputs(bundle,start,end)
        upstream=mx.array((derivative[start:end]*bundle.dataset.target_scales[start:end,None,None]).reshape(end-start,-1).astype(np.float32))
        def linear(current): return mx.sum(current.raw_values(*inputs).reshape(end-start,-1)*upstream)
        _,gradient=nn.value_and_grad(model,linear)(model)
        total=gradient if total is None else gradient_add(total,gradient)
        mx.eval(total)
    return total


def worker(args,checkpoints):
    if os.environ.get("MLX_ENABLE_TF32")!="0":raise ValueError("full-float32 worker required")
    dataset,contexts,queries,split,bundles,_=base.prepare(args)
    del dataset,contexts,queries,split;gc.collect()
    rows=[]
    for seed in (10601,10602):
        for step in (0,600):
            receipt=checkpoints[str(seed)][str(step)]
            if sha256(Path(receipt["path"]))!=receipt["sha256"]:raise ValueError("frozen checkpoint changed")
            model=training.SharedComboValueNetwork(True,"wide","payoff-exposure",training.FEATURE_SCHEMA_EXACT_RUNOUT)
            import_retained_weights(model,json.loads(Path(receipt["path"]).read_text()),seed)
            reports=[]; original_gradient=margin_gradient=None
            for bundle in bundles:
                data=bundle.dataset
                predictions=[]
                for start in range(0,len(data.targets),4):
                    end=min(start+4,len(data.targets))
                    predictions.append(np.asarray(model.raw_values(*base.BundleObjective.inputs(bundle,start,end))).reshape(-1,2,1326)*data.target_scales[start:end,None,None])
                raw=np.concatenate(predictions)
                weights=np.asarray([r*native.compatible_masses(r) for r in data.ranges])
                legal=np.asarray([native.legal_combos(b) for b in data.boards])
                projection=BoundedValueProjection(raw,weights,legal)
                original=np.zeros_like(raw,dtype=np.float64);alternative=np.zeros_like(original)
                for group in bundle.groups:
                    q=group.backup(projection.values[:,group.actor]);reach=group.weights*group.support
                    pieces,report=contrast_allocation(q,group.target,reach)
                    loss,derivative=contrast_loss_and_q_gradient(q,group.target,reach)
                    error=float(abs(sum(pieces.values())-derivative).max())
                    if not np.isfinite(error) or error>1e-12:raise ValueError("pair decomposition changed the real objective")
                    margin_loss,margin_derivative=margin_loss_and_q_gradient(q,group.target,reach)
                    original[:,group.actor]+=leaf_gradient(group,derivative)/len(bundle.groups)
                    alternative[:,group.actor]+=leaf_gradient(group,margin_derivative)/len(bundle.groups)
                    reports.append(dict(family=list(bundle.family),history=group.history,actor=group.actor,
                        contrastLoss=loss,marginLoss=margin_loss,maximumDerivativeReconstructionError=error,**report))
                first=parameter_vjp(model,bundle,projection.vjp(original))
                second=parameter_vjp(model,bundle,projection.vjp(alternative))
                zero=gradient_add(first,first,-1.)
                original_gradient=gradient_add(zero if original_gradient is None else original_gradient,first,1/len(bundles))
                margin_gradient=gradient_add(zero if margin_gradient is None else margin_gradient,second,1/len(bundles))
            if len(reports)!=12:raise ValueError("six complete TRAIN families/two nodes required")
            row=dict(seed=seed,step=step,decisions=reports,
                meanNativeRankingLossBb=float(np.mean([r["nativeRankingLossBb"] for r in reports])),
                meanCorrectDerivativeShare=float(np.mean([r["pairDerivativeShares"]["correct"] for r in reports])),
                parameterGradientRelationship=gradient_relationship(original_gradient,margin_gradient))
            rows.append(row);atomic_json(args.output/"analysis.json",dict(status="running",results=rows,releaseAccepted=False))
            print(json.dumps(dict(event="decision-gradient-allocation",seed=seed,step=step,
                nativeRankingLossBb=row["meanNativeRankingLossBb"],correctDerivativeShare=row["meanCorrectDerivativeShare"],
                gradientCosine=row["parameterGradientRelationship"]["cosine"])),flush=True)
            del model;gc.collect();mx.clear_cache()
    atomic_json(args.output/"analysis.json",dict(status="complete",results=rows,
        rankingAuxiliaryPilotSupported=ranking_pilot_supported(rows),releaseAccepted=False,
        interpretation="Frozen TRAIN profile rankings and raw parameter-gradient geometry. No fitting, checkpoint selection, new labels, response gains or equilibrium guarantee."))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gradient-probe",type=Path,required=True);p.add_argument("--gradient-probe-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--worker",action="store_true")
    args=p.parse_args();args.gradient_probe=args.gradient_probe.resolve();args.output=args.output.resolve()
    if sha256(args.gradient_probe)!=args.gradient_probe_sha256:raise ValueError("frozen input receipt changed")
    previous=json.loads(args.gradient_probe.read_text())
    if previous.get("status")!="complete" or previous.get("schema")!="training-gradient-conflict-probe-v1":
        raise ValueError("completed TRAIN checkpoint/input receipt required")
    pinned=dict(previous["pinnedInputs"]);pinned[str(args.gradient_probe)]=args.gradient_probe_sha256
    controls=[Path(path) for path in pinned if path.endswith("local-serving-contrast-students-20261004-a/manifest.json")]
    if len(controls)!=1:raise ValueError("exact aligned-model control required")
    control=json.loads(controls[0].read_text());command=control["fitWorker"]["command"]
    if control.get("status")!="complete" or control.get("armDifference")!="serving_contrast_forward_vjp_only":
        raise ValueError("frozen aligned pair required")
    for name in ("corpus","split-reference","bundles","feature-cache"):
        path=Path(command[command.index("--"+name)+1]).resolve();setattr(args,name.replace("-","_"),path)
        if name!="feature-cache":setattr(args,name.replace("-","_")+"_sha256",command[command.index("--"+name+"-sha256")+1])
    args.feature_workers=1
    for name in (Path(__file__).name,"decision_gradient_allocation.py","run_native_value_preflight.py","run_postflop_gap_pilot.py"):
        path=Path(__file__).with_name(name);pinned[str(path)]=sha256(path)
    if any(sha256(Path(path))!=digest for path,digest in pinned.items()):raise ValueError("diagnostic dependency changed")
    if args.worker:worker(args,previous["checkpoints"]);return
    if args.output.exists():raise ValueError("never overwrite frozen allocation probe")
    args.output.mkdir();stop=threading.Event();started=time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    pressure=PilotMemoryGuard(stop,args.output/"system-memory.json").start()
    record=dict(schema="decision-gradient-allocation-probe-v1",status="running",pinnedInputs=pinned,releaseAccepted=False)
    atomic_json(args.output/"manifest.json",record)
    try:
        record["worker"]=guarded([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
            {"MLX_ENABLE_TF32":"0"},args.output/"worker",1200,6*1024**3,stop)
        if stop.is_set() or any(sha256(Path(path))!=digest for path,digest in pinned.items()):raise ValueError("probe stopped or source changed")
        record["analysisSha256"]=sha256(args.output/"analysis.json");record["status"]="complete"
    except Exception as error:
        record["status"]="failed";record["failure"]=str(error);stop.set();raise
    finally:
        record["elapsedSeconds"]=time.monotonic()-started;record["systemMemoryGuard"]=pressure.finish()
        atomic_json(args.output/"manifest.json",record)


if __name__=="__main__":main()

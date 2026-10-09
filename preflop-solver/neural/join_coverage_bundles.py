"""Join old full49 TRAIN data with exactly three registered coverage shards."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from action_contrast_dataset import build_groups
from complete_action_bundle_chance import verified_json
from join_full_chance_bundles import data_decision
from run_native_value_pilot import read_capture
from run_native_value_preflight import atomic_json, sha256
from training_coverage import extension_families, extra_calibration


def join(args):
    old_receipt=dict(path=str(args.baseline.resolve()),sha256=args.baseline_sha256)
    new_receipt=dict(path=str(args.coverage.resolve()),sha256=args.coverage_sha256)
    old,new=verified_json(old_receipt),verified_json(new_receipt)
    if (old.get("schema")!="full-chance-action-bundle-set-v1" or new.get("schema")!="training-coverage-action-bundle-pilot-v1"
            or any(m.get("status")!="complete" or m.get("releaseAccepted") is not False for m in (old,new))
            or sorted(f["root"] for f in old["families"])!=[2,3,4]
            or sorted(f["root"] for f in new["families"])!=[100,101,102]):
        raise ValueError("only the frozen three old plus three registered TRAIN shards may join")
    registry=verified_json(new["trainingExtension"])
    existing={tuple(f["family"]) for f in old["families"]}
    forbidden={tuple(f) for f in registry["excludedFamilies"]}-existing
    extension_families(new["trainingExtension"],registry["corpusSha256"],registry["splitReferenceSha256"],existing,forbidden)
    registered={f["root"]:f for f in registry["families"]}
    if any(tuple(f["family"])!=tuple(registered[f["root"]]["family"]) or f.get("role")!="TRAIN"
           or f.get("rootSha256")!=registered[f["root"]]["rootSha256"] for f in new["families"]):
        raise ValueError("coverage families differ from the scores-blind registry")
    qualities=[]; families=copy.deepcopy(old["families"]+new["families"])
    for family in families:
        if family.get("all49Turns") is not True or len(family["labels"])!=49: raise ValueError("incomplete chance coverage")
        prefix=verified_json(family["prefix"]); packets=[verified_json(r) for r in family["labels"]]
        if any(p["turn_iterations"]!=64 for p in packets): raise ValueError("teacher budget changed")
        groups,ordered=build_groups(prefix,packets,require_full_chance=True)
        if family.get("extraCapture"):
            receipt=family["extraCapture"]; path=Path(receipt["path"])
            if sha256(path)!=receipt["sha256"]: raise ValueError("search capture changed")
            groups,ordered=extra_calibration(groups,ordered,read_capture(path),family["candidateSha256"])
        receipt=family["calibrationCorpus"]; path=Path(receipt["path"])
        if sha256(path)!=receipt["sha256"] or read_capture(path)["targets"]!=ordered:
            raise ValueError("calibration/affine ordering changed")
        if not family.get("quality"):
            source=next(verified_json(r) for r in old["references"] if verified_json(r)["families"][0]["root"]==family["root"])
            family["quality"]=source["quality"]
        quality=verified_json(family["quality"])
        if quality["root"]!=family["root"]: raise ValueError("quality family differs")
        qualities.append(quality)
    if len({tuple(f["family"]) for f in families})!=6: raise ValueError("duplicate TRAIN family")
    decision=data_decision(qualities,"0"*64,expected_roots=[2,3,4,100,101,102])
    if args.output.exists(): raise ValueError("never overwrite a joined TRAIN extension")
    args.output=args.output.resolve(); args.output.mkdir()
    manifest=dict(schema="full-chance-coverage-bundle-set-v1",status="complete",releaseAccepted=False,
        calibrationOnly=False,all49Turns=True,nativeBudget=64,trainingExtension=new["trainingExtension"],
        references=[old_receipt,new_receipt],families=sorted(families,key=lambda f:f["root"]))
    path=args.output/"manifest.json"; atomic_json(path,manifest)
    decision["bundleManifestSha256"]=decision["primaryTrainingManifestSha256"]=sha256(path)
    atomic_json(args.output/"decision.json",decision)
    print(json.dumps(dict(manifestSha256=sha256(path),decisionSha256=sha256(args.output/"decision.json"),decision=decision)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","coverage"):
        p.add_argument("--"+name,type=Path,required=True); p.add_argument("--"+name+"-sha256",required=True)
    p.add_argument("--output",type=Path,required=True); join(p.parse_args())

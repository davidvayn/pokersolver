"""Attribute an unchanged sampled update's training/played-policy target gap."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_native_value_preflight import atomic_json, sha256


def inspect_tick(d):
    training = np.asarray(d["trainingRootActionValuesBb"], dtype=float)
    profile = np.asarray(d["profileRootActionValuesBb"], dtype=float)
    mix = np.asarray(d["rootProbabilities"], dtype=float)
    weights = np.asarray(d["multiplicities"], dtype=float)/1326
    if (training.shape != profile.shape or mix.shape != profile.shape
            or mix.shape != (len(d["actions"]), len(d["classes"]))
            or not np.isfinite([training, profile, mix]).all()
            or (mix < 0).any() or np.max(np.abs(mix.sum(axis=0)-1)) > 1e-12
            or abs(weights.sum()-1) > 1e-12):
        raise ValueError("invalid target diagnostic")
    train_value = (training*mix).sum(axis=0)
    profile_value = (profile*mix).sum(axis=0)
    a = training-train_value
    b = profile-profile_value
    gap = a-b
    changed = training.argmax(axis=0) != profile.argmax(axis=0)
    spurious = (a > 1e-8) & (b <= 0)
    examples = []
    for action, hand in zip(*np.where(spurious)):
        examples.append(dict(hand=d["classes"][hand], action=d["actions"][action],
            trainingAdvantageEstimateBb=float(a[action, hand]),
            playedAdvantageEstimateBb=float(b[action, hand]),
            currentActionProbability=float(mix[action, hand])))
    examples.sort(key=lambda r: -(r["trainingAdvantageEstimateBb"]-r["playedAdvantageEstimateBb"]))
    return dict(rootUpdatedThisRound=d["rootUpdatedThisRound"],
        maximumRootAdvantageGapBb=float(np.abs(gap).max()),
        rootAdvantageGapRmsBb=float(np.sqrt((gap**2*weights).mean(axis=0).sum())),
        trainingRootAdvantageRmsBb=float(np.sqrt((a**2*weights).mean(axis=0).sum())),
        bestActionChangedComboFraction=float(weights[changed].sum()),
        trainingOnlyPositiveRegretComboFraction=float(weights[spurious.any(axis=0)].sum()),
        maximumCurrentProfileValueChangeBb=float(np.abs(train_value-profile_value).max()),
        zeroOwnRootHoldings=d["zeroOwnRootHoldings"],
        maxNativeGapOnPositiveRootReachBb=d["maxNativeGapOnPositiveRootReachBb"],
        maxNativeGapOnZeroRootReachBb=d["maxNativeGapOnZeroRootReachBb"], examples=examples[:5])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "reference"):
        p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--"+name+"-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite diagnosis")
    manifests = []
    for name in ("diagnostic", "reference"):
        path = getattr(args,name)
        if sha256(path) != getattr(args,name+"_sha256"):
            raise ValueError("changed replay manifest")
        m = json.loads(path.read_text())
        if m["status"] != "complete" or [j["seed"] for j in m["jobs"]] != [27001,27002]:
            raise ValueError("complete ordered pair required")
        manifests.append(m)
    results = []
    for new, old in zip(*(m["jobs"] for m in manifests)):
        values = []
        for job in (new,old):
            path = Path(job["output"])
            if sha256(path) != job["outputSha256"]:
                raise ValueError("changed replay output")
            value = json.loads(path.read_text())
            if sha256(Path(value["frozenPolicy"])) != value["frozenPolicySha256"]:
                raise ValueError("changed policy export")
            values.append(value)
        current, reference = values
        if (current["frozenPolicySha256"] != reference["frozenPolicySha256"]
                or current["targetDiagnosticEnabled"] is not True):
            raise ValueError("diagnostic changed the policy: do not interpret the probe")
        ticks = []
        for tick in current["progress"]:
            if tick["targetDiagnostic"] is not None:
                ticks.append(dict(round=tick["round"], selectedHistory=tick["selectedHistory"],
                                  **inspect_tick(tick["targetDiagnostic"])))
        results.append(dict(seed=new["seed"], frozenPolicySha256=current["frozenPolicySha256"],
                            policyBytesUnchanged=True, ticks=ticks))
    report = dict(schema="preflop-training-profile-target-diagnosis-v1", results=results,
        diagnosticManifestSha256=args.diagnostic_sha256, referenceManifestSha256=args.reference_sha256,
        releaseAccepted=False, interpretation="Observed importance-weighted updates, not expected action EVs or proof of the cause of the 128-update regression. No policy changes in this replay.")
    atomic_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), outputSha256=sha256(args.output),
        seeds=[dict(seed=r["seed"], policyBytesUnchanged=True,
            rootUpdates=[{k:t[k] for k in ("round","maximumRootAdvantageGapBb", "bestActionChangedComboFraction",
                "trainingOnlyPositiveRegretComboFraction")} for t in r["ticks"] if t["rootUpdatedThisRound"]]) for r in results]),indent=2))


if __name__ == "__main__":
    main()

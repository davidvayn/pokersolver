"""Read-only teacher/chance sensitivity of complete TRAIN decision groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action_contrast_dataset import build_groups
from run_native_value_preflight import atomic_json, sha256
from decision_pilot_screen import MINIMUM_IMPROVEMENT_BB


def qualification(report):
    """Conservative DATA-pilot rule, not an equilibrium or release gate."""
    groups = [g for f in report["families"] for g in f["groups"]]
    if len(groups) != 6 or any(g["sensitivityTurnCount"] != 8 for g in groups):
        raise ValueError("qualification needs both complete eight-turn blocks")
    chance = [g["primaryVsDisjointSensitivity"]["firstTargetLossFromSecondBestBb"] for g in groups]
    teacher = [g["sameTurn64To256"]["firstTargetLossFromSecondBestBb"] for g in groups]
    if not np.isfinite(chance + teacher).all() or min(chance + teacher) < -1e-9:
        raise ValueError("invalid teacher/chance ranking loss")
    uncertain = max(chance + teacher) >= MINIMUM_IMPROVEMENT_BB
    return dict(schema="action-contrast-data-decision-v1",
                status="inconclusive" if uncertain else "ready_for_fit_preflight",
                bundleManifestSha256=report["bundleManifestSha256"],
                primaryTrainingManifestSha256=report["primaryTrainingManifestSha256"],
                usefulEffectThresholdBb=MINIMUM_IMPROVEMENT_BB,
                maximumChanceBlockRankingLossBb=max(chance), maximumSentinelRankingLossBb=max(teacher),
                reason="Target ranking sensitivity is comparable to or exceeds the intended policy effect."
                       if uncertain else "Bounded sensitivity screen passed; actual policy benefit remains unmeasured.",
                releaseAccepted=False, modelPromotionAllowed=False)


def compare(first, second):
    if (first.history != second.history or first.actions != second.actions
            or not np.array_equal(first.weights, second.weights)):
        raise ValueError("sensitivity changed prefix or decision group")
    weight = first.weights * first.support * second.support
    if weight.sum() <= 0: raise ValueError("no coherent sensitivity reach")
    errors = []
    for a in range(len(first.actions)):
        for b in range(a + 1, len(first.actions)):
            errors.append((second.target[a] - second.target[b]) - (first.target[a] - first.target[b]))
    error = np.asarray(errors)
    rms = float(np.sqrt(np.sum(error**2 * weight) / weight.sum() / len(errors)))
    best = np.argmax(first.target, axis=0)
    other = np.argmax(second.target, axis=0)
    actual_best = np.max(first.target, axis=0)
    chosen = np.take_along_axis(first.target, other[None, :], axis=0)[0]
    loss = float(np.sum(weight * (actual_best - chosen)) / weight.sum())
    return dict(actionContrastRmseBb=rms, firstTargetLossFromSecondBestBb=loss,
                bestActionAgreement=float(np.sum(weight * (best == other)) / weight.sum()),
                coherentReachFraction=float(weight.sum() / first.weights.sum()))


def inspect(path: Path, digest: str):
    if sha256(path) != digest: raise ValueError("bundle manifest changed")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("calibrationOnly"):
        raise ValueError("complete training bundles required")
    result = dict(schema="action-bundle-quality-v1", bundleManifestSha256=digest,
                  primaryTrainingManifestSha256=manifest.get("extendedFromSha256", digest),
                  releaseAccepted=False, families=[],
                  interpretation="Training-only finite-label/chance sensitivity; not action-EV sampling confidence or a release metric.")
    for family in manifest["families"]:
        prefix_row = family["prefix"]
        if sha256(Path(prefix_row["path"])) != prefix_row["sha256"]: raise ValueError("prefix changed")
        prefix = json.loads(Path(prefix_row["path"]).read_text())
        packets = []
        for receipt in family["labels"] + family["sentinels"]:
            path = Path(receipt["path"])
            if sha256(path) != receipt["sha256"]: raise ValueError("label changed")
            packets.append(json.loads(path.read_text()))
        primary = [p for p in packets if p["turn"] in family["trainingTurns"] and p["turn_iterations"] == 64]
        groups, _ = build_groups(prefix, primary)
        high = [p for p in packets if p["turn_iterations"] == 256]
        sensitivity = [p for p in packets if p["turn"] in family["sensitivityTurns"]]
        if len(primary) != 8 or len(high) != 1 or not sensitivity: raise ValueError("planned sensitivity incomplete")
        low = [p for p in primary if p["turn"] == high[0]["turn"]]
        low_groups, _ = build_groups(prefix, low)
        high_groups, _ = build_groups(prefix, high)
        changed = [high[0] if p["turn"] == high[0]["turn"] else p for p in primary]
        hybrid_groups, _ = build_groups(prefix, changed)
        sensitivity_groups, _ = build_groups(prefix, sensitivity)
        result["families"].append(dict(root=family["root"], groups=[dict(
            history=g.history, support=g.report(),
            sameTurn64To256=compare(l, h),
            oneTurnUpgradeEffectOnEightTurnTarget=compare(g, hybrid),
            primaryVsDisjointSensitivity=compare(g, sample),
            sensitivityTurnCount=len(sensitivity))
            for g, l, h, hybrid, sample in zip(groups, low_groups, high_groups, hybrid_groups, sensitivity_groups, strict=True)]))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decision-output", type=Path)
    a = p.parse_args()
    if a.output.exists(): raise ValueError("never overwrite quality diagnosis")
    result = inspect(a.manifest.resolve(), a.manifest_sha256)
    if a.decision_output:
        if a.decision_output.exists(): raise ValueError("never overwrite data decision")
        atomic_json(a.decision_output, qualification(result))
    atomic_json(a.output, result)
    print(json.dumps(result))

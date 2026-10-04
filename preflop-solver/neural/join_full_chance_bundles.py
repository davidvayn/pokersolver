"""Join the three pinned full-chance TRAIN shards; never activate a model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from action_contrast_dataset import build_groups
from complete_action_bundle_chance import verified_json
from decision_pilot_screen import MINIMUM_IMPROVEMENT_BB
from run_native_value_pilot import read_capture
from run_native_value_preflight import atomic_json, sha256


def data_decision(qualities, bundle_sha):
    """Finite-teacher training screen. Exact chance integration is not GTO proof."""
    if len(qualities) != 3 or sorted(q["root"] for q in qualities) != [2, 3, 4]:
        raise ValueError("exactly the three predeclared training families required")
    groups = []
    for quality in qualities:
        if (quality.get("schema") != "full-chance-action-bundle-quality-v1"
                or quality.get("all49Turns") is not True or quality.get("nativeBudget") != 64
                or quality.get("releaseAccepted") is not False or len(quality["groups"]) != 2):
            raise ValueError("incomplete full-chance target qualification")
        groups.extend(quality["groups"])
    losses = [g["sentinel64To256"]["firstTargetLossFromSecondBestBb"] for g in groups]
    full_losses = [g["oneSentinelUpgradeEffectOnAll49"]["firstTargetLossFromSecondBestBb"] for g in groups]
    support = [g["support"]["profileConsistentReachFraction"] for g in groups]
    if (not np.isfinite(losses + full_losses + support).all()
            or min(losses + full_losses) < -1e-9 or min(support) <= 0):
        raise ValueError("invalid teacher drift or empty profile-consistent support")
    uncertain = max(losses + full_losses) >= MINIMUM_IMPROVEMENT_BB
    return dict(schema="action-contrast-data-decision-v1",
        status="inconclusive" if uncertain else "ready_for_fit_preflight",
        bundleManifestSha256=bundle_sha, primaryTrainingManifestSha256=bundle_sha,
        usefulEffectThresholdBb=MINIMUM_IMPROVEMENT_BB,
        maximumSentinelRankingLossBb=max(losses), maximumOneTurnAll49UpgradeLossBb=max(full_losses),
        minimumProfileConsistentReachFraction=min(support),
        chanceIntegration="all49_exact_for_fixed_native64_labels",
        reason="Measured teacher drift is comparable to intended policy benefit." if uncertain else
               "Complete public-turn coverage and bounded teacher sentinels permit a matched training pilot, not promotion.",
        limitation="Sentinels do not certify every continuation; native64 targets remain approximate. Actual-policy benefit is unmeasured.",
        releaseAccepted=False, modelPromotionAllowed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", nargs=2, action="append", required=True, metavar=("MANIFEST", "SHA256"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError("never overwrite a joined target set")
    families, qualities, references, source_ids = [], [], [], set()
    for name, digest in a.reference:
        path = Path(name).resolve()
        if sha256(path) != digest: raise ValueError("reference manifest changed")
        manifest = json.loads(path.read_text())
        if (manifest.get("schema") != "full-chance-action-bundle-pilot-v1"
                or manifest.get("status") != "complete" or manifest.get("releaseAccepted") is not False
                or len(manifest["families"]) != 1):
            raise ValueError("complete immutable full-chance reference required")
        family = manifest["families"][0]
        if family.get("all49Turns") is not True or len(family["labels"]) != 49:
            raise ValueError("missing public turns")
        prefix = verified_json(family["prefix"])
        packets = [verified_json(r) for r in family["labels"]]
        if any(p["turn_iterations"] != 64 for p in packets): raise ValueError("teacher budget changed")
        _, ordered = build_groups(prefix, packets, require_full_chance=True)
        # Preserve separate 245/441/441-state shards, never bypass the 640-state
        # bound by flattening them into the existing 615-state calibration set.
        corpus_path = Path(family["calibrationCorpus"]["path"])
        if sha256(corpus_path) != family["calibrationCorpus"]["sha256"]:
            raise ValueError("full-chance calibration hash changed")
        corpus = read_capture(corpus_path)
        if corpus["targets"] != ordered:
            raise ValueError("full-chance target/calibration ordering changed")
        quality = verified_json(manifest["quality"])
        if quality["root"] != family["root"]: raise ValueError("quality family mismatch")
        families.append(copy.deepcopy(family)); qualities.append(quality)
        references.append(dict(path=str(path), sha256=digest)); source_ids.add(manifest["sourceManifestSha256"])
    if len(source_ids) != 1 or len({tuple(f["family"]) for f in families}) != 3:
        raise ValueError("changed source or non-disjoint training families")
    # Qualify BEFORE creating anything. Final decision pins the saved set's hash.
    data_decision(qualities, "0" * 64)
    if any(sha256(Path(r["path"])) != r["sha256"] for r in references):
        raise ValueError("reference changed while joining")
    a.output = a.output.resolve(); a.output.mkdir()
    joined = dict(schema="full-chance-action-bundle-set-v1", status="complete", calibrationOnly=False,
        all49Turns=True, nativeBudget=64, sourceManifestSha256=source_ids.pop(), references=references,
        families=sorted(families, key=lambda f: f["root"]), releaseAccepted=False)
    path = a.output / "manifest.json"; atomic_json(path, joined)
    decision = data_decision(qualities, sha256(path))
    atomic_json(a.output / "decision.json", decision)
    print(json.dumps(dict(manifestSha256=sha256(path), decisionSha256=sha256(a.output / "decision.json"),
                         decision=decision)), flush=True)


if __name__ == "__main__": main()

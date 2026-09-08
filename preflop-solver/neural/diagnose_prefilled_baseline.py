"""Saved-data endpoint-noise diagnostic; never fits or exports a playing policy."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from run_frozen_preflop_response import IDENTITY, validate_capture
from run_native_value_preflight import atomic_json, sha256


def endpoint_lottery(contributions, baseline):
    """Enumerate the uniform one-endpoint correction, not a sampled simulation.

    Arrays are [board, endpoint, class, action]. The baseline is fixed before
    these boards and has shape [endpoint, class, action]. Exact terminal and
    checkdown contributions cancel from the variance calculation.
    """
    values = np.asarray(contributions, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if (values.ndim != 4 or values.shape[1:] != base.shape
            or not np.isfinite(values).all() or not np.isfinite(base).all()):
        raise ValueError("invalid endpoint lottery")
    estimates = base.sum(axis=0) + values.shape[1] * (values-base)
    if not np.allclose(estimates.mean(axis=1), values.sum(axis=1), atol=1e-12, rtol=1e-12):
        raise ValueError("baseline changed expected update")
    return estimates


def fitting_shrinkage(contributions, weights):
    """Two-fold fitting-only least-squares weight for a past-board baseline.

    Center across endpoints because only that lottery's variance is targeted.
    The fitted mean of both boards is used later; this one-board cross-fit
    weight deliberately does not use either evaluation board for tuning.
    """
    x = np.asarray(contributions, dtype=float)
    if x.ndim != 4 or x.shape[0] != 2 or not np.isfinite(x).all():
        raise ValueError("exactly two finite fitting boards required")
    centered = x-x.mean(axis=1, keepdims=True)
    covariance = float((centered[0]*centered[1]*weights).sum())
    square = float(((centered[0]**2+centered[1]**2)*weights).sum()/2)
    return 0.0 if square == 0 else float(np.clip(covariance/square, 0, 1))


def inspect(outputs, kernel):
    outputs = sorted(outputs, key=lambda o: o["boardIndex"])
    if [o["boardIndex"] for o in outputs] != [0, 1, 2, 3]:
        raise ValueError("four prescribed development clusters required")
    first = outputs[0]
    if any(any(o[k] != first[k] for k in IDENTITY) for o in outputs):
        raise ValueError("mixed policy or continuation")
    parsed = [validate_capture(o, True) for o in outputs]
    if (kernel["schema"] != "exact-preflop-checkdown-class-kernel-v1"
            or kernel["complete"] is not True or kernel["rawFlops"] != 22100
            or kernel["canonicalFlops"] != 1755 or kernel["classes"] != first["classes"]
            or kernel["classMultiplicities"] != first["multiplicities"]):
        raise ValueError("incompatible exact kernel")
    pairs = np.asarray(kernel["compatiblePairs"], dtype=float).reshape(169, 169)
    wins = np.asarray(kernel["weightedWinUnits"], dtype=float).reshape(169, 169)
    counts = np.asarray(kernel["weightedFlopPairs"], dtype=float).reshape(169, 169)
    if not np.array_equal(counts, pairs*17296) or not np.array_equal(wins+wins.T, counts*1980):
        raise ValueError("kernel chance/zero-sum counts differ")
    mult = np.asarray(first["multiplicities"], dtype=float)
    rows, _ = parsed[0]
    root = tuple(first["rootHistory"])
    root_probs = np.asarray(rows[root]["probabilities"]).T
    if rows[root]["actor"] != 0:
        raise ValueError("diagnostic expects SB initial decision")
    # Track own action reach below (not including) the root action separately.
    reaches, downstream, root_actions = {}, {}, {}
    def visit(h, prior, own, root_action):
        reaches[h], downstream[h], root_actions[h] = prior, own, root_action
        if h not in rows:
            return
        row = rows[h]
        for a, child in enumerate(row["children"]):
            mix = np.asarray(row["probabilities"][a])
            updated = prior.copy()
            updated[row["actor"]] *= mix
            next_own = own * mix if h != root and row["actor"] == 0 else own
            visit(tuple(child), updated, next_own, a if h == root else root_action)
    visit(root, np.ones((2, 169)), np.ones(169), None)
    histories = sorted(tuple(c["history"]) for c in first["captures"])
    invested = {tuple(c["history"]): c["investedBb"] for c in first["captures"]}
    contributions = []
    for h in histories:
        stake = invested[h]
        matrix = ((sum(stake)*wins/(17296*1980) - stake[0]*pairs)
                  / (mult[:, None]*1225))
        exact = matrix @ reaches[h][1]
        # A residual in root action a changes every regret by -sigma(a)*r,
        # and its own regret by an additional r. No own root probability here.
        action = root_actions[h]
        direction = -np.repeat(root_probs[:, action, None], root_probs.shape[1], axis=1)
        direction[:, action] += 1
        scalar = np.asarray([end[h][0]-exact for _, end in parsed])
        contributions.append(scalar[:, :, None]*downstream[h][None, :, None]*direction)
    contributions = np.stack(contributions, axis=1)
    fitted = contributions[:2].mean(axis=0)
    weights = mult[:, None]/1326/root_probs.shape[1]
    shrinkage = fitting_shrinkage(contributions[:2], weights)
    results = []
    for index in (2, 3):
        values = contributions[index:index+1]
        old = endpoint_lottery(values, np.zeros_like(fitted))
        new = endpoint_lottery(values, fitted)
        shrunk = endpoint_lottery(values, shrinkage*fitted)
        # Enumerated endpoint variance conditional on each saved chance board.
        # Not action-EV SE, full-game variance, or a strength measurement.
        before = float((old.var(axis=1)[0]*weights).sum())
        after = float((new.var(axis=1)[0]*weights).sum())
        conservative = float((shrunk.var(axis=1)[0]*weights).sum())
        results.append(dict(boardIndex=index, oldEndpointVarianceBb2=before,
                            prefilledEndpointVarianceBb2=after,
                            ratio=None if before == 0 else after/before,
                            crossFitShrunkVarianceBb2=conservative,
                            crossFitShrunkRatio=None if before == 0 else conservative/before))
    return dict(fitBoardIndices=[0, 1], evaluationBoardIndices=[2, 3],
                fittingOnlyShrinkage=shrinkage,
                results=results, releaseAccepted=False,
                interpretation="Fixed-reference SB root regret estimator only. Exact enumeration of uniform endpoint lottery on two development evaluation boards; fit baseline uses boards 0/1 only. Does not measure strength or establish variance reduction when training ranges change.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("changed manifest or existing output")
    source = json.loads(args.manifest.read_text())
    if source["status"] != "complete" or source["completeCapture"] is not True:
        raise ValueError("complete native captures required")
    outputs = []
    for job in source["jobs"]:
        path = Path(job["output"])
        if sha256(path) != job["outputSha256"]:
            raise ValueError("changed native endpoint values")
        outputs.append(json.loads(path.read_text()))
    if any(o["kernelSha256"] != sha256(args.kernel) for o in outputs):
        raise ValueError("kernel differs from captures")
    kernel = json.loads(gzip.decompress(args.kernel.read_bytes()))
    result = inspect(outputs, kernel)
    result.update(sourceManifestSha256=args.manifest_sha256, runnerSha256=sha256(Path(__file__)))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

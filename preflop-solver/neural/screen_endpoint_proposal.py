"""Offline, development-only importance allocation; never runs native training."""
import argparse
import json
from pathlib import Path

import numpy as np

from endpoint_variance import decision_residuals, fit_proposal, gradient_energies, importance_variance
from run_native_value_preflight import atomic_json, sha256


def pinned(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--response", action="append", required=True, nargs=2,
                        metavar=("MANIFEST", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite a screen")
    base_manifest = pinned(args.baseline, args.baseline_sha)
    if base_manifest["status"] != "complete" or len(args.response) != 2:
        raise ValueError("complete paired sources required")
    baselines = {}
    for job in base_manifest["jobs"]:
        base = pinned(job["output"], job["outputSha256"])
        baselines[base["preflopSha256"]] = base
    frames, sources = [], {str(args.baseline.resolve()): args.baseline_sha}
    histories = None
    for manifest_path, digest in args.response:
        manifest = pinned(manifest_path, digest)
        sources[str(Path(manifest_path).resolve())] = digest
        if manifest["status"] != "complete" or sorted(j["index"] for j in manifest["jobs"]) != list(range(4)):
            raise ValueError("complete four-board source required")
        for job in sorted(manifest["jobs"], key=lambda j: j["index"]):
            capture = pinned(job["output"], job["outputSha256"])
            if capture["boardIndex"] != job["index"]:
                raise ValueError("capture index mismatch")
            base = baselines[capture["preflopSha256"]]
            if histories is None:
                histories = base["liveHistories"]
            if histories != base["liveHistories"]:
                raise ValueError("endpoint order mismatch")
            energies, totals, costs = gradient_energies(capture, base)
            frames.append(dict(capture=capture, baseline=base, energies=energies,
                               totals=totals, costs=costs, output=job["output"], sha=job["outputSha256"]))
            print(json.dumps(dict(board=capture["boardIndex"], policy=capture["preflopSha256"], stage="energies_complete")), flush=True)
    if len({f["capture"]["preflopSha256"] for f in frames}) != 2:
        raise ValueError("two distinct frozen policies required")
    fitting = [f for f in frames if f["capture"]["boardIndex"] < 2]
    proposals = np.array([fit_proposal(np.array([f["energies"][a] for f in fitting])) for a in range(2)])
    uniform = np.full(49, 1/49)
    results = []
    for frame in frames:
        capture, base = frame["capture"], frame["baseline"]
        actor_results = []
        for actor, q in enumerate(proposals):
            old = float((frame["energies"][actor]/uniform).sum()-frame["totals"][actor])
            new = float((frame["energies"][actor]/q).sum()-frame["totals"][actor])
            cost_ratio = float((q@frame["costs"])/(uniform@frame["costs"]))
            actor_results.append(dict(actor=actor, uniformGradientVariance=old,
                proposalGradientVariance=new, varianceRatio=new/old,
                expectedCostRatio=cost_ratio, varianceCostRatio=new/old*cost_ratio))
        decisions = []
        root = capture["rootHistory"]
        selected = [r for r in capture["rows"] if r["history"] == root or
                    r["history"] in [root+["Preflop:p0:limp"], root+["Preflop:p0:raise_to_4.000bb"]]]
        if len(selected) != 3:
            raise ValueError("missing opening/limp/4bb diagnostics")
        for row in selected:
            residuals, _, costs = decision_residuals(capture, base, row["history"])
            weights = np.asarray(capture["multiplicities"])/1326
            old = float((importance_variance(residuals, uniform)*weights).sum(axis=-1).mean())
            new = float((importance_variance(residuals, proposals[row["actor"]])*weights).sum(axis=-1).mean())
            decisions.append(dict(history=row["history"], varianceRatio=new/old))
        results.append(dict(boardIndex=capture["boardIndex"], preflopSha256=capture["preflopSha256"] ,
            role="fit" if capture["boardIndex"] < 2 else "development_evaluation",
            output=frame["output"], outputSha256=frame["sha"], actors=actor_results, decisions=decisions))
    report = dict(schema="preflop-endpoint-importance-screen-v1", sourceManifests=sources,
        analysisSha256=sha256(Path(__file__)), helperSha256=sha256(Path(__file__).with_name("endpoint_variance.py")),
        fitBoardIndices=[0, 1], evaluationBoardIndices=[2, 3], uniformMixture=.5,
        liveHistories=histories, probabilitiesByActor=proposals.tolist(), results=results,
        nativeQueries=0, releaseAccepted=False,
        interpretation="Conditional endpoint-selection gradient variance, not chance variance or policy strength. Fits both seeds on boards 0/1; evaluates reused development boards 2/3. No untouched qualification, exploitability or convergence claim. Timing reuses saved observations.")
    atomic_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output),
        evaluation=[r for r in results if r["role"] == "development_evaluation"])), flush=True)


if __name__ == "__main__":
    main()

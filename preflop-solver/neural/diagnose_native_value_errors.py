"""Inspect saved native predictions without fitting, feature caches, or new solves.

These are conditional-value approximation errors against finite-budget native
labels, not action-EV sampling errors or an exploitability estimate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import native_value_dataset as native
from run_native_value_pilot import read_capture
from run_native_value_preflight import atomic_json, sha256


def error_statistics(truth, prediction, weights):
    truth, prediction, weights = (np.asarray(x, dtype=np.float64) for x in (truth, prediction, weights))
    if (truth.shape != prediction.shape or truth.shape != weights.shape
            or not all(np.isfinite(x).all() for x in (truth, prediction, weights))
            or (weights < 0).any()):
        raise ValueError("error diagnostics require aligned finite values and nonnegative weights")
    mass = float(weights.sum())
    if mass == 0:
        return dict(weightMass=0.0, weightedRmseBb=None, weightedMaeBb=None,
                    weightedBiasBb=None, weightAbove050Bb=None)
    error = prediction - truth
    return dict(weightMass=mass,
                weightedRmseBb=float(np.sqrt(np.sum(weights * error**2) / mass)),
                weightedMaeBb=float(np.sum(weights * np.abs(error)) / mass),
                weightedBiasBb=float(np.sum(weights * error) / mass),
                weightAbove050Bb=float(np.sum(weights * (np.abs(error) > .5)) / mass))


def paired_statistics(truth, prediction, weights):
    result = error_statistics(truth, prediction, weights)
    result["perSeat"] = [error_statistics(truth[..., p, :], prediction[..., p, :], weights[..., p, :])
                         for p in (0, 1)]
    return result


def diagnose(source, predictions, split):
    truth = np.asarray([t["counterfactual_values_bb"] for t in source["targets"]])
    ranges = np.asarray([t["ranges"] for t in source["targets"]])
    masses = np.asarray([t["opponent_compatible_mass"] for t in source["targets"]])
    if predictions.shape != truth.shape or not np.isfinite(predictions).all():
        raise ValueError("saved prediction shape/finiteness mismatch")
    weights = ranges * masses
    zero_own_weights = np.asarray([
        native.training_weights(t["board"], ranges[i], masses[i]) * (ranges[i] == 0)
        for i, t in enumerate(source["targets"])
    ])
    raw_masses = masses * np.asarray([t["raw_reach_totals"] for t in source["targets"]])[:, ::-1, None]
    profile = np.divide(np.asarray([t["raw_profile_counterfactual_bb"] for t in source["targets"]]),
                        raw_masses, out=np.zeros_like(truth), where=raw_masses > 0)
    reports = {}
    for name, indices in split.items():
        groups = {key: defaultdict(list) for key in ("flopFamily", "distribution", "investment")}
        states = []
        for i in indices:
            target = source["targets"][i]
            groups["flopFamily"][str(native.board_family(target["board"]))].append(i)
            groups["distribution"][target["state_distribution"]].append(i)
            groups["investment"][str(target["invested_bb"])].append(i)
            stats = paired_statistics(truth[i], predictions[i], weights[i])
            # All states remain in summaries; the most costly state list is
            # descriptive only and never an alternate model-selection split.
            states.append(dict(state=int(i), board=target["board"], investedBb=target["invested_bb"],
                               history=target["public_state"]["public_history"], **stats))
        reports[name] = dict(
            authentic=paired_statistics(truth[indices], predictions[indices], weights[indices]),
            zeroOwnCounterfactual=paired_statistics(truth[indices], predictions[indices], zero_own_weights[indices]),
            zeroOwnAgainstFrozenProfile=paired_statistics(profile[indices], predictions[indices], zero_own_weights[indices]),
            zeroOwnCompletionGap=paired_statistics(profile[indices], truth[indices], zero_own_weights[indices]),
            groups={facet: {key: dict(states=len(rows), **paired_statistics(truth[rows], predictions[rows], weights[rows]))
                            for key, rows in members.items()} for facet, members in groups.items()},
            largestStateRmse=sorted(states, key=lambda row: row["weightedRmseBb"] or 0, reverse=True)[:8],
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "students", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--students-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.students) != args.students_sha256:
        raise ValueError("existing output or student identity mismatch")
    students = json.loads(args.students.read_text())
    if students["status"] != "complete" or sha256(args.corpus) != students["corpusSha256"]:
        raise ValueError("complete matching student/corpus required")
    source = read_capture(args.corpus)
    native.validate_dataset(source)
    if "split" in students:
        split = {name: np.asarray(students["split"][name], dtype=np.int64) for name in ("train", "tuning", "holdout")}
        if sorted(np.concatenate(list(split.values())).tolist()) != list(range(len(source["targets"]))):
            raise ValueError("saved split is not a complete disjoint partition")
        families = [{native.board_family(source["targets"][i]["board"]) for i in indices} for indices in split.values()]
        if any(families[a] & families[b] for a, b in ((0,1), (0,2), (1,2))):
            raise ValueError("saved split leaks flop families")
    else:
        split = dict(zip(("train", "tuning", "holdout"), native.family_split(source, 10601, .25, .25)))
    report = dict(schema="native-value-error-diagnostics-v1", releaseAccepted=False,
                  corpusSha256=students["corpusSha256"], studentsSha256=args.students_sha256,
                  scriptSha256=sha256(Path(__file__)), splitUnit="whole_flop_family",
                  interpretation="Finite-budget conditional value errors, not action EV SE or exploitability; no retraining or selection.",
                  weightInterpretation="Authentic private-hand reach within each sampled state; public states are not weighted by full-game reach. Zero-sum projection can force combined signed bias to zero while individual-seat errors remain large.",
                  seeds=[])
    for entry in students["predictions"]:
        path = args.students.parent / ("predictions-%d.json" % entry["seed"])
        prediction_sha = sha256(path)
        if entry.get("predictionsSha256", prediction_sha) != prediction_sha:
            raise ValueError("saved verified predictions changed")
        prediction = json.loads(path.read_text())
        if prediction["modelSha256"] != entry["modelSha256"]:
            raise ValueError("prediction model identity mismatch")
        report["seeds"].append(dict(seed=entry["seed"], predictionsSha256=prediction_sha,
                                  predictionDigestRecordedAtVerification="predictionsSha256" in entry,
                                  modelSha256=entry["modelSha256"],
                                  errors=diagnose(source, np.asarray(prediction["predictions"]), split)))
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "seeds": [
        dict(seed=s["seed"], splits={k: v["authentic"] for k, v in s["errors"].items()})
        for s in report["seeds"]]}))


if __name__ == "__main__":
    main()

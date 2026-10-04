"""Predeclared control-first screen; conditional research, never a release gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from run_flop_update_pilot import DEFAULT_BASELINE, BASELINE_SHA, check_baseline
from run_native_value_preflight import atomic_json, sha256

SPOTS = ("three-bet-high-rainbow", "three-bet-monotone")
SEEDS = (100101, 100102)
CASES = {(spot, seed) for spot in SPOTS for seed in SEEDS}
MAXIMUM_REGRESSION_BB = .01
MINIMUM_IMPROVEMENT_BB = .02


def screen(rows: list[dict]) -> dict:
    if (len(rows) != len(CASES)
            or {(r["spot"], r["seed"]) for r in rows} != CASES):
        raise ValueError("screen requires exactly both declared roots and seeds")
    matched = ["matchedGainBb" in r for r in rows]
    if any(matched) and not all(matched):
        raise ValueError("matched control must cover every case")
    if any(not math.isfinite(r[key]) or r[key] < -1e-8
           for r in rows for key in ("gainBb", "oldGainBb", *(('matchedGainBb',) if all(matched) else ()))):
        raise ValueError("screen gains must be finite and nonnegative")
    differences = [{**r, "improvementBb": r["oldGainBb"] - r["gainBb"]}
                   for r in rows]
    means = {str(seed): math.fsum(r["improvementBb"] for r in differences
                                 if r["seed"] == seed) / len(SPOTS) for seed in SEEDS}
    mean = math.fsum(r["improvementBb"] for r in differences) / len(rows)
    regressions = [dict(spot=r["spot"], seed=r["seed"], regressionBb=-r["improvementBb"])
                   for r in differences if r["improvementBb"] < -MAXIMUM_REGRESSION_BB]
    matched_means = ({str(seed): math.fsum(r["matchedGainBb"] - r["gainBb"]
                                          for r in rows if r["seed"] == seed) / len(SPOTS)
                     for seed in SEEDS} if all(matched) else None)
    status = ("rejected" if regressions else "promising"
              if mean >= MINIMUM_IMPROVEMENT_BB and all(v > 0 for v in means.values())
              and (matched_means is None or all(v > 0 for v in matched_means.values()))
              else "inconclusive")
    return dict(status=status, cases=differences, equalCaseMeanImprovementBb=mean,
                seedMeanImprovementBb=means, matchedSeedMeanImprovementBb=matched_means,
                materialRegressions=regressions, releaseAccepted=False,
                interpretation="Four consumed three-bet cases with all49 native64 continuations; not fresh-board or full-game qualification.")


def verified_rows(path: Path, expected_sha: str) -> tuple[dict, list[dict]]:
    if sha256(path) != expected_sha:
        raise ValueError("screen response manifest hash changed")
    manifest = json.loads(path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("baselineSha256") != BASELINE_SHA
            or manifest.get("playedTurnIterations") != 64):
        raise ValueError("screen requires completed matched native64 responses")
    rows = [r for r in manifest["cases"] if (r["spot"], r["seed"]) in CASES]
    if len(rows) != len(CASES):
        raise ValueError("cached screen is incomplete")
    for row in rows:
        work = path.parent / row["spot"] / str(row["seed"])
        candidate_path, response_path = work / "candidate.json", work / "response.json"
        if (sha256(candidate_path) != row["candidateSha256"]
                or sha256(response_path) != row["responseSha256"]):
            raise ValueError("cached candidate or response changed")
        candidate = json.loads(candidate_path.read_text())
        response = json.loads(response_path.read_text())
        if (candidate["turn_iterations"] != 64 or candidate["iterations"] not in (128, 256)
                or response["candidate_sha256"] != row["candidateSha256"]
                or response["public_turns"] != 49
                or not math.isclose(response["half_summed_gain_bb"], row["gainBb"], abs_tol=1e-12)):
            raise ValueError("cached game/chance/response contract changed")
        for turn in range(52):
            if turn in candidate["state"]["board"]:
                continue
            packet = work / "packets" / f"turn-{turn}.json"
            receipt = json.loads((work / f"turn-{turn}" / "completed.json").read_text())
            if (sha256(packet) != receipt["outputs"][str(packet)]
                    or receipt["identity"]["environment"]["POKER_NATIVE_FLOP_CANDIDATE_SHA"] != row["candidateSha256"]
                    or receipt["worker"]["status"] != "complete"):
                raise ValueError("cached native turn packet/receipt changed")
        audit = json.loads((work / "audit" / "completed.json").read_text())
        if audit["worker"]["status"] != "complete":
            raise ValueError("cached independent backup audit incomplete")
    return manifest, rows


def calibrate(output: Path):
    if output.exists():
        raise ValueError("refusing to overwrite calibration")
    here = Path(__file__).resolve().parent
    inputs = [
        ("native64-student", "local-student-response-label64-20261003-confirm-a",
         "a0abb78df7c1b8b36169c6b0332d8400da6f1fda7154ecdb3917dc095a917088"),
        ("native256-student", "local-student-response-label256-20261003-confirm-a",
         "0375d5d5ccdd74f8f154486155506ef12b18a469cbfd171a97c7a995dabe8996"),
        ("256-flop-updates", "local-flop-update-20261003-confirm-a",
         "79da5ceb3127a9bc9e0616981f8d3b759e82b3d0b7398473b336d74079a93b35"),
    ]
    started = time.monotonic()
    _, baseline = check_baseline(DEFAULT_BASELINE, BASELINE_SHA, list(SPOTS))
    old = {(spot["id"], model["seed"]): row["halfSummedGainBb"] for spot, model, row in baseline}
    record = dict(schema="decision-pilot-calibration-v1", status="running",
                  baselineSha256=BASELINE_SHA, screenSpots=list(SPOTS), arms=[],
                  codeSha256=sha256(Path(__file__)), releaseAccepted=False)
    output.mkdir()
    try:
        for name, directory, digest in inputs:
            path = here / "runs" / directory / "manifest.json"
            manifest, cached = verified_rows(path, digest)
            rows = [dict(spot=r["spot"], seed=r["seed"], gainBb=r["gainBb"],
                         oldGainBb=old[(r["spot"], r["seed"])]) for r in cached]
            result = screen(rows)
            if (name.endswith("student") and result["status"] != "rejected"
                    or name == "256-flop-updates" and result["status"] != "promising"):
                raise ValueError("cheap screen failed its known negative/positive controls")
            record["arms"].append(dict(name=name, manifestSha256=digest,
                                      previousElapsedSeconds=manifest["elapsedSeconds"], screen=result))
        # Opposite off-/on-policy conclusions are important; preserve the pinned
        # diagnosis without pretending alternative values changed a policy.
        diagnostic_path = here / "runs/local-student-policy-diagnosis-20261003-a/control-100101/action-values.json"
        if sha256(diagnostic_path) != "385ed78eb957084af78bb08e21463eb47c3c56b7f4771ed38adbf1f6af71a53a":
            raise ValueError("on-policy action diagnosis changed")
        diagnosed = json.loads(diagnostic_path.read_text())
        top = max(diagnosed["actionValueDiagnostics"], key=lambda r: r["rootWeightedRankingLossBb"])
        record["ownPolicyControlDiagnosis"] = {key: top[key] for key in (
            "history", "nativeBestAgreement", "nativeLossFromPredictedBestBb",
            "nativePolicyDeviationLossBb", "predictedPolicyDeviationLossBb", "actionContrastRmseBb")}
        record["status"] = "complete"
    except (OSError, ValueError, KeyError) as error:
        record["status"], record["failure"] = "failed", str(error)
    record["elapsedSeconds"] = time.monotonic() - started
    atomic_json(output / "manifest.json", record)
    print(json.dumps(record))
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.calibrate:
        parser.error("use --calibrate; policy scoring is imported by controllers")
    calibrate(args.output.resolve())

"""Read-only paired response summary for a completed postflop leaf pilot."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_native_value_preflight import atomic_json, sha256


def summarize(pilot, protocol):
    if (pilot.get("schema") != "postflop-gap-matched-leaf-pilot-v1"
            or pilot.get("phase") != "compare" or pilot.get("status") != "complete"
            or pilot.get("iterations") != 32
            or protocol.get("schema") != "postflop-benchmark-protocol-v1"):
        raise ValueError("complete matched 32-update pilot and benchmark protocol required")
    spots = {spot["id"]: spot for spot in protocol["spots"]}
    requested = pilot["spots"]
    if (len(requested) != len(set(requested)) or not requested
            or set(requested) - spots.keys()):
        raise ValueError("invalid selected development roots")
    rows = {}
    for row in pilot["cases"]:
        key = row["spot"], row["seed"], row["arm"]
        if (key in rows or key[0] not in requested or key[1] not in (100101, 100102)
                or key[2] not in ("native", "learned")
                or not math.isfinite(row.get("gainBb", math.nan))
                or row["gainBb"] < -1e-8):
            raise ValueError("duplicate, invalid or nonresponse pilot case")
        rows[key] = row
    expected = {(spot, seed, arm) for spot in requested
                for seed in (100101, 100102) for arm in ("native", "learned")}
    if rows.keys() != expected:
        raise ValueError("incomplete paired response; cannot issue a headline result")
    pairs = []
    for spot in requested:
        pot = spots[spot]["startingPotBb"]
        if not math.isfinite(pot) or pot <= 0:
            raise ValueError("invalid starting pot")
        for seed in (100101, 100102):
            native = rows[spot, seed, "native"]["gainBb"]
            learned = rows[spot, seed, "learned"]["gainBb"]
            pairs.append(dict(spot=spot, potType=spots[spot]["potType"],
                              seed=seed, startingPotBb=pot,
                              nativeGainBb=native, learnedGainBb=learned,
                              nativePercentPot=100*native/pot,
                              learnedPercentPot=100*learned/pot,
                              improvementBb=learned-native,
                              improvementPercentagePointsPot=100*(learned-native)/pot))
    by_spot = []
    for spot in requested:
        selected = [row for row in pairs if row["spot"] == spot]
        by_spot.append(dict(spot=spot, meanImprovementBb=math.fsum(
            row["improvementBb"] for row in selected)/2,
            pairedSeedImprovementsBb=[row["improvementBb"] for row in selected]))
    return dict(schema="postflop-gap-paired-response-summary-v1",
                releaseAccepted=False, fullGameGateEvaluated=False,
                interpretation="Native versus learned leaves at matched 32 flop updates and frozen native64 played continuation; selected development roots, not whole-game exploitability or untouched confirmation.",
                pairs=pairs, bySpot=by_spot,
                meanNativeGainBb=math.fsum(row["nativeGainBb"] for row in pairs)/len(pairs),
                meanLearnedGainBb=math.fsum(row["learnedGainBb"] for row in pairs)/len(pairs),
                meanNativePercentPot=math.fsum(row["nativePercentPot"] for row in pairs)/len(pairs),
                meanLearnedPercentPot=math.fsum(row["learnedPercentPot"] for row in pairs)/len(pairs),
                meanImprovementBb=math.fsum(row["improvementBb"] for row in pairs)/len(pairs),
                meanImprovementPercentagePointsPot=math.fsum(
                    row["improvementPercentagePointsPot"] for row in pairs)/len(pairs),
                allSeedRootPairsImprove=all(row["improvementBb"] > 0 for row in pairs))


def verify_saved_outputs(pilot, root):
    for row in pilot["cases"]:
        work = root / row["spot"] / str(row["seed"]) / row["arm"]
        candidate, response = work / "candidate.json", work / "response.json"
        if (sha256(candidate) != row["candidateSha256"]
                or sha256(response) != row["responseSha256"]):
            raise ValueError("completed pilot artifact changed")
        actual = json.loads(response.read_text())
        if not math.isclose(actual["half_summed_gain_bb"], row["gainBb"],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("manifest response gain differs from frozen response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--pilot-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.pilot) != args.pilot_sha256 or sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("existing output or changed pinned pilot/protocol")
    pilot, protocol = json.loads(args.pilot.read_text()), json.loads(args.protocol.read_text())
    if pilot["protocolSha256"] != args.protocol_sha256:
        raise ValueError("pilot did not use the pinned protocol")
    summary = summarize(pilot, protocol)
    verify_saved_outputs(pilot, args.pilot.parent)
    summary.update(pilotSha256=args.pilot_sha256, protocolSha256=args.protocol_sha256)
    atomic_json(args.output, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

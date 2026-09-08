"""Read-only 2x2 diagnostic: solver endpoint seeds versus continuation chance.

Two replicates cannot establish a variance component or equilibrium guarantee.
The comparisons identify which source deserves the next bounded pilot.
"""
import argparse
import json
from pathlib import Path

from cloud_blueprint_run import policy_stability_summary
from run_preflop_average_pilot import stability
from run_native_value_preflight import atomic_json, sha256


def root_metrics(outputs):
    roots = []
    for result in outputs:
        length = min(len(r["history"]) for r in result["rows"])
        root = [r for r in result["rows"] if len(r["history"]) == length]
        roots.append(dict(rootStrategies=[dict(hand=r["hand"], averageVisits=r["averageVisits"],
            regretUpdates=r["regretUpdates"], trainedAverage=r["trained"],
            actions=[dict(action=a, probability=p) for a,p in zip(r["actions"],r["probabilities"])]) for r in root]))
    return policy_stability_summary(roots)


def analyze(first, second):
    if (first["status"] != "complete" or second["status"] != "complete"
            or first["rounds"] != second["rounds"]
            or [first["continuationSeed"],second["continuationSeed"]] != [28001,28002]):
        raise ValueError("completed matched-round independent continuation pairs required")
    values = []
    for m in (first,second):
        if [j["seed"] for j in m["jobs"]] != [27001,27002]:
            raise ValueError("independent ordered solver seeds required")
        pair = []
        for job in m["jobs"]:
            path = Path(job["output"])
            if sha256(path) != job["outputSha256"]: raise ValueError("changed paired result")
            result = json.loads(path.read_text())
            if (result["continuationSeed"] != m["continuationSeed"]
                    or result["config"]["seed"] != job["seed"]
                    or result["config"]["iterations"] != m["rounds"]):
                raise ValueError("paired output identity mismatch")
            pair.append(result)
        values.append(pair)
    if len({v["valueModelSha256"] for pair in values for v in pair}) != 1:
        raise ValueError("continuation weights differ")
    if len({v.get("exactCheckdownSha256") for pair in values for v in pair}) != 1:
        raise ValueError("checkdown expectation differs")
    if len({v.get("endpointSampling", "uniform_one") for pair in values for v in pair}) != 1:
        raise ValueError("endpoint sampling differs")
    if len({v.get("historyBaseline", False) for pair in values for v in pair}) != 1:
        raise ValueError("history baseline differs")
    if len({v.get("completeTurnBaseline", False) for pair in values for v in pair}) != 1:
        raise ValueError("turn baseline differs")
    if len({v.get("flopCheckdownScale", 1.0) for pair in values for v in pair}) != 1:
        raise ValueError("flop checkdown scale differs")
    if len({v.get("simultaneousUpdates", False) for pair in values for v in pair}) != 1:
        raise ValueError("player update schedule differs")
    if len({v.get("rootRealizationTurnAverages", False) for pair in values for v in pair}) != 1:
        raise ValueError("turn average definition differs")
    return dict(schema="compact-preflop-noise-source-diagnostic-v1", releaseAccepted=False,
        rounds=first["rounds"], sameChanceDifferentSolver=[root_metrics(pair) for pair in values],
        sameSolverDifferentChance=[dict(seed=27001+i,
            root=root_metrics([values[0][i],values[1][i]]),
            withinPublicState=stability(values[0][i],values[1][i])) for i in range(2)],
        interpretation=__doc__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first","second"):
        parser.add_argument("--"+name,type=Path,required=True)
        parser.add_argument("--"+name+"-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    for name in ("first","second"):
        if sha256(getattr(args,name)) != getattr(args,name+"_sha256"):
            raise ValueError("changed paired manifest")
    if args.output.exists(): raise ValueError("refusing to overwrite diagnostic")
    result=analyze(json.loads(args.first.read_text()),json.loads(args.second.read_text()))
    result["sourceManifestSha256"]=[args.first_sha256,args.second_sha256]
    result["runnerSha256"]=sha256(Path(__file__))
    atomic_json(args.output,result)
    print(json.dumps(dict(
        solverMae=[v["maximumComboWeightedPerActionMae"] for v in result["sameChanceDifferentSolver"]],
        chanceMae=[v["root"]["maximumComboWeightedPerActionMae"] for v in result["sameSolverDifferentChance"]],
        outputSha256=sha256(args.output))),flush=True)


if __name__=="__main__": main()

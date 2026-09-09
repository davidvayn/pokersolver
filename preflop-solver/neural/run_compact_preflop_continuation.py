"""Bounded paired compact preflop solves against one pinned continuation model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import signal
import threading
import time

from cloud_blueprint_run import policy_stability_summary
from run_preflop_average_pilot import stability
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::preflop_continuation::compact_preflop_continuation_pilot"


def compare(outputs):
    if len(outputs) != 2:
        raise ValueError("two complete independent solver seeds required")
    roots = []
    for result in outputs:
        if (result.get("schema") != "compact-preflop-continuation-pilot-v1"
                or result.get("releaseAccepted") is not False
                or result["totalNodes"] != 16900 or len(result["rows"]) != 16900
                or any(not r["trained"] for r in result["rows"])):
            raise ValueError("incomplete compact preflop policy")
        length = min(len(r["history"]) for r in result["rows"])
        root = [r for r in result["rows"] if len(r["history"]) == length]
        roots.append(dict(rootStrategies=[dict(hand=r["hand"], averageVisits=r["averageVisits"],
            regretUpdates=r["regretUpdates"], trainedAverage=r["trained"],
            actions=[dict(action=a, probability=p) for a,p in zip(r["actions"],r["probabilities"])]) for r in root]))
    if outputs[0]["valueModelSha256"] != outputs[1]["valueModelSha256"]:
        raise ValueError("paired continuation model differs")
    if outputs[0].get("exactCheckdownSha256") != outputs[1].get("exactCheckdownSha256"):
        raise ValueError("paired checkdown expectation differs")
    if outputs[0].get("endpointSampling", "uniform_one") != outputs[1].get("endpointSampling", "uniform_one"):
        raise ValueError("paired endpoint sampling differs")
    if outputs[0].get("endpointProposalSha256") != outputs[1].get("endpointProposalSha256"):
        raise ValueError("paired endpoint proposal differs")
    if outputs[0].get("endpointProposalRefreshAfterRound") != outputs[1].get("endpointProposalRefreshAfterRound"):
        raise ValueError("paired endpoint proposal schedule differs")
    if outputs[0].get("historyBaseline", False) != outputs[1].get("historyBaseline", False):
        raise ValueError("paired history baseline differs")
    if outputs[0].get("completeTurnBaseline", False) != outputs[1].get("completeTurnBaseline", False):
        raise ValueError("paired turn baseline differs")
    if outputs[0].get("flopCheckdownScale", 1.0) != outputs[1].get("flopCheckdownScale", 1.0):
        raise ValueError("paired flop checkdown scale differs")
    if outputs[0].get("simultaneousUpdates", False) != outputs[1].get("simultaneousUpdates", False):
        raise ValueError("paired player update schedule differs")
    if outputs[0].get("rootRealizationTurnAverages", False) != outputs[1].get("rootRealizationTurnAverages", False):
        raise ValueError("paired turn average definition differs")
    if outputs[0].get("playedProfileTargets", False) != outputs[1].get("playedProfileTargets", False):
        raise ValueError("paired continuation target definition differs")
    if outputs[0].get("regretSchedule", "dcfr") != outputs[1].get("regretSchedule", "dcfr"):
        raise ValueError("paired regret weighting differs")
    if outputs[0].get("independentBoards", 1) != outputs[1].get("independentBoards", 1):
        raise ValueError("paired public chance batch differs")
    if [v["config"]["seed"] for v in outputs] != [27001,27002]:
        raise ValueError("unexpected paired solver seeds")
    return dict(establishedRootStability=policy_stability_summary(roots),
                withinPublicState=stability(*outputs),
                interpretation="Same continuation model/function and exact private integration; stability is not equilibrium proof or response resistance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--rounds", type=int, choices=[2,4,8,16,32,128], default=2)
    parser.add_argument("--workers", type=int, choices=[1,2], default=1)
    parser.add_argument("--independent-boards", type=int, choices=[1,2], default=1)
    parser.add_argument("--checkdown",type=Path)
    parser.add_argument("--checkdown-sha256")
    parser.add_argument("--continuation-seed", type=int, choices=[28001,28002], default=28001)
    parser.add_argument("--endpoint-sampling", choices=["uniform_one", "uniform_batch", "root_stratified", "opponent_reach", "fixed_importance"], default="uniform_one")
    parser.add_argument("--endpoint-proposal", type=Path)
    parser.add_argument("--endpoint-proposal-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-baseline", action="store_true")
    parser.add_argument("--turn-baseline", action="store_true")
    parser.add_argument("--flop-baseline-scale", type=float, choices=[1.0,2.0], default=1.0)
    parser.add_argument("--simultaneous-updates", action="store_true")
    parser.add_argument("--turn-root-averages", action="store_true")
    parser.add_argument("--diagnose-targets", action="store_true")
    parser.add_argument("--played-profile-targets", action="store_true")
    parser.add_argument("--trace-root-updates", action="store_true")
    parser.add_argument("--regret-schedule", choices=["dcfr", "lcfr"], default="dcfr")
    parser.add_argument("--checkpoint-interval", type=int, choices=[0,8,16,32], default=0)
    parser.add_argument("--maximum-worker-seconds", type=int, choices=[3600,5400])
    parser.add_argument("--maximum-worker-memory-mib", type=int, choices=[2048,2560], default=2048)
    parser.add_argument("--resume", action="append", nargs=3, default=[], metavar=("SEED","RECEIPT","SHA256"))
    args = parser.parse_args()
    if args.independent_boards != 1 and (args.rounds > 32 or not args.played_profile_targets
            or not args.turn_baseline or args.endpoint_sampling != "fixed_importance"
            or args.history_baseline or args.simultaneous_updates or args.turn_root_averages
            or args.diagnose_targets or args.trace_root_updates or args.flop_baseline_scale != 1.0):
        raise ValueError("independent boards require isolated <=32-update fixed-proposal played-target pilot")
    proposal_refresh_round = None
    pinned = {}
    for name in ("binary", "model"):
        path = getattr(args,name).resolve()
        digest = getattr(args,name+"_sha256")
        if sha256(path) != digest: raise ValueError("pinned input changed")
        setattr(args,name,path)
        pinned[str(path)] = digest
    if bool(args.checkdown) != bool(args.checkdown_sha256):
        raise ValueError("exact checkdown path and digest must be supplied together")
    if args.endpoint_sampling != "uniform_one" and not args.checkdown:
        raise ValueError("stratified pilot requires the pinned exact checkdown expectation")
    if args.history_baseline and (not args.checkdown or args.endpoint_sampling != "uniform_one"):
        raise ValueError("history baseline pilot requires uniform selection and exact checkdown mean")
    if args.turn_baseline and (args.history_baseline or not args.checkdown or args.endpoint_sampling not in ("uniform_one", "fixed_importance")):
        raise ValueError("turn baseline pilot requires uniform or fixed-importance exact-mean control")
    if bool(args.endpoint_proposal) != bool(args.endpoint_proposal_sha256) or bool(args.endpoint_proposal) != (args.endpoint_sampling == "fixed_importance"):
        raise ValueError("fixed importance requires an explicit pinned proposal")
    if args.endpoint_proposal:
        if (args.rounds > 128 or not args.played_profile_targets or not args.turn_baseline
                or args.simultaneous_updates or args.turn_root_averages or args.diagnose_targets
                or args.history_baseline or args.flop_baseline_scale != 1.0):
            raise ValueError("fixed importance pilot requires isolated <=128-update played-target control")
        args.endpoint_proposal = args.endpoint_proposal.resolve()
        if sha256(args.endpoint_proposal) != args.endpoint_proposal_sha256:
            raise ValueError("endpoint proposal changed")
        pinned[str(args.endpoint_proposal)] = args.endpoint_proposal_sha256
        proposal = json.loads(args.endpoint_proposal.read_text())
        if proposal.get("probabilitiesAfterRound32") is not None:
            if args.regret_schedule != "lcfr":
                raise ValueError("late proposal refresh requires isolated LCFR pilot")
            proposal_refresh_round = 32
    if args.flop_baseline_scale!=1.0 and not args.turn_baseline:
        raise ValueError("scale pilot requires the retained complete-turn control")
    if args.simultaneous_updates and (not args.turn_baseline or args.flop_baseline_scale != 1.0):
        raise ValueError("simultaneous pilot requires isolated unscaled complete-turn control")
    if args.turn_root_averages and (args.simultaneous_updates or not args.turn_baseline or args.flop_baseline_scale != 1.0):
        raise ValueError("turn average pilot requires isolated alternating unscaled turn control")
    if args.rounds > 32 and (not args.turn_baseline or args.simultaneous_updates
            or args.turn_root_averages or args.history_baseline
            or args.endpoint_sampling not in ("uniform_one", "fixed_importance") or args.flop_baseline_scale != 1.0):
        raise ValueError("128-update cost audit covers only the retained alternating turn control")
    if args.diagnose_targets and (args.rounds > 8 or not args.turn_baseline
            or args.simultaneous_updates or args.turn_root_averages or args.flop_baseline_scale != 1.0):
        raise ValueError("target diagnostic requires a <=8-update unchanged turn-control replay")
    if args.played_profile_targets and (args.rounds > 128 or not args.turn_baseline
            or args.simultaneous_updates or args.diagnose_targets
            or args.flop_baseline_scale != 1.0):
        raise ValueError("played-target pilot requires isolated <=128-update alternating turn control")
    if args.trace_root_updates and (args.rounds > 32 or not args.played_profile_targets
            or args.turn_root_averages or args.diagnose_targets):
        raise ValueError("root trace requires a <=32-update unchanged played-target replay")
    if (args.checkpoint_interval or args.resume) and (not args.played_profile_targets
            or args.history_baseline or args.simultaneous_updates or args.turn_root_averages
            or args.diagnose_targets or args.flop_baseline_scale != 1.0):
        raise ValueError("recovery supports only isolated played-profile continuation training")
    if args.maximum_worker_seconds and (args.rounds != 128 or not args.checkpoint_interval):
        raise ValueError("extended runtime requires checkpointed128 training")
    if args.maximum_worker_memory_mib > 2048 and (args.rounds != 128 or not args.checkpoint_interval):
        raise ValueError("extended memory requires checkpointed128 training")
    if args.regret_schedule == "lcfr" and (args.rounds > 128 or not args.played_profile_targets
            or not args.turn_baseline or args.endpoint_sampling != "fixed_importance"
            or args.simultaneous_updates or args.turn_root_averages
            or args.history_baseline or args.diagnose_targets or args.flop_baseline_scale != 1.0):
        raise ValueError("LCFR screen permits only isolated <=128-update importance pilots")
    resumes = {}
    for seed, path, digest in args.resume:
        seed, path = int(seed), Path(path).resolve()
        if seed not in (27001,27002) or seed in resumes or sha256(path) != digest:
            raise ValueError("invalid or duplicate pinned seed recovery")
        receipt = json.loads(path.read_text())
        count = receipt["completedIterations"]
        if (receipt["schema"] != "compact-preflop-checkpoint-v1" or type(count) is not int
                or not 0 < count <= args.rounds or receipt["checkpoint"] != f"round{count:04}.mpk.gz"):
            raise ValueError("invalid compact checkpoint generation")
        state = path.parent/receipt["checkpoint"]
        if sha256(state) != receipt["checkpointSha256"]:
            raise ValueError("checkpoint state changed")
        pinned[str(path)], pinned[str(state)] = digest, receipt["checkpointSha256"]
        resumes[seed] = dict(receipt=str(path),sha256=digest,round=count)
    if args.checkdown:
        args.checkdown=args.checkdown.resolve()
        if sha256(args.checkdown)!=args.checkdown_sha256: raise ValueError("checkdown kernel changed")
        pinned[str(args.checkdown)]=args.checkdown_sha256
    for name in ("cloud_blueprint_run.py", "run_preflop_average_pilot.py", "run_native_value_pilot.py",
                 "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    started = time.monotonic()
    endpoint_budget = 1 if args.endpoint_sampling in ("uniform_one", "opponent_reach", "fixed_importance") else 6
    seconds_budget = args.maximum_worker_seconds or min(3600, 150*args.rounds*endpoint_budget*args.independent_boards)
    record = dict(schema="compact-preflop-continuation-controller-v1", status="running",
        pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)), rounds=args.rounds,
        continuationSeed=args.continuation_seed, maximumWorkerSeconds=seconds_budget,
        endpointSampling=args.endpoint_sampling, maximumEndpointsPerRound=endpoint_budget,
        independentBoards=args.independent_boards,
        endpointProposalSha256=args.endpoint_proposal_sha256,
        endpointProposalRefreshAfterRound=proposal_refresh_round,
        historyBaseline=args.history_baseline,
        completeTurnBaseline=args.turn_baseline,
        flopCheckdownScale=args.flop_baseline_scale,
        simultaneousUpdates=args.simultaneous_updates,
        rootRealizationTurnAverages=args.turn_root_averages,
        targetDiagnosticEnabled=args.diagnose_targets,
        playedProfileTargets=args.played_profile_targets,
        rootUpdateTraceEnabled=args.trace_root_updates,
        regretSchedule=args.regret_schedule,
        checkpointInterval=args.checkpoint_interval, resumedSeeds=resumes,
        maximumConcurrentWorkers=args.workers,
        maximumWorkerMemoryBytes=args.maximum_worker_memory_mib*1024**2, jobs=[], releaseAccepted=False,
        continuationFunction=dict(flopIterations=128,trainingTurnIterations=64,playedTurnIterations=64,
            completeRootSupport=True,rootSeedRule="100101-xor-first8-le-sha256-public-input-v1"),
        baseline="exact_preflop_checkdown_mean" if args.checkdown else "sampled_flop_checkdown",
        exactCheckdownSha256=args.checkdown_sha256)
    atomic_json(args.output/"manifest.json",record)
    results = {}
    def job(seed):
        if stop.is_set() or any(sha256(Path(p))!=h for p,h in pinned.items()):
            raise ValueError("stopped stage or changed input")
        output = args.output/("seed%d.json"%seed)
        environment = {
            "POKER_COMPACT_SEED":str(seed), "POKER_COMPACT_CHANCE_SEED":str(args.continuation_seed),
            "POKER_COMPACT_ROUNDS":str(args.rounds), "POKER_COMPACT_MODEL":str(args.model),
            "POKER_COMPACT_MODEL_SHA":args.model_sha256, "POKER_COMPACT_OUTPUT":str(output),
            "POKER_COMPACT_ENDPOINT_SAMPLING":args.endpoint_sampling,
            "POKER_COMPACT_HISTORY_BASELINE":"1" if args.history_baseline else "0",
            "POKER_COMPACT_TURN_BASELINE":"1" if args.turn_baseline else "0",
            "POKER_COMPACT_FLOP_CHECKDOWN_SCALE":str(args.flop_baseline_scale),
            "POKER_COMPACT_SIMULTANEOUS":"1" if args.simultaneous_updates else "0",
            "POKER_COMPACT_TURN_ROOT_AVERAGES":"1" if args.turn_root_averages else "0",
            "POKER_COMPACT_DIAGNOSE_TARGETS":"1" if args.diagnose_targets else "0",
            "POKER_COMPACT_PLAYED_TARGETS":"1" if args.played_profile_targets else "0",
            "POKER_COMPACT_TRACE_ROOT":"1" if args.trace_root_updates else "0",
            "POKER_COMPACT_REGRET_SCHEDULE":args.regret_schedule,
            "POKER_COMPACT_CHECKPOINT_INTERVAL":str(args.checkpoint_interval),
            "POKER_COMPACT_BINARY_SHA":args.binary_sha256,
            "POKER_COMPACT_INDEPENDENT_BOARDS":str(args.independent_boards),
        }
        if args.checkdown:
            environment.update(POKER_COMPACT_CHECKDOWN=str(args.checkdown),
                               POKER_COMPACT_CHECKDOWN_SHA=args.checkdown_sha256)
        if args.endpoint_proposal:
            environment.update(POKER_COMPACT_PROPOSAL=str(args.endpoint_proposal),
                               POKER_COMPACT_PROPOSAL_SHA=args.endpoint_proposal_sha256)
        if seed in resumes:
            environment.update(POKER_COMPACT_RESUME_RECEIPT=resumes[seed]["receipt"],
                               POKER_COMPACT_RESUME_SHA=resumes[seed]["sha256"])
        worker = guarded(test_command(args.binary, TEST), environment,
            args.output/("seed%d-worker"%seed), seconds_budget, args.maximum_worker_memory_mib*1024**2, stop)
        result = json.loads(output.read_text())
        if (result["config"]["seed"]!=seed or result["config"]["iterations"]!=args.rounds
                or result["continuationSeed"]!=args.continuation_seed
                or result["valueModelSha256"]!=args.model_sha256
                or result.get("exactCheckdownSha256")!=args.checkdown_sha256
                or result.get("endpointSampling", "uniform_one")!=args.endpoint_sampling
                or result.get("endpointProposalSha256")!=args.endpoint_proposal_sha256
                or result.get("endpointProposalRefreshAfterRound")!=proposal_refresh_round
                or result.get("historyBaseline", False)!=args.history_baseline
                or result.get("completeTurnBaseline", False)!=args.turn_baseline
                or result.get("flopCheckdownScale", 1.0)!=args.flop_baseline_scale
                or result.get("simultaneousUpdates", False)!=args.simultaneous_updates
                or result.get("rootRealizationTurnAverages", False)!=args.turn_root_averages
                or result.get("targetDiagnosticEnabled", False)!=args.diagnose_targets
                or result.get("playedProfileTargets", False)!=args.played_profile_targets
                or result.get("rootUpdateTraceEnabled", False)!=args.trace_root_updates
                or result.get("regretSchedule", "dcfr")!=args.regret_schedule
                or result.get("independentBoards", 1)!=args.independent_boards
                or result.get("checkpointInterval",0)!=args.checkpoint_interval
                or result.get("resumedFromRound",0)!=resumes.get(seed,{}).get("round",0)
                or result.get("resumeReceiptSha256")!=resumes.get(seed,{}).get("sha256")
                or sha256(Path(result["frozenPolicy"]))!=result["frozenPolicySha256"]):
            raise ValueError("compact pilot identity mismatch")
        if len(result["progress"]) != args.rounds or [t["round"] for t in result["progress"]] != list(range(1,args.rounds+1)):
            raise ValueError("incomplete or discontinuous training progress")
        for tick in result["progress"]:
            samples=tick.get("endpointSamples")
            if samples is not None and (len(samples)!=endpoint_budget*args.independent_boards
                    or len({tuple(s["selectedHistory"]) for s in samples})!=endpoint_budget):
                raise ValueError("incomplete sampled endpoint batch")
            if args.independent_boards > 1:
                boards = tick.get("boardSamples", [])
                if len(boards) != args.independent_boards or tick.get("independentBoards") != args.independent_boards:
                    raise ValueError("missing independent chance provenance")
                expected = [((tick["round"]-1)//2)*(2*args.independent_boards)
                            +(tick["round"]-1)%2+2*b+1 for b in range(args.independent_boards)]
                if [b["chanceRound"] for b in boards] != expected:
                    raise ValueError("independent chance stream changed")
        return result, dict(seed=seed,output=str(output),outputSha256=sha256(output),worker=worker)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(job, seed) for seed in (27001,27002)]
            try:
                for future in as_completed(futures):
                    result, report = future.result()
                    seed = report["seed"]
                    results[seed] = result
                    record["jobs"].append(report)
                    record["jobs"].sort(key=lambda row: row["seed"])
                    atomic_json(args.output/"manifest.json",record)
                    print(json.dumps(dict(event="compact-preflop-seed-complete", seed=seed,
                        seconds=result["trainingSeconds"], nodes=result["totalNodes"])),flush=True)
            except Exception:
                stop.set()
                raise
        if stop.is_set() or any(sha256(Path(p))!=h for p,h in pinned.items()):
            raise ValueError("stopped stage or changed input")
        record["comparison"] = compare([results[seed] for seed in (27001,27002)])
        record["status"] = "complete"
    except (OSError,ValueError,KeyError,AssertionError) as error:
        record.update(status="failed",failure=str(error))
    record["elapsedSeconds"] = time.monotonic()-started
    atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","failure")}),flush=True)
    if record["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()

"""Frozen preflop response diagnostic; disjoint fitting/evaluation chance clusters."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import re
import signal
import threading
import time

import numpy as np
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::preflop_continuation::frozen_response::frozen_preflop_response_capture"
IDENTITY = ("preflopSha256", "modelSha256", "kernelSha256", "chanceSeed", "classes",
            "multiplicities", "rootHistory", "rows", "continuationFunction")


def projected_board_seconds(preflight):
    """Cost estimate only; poker state/value calculations stay in Rust.

    Charge each root the measured smaller-investment representative, with a
    doubled 2bb cost for the unmeasured 1bb pot and 50% total headroom.
    Runtime/memory guards remain authoritative if timing extrapolation fails.
    """
    validate_capture(preflight, False)
    costs = {c["investedBb"][0]: c["seconds"] for c in preflight["captures"]}
    if set(costs) != {2.0, 9.0, 18.0} or any(not np.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("missing representative endpoint cost")
    live = {tuple(h) for r in preflight["rows"] for h in r["children"] if h[-1] == "deal:Flop"}
    if len(live) != 49:
        raise ValueError("cost preflight has unexpected live tree")
    total = 0.0
    for h in live:
        raises = [float(s.split("raise_to_")[1][:-2]) for s in h if "raise_to_" in s]
        amount = raises[-1] if raises else 1.0
        total += (2*costs[2.0] if amount < 2 else costs[2.0] if amount < 9
                  else costs[9.0] if amount < 18 else costs[18.0])
    return total*1.5


def tree_values(rows, endpoints, root, seat, choices=None, fit=False):
    """CFVs contain opponent reaches already. Max only at own observable nodes.

    Inputs must first be averaged over fitting chance and private information
    classes. Evaluation uses frozen choices, never per-board maximization.
    """
    if fit and choices is None:
        raise ValueError("fit requires an explicit response destination")
    if root in endpoints:
        return endpoints[root][seat].copy()
    row = rows[root]
    children = np.asarray([tree_values(rows, endpoints, tuple(h), seat, choices, fit)
                           for h in row["children"]])
    if row["actor"] != seat:
        return children.sum(axis=0)
    if fit:
        choices[root] = children.argmax(axis=0)
    if choices is not None:
        return children[choices[root], np.arange(children.shape[1])]
    return (children * np.asarray(row["probabilities"])).sum(axis=0)


def validate_capture(value, complete):
    if (value["schema"] != "frozen-preflop-response-capture-v1"
            or value["releaseAccepted"] is not False or value["complete"] is not complete
            or value["chanceSeed"] != 32001 or value["boardIndex"] not in range(4)
            or len(value["board"]) != 3 or len(set(value["board"] + [value["turn"]])) != 4
            or any(type(c) is not int or c not in range(52) for c in value["board"] + [value["turn"]])
            or len(value["classes"]) != 169 or value["classes"] != sorted(set(value["classes"]))
            or len(value["multiplicities"]) != 169 or sum(value["multiplicities"]) != 1326
            or any(v not in (4, 6, 12) for v in value["multiplicities"])
            or len(value["captures"]) != (49 if complete else 3)):
        raise ValueError("invalid frozen response capture or incomplete board")
    rows = {tuple(r["history"]): r for r in value["rows"]}
    endpoints = {tuple(r["history"]): np.asarray(r["cfvBb"], dtype=float) for r in value["endpoints"]}
    if len(rows) != 100 or len(rows) != len(value["rows"]) or len(endpoints) != len(value["endpoints"]) or rows.keys() & endpoints.keys():
        raise ValueError("duplicate or invalid public tree")
    for row in rows.values():
        probs = np.asarray(row["probabilities"], dtype=float)
        if (row["actor"] not in (0, 1) or not row["actions"]
                or len(set(row["actions"])) != len(row["actions"])
                or len(row["actions"]) != len(row["children"])
                or probs.shape != (len(row["actions"]), 169)
                or not np.isfinite(probs).all() or (probs < 0).any()
                or np.max(np.abs(probs.sum(axis=0)-1)) > 1e-12):
            raise ValueError("invalid frozen response probabilities")
    if any(v.shape != (2, 169) or not np.isfinite(v).all() for v in endpoints.values()):
        raise ValueError("invalid frozen response CFVs")
    captured = [tuple(c["history"]) for c in value["captures"]]
    if len(set(captured)) != len(captured) or any(h not in endpoints for h in captured):
        raise ValueError("duplicate or missing native endpoints")
    if complete:
        visited = set()
        def visit(key):
            if key in visited:
                raise ValueError("cycle or multiply reached public history")
            visited.add(key)
            if key in endpoints:
                return
            if key not in rows:
                raise ValueError("missing endpoint; no fallback")
            for h in rows[key]["children"]:
                visit(tuple(h))
        visit(tuple(value["rootHistory"]))
        if visited != rows.keys() | endpoints.keys():
            raise ValueError("unreachable extra public states")
    return rows, endpoints


def summarize(outputs):
    if len(outputs) != 4 or sorted(o["boardIndex"] for o in outputs) != [0, 1, 2, 3]:
        raise ValueError("exactly four disjoint chance clusters required")
    outputs = sorted(outputs, key=lambda o: o["boardIndex"])
    first = outputs[0]
    parsed = []
    for o in outputs:
        if any(o[k] != first[k] for k in IDENTITY):
            raise ValueError("mixed frozen policy, public tree or continuation")
        parsed.append(validate_capture(o, True))
    rows, _ = parsed[0]
    root = tuple(first["rootHistory"])
    keys = parsed[0][1].keys()
    if any(p[1].keys() != keys for p in parsed):
        raise ValueError("endpoint coverage differs")
    train = {h: (parsed[0][1][h]+parsed[1][1][h])/2 for h in keys}
    weights = np.asarray(first["multiplicities"])/1326
    choices = [{}, {}]
    training = []
    for seat in range(2):
        response = tree_values(rows, train, root, seat, choices[seat], True)
        profile = tree_values(rows, train, root, seat)
        gain = float((weights*(response-profile)).sum())
        if gain < -1e-9:
            raise ValueError("fitted response is below reference on fitting sample")
        training.append(gain)
    evaluated = []
    for index in (2, 3):
        end = parsed[index][1]
        profile = [float(weights @ tree_values(rows, end, root, s)) for s in range(2)]
        response = [float(weights @ tree_values(rows, end, root, s, choices[s])) for s in range(2)]
        if abs(sum(profile)) > 1e-8:
            raise ValueError("frozen profile lost zero sum")
        gains = [response[s]-profile[s] for s in range(2)]
        evaluated.append(dict(boardIndex=index, profileBb=profile, responseBb=response,
                              gainsBb=gains, summedGainBb=sum(gains)))
    gains = np.asarray([r["gainsBb"] for r in evaluated])
    totals = gains.sum(axis=1)
    policy = [dict(seat=s, history=list(h), actions=v.tolist())
              for s in range(2) for h, v in sorted(choices[s].items())]
    return dict(fitBoardIndices=[0, 1], evaluationBoardIndices=[2, 3],
        trainingGainsBb=training, evaluation=evaluated,
        evaluationMeanGainsBb=gains.mean(axis=0).tolist(),
        evaluationMeanSummedGainBb=float(totals.mean()),
        summedGainStandardErrorBb=float(totals.std(ddof=1)/2**0.5),
        confidenceUpperBound99Bb=None, releaseAccepted=False, fullGameGateEvaluated=False,
        frozenResponse=policy,
        interpretation="Restricted preflop attacker fitted on two chance clusters, evaluated on two different clusters. Postflop continues with the original reference ranges/policies, NOT ranges re-solved for the attacker. Too few independent clusters for a qualifying confidence bound. Not full-game exploitability or an accepted self-play policy.")


def recover_captures(manifest, digest, pinned, expected):
    """Recover finished numerical captures, never infer missing guard telemetry.

    An interrupted controller may leave detached native workers alive. Refuse
    to launch replacements for any started board lacking a successful final
    result. Recovery writes a NEW manifest; the source remains immutable.
    """
    manifest = manifest.resolve()
    if not digest or sha256(manifest) != digest:
        raise ValueError("recovery manifest changed")
    prior = json.loads(manifest.read_text())
    if (prior.get("schema") != "frozen-preflop-response-controller-v1"
            or prior.get("completeCapture") is not True
            or prior.get("rootRealizationTurnAverages", False) !=
                expected["continuationFunction"].get("rootRealizationTurnAverages", False)
            or any(prior.get("pinnedInputs", {}).get(k) != v
                   for k, v in pinned.items() if k in prior.get("pinnedInputs", {}))):
        raise ValueError("incompatible recovery controller")
    # The four primary inputs must be explicitly pinned by the old controller.
    for key in ("preflopSha256", "modelSha256", "kernelSha256"):
        if expected[key] not in prior.get("pinnedInputs", {}).values():
            raise ValueError("missing recovery input identity")
    for path, value in prior["pinnedInputs"].items():
        if sha256(Path(path)) != value:
            raise ValueError("recovery input changed")
    pinned.update(prior["pinnedInputs"])
    pinned[str(manifest)] = digest
    values, reports = [], []
    for index in range(4):
        output = manifest.parent/f"board{index}.json"
        worker_dir = manifest.parent/f"board{index}-worker"
        if not output.exists() and not worker_dir.exists():
            continue
        log = worker_dir/"worker.log"
        if not output.is_file() or not log.is_file() or not re.search(
                r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in [\d.]+s\s*\Z",
                log.read_text()):
            raise ValueError("started recovery board is incomplete; do not duplicate a possibly live worker")
        value = json.loads(output.read_text())
        validate_capture(value, True)
        if value["boardIndex"] != index or any(value[k] != expected[k] for k in IDENTITY):
            raise ValueError("recovered board identity differs")
        output_sha = sha256(output)
        prior_jobs = [j for j in prior.get("jobs", []) if j["index"] == index]
        if len(prior_jobs) > 1 or (prior_jobs and prior_jobs[0]["outputSha256"] != output_sha):
            raise ValueError("duplicate or changed recorded recovery board")
        pinned[str(output)] = output_sha
        pinned[str(log)] = sha256(log)
        values.append(value)
        reports.append(dict(index=index, output=str(output), outputSha256=output_sha,
            recovered=True, worker=None,
            telemetry="unavailable: original controller interrupted; native test completed successfully"))
    if not values:
        raise ValueError("no completed captures to recover")
    return values, reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "preflop", "model", "kernel"):
        p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--"+name+"-sha256", required=True)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--preflight-sha256")
    p.add_argument("--complete", action="store_true")
    p.add_argument("--turn-root-averages", action="store_true")
    p.add_argument("--recover-from", type=Path, help="Pinned interrupted controller manifest; use a new output directory")
    p.add_argument("--recover-sha256")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (args.recover_from is not None) != (args.recover_sha256 is not None) or (args.recover_from and not args.complete):
        raise ValueError("recovery requires complete mode and a manifest hash")
    pinned = {}
    for name in ("binary", "preflop", "model", "kernel"):
        path = getattr(args, name).resolve()
        digest = getattr(args, name+"_sha256")
        if sha256(path) != digest:
            raise ValueError("frozen response input changed")
        setattr(args, name, path)
        pinned[str(path)] = digest
    projection = None
    if args.complete:
        if not args.preflight or sha256(args.preflight) != args.preflight_sha256:
            raise ValueError("complete capture requires pinned three-endpoint preflight")
        prior = json.loads(args.preflight.read_text())
        if (prior["status"] != "complete" or prior["completeCapture"] is not False
                or prior.get("rootRealizationTurnAverages", False) != args.turn_root_averages
                or any(prior["pinnedInputs"].get(k) != v for k, v in pinned.items())):
            raise ValueError("incompatible frozen response preflight")
        source = Path(prior["jobs"][0]["output"])
        if sha256(source) != prior["jobs"][0]["outputSha256"]:
            raise ValueError("preflight endpoint output changed")
        preflight_value = json.loads(source.read_text())
        projection = projected_board_seconds(preflight_value)
        if projection > 3600:
            raise ValueError("projected board capture exceeds one-hour worker cap")
        pinned[str(args.preflight.resolve())] = args.preflight_sha256
        pinned[str(source.resolve())] = prior["jobs"][0]["outputSha256"]
    for name in ("run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    recovered_values, recovered_jobs = [], []
    if args.recover_from:
        prior_pins = json.loads(args.recover_from.read_text()).get("pinnedInputs", {})
        if any(prior_pins.get(str(getattr(args, name))) != getattr(args, name+"_sha256")
               for name in ("binary", "preflop", "model", "kernel")):
            raise ValueError("recovery primary inputs differ")
        recovered_values, recovered_jobs = recover_captures(
            args.recover_from, args.recover_sha256, pinned, preflight_value)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    started = time.monotonic()
    record = dict(schema="frozen-preflop-response-controller-v1", status="running",
        completeCapture=args.complete, pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)),
        rootRealizationTurnAverages=args.turn_root_averages,
        projectedSecondsPerBoard=projection,
        maximumWorkerSeconds=3600 if args.complete else 600, maximumWorkerMemoryBytes=2*1024**3,
        maximumConcurrentWorkers=2 if args.complete else 1, jobs=recovered_jobs, releaseAccepted=False)
    if args.recover_from:
        record["recoveredFrom"] = str(args.recover_from.resolve())
    atomic_json(args.output/"manifest.json", record)
    def job(index):
        if stop.is_set() or any(sha256(Path(k)) != v for k, v in pinned.items()):
            raise ValueError("stopped or changed input")
        output = args.output/f"board{index}.json"
        worker = guarded(test_command(args.binary, TEST), dict(
            POKER_NOISE_PREFLOP=str(args.preflop), POKER_NOISE_PREFLOP_SHA=args.preflop_sha256,
            POKER_COMPACT_MODEL=str(args.model), POKER_COMPACT_MODEL_SHA=args.model_sha256,
            POKER_COMPACT_CHECKDOWN=str(args.kernel), POKER_COMPACT_CHECKDOWN_SHA=args.kernel_sha256,
            POKER_NOISE_BOARD_INDEX=str(index), POKER_RESPONSE_PREFLIGHT="0" if args.complete else "1",
            POKER_COMPACT_TURN_ROOT_AVERAGES="1" if args.turn_root_averages else "0",
            POKER_COMPACT_OUTPUT=str(output)), args.output/f"board{index}-worker",
            record["maximumWorkerSeconds"], record["maximumWorkerMemoryBytes"], stop)
        value = json.loads(output.read_text())
        validate_capture(value, args.complete)
        if (value["boardIndex"] != index or value["preflopSha256"] != args.preflop_sha256
                or value["continuationFunction"].get("rootRealizationTurnAverages", False) != args.turn_root_averages
                or value["modelSha256"] != args.model_sha256 or value["kernelSha256"] != args.kernel_sha256):
            raise ValueError("wrong frozen response output")
        return value, dict(index=index, output=str(output), outputSha256=sha256(output), worker=worker)
    try:
        values = list(recovered_values)
        finished = {v["boardIndex"] for v in values}
        with ThreadPoolExecutor(max_workers=record["maximumConcurrentWorkers"]) as pool:
            futures = [pool.submit(job, i) for i in range(4 if args.complete else 1) if i not in finished]
            try:
                for f in as_completed(futures):
                    value, report = f.result()
                    values.append(value)
                    record["jobs"].append(report)
                    record["jobs"].sort(key=lambda r: r["index"])
                    atomic_json(args.output/"manifest.json", record)
                    print(json.dumps(dict(event="frozen-response-board-complete", index=value["boardIndex"], seconds=value["seconds"])), flush=True)
            except Exception:
                stop.set()
                raise
        if stop.is_set() or any(sha256(Path(k)) != v for k, v in pinned.items()):
            raise ValueError("stopped or changed inputs")
        if args.complete:
            record["summary"] = summarize(values)
        else:
            record["preflightCaptures"] = values[0]["captures"]
        record["status"] = "complete"
    except (OSError, ValueError, KeyError, AssertionError) as error:
        record.update(status="failed", failure=str(error))
    record["elapsedSeconds"] = time.monotonic()-started
    atomic_json(args.output/"manifest.json", record)
    print(json.dumps({k: record.get(k) for k in ("status", "elapsedSeconds", "failure")}), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

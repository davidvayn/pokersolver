"""Scores-blind TRAIN coverage: three bounded all49/search-belief shards.

Freeze the protocol before labeling. Native labels, not learned predictions,
are the targets. No fitting, evaluation-family reuse or model activation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

from action_contrast_dataset import build_groups
from complete_action_bundle_chance import verified_json
from inspect_action_bundle_quality import compare
from join_full_chance_bundles import data_decision
import native_value_dataset as native
from run_action_bundle_pilot import BASE
from run_native_value_pilot import guarded, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from run_search_distribution_pilot import TEST
from run_student_value_pilot import select_students
from training_coverage import extra_calibration, extension_families, select_families


def freeze(args):
    pinned = {}
    for name in ("binary", "corpus", "split_reference", "roots", "students", "benchmark_protocol", "preflop"):
        path = getattr(args, name).resolve(); setattr(args, name, path)
        digest = getattr(args, name + "_sha256")
        if sha256(path) != digest: raise ValueError("freeze input changed: " + name)
        pinned[str(path)] = digest
    source, reference = read_capture(args.corpus), read_capture(args.split_reference)
    split = native.family_split(source, 10601, .25, .25, reference=reference, refresh_training=True)
    if tuple(map(len, split)) != (474, 69, 72): raise ValueError("frozen split changed")
    excluded = {native.board_family(t["board"]) for t in source["targets"]}
    benchmark = json.loads(args.benchmark_protocol.read_text())
    excluded |= {native.board_family(s["board"]) for s in benchmark["spots"]}
    bank = json.loads(args.roots.read_text())
    if (bank.get("schema") != "native-value-public-roots-v1" or bank.get("releaseAccepted") is not False
            or bank.get("seed") != 2026100417 or bank.get("source_preflop_sha256") != args.preflop_sha256
            or len(bank["roots"]) != 128):
        raise ValueError("wrong authentic root bank")
    indices = select_families(bank["roots"], excluded)
    _, students = select_students(args.students, args.students_sha256)
    if args.output.exists(): raise ValueError("never overwrite a frozen TRAIN protocol")
    args.output = args.output.resolve(); args.output.mkdir()
    families = []
    for offset, index in enumerate(indices):
        root = bank["roots"][index]; path = args.output / f"root-{100+offset}.json"
        atomic_json(path, root); pinned[str(path)] = sha256(path)
        seed = 100101 + offset % 2; model = students[seed]
        pinned[str(Path(model["path"]).resolve())] = model["sha256"]
        families.append(dict(root=100+offset, originalRootIndex=index, role="TRAIN",
            family=list(native.board_family(root["public"]["board"])), rootPath=str(path), rootSha256=sha256(path),
            solverSeed=seed, sampleSeed=2026100400+100+offset, modelPath=model["path"], modelSha256=model["sha256"]))
    plan = dict(schema="new-training-coverage-protocol-v1", status="complete", releaseAccepted=False,
        corpusSha256=args.corpus_sha256, splitReferenceSha256=args.split_reference_sha256,
        binaryPath=str(args.binary), pinnedInputs=pinned, families=families,
        excludedFamilies=[list(f) for f in sorted(excluded)],
        selection="first two unpaired Q-or-higher rainbow and first two-tone five-leaf three-bet families in bank order; scores blind",
        proposer="rejected C1 pair used for exploratory search inputs only, never active strategy",
        maximumLabelingSeconds=5400, maximumFinalizationSeconds=180, maximumWorkers=2,
        maximumWorkerMemoryBytes=2*1024**3, maximumNewArtifactBytes=10*1024**3,
        minimumFreeDiskBytes=20*1024**3, flopIterations=128, nativeIterations=64,
        searchStatesPerFamily=16, publicTurnCount=49, sentinelNativeIterations=256,
        calibrationStatesPerFamily=261, primarySplitUnchanged=True)
    path = args.output / "protocol.json"; atomic_json(path, plan)
    print(json.dumps(dict(protocol=str(path), sha256=sha256(path), families=families)), flush=True)


def quality(prefix, packets, high, root):
    full, _ = build_groups(prefix, packets, require_full_chance=True)
    same = [p for p in packets if p["turn"] == high["turn"]]
    low, _ = build_groups(prefix, same); stronger, _ = build_groups(prefix, [high])
    hybrid, _ = build_groups(prefix, [high if p["turn"] == high["turn"] else p for p in packets], require_full_chance=True)
    return dict(schema="full-chance-action-bundle-quality-v1", root=root, all49Turns=True,
        nativeBudget=64, releaseAccepted=False, groups=[dict(history=g.history, support=g.report(),
            sentinel64To256=compare(a, b), oneSentinelUpgradeEffectOnAll49=compare(g, h))
            for g, a, b, h in zip(full, low, stronger, hybrid, strict=True)])


def collect(args):
    receipt = dict(path=str(args.protocol.resolve()), sha256=args.protocol_sha256)
    plan = verified_json(receipt)
    extension_families(receipt, plan["corpusSha256"], plan["splitReferenceSha256"], set(), set())
    pinned = dict(plan["pinnedInputs"]); pinned[receipt["path"]] = receipt["sha256"]
    for name in (Path(__file__).name, "training_coverage.py", "action_contrast_dataset.py", "native_value_dataset.py",
                 "run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py",
                 "run_postflop_gap_pilot.py", "inspect_action_bundle_quality.py", "join_full_chance_bundles.py"):
        path = Path(__file__).with_name(name).resolve(); pinned[str(path)] = sha256(path)
    if args.output.exists(): raise ValueError("never overwrite coverage labels")
    args.output = args.output.resolve(); args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(plan["maximumLabelingSeconds"], stop.set); timer.daemon=True; timer.start()
    pressure = PilotMemoryGuard(stop, args.output / "system-memory.json").start()
    started = time.monotonic(); binary = Path(plan["binaryPath"])
    record = dict(schema="training-coverage-action-bundle-pilot-v1", status="running", releaseAccepted=False,
        pinnedInputs=pinned, trainingExtension=receipt, families=[], maximumWorkers=2,
        maximumLabelingSeconds=5400, maximumFinalizationSeconds=180, all49Turns=True, nativeBudget=64)
    atomic_json(args.output / "manifest.json", record)

    def check():
        if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
            raise ValueError("resource/operator stop or immutable input drift")
        if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > plan["maximumNewArtifactBytes"]:
            raise ValueError("new artifact budget exceeded")

    def export(candidate, output, test, extra=None):
        check()
        worker = guarded(test_command(binary, BASE+test), dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),
            POKER_NATIVE_FLOP_CANDIDATE_SHA=sha256(candidate), POKER_NATIVE_FLOP_OUTPUT=str(output), **(extra or {})),
            output.parent/(output.stem+"-worker"), 900, 2*1024**3, stop)
        return dict(path=str(output), sha256=sha256(output), worker=worker)

    try:
        selected = []
        for i, family in enumerate(plan["families"]):
            check(); work = args.output/f"root-{family['root']}"; work.mkdir()
            capture, candidate = work/"search.json.gz", work/"proposer.json"
            worker = guarded(test_command(binary, TEST), dict(
                POKER_SEARCH_ROOT=family["rootPath"], POKER_SEARCH_ROOT_SHA=family["rootSha256"],
                POKER_SEARCH_MODEL=family["modelPath"], POKER_SEARCH_MODEL_SHA=family["modelSha256"],
                POKER_SEARCH_OUTPUT=str(capture), POKER_SEARCH_POLICY_OUTPUT=str(candidate),
                POKER_SEARCH_TRUNK_SEED=str(family["solverSeed"]), POKER_SEARCH_SAMPLE_SEED=str(family["sampleSeed"]),
                POKER_SEARCH_LABEL_COUNT="16", POKER_SEARCH_WORKERS="2", POKER_SEARCH_FLOP_ITERATIONS="128",
                POKER_SEARCH_LABEL_TURN_ITERATIONS="64", POKER_SEARCH_VERIFY_POLICY_PARITY="1" if i==0 else "0"),
                work/"search-worker", 900, 2*1024**3, stop)
            source = read_capture(capture)
            if (source["source_public_input_sha256"] != family["rootSha256"]
                    or source["proposal_model_sha256"] != family["modelSha256"]
                    or source["seed"] != family["solverSeed"] or source["sampling_seed"] != family["sampleSeed"]
                    or (i==0 and not source["policy_observation_parity_checked"])):
                raise ValueError("search provenance or observation parity changed")
            row = {**family, "candidatePath":str(candidate), "candidateSha256":sha256(candidate),
                   "extraCapture":dict(path=str(capture), sha256=sha256(capture), worker=worker), "labels":[], "sentinels":[]}
            pinned[str(candidate)] = sha256(candidate)
            row["prefix"] = export(candidate, work/"prefix.json", "saved_action_bundle_prefix")
            prefix = verified_json(row["prefix"])
            if native.board_family(prefix["root"]["board"]) != tuple(family["family"]) or len(prefix["leaves"]) != 5:
                raise ValueError("prefix family/tree changed")
            # Validate the capture before launching complete chance work.
            extra_calibration([], [dict(board=prefix["root"]["board"])], source, row["candidateSha256"])
            turns = sorted(set(range(52))-set(prefix["root"]["board"]))
            selected.append((work, prefix, row, source, turns)); record["families"].append(row)
            atomic_json(args.output/"manifest.json", record)
            print(json.dumps(dict(event="coverage-search-capture", root=family["root"], seconds=worker["workerElapsedSeconds"])), flush=True)

        def label(job, turn, iterations=64):
            work, _, row, _, _ = job
            return export(Path(row["candidatePath"]), work/f"turn-{turn}-{iterations}.json", "saved_action_bundle_turn_labels",
                dict(POKER_NATIVE_FLOP_TURN=str(turn), POKER_ACTION_LABEL_ITERATIONS=str(iterations)))

        with ThreadPoolExecutor(max_workers=2) as pool:
            timings=[]
            for job in selected:
                before=time.monotonic()
                job[2]["labels"] += list(pool.map(lambda t: label(job,t), job[4][:2]))
                timings.append(time.monotonic()-before)
            # Two native64 packets run concurrently; the native256 sentinel
            # runs alone, conservatively budgeted at 4x a full warm batch.
            remaining=1.5*sum(t*(47/2+4) for t in timings)
            record["preflight"] = dict(twoTurnFamilySeconds=timings, projectedRemainingLabelSeconds=remaining)
            atomic_json(args.output/"manifest.json", record)
            print(json.dumps(dict(event="coverage-label-preflight", **record["preflight"])), flush=True)
            if remaining > 5400-(time.monotonic()-started):
                raise ValueError("full chance/sentinel labels exceed declared 90-minute cap")
            for job in selected:
                futures=[pool.submit(label,job,t) for t in job[4][2:]]
                for future in as_completed(futures):
                    try: row=future.result()
                    except Exception: stop.set(); raise
                    job[2]["labels"].append(row)
                    atomic_json(args.output/"manifest.json", record)
                job[2]["sentinels"]=[label(job,job[4][0],256)]
                print(json.dumps(dict(event="coverage-family-labeled", root=job[2]["root"], turns=49)), flush=True)

        check(); record["labelingElapsedSeconds"]=time.monotonic()-started
        timer.cancel(); timer=threading.Timer(180,stop.set); timer.daemon=True; timer.start()
        qualities=[]
        for work, prefix, row, source, _ in selected:
            packets=[verified_json(r) for r in row["labels"]]
            groups, ordered=build_groups(prefix,packets,require_full_chance=True)
            groups, ordered=extra_calibration(groups,ordered,source,row["candidateSha256"])
            corpus=dict(schema=native.SCHEMA,game=prefix["game"],
                source_public_input_sha256=row["rootSha256"],source_policy_sha256=row["candidateSha256"],
                validation=dict(status="research_only"),flop_iterations=128,turn_iterations=64,
                maximum_states=len(ordered),observed_queries=len(ordered),targets=ordered)
            native.validate_dataset(corpus)
            encoded=json.dumps(corpus,separators=(",",":"),allow_nan=False).encode()
            if len(ordered)!=261 or len(encoded)>256*1024**2: raise ValueError("coverage shard exceeds existing guards")
            path=work/"calibration.json.gz"
            with path.open("xb") as file: file.write(gzip.compress(encoded,mtime=0))
            if path.stat().st_size>128*1024**2: raise ValueError("compressed coverage guard exceeded")
            row.update(all49Turns=True,groups=[g.report() for g in groups],calibrationCorpus=dict(path=str(path),sha256=sha256(path),states=261))
            q=quality(prefix,packets,verified_json(row["sentinels"][0]),row["root"]); qualities.append(q)
            atomic_json(work/"quality.json",q); row["quality"]=dict(path=str(work/"quality.json"),sha256=sha256(work/"quality.json"))
        check(); record["status"]="complete"
        atomic_json(args.output/"manifest.json",record)
        # Same finite-teacher rule, with explicitly declared new root IDs.
        decision=data_decision(qualities,sha256(args.output/"manifest.json"), expected_roots=[100,101,102])
        atomic_json(args.output/"decision.json",decision)
    except Exception as error:
        record["status"]="failed"; record["failure"]=str(error); stop.set(); raise
    finally:
        timer.cancel(); record["elapsedSeconds"]=time.monotonic()-started
        record["systemMemoryGuard"]=pressure.finish(); atomic_json(args.output/"manifest.json",record)
    # Final manifest includes resource receipts; pin that final identity.
    decision["bundleManifestSha256"]=decision["primaryTrainingManifestSha256"]=sha256(args.output/"manifest.json")
    atomic_json(args.output/"decision.json",decision)
    print(json.dumps(dict(status=record["status"],seconds=record["elapsedSeconds"],decision=decision)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="mode",required=True)
    f=sub.add_parser("freeze")
    for name in ("binary","corpus","split-reference","roots","students","benchmark-protocol","preflop"):
        f.add_argument("--"+name,type=Path,required=True); f.add_argument("--"+name+"-sha256",required=True)
    c=sub.add_parser("collect"); c.add_argument("--protocol",type=Path,required=True); c.add_argument("--protocol-sha256",required=True)
    for parser in (f,c): parser.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); freeze(a) if a.mode=="freeze" else collect(a)

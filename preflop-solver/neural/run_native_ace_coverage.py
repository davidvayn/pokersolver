"""Bounded six-family ace-high coverage test; append 96 native labels, no fitting.

Roots come from authentic deals and frozen preflop play, then are explicitly
conditioned on ace-high boards. This is targeted training, not an authentic
evaluation distribution. Existing corpus targets and held-out families stay fixed.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import signal
import shutil
import threading
import time

import native_value_dataset as native
from run_native_value_pilot import ROOT_TEST, guarded, merge_captures, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_search_distribution_pilot import TEST


def select_ace_families(roots, excluded, count=6):
    seen = {native.board_family(board) for board in excluded}
    selected = []
    for index, root in enumerate(roots):
        board = root["public"]["board"]
        family = native.board_family(board)
        if max(int(card)//4 for card in board) != 12 or family in seen:
            continue
        seen.add(family)
        selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError("insufficient unused ace-high families; do not pad or reuse holdouts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("binary", "reference", "split-reference", "students", "response", "roots", "preflop")
    for name in names:
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--exclude-root", type=Path, action="append", required=True)
    parser.add_argument("--exclude-root-sha256", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pinned = {}
    for key in [name.replace("-", "_") for name in names]:
        path = getattr(args,key).resolve()
        setattr(args,key,path)
        if sha256(path) != getattr(args,key+"_sha256"):
            raise ValueError("pinned input mismatch: "+key)
        pinned[str(path)] = sha256(path)
    if len(args.exclude_root) != len(args.exclude_root_sha256):
        raise ValueError("each excluded evaluation root needs its own hash")
    excluded = []
    for path, expected in zip(args.exclude_root,args.exclude_root_sha256):
        path = path.resolve()
        if sha256(path) != expected: raise ValueError("excluded root changed")
        pinned[str(path)] = expected
        excluded.append(json.loads(path.read_text())["public"]["board"])
    reference, students, response = [json.loads(p.read_text()) for p in
                                    (args.reference,args.students,args.response)]
    if any(m.get("status") != "complete" for m in (reference,students,response)):
        raise ValueError("completed source, students and response required")
    if (len(students["predictions"]) != 2 or len(response["profiles"]) != 2
            or response.get("flopIterations") != 128
            or response["pinnedInputs"].get(str(args.students)) != args.students_sha256):
        raise ValueError("preceding response did not evaluate the proposer pair")
    corpus_path = args.reference.parent/"corpus.json.gz"
    if sha256(corpus_path) != reference["analysis"]["corpusSha256"]:
        raise ValueError("reference corpus changed")
    pinned[str(corpus_path)] = sha256(corpus_path)
    corpus, split_reference = read_capture(corpus_path),read_capture(args.split_reference)
    if len(corpus["targets"]) != 412: raise ValueError("expected 412-state reference")
    split = native.family_split(corpus,10601,.25,.25,reference=split_reference,refresh_training=True)
    excluded.extend(t["board"][:3] for t in corpus["targets"])
    old_roots = json.loads(args.roots.read_text())
    if old_roots["source_preflop_sha256"] != args.preflop_sha256:
        raise ValueError("root proposer changed")
    models = {}
    for entry in students["predictions"]:
        path = Path(entry["model"])
        if sha256(path) != entry["modelSha256"] or entry["maximumParityErrorBb"] > 1e-4:
            raise ValueError("unqualified proposer")
        models[entry["seed"]] = path
        pinned[str(path)] = entry["modelSha256"]
    if set(models) != {10601,10602}: raise ValueError("unexpected proposer seeds")
    for name in ("native_value_dataset.py","run_native_value_pilot.py","run_native_value_preflight.py","worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    timer = threading.Timer(7200,stop.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    record = dict(schema="native-targeted-ace-coverage-v1",status="running",releaseAccepted=False,
                  pinnedInputs=pinned,runnerSha256=sha256(Path(__file__)),workers=[],
                  nativeWorkers=2,additionalStates=96,maximumStageSeconds=7200,
                  interpretation="Six unused ace-high families selected from authentic preflop deals; targeted training only, exact old targets and held-out split retained.")
    atomic_json(args.output/"manifest.json",record)
    try:
        root_output = args.output/"roots.json"
        record["rootWorker"] = guarded(test_command(args.binary,ROOT_TEST),{
            "POKER_NATIVE_CHECKPOINT":str(args.preflop),"POKER_NATIVE_CHECKPOINT_SHA":args.preflop_sha256,
            "POKER_NATIVE_ROOTS_OUTPUT":str(root_output),"POKER_NATIVE_ROOTS_COUNT":"128",
            "POKER_NATIVE_ROOTS_SEED":str(old_roots["seed"])},args.output/"root-worker",180,stop=stop)
        roots = json.loads(root_output.read_text())["roots"]
        if roots[:len(old_roots["roots"])] != old_roots["roots"]:
            raise ValueError("authentic root prefix changed")
        selected = select_ace_families(roots,excluded)
        if any(not 1 <= roots[i]["turn_leaf_count"] <= 13 for i in selected):
            raise ValueError("16-label quota cannot cover these roots; do not silently exclude them")
        # Run the largest selected tree first as the resource/provenance preflight.
        selected.sort(key=lambda i:(-roots[i]["turn_leaf_count"],i))
        record.update(rootsSha256=sha256(root_output),selectedRootIndices=selected,originalRootPrefixParity=True)
        captures = []
        for offset,index in enumerate(selected):
            if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
                raise ValueError("stopped stage or pinned input changed")
            root,output,policy = [args.output/("capture-%03d"%index+suffix)
                                  for suffix in ("-root.json",".json.gz","-policy.json")]
            atomic_json(root,roots[index])
            model = models[10601+index%2]
            env = {"POKER_SEARCH_ROOT":str(root),"POKER_SEARCH_ROOT_SHA":sha256(root),
                   "POKER_SEARCH_MODEL":str(model),"POKER_SEARCH_MODEL_SHA":pinned[str(model)],
                   "POKER_SEARCH_OUTPUT":str(output),"POKER_SEARCH_POLICY_OUTPUT":str(policy),
                   "POKER_SEARCH_TRUNK_SEED":str(100101+index%2),"POKER_SEARCH_SAMPLE_SEED":str(772801+index),
                   "POKER_SEARCH_LABEL_COUNT":"16","POKER_SEARCH_WORKERS":"2",
                   "POKER_SEARCH_FLOP_ITERATIONS":"128","POKER_SEARCH_VERIFY_POLICY_PARITY":"1" if offset==0 else "0"}
            worker = guarded(test_command(args.binary,TEST),env,args.output/("worker-%03d"%index),900,2*1024**3,stop)
            source = read_capture(output)
            expected = {"source_public_input_sha256":sha256(root),"source_policy_sha256":sha256(policy),
                        "proposal_model_sha256":pinned[str(model)],"seed":100101+index%2,
                        "sampling_seed":772801+index,"flop_iterations":128,"turn_iterations":64,"native_label_queries":16}
            if (any(source.get(k)!=v for k,v in expected.items()) or len(source["targets"])!=16
                    or (offset==0 and source.get("policy_observation_parity_checked") is not True)):
                raise ValueError("targeted capture provenance mismatch")
            captures.append(output)
            record["workers"].append(dict(root=index,path=str(output),sha256=sha256(output),worker=worker))
            if offset==0:
                projected_compressed = corpus_path.stat().st_size + 6*output.stat().st_size
                projected_decoded = reference["analysis"]["decodedBytes"] + 6*len(json.dumps(source).encode())
                # A conservative extra copy for captures plus the merged output.
                projected_new_disk = 2*projected_compressed
                if (projected_compressed > 128*1024**2 or projected_decoded > 256*1024**2
                        or shutil.disk_usage(args.output).free < 20*1024**3 + projected_new_disk):
                    raise ValueError("preflight projects beyond corpus/storage budget")
                record.update(preflightPassed=True,preflightProjection=dict(
                    compressedBytes=projected_compressed,decodedBytes=projected_decoded,
                    additionalDiskBytes=projected_new_disk,
                    sixCaptureSeconds=6*worker["workerElapsedSeconds"]))
            atomic_json(args.output/"manifest.json",record)
            print(json.dumps(dict(event="ace-coverage-labels",families=offset+1,total=6,seconds=worker["workerElapsedSeconds"])),flush=True)
        merged = merge_captures(captures,search_proposals=True,reference=corpus)
        actual = native.family_split(merged,10601,.25,.25,reference=split_reference,refresh_training=True)
        if (merged["targets"][:412] != corpus["targets"] or len(merged["targets"])!=508
                or any(actual[i].tolist()!=split[i].tolist() for i in (1,2))
                or len(actual[0])!=367):
            raise ValueError("prefix, held-out split or state budget changed")
        encoded = json.dumps(merged,separators=(",",":"),allow_nan=False).encode()
        if len(encoded)>256*1024**2: raise ValueError("native decoded corpus limit exceeded")
        output = args.output/"corpus.json.gz"
        compressed = gzip.compress(encoded,mtime=0)
        if len(compressed)>128*1024**2: raise ValueError("native compressed corpus limit exceeded")
        with output.open("xb") as stream: stream.write(compressed)
        if stop.is_set() or any(sha256(Path(p))!=h for p,h in pinned.items()):
            raise ValueError("stopped stage or pinned input changed")
        record.update(status="complete",analysis=dict(states=508,corpusSha256=sha256(output),
            compressedBytes=len(compressed),decodedBytes=len(encoded),exactOldTargetParity=True,
            split={k:v.tolist() for k,v in zip(("train","tuning","holdout"),actual)},
            maximumWorkerMemoryBytes=max(w["worker"]["sampledPeakMemoryBytes"] for w in record["workers"])))
    except (OSError,ValueError,KeyError) as error:
        stop.set()
        record["status"],record["failure"] = "failed",str(error)
    finally:
        timer.cancel()
        record["elapsedSeconds"] = time.monotonic()-started
        atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","failure")}),flush=True)
    if record["status"]!="complete": raise SystemExit(1)


if __name__ == "__main__": main()

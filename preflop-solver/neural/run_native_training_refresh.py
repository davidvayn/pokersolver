"""Refresh only training-family beliefs with a frozen 128-update proposer pair.

Same 412-state budget and exact held-out targets. Native 64-update solves supply
all fresh values; proposer predictions are inputs, never supervision.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import signal
import threading
import time

import native_value_dataset as native
from run_native_value_pilot import guarded, merge_captures, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_search_distribution_pilot import TEST


def capture_training_flags(corpus, train):
    train = set(map(int, train))
    flags, offset = [], 0
    for index, capture in enumerate(corpus["source_captures"]):
        indices = list(range(offset, offset + capture["states"]))
        if any(corpus["targets"][i]["source_capture_index"] != index for i in indices):
            raise ValueError("capture order changed")
        membership = {i in train for i in indices}
        if len(membership) != 1: raise ValueError("one capture crosses a held-out split")
        flags.append(membership.pop())
        offset += capture["states"]
    if offset != len(corpus["targets"]): raise ValueError("incomplete source capture counts")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "original", "reference", "split-reference", "students", "response", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
        if key != "output": parser.add_argument("--"+key+"-sha256", required=True)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    pinned = {}
    for key in ("binary", "original", "reference", "split_reference", "students", "response", "output"):
        path = getattr(args,key).resolve()
        setattr(args,key,path)
        if key != "output":
            if sha256(path) != getattr(args,key+"_sha256"): raise ValueError("pinned input changed: "+key)
            pinned[str(path)] = sha256(path)
    original, reference, students, response = [json.loads(p.read_text()) for p in
                                              (args.original,args.reference,args.students,args.response)]
    if any(m.get("status") != "complete" for m in (original,reference,students,response)):
        raise ValueError("completed reference corpus, proposers and response comparison required")
    if (response.get("flopIterations") != 128 or len(response["profiles"]) != 2
            or len(students["predictions"]) != 2
            or response["pinnedInputs"].get(str(args.students)) != args.students_sha256):
        raise ValueError("reference did not evaluate this frozen proposer pair at 128 updates")
    corpus_path, roots_path = args.reference.parent/"corpus.json.gz", args.reference.parent/"roots.json"
    for path, digest in ((corpus_path,reference["analysis"]["corpusSha256"]),
                         (roots_path,reference["rootsSha256"])):
        if sha256(path) != digest: raise ValueError("reference artifact changed")
        pinned[str(path)] = digest
    corpus, split_reference = read_capture(corpus_path), read_capture(args.split_reference)
    if len(corpus["targets"]) != 412: raise ValueError("expected fixed 412-state budget")
    split = native.family_split(corpus,10601,.25,.25,reference=split_reference)
    flags = capture_training_flags(corpus,split[0])
    roots = json.loads(roots_path.read_text())["roots"]
    indices = list(range(8)) + reference["selectedRootIndices"]
    if len(indices) != len(flags): raise ValueError("source family count changed")
    paths = {w["sha256"]:Path(w["path"]) for m in (original,reference) for w in m["workers"]}
    old_paths = []
    for index, capture in enumerate(corpus["source_captures"]):
        path = paths[capture["capture_sha256"]]
        if sha256(path) != capture["capture_sha256"]: raise ValueError("source capture changed")
        pinned[str(path)] = capture["capture_sha256"]
        old_paths.append(path)
        if any(native.board_family(t["board"]) != native.board_family(roots[indices[index]]["public"]["board"])
               for t in read_capture(path)["targets"]): raise ValueError("source root family mismatch")
    models = []
    for p in students["predictions"]:
        path = Path(p["model"])
        if sha256(path) != p["modelSha256"] or p["maximumParityErrorBb"] > 1e-4:
            raise ValueError("unqualified proposer inference")
        pinned[str(path)] = p["modelSha256"]
        models.append(path)
    for name in ("native_value_dataset.py","run_native_value_pilot.py","run_native_value_preflight.py","worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    if args.output.exists(): raise ValueError("refusing to overwrite a refresh")
    args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    timer = threading.Timer(7200,stop.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    record = dict(schema="native-current-search-training-refresh-v1",status="running",releaseAccepted=False,
                  pinnedInputs=pinned,runnerSha256=sha256(Path(__file__)),workers=[],
                  flopIterations=128,turnIterations=64,nativeWorkers=args.workers,maximumStageSeconds=7200,
                  refreshedTrainingStates=len(split[0]),unchangedHeldoutStates=len(split[1])+len(split[2]),
                  interpretation="Current search beliefs on existing training families only; fixed count, exact held-out targets; no activation.")
    atomic_json(args.output/"manifest.json",record)

    def capture(index,count,name,parity=False):
        if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
            raise ValueError("stopped stage or changed pinned input")
        root_index = indices[index]
        root, output, policy = [args.output/(name+suffix) for suffix in ("-root.json",".json.gz","-policy.json")]
        atomic_json(root,roots[root_index])
        model = models[root_index%2]
        env = {"POKER_SEARCH_ROOT":str(root),"POKER_SEARCH_ROOT_SHA":sha256(root),
               "POKER_SEARCH_MODEL":str(model),"POKER_SEARCH_MODEL_SHA":pinned[str(model)],
               "POKER_SEARCH_OUTPUT":str(output),"POKER_SEARCH_POLICY_OUTPUT":str(policy),
               "POKER_SEARCH_TRUNK_SEED":str(100101+root_index%2),"POKER_SEARCH_SAMPLE_SEED":str(772801+root_index),
               "POKER_SEARCH_LABEL_COUNT":str(count),"POKER_SEARCH_WORKERS":str(args.workers),
               "POKER_SEARCH_FLOP_ITERATIONS":"128","POKER_SEARCH_VERIFY_POLICY_PARITY":"1" if parity else "0"}
        worker = guarded(test_command(args.binary,TEST),env,args.output/(name+"-worker"),900,2*1024**3,stop)
        source = read_capture(output)
        expected = {"source_public_input_sha256":sha256(root),"source_policy_sha256":sha256(policy),
                    "proposal_model_sha256":pinned[str(model)],"seed":100101+root_index%2,
                    "sampling_seed":772801+root_index,"flop_iterations":128,"turn_iterations":64,
                    "native_label_queries":count}
        if (any(source.get(k) != v for k,v in expected.items()) or len(source["targets"]) != count
                or (parity and source.get("policy_observation_parity_checked") is not True)):
            raise ValueError("refreshed capture provenance drift")
        record["workers"].append(dict(root=root_index,path=str(output),sha256=sha256(output),preflight=parity,worker=worker))
        atomic_json(args.output/"manifest.json",record)
        print(json.dumps(dict(event="native-current-search-labels",root=root_index,states=count,preflight=parity,
                              seconds=worker["workerElapsedSeconds"])),flush=True)
        return output

    try:
        eligible = [i for i,flag in enumerate(flags) if flag]
        # Exercise the most complex training root before collecting full counts.
        first = max(eligible,key=lambda i:roots[indices[i]]["turn_leaf_count"])
        capture(first,16,"preflight",True)
        record["preflightPassed"] = True
        captures = [capture(i,corpus["source_captures"][i]["states"],"capture-%02d"%i) if flag else old_paths[i]
                    for i,flag in enumerate(flags)]
        refreshed = merge_captures(captures,search_proposals=True)
        actual = native.family_split(refreshed,10601,.25,.25,reference=split_reference,refresh_training=True)
        if len(refreshed["targets"]) != 412 or any(a.tolist()!=b.tolist() for a,b in zip(actual,split)):
            raise ValueError("refresh changed the split or label budget")
        for i in list(split[1])+list(split[2]):
            if refreshed["targets"][i] != corpus["targets"][i]: raise ValueError("held-out target changed")
        encoded = json.dumps(refreshed,separators=(",",":"),allow_nan=False).encode()
        output = args.output/"corpus.json.gz"
        with output.open("xb") as stream: stream.write(gzip.compress(encoded,mtime=0))
        if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
            raise ValueError("stopped stage or changed pinned input")
        record["status"] = "complete"
        record["analysis"] = dict(states=412,corpusSha256=sha256(output),compressedBytes=output.stat().st_size,
                                  decodedBytes=len(encoded),exactHeldoutParity=True,
                                  split={k:v.tolist() for k,v in zip(("train","tuning","holdout"),actual)},
                                  maximumWorkerMemoryBytes=max(w["worker"]["sampledPeakMemoryBytes"] for w in record["workers"]))
    except (OSError,ValueError,KeyError) as error:
        stop.set()
        record["status"],record["failure"] = "failed",str(error)
    finally:
        timer.cancel()
        record["elapsedSeconds"] = time.monotonic()-started
        atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status","elapsedSeconds","failure")}),flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()

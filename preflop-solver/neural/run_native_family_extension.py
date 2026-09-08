"""Add eight training-only flop families while retaining every old target/split.

Sequential, bounded 16-label captures. Uses the same frozen search proposers as
the reference corpus; predictions never become labels. No fitting or activation.
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
from run_native_value_pilot import ROOT_TEST, guarded, merge_captures, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_search_distribution_pilot import TEST as CAPTURE_TEST


def choose_new_families(roots, reference_roots, count=8):
    if roots[:len(reference_roots)] != reference_roots:
        raise ValueError("authentic root prefix changed")
    seen = {native.board_family(r["public"]["board"]) for r in reference_roots}
    selected = []
    for index in range(len(reference_roots), len(roots)):
        family = native.board_family(roots[index]["public"]["board"])
        if family in seen:
            continue
        seen.add(family)
        selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError("insufficient new authentic flop families")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "reference", "baseline", "proposers", "response", "preflop", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
        if name != "output":
            parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    pinned = {}
    for name in ("binary", "reference", "baseline", "proposers", "response", "preflop", "output"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        if name != "output":
            expected = getattr(args, name + "_sha256")
            if sha256(path) != expected:
                raise ValueError("pinned input mismatch: " + name)
            pinned[str(path)] = expected
    reference_record, baseline, proposers, response = (
        json.loads(path.read_text()) for path in (args.reference, args.baseline, args.proposers, args.response)
    )
    if any(m.get("status") != "complete" for m in (reference_record, baseline, proposers, response)):
        raise ValueError("complete reference, baseline, proposers and preceding paired evaluation required")
    if (len(response["profiles"]) != 2 or len(proposers["predictions"]) != 2
            or reference_record["pinnedInputs"].get(str(args.proposers)) != args.proposers_sha256
            or reference_record["pinnedInputs"].get(str(args.baseline)) != args.baseline_sha256
            or baseline["analysis"]["preflopSha256"] != args.preflop_sha256):
        raise ValueError("reference proposer/preflop identity or paired evaluation mismatch")
    reference_path = args.reference.parent / "corpus.json.gz"
    roots_reference_path = args.baseline.parent / "roots.json"
    for path, expected in ((reference_path, reference_record["analysis"]["corpusSha256"]),
                           (roots_reference_path, baseline["analysis"]["rootsSha256"])):
        if sha256(path) != expected:
            raise ValueError("completed source artifact changed")
        pinned[str(path)] = expected
    reference = read_capture(reference_path)
    roots_reference = json.loads(roots_reference_path.read_text())
    if len(reference["targets"]) != 284 or len(roots_reference["roots"]) != 8:
        raise ValueError("expected the pinned 284-state/eight-family reference")
    captures = []
    for index, source in enumerate(reference["source_captures"]):
        matches = [r for r in reference_record["workers"] if r["root"] == index and r["states"] == source["states"]]
        if len(matches) != 1:
            raise ValueError("ambiguous reference capture")
        path = Path(matches[0]["path"])
        if sha256(path) != source["capture_sha256"]:
            raise ValueError("reference capture changed")
        pinned[str(path)] = source["capture_sha256"]
        captures.append(path)
    models = []
    for entry in proposers["predictions"]:
        path = Path(entry["model"])
        if sha256(path) != entry["modelSha256"] or entry["maximumParityErrorBb"] > 1e-4:
            raise ValueError("proposer qualification changed")
        models.append(path)
        pinned[str(path)] = entry["modelSha256"]
    for name in ("native_value_dataset.py", "run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    if args.output.exists():
        raise ValueError("refusing to overwrite a family extension")
    args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    record = dict(schema="native-training-family-extension-v1", status="running", releaseAccepted=False,
                  pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)), workers=[],
                  additionalStates=128, nativeWorkers=2, maximumStageSeconds=7200,
                  splitReferenceCorpus=str(reference_path), splitReferenceSha256=sha256(reference_path),
                  interpretation="Eight new training-only public flop families; old tuning/holdout and labels unchanged; not full-game qualification.")
    atomic_json(args.output / "manifest.json", record)
    try:
        root_output = args.output / "roots.json"
        record["rootWorker"] = guarded(test_command(args.binary, ROOT_TEST),
            {"POKER_NATIVE_CHECKPOINT": str(args.preflop), "POKER_NATIVE_CHECKPOINT_SHA": args.preflop_sha256,
             "POKER_NATIVE_ROOTS_OUTPUT": str(root_output), "POKER_NATIVE_ROOTS_SEED": str(roots_reference["seed"]),
             "POKER_NATIVE_ROOTS_COUNT": "32"}, args.output / "root-worker", 180, stop=stop)
        roots = json.loads(root_output.read_text())["roots"]
        selected = choose_new_families(roots, roots_reference["roots"])
        record.update(rootsSha256=sha256(root_output), originalRootPrefixParity=True, selectedRootIndices=selected)
        atomic_json(args.output / "manifest.json", record)
        for offset, index in enumerate(selected):
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
                raise ValueError("stopped stage or changed pinned input")
            root = args.output / ("root-%02d.json" % index)
            atomic_json(root, roots[index])
            model = models[index % 2]
            target = args.output / ("capture-%02d.json.gz" % index)
            policy = args.output / ("proposer-%02d.json" % index)
            seed, sample_seed = 100101 + index % 2, 772801 + index
            env = {"POKER_SEARCH_ROOT": str(root), "POKER_SEARCH_ROOT_SHA": sha256(root),
                   "POKER_SEARCH_MODEL": str(model), "POKER_SEARCH_MODEL_SHA": pinned[str(model)],
                   "POKER_SEARCH_OUTPUT": str(target), "POKER_SEARCH_POLICY_OUTPUT": str(policy),
                   "POKER_SEARCH_TRUNK_SEED": str(seed), "POKER_SEARCH_SAMPLE_SEED": str(sample_seed),
                   "POKER_SEARCH_LABEL_COUNT": "16", "POKER_SEARCH_WORKERS": "2",
                   "POKER_SEARCH_VERIFY_POLICY_PARITY": "1" if offset == 0 else "0"}
            worker = guarded(test_command(args.binary, CAPTURE_TEST), env,
                             args.output / ("worker-%02d" % index), 900, 2 * 1024**3, stop)
            source = read_capture(target)
            if (len(source["targets"]) != 16 or source["native_label_queries"] != 16
                    or source["source_public_input_sha256"] != sha256(root)
                    or source["source_policy_sha256"] != sha256(policy)
                    or source["proposal_model_sha256"] != pinned[str(model)]
                    or source["seed"] != seed or source["sampling_seed"] != sample_seed
                    or source["flop_iterations"] != 32 or source["turn_iterations"] != 64
                    or (offset == 0 and not source["policy_observation_parity_checked"])):
                raise ValueError("extension label provenance/configuration mismatch")
            captures.append(target)
            record["workers"].append(dict(root=index, path=str(target), sha256=sha256(target), worker=worker))
            if offset == 0:
                record["preflightPassed"] = True
            atomic_json(args.output / "manifest.json", record)
            print(json.dumps(dict(event="native-family-extension", families=offset+1, total=8,
                                  states=16, preflight=offset == 0, seconds=worker["workerElapsedSeconds"])), flush=True)
        corpus = merge_captures(captures, search_proposals=True)
        split = native.family_split(corpus, 10601, .25, .25, reference=reference)
        if len(corpus["targets"]) != 412 or [len(s) for s in split] != [271, 69, 72]:
            raise ValueError("extension changed reference split sizes or labeling budget")
        output = args.output / "corpus.json.gz"
        encoded = json.dumps(corpus, separators=(",", ":"), allow_nan=False).encode()
        with output.open("xb") as stream:
            stream.write(gzip.compress(encoded, mtime=0))
        if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
            raise ValueError("stopped stage or changed pinned input")
        record["status"] = "complete"
        record["analysis"] = dict(states=412, corpusSha256=sha256(output), compressedBytes=output.stat().st_size,
                                  decodedBytes=len(encoded), split={k: v.tolist() for k,v in zip(("train","tuning","holdout"),split)},
                                  maximumWorkerMemoryBytes=max(w["worker"]["sampledPeakMemoryBytes"] for w in record["workers"]))
    except (OSError, ValueError, KeyError) as error:
        stop.set()
        record["status"], record["failure"] = "failed", str(error)
    finally:
        timer.cancel()
        record["elapsedSeconds"] = time.monotonic() - started
        atomic_json(args.output / "manifest.json", record)
    print(json.dumps({k: record.get(k) for k in ("status", "elapsedSeconds", "failure")}), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

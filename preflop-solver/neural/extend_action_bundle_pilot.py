"""One predeclared disjoint-eight-turn TRAIN sensitivity block; reuse all labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import signal
import threading
import time

from run_action_bundle_pilot import BASE
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.source, a.output = a.source.resolve(), a.output.resolve()
    if sha256(a.source) != a.source_sha256 or a.output.exists(): raise ValueError("changed source or existing output")
    original = json.loads(a.source.read_text())
    if original.get("status") != "complete" or original.get("calibrationOnly"):
        raise ValueError("complete initial TRAIN bundle stage required")
    candidates = {str(Path(f["candidatePath"])): f["candidateSha256"] for f in original["families"]}
    binaries = [path for path in original["pinnedInputs"] if "local-compact-binaries/" in path]
    if len(binaries) != 1: raise ValueError("ambiguous native binary")
    binary = Path(binaries[0]); pinned = {str(binary): original["pinnedInputs"][str(binary)], **candidates,
                                        str(a.source): a.source_sha256}
    for name in ("extend_action_bundle_pilot.py", "run_native_value_pilot.py", "worker_resources.py"):
        path = Path(__file__).with_name(name); pinned[str(path)] = sha256(path)
    remaining = 5400 - original["elapsedSeconds"]
    if remaining < 1200: raise ValueError("not enough of original label budget remains for sensitivity")
    a.output.mkdir(); stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(remaining, stop.set); timer.daemon=True; timer.start()
    memory = PilotMemoryGuard(stop, a.output/"system-memory.json").start(); started=time.monotonic()
    record = {**copy.deepcopy(original), "status":"running", "extendedFromSha256":a.source_sha256,
              "extensionPinnedInputs":pinned, "extensionMaximumSeconds":remaining,
              "extensionKind":"complete_reserved_disjoint_eight_turn_training_sensitivity"}
    atomic_json(a.output/"manifest.json",record)

    def label(job):
        family, turn = job
        if stop.is_set() or any(sha256(Path(path)) != digest for path,digest in pinned.items()):
            raise ValueError("stopped or immutable identity changed")
        work = a.output/f"root-{family['root']}"; work.mkdir(exist_ok=True)
        output = work/f"turn-{turn}-64.json"
        worker = guarded(test_command(binary, BASE+"saved_action_bundle_turn_labels"), dict(
            POKER_NATIVE_FLOP_CANDIDATE=family["candidatePath"], POKER_NATIVE_FLOP_CANDIDATE_SHA=family["candidateSha256"],
            POKER_NATIVE_FLOP_OUTPUT=str(output), POKER_NATIVE_FLOP_TURN=str(turn), POKER_ACTION_LABEL_ITERATIONS="64"),
            work/(output.stem+"-worker"),900,2*1024**3,stop)
        return family["root"],dict(path=str(output),sha256=sha256(output),worker=worker)
    try:
        jobs=[]
        for family in record["families"]:
            existing={json.loads(Path(r["path"]).read_text())["turn"] for r in family["sentinels"]
                      if json.loads(Path(r["path"]).read_text())["turn_iterations"]==64}
            if len(existing)!=1 or len(family["sensitivityTurns"])!=8:
                raise ValueError("initial reserved sensitivity identity changed")
            jobs += [(family,t) for t in family["sensitivityTurns"] if t not in existing]
        if len(jobs)!=21: raise ValueError("one seven-turn-per-family extension required")
        with ThreadPoolExecutor(max_workers=2) as pool:
            for root, receipt in pool.map(label,jobs):
                next(f for f in record["families"] if f["root"]==root)["sentinels"].append(receipt)
                atomic_json(a.output/"manifest.json",record)
                print(json.dumps(dict(event="bundle-sensitivity-turn",root=root,path=receipt["path"])),flush=True)
        if stop.is_set(): raise ValueError("extension resource stop")
        record["status"]="complete"
    except Exception as error:
        record["status"]="failed";record["failure"]=str(error);stop.set();raise
    finally:
        timer.cancel();record["extensionElapsedSeconds"]=time.monotonic()-started
        record["elapsedSeconds"]=original["elapsedSeconds"]+record["extensionElapsedSeconds"]
        record["extensionMemoryGuard"]=memory.finish();atomic_json(a.output/"manifest.json",record)
    print(json.dumps(dict(status=record["status"],combinedSeconds=record["elapsedSeconds"])),flush=True)


if __name__=="__main__":main()

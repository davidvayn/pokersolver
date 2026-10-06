"""Cheap serving-speed check at the website's pinned budget, not a strength eval.

Keeps cold/warm outputs for exact differential comparison and separately
measures whether a ready preflop query waits behind a cold flop sibling.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_native_value_preflight import atomic_json, sha256
from worker_resources import WorkerResourceGuard

ROOT = Path(__file__).resolve().parents[2]
VERSION = "hu-20bb-v102-consensus-continual-resolver-experimental"


def fixtures(version, cross_batch=False):
    preflop = dict(requestId="startup-preflop", stateHash="e" * 64,
                   modelVersion=version, depthBb=20, privateCards=[7, 34],
                   board=[], street="preflop", actor=0, totalPotBb=1.5,
                   stacksBb=[19.5, 19], streetBetsBb=[.5, 1],
                   totalCommittedBb=[.5, 1], lastFullRaiseBb=1,
                   raiseReopened=True, actions=[])
    limp = {**preflop, "street": "flop", "actor": 1, "board": [13, 5, 15],
            "totalPotBb": 2, "stacksBb": [19, 19], "streetBetsBb": [0, 0],
            "totalCommittedBb": [1, 1], "actions": [
                dict(actor=0, street="preflop", kind="call", amountToBb=None),
                dict(actor=1, street="preflop", kind="check", amountToBb=None)]}
    raised = {**limp, "board": [1, 22, 43], "totalPotBb": 5,
              "stacksBb": [17.5, 17.5], "totalCommittedBb": [2.5, 2.5],
              "actions": [
                  dict(actor=0, street="preflop", kind="raise", amountToBb=2.5),
                  dict(actor=1, street="preflop", kind="call", amountToBb=None)]}
    queries = [(name, {**query, "requestId": name}) for name, query in [
        ("startup-preflop", preflop), ("limped-cold", limp), ("limped-warm", limp),
        ("raised-cold", raised), ("raised-warm", raised),
        ("batch-slow-flop", {**limp, "board": [2, 24, 41]}),
        ("batch-ready-preflop", preflop)]]
    if cross_batch:
        queries += [("cross-batch-slow-flop", {**limp, "board": [3, 21, 42],
                                               "requestId": "cross-batch-slow-flop"}),
                    ("cross-batch-ready-preflop", {**preflop,
                                                   "requestId": "cross-batch-ready-preflop"})]
    return queries


def command(binary, manifest, threads):
    runtime = manifest["runtime"]
    model_root = ROOT / "preflop-solver/models/practice"
    args = [str(binary), "practice-policy-server", "--model-version", manifest["version"],
            "--effective-stack-bb", "20"]
    for flag, key in [("--networks", "networks"), ("--range-policy", "rangePolicy"),
                      ("--preflop-action-values", "preflopActionValues"),
                      ("--flop-resolver-value-network", "flopValueNetwork")]:
        args += [flag, str(model_root / runtime["artifactFiles"][key])]
    for flag, key in [("--dcfr-alpha", "positiveRegretExponent"),
                      ("--dcfr-beta", "negativeRegretExponent"),
                      ("--dcfr-gamma", "strategyExponent")]:
        args += [flag, str(runtime["dcfr"][key])]
    profile = runtime["resolver"]
    # Refuse silently incomplete profiles if serving settings change later.
    if profile["riverSafeResolving"] or profile["riverSafeMaxmargin"]:
        raise ValueError("extend the benchmark command for the new safe resolver profile")
    if profile["flopDeploySolvedPolicy"]:
        args += ["--flop-resolver-deploy-solved-policy"]
    for street in ("flop", "turn", "river"):
        args += [f"--{street}-resolver-iterations", str(profile[f"{street}Iterations"]),
                 f"--{street}-resolver-averaging-delay", "0"]
        actor = profile[f"{street}ResolvedActor"]
        if actor is not None:
            args += [f"--{street}-resolver-actor", str(actor)]
        if street != "river":
            args += [f"--{street}-resolver-threads", str(threads)]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new receipt directory")
    parser.add_argument("--binary", type=Path, default=ROOT / "preflop-solver/target/release/preflop-solver")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--baseline", type=Path, help="require identical outputs to this prior receipt")
    parser.add_argument("--cross-batch", action="store_true", help="include a later ready batch behind a cold one")
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("threads must be between 1 and 8")
    manifest_path = ROOT / "data/practice/full-hand-manifests.json"
    manifest = next(m for m in json.loads(manifest_path.read_text()) if m["version"] == VERSION)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    queries = fixtures(manifest["version"], args.cross_batch)
    launch = command(args.binary.resolve(), manifest, args.threads)
    record = dict(schema="practice-serving-latency-v1", status="running",
                  binarySha256=sha256(args.binary), manifestSha256=sha256(manifest_path),
                  command=launch, rows=[], queries=[query for _, query in queries])
    with (output / "stderr.log").open("x") as error_log:
        process = subprocess.Popen(launch, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=error_log, text=True, cwd=ROOT, start_new_session=True)
        guard = WorkerResourceGuard(process, output, max_memory_bytes=2 * 1024**3,
                                    max_seconds=120, minimum_free_disk_bytes=20 * 1024**3).start()
        record["pid"] = process.pid
        seen = set()

        def save(value, elapsed):
            name = value.get("requestId")
            if value.get("error") or name not in dict(queries) or name in seen:
                raise ValueError(f"resolver returned an invalid response: {value}")
            if abs(sum(action["probability"] for action in value["actions"]) - 1) > 1e-6:
                raise ValueError("invalid probability sum")
            if args.baseline and value != json.loads((args.baseline / f"{name}.json").read_text()):
                raise ValueError(f"policy/EV parity failed for {name}")
            seen.add(name)
            atomic_json(output / f"{name}.json", value)
            row = dict(name=name, seconds=elapsed, resultSha256=sha256(output / f"{name}.json"))
            record["rows"].append(row)
            atomic_json(output / "manifest.json", record)
            print(json.dumps(row), flush=True)

        def read():
            line = process.stdout.readline()
            if not line:
                raise ValueError("resolver exited without completing the receipt")
            return json.loads(line)

        try:
            atomic_json(output / "manifest.json", record)
            for _, query in queries[:5]:
                started = time.perf_counter()
                process.stdin.write(json.dumps(query) + "\n")
                process.stdin.flush()
                value = read()
                save(value, time.perf_counter() - started)
            batch = dict(schema="hu-practice-continual-resolver-batch-query-v1",
                         requestId="latency-batch", streamResults=True,
                         queries=[query for _, query in queries[5:7]])
            started = time.perf_counter()
            process.stdin.write(json.dumps(batch) + "\n")
            process.stdin.flush()
            first = read()
            elapsed = time.perf_counter() - started
            if first.get("schema") == "hu-practice-continual-resolver-batch-result-v1":
                for value in first["results"]:
                    save(value, elapsed)
            else:
                save(first, elapsed)
                second = read()
                if second.get("requestId") == first.get("requestId"):
                    raise ValueError("duplicate streamed response")
                save(second, time.perf_counter() - started)
            if args.cross_batch:
                started = time.perf_counter()
                for name, query in queries[7:]:
                    batch = dict(schema="hu-practice-continual-resolver-batch-query-v1",
                                 requestId=name + "-container", streamResults=True, queries=[query])
                    process.stdin.write(json.dumps(batch) + "\n")
                process.stdin.flush()
                for _ in range(2):
                    value = read()
                    elapsed = time.perf_counter() - started
                    if value.get("schema") == "hu-practice-continual-resolver-batch-result-v1":
                        for result in value["results"]:
                            save(result, elapsed)
                    else:
                        save(value, elapsed)
            if seen != set(dict(queries)):
                raise ValueError("incomplete request coverage")
            record["status"] = "complete"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            raise
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            record["worker"] = guard.finish()
            atomic_json(output / "manifest.json", record)
        if record["worker"]["resourceStopReason"] or process.returncode:
            record["status"] = "failed"
            record["error"] = "resource stop or nonzero process exit"
            atomic_json(output / "manifest.json", record)
            raise ValueError("resource stop or nonzero process exit invalidated this receipt")


if __name__ == "__main__":
    main()

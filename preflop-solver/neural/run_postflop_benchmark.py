"""Frozen, board-family-disjoint 20bb postflop benchmark; never promotes a model.

Reports half-summed conditional best-response gain / STARTING flop pot.
The JS audit independently backs up the flop using shared Rust continuation
packets and equity; it is not an independent poker engine or a Pio comparison.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import math
from pathlib import Path
import random
import shutil
import signal
import threading
import time

import numpy as np
from native_value_dataset import board_family
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

HERE = Path(__file__).resolve().parent
PREFIX = "blueprint::public_belief::counterfactual_turn::flop_pilot::"
RESPONSE = PREFIX + "frozen_response::tests::"
EXPORT = "blueprint::preflop_continuation::postflop_benchmark::export_postflop_benchmark_roots"
PREFLOP = "runs/local-compact-preflop-20260908-lcfr32-b/seed27001.preflop.json.gz"
STUDENTS = "runs/local-native-value-20260907-student-ace-coverage-a/manifest.json"
CORPORA = ["runs/local-native-value-20260907-ace-coverage-a/corpus.json.gz",
           "runs/local-path-a-values-20260908-a/corpus.json.gz"]
TEXTURES = ("paired", "monotone", "connected-two-tone", "high-rainbow")
POTS = (("limped", 2.0), ("single-raised", 5.0), ("three-bet", 15.0))
MEMORY = 1536 * 1024**2
STAGE_SECONDS = 12 * 3600


def texture(board):
    ranks = sorted(c // 4 for c in board)
    suits = {c % 4 for c in board}
    if len(set(ranks)) == 2:
        return "paired"
    if len(set(ranks)) != 3:
        return None
    if len(suits) == 1:
        return "monotone"
    if len(suits) == 2 and ranks[-1] - ranks[0] <= 4:
        return "connected-two-tone"
    if len(suits) == 3 and ranks[-1] >= 10:
        return "high-rainbow"
    return None


def choose_spots(excluded, seed=20260921):
    rng = random.Random(seed)
    seen = set(excluded)
    spots = []
    for pot_type, pot in POTS:
        for wanted in TEXTURES:
            for _ in range(100000):
                board = sorted(rng.sample(range(52), 3))
                family = board_family(board)
                if texture(board) == wanted and family not in seen:
                    break
            else:
                raise ValueError("unable to select disjoint benchmark texture")
            seen.add(family)
            spots.append(dict(id=f"{pot_type}-{wanted}", potType=pot_type,
                              texture=wanted, board=board, family=list(family), startingPotBb=pot))
    return spots


def check_pins(pins):
    for path, digest in pins.items():
        if sha256(Path(path)) != digest:
            raise ValueError("pinned benchmark input changed: " + path)


def same_public_state(left, right):
    """Allow only roundoff from native range re-normalization, never support drift."""
    if {k: v for k, v in left.items() if k != "ranges"} != {
            k: v for k, v in right.items() if k != "ranges"}:
        return False
    a, b = left.get("ranges", []), right.get("ranges", [])
    if len(a) != 2 or len(b) != 2:
        return False
    for x, y in zip(a, b):
        if len(x) != len(y) or not x:
            return False
        if any(not math.isfinite(p) or not math.isfinite(q) or p < 0 or q < 0
               or (p == 0) != (q == 0) or abs(p - q) > 1e-14 for p, q in zip(x, y)):
            return False
        if abs(math.fsum(x) - 1) > 1e-12 or abs(math.fsum(y) - 1) > 1e-12:
            return False
    return True


def prepare(binary, output):
    output.mkdir(exist_ok=False)
    students_path = HERE / STUDENTS
    students = json.loads(students_path.read_text())
    if students["status"] != "complete":
        raise ValueError("completed retained students required")
    inputs = [binary, HERE / PREFLOP, students_path, Path(__file__),
              HERE / "run_native_value_pilot.py", HERE / "run_native_value_preflight.py",
              HERE / "worker_resources.py", HERE / "native_value_dataset.py",
              HERE / "audit_native_flop_response.mjs", HERE / "native_action_diagnostics.mjs",
              HERE / "native_action_value_probe.mjs"]
    excluded = set()
    exclusion_sources = []
    for name in CORPORA:
        path = HERE / name
        with gzip.open(path, "rb") as stream:
            data = stream.read(256 * 1024**2 + 1)
        if len(data) > 256 * 1024**2:
            raise ValueError("exclusion corpus exceeds decoded budget")
        excluded.update(board_family(t["board"]) for t in json.loads(data)["targets"])
        exclusion_sources.append(str(path)); inputs.append(path)
    # Also exclude known development roots and the consumed preflop response boards.
    prior_roots = list((HERE / "runs/local-native-flop-20260906-broader-inputs").glob("source-*.json"))
    prior_roots += list((HERE.parent / "tests/fixtures").glob("flop-continuation-public-root-*.json"))
    for seed in (27001, 27002):
        prior_roots += list((HERE / f"runs/local-independent-boards-response-20260908-{seed}-reboot-complete").glob("board[0-3].json"))
    for path in prior_roots:
        record = json.loads(path.read_text())
        board = record.get("board", record.get("public", {}).get("board"))
        if board is not None:
            excluded.add(board_family(board)); inputs.append(path); exclusion_sources.append(str(path))
    models = []
    for i, model in enumerate(students["predictions"]):
        path = Path(model["model"])
        if sha256(path) != model["modelSha256"] or model["maximumParityErrorBb"] > 1e-4:
            raise ValueError("retained model identity/parity changed")
        inputs.append(path)
        models.append(dict(seed=100101+i, trainingSeed=model["seed"], path=str(path), sha256=sha256(path)))
    if len(models) != 2:
        raise ValueError("exactly two retained model seeds required")
    protocol = dict(schema="postflop-benchmark-protocol-v1", selectionSeed=20260921,
        spots=choose_spots(excluded), models=models, preflop=str(HERE / PREFLOP),
        binary=str(binary), flopIterations=128, turnIterations=64,
        excludedFamilies=[list(f) for f in sorted(excluded)], exclusionSources=exclusion_sources,
        exclusionScope="Pinned retained/rejected value corpora, named native development roots, and consumed preflop-response boards; not every historical artifact in the repository",
        pinnedInputs={str(p.resolve()): sha256(p) for p in inputs},
        maximumNativeWorkers=2, maximumWorkerMemoryBytes=MEMORY, maximumStageSeconds=STAGE_SECONDS,
        metric="100 * (gain_seat0 + gain_seat1) / (2 * starting_flop_pot_bb)",
        spotWeighting="Equal weight per prespecified spot; seeds averaged within each spot, not counted as independent boards",
        referenceMeanPercentPot=0.6, referenceSource="https://deepsolver.com/blog/speed-precision-benchmarks-and-testing",
        referenceInterpretation="Historical vendor average on different spots; descriptive reference, NOT a matched head-to-head pass gate",
        releaseAccepted=False, deployedModelEvaluated=False, fullGameGateEvaluated=False,
        evaluator="Frozen native full postflop conditional response over all 49 legal turn cards, exact combo information sets, f32 policy/equity arithmetic; independent JS flop accounting only, shared native turn/river packets",
        limitations=["20bb rake-free abstraction only", "Not an external Pio/Deepsolver replication", "No whole-game exploitability certificate", "No model selection or fitting on this benchmark"])
    atomic_json(output / "protocol.json", protocol)
    env = dict(POKER_NOISE_PREFLOP=protocol["preflop"], POKER_NOISE_PREFLOP_SHA=sha256(HERE / PREFLOP),
        POKER_BENCHMARK_PROTOCOL=str(output / "protocol.json"), POKER_BENCHMARK_PROTOCOL_SHA=sha256(output / "protocol.json"),
        POKER_COMPACT_OUTPUT=str(output / "roots.json"))
    guarded(test_command(binary, EXPORT), env, output / "root-export", 180, MEMORY)
    roots = json.loads((output / "roots.json").read_text())
    if roots["preflopSha256"] != sha256(HERE / PREFLOP):
        raise ValueError("root export policy changed")
    root_map = {r["id"]: r["input"] for r in roots["roots"]}
    root_pins = {}
    (output / "inputs").mkdir()
    for spot in protocol["spots"]:
        value = root_map[spot["id"]]
        if (value["public"]["board"] != spot["board"] or
                sum(value["public"]["invested_bb"]) != spot["startingPotBb"] or
                value["public"]["street_invested_bb"] != [0, 0]):
            raise ValueError("root export board/pot mismatch")
        path = output / "inputs" / (spot["id"] + ".json")
        atomic_json(path, value); root_pins[str(path)] = sha256(path)
    atomic_json(output / "inputs.json", dict(protocolSha256=sha256(output / "protocol.json"), roots=root_pins))
    print(json.dumps(dict(event="benchmark-frozen", spots=12, seeds=2, excludedFamilies=len(excluded),
                          protocolSha256=sha256(output / "protocol.json"))), flush=True)


def run_job(binary, test, env, stage, outputs, stop, seconds=900, command=None):
    """Completed immutable receipts resume; interrupted attempts stay as evidence."""
    stage.mkdir(exist_ok=True)
    receipt = stage / "completed.json"
    identity = dict(command=command or test_command(binary, test), environment=env)
    if receipt.exists():
        r = json.loads(receipt.read_text())
        if r["identity"] != identity:
            raise ValueError("recovered job identity changed")
        check_pins(r["outputs"])
        return r
    attempt = 0
    while (stage / f"attempt-{attempt}").exists():
        attempt += 1
    attempt_dir = stage / f"attempt-{attempt}"
    temporary_receipt = receipt.with_suffix('.json.tmp')
    if temporary_receipt.exists():
        temporary_receipt.rename(stage / f"completed.interrupted-{attempt}.tmp")
    # Orphan outputs from interrupted attempts are kept, not overwritten/scored.
    for path in outputs:
        if path.exists():
            orphan = stage / (path.name + f".interrupted-{attempt}")
            path.rename(orphan)
    worker = guarded(identity["command"], env, attempt_dir, seconds, MEMORY, stop)
    record = dict(identity=identity, worker=worker, outputs={str(p): sha256(p) for p in outputs})
    atomic_json(receipt, record)
    return record


def summarize(rows, spots):
    if len(rows) != 2 * len(spots):
        raise ValueError("incomplete benchmark cannot receive a headline score")
    values = []
    for spot in spots:
        selected = [r for r in rows if r["spot"] == spot["id"]]
        if len(selected) != 2 or {r["seed"] for r in selected} != {100101, 100102}:
            raise ValueError("missing/duplicate paired seed")
        for row in selected:
            if (not math.isfinite(row["halfSummedGainBb"]) or row["halfSummedGainBb"] < -1e-8
                    or not math.isclose(row["percentPot"],100*row["halfSummedGainBb"]/spot["startingPotBb"],abs_tol=1e-10)):
                raise ValueError("invalid response or pot normalization")
        values.append(dict(spot=spot["id"], potType=spot["potType"], texture=spot["texture"],
                           meanPercentPot=float(np.mean([r["percentPot"] for r in selected]))))
    x = np.asarray([r["meanPercentPot"] for r in values])
    return dict(spotCount=len(spots), seedCount=2, perSpot=values,
        meanPercentPot=float(x.mean()), medianPercentPot=float(np.median(x)),
        p90PercentPot=float(np.quantile(x,.9)), maximumPercentPot=float(x.max()),
        maximumIndividualSeedSpotPercentPot=max(r['percentPot'] for r in rows),
        fractionSpotsBelow1Percent=float(np.mean(x < 1.0)),
        perSeedMeanPercentPot={str(seed):float(np.mean([r["percentPot"] for r in rows if r["seed"]==seed])) for seed in (100101,100102)},
        perPotTypeMeanPercentPot={kind:float(np.mean([r["meanPercentPot"] for r in values if r["potType"]==kind])) for kind,_ in POTS if any(r["potType"]==kind for r in values)},
        coldFlopSolveMedianSeconds=float(np.median([r["solveSeconds"] for r in rows])),
        coldFlopSolveP95Seconds=float(np.quantile([r["solveSeconds"] for r in rows],.95)),
        confidenceUpperBound99=None, releaseAccepted=False, fullGameGateEvaluated=False,
        interpretation="Descriptive small stratified fresh-board suite, NOT a random population confidence bound, current website score, or matched commercial comparison. Cold flop solve time excludes full-strategy materialization/evaluation.")


def run(output, preflight):
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    roots = json.loads((output / "inputs.json").read_text())
    if roots["protocolSha256"] != sha256(protocol_path):
        raise ValueError("benchmark protocol changed")
    pins = {**protocol["pinnedInputs"], **roots["roots"]}
    check_pins(pins)
    binary = Path(protocol["binary"])
    # Wider limped root first: cost/memory only, never a quality-based spot selection.
    selected_spots = protocol["spots"][:1] if preflight else protocol["spots"]
    stage = output / ("preflight" if preflight else "complete")
    stage.mkdir(exist_ok=True)
    temporary_manifest = stage / 'manifest.json.tmp'
    if temporary_manifest.exists():
        temporary_manifest.rename(stage / f'manifest.interrupted-{time.time_ns()}.tmp')
    record = dict(schema="postflop-benchmark-results-v1", status="running", protocolSha256=sha256(protocol_path),
                  rows=[], releaseAccepted=False, fullGameGateEvaluated=False)
    atomic_json(stage / "manifest.json", record)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(STAGE_SECONDS,stop.set); timer.daemon=True; timer.start()
    started = time.monotonic()
    try:
        for spot in selected_spots:
            root = output / "inputs" / (spot["id"] + ".json")
            for model in protocol["models"]:
                if stop.is_set(): raise ValueError("stage stopped")
                check_pins(pins)
                work = output / "jobs" / spot["id"] / str(model["seed"])
                work.mkdir(parents=True,exist_ok=True)
                candidate = work / "candidate.json"
                env = dict(POKER_NATIVE_FLOP_INPUT=str(root),POKER_NATIVE_FLOP_INPUT_SHA=sha256(root),
                    POKER_NATIVE_FLOP_SEED=str(model["seed"]),POKER_NATIVE_FLOP_ITERATIONS="128",
                    POKER_NATIVE_FLOP_TURN_ITERATIONS="64",POKER_NATIVE_FLOP_CHANCE_BASELINE="none",
                    POKER_NATIVE_FLOP_TURN_SAMPLES="1",POKER_NATIVE_FLOP_LEAF_WORKERS="1",
                    POKER_NATIVE_FLOP_VALUE_MODEL=model["path"],POKER_NATIVE_FLOP_VALUE_MODEL_SHA=model["sha256"],
                    POKER_NATIVE_FLOP_OUTPUT=str(candidate))
                train = run_job(binary,PREFIX+"saved_20bb_native_flop_pilot",env,work/"solve",[candidate],stop)
                value = json.loads(candidate.read_text()); source=json.loads(root.read_text())
                if (not same_public_state(value["state"],source["public"]) or value["game"]!=source["game"] or value["iterations"]!=128
                        or value["turn_iterations"]!=64 or value["learned_leaf_model_sha256"]!=model["sha256"]):
                    raise ValueError("candidate does not match frozen benchmark")
                common=dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),POKER_NATIVE_FLOP_CANDIDATE_SHA=sha256(candidate))
                equity=work/"equity.json"
                run_job(binary,RESPONSE+"saved_native_flop_equity_audit_input",
                        {**common,"POKER_NATIVE_FLOP_OUTPUT":str(equity)},work/"equity-export",
                        [equity,equity.with_suffix('.f32le')],stop)
                packets=work/"packets";packets.mkdir(exist_ok=True)
                def packet(turn):
                    if stop.is_set(): raise ValueError("stopped before packet")
                    path=packets/f"turn-{turn}.json"
                    result=run_job(binary,RESPONSE+"saved_native_flop_turn_packet",
                        {**common,"POKER_NATIVE_FLOP_TURN":str(turn),"POKER_NATIVE_FLOP_OUTPUT":str(path)},
                        work/f"turn-{turn}",[path],stop,seconds=900)
                    packet_value=json.loads(path.read_text())
                    if packet_value['candidate_sha256']!=common['POKER_NATIVE_FLOP_CANDIDATE_SHA'] or packet_value['turn']!=turn:
                        raise ValueError("packet identity changed")
                    return result
                turns=[c for c in range(52) if c not in spot["board"]]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(packet,c) for c in turns]
                    try:
                        for n,future in enumerate(as_completed(futures),1):
                            future.result()
                            if n%7==0: print(json.dumps(dict(event="benchmark-turns",spot=spot["id"],seed=model["seed"],completed=n,total=49)),flush=True)
                    except BaseException:
                        stop.set();raise
                response=work/"response.json"
                run_job(binary,RESPONSE+"saved_native_flop_response_aggregate",
                        {**common,"POKER_NATIVE_FLOP_PACKET_DIRECTORY":str(packets),"POKER_NATIVE_FLOP_OUTPUT":str(response)},
                        work/"aggregate",[response],stop,seconds=180)
                audit=HERE/"audit_native_flop_response.mjs"
                run_job(binary,None,{},work/"audit",[],stop,seconds=180,
                        command=["node",str(audit),str(candidate),str(packets),str(equity),str(response)])
                result=json.loads(response.read_text());gain=result["half_summed_gain_bb"]
                row=dict(spot=spot["id"],seed=model["seed"],board=spot["board"],startingPotBb=spot["startingPotBb"],
                         halfSummedGainBb=gain,percentPot=100*gain/spot["startingPotBb"],
                         solveSeconds=train["worker"]["workerElapsedSeconds"],candidateSha256=sha256(candidate),
                         response=str(response),responseSha256=sha256(response),all49Turns=True,flopAccountingAuditPassed=True)
                record["rows"].append(row); atomic_json(stage/"manifest.json",record)
                print(json.dumps(dict(event="benchmark-spot-seed-complete",**row)),flush=True)
        check_pins(pins)
        record.update(status="complete",summary=summarize(record["rows"],selected_spots))
    except (ValueError,OSError,KeyError,AssertionError) as error:
        stop.set();record.update(status="failed",failure=str(error))
    finally:
        timer.cancel();record["elapsedSeconds"]=time.monotonic()-started
        atomic_json(stage/"manifest.json",record)
    print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
    if record["status"]!='complete': raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','preflight','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--binary',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if shutil.disk_usage(output.parent).free<24*1024**3:
        raise ValueError('24GiB disk headroom required')
    if args.mode=='prepare':
        if args.binary is None: raise ValueError('prepare requires compiled native binary')
        prepare(args.binary.resolve(),output)
    else:
        if args.mode=='run':
            pre=json.loads((output/'preflight/manifest.json').read_text())
            if pre['status']!='complete' or pre['protocolSha256']!=sha256(output/'protocol.json'):
                raise ValueError('completed pinned preflight required')
            if pre['elapsedSeconds']*12>STAGE_SECONDS:
                raise ValueError('conservative 12-spot cost projection exceeds twelve-hour cap')
        run(output,args.mode=='preflight')


if __name__=='__main__': main()

"""Bounded paired exact-averaging diagnostic; never promotes a policy."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from cloud_blueprint_run import atomic_json, file_sha256, policy_stability_summary
from worker_resources import WorkerResourceGuard


def stability(left, right):
    """Same public-state/action/169-class diagnostic, not exploitability."""
    right_rows = {r['key']: r for r in right['rows']}
    groups = {}
    for a in left['rows']:
        b = right_rows[a['key']]
        assert (a['actor'], a['history'], a['hand'], a['actions'], a['comboWeight']) == (
            b['actor'], b['history'], b['hand'], b['actions'], b['comboWeight'])
        group = groups.setdefault((a['actor'], tuple(a['history'])), [])
        group.append((a, b))
    result = []
    for (actor, history), rows in groups.items():
        complete = [(a, b) for a, b in rows if a['probabilities'] is not None and b['probabilities'] is not None]
        weight = sum(a['comboWeight'] for a, _ in complete)
        if weight == 0:
            result.append(dict(actor=actor, history=history, available=False))
            continue
        actions = complete[0][0]['actions']
        mean = [0.0] * len(actions)
        mae = agreement = 0.0
        for a, b in complete:
            w = a['comboWeight'] / weight
            pa, pb = a['probabilities'], b['probabilities']
            for p in (pa, pb):
                assert all(0 <= x <= 1 for x in p) and abs(sum(p)-1) < 1e-12
            delta = [x-y for x, y in zip(pa, pb)]
            mae += w * sum(map(abs, delta)) / len(actions)
            agreement += w * (max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__))
            for i, d in enumerate(delta):
                mean[i] += w*d
        result.append(dict(actor=actor, history=history, available=True, comparedCombos=weight,
            completeCoverage=weight == 1326, comboWeightedPerActionMae=mae,
            comboWeightedPrimaryAgreement=agreement, maximumAggregateDelta=max(map(abs, mean))))
    # Authentic initial histories include posted blinds; they are not empty.
    first_length = min(len(r['history']) for r in result)
    roots = [r for r in result if len(r['history']) == first_length]
    assert len(roots) == 1 and roots[0]['actor'] == 0
    return dict(groups=result, initialRoot=roots[0],
        completePublicStates=sum(r.get('completeCoverage', False) for r in result),
        interpretation='Within-public-state combo weights, not a held-out full-hand reach distribution. Incomplete groups are explicitly marked.')


def analyze(record):
    outputs = {}
    for job in record['jobs']:
        assert job['status'] == 'complete'
        path = Path(job['output'])
        assert file_sha256(path) == job['outputSha256']
        value = json.loads(path.read_text())
        assert value['config']['seed'] == job['seed']
        assert value['config']['iterations'] == record['rounds']
        assert value['config'].get('exact_preflop_averaging', False) == (job['mode'] == 'exact')
        outputs[job['seed'], job['mode']] = value
    assert len(outputs) == 4
    comparisons = []
    for seed in (26001, 26002):
        a, b = outputs[seed, 'sampled'], outputs[seed, 'exact']
        for key in ('regretStateSha256', 'postflopStateSha256', 'rngState', 'sampledDeals', 'terminalEvaluations'):
            assert a[key] == b[key], (seed, key, a[key], b[key])
        comparisons.append(dict(seed=seed, trainingParity=True,
            sampledMissingAverageCombos=a['absentAverageCombos'], exactMissingAverageCombos=b['absentAverageCombos'],
            sampledUntrainedCombos=a['untrainedCombos'], exactUntrainedCombos=b['untrainedCombos'],
            sampledTrainingSeconds=a['trainingSeconds'], exactTrainingSeconds=b['trainingSeconds']))
    established = {}
    for mode in ('sampled', 'exact'):
        summaries = []
        for seed in (26001, 26002):
            rows = outputs[seed, mode]['rows']
            first_length = min(len(r['history']) for r in rows)
            root = [r for r in rows if len(r['history']) == first_length]
            summaries.append({'rootStrategies': [dict(hand=r['hand'],
                averageVisits=r['averageVisits'] or 0, regretUpdates=r['regretUpdates'] or 0,
                trainedAverage=r['trained'], actions=[dict(action=a, probability=p)
                    for a, p in zip(r['actions'], r['probabilities'] or [])]) for r in root]})
        try:
            # Reuse the established gate exactly: MAXIMUM individual-action
            # MAE and unweighted primary agreement. The per-public-state
            # diagnostic above instead reports the MEAN across actions.
            established[mode] = policy_stability_summary(summaries)
        except ValueError as error:
            established[mode] = dict(available=False, passed=False, reason=str(error))
    return dict(comparisons=comparisons, establishedRootStability=established,
        stability={mode: stability(outputs[26001, mode], outputs[26002, mode])
        for mode in ('sampled', 'exact')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, choices=[2, 8, 16, 32, 64, 800], default=8)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--analyze-existing', action='store_true', help='Verify completed outputs and write a separate analysis; no training or overwrite')
    args = parser.parse_args()
    assert args.rounds != 800 or args.workers == 1, '800-round matched replay is serial on the 16GiB host'
    worker_memory_gib = 8 if args.rounds == 800 else 4
    binary = args.binary.resolve()
    assert file_sha256(binary) == args.binary_sha256
    stage = args.output.resolve()
    if args.analyze_existing:
        source = stage/'manifest.json'
        record = json.loads(source.read_text())
        assert record['binarySha256'] == args.binary_sha256
        output = stage/'analysis.json'
        assert not output.exists()
        result = dict(analyze(record), sourceManifestSha256=file_sha256(source),
            analysisCodeSha256=file_sha256(Path(__file__)), status='complete')
        atomic_json(output, result)
        print(json.dumps(result['comparisons']), flush=True)
        return
    assert not stage.exists()
    assert shutil.disk_usage(stage.parent).free > 20*1024**3 + 64*1024**2
    stage.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(3600, stop.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    record = dict(schema='paired-exact-preflop-averaging-controller-v1', status='running',
        binarySha256=args.binary_sha256, runnerSha256=file_sha256(Path(__file__)),
        rounds=args.rounds, maximumWorkers=args.workers, maximumWorkerMemoryGiB=worker_memory_gib,
        maximumWorkerSeconds=900, maximumStageSeconds=3600, minimumFreeDiskGiB=20,
        jobs=[], comparisons=[], stability={}, interpretation='Research diagnostic only. No promotion or full-game exploitability claim.')

    def save():
        atomic_json(stage/'manifest.json', record)

    def job(seed, mode):
        if stop.is_set():
            return dict(seed=seed, mode=mode, status='cancelled_before_start')
        directory = stage/f'seed{seed}-{mode}'
        directory.mkdir()
        output = directory/'result.json'
        env = dict(POKER_AVERAGE_SEED=str(seed), POKER_AVERAGE_MODE=mode,
            POKER_AVERAGE_ROUNDS=str(args.rounds), POKER_AVERAGE_OUTPUT=str(output))
        if args.rounds == 800:
            env['POKER_AVERAGE_POLICY_OUTPUT'] = str(directory/'frozen-preflop.json.gz')
        command = [str(binary), 'blueprint::preflop_average::pilot::matched_preflop_averaging_pilot',
            '--exact', '--ignored', '--nocapture', '--test-threads=1']
        result = dict(seed=seed, mode=mode, command=command, environment=env, output=str(output),
            startedAtUnix=time.time(), status='running')
        with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, env=dict(os.environ, **env), stdout=stdout,
                stderr=stderr, start_new_session=True)
            guard = WorkerResourceGuard(process, directory, worker_memory_gib*1024**3, 900, 20*1024**3, stop_event=stop).start()
            result['pid'] = process.pid
            atomic_json(directory/'manifest.json', result)
            print(seed, mode, 'pid', process.pid, flush=True)
            try:
                process.wait()
            except BaseException:
                guard.request_stop('averaging pilot interrupted')
                process.wait()
                raise
            finally:
                result.update(guard.finish(), exitCode=process.poll(), finishedAtUnix=time.time())
        result['status'] = 'complete' if result['exitCode'] == 0 and result['resourceStopReason'] is None else 'failed'
        if result['status'] == 'complete':
            result['outputSha256'] = file_sha256(output)
            frozen = directory/'frozen-preflop.json.gz'
            if frozen.exists():
                result['frozenPolicySha256'] = file_sha256(frozen)
        else:
            stop.set()
        atomic_json(directory/'manifest.json', result)
        return result

    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(job, seed, mode) for seed in (26001, 26002) for mode in ('sampled', 'exact')]
            for future in as_completed(futures):
                result = future.result()
                record['jobs'].append(result)
                save()
                assert result['status'] == 'complete', result
        record.update(analyze(record))
        assert file_sha256(binary) == args.binary_sha256
        record.update(status='complete', elapsedSeconds=time.monotonic()-started)
    except BaseException:
        stop.set()
        record['status'] = 'failed'
        save()
        raise
    finally:
        timer.cancel()
    save()
    print(json.dumps(record['comparisons']), flush=True)


if __name__ == '__main__':
    main()

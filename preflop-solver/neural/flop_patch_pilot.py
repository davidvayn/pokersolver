"""Resource-guarded fixed-opponent flop-action experiments; no model promotion."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import threading

from cloud_blueprint_run import atomic_json, file_sha256
from worker_resources import WorkerResourceGuard


def matching_reports(paths, checkpoint_sha):
    return [path for path in paths
            if json.loads(path.read_text())['policy_sha256'] == checkpoint_sha]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--checkpoint-stage', type=Path, required=True)
    parser.add_argument('--proposal-responses', required=True)
    parser.add_argument('--opponent-responses', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', default='26001,26002')
    parser.add_argument('--seed-offset', type=int, required=True)
    parser.add_argument('--weight', type=float, default=0.25)
    parser.add_argument('--evaluation-deals', type=int, default=1024)
    parser.add_argument('--response-workers', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--all-in-samples', type=int)
    parser.add_argument('--integrate-terminal', action='store_true')
    parser.add_argument('--max-worker-minutes', type=float, default=30)
    parser.add_argument('--max-worker-memory-gib', type=float, default=7.5)
    args = parser.parse_args()
    if not math.isfinite(args.weight) or not 0 <= args.weight <= 0.5:
        parser.error('weight must be 0..0.5')
    if not 2 <= args.evaluation_deals <= 1_000_000:
        parser.error('evaluation hands must be 2..1000000')
    if args.all_in_samples is not None and not 128 <= args.all_in_samples <= 16384:
        parser.error('all-in samples must be 128..16384')
    if args.integrate_terminal and args.all_in_samples is None:
        parser.error('terminal integration requires the terminal-only all-in correction')
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.max_worker_minutes, args.max_worker_memory_gib)):
        parser.error('positive finite resource stops required')
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(seeds) != len(set(seeds)):
        parser.error('duplicate seeds')
    parent = json.loads((args.checkpoint_stage / 'run-manifest.json').read_text())
    if parent['status'] != 'complete':
        parser.error('checkpoint stage must be complete')
    proposals = [Path(p).resolve() for p in args.proposal_responses.split(',')]
    opponents = [Path(p).resolve() for p in args.opponent_responses.split(',')]
    jobs = []
    for seed in seeds:
        source = parent['seeds'][str(seed)]
        checkpoint = Path(source['checkpoint'])
        digest = source['checkpointSha256']
        if source['status'] != 'complete' or file_sha256(checkpoint) != digest:
            parser.error('checkpoint integrity failure')
        proposal = matching_reports(proposals, digest)
        panel = matching_reports(opponents, digest)
        if len(proposal) != 1 or not panel:
            parser.error('need exactly one proposal and at least one opponent per checkpoint')
        paths = [proposal[0], *panel]
        if any(json.loads(p.read_text())['seed'] == seed + args.seed_offset for p in paths):
            parser.error('fresh seed required')
        jobs.append((seed, checkpoint, digest, proposal[0], panel))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'cohort.json'
    binary = (args.output_dir / 'solver-frozen').resolve()
    if manifest.exists() or binary.exists():
        parser.error('refusing to overwrite a cohort')
    if shutil.disk_usage(args.output_dir).free < 20 * 1024**3:
        parser.error('20GiB disk reserve required')
    shutil.copy2(args.binary, binary)
    state = {'schema': 'flop-patch-pilot-v1', 'status': 'running',
             'binarySha256': file_sha256(binary), 'runs': [],
             'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    event = threading.Event()
    def stop(_signal, _frame):
        event.set()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    atomic_json(manifest, state)
    for seed, checkpoint, digest, proposal, panel in jobs:
        if event.is_set():
            state['status'] = 'interrupted'; atomic_json(manifest, state); return 130
        output = (args.output_dir / f'flop-seed{seed}.json').resolve()
        command = [str(binary), 'tabular-flop-pilot', '--tabular-checkpoint', str(checkpoint),
                   '--proposal-response', str(proposal), '--opponent-responses', ','.join(map(str, panel)),
                   '--weight', str(args.weight), '--evaluation-deals', str(args.evaluation_deals),
                   '--seed', str(seed + args.seed_offset), '--response-workers', str(args.response_workers),
                   '--output', str(output)]
        if args.all_in_samples is not None:
            command += ['--all-in-samples', str(args.all_in_samples)]
        if args.integrate_terminal:
            command += ['--integrate-terminal']
        record = {'seed': seed, 'status': 'running', 'command': command, 'checkpointSha256': digest,
                  'inputReportSha256': {str(p): file_sha256(p) for p in [proposal, *panel]}}
        state['runs'].append(record); atomic_json(manifest, state)
        print(f'Starting paired flop correction seed {seed}', flush=True)
        with output.with_suffix('.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            guard = WorkerResourceGuard(child, args.output_dir,
                max_memory_bytes=int(args.max_worker_memory_gib * 1024**3),
                max_seconds=args.max_worker_minutes * 60, minimum_free_disk_bytes=20 * 1024**3,
                stop_event=event).start()
            try:
                code = child.wait()
            finally:
                if child.poll() is None:
                    guard.request_stop('pilot interrupted'); child.wait(timeout=10)
                record.update(guard.finish())
        record['returnCode'] = code
        if code != 0 or record['resourceStopReason'] is not None:
            record['status'] = 'failed'; state['status'] = 'stopped'; atomic_json(manifest, state); return 1
        report = json.loads(output.read_text())
        if report['policy_sha256'] != digest:
            raise ValueError('unexpected policy identity')
        record.update(status='complete', outputSha256=file_sha256(output),
                      results=[{k: v for k, v in r.items() if not k.startswith('paired_samples')}
                               for r in report['results']])
        atomic_json(manifest, state)
        print(f'Completed flop correction seed {seed}: {record["results"]}', flush=True)
    state['status'] = 'complete'; atomic_json(manifest, state); return 0


if __name__ == '__main__':
    raise SystemExit(main())

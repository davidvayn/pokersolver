"""Sequential, resource-guarded paired response pilots on immutable checkpoints.

No training checkpoint is modified. Stop on the first failed job; never convert
a missing candidate output or a rejected responder into an apparent policy win.
"""
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


def response_comparison_eligible(report):
    """A rejected response is not evidence for a low-exploitability candidate."""
    return (report.get('response_deployed') == [True, True]
            and all(math.isfinite(player.get('estimated_gain_bb', math.nan))
                    and math.isfinite(player.get('gain_standard_error_bb', math.nan))
                    and player['gain_standard_error_bb'] >= 0
                    for player in report.get('players', []))
            and len(report.get('players', [])) == 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--checkpoint-stage', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--arms', default='baseline,joint:8,safe:8')
    parser.add_argument('--seeds', default='26001,26002')
    parser.add_argument('--seed-offset', type=int, default=700000)
    parser.add_argument('--training-deals', type=int, default=256)
    parser.add_argument('--response-workers', type=int, choices=range(1,5), default=1)
    parser.add_argument('--terminal-flop-samples', type=int)
    parser.add_argument('--terminal-flop-weight', type=float, default=0.25)
    parser.add_argument('--flop-backoff-minimum-visits', type=int)
    parser.add_argument('--flop-backoff-weight', type=float, default=1.0)
    parser.add_argument('--response-terminal-expectations', action='store_true')
    parser.add_argument('--calibration-deals', type=int, default=256)
    parser.add_argument('--evaluation-deals', type=int, default=1024)
    parser.add_argument('--rollouts-per-action', type=int, default=4)
    parser.add_argument('--minimum-range-particles', type=int, default=4)
    parser.add_argument('--max-worker-minutes', type=float, default=30)
    parser.add_argument('--max-worker-memory-gib', type=float, default=7.5)
    args = parser.parse_args()
    if args.flop_backoff_minimum_visits is not None and (args.flop_backoff_minimum_visits < 1
            or not math.isfinite(args.flop_backoff_weight) or not 0 <= args.flop_backoff_weight <= 1):
        parser.error('flop pooling requires positive support and weight 0..1')
    if args.terminal_flop_samples is not None and not 128 <= args.terminal_flop_samples <= 16384:
        parser.error('terminal flop equity samples must be 128..16384')
    if not math.isfinite(args.terminal_flop_weight) or not 0 <= args.terminal_flop_weight <= 0.5:
        parser.error('terminal flop weight must be 0..0.5')
    if min(args.training_deals, args.calibration_deals, args.evaluation_deals, args.rollouts_per_action, args.minimum_range_particles) < 2:
        parser.error('all sample budgets must be at least two')
    if any(not math.isfinite(value) or value <= 0 for value in (args.max_worker_minutes,args.max_worker_memory_gib)):
        parser.error('positive worker stops required')
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms):
        parser.error('duplicate arms would overwrite a result')
    for arm in arms:
        if arm != 'baseline':
            name, separator, count = arm.partition(':')
            if not separator or name not in ('joint', 'safe') or not count.isdigit() or int(count) < 2:
                parser.error('arms must be baseline, joint:N or safe:N, N >= 2')
    seeds = [int(seed) for seed in args.seeds.split(',')]
    if len(set(seeds)) != len(seeds):
        parser.error('duplicate seeds')
    parent = json.loads((args.checkpoint_stage / 'run-manifest.json').read_text())
    if parent['status'] != 'complete':
        parser.error('checkpoint stage must be complete')
    if any(str(seed) not in parent['seeds'] for seed in seeds):
        parser.error('requested seed is absent from checkpoint stage')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'cohort.json'
    if manifest.exists():
        parser.error('refusing to overwrite an existing cohort')
    if shutil.disk_usage(args.output_dir).free < 20 * 1024**3:
        parser.error('20GiB disk reserve required')
    # Freeze the executable, since later development may rebuild target/release.
    binary = (args.output_dir / 'solver-frozen').resolve()
    if binary.exists():
        parser.error('frozen executable already exists')
    shutil.copy2(args.binary, binary)
    state = {'schema':'tabular-turn-pilot-v1', 'status':'running',
             'binarySha256':file_sha256(binary), 'runs':[],
             'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             'interpretation':'Research full-hand restricted-response pilot, not an exploitability upper bound. Calibration rejection is inconclusive, not proof of low exploitability.'}
    event = threading.Event()
    def stop(_signal, _frame):
        event.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    atomic_json(manifest, state)
    verified = set()
    for arm in arms:
        if arm != 'baseline' and 'baseline' in arms:
            baselines = [run for run in state['runs'] if run['arm'] == 'baseline']
            if len(baselines) != len(seeds) or not all(run.get('responseComparisonEligible') for run in baselines):
                state.update(status='inconclusive', reason='baseline responders failed calibration; increase evidence before expensive candidate comparisons')
                atomic_json(manifest, state)
                print(state['reason'], flush=True)
                return 2
        for seed in seeds:
            source = parent['seeds'][str(seed)]
            checkpoint = Path(source['checkpoint'])
            if seed not in verified:
                if source['status'] != 'complete' or file_sha256(checkpoint) != source['checkpointSha256']:
                    state.update(status='stopped', reason=f'checkpoint integrity check failed for seed {seed}')
                    atomic_json(manifest, state)
                    return 1
                verified.add(seed)
            if event.is_set():
                state['status']='interrupted'; atomic_json(manifest,state); return 130
            output = (args.output_dir / f'{arm.replace(":","-")}-seed{seed}.json').resolve()
            command = [str(binary),'full-game-lbr','--tabular-checkpoint',str(checkpoint),
                       '--training-deals',str(args.training_deals),'--calibration-deals',str(args.calibration_deals),
                       '--evaluation-deals',str(args.evaluation_deals),'--rollouts-per-action',str(args.rollouts_per_action),
                       '--minimum-range-particles',str(args.minimum_range_particles),'--maximum-response-granularity','strategic',
                       '--seed',str(seed+args.seed_offset),'--output',str(output)]
            command += ['--response-workers',str(args.response_workers)]
            if args.response_terminal_expectations:
                command += ['--response-terminal-expectations']
            if arm != 'baseline':
                name,count=arm.split(':'); command += ['--tabular-turn-iterations',count]
                if name == 'joint': command += ['--tabular-turn-unconstrained']
                if args.terminal_flop_samples is not None:
                    command += ['--terminal-flop-samples', str(args.terminal_flop_samples),
                                '--terminal-flop-weight', str(args.terminal_flop_weight)]
                if args.flop_backoff_minimum_visits is not None:
                    command += ['--flop-backoff-minimum-visits', str(args.flop_backoff_minimum_visits),
                                '--flop-backoff-weight', str(args.flop_backoff_weight)]
            record = {'seed':seed,'arm':arm,'command':command,'checkpointSha256':source['checkpointSha256'],'status':'running'}
            state['runs'].append(record); atomic_json(manifest,state)
            print(f'Starting {arm} seed {seed}',flush=True)
            with output.with_suffix('.log').open('wb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                guard=WorkerResourceGuard(child,args.output_dir,
                    max_memory_bytes=int(args.max_worker_memory_gib*1024**3),
                    max_seconds=args.max_worker_minutes*60,
                    minimum_free_disk_bytes=20*1024**3,stop_event=event).start()
                try:
                    code=child.wait()
                finally:
                    if child.poll() is None:
                        guard.request_stop('pilot interrupted'); child.wait(timeout=10)
                    record.update(guard.finish())
            record['returnCode']=code
            if code != 0 or record['resourceStopReason'] is not None:
                record['status']='failed';state['status']='stopped';atomic_json(manifest,state)
                print('Stopped; inspect worker log and resource record.',flush=True); return 1
            try:
                report=json.loads(output.read_text())
                if report['policy_sha256'] != source['checkpointSha256']:
                    raise ValueError('candidate used a different source policy')
                if not math.isfinite(report['total_response_gain_bb_per_hand']):
                    raise ValueError('nonfinite response result')
                eligible = response_comparison_eligible(report)
            except (OSError, ValueError, KeyError, TypeError) as error:
                record.update(status='failed', resultError=str(error))
                state['status']='stopped'; atomic_json(manifest,state)
                print('Stopped; invalid or missing worker result.',flush=True)
                return 1
            record.update(status='complete',outputSha256=file_sha256(output),
                totalResponseGainBbPerHand=report['total_response_gain_bb_per_hand'],
                responseDeployed=report['response_deployed'],
                responseComparisonEligible=eligible,
                resolutionDiagnostics=report.get('resolution_diagnostics',{}))
            atomic_json(manifest,state)
            print(f'Completed {arm} seed {seed}: gain={record["totalResponseGainBbPerHand"]:.6f}, deployed={record["responseDeployed"]}',flush=True)
    state['status']='complete';atomic_json(manifest,state);return 0


if __name__=='__main__':
    raise SystemExit(main())

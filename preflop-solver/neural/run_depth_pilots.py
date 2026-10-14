#!/usr/bin/env python3
"""Bounded, cold-start paired Home pilots; never modify serving manifests/weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
JOBS = ((50, 50121), (100, 100121), (50, 50122), (100, 100122))
# Three deliberately separated regular sizes at most. All-in is a distinct
# legal action, deduplicated by the hand engine when a target reaches the stack.
PILOT_GRID = {
    'open_sizes_bb': [2.0, 3.5, 5.0],
    'limp_raise_sizes_bb': [3.0, 5.0, 7.0],
    'three_bet_sizes_bb': [8.0, 12.0, 16.0],
    'four_bet_sizes_bb': [20.0, 30.0, 45.0],
    'deeper_raise_pot_fractions': [0.5, 1.0, 2.0],
    'preflop_raise_cap': 4,
    'flop_bet_pot_fractions': [1.0 / 3.0, 0.75, 1.25],
    'turn_river_bet_pot_fractions': [0.5, 1.0],
    'postflop_raise_pot_fractions': [1.0],
    'postflop_raise_cap': 1,
    'include_all_in': True,
}

def command(python: str, output: Path, depth: int, seed: int, max_minutes: float) -> list[str]:
    return [python, str(ROOT / 'preflop-solver/neural/train.py'), '--run-dir', str(output),
            '--depth-bb', str(depth), '--seed', str(seed), '--target-round', '3',
            '--max-minutes', str(max_minutes), '--traversals-per-round', '64',
            '--reservoir-capacity', '40000', '--hidden-sizes', '256,128',
            '--batch-size', '512', '--steps-per-round', '60', '--artifact-every', '1',
            '--preflop-runout-samples', '64', '--flop-runout-samples', '32',
            '--value-rollouts-per-action', '1', '--sample-turn-rivers',
            '--action-abstraction-json', json.dumps(PILOT_GRID, sort_keys=True, separators=(',', ':')),
            '--replay-street-proposal', '.25,.25,.25,.25']

def process_family(pid: int) -> dict[int, int]:
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    processes = {int(parts[0]): (int(parts[1]), int(parts[2]) * 1024)
                 for line in rows.splitlines() if len(parts := line.split()) == 3}
    owned = {pid}
    while True:
        added = {child for child, (parent, _) in processes.items() if parent in owned}
        expanded = owned | added
        if expanded == owned: break
        owned = expanded
    return {child: processes[child][1] for child in owned if child in processes}

def stop_family(process: subprocess.Popen) -> None:
    # The trainer's Rust sampler uses its own session. Track exact descendants,
    # not a broad name/pkill pattern, so a deadline cannot leave that sampler running.
    owned = process_family(process.pid)
    for pid in sorted(owned, reverse=True):
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        for pid in process_family(process.pid):
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
        process.wait(timeout=15)

def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def merge_results(previous: list[dict], result: dict) -> list[dict]:
    return [old for old in previous if (old.get('depthBb'),old.get('seed')) !=
            (result['depthBb'],result['seed'])] + [result]

def run_job(args: argparse.Namespace, depth: int, seed: int) -> dict:
    output = args.output_root / f'{depth}bb-seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    argv = command(args.python, output, depth, seed, args.max_minutes)
    environment = dict(os.environ)
    environment['PATH'] = '/Users/davidvayntrub/.cargo/bin:' + environment.get('PATH', '')
    started = time.monotonic()
    with (output / 'train.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info = {'depthBb': depth, 'seed': seed, 'pid': process.pid, 'status': 'training',
                'output': str(output), 'command': argv, 'startedUtc': datetime.now(timezone.utc).isoformat(),
                'hardDeadlineMinutes': args.deadline_minutes, 'maxCpuRssGb': args.max_rss_gb,
                'validationStatus': 'pilot_not_activated'}
        info['actionAbstraction'] = PILOT_GRID
        prior_file = output / 'pilot.json'
        if prior_file.exists():
            prior = json.loads(prior_file.read_text(encoding='utf-8'))
            info['previousAttempts'] = prior.get('previousAttempts', []) + [
                {key: prior.get(key) for key in ('pid','status','exitCode','elapsedSeconds','completedRounds')}]
        atomic_json(output / 'pilot.json', info)
        peak = 0
        while process.poll() is None:
            rss = sum(process_family(process.pid).values()); peak = max(peak, rss)
            if time.monotonic() - started > args.deadline_minutes * 60 or rss > args.max_rss_gb * 1024**3:
                info['status'] = 'stopped_deadline' if rss <= args.max_rss_gb * 1024**3 else 'stopped_memory'
                stop_family(process); break
            time.sleep(5)
        info.update(exitCode=process.returncode, elapsedSeconds=time.monotonic()-started, peakCpuRssBytes=peak)
        if info['status'] == 'training': info['status'] = 'completed' if process.returncode == 0 else 'failed'
    state_file = output / 'state.json'
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding='utf-8'))
        if state.get('config', {}).get('depth_bb') != depth or state.get('config', {}).get('seed') != seed:
            raise RuntimeError('Pilot checkpoint identity differs from its actual training depth/seed')
        info.update(completedRounds=state.get('completed_rounds',0), artifacts=state.get('artifacts',[]),
                    latestMetrics=state.get('metrics',[None])[-1] if state.get('metrics') else None)
    atomic_json(output / 'pilot.json', info)
    return info

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--workers', type=int, choices=(1,2), default=2)
    parser.add_argument('--max-minutes', type=float, default=12)
    parser.add_argument('--deadline-minutes', type=float, default=30)
    parser.add_argument('--max-rss-gb', type=float, default=3.5)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--only-seed', type=int, choices=tuple(seed for _,seed in JOBS),
                        help='retry one checkpoint with unchanged depth/grid/config')
    args = parser.parse_args(); args.output_root = args.output_root.resolve()
    if not 0 < args.max_minutes <= args.deadline_minutes <= 60 or not 0 < args.max_rss_gb <= 4:
        parser.error('Pilot time/memory bounds are invalid')
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = tuple((depth,seed) for depth,seed in JOBS if args.only_seed is None or seed == args.only_seed)
    if args.detach:
        argv = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != '--detach']]
        with (args.output_root / 'runner.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        launcher = {'pid':process.pid,'output':str(args.output_root),'depthsBb':sorted({d for d,_ in jobs}),
                    'jobs':[{'depthBb':d,'seed':s} for d,s in jobs],
                    'workers':min(args.workers,len(jobs)),'status':'started','validationStatus':'pilot_not_activated'}
        atomic_json(args.output_root / 'launcher.json',launcher); print(json.dumps(launcher)); return
    summary = args.output_root / 'summary.json'
    results = json.loads(summary.read_text(encoding='utf-8')).get('results',[]) if summary.exists() else []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job,args,depth,seed) for depth,seed in jobs]
        for future in as_completed(futures):
            result = future.result()
            results = merge_results(results, result)
            atomic_json(args.output_root / 'summary.json',{'schema':'hu-depth-training-pilot-v1','validationStatus':'pilot_not_activated','results':results})
            print(json.dumps({k:result.get(k) for k in ('depthBb','seed','status','elapsedSeconds','completedRounds')}),flush=True)

if __name__ == '__main__': main()

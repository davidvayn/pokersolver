#!/usr/bin/env python3
"""Bounded paired 50/100bb training; checkpointed research weights, never activation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from run_depth_pilots import JOBS, PILOT_GRID, ROOT, atomic_json, process_family
from long_run import training_seconds

PROFILE = {
    'target_round': 1_000_000, 'traversals_per_round': 400,
    'reservoir_capacity': 100_000, 'hidden_sizes': '256,128', 'batch_size': 1024,
    'steps_per_round': 100, 'artifact_every': 100, 'preflop_runout_samples': 256,
    'flop_runout_samples': 128, 'value_rollouts_per_action': 4,
    'learning_rate': .001, 'advantage_alpha': 2., 'variance_baseline_scale': .5,
    'replay_street_proposal': '.25,.25,.25,.25',
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def remaining_seconds(run: Path, hours: float) -> float:
    return max(0., hours * 3600 - training_seconds(run))

def trainer_command(python: str, run: Path, depth: int, seed: int,
                    seconds: float, sampler: Path, digest: str) -> list[str]:
    argv = [python, str(ROOT/'preflop-solver/neural/train.py'), '--run-dir', str(run),
            '--depth-bb', str(depth), '--seed', str(seed), '--max-minutes', str(seconds/60),
            '--sampler-binary', str(sampler), '--sampler-sha256', digest,
            '--sample-turn-rivers', '--action-abstraction-json',
            json.dumps(PILOT_GRID,sort_keys=True,separators=(',',':'))]
    for name,value in PROFILE.items():
        argv.extend(('--'+name.replace('_','-'),str(value)))
    return argv

def finite_metrics(value: object) -> bool:
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(finite_metrics(v) for v in value.values())
    if isinstance(value,list): return all(finite_metrics(v) for v in value)
    return True  # null denotes an unavailable held-out metric, not zero.

def progress(run: Path) -> dict:
    state_path = run/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    return {'completedRounds':state.get('completed_rounds',0),
            'completedTraversals':state.get('completed_traversals',0),
            'checkpoint':state.get('checkpoint'),
            'latestMetrics':(state.get('metrics') or [None])[-1],
            'artifacts':state.get('artifacts',[])}

def process_births() -> dict[int,str]:
    lines = subprocess.check_output(['ps','-axo','pid=,lstart='],text=True).splitlines()
    return {int(parts[0]):parts[1] for line in lines if len(parts:=line.strip().split(maxsplit=1))==2}

def terminate_owned(process: subprocess.Popen) -> None:
    # Capture births before terminating parents: a sampler has its own session
    # and can become orphaned. Re-check births to avoid targeting reused PIDs.
    births = process_births()
    owned = {pid:births[pid] for pid in process_family(process.pid) if pid in births}
    for pid in owned:
        try: os.kill(pid,signal.SIGTERM)
        except ProcessLookupError: pass
    deadline = time.monotonic()+10
    while time.monotonic()<deadline:
        process.poll()
        alive = process_births()
        if not any(alive.get(pid)==birth for pid,birth in owned.items()): break
        time.sleep(.25)
    alive = process_births()
    for pid,birth in owned.items():
        if alive.get(pid)==birth:
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass
    process.wait(timeout=10)

def run_job(args: argparse.Namespace, plan: dict, depth: int, seed: int,
            stop: threading.Event) -> dict:
    run = args.output_root/f'{depth}bb-seed{seed}'
    run.mkdir(exist_ok=True)
    info = {'depthBb':depth,'seed':seed,'output':str(run),'status':'queued',
            'validationStatus':'research_not_activated','targetHours':args.hours,
            'startedUtc':datetime.now(timezone.utc).isoformat()}
    process = None
    started = time.monotonic()
    try:
        if stop.is_set(): raise RuntimeError('controller stopped before this seed started')
        for path,digest in plan['pinnedInputs'].items():
            if sha256(Path(path))!=digest: raise RuntimeError('pinned training input changed')
        remaining = remaining_seconds(run,args.hours)
        if remaining<=0:
            info.update(status='budget_completed',**progress(run)); return info
        sampler = args.output_root/'bin/preflop-solver'
        argv = trainer_command(args.python,run,depth,seed,remaining,sampler,plan['samplerSha256'])
        info['command'] = argv
        with (run/'train.log').open('a') as log:
            process = subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            info.update(pid=process.pid,status='training')
            last_round = progress(run)['completedRounds']; last_progress = started; peak=0
            while process.poll() is None:
                snapshot = progress(run); rounds=snapshot['completedRounds']
                if rounds!=last_round: last_progress=time.monotonic(); last_round=rounds
                rss=sum(process_family(process.pid).values()); peak=max(peak,rss)
                metrics=snapshot['latestMetrics']
                reason = ('stopped_by_request' if stop.is_set() else
                    'stopped_memory' if rss>args.max_rss_gb*1024**3 else
                    'stopped_disk' if shutil.disk_usage(args.output_root).free<args.min_free_gb*1024**3 else
                    'stopped_deadline' if time.monotonic()-started>remaining+30*60 else
                    'stopped_no_progress' if time.monotonic()-last_progress>args.stall_minutes*60 else
                    'stopped_nonfinite' if not finite_metrics(metrics) else
                    'stopped_truncated_samples' if metrics and metrics.get('records_truncated') else None)
                info.update(**snapshot,peakCpuRssBytes=peak,elapsedSeconds=time.monotonic()-started)
                atomic_json(run/'run.json',info)
                if reason:
                    info['status']=reason
                    if reason in ('stopped_disk','stopped_by_request'): stop.set()
                    terminate_owned(process); break
                stop.wait(10)
            if info['status']=='training':
                info['status']='completed_training_budget' if process.returncode==0 else 'failed'
            info.update(exitCode=process.returncode,**progress(run))
    except Exception as error:
        info.update(status='failed',failure=str(error))
    finally:
        if process is not None and process.poll() is None: terminate_owned(process)
        info['elapsedSeconds']=time.monotonic()-started
        atomic_json(run/'run.json',info)
    return info

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--max-rss-gb',type=float,default=3.5)
    parser.add_argument('--min-free-gb',type=float,default=25)
    parser.add_argument('--stall-minutes',type=float,default=30)
    parser.add_argument('--only-seed',type=int,choices=tuple(s for _,s in JOBS))
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--keep-awake',action='store_true')
    args=parser.parse_args(); args.output_root=args.output_root.resolve()
    if not 8<=args.hours<=12 or not 0<args.max_rss_gb<=4 or args.min_free_gb<20 or args.stall_minutes<5:
        parser.error('full training requires 8–12h, <=4GiB per worker, >=20GiB disk floor and >=5min stall limit')
    args.output_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output_root).free<args.min_free_gb*1024**3:
        parser.error('not enough free disk for the full run')
    if args.detach:
        argv=[sys.executable,str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!='--detach']]
        with (args.output_root/'controller.log').open('a') as log:
            process=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        launcher={'pid':process.pid,'output':str(args.output_root),'hoursPerSeed':args.hours,
            'workers':args.workers,'validationStatus':'research_not_activated'}
        atomic_json(args.output_root/'launcher.json',launcher); print(json.dumps(launcher)); return
    with (args.output_root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
        sampler=args.output_root/'bin/preflop-solver'
        source=ROOT/'preflop-solver/target/release/preflop-solver'
        sampler.parent.mkdir(exist_ok=True)
        if not sampler.exists(): shutil.copy2(source,sampler); sampler.chmod(0o555)
        subprocess.run([str(sampler),'--help'],check=True,stdout=subprocess.DEVNULL,timeout=15)
        paths=[ROOT/'preflop-solver/neural/train.py',ROOT/'preflop-solver/neural/cash_profiles.py',
               ROOT/'scripts/policy/export-neural-artifact.mjs',sampler,Path(__file__).resolve()]
        plan={'schema':'hu-depth-full-run-v1','hoursPerSeed':args.hours,'workers':args.workers,
            'profile':PROFILE,'actionAbstraction':PILOT_GRID,'jobs':JOBS,
            'samplerSha256':sha256(sampler),'pinnedInputs':{str(p):sha256(p) for p in paths},
            'maxCpuRssGb':args.max_rss_gb,'minimumFreeDiskGb':args.min_free_gb,
            'validationStatus':'research_not_activated'}
        plan_path=args.output_root/'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=json.loads(json.dumps(plan)):
            raise RuntimeError('full-run plan or source changed; use a new output directory')
        atomic_json(plan_path,plan)
        awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]) if args.keep_awake else None
        try:
            jobs=[(d,s) for d,s in JOBS if args.only_seed is None or args.only_seed==s]
            summary=args.output_root/'summary.json'
            results=json.loads(summary.read_text()).get('results',[]) if summary.exists() else []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_job,args,plan,d,s,stop) for d,s in jobs]
                for future in as_completed(futures):
                    result=future.result()
                    results=[r for r in results if r['seed']!=result['seed']]+[result]
                    atomic_json(summary,{'schema':'hu-depth-full-run-v1','validationStatus':'research_not_activated','results':results})
                    print(json.dumps({k:result.get(k) for k in ('depthBb','seed','status','completedRounds','elapsedSeconds','failure')}),flush=True)
        finally:
            if awake is not None:
                awake.terminate(); awake.wait(timeout=5)

if __name__=='__main__': main()

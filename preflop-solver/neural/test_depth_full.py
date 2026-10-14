import json
from pathlib import Path
import tempfile
import unittest
import argparse
import signal
from unittest.mock import MagicMock, call, patch

from run_depth_full import PROFILE, finite_metrics, remaining_seconds, trainer_command, run_job, terminate_owned
import threading
from run_depth_pilots import JOBS, PILOT_GRID

class DepthFullTests(unittest.TestCase):
    def test_full_run_is_not_the_three_round_smoke_or_a_mislabeled_20bb_model(self):
        for depth,seed in JOBS:
            argv=trainer_command('python',Path('/run'),depth,seed,8*3600,Path('/frozen/sampler'),'a'*64)
            value=lambda flag:argv[argv.index(flag)+1]
            self.assertEqual(value('--depth-bb'),str(depth))
            self.assertEqual(value('--seed'),str(seed))
            self.assertEqual(value('--max-minutes'),'480.0')
            self.assertEqual(value('--target-round'),'1000000')
            self.assertEqual(value('--sampler-binary'),'/frozen/sampler')
            self.assertEqual(value('--sampler-sha256'),'a'*64)
            self.assertEqual(value('--traversals-per-round'),'400')
            self.assertEqual(value('--reservoir-capacity'),'100000')
            self.assertEqual(value('--value-rollouts-per-action'),'4')
            self.assertEqual(value('--artifact-every'),'100')
            self.assertEqual(json.loads(value('--action-abstraction-json')),PILOT_GRID)
            self.assertNotIn('--cash-profile',argv)
            self.assertNotIn('--networks',argv)

    def test_resume_consumes_the_total_completed_training_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            self.assertEqual(remaining_seconds(run,8),28800)
            (run/'metrics.jsonl').write_text('\n'.join(json.dumps({'elapsed_seconds':v}) for v in [100,200,300]))
            self.assertEqual(remaining_seconds(run,8),28200)
            self.assertEqual(remaining_seconds(run,.1),0)

    def test_guard_rejects_nans_but_does_not_fabricate_unavailable_metrics(self):
        self.assertTrue(finite_metrics({'heldout':None,'losses':[1.,.1]}))
        self.assertFalse(finite_metrics({'heldout':{'value':float('nan')}}))
        self.assertFalse(finite_metrics([1,float('inf')]))
        self.assertEqual(PROFILE['steps_per_round'],100)

    def test_memory_guard_stops_the_owned_worker_and_records_the_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(output_root=Path(directory),python='python',hours=8,
                max_rss_gb=3.5,min_free_gb=25,stall_minutes=30)
            process=MagicMock(pid=123,returncode=-15);process.poll.return_value=None
            def terminate(p): p.poll.return_value=-15
            with patch('run_depth_full.subprocess.Popen',return_value=process), \
                 patch('run_depth_full.process_family',return_value={123:4*1024**3}), \
                 patch('run_depth_full.terminate_owned',side_effect=terminate) as stop:
                result=run_job(args,{'pinnedInputs':{},'samplerSha256':'a'*64},50,50121,threading.Event())
            self.assertEqual(result['status'],'stopped_memory')
            self.assertEqual(result['exitCode'],-15)
            stop.assert_called_once_with(process)
            self.assertEqual(json.loads((Path(directory)/'50bb-seed50121/run.json').read_text())['status'],'stopped_memory')

    def test_cleanup_remembers_orphaned_children_without_killing_reused_pids(self):
        process=MagicMock(pid=100)
        with patch('run_depth_full.process_births',side_effect=[{100:'original',101:'child'}, {100:'new-process',101:'child'}]), \
             patch('run_depth_full.process_family',return_value={100:1,101:1}), \
             patch('run_depth_full.time.monotonic',side_effect=[0,11]), \
             patch('run_depth_full.os.kill') as kill:
            terminate_owned(process)
        self.assertEqual(kill.call_args_list,[call(100,signal.SIGTERM),call(101,signal.SIGTERM),call(101,signal.SIGKILL)])

if __name__=='__main__': unittest.main()

import unittest
import json
import argparse
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path
from run_depth_pilots import JOBS, PILOT_GRID, command, merge_results, run_job

class DepthPilotTests(unittest.TestCase):
    def test_paired_distinct_depths_and_seeds_are_not_rescaled_20bb_training(self):
        self.assertEqual({d for d,_ in JOBS},{50,100})
        self.assertEqual(len({s for _,s in JOBS}),4)
        for depth,seed in JOBS:
            argv=command('python',Path('/tmp/depth-pilot-test'),depth,seed,12)
            self.assertEqual(argv[argv.index('--depth-bb')+1],str(depth))
            self.assertEqual(argv[argv.index('--seed')+1],str(seed))
            self.assertNotIn('--cash-profile',argv)
            self.assertNotIn('--networks',argv)
            self.assertNotIn('--compact-serving-grid',argv)
            self.assertEqual(json.loads(argv[argv.index('--action-abstraction-json')+1]), PILOT_GRID)
            self.assertEqual(argv[argv.index('--target-round')+1],'3')

    def test_regular_sizes_are_capped_and_separated(self):
        for name, sizes in PILOT_GRID.items():
            if isinstance(sizes, list):
                self.assertLessEqual(len(sizes), 3, name)
                self.assertEqual(sizes, sorted(set(sizes)), name)
        self.assertEqual(PILOT_GRID['open_sizes_bb'], [2.0, 3.5, 5.0])
        self.assertGreaterEqual(min(b-a for a,b in zip(PILOT_GRID['three_bet_sizes_bb'],PILOT_GRID['three_bet_sizes_bb'][1:])),4)
        self.assertEqual(PILOT_GRID['postflop_raise_cap'],1)
        self.assertTrue(PILOT_GRID['include_all_in'])

    def test_retry_keeps_other_results_and_the_failed_attempt_record(self):
        previous = [{'depthBb':d,'seed':s,'status':'completed' if s!=100122 else 'failed'} for d,s in JOBS]
        retried = {'depthBb':100,'seed':100122,'status':'completed'}
        updated = merge_results(previous, retried)
        self.assertEqual(len(updated),4)
        self.assertTrue(all(r['status']=='completed' for r in updated))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'100bb-seed100122'; output.mkdir()
            (output/'pilot.json').write_text(json.dumps({'pid':123,'status':'failed','exitCode':1,'elapsedSeconds':30}))
            process = MagicMock(pid=124,returncode=0); process.poll.return_value=0
            args = argparse.Namespace(output_root=Path(directory),python='python',max_minutes=12,deadline_minutes=30,max_rss_gb=3.5)
            with patch('run_depth_pilots.subprocess.Popen',return_value=process):
                result = run_job(args,100,100122)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['previousAttempts'][0]['status'],'failed')
            self.assertEqual(result['previousAttempts'][0]['pid'],123)
            self.assertEqual(result['actionAbstraction'],PILOT_GRID)

if __name__=='__main__': unittest.main()

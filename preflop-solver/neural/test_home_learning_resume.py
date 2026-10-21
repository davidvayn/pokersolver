import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np


class HomeLearningResumeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('POKER_CERTIFICATE_TEST_BINARY'),'explicit native binary required')
    def test_cpu_prior_and_separate_average_budget_resume_deterministically(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = Path(__file__).with_name('train.py')
            binary = os.environ['POKER_CERTIFICATE_TEST_BINARY']
            def run(name, target, *, reject_precision=None):
                command = [sys.executable,str(trainer),'--run-dir',str(root/name),'--depth-bb','50',
                    '--seed','2919071','--target-round',str(target),'--sampler-binary',binary,
                    '--sampler-sha256',hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                    '--traversals-per-round','4','--steps-per-round','4','--average-policy-steps-per-round','8',
                    '--bootstrap-cpu-prior','--card-arrival-memory','--average-regret-rollouts',
                    '--hidden-sizes','8,4','--batch-size','8','--reservoir-capacity','2000',
                    '--sample-turn-rivers','--preflop-runout-samples','4','--flop-runout-samples','4',
                    '--value-rollouts-per-action','2','--artifact-every','1']
                environment = dict(os.environ)
                if reject_precision is not None:
                    environment['MLX_ENABLE_TF32']=reject_precision
                result = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=60)
                if reject_precision is not None:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('resume configuration differs',result.stderr)
                    return
                self.assertEqual(result.returncode,0,result.stderr[-4000:])
            run('continuous',4)
            run('resumed',2)
            run('resumed',4)
            a,b = [json.loads((root/name/'state.json').read_text()) for name in ('continuous','resumed')]
            self.assertEqual(a['config_hash'],b['config_hash'])
            self.assertEqual(a['config'].get('matmul_precision'),
                'full-fp32-v1' if os.environ.get('MLX_ENABLE_TF32')=='0' else None)
            self.assertEqual(a['numpy_rng_state'],b['numpy_rng_state'])
            for head in ('advantage_p0','advantage_p1','average_strategy','value'):
                weights = [mx.load(str(root/name/state['checkpoint']/f'{head}.safetensors'))
                           for name,state in (('continuous',a),('resumed',b))]
                self.assertEqual(weights[0].keys(),weights[1].keys())
                for key in weights[0]:
                    np.testing.assert_array_equal(np.asarray(weights[0][key]),np.asarray(weights[1][key]))
            before = (root/'resumed/state.json').read_bytes()
            run('resumed',4,reject_precision='1' if os.environ.get('MLX_ENABLE_TF32')=='0' else '0')
            self.assertEqual((root/'resumed/state.json').read_bytes(),before)

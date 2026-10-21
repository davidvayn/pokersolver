import tempfile
import unittest
from pathlib import Path

from depth_repair import training_command


class HomeLearningProfileTests(unittest.TestCase):
    def test_cpu_and_average_fit_are_separate_single_factor_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder)/'binary'
            binary.write_bytes(b'fixture')
            commands = {p:training_command(binary,Path(folder)/p,p,51701,24,128,240,depth=50)
                        for p in ('recall','recall-cpu','recall-fit')}
            for command in commands.values():
                self.assertIn('--card-arrival-memory',command)
                self.assertIn('--average-regret-rollouts',command)
                self.assertEqual(command[command.index('--steps-per-round')+1],'240')
                self.assertEqual(command[command.index('--traversals-per-round')+1],'128')
            self.assertNotIn('--bootstrap-cpu-prior',commands['recall'])
            self.assertIn('--bootstrap-cpu-prior',commands['recall-cpu'])
            self.assertNotIn('--average-policy-steps-per-round',commands['recall-cpu'])
            self.assertNotIn('--bootstrap-cpu-prior',commands['recall-fit'])
            fit = commands['recall-fit']
            self.assertEqual(fit[fit.index('--average-policy-steps-per-round')+1],'480')

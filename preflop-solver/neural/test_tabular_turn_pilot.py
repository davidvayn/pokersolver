import math
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from tabular_turn_pilot import load_recheck_reports, response_comparison_eligible


class ResponseComparisonTests(unittest.TestCase):
    def test_preflop_resampling_rejects_noops_and_unbounded_allocations_before_io(self):
        base = [sys.executable, str(Path(__file__).with_name('tabular_turn_pilot.py')),
                '--binary', 'unused', '--checkpoint-stage', 'unused',
                '--output-dir', 'unused', '--seed-offset', '123',
                '--response-preflop-runouts']
        for options, message in [(['--postflop-response-only'], 'require preflop response training'),
                                 (['--rollouts-per-action', '4097'], 'limited to 4096'),
                                 (['--arms', 'recheck', '--recheck-responses', 'unused'], 'inherits all training/profile settings')]:
            result = subprocess.run([*base, *options], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def report(self):
        return {'response_deployed': [True, True], 'players': [
            {'estimated_gain_bb': .4, 'gain_standard_error_bb': .02},
            {'estimated_gain_bb': .3, 'gain_standard_error_bb': .03},
        ]}

    def test_two_calibrated_responses_are_eligible(self):
        self.assertTrue(response_comparison_eligible(self.report()))

    def test_rejected_response_cannot_be_a_zero_gain_win(self):
        report = self.report()
        report['response_deployed'][0] = False
        report['players'][0]['estimated_gain_bb'] = 0.0
        self.assertFalse(response_comparison_eligible(report))

    def test_missing_seat_or_metrics_are_inconclusive(self):
        self.assertFalse(response_comparison_eligible({}))
        report = self.report()
        report['players'].pop()
        self.assertFalse(response_comparison_eligible(report))
        report = self.report()
        del report['players'][0]['gain_standard_error_bb']
        self.assertFalse(response_comparison_eligible(report))

    def test_nonfinite_or_negative_errors_are_inconclusive(self):
        for field, invalid in [('estimated_gain_bb', math.nan),
                               ('estimated_gain_bb', math.inf),
                               ('gain_standard_error_bb', math.inf),
                               ('gain_standard_error_bb', -.1)]:
            report = self.report()
            report['players'][0][field] = invalid
            self.assertFalse(response_comparison_eligible(report))

    def test_signed_negative_holdout_is_not_clamped(self):
        # Eligibility is not a quality win. A response accepted on separate
        # calibration data may have a negative noisy held-out point estimate.
        report = self.report()
        report['players'][0]['estimated_gain_bb'] = -.01
        self.assertTrue(response_comparison_eligible(report))

    def test_recheck_pins_original_reports_and_rejects_ambiguous_or_chained_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'response.json'
            report = {'schema': 'hu-tabular-checkpoint-information-set-response-v1',
                      'policy_sha256': 'a' * 64, 'seed': 123}
            path.write_text(json.dumps(report))
            pinned = load_recheck_reports([path])['a' * 64]
            self.assertEqual(pinned[0], path.resolve())
            self.assertEqual(len(pinned[1]), 64)
            self.assertEqual(pinned[2], 123)
            with self.assertRaises(ValueError):
                load_recheck_reports([path, path])
            report['retained_training'] = {'seed': 122}
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                load_recheck_reports([path])


if __name__ == '__main__':
    unittest.main()

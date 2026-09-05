import math
import unittest

from tabular_turn_pilot import response_comparison_eligible


class ResponseComparisonTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()

import unittest

from distill_checkpoint_ensemble import parse_round_weights


class DistillCheckpointEnsembleTests(unittest.TestCase):
    def test_round_weights_are_normalized(self):
        self.assertEqual(parse_round_weights("100:8,200:2"), [(100, 0.8), (200, 0.2)])

    def test_round_weights_reject_non_positive_values(self):
        with self.assertRaises(ValueError):
            parse_round_weights("100:0")


if __name__ == "__main__":
    unittest.main()

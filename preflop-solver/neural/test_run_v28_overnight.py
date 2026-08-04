import unittest

from run_v28_overnight import select_candidate, should_extend


def candidate(label: str, mean: float, worst: float, stable: bool = True):
    return {
        "label": label,
        "meanValidationExploitabilityBbPerHand": mean,
        "worstValidationExploitabilityBbPerHand": worst,
        "passesStability": stable,
    }


class RunV28OvernightTests(unittest.TestCase):
    def test_extension_requires_stability_progress_and_a_bounded_validation_result(self):
        ten = candidate("10m", 0.18, 0.19)
        self.assertTrue(should_extend(ten, candidate("100m", 0.16, 0.18)))
        self.assertFalse(should_extend(ten, candidate("100m", 0.179, 0.18)))
        self.assertFalse(should_extend(ten, candidate("100m", 0.14, 0.21)))
        self.assertFalse(should_extend(ten, candidate("100m", 0.08, 0.09, False)))

    def test_a_subpoint_one_stable_result_always_earns_the_long_checkpoint(self):
        self.assertTrue(
            should_extend(
                candidate("10m", 0.08, 0.09),
                candidate("100m", 0.081, 0.09),
            )
        )

    def test_selection_uses_validation_mean_and_rejects_unstable_candidates(self):
        selected = select_candidate(
            [
                candidate("unstable-best", 0.01, 0.02, False),
                candidate("stable-second", 0.10, 0.11),
                candidate("stable-third", 0.12, 0.13),
            ]
        )
        self.assertEqual(selected["label"], "stable-second")

    def test_selection_fails_when_no_pair_passes_stability(self):
        with self.assertRaisesRegex(RuntimeError, "no paired candidate"):
            select_candidate([candidate("bad", 0.01, 0.02, False)])


if __name__ == "__main__":
    unittest.main()

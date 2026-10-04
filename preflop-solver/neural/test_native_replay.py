import copy
import unittest

import numpy as np

import native_replay as replay
import native_value_dataset as native


class NativeReplayTests(unittest.TestCase):
    def test_exact_target_and_capture_prefix_required(self):
        reference = dict(schema=native.SCHEMA, game={"effective_stack_bb": 20},
                         targets=[{"board": [0, 1, 2, 3]}, {"board": [4, 5, 6, 7]}],
                         source_captures=[{"states": 2, "capture_sha256": "a" * 64}])
        source = copy.deepcopy(reference)
        source["targets"].append({"board": [8, 9, 10, 11]})
        source["source_captures"].append({"states": 1})
        self.assertEqual(replay.reference_boundary(source, reference), 2)
        source["targets"][0]["board"][0] = 51
        with self.assertRaisesRegex(ValueError, "prefix"):
            replay.reference_boundary(source, reference)
        source = copy.deepcopy(reference)
        source["targets"].append({"board": [8, 9, 10, 11]})
        source["source_captures"][0]["capture_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "provenance"):
            replay.reference_boundary(source, reference)

    def test_partition_excludes_held_out_states_and_handles_augmentations(self):
        groups = np.repeat(np.arange(8), 2)
        first, second = replay.partition_rows(groups, np.array([0, 2, 4, 6]),
                                              np.array([1, 5]), np.array([3, 7]), 4)
        np.testing.assert_array_equal(groups[first], [0, 0, 2, 2])
        np.testing.assert_array_equal(groups[second], [4, 4, 6, 6])
        with self.assertRaisesRegex(ValueError, "overlap"):
            replay.partition_rows(groups, np.array([0, 1, 2, 4, 6]),
                                  np.array([1, 5]), np.array([3, 7]), 4)

    def test_rotating_cohort_preserves_exact_pot_quotas(self):
        invested = np.repeat([[1., 1.], [5., 5.], [10., 10.]], 2, axis=0)
        sampler = replay.NativeReplaySampler(np.array([0, 2, 4]), np.array([1, 3, 5]),
                                              invested, 8, .875)
        rng = np.random.default_rng(17)
        for step in range(80):
            chosen = sampler.sample(rng, step)
            self.assertEqual(sum(row % 2 == 0 for row in chosen), 7)
            np.testing.assert_array_equal(np.bincount(sampler.bands[chosen]), [3, 3, 2])
        report = sampler.report()
        self.assertEqual(report["sampledRowsByOriginPot"]["appended"],
                         dict(small=30, medium=30, large=20))
        self.assertEqual(report["sampledRowsByOriginPot"]["retained"],
                         dict(small=210, medium=210, large=140))
        self.assertEqual(sum(sum(v.values()) for v in
                             report["emptyBandFallbacksByOriginPot"].values()), 0)

    def test_empty_band_fallback_is_explicit_and_never_changes_inputs(self):
        invested = np.array([[1., 1.], [5., 5.], [10., 10.], [1., 1.]])
        before = invested.copy()
        sampler = replay.NativeReplaySampler(np.array([0, 1, 2]), np.array([3]),
                                              invested, 8, .875)
        for step in range(8):
            sampler.sample(np.random.default_rng(step), step)
        self.assertEqual(sampler.report()["emptyBandFallbacksByOriginPot"]["appended"],
                         dict(small=0, medium=3, large=2))
        np.testing.assert_array_equal(invested, before)

    def test_sampler_is_deterministic_and_supports_weighted_rows(self):
        invested = np.tile([[1., 1.]], (4, 1))
        weights = np.array([1., 99., 1., 1.])
        outputs = []
        for _ in range(2):
            sampler = replay.NativeReplaySampler(np.array([0, 1]), np.array([2, 3]),
                                                  invested, 8, .875, weights)
            rng = np.random.default_rng(123)
            outputs.append(np.concatenate([sampler.sample(rng, n) for n in range(80)]))
        np.testing.assert_array_equal(*outputs)
        self.assertGreater(np.mean(outputs[0] == 1), .8)


if __name__ == "__main__":
    unittest.main()

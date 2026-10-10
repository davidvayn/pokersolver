import unittest
from unittest.mock import Mock

import mlx.core as mx
import numpy as np

from cash_feature_cache import CashFeatureCache


def features():
    return tuple(np.arange(6, dtype=np.float32).reshape(2, 3).copy() + index
                 for index in range(8))


class CashFeatureCacheTests(unittest.TestCase):
    def test_same_content_and_schema_prepare_once_with_exact_readonly_arrays(self):
        arrays = features()
        expected = [array.copy() for array in arrays]
        prepare = Mock(return_value=arrays)
        cache = CashFeatureCache()
        first, miss = cache.get("a" * 64, "features-v2", prepare)
        second, hit = cache.get("a" * 64, "features-v2", prepare)
        prepare.assert_called_once_with()
        self.assertIs(first, second)
        self.assertFalse(miss["cache_hit"])
        self.assertTrue(hit["cache_hit"])
        self.assertTrue(hit["retained"])
        self.assertEqual(hit["array_bytes"], sum(a.nbytes for a in arrays))
        self.assertEqual(hit["dataset_sha256"], "a" * 64)
        self.assertEqual(hit["feature_schema"], "features-v2")
        self.assertEqual(hit["preparation_seconds"], 0.)
        for actual, frozen in zip(second, expected):
            np.testing.assert_array_equal(actual, frozen)
            self.assertFalse(actual.flags.writeable)
            with self.assertRaises(ValueError):
                actual[0, 0] = -1
        # These arrays cross the trainer's real NumPy -> MLX seam.
        tensors = [mx.array(a) for a in second]
        mx.eval(tensors)
        for tensor, frozen in zip(tensors, expected):
            np.testing.assert_array_equal(np.array(tensor), frozen)

    def test_content_or_schema_changes_miss_and_only_last_entry_is_retained(self):
        cache = CashFeatureCache()
        prepare = Mock(side_effect=features)
        for digest, schema in [("a" * 64, "v2"), ("b" * 64, "v2"),
                               ("b" * 64, "v3"), ("a" * 64, "v2")]:
            _, info = cache.get(digest, schema, prepare)
            self.assertFalse(info["cache_hit"])
        self.assertEqual(prepare.call_count, 4)
        _, info = cache.get("a" * 64, "v2", prepare)
        self.assertTrue(info["cache_hit"])
        self.assertEqual(prepare.call_count, 4)

    def test_oversize_features_are_valid_but_never_retained(self):
        cache = CashFeatureCache(maximum_bytes=1)
        prepare = Mock(side_effect=features)
        for _ in range(2):
            arrays, info = cache.get("c" * 64, "v2", prepare)
            self.assertFalse(info["cache_hit"])
            self.assertFalse(info["retained"])
            self.assertTrue(all(not a.flags.writeable for a in arrays))
            self.assertGreater(info["array_bytes"], info["maximum_retained_bytes"])
        self.assertEqual(prepare.call_count, 2)

    def test_budget_boundary_retains_exact_size(self):
        arrays = features()
        cache = CashFeatureCache(maximum_bytes=sum(a.nbytes for a in arrays))
        prepare = Mock(return_value=arrays)
        _, info = cache.get("d" * 64, "v2", prepare)
        self.assertTrue(info["retained"])
        cache.get("d" * 64, "v2", prepare)
        prepare.assert_called_once_with()

    def test_failed_preparation_does_not_poison_key_or_retain_previous_entry(self):
        cache = CashFeatureCache()
        cache.get("a" * 64, "v2", features)
        prepare = Mock(side_effect=[RuntimeError("preparation failed"), features()])
        with self.assertRaisesRegex(RuntimeError, "preparation failed"):
            cache.get("b" * 64, "v2", prepare)
        _, info = cache.get("b" * 64, "v2", prepare)
        self.assertFalse(info["cache_hit"])
        self.assertTrue(info["retained"])
        old = Mock(side_effect=features)
        _, info = cache.get("a" * 64, "v2", old)
        self.assertFalse(info["cache_hit"])
        old.assert_called_once_with()

    def test_invalid_identity_and_budget_are_rejected_before_preparation(self):
        for budget in (0, -1, True, 1.5, None):
            with self.assertRaisesRegex(ValueError, "budget"):
                CashFeatureCache(maximum_bytes=budget)
        cache = CashFeatureCache()
        prepare = Mock(side_effect=features)
        for digest in (None, "", "g" * 64, "a" * 63, "a" * 65, "A" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                cache.get(digest, "v2", prepare)
        for schema in (None, "", 2, True):
            with self.assertRaisesRegex(ValueError, "schema"):
                cache.get("a" * 64, schema, prepare)
        prepare.assert_not_called()

    def test_malformed_features_are_not_cached_or_partially_frozen(self):
        cache = CashFeatureCache()
        good = features()
        for bad in (list(good), good[:7], good + (good[0],),
                    good[:7] + (None,), good[:7] + (good[7].astype(np.float64),),
                    good[:7] + (good[7][:1],)):
            with self.assertRaisesRegex(ValueError, "eight owned FP32"):
                cache.get("e" * 64, "v2", lambda: bad)
            self.assertTrue(all(a.flags.writeable for a in good))
        prepare = Mock(return_value=good)
        _, info = cache.get("e" * 64, "v2", prepare)
        self.assertFalse(info["cache_hit"])
        prepare.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

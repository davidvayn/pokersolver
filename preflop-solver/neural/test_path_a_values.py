import copy
import unittest
import numpy as np
import native_value_dataset as native
from run_path_a_values import forced_call_public


class ForcedCallTests(unittest.TestCase):
    def test_restore_preaction_range_including_zero_call_hands(self):
        ranks = "23456789TJQKA"
        def label(a, b):
            high, low = max(a//4,b//4),min(a//4,b//4)
            return ranks[high]+ranks[low]+("" if high==low else "s" if a%4==b%4 else "o")
        labels = sorted({label(a,b) for a,b in native.COMBOS})
        prior = [0.0]*169; prior[labels.index("AA")] = 1.0
        public = dict(board=[0,5,10], ranges=[[0.0]*1326, [1/1326]*1326], trajectory=["unchanged"])
        row = dict(seat=0, ownPrefixReach=prior, publicInput=public)
        original = copy.deepcopy(row)
        actual = forced_call_public(row, labels)
        self.assertEqual(row, original)
        self.assertEqual(actual["ranges"][1], public["ranges"][1])
        self.assertEqual(actual["trajectory"], public["trajectory"])
        weights = np.asarray(actual["ranges"][0])
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertEqual(np.count_nonzero(weights), 6)
        for i,(a,b) in enumerate(native.COMBOS):
            if a in public["board"] or b in public["board"]: self.assertEqual(weights[i], 0)
        row["publicInput"]["board"][0] = 48
        blocked = forced_call_public(row, labels)["ranges"][0]
        self.assertEqual(np.count_nonzero(blocked), 3)
        self.assertAlmostEqual(sum(blocked), 1)
        for i,(a,b) in enumerate(native.COMBOS):
            if 48 in (a,b): self.assertEqual(blocked[i], 0)
        row["ownPrefixReach"] = [0.0]*169
        with self.assertRaises(ValueError): forced_call_public(row, labels)


if __name__ == "__main__": unittest.main()

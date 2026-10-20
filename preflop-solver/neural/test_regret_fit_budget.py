import unittest
import numpy as np
from probe_regret_fit_budget import split_indices


class FixedFitSplitTests(unittest.TestCase):
    def test_repeated_information_sets_cannot_leak_across_split(self):
        states = np.repeat(np.arange(200,dtype=np.float16)[:,None],716,axis=1)
        states = np.concatenate([states,states])
        train,test = split_indices(states)
        self.assertTrue(len(train) > 100 and len(test) > 50)
        self.assertFalse(set(states[train,0]) & set(states[test,0]))
        again = split_indices(states)
        np.testing.assert_array_equal(train,again[0])
        np.testing.assert_array_equal(test,again[1])

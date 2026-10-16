import unittest
from probe_snapshot_compression import family_splits


class SnapshotCompressionProbeTests(unittest.TestCase):
    def test_whole_families_stay_in_one_split_including_duplicate_policy_trajectories(self):
        records=[{'iteration':i} for _ in range(2) for i in range(32)]
        train,tune,test=family_splits(records,1701)
        self.assertFalse(train&tune or train&test or tune&test)
        self.assertEqual(train|tune|test,set(range(32)))
        self.assertEqual([train,tune,test],family_splits(records,1701))


if __name__=='__main__':unittest.main()

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from distill_tabular_preflop import batch_features, dataset_arrays
from train import (
    ACTION_FEATURE_COUNT,
    MAX_POLICY_ACTIONS,
    STATE_FEATURE_COUNT,
    STATE_FEATURE_SCHEMA,
)


class DistillTabularPreflopTests(unittest.TestCase):
    def test_batch_features_broadcasts_state_across_actions(self):
        data = {
            "states": np.ones((2, STATE_FEATURE_COUNT), dtype=np.float32),
            "actions": np.zeros(
                (2, MAX_POLICY_ACTIONS, ACTION_FEATURE_COUNT), dtype=np.float32
            ),
        }
        features = np.asarray(batch_features(data, np.asarray([1])))
        self.assertEqual(features.shape, (1, MAX_POLICY_ACTIONS, 725))
        np.testing.assert_array_equal(features[0, 0, :STATE_FEATURE_COUNT], 1.0)

    def test_dataset_arrays_streams_declared_records(self):
        metadata = {
            "schema": "hu-neural-traversal-jsonl-v7",
            "state_feature_count": STATE_FEATURE_COUNT,
            "state_feature_schema": STATE_FEATURE_SCHEMA,
            "action_feature_count": ACTION_FEATURE_COUNT,
            "depth_bb": 20,
            "records": 1,
        }
        record = {
            "actions": [{"kind": "fold", "amount_to_bb": None}],
            "targets": [1.0],
            "weight": 2.0,
            "state": {
                "actor": 0,
                "street": "preflop",
                "button": 0,
                "private_cards": [51, 50],
                "board": [],
                "pot_bb": 1.5,
                "stacks_bb": [19.5, 19.0],
                "street_bets_bb": [0.5, 1.0],
                "total_committed_bb": [0.5, 1.0],
                "to_call_bb": 0.5,
                "last_full_raise_bb": 1.0,
                "raise_reopened": True,
                "trajectory": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata) + "\n")
                stream.write(json.dumps(record) + "\n")
            loaded_metadata, data = dataset_arrays(path)
        self.assertEqual(loaded_metadata["records"], 1)
        self.assertEqual(data["states"].shape, (1, STATE_FEATURE_COUNT))
        self.assertEqual(float(data["weights"][0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()

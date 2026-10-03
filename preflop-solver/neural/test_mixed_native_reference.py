"""Training-only stronger-label assembly keeps the retained split frozen."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mixed_native_reference import append_training_labels


class StrongerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.train_board = [2, 12, 15, 19]
        self.tuning_board = [3, 24, 49, 36]
        self.holdout_board = [1, 25, 50, 37]
        self.retained = dict(schema="hu-native-turn-cfv-dataset-v1",
                             game={"effective_stack_bb": 20.0},
                             turn_iterations=64, flop_iterations=32,
                             observed_queries=256,
                             source_captures=[{"input_sha256": "1" * 64,
                                               "policy_sha256": "2" * 64,
                                               "capture_sha256": "3" * 64,
                                               "states": 256, "flop_iterations": 32,
                                               "turn_iterations": 64}],
                             targets=[{"board": self.train_board, "source_capture_index": 0}
                                      for _ in range(254)] +
                                     [{"board": self.tuning_board, "source_capture_index": 0},
                                      {"board": self.holdout_board, "source_capture_index": 0}])
        self.split_reference = copy.deepcopy(self.retained)
        distributions = ["learned_flop_search_early_belief_native_label",
                         "learned_flop_search_middle_belief_native_label",
                         "learned_flop_search_late_belief_native_label",
                         "learned_flop_final_average_belief_native_label"]
        self.capture = dict(game=self.retained["game"],
                            capture_selection="stratified_learned_search_and_final_average_beliefs_with_native_labels",
                            proposal_policy_kind="frozen_learned_leaf_search_only",
                            proposal_turn_iterations=64, turn_iterations=256,
                            native_label_queries=4, flop_iterations=32,
                            source_public_input_sha256="4" * 64,
                            source_policy_sha256="5" * 64,
                            proposal_model_sha256="6" * 64,
                            seed=100101, sampling_seed=77,
                            observed_queries=8,
                            targets=[{"board": self.train_board, "state_distribution": name}
                                     for name in distributions])

    def test_appends_capture_specific_budget_without_changing_evaluation(self):
        old_train = np.arange(254)
        expanded_train = np.r_[old_train, np.arange(256, 260)]
        split = [(old_train, np.array([254]), np.array([255])),
                 (expanded_train, np.array([254]), np.array([255]))]
        with (patch("mixed_native_reference.native.validate_dataset"),
              patch("mixed_native_reference.native.family_split", side_effect=split),
              patch("mixed_native_reference.read_capture", return_value=self.capture),
              patch("mixed_native_reference.sha256", return_value="a" * 64)):
            result = append_training_labels(self.retained, self.split_reference, [Path("capture.json.gz")])
        self.assertEqual(result["turn_iterations"], 64)
        self.assertEqual(result["source_captures"][-1]["turn_iterations"], 256)
        self.assertEqual(result["targets"][:256], self.retained["targets"])
        self.assertTrue(all(row["source_capture_index"] == 1 for row in result["targets"][256:]))

    def test_rejects_new_label_on_frozen_tuning_family(self):
        contaminated = copy.deepcopy(self.capture)
        contaminated["targets"][0]["board"] = self.tuning_board
        with (patch("mixed_native_reference.native.validate_dataset"),
              patch("mixed_native_reference.native.family_split",
                    return_value=(np.arange(254), np.array([254]), np.array([255]))),
              patch("mixed_native_reference.read_capture", return_value=contaminated),
              patch("mixed_native_reference.sha256", return_value="a" * 64)):
            with self.assertRaisesRegex(ValueError, "tuning, holdout"):
                append_training_labels(self.retained, self.split_reference, [Path("capture.json.gz")])


if __name__ == "__main__":
    unittest.main()

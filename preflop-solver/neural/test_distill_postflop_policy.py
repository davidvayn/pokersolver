import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from distill_postflop_policy import (
    bounded_expected_regret_bb,
    corpus_groups,
    resolved_ev_regret_cap_bb,
    split_indices,
)


class PostflopPolicyDistillationTest(unittest.TestCase):
    def write_corpus(self, path: Path, status: str, boards: list[list[int]]) -> None:
        metadata = {
            "teacher": {
                "schema": "hu-range-conditioned-postflop-action-teacher-v1",
                "validation": {"status": status},
            }
        }
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata) + "\n")
            for board in boards:
                stream.write(
                    json.dumps({"state": {"street": "flop", "board": board}}) + "\n"
                )

    def test_board_groups_do_not_leak_between_training_and_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl.gz"
            boards = [[0, 1, 2]] * 4 + [[3, 4, 5]] * 4
            self.write_corpus(path, "accepted_for_training", boards)
            groups, streets = corpus_groups(path)
            training, heldout = split_indices(groups)
            self.assertEqual(streets, {"flop": 8, "turn": 0, "river": 0})
            self.assertFalse(
                set(groups[training].tolist()).intersection(groups[heldout].tolist())
            )

    def test_rejected_teacher_cannot_enter_distillation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl.gz"
            self.write_corpus(path, "rejected", [[0, 1, 2]])
            with self.assertRaisesRegex(ValueError, "not accepted for training"):
                corpus_groups(path)

    def test_preflop_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl.gz"
            metadata = {
                "teacher": {
                    "schema": "hu-range-conditioned-postflop-action-teacher-v1",
                    "validation": {"status": "accepted_for_training"},
                }
            }
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata) + "\n")
                stream.write(
                    json.dumps({"state": {"street": "preflop", "board": []}}) + "\n"
                )
            with self.assertRaisesRegex(ValueError, "preflop or unknown"):
                corpus_groups(path)

    def test_bounded_expected_regret_ignores_masks_and_caps_losses(self) -> None:
        predicted = np.asarray([[0.25, 0.25, 0.5], [0.5, 0.5, 0.0]])
        values = np.asarray([[1.0, -9.0, 0.0], [2.0, 2.0, -100.0]])
        masks = np.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        np.testing.assert_allclose(
            bounded_expected_regret_bb(predicted, values, masks, 0.4),
            [0.2, 0.0],
        )

    def test_dominated_action_mass_increases_expected_regret(self) -> None:
        values = np.asarray([[1.0, 0.0]])
        masks = np.ones_like(values)
        low = bounded_expected_regret_bb(
            np.asarray([[0.9, 0.1]]), values, masks, 5.0
        )
        high = bounded_expected_regret_bb(
            np.asarray([[0.1, 0.9]]), values, masks, 5.0
        )
        self.assertGreater(float(high[0]), float(low[0]))

    def test_default_ev_regret_cap_covers_the_depth_utility_span(self) -> None:
        self.assertEqual(resolved_ev_regret_cap_bb(None, 20.0), 40.0)
        self.assertEqual(resolved_ev_regret_cap_bb(5.0, 20.0), 5.0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            resolved_ev_regret_cap_bb(float("inf"), 20.0)


if __name__ == "__main__":
    unittest.main()

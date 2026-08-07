import gzip
import json
import tempfile
import unittest
from pathlib import Path

from distill_postflop_policy import corpus_groups, split_indices


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


if __name__ == "__main__":
    unittest.main()

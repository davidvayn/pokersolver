import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import validate_resolver_reach_corpus as module


class ResolverReachCorpusValidationTests(unittest.TestCase):
    @staticmethod
    def hashed(path: Path) -> dict[str, str]:
        return {
            "path": str(path.relative_to(path.parents[1])),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def config(self, root: Path) -> tuple[Path, dict]:
        neural = root / "neural"
        neural.mkdir()
        first = neural / "first.json"
        second = neural / "second.json"
        first.write_text('{"source":1}\n')
        second.write_text('{"source":2}\n')
        legacy = neural / "legacy.json"
        legacy.write_text(
            json.dumps({"targets": [{"resolver_root_board": [14, 18, 22]}]})
        )
        training = ["2c,7d,Jh", "3d,9h,Ac", "2c,4d,5h"]
        evaluation = ["4h,Tc,Ks", "7c,9d,Th", "4c,Jc,Ad"]

        def coverage(boards: list[str]) -> dict[str, int]:
            return dict(
                Counter(module.flop_texture_key(module.parse_board(board)) for board in boards)
            )

        payload = {
            "schema": module.SCHEMA,
            "activationAllowed": False,
            "sourceValueNetworks": [
                {"trainingSeed": 1, **self.hashed(first)},
                {"trainingSeed": 2, **self.hashed(second)},
            ],
            "trainingShards": [
                {
                    "seed": 11,
                    "sourceTrainingSeed": 1,
                    "boards": training,
                    "expectedStateCount": 9,
                    "output": "neural/missing-training.json",
                }
            ],
            "reservedEvaluationShards": [
                {
                    "seed": 12,
                    "sourceTrainingSeed": 2,
                    "boards": evaluation,
                    "expectedStateCount": 9,
                    "output": "neural/missing-evaluation.json",
                }
            ],
            "legacyDiagnosticDatasets": [self.hashed(legacy)],
            "separationPolicy": {
                "trainingRootCount": 3,
                "reservedEvaluationRootCount": 3,
            },
            "coverage": {
                "training": coverage(training),
                "reservedEvaluation": coverage(evaluation),
            },
        }
        path = neural / "config.json"
        path.write_text(json.dumps(payload))
        return path, payload

    def test_suit_isomorphism_ignores_suit_names_and_card_order(self) -> None:
        first = module.parse_board("2c,7d,Jh")
        second = module.parse_board("Js,2h,7c")
        self.assertEqual(
            module.suit_isomorphism_key(first), module.suit_isomorphism_key(second)
        )
        self.assertNotEqual(
            module.suit_isomorphism_key(first),
            module.suit_isomorphism_key(module.parse_board("2c,7c,Jh")),
        )

    def test_valid_freeze_is_accepted_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            path, _ = self.config(root)
            result = module.validate_config(path, root)
            self.assertEqual(result["status"], "accepted")
            self.assertFalse(result["activationAllowed"])
            self.assertEqual(result["trainingRootCount"], 3)
            self.assertEqual(result["completedShards"]["training"], [])

    def test_training_evaluation_isomorphic_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            path, payload = self.config(root)
            payload["reservedEvaluationShards"][0]["boards"][0] = "Js,2h,7c"
            payload["coverage"]["reservedEvaluation"] = dict(
                Counter(
                    module.flop_texture_key(module.parse_board(board))
                    for board in payload["reservedEvaluationShards"][0]["boards"]
                )
            )
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "suit-isomorphic"):
                module.validate_config(path, root)


if __name__ == "__main__":
    unittest.main()

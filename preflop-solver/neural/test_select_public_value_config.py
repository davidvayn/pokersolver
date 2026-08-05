import json
import tempfile
import unittest
from pathlib import Path

import select_public_value_config as module


class PublicValueConfigSelectionTests(unittest.TestCase):
    @staticmethod
    def report(directory: Path, name: str, tuning: list[float], validation: float) -> Path:
        weights = []
        variants = []
        for offset, value in enumerate(tuning):
            weight = directory / f"{name}-seed-{offset}.json"
            weight.write_text(f'{{"seed":{offset}}}\n')
            weights.append(weight)
            variants.append(
                {
                    "seed": offset + 1,
                    "weights": weight.name,
                    "metrics": {
                        "bestTuningRmseBb": value,
                        "finalTuningMetrics": {"weightedRmseBb": value},
                        "weightedRmseBb": validation,
                    },
                }
            )
        payload = {
            "schema": module.REPORT_SCHEMA,
            "sourceValidation": {"status": "accepted"},
            "datasetSha256": "a" * 64,
            "componentDatasetSha256": ["b" * 64],
            "sourcePolicySha256": "c" * 64,
            "splitSeed": 9,
            "trainStates": [0],
            "tuningStates": [1],
            "validationStates": [2],
            "architecture": name,
            "variants": {"range": variants},
            "tuningCrossSeedPredictionCorrelation": {"range": 0.99},
            "twoSeedOutputEnsembleMetrics": {
                "range": {"tuning": {"weightedRmseBb": sum(tuning) / 2}}
            },
        }
        path = directory / f"{name}-report.json"
        path.write_text(json.dumps(payload))
        return path

    def test_selection_uses_tuning_and_ignores_better_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            better_tuning = self.report(directory, "better-tuning", [0.20, 0.21], 9.0)
            better_validation = self.report(
                directory, "better-validation", [0.22, 0.23], 0.01
            )
            result = module.select_candidate([better_tuning, better_validation])
            self.assertFalse(result["holdoutMetricsConsulted"])
            self.assertEqual(
                result["selectedConfiguration"]["architecture"], "better-tuning"
            )

    def test_selection_rejects_mismatched_split_or_unrestored_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = self.report(directory, "first", [0.20, 0.21], 0.3)
            second = self.report(directory, "second", [0.19, 0.20], 0.3)
            payload = json.loads(second.read_text())
            payload["tuningStates"] = [3]
            second.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "identical data"):
                module.select_candidate([first, second])

            payload["tuningStates"] = [1]
            payload["variants"]["range"][0]["metrics"]["finalTuningMetrics"][
                "weightedRmseBb"
            ] = 0.5
            second.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "restore"):
                module.select_candidate([second])


if __name__ == "__main__":
    unittest.main()

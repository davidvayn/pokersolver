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

    def test_replicated_configuration_is_ranked_across_every_seed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            stable = self.report(directory, "stable", [0.20, 0.21], 9.0)
            first = self.report(directory, "replicated", [0.18, 0.19], 0.01)
            first_payload = json.loads(first.read_text())
            first_payload["architecture"] = "pooled"
            first.write_text(json.dumps(first_payload))
            second = self.report(directory, "replicated-confirm", [0.17, 0.24], 0.01)
            second_payload = json.loads(second.read_text())
            second_payload["architecture"] = "pooled"
            for offset, variant in enumerate(second_payload["variants"]["range"]):
                variant["seed"] = 11 + offset
            second.write_text(json.dumps(second_payload))

            result = module.select_candidate([stable, first, second])

            self.assertEqual(result["selectedConfiguration"]["architecture"], "stable")
            pooled = next(
                group
                for group in result["configurationGroups"]
                if group["configuration"]["architecture"] == "pooled"
            )
            self.assertEqual(pooled["reportCount"], 2)
            self.assertEqual(pooled["trainingSeeds"], [1, 2, 11, 12])
            self.assertEqual(pooled["maximumSeedTuningRmseBb"], 0.24)

    def test_replicated_configuration_rejects_reused_seed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = self.report(directory, "first", [0.20, 0.21], 0.3)
            second = self.report(directory, "second", [0.19, 0.20], 0.3)
            payload = json.loads(second.read_text())
            payload["architecture"] = "first"
            second.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "reuse a training seed"):
                module.select_candidate([first, second])

    def test_per_dataset_replay_weights_are_part_of_configuration_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = self.report(directory, "first", [0.20, 0.21], 0.3)
            second = self.report(directory, "second", [0.19, 0.20], 0.3)
            for path, weights in ((first, [1.0, 0.5]), (second, [0.5, 1.0])):
                payload = json.loads(path.read_text())
                payload["architecture"] = "pooled"
                payload["supplementalDatasetSamplingWeights"] = weights
                path.write_text(json.dumps(payload))

            result = module.select_candidate([first, second])

            self.assertEqual(len(result["configurationGroups"]), 2)
            self.assertEqual(
                result["selectedConfiguration"]["supplementalDatasetSamplingWeights"],
                [0.5, 1.0],
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

    def test_exact_tie_is_invariant_to_validation_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = self.report(directory, "alpha", [0.20, 0.21], 0.01)
            second = self.report(directory, "omega", [0.20, 0.21], 9.0)
            before = module.select_candidate([first, second])
            selected_before = before["selectedConfiguration"]["architecture"]

            for path, validation in ((first, 99.0), (second, -99.0)):
                payload = json.loads(path.read_text())
                for variant in payload["variants"]["range"]:
                    variant["metrics"]["weightedRmseBb"] = validation
                path.write_text(json.dumps(payload))

            after = module.select_candidate([first, second])
            self.assertEqual(
                after["selectedConfiguration"]["architecture"], selected_before
            )
            self.assertEqual(
                after["candidates"][0]["selectionTieBreaker"],
                before["candidates"][0]["selectionTieBreaker"],
            )


if __name__ == "__main__":
    unittest.main()

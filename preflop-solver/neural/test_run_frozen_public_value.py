import unittest

import run_frozen_public_value as module


class FrozenPublicValueRunTests(unittest.TestCase):
    def config(self):
        return {
            "primaryDataset": {
                "path": "primary.json",
                "holdoutStartIndex": 384,
            },
            "trainingOnlySupplements": [
                {"path": "old.json"},
                {"path": "coverage.json"},
            ],
            "trainer": {
                "supplementalSamplingWeight": 1.0,
                "minimumPrimaryBatchFraction": 0.5,
                "outputDirectory": "output",
                "architecture": "xwide-gelu",
                "featureSchema": "rank-suit-invariant-combo-query-v3",
                "featureWorkers": 8,
                "featureCacheDirectory": "cache",
                "valueNormalization": "pot",
                "variantSet": "range-only",
                "steps": 10000,
                "batchSize": 24,
                "learningRate": 0.0003,
                "learningRateFinal": None,
                "adamBiasCorrection": False,
                "trainingSeeds": [14721, 14722],
                "splitSeed": 10901,
                "validationFraction": 0.25,
                "tuningFraction": 0.15,
                "evaluationInterval": 50,
                "earlyStoppingPatience": 25,
                "suitAugmentationsPerState": 1,
                "huberDelta": 0.05,
                "rawBbAuxiliaryWeight": 0.25,
            },
            "valueReleaseGates": {
                "maximumPerSeedHoldoutRmseBb": 0.25,
                "minimumHoldoutCrossSeedPredictionCorrelation": 0.95,
                "minimumTuningCrossSeedPredictionCorrelation": 0.95,
            },
        }

    def test_command_preserves_every_frozen_training_control(self) -> None:
        command = module.trainer_command(self.config())
        rendered = " ".join(command)
        self.assertIn("--dataset primary.json", rendered)
        self.assertEqual(command.count("--supplemental-dataset"), 2)
        self.assertIn("--steps 10000", rendered)
        self.assertIn("--seeds 14721,14722", rendered)
        self.assertIn("--holdout-start-index 384", rendered)
        self.assertIn("--minimum-tuning-cross-seed-correlation 0.95", rendered)
        self.assertNotIn("--learning-rate-final", command)
        self.assertNotIn("--adam-bias-correction", command)

    def test_optional_optimizer_controls_are_explicit(self) -> None:
        config = self.config()
        config["trainer"]["learningRateFinal"] = 0.00001
        config["trainer"]["adamBiasCorrection"] = True
        command = module.trainer_command(config)
        self.assertIn("--learning-rate-final", command)
        self.assertIn("--adam-bias-correction", command)

    def test_target_dataset_validation_fails_closed(self) -> None:
        valid = {
            "schema": module.TARGET_SCHEMA,
            "validation": {"status": "accepted"},
            "targets": [{}, {}],
        }
        module.validate_target_dataset(valid, 2, "fixture")
        rejected = dict(valid, validation={"status": "rejected"})
        with self.assertRaisesRegex(ValueError, "rejected"):
            module.validate_target_dataset(rejected, 2, "fixture")
        with self.assertRaisesRegex(ValueError, "count"):
            module.validate_target_dataset(valid, 3, "fixture")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
                "learningRateSchedule": "constant",
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
                "sourceValidationStatus": "accepted",
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

    def test_pooled_architecture_requires_v5_runtime_schema(self) -> None:
        self.assertEqual(
            module.expected_network_schema("xwide-gelu-pooled"),
            module.POOLED_NETWORK_SCHEMA,
        )
        self.assertEqual(
            module.expected_network_schema("xwide-gelu"),
            module.SHARED_NETWORK_SCHEMA,
        )
        self.assertEqual(
            module.expected_range_aggregation("xwide-gelu-pooled"),
            "joint-reach-weighted-own-and-opponent-query-pooling",
        )
        self.assertIsNone(module.expected_range_aggregation("xwide-gelu"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            module.expected_network_schema("typo")

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
        standalone_reason = "too small to stand alone"
        standalone = dict(
            valid,
            validation={"status": "rejected", "reasons": [standalone_reason]},
        )
        module.validate_target_dataset(
            standalone, 2, "fixture", [standalone_reason]
        )
        with self.assertRaisesRegex(ValueError, "rejected"):
            module.validate_target_dataset(
                standalone, 2, "fixture", ["different reason"]
            )
        with self.assertRaisesRegex(ValueError, "count"):
            module.validate_target_dataset(valid, 3, "fixture")

    def test_missing_json_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                module.load_json(missing)

    def test_replicated_tuning_selection_verifies_every_report_and_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            controls = module.frozen_training_controls(config["trainer"])
            selected = []
            all_seeds = []
            for pair, seeds in enumerate(((11, 12), (21, 22))):
                pair_directory = root / f"pair-{pair}"
                pair_directory.mkdir()
                variants = []
                weights = []
                for seed in seeds:
                    weight_path = pair_directory / f"seed-{seed}.json"
                    weight_path.write_text(json.dumps({"seed": seed}))
                    variants.append(
                        {"seed": seed, "weights": weight_path.name}
                    )
                    weights.append(
                        {
                            "seed": seed,
                            "path": str(weight_path),
                            "sha256": module.sha256_file(weight_path),
                        }
                    )
                report_path = pair_directory / "report.json"
                report_path.write_text(
                    json.dumps({"variants": {"range": variants}})
                )
                selected.append(
                    {
                        "report": str(report_path),
                        "reportSha256": module.sha256_file(report_path),
                        "configuration": controls,
                        "trainingSeeds": list(seeds),
                        "weights": weights,
                    }
                )
                all_seeds.extend(seeds)
            configuration_sha = module.canonical_sha256(controls)
            selector_path = root / "selection.json"
            selector_path.write_text(
                json.dumps(
                    {
                        "schema": module.SELECTOR_SCHEMA_V2,
                        "status": "frozen-for-fresh-holdout",
                        "holdoutMetricsConsulted": False,
                        "selectedConfiguration": controls,
                        "selectedConfigurationSha256": configuration_sha,
                        "selectedEvidence": selected,
                    }
                )
            )
            config["selectionEvidence"] = {
                "selectorArtifact": str(selector_path),
                "selectorArtifactSha256": module.sha256_file(selector_path),
                "selectedConfigurationSha256": configuration_sha,
                "selectedTrainingSeeds": all_seeds,
            }
            config.update(
                {
                    "schema": module.SCHEMA,
                    "status": "frozen-for-fresh-holdout",
                    "activationAllowed": False,
                }
            )
            config_path = root / "frozen.json"
            config_path.write_text(json.dumps(config))

            with mock.patch.object(module, "SOLVER_ROOT", root):
                self.assertEqual(module.verify_selection(config), set(all_seeds))
                verified, verified_seeds = module.verify_frozen_selection(config_path)
                self.assertFalse(verified["activationAllowed"])
                self.assertEqual(verified_seeds, set(all_seeds))
                config["trainer"]["trainingSeeds"] = [11, 14722]
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "independent"):
                    module.verify_frozen_selection(config_path)
                config["trainer"]["trainingSeeds"] = [14721, 14722]
                selected[1]["weights"][1]["sha256"] = "0" * 64
                selector_path.write_text(
                    json.dumps(
                        {
                            "schema": module.SELECTOR_SCHEMA_V2,
                            "status": "frozen-for-fresh-holdout",
                            "holdoutMetricsConsulted": False,
                            "selectedConfiguration": controls,
                            "selectedConfigurationSha256": configuration_sha,
                            "selectedEvidence": selected,
                        }
                    )
                )
                config["selectionEvidence"]["selectorArtifactSha256"] = (
                    module.sha256_file(selector_path)
                )
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    module.verify_selection(config)

    def test_output_report_requires_both_seed_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            (root / "primary.json").write_text("{}")
            (root / "old.json").write_text("{}")
            config = self.config()
            config["primaryDataset"]["expectedStateCount"] = 4
            config["primaryDataset"]["holdoutStartIndex"] = 2
            config["trainingOnlySupplements"] = [
                {
                    "path": "old.json",
                    "sha256": module.sha256_file(root / "old.json"),
                }
            ]
            for seed in (14721, 14722):
                (root / "output" / f"seed-{seed}.json").write_text(
                    json.dumps(
                        {
                            "schema": module.SHARED_NETWORK_SCHEMA,
                            "architecture": "xwide-gelu",
                            "featureSchema": "rank-suit-invariant-combo-query-v3",
                            "valueNormalization": "pot",
                            "seed": seed,
                            "sourceDatasetSha256": "combined",
                            "sourceValidationStatus": "accepted",
                        }
                    )
                )
            report = {
                "schema": module.REPORT_SCHEMA,
                "networkSchema": module.SHARED_NETWORK_SCHEMA,
                "architecture": "xwide-gelu",
                "featureSchema": "rank-suit-invariant-combo-query-v3",
                "valueNormalization": "pot",
                "variantSet": "range-only",
                "datasetSha256": "combined",
                "componentDatasetSha256": [
                    module.sha256_file(root / "primary.json"),
                    module.sha256_file(root / "old.json"),
                ],
                "sourceValidation": {"status": "accepted"},
                "primaryStates": 4,
                "holdoutStartIndex": 2,
                "validationStates": [2, 3],
                "splitSeed": 10901,
                "variants": {
                    "range": [
                        {
                            "seed": seed,
                            "weights": f"seed-{seed}.json",
                            "metrics": {"weightedRmseBb": 0.2},
                        }
                        for seed in (14721, 14722)
                    ]
                },
                "crossSeedPredictionCorrelation": {"range": 0.99},
                "tuningCrossSeedPredictionCorrelation": {"range": 0.98},
                "validation": {"status": "accepted"},
            }
            report_path = root / "output" / "turn-value-paired-report.json"
            report_path.write_text(json.dumps(report))
            with mock.patch.object(module, "SOLVER_ROOT", root):
                result = module.verify_output_report(config)
                self.assertEqual(result["status"], "accepted")
                report["variants"]["range"][1]["metrics"][
                    "weightedRmseBb"
                ] = 0.251
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "RMSE"):
                    module.verify_output_report(config)
                report["variants"]["range"][1]["metrics"][
                    "weightedRmseBb"
                ] = 0.2
                report["networkSchema"] = module.POOLED_NETWORK_SCHEMA
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "model controls"):
                    module.verify_output_report(config)


if __name__ == "__main__":
    unittest.main()

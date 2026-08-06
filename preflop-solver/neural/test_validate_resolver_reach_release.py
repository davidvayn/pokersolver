import json
import tempfile
import unittest
from pathlib import Path

import validate_resolver_reach_release as module


class ValidateResolverReachReleaseTests(unittest.TestCase):
    def test_sufficient_statistics_aggregate_without_averaging_rmse(self) -> None:
        reports = [
            {
                "weightMass": 1.0,
                "weightedSquaredErrorBb2Sum": 1.0,
                "weightedAbsoluteErrorBbSum": 1.0,
            },
            {
                "weightMass": 3.0,
                "weightedSquaredErrorBb2Sum": 27.0,
                "weightedAbsoluteErrorBbSum": 9.0,
            },
        ]
        result = module.aggregate_error(reports, resolver=False)
        self.assertAlmostEqual(result["weightedRmseBb"], 7.0**0.5)
        self.assertAlmostEqual(result["weightedMaeBb"], 2.5)

    def test_resolver_sufficient_statistics_use_reach_mass(self) -> None:
        reports = [
            {
                "resolverReachEvaluation": {
                    "reachWeightMass": 2.0,
                    "reachWeightedSquaredErrorBb2Sum": 0.5,
                    "reachWeightedAbsoluteErrorBbSum": 1.0,
                }
            },
            {
                "resolverReachEvaluation": {
                    "reachWeightMass": 2.0,
                    "reachWeightedSquaredErrorBb2Sum": 1.5,
                    "reachWeightedAbsoluteErrorBbSum": 1.0,
                }
            },
        ]
        result = module.aggregate_error(reports, resolver=True)
        self.assertAlmostEqual(result["weightedRmseBb"], 0.5**0.5)
        self.assertAlmostEqual(result["weightedMaeBb"], 0.5)

    def test_fresh_holdout_requires_unique_accepted_exact_source_states(self) -> None:
        payload = {
            "schema": "hu-turn-public-belief-cfv-dataset-v2",
            "seed": 15401,
            "source_policy_sha256": "a" * 64,
            "state_distribution": "frozen_v26_self_play_exact_reach_factor_public_beliefs",
            "targets": [
                {"input_sha256": "b" * 64},
                {"input_sha256": "c" * 64},
            ],
            "validation": {"status": "accepted", "reasons": []},
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "holdout.json"
            path.write_text(json.dumps(payload))
            module.validate_fresh_holdout_dataset(path, 15401, 2, "a" * 64)
            payload["targets"][1]["input_sha256"] = "b" * 64
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "repeats"):
                module.validate_fresh_holdout_dataset(path, 15401, 2, "a" * 64)

    def test_dataset_fingerprints_require_unique_exact_state_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "dataset.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": [
                            {"input_sha256": "a" * 64},
                            {"input_sha256": "b" * 64},
                        ]
                    }
                )
            )
            self.assertEqual(
                module.dataset_fingerprints(path), {"a" * 64, "b" * 64}
            )
            path.write_text(
                json.dumps(
                    {
                        "targets": [
                            {"input_sha256": "a" * 64},
                            {"input_sha256": "a" * 64},
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "repeated"):
                module.dataset_fingerprints(path)


if __name__ == "__main__":
    unittest.main()

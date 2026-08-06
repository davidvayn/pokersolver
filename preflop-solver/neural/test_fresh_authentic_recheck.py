import tempfile
import unittest
from pathlib import Path

import run_fresh_authentic_recheck as runner
import run_resolver_reach_crossfit as crossfit
import validate_fresh_authentic_recheck as validator


class FreshAuthenticRecheckTests(unittest.TestCase):
    def protocol(self):
        return {
            "sourcePolicy": {"path": "policy.json", "sha256": "f" * 64},
            "models": [
                {"seed": 15301, "path": "model-15301.json", "sha256": "a" * 64},
                {"seed": 15302, "path": "model-15302.json", "sha256": "b" * 64},
            ],
            "generator": {
                "command": "turn-pbs-self-play-targets",
                "effectiveStackBb": 20.0,
                "statesPerShard": 64,
                "rangeParticles": 4096,
                "riverIterations": 200,
                "riverAveragingDelay": 20,
                "threads": 10,
                "beliefReplicates": 2,
                "explorationProbability": 0.0,
                "minimumPotBb": 0.0,
            },
            "shards": [
                {
                    "seed": 15501,
                    "checkpointDirectory": "checkpoints-15501",
                    "output": "holdout-15501.json",
                },
                {
                    "seed": 15502,
                    "checkpointDirectory": "checkpoints-15502",
                    "output": "holdout-15502.json",
                },
            ],
            "diagnosticOutputDirectory": "diagnostics",
            "gates": {
                "maximumPerSeedRmseBb": 0.25,
                "minimumCrossSeedPredictionCorrelation": 0.95,
            },
        }

    def test_plan_generates_two_holdouts_and_every_model_dataset_pair(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            protocol_path = Path(raw_directory) / "protocol.json"
            protocol_path.write_text("protocol")
            plan = runner.build_plan(self.protocol(), protocol_path)
            self.assertFalse(plan["activationAllowed"])
            self.assertEqual(len(plan["holdoutJobs"]), 2)
            self.assertEqual(len(plan["diagnosticJobs"]), 4)
            first = plan["holdoutJobs"][0]["command"]
            self.assertEqual(crossfit.option_values(first, "--seed"), ["15501"])
            self.assertEqual(crossfit.option_values(first, "--states"), ["64"])
            pairs = {
                (job["datasetSeed"], job["modelSeed"])
                for job in plan["diagnosticJobs"]
            }
            self.assertEqual(
                pairs,
                {
                    (15501, 15301),
                    (15501, 15302),
                    (15502, 15301),
                    (15502, 15302),
                },
            )

    def test_summary_passes_only_the_frozen_rmse_and_correlation_gates(self):
        protocol = self.protocol()
        metrics = {
            15301: {"weightedRmseBb": 0.21},
            15302: {"weightedRmseBb": 0.22},
        }
        artifacts = [{} for _ in range(6)]
        with tempfile.TemporaryDirectory() as raw_directory:
            protocol_path = Path(raw_directory) / "protocol.json"
            protocol_path.write_text("protocol")
            accepted = validator.summarize(
                protocol, protocol_path, metrics, 0.97, 128, 1000, artifacts
            )
            self.assertTrue(all(accepted["gates"].values()))
            self.assertFalse(accepted["activationAllowed"])
            rejected = validator.summarize(
                protocol,
                protocol_path,
                {**metrics, 15302: {"weightedRmseBb": 0.251}},
                0.97,
                128,
                1000,
                artifacts,
            )
            self.assertEqual(rejected["status"], "rejected")
            self.assertFalse(rejected["gates"]["freshAuthenticPerSeedRmse"])


if __name__ == "__main__":
    unittest.main()

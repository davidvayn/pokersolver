import unittest

import run_resolver_reach_release as module


class RunResolverReachReleaseTests(unittest.TestCase):
    def payload(self):
        return {
            "primaryDataset": {"path": "primary.json", "sha256": "a" * 64},
            "supplementalDatasets": [
                {"path": f"supplement-{index}.json", "sha256": str(index) * 64}
                for index in range(7)
            ],
            "supplementalDatasetSamplingWeights": [1.0] * 5 + [1.5, 1.5],
            "minimumPrimaryBatchFraction": 0.75,
            "trainer": {
                "architecture": "xwide-gelu-pooled",
                "featureSchema": "rank-suit-invariant-combo-query-v3",
                "featureWorkers": 8,
                "featureCacheDirectory": "cache",
                "valueNormalization": "pot",
                "variantSet": "range-only",
                "steps": 5000,
                "batchSize": 24,
                "evaluationInterval": 50,
                "learningRate": 0.0003,
                "adamBiasCorrection": False,
                "earlyStoppingPatience": 25,
                "huberDelta": 0.05,
                "rawBbAuxiliaryWeight": 0.25,
                "suitAugmentationsPerState": 1,
                "splitSeed": 10901,
                "validationFraction": 0.25,
                "tuningFraction": 0.15,
                "trainingSeeds": [15301, 15302],
                "outputDirectory": "runs/release",
            },
            "reservedResolverEvaluation": [
                {
                    "seed": seed,
                    "sourceTrainingSeed": source,
                    "boards": ["2c,7d,Jh"],
                    "expectedStateCount": 3,
                    "checkpointDirectory": f"resolver-{seed}-checkpoints",
                    "output": f"resolver-{seed}.json",
                }
                for seed, source in ((15103, 14921), (15104, 14922))
            ],
            "freshAuthenticHoldout": {
                "sourcePolicy": {"path": "policy.json", "sha256": "f" * 64},
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
                        "seed": seed,
                        "checkpointDirectory": f"authentic-{seed}-checkpoints",
                        "output": f"authentic-{seed}.json",
                    }
                    for seed in (15401, 15402)
                ],
            },
            "releaseGates": {
                "minimumCrossSeedPredictionCorrelation": 0.95,
                "maximumPerSeedAuthenticFreshHoldoutRmseBb": 0.25,
                "maximumRustPythonParityErrorBb": 0.0001,
            },
        }

    def corpus(self):
        return {
            "generator": {
                "effectiveStackBb": 20.0,
                "statesPerBoard": 3,
                "rootPotBb": 4.0,
                "rootActor": 1,
                "resolverIterations": 20,
                "resolverAveragingDelay": 2,
                "riverIterations": 200,
                "riverAveragingDelay": 20,
                "threads": 10,
            },
            "sourceValueNetworks": [
                {"trainingSeed": 14921, "path": "value-a.json"},
                {"trainingSeed": 14922, "path": "value-b.json"},
            ],
        }

    def test_plan_covers_training_both_fresh_corpora_diagnostics_and_parity(self):
        plan = module.build_plan(self.payload(), self.corpus())
        self.assertFalse(plan["activationAllowed"])
        self.assertEqual(len(plan["resolverEvaluationJobs"]), 2)
        self.assertEqual(len(plan["freshAuthenticHoldoutJobs"]), 2)
        self.assertEqual(len(plan["baselineResolverDiagnosticJobs"]), 4)
        self.assertEqual(len(plan["diagnosticJobs"]), 8)
        self.assertEqual(len(plan["parityJobs"]), 2)
        command = plan["trainingJob"]["command"]
        self.assertEqual(
            module.crossfit.option_values(command, "--supplemental-dataset-weight"),
            ["1.0"] * 5 + ["1.5", "1.5"],
        )
        self.assertEqual(
            module.crossfit.option_values(command, "--seeds"), ["15301,15302"]
        )
        self.assertNotIn("--holdout-start-index", command)

    def test_resolver_job_pins_source_model_and_every_solver_control(self):
        command = module.resolver_evaluation_command(
            self.corpus(), self.payload()["reservedResolverEvaluation"][0]
        )
        self.assertEqual(
            module.crossfit.option_values(command, "--value-network"),
            ["value-a.json"],
        )
        self.assertEqual(
            module.crossfit.option_values(command, "--resolver-iterations"), ["20"]
        )
        self.assertEqual(
            module.crossfit.option_values(command, "--states-per-board"), ["3"]
        )

    def test_authentic_job_uses_precommitted_particles_and_seed(self):
        payload = self.payload()
        command = module.authentic_holdout_command(
            payload["freshAuthenticHoldout"],
            payload["freshAuthenticHoldout"]["shards"][0],
        )
        self.assertEqual(
            module.crossfit.option_values(command, "--range-particles"), ["4096"]
        )
        self.assertEqual(module.crossfit.option_values(command, "--seed"), ["15401"])
        self.assertEqual(
            module.crossfit.option_values(command, "--networks"), ["policy.json"]
        )


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path

import run_resolver_reach_crossfit as module


class ResolverReachCrossfitRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = Path(__file__).parent
        cls.payload = json.loads(
            (directory / "20bb-v49-resolver-reach-experiments.json").read_text()
        )
        cls.corpus = json.loads(
            (directory / "20bb-v49-resolver-reach-corpus.json").read_text()
        )
        cls.plan = module.build_plan(cls.payload, cls.corpus)

    def test_plan_contains_every_frozen_pair_and_diagnostic(self) -> None:
        self.assertFalse(self.plan["activationAllowed"])
        self.assertEqual(len(self.plan["baselineJobs"]), 4)
        self.assertEqual(len(self.plan["candidateJobs"]), 8)
        self.assertEqual(
            sum(len(job["diagnosticJobs"]) for job in self.plan["candidateJobs"]),
            16,
        )
        self.assertEqual(len(self.plan["selectorSpec"]["candidates"]), 4)

    def test_training_commands_use_six_weighted_supplements_without_old_holdout(self) -> None:
        seeds = set()
        for job in self.plan["candidateJobs"]:
            command = job["trainingCommand"]
            self.assertEqual(command.count("--supplemental-dataset"), 6)
            self.assertEqual(command.count("--supplemental-dataset-weight"), 6)
            self.assertNotIn("--holdout-start-index", command)
            raw_seeds = command[command.index("--seeds") + 1]
            pair = {int(value) for value in raw_seeds.split(",")}
            self.assertEqual(len(pair), 2)
            self.assertFalse(seeds & pair)
            seeds |= pair
        self.assertEqual(len(seeds), 16)

    def test_every_diagnostic_enumerates_all_thirty_six_states(self) -> None:
        expected = ",".join(str(index) for index in range(36))
        jobs = self.plan["baselineJobs"] + [
            diagnostic
            for candidate in self.plan["candidateJobs"]
            for diagnostic in candidate["diagnosticJobs"]
        ]
        for job in jobs:
            command = job["command"]
            self.assertEqual(command[command.index("--state-indices") + 1], expected)


if __name__ == "__main__":
    unittest.main()

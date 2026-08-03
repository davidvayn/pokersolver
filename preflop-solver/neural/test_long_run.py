import json
import math
import tempfile
import unittest
from pathlib import Path

from long_run import load_plan, trainer_command, training_seconds, validation_command


class LongRunTests(unittest.TestCase):
    def setUp(self):
        self.plan_path = Path(__file__).resolve().with_name("long-run-20bb-v1.json")
        self.plan = load_plan(self.plan_path)

    def test_plan_pins_two_fresh_seeds_and_the_authorized_extension(self):
        self.assertEqual(self.plan["seeds"], [5101, 5102])
        self.assertEqual(
            sum(stage["minutes"] for stage in self.plan["stages"]), 17.5 * 60
        )
        self.assertEqual(self.plan["targetTrainingHoursPerSeed"], 8)
        self.assertEqual(self.plan["authorizedExtensionHoursPerSeed"], 9.5)
        self.assertEqual(self.plan["sharedTraining"]["valueRolloutsPerAction"], 4)
        self.assertEqual(self.plan["monitorIntervalSeconds"], 600)

    def test_certificate_budget_can_clear_the_bound_at_zero_sample_loss(self):
        validation = self.plan["postRunValidation"]
        family_confidence = validation["exploitabilityCertificateConfidence"]
        per_seed_alpha = (1.0 - family_confidence) / len(self.plan["seeds"])
        margin = self.plan["depthBb"] * math.sqrt(
            math.log(1.0 / per_seed_alpha)
            / (2.0 * validation["exploitabilityCertificateDeals"])
        )
        self.assertLess(margin, 0.10)

    def test_commands_pin_every_material_training_control(self):
        command = trainer_command(self.plan, self.plan["stages"][0], 5101, 90)
        joined = " ".join(command)
        self.assertIn("20bb-long-v1-narrow-seed5101", joined)
        self.assertIn("--max-minutes 1.5", joined)
        self.assertIn("--value-rollouts-per-action 4", joined)
        self.assertIn("--learning-rate-decay-start-round 50", joined)
        self.assertIn("--artifact-every 50", joined)

        wide_command = " ".join(
            trainer_command(self.plan, self.plan["stages"][1], 5101, 90)
        )
        self.assertIn("20bb-long-v1-wide-seed5101", wide_command)

    def test_validation_command_routes_stages_and_pins_release_budgets(self):
        joined = " ".join(validation_command(self.plan))
        self.assertIn("20bb-long-v1-narrow-seed5101", joined)
        self.assertIn("20bb-long-v1-narrow-seed5102", joined)
        self.assertIn("--postflop-run-a", joined)
        self.assertIn("20bb-long-v1-wide-seed5101", joined)
        self.assertIn("20bb-long-v1-wide-seed5102", joined)
        self.assertIn("--round 250", joined)
        self.assertIn("--postflop-latest", joined)
        self.assertIn("--traversals 10000", joined)
        self.assertIn("--action-value-rollouts-per-action 64", joined)
        self.assertIn("--exploitability-certificate-deals 125000", joined)

    def test_resume_budget_uses_all_atomic_round_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metrics.jsonl").write_text(
                "\n".join(
                    json.dumps({"elapsed_seconds": value}) for value in (2.5, 3, 4.25)
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(training_seconds(path), 9.75)


if __name__ == "__main__":
    unittest.main()

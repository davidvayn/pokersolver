import json
import tempfile
import unittest
from pathlib import Path

import run_matched_continual_resolver as module


class RunMatchedContinualResolverTests(unittest.TestCase):
    def controls(self):
        return {
            "runOnlyAfterValueReleaseGatesPass": True,
            "rootSource": "all reservedEvaluationShards boards in the pinned resolver corpus",
            "crossEvaluateBothReleaseSeeds": True,
            "effectiveStackBb": 20.0,
            "rootPotBb": 4.0,
            "rootActor": 1,
            "iterations": 100,
            "averagingDelay": 10,
            "threads": 10,
            "maximumExploitabilityBbPerHand": 0.05,
        }

    def write_models(self, root: Path) -> None:
        directory = root / "runs/release"
        directory.mkdir(parents=True)
        for seed, marker in ((15301, "a"), (15302, "b")):
            (directory / f"turn-value-range-seed{seed}.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "usesExactRanges": True,
                        "sourceValidationStatus": "accepted",
                        "sourceDatasetSha256": marker * 64,
                        "sourcePolicySha256": "c" * 64,
                    }
                )
            )

    def release(self):
        return {
            "trainer": {
                "trainingSeeds": [15301, 15302],
                "outputDirectory": "runs/release",
            },
            "reservedResolverEvaluation": [
                {"boards": ["2d,8h,Ks"]},
                {"boards": ["Qc,Qd,Qh"]},
            ],
            "matchedContinualResolver": self.controls(),
        }

    def test_plan_cross_evaluates_both_models_on_every_reserved_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            self.write_models(root)
            release_path = root / "release.json"
            release_path.write_text("release")
            validation_path = root / "value.json"
            validation_path.write_text("value")
            plan = module.build_plan(
                self.release(),
                {"status": "accepted-awaiting-strategy-and-full-game-gates"},
                release_path,
                validation_path,
                root,
            )
            self.assertFalse(plan["activationAllowed"])
            self.assertEqual(plan["rootCount"], 2)
            self.assertEqual(len(plan["jobs"]), 4)
            directions = {
                (job["strategyModel"]["seed"], job["evaluationModel"]["seed"])
                for job in plan["jobs"]
                if job["board"] == "2d,8h,Ks"
            }
            self.assertEqual(directions, {(15301, 15302), (15302, 15301)})
            command = plan["jobs"][0]["command"]
            self.assertEqual(
                module.crossfit.option_values(command, "--iterations"), ["100"]
            )
            self.assertEqual(
                module.crossfit.option_values(command, "--evaluation-value-network"),
                ["runs/release/turn-value-range-seed15302.json"],
            )

    def test_value_gate_must_be_accepted_and_bound_to_exact_release(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            release_path = root / "release.json"
            release_path.write_text("frozen release")
            validation_path = root / "value.json"
            report = {
                "schema": module.VALUE_VALIDATION_SCHEMA,
                "status": "accepted-awaiting-strategy-and-full-game-gates",
                "activationAllowed": False,
                "releaseFreeze": {
                    "path": str(release_path),
                    "sha256": module.release_freeze.sha256_file(release_path),
                },
                "gates": {"first": True, "second": True},
            }
            validation_path.write_text(json.dumps(report))
            self.assertEqual(
                module.validate_value_report(
                    validation_path, release_path, root
                )["status"],
                "accepted-awaiting-strategy-and-full-game-gates",
            )
            report["gates"]["second"] = False
            validation_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "remains sealed"):
                module.validate_value_report(validation_path, release_path, root)


if __name__ == "__main__":
    unittest.main()

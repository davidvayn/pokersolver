import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import select_resolver_reach_value_config as module


class ResolverReachValueSelectionTests(unittest.TestCase):
    @staticmethod
    def relative(root: Path, path: Path) -> str:
        return str(path.relative_to(root))

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def dataset(self, root: Path, name: str) -> tuple[Path, str]:
        path = root / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": module.DATASET_SCHEMA,
                    "validation": {"status": "accepted", "reasons": []},
                    "targets": [],
                    "name": name,
                }
            )
        )
        return path, self.sha(path)

    def diagnostic(
        self,
        root: Path,
        name: str,
        dataset_sha: str,
        seed: int,
        model_sha: str,
        rmse: float,
    ) -> Path:
        path = root / f"{name}-diagnostic-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": module.DIAGNOSTIC_SCHEMA,
                    "sourceDatasetSha256": dataset_sha,
                    "modelSeed": seed,
                    "modelSha256": model_sha,
                    "resolverReachEvaluation": {
                        "reachWeightedRmseBb": rmse,
                        "sampledLeafReachMass": 0.1,
                    },
                }
            )
        )
        return path

    def report(
        self,
        root: Path,
        name: str,
        architecture: str,
        seeds: tuple[int, int],
        tuning: tuple[float, float],
        training_fold_sha: str,
    ) -> tuple[Path, list[tuple[int, str]]]:
        directory = root / name
        directory.mkdir()
        variants = []
        models = []
        for seed, rmse in zip(seeds, tuning, strict=True):
            model = directory / f"model-{seed}.json"
            model.write_text(json.dumps({"seed": seed, "candidate": name}))
            digest = self.sha(model)
            models.append((seed, digest))
            variants.append(
                {
                    "seed": seed,
                    "weights": model.name,
                    "metrics": {
                        "bestTuningRmseBb": rmse,
                        "finalTuningMetrics": {"weightedRmseBb": rmse},
                    },
                }
            )
        report = directory / "report.json"
        report.write_text(
            json.dumps(
                {
                    "schema": module.value_selection.REPORT_SCHEMA,
                    "sourceValidation": {"status": "accepted"},
                    "datasetSha256": "a" * 64,
                    "componentDatasetSha256": [training_fold_sha],
                    "sourcePolicySha256": "b" * 64,
                    "splitSeed": 7,
                    "trainStates": [0],
                    "tuningStates": [1],
                    "validationStates": [2],
                    "architecture": architecture,
                    "variantSet": "range-only",
                    "featureSchema": "features",
                    "supplementalDatasetSamplingWeights": [1.0],
                    "variants": {"range": variants},
                    "tuningCrossSeedPredictionCorrelation": {"range": 0.99},
                    "twoSeedOutputEnsembleMetrics": {
                        "range": {"tuning": {"weightedRmseBb": sum(tuning) / 2}}
                    },
                }
            )
        )
        return report, models

    def spec(self, root: Path) -> tuple[Path, dict]:
        first_dataset, first_sha = self.dataset(root, "fold-first")
        second_dataset, second_sha = self.dataset(root, "fold-second")
        baseline_models = []
        baseline_folds = []
        baseline_hashes = []
        for seed in (91, 92):
            path = root / f"baseline-{seed}.json"
            path.write_text(json.dumps({"seed": seed}))
            digest = self.sha(path)
            baseline_hashes.append((seed, digest))
            baseline_models.append(
                {"seed": seed, "path": self.relative(root, path), "sha256": digest}
            )
        for fold_name, dataset, dataset_sha in (
            ("first", first_dataset, first_sha),
            ("second", second_dataset, second_sha),
        ):
            diagnostics = [
                self.diagnostic(
                    root,
                    f"baseline-{fold_name}",
                    dataset_sha,
                    seed,
                    digest,
                    0.5 - offset * 0.01,
                )
                for offset, (seed, digest) in enumerate(baseline_hashes)
            ]
            baseline_folds.append(
                {
                    "name": fold_name,
                    "evaluationDataset": self.relative(root, dataset),
                    "diagnostics": [self.relative(root, path) for path in diagnostics],
                }
            )

        candidates = []
        for candidate_name, architecture, resolver_rmse in (
            ("better", "pooled-a", 0.30),
            ("worse", "pooled-b", 0.36),
        ):
            folds = []
            for offset, (fold_name, evaluation, evaluation_sha, training_sha) in enumerate(
                (
                    ("first", first_dataset, first_sha, second_sha),
                    ("second", second_dataset, second_sha, first_sha),
                )
            ):
                seeds = (101 + offset * 2, 102 + offset * 2)
                report, models = self.report(
                    root,
                    f"{candidate_name}-{fold_name}",
                    architecture,
                    seeds,
                    (0.20, 0.21),
                    training_sha,
                )
                diagnostics = [
                    self.diagnostic(
                        root,
                        f"{candidate_name}-{fold_name}",
                        evaluation_sha,
                        seed,
                        digest,
                        resolver_rmse + model_offset * 0.01,
                    )
                    for model_offset, (seed, digest) in enumerate(models)
                ]
                folds.append(
                    {
                        "name": fold_name,
                        "trainingReport": self.relative(root, report),
                        "evaluationDataset": self.relative(root, evaluation),
                        "diagnostics": [
                            self.relative(root, path) for path in diagnostics
                        ],
                    }
                )
            candidates.append({"name": candidate_name, "folds": folds})
        payload = {
            "schema": module.SPEC_SCHEMA,
            "gates": {
                "minimumCrossSeedPredictionCorrelation": 0.95,
                "maximumAuthenticTuningRmseBb": 0.25,
                "minimumMaximumResolverReachRmseImprovementFraction": 0.2,
            },
            "baseline": {"models": baseline_models, "folds": baseline_folds},
            "candidates": candidates,
        }
        path = root / "spec.json"
        path.write_text(json.dumps(payload))
        return path, payload

    def test_selects_best_passing_crossfit_without_release_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            path, _ = self.spec(root)
            result = module.select(path, root)
            self.assertEqual(result["status"], "frozen-for-fresh-evaluation")
            self.assertFalse(result["activationAllowed"])
            self.assertFalse(result["releaseHoldoutMetricsConsulted"])
            self.assertEqual(result["selectedCandidate"]["name"], "better")
            self.assertEqual(
                result["gates"]["maximumCrossFitResolverReachWeightedRmseBb"],
                0.4,
            )

    def test_rejects_diagnostic_whose_model_bytes_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            path, payload = self.spec(root)
            diagnostic = root / payload["candidates"][0]["folds"][0]["diagnostics"][0]
            report = json.loads(diagnostic.read_text())
            report["modelSha256"] = "0" * 64
            diagnostic.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "model hash mismatch"):
                module.select(path, root)

    def test_rejects_training_evaluation_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            path, payload = self.spec(root)
            fold = payload["candidates"][0]["folds"][0]
            report_path = root / fold["trainingReport"]
            report = json.loads(report_path.read_text())
            report["componentDatasetSha256"].append(
                module.sha256_file(root / fold["evaluationDataset"])
            )
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "leaked"):
                module.select(path, root)

    def test_no_passing_candidate_remains_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            path, payload = self.spec(root)
            for candidate in payload["candidates"]:
                for fold in candidate["folds"]:
                    for raw_diagnostic in fold["diagnostics"]:
                        diagnostic = root / raw_diagnostic
                        report = json.loads(diagnostic.read_text())
                        report["resolverReachEvaluation"]["reachWeightedRmseBb"] = 0.41
                        diagnostic.write_text(json.dumps(report))

            result = module.select(path, root)

            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(result["selectedCandidate"])
            self.assertFalse(result["activationAllowed"])


if __name__ == "__main__":
    unittest.main()

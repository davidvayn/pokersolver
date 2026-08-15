import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

import distill_range_policy as distill
import fine_tune_range_policy_causal as module
import train_public_value_network as value_features


class CausalRangePolicyTests(unittest.TestCase):
    def test_per_seed_learning_rates_override_the_shared_default(self):
        self.assertEqual(
            module.resolve_learning_rates(1e-6, None, None),
            [1e-6, 1e-6],
        )
        self.assertEqual(
            module.resolve_learning_rates(1e-6, 1.5e-6, 3e-6),
            [1.5e-6, 3e-6],
        )
        for invalid in (0.0, -1e-6, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                module.resolve_learning_rates(1e-6, None, invalid)

    def test_cached_exact_dataset_parity_must_pin_every_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.json"
            attribution = directory / "attribution.json"
            source.write_text("source")
            attribution.write_text("attribution")
            dataset = SimpleNamespace(
                sha256="d" * 64,
                records=[{}],
                metadata={"target_actor": 0},
            )
            exact = {
                "schema": "hu-range-conditioned-causal-policy-rust-evaluation-v1",
                "networkSha256": module.sha256(source),
                "frozenNetworkSha256": module.sha256(source),
                "attributionNetworkSha256": module.sha256(attribution),
                "datasetSha256": dataset.sha256,
                "targetActor": 0,
                "records": 1,
                "maximumStoredSourceProbabilityDifference": 1e-7,
                "maximumProbabilitySumError": 1e-8,
            }
            report = directory / "report.json"
            report.write_text(json.dumps({"exactRustDatasetParity": [exact]}) + "\n")
            self.assertEqual(
                module.reuse_exact_dataset_parity(
                    report, [dataset], [source], [attribution]
                ),
                [exact],
            )
            exact["maximumStoredSourceProbabilityDifference"] = (
                module.MAXIMUM_EXACT_RUST_STORED_SOURCE_PROBABILITY_DIFFERENCE
            )
            report.write_text(json.dumps({"exactRustDatasetParity": [exact]}) + "\n")
            self.assertEqual(
                module.reuse_exact_dataset_parity(
                    report, [dataset], [source], [attribution]
                ),
                [exact],
            )
            exact["maximumStoredSourceProbabilityDifference"] = np.nextafter(
                module.MAXIMUM_EXACT_RUST_STORED_SOURCE_PROBABILITY_DIFFERENCE,
                np.inf,
            )
            report.write_text(json.dumps({"exactRustDatasetParity": [exact]}) + "\n")
            with self.assertRaisesRegex(ValueError, "not pinned"):
                module.reuse_exact_dataset_parity(
                    report, [dataset], [source], [attribution]
                )
            exact["maximumStoredSourceProbabilityDifference"] = 1e-7
            exact["targetActor"] = 1
            report.write_text(json.dumps({"exactRustDatasetParity": [exact]}) + "\n")
            with self.assertRaisesRegex(ValueError, "not pinned"):
                module.reuse_exact_dataset_parity(
                    report, [dataset], [source], [attribution]
                )
            exact["targetActor"] = 0
            exact["datasetSha256"] = "0" * 64
            report.write_text(json.dumps({"exactRustDatasetParity": [exact]}) + "\n")
            with self.assertRaisesRegex(ValueError, "not pinned"):
                module.reuse_exact_dataset_parity(
                    report, [dataset], [source], [attribution]
                )

    def test_rust_evaluator_report_must_pin_every_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / "candidate.json"
            frozen = directory / "frozen.json"
            dataset = directory / "dataset.jsonl.gz"
            candidate.write_text("candidate")
            frozen.write_text("frozen")
            dataset.write_text("dataset")
            report = {
                "schema": "hu-range-conditioned-causal-policy-rust-evaluation-v1",
                "networkSha256": module.sha256(candidate),
                "frozenNetworkSha256": module.sha256(frozen),
                "attributionNetworkSha256": module.sha256(frozen),
                "networkArtifactSha256": module.sha256(candidate),
                "frozenNetworkArtifactSha256": module.sha256(frozen),
                "attributionNetworkArtifactSha256": module.sha256(frozen),
                "datasetSha256": module.sha256(dataset),
                "validation": {"status": "accepted_for_directional_evaluation"},
            }
            completed = SimpleNamespace(
                returncode=0, stdout=json.dumps(report), stderr=""
            )
            with patch.object(module.subprocess, "run", return_value=completed) as run:
                measured = module.rust_evaluate(
                    Path("solver"),
                    candidate,
                    frozen,
                    frozen,
                    dataset,
                    1e-6,
                    0.005,
                    0.0015,
                    None,
                )
            self.assertEqual(measured, report)
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["solver", "range-policy-causal-evaluate"])
            run.assert_called_once_with(
                command, check=False, capture_output=True, text=True
            )
            report["datasetSha256"] = "0" * 64
            with patch.object(
                module.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(report), stderr=""
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "not pinned"):
                    module.rust_evaluate(
                        Path("solver"),
                        candidate,
                        frozen,
                        frozen,
                        dataset,
                        1e-6,
                        0.005,
                        0.0015,
                        None,
                    )
            report["datasetSha256"] = module.sha256(dataset)
            report["targetActor"] = 1
            with patch.object(
                module.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(report), stderr=""
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "not pinned"):
                    module.rust_evaluate(
                        Path("solver"),
                        candidate,
                        frozen,
                        frozen,
                        dataset,
                        1e-6,
                        0.005,
                        0.0015,
                        0,
                    )
            report["targetActor"] = None
            report["attributionNetworkArtifactSha256"] = "0" * 64
            with patch.object(
                module.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(report), stderr=""
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "not pinned"):
                    module.rust_evaluate(
                        Path("solver"),
                        candidate,
                        frozen,
                        frozen,
                        dataset,
                        1e-6,
                        0.005,
                        0.0015,
                        None,
                    )

    def test_rust_evaluator_surfaces_solver_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = [directory / name for name in ("candidate", "frozen", "dataset")]
            for path in paths:
                path.write_text(path.name)
            completed = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Error: focal combo has no reach\n",
            )
            with patch.object(module.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "focal combo has no reach"):
                    module.rust_evaluate(
                        Path("solver"),
                        paths[0],
                        paths[1],
                        paths[1],
                        paths[2],
                        1e-6,
                        0.005,
                        0.0015,
                        None,
                    )

    def test_loader_builds_exact_belief_features_and_metrics_reward_better_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "causal.jsonl.gz"
            records = []
            for index, street in enumerate(
                ("flop", "flop", "turn", "turn", "river", "river")
            ):
                board = [0, 5, 10, 15, 20][: {"flop": 3, "turn": 4, "river": 5}[street]]
                blocked = np.isin(value_features.COMBO_CARDS[:, 0], board) | np.isin(
                    value_features.COMBO_CARDS[:, 1], board
                )
                ranges = np.ones((2, module.COMBO_COUNT), dtype=np.float32)
                ranges[:, blocked] = 0.0
                ranges /= ranges.sum(axis=1, keepdims=True)
                focal = int(np.flatnonzero(~blocked)[index])
                records.append(
                    {
                        "record_type": "range_conditioned_causal_policy_attribution",
                        "weight": float(index + 1),
                        "state": {
                            "board": board,
                            "street": street,
                            "actor": index % 2,
                            "invested_bb": [2.0, 2.0],
                            "street_invested_bb": [0.0, 0.0],
                            "last_full_raise_bb": 1.0,
                            "aggressions": 0,
                            "checks": 0,
                            "raise_reopened": True,
                            "public_history": [f"node-{index}"],
                            "trajectory": [],
                        },
                        "ranges": ranges.tolist(),
                        "focal_combo": focal,
                        "action_labels": ["check", "bet_to_3.000bb"],
                        "action_features": [
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.15, 0.15, 0.5],
                        ],
                        "probabilities": [0.5, 0.5],
                        "action_values_bb": [0.0, 1.0],
                    }
                )
            metadata = {
                "record_type": "metadata",
                "schema": module.CAUSAL_SCHEMA,
                "state_feature_schema": module.RANGE_POLICY_FEATURE_SCHEMA,
                "state_feature_count": module.CONTEXT_SIZE,
                "action_feature_schema": module.ACTION_FEATURE_SCHEMA,
                "action_feature_count": module.ACTION_FEATURE_COUNT,
                "depth_bb": 20.0,
                "seed": 101,
                "records": len(records),
                "source_network_sha256": "a" * 64,
                "source_range_policy_sha256": "b" * 64,
                "uses_exact_ranges": True,
                "focal_combo_attribution": True,
                "postflop_only": True,
                "preflop_policy_frozen": True,
            }
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata) + "\n")
                for record in records:
                    stream.write(json.dumps(record) + "\n")

            dataset = module.load_dataset(path, 6, None, 1)
            self.assertEqual(len(dataset.records), 6)
            self.assertEqual(dataset.contexts.shape, (6, 2, module.CONTEXT_SIZE))
            self.assertEqual(
                dataset.queries.shape,
                (6, 2, module.COMBO_COUNT, distill.QUERY_SIZE),
            )
            targeted_path = Path(temporary) / "targeted.jsonl.gz"
            targeted_metadata = dict(metadata)
            targeted_metadata["target_actor"] = 0
            targeted_records = json.loads(json.dumps(records))
            for record in targeted_records:
                record["state"]["actor"] = 0
            for invalid_target_actor in (-1, 2, 0.0, "0", False):
                invalid_metadata = dict(targeted_metadata)
                invalid_metadata["target_actor"] = invalid_target_actor
                with gzip.open(targeted_path, "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps(invalid_metadata) + "\n")
                    for record in targeted_records:
                        stream.write(json.dumps(record) + "\n")
                with self.assertRaisesRegex(ValueError, "invalid.*target actor"):
                    module.load_dataset(targeted_path, 6, None, 1)
            with gzip.open(targeted_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(targeted_metadata) + "\n")
                for record in targeted_records:
                    stream.write(json.dumps(record) + "\n")
            targeted = module.load_dataset(targeted_path, 6, None, 1)
            self.assertTrue(np.all(targeted.actors == 0))
            targeted_records[0]["state"]["actor"] = 1
            with gzip.open(targeted_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(targeted_metadata) + "\n")
                for record in targeted_records:
                    stream.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid causal range-policy"):
                module.load_dataset(targeted_path, 6, None, 1)
            for invalid_record_actor in (0.0, "0", False):
                targeted_records[0]["state"]["actor"] = invalid_record_actor
                with gzip.open(targeted_path, "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps(targeted_metadata) + "\n")
                    for record in targeted_records:
                        stream.write(json.dumps(record) + "\n")
                with self.assertRaisesRegex(ValueError, "invalid causal range-policy"):
                    module.load_dataset(targeted_path, 6, None, 1)
            self_play_path = Path(temporary) / "self-play.jsonl.gz"
            self_play_metadata = dict(metadata)
            self_play_metadata["schema"] = module.SELF_PLAY_SCHEMA
            self_play_records = []
            for record in records:
                converted = dict(record)
                converted["record_type"] = "range_conditioned_self_play_regret"
                converted["action_value_standard_errors_bb"] = [0.01, 0.01]
                self_play_records.append(converted)
            with gzip.open(self_play_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(self_play_metadata) + "\n")
                for record in self_play_records:
                    stream.write(json.dumps(record) + "\n")
            self_play_dataset = module.load_dataset(self_play_path, 6, None, 1)
            self.assertEqual(len(self_play_dataset.records), 6)
            np.testing.assert_allclose(
                self_play_dataset.action_values, dataset.action_values
            )
            self.assertIsNotNone(self_play_dataset.action_value_standard_errors)
            np.testing.assert_allclose(
                self_play_dataset.action_value_standard_errors[:, :2], 0.01
            )
            candidate = np.tile([0.4, 0.6], (6, 1)).astype(np.float32)
            measured = module.metrics(candidate, dataset)
            self.assertGreater(measured["weightedPolicyValueGainBb"], 0.0)
            self.assertGreater(measured["weightedReverseKlFromFrozen"], 0.0)
            measured_self_play = module.metrics(candidate, self_play_dataset)
            self.assertGreater(
                measured_self_play[
                    "weightedPolicyValueGainActionRolloutStandardErrorBb"
                ],
                0.0,
            )
            self.assertLess(
                measured_self_play[
                    "weightedPolicyValueGainActionRolloutLowerBound99Bb"
                ],
                measured_self_play["weightedPolicyValueGainBb"],
            )
            invalid_self_play = dict(self_play_records[0])
            invalid_self_play["action_value_standard_errors_bb"] = [0.01, -0.01]
            with gzip.open(self_play_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(self_play_metadata) + "\n")
                stream.write(json.dumps(invalid_self_play) + "\n")
                for record in self_play_records[1:]:
                    stream.write(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid causal range-policy"):
                module.load_dataset(self_play_path, 6, None, 1)
            local_control = module.metrics(candidate, dataset, frozen=candidate)
            self.assertAlmostEqual(local_control["weightedPolicyValueGainBb"], 0.0)
            self.assertAlmostEqual(local_control["weightedReverseKlFromFrozen"], 0.0)
            self.assertAlmostEqual(local_control["weightedL1ActionDelta"], 0.0)
            arithmetic_drift = np.tile([0.5005, 0.4995], (6, 1)).astype(np.float32)
            self.assertTrue(
                module.source_parity_metrics(arithmetic_drift, dataset)["accepted"]
            )
            primary_drift = np.tile([0.4995, 0.5005], (6, 1)).astype(np.float32)
            self.assertFalse(
                module.source_parity_metrics(primary_drift, dataset)["accepted"]
            )
            rare_primary_drift = dataset.current.copy()
            rare_primary_drift[0] = [0.4995, 0.5005]
            rare_primary_dataset = replace(
                dataset,
                weights=np.asarray([1e-6, 1, 1, 1, 1, 1], dtype=np.float64),
            )
            rare_primary_metrics = module.source_parity_metrics(
                rare_primary_drift, rare_primary_dataset
            )
            self.assertGreaterEqual(
                rare_primary_metrics["primaryActionAgreement"],
                module.MINIMUM_SOURCE_PARITY_PRIMARY_AGREEMENT,
            )
            self.assertTrue(rare_primary_metrics["accepted"])
            skewed_current = dataset.current.copy()
            skewed_current[0] = [0.001, 0.999]
            skewed_measured = skewed_current.copy()
            skewed_measured[0] = [0.0034, 0.9966]
            skewed_dataset = replace(
                dataset,
                current=skewed_current,
                weights=np.asarray([1e-6, 1, 1, 1, 1, 1], dtype=np.float64),
            )
            excessive_backend_drift = module.source_parity_metrics(
                skewed_measured, skewed_dataset
            )
            self.assertLessEqual(
                excessive_backend_drift["maximumAbsoluteError"],
                module.MAXIMUM_SOURCE_PARITY_ABSOLUTE_ERROR,
            )
            self.assertLessEqual(
                excessive_backend_drift["weightedReverseKlFromFrozen"],
                module.MAXIMUM_SOURCE_PARITY_WEIGHTED_KL,
            )
            self.assertGreater(
                excessive_backend_drift["maximumReverseKlFromFrozen"],
                module.MAXIMUM_SOURCE_PARITY_NODE_KL,
            )
            self.assertFalse(excessive_backend_drift["accepted"])

            source_path = Path(temporary) / "source.json"
            source_model = distill.RangeConditionedPolicy("compact", "replace")
            source_model.head.layers[-1].weight = mx.zeros_like(
                source_model.head.layers[-1].weight
            )
            source_model.head.layers[-1].bias = mx.zeros_like(
                source_model.head.layers[-1].bias
            )
            distill.export_model(
                source_model,
                source_path,
                103,
                SimpleNamespace(
                    metadata={"depth_bb": 20.0, "source_policy_baseline": {}},
                    sha256="c" * 64,
                ),
                SimpleNamespace(sha256="d" * 64),
            )
            for full_corpus_gradient, paired_corpus_gradient in (
                (False, False),
                (True, False),
                (True, True),
            ):
                trained, _, diagnostics = module.train_candidate(
                    source_path,
                    dataset,
                    dataset,
                    107,
                    1,
                    2,
                    1e-5,
                    0.1,
                    0.002,
                    0.005,
                    0.0015,
                    0.25,
                    1.0,
                    1,
                    full_corpus_gradient,
                    None,
                    paired_corpus_gradient,
                )
                self.assertLess(
                    diagnostics["sourceParity"]["maximumAbsoluteError"], 1e-6
                )
                self.assertEqual(diagnostics["selectedCheckpoint"]["step"], 1)
                self.assertEqual(
                    diagnostics["pairedCorpusGradient"], paired_corpus_gradient
                )
                self.assertEqual(
                    diagnostics["targetAnchorProbabilities"],
                    "exact_stored_rust_training_and_mlx_cross_seed_probabilities",
                )
                del trained

            # A cross-seed corpus stores the other policy's Rust probabilities.
            # Those probabilities are evidence about its attribution policy, not
            # the candidate source's frozen baseline on the same public states.
            # Keep this deliberately far from the uniform source so a regression
            # to anchoring validation metrics at ``validation.current`` fails.
            cross_seed_current = np.tile([0.9, 0.1], (6, 1)).astype(np.float32)
            cross_seed_dataset = replace(dataset, current=cross_seed_current)
            trained, _, diagnostics = module.train_candidate(
                source_path,
                dataset,
                cross_seed_dataset,
                108,
                1,
                2,
                1e-5,
                0.1,
                0.002,
                0.005,
                0.0015,
                0.25,
                1.0,
                1,
                True,
                None,
                True,
            )
            validation = diagnostics["selectedCheckpoint"]["validation"]
            self.assertLess(validation["maximumReverseKlFromFrozen"], 0.005)
            self.assertGreater(validation["weightedPolicyValueGainBb"], 0.0)
            self.assertTrue(
                diagnostics["selectedCheckpoint"]["insideRealizedTrustRegion"]
            )
            del trained

            attribution_path = Path(temporary) / "attribution.json.gz"
            attribution_payload = json.loads(source_path.read_text())
            attribution_payload["seed"] = 109
            with gzip.open(attribution_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(attribution_payload) + "\n")
            self.assertNotEqual(
                module.sha256(source_path), module.sha256(attribution_path)
            )
            trained, _, diagnostics = module.train_candidate(
                source_path,
                dataset,
                dataset,
                111,
                1,
                2,
                1e-5,
                0.1,
                0.002,
                0.005,
                0.0015,
                0.25,
                1.0,
                1,
                False,
                attribution_path,
            )
            self.assertIsNone(diagnostics["sourceParity"])
            self.assertFalse(diagnostics["attributionParityRequiredForTraining"])
            self.assertTrue(diagnostics["attributionParity"]["accepted"])
            self.assertEqual(
                diagnostics["attributionRangePolicySha256"],
                module.sha256(attribution_path),
            )
            self.assertEqual(
                diagnostics["targetAnchorRangePolicySha256"],
                module.sha256(source_path),
            )
            self.assertEqual(
                diagnostics["targetAnchorProbabilities"],
                "mlx_frozen_parent_probabilities",
            )
            del trained

    def test_cap_preserves_two_rows_per_street(self):
        records = [
            {
                "weight": float(index + 1),
                "state": {"street": ("flop", "turn", "river")[index % 3]},
                "focal_combo": index,
                "action_labels": ["check"],
            }
            for index in range(12)
        ]
        selected = module._cap_records(records, 6)
        counts = {
            street: sum(record["state"]["street"] == street for record in selected)
            for street in ("flop", "turn", "river")
        }
        self.assertEqual(counts, {"flop": 2, "turn": 2, "river": 2})


if __name__ == "__main__":
    unittest.main()

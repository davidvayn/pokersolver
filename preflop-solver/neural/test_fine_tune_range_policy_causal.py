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
                "datasetSha256": module.sha256(dataset),
                "validation": {"status": "accepted_for_directional_evaluation"},
            }
            completed = SimpleNamespace(stdout=json.dumps(report))
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
                )
            self.assertEqual(measured, report)
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["solver", "range-policy-causal-evaluate"])
            run.assert_called_once_with(
                command, check=True, capture_output=True, text=True
            )
            report["datasetSha256"] = "0" * 64
            with patch.object(
                module.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=json.dumps(report)),
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
                    )

    def test_loader_builds_exact_belief_features_and_metrics_reward_better_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "causal.jsonl.gz"
            records = []
            for index, street in enumerate(("flop", "flop", "turn", "turn", "river", "river")):
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
                (6, 2, module.COMBO_COUNT, module.QUERY_SIZE),
            )
            candidate = np.tile([0.4, 0.6], (6, 1)).astype(np.float32)
            measured = module.metrics(candidate, dataset)
            self.assertGreater(measured["weightedPolicyValueGainBb"], 0.0)
            self.assertGreater(measured["weightedReverseKlFromFrozen"], 0.0)
            local_control = module.metrics(candidate, dataset, frozen=candidate)
            self.assertAlmostEqual(local_control["weightedPolicyValueGainBb"], 0.0)
            self.assertAlmostEqual(local_control["weightedReverseKlFromFrozen"], 0.0)
            self.assertAlmostEqual(local_control["weightedL1ActionDelta"], 0.0)
            arithmetic_drift = np.tile([0.5005, 0.4995], (6, 1)).astype(
                np.float32
            )
            self.assertTrue(
                module.source_parity_metrics(arithmetic_drift, dataset)["accepted"]
            )
            primary_drift = np.tile([0.4995, 0.5005], (6, 1)).astype(
                np.float32
            )
            self.assertFalse(
                module.source_parity_metrics(primary_drift, dataset)["accepted"]
            )
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
            trained, _, diagnostics = module.train_candidate(
                source_path,
                dataset,
                dataset,
                107,
                1,
                1,
                1e-5,
                0.1,
                0.002,
                0.005,
                0.0015,
                0.25,
                1.0,
                1,
            )
            self.assertLess(
                diagnostics["sourceParity"]["maximumAbsoluteError"], 1e-6
            )
            self.assertEqual(diagnostics["selectedCheckpoint"]["step"], 1)
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

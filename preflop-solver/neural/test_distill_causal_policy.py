import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

import distill_causal_policy as causal
from train import ActionScorer, scorer_json


class CausalPolicyTrustRegionTests(unittest.TestCase):
    def test_mirror_target_improves_policy_value_within_node_kl_bound(self):
        current = np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32)
        values = np.asarray([[-1.0, 1.0, 0.0]], dtype=np.float32)
        masks = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
        targets = causal.mirror_descent_targets(
            current, values, masks, step_per_bb=10.0, maximum_node_kl=0.001
        )
        kl = causal.categorical_kl(targets, current)[0]
        self.assertGreater(targets[0, 1], current[0, 1])
        self.assertLessEqual(kl, 0.001 + 1e-8)
        self.assertAlmostEqual(float(np.sum(targets)), 1.0, places=7)
        self.assertEqual(targets[0, 2], 0.0)

    def test_equal_action_values_leave_the_frozen_policy_unchanged(self):
        current = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32)
        values = np.asarray([[4.0, 4.0, 4.0]], dtype=np.float32)
        masks = np.ones_like(current)
        targets = causal.mirror_descent_targets(
            current, values, masks, step_per_bb=0.5, maximum_node_kl=0.01
        )
        np.testing.assert_allclose(targets, current, atol=1e-7, rtol=0.0)

    def test_policy_metrics_measure_negated_responder_value_gain(self):
        data = {
            "current": np.asarray([[0.5, 0.5]], dtype=np.float32),
            "action_values_bb": np.asarray([[-2.0, 1.0]], dtype=np.float32),
            "masks": np.ones((1, 2), dtype=np.float32),
            "weights": np.ones((1, 1), dtype=np.float32),
        }
        candidate = np.asarray([[0.25, 0.75]], dtype=np.float32)
        metrics = causal.policy_metrics(candidate, data)
        self.assertAlmostEqual(metrics["weightedBaselinePolicyValueBb"], -0.5)
        self.assertAlmostEqual(metrics["weightedCandidatePolicyValueBb"], 0.25)
        self.assertAlmostEqual(metrics["weightedPolicyValueGainBb"], 0.75)
        self.assertGreater(metrics["weightedReverseKlFromFrozen"], 0.0)

    def test_source_identity_uses_exact_exported_parameters(self):
        model = ActionScorer(725, (2, 2))
        mx.eval(model.parameters())
        scorer = scorer_json(model)
        bundle = {"postflop_networks": [scorer, scorer]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "networks.json"
            payload = json.dumps(bundle, separators=(",", ":")).encode()
            path.write_bytes(payload)
            causal.validate_source_artifact_identity(
                model, path, hashlib.sha256(payload).hexdigest()
            )
            bundle["postflop_networks"][1] = {"layers": []}
            changed = json.dumps(bundle, separators=(",", ":")).encode()
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "parameter-identical"):
                causal.validate_source_artifact_identity(
                    model, path, hashlib.sha256(changed).hexdigest()
                )


if __name__ == "__main__":
    unittest.main()

import copy
import json
import unittest
from pathlib import Path
import numpy as np
from cash_profiles import (CASH_DATASET_SCHEMA, PAYOFF_CONTRACT, profile_rules, rules_digest,
    validate_cash_metadata, validate_training_profile)
from serving_value_projection import OwnPayoffValueProjection, own_payoff_conservation_residual
import tempfile
from train import ActionScorer, INPUT_FEATURE_COUNT, input_feature_count, export_traversal_networks
from train import parse_args


class CashTrainingTests(unittest.TestCase):
    def test_requested_cash_depths_are_trainable_not_automatically_home_or_active(self):
        for depth in (20,40,50,100,200,1000,2000):
            args = parse_args(["--run-dir","/tmp/cash-depth-test","--seed","41","--cash-profile","nl25","--depth-bb",str(depth),"--variance-baseline-scale","0"])
            validate_training_profile(args.cash_profile,args.depth_bb,args.variance_baseline_scale)
        with self.assertRaisesRegex(ValueError,"legacy Home"):
            validate_training_profile(None,40,0.)

    def test_cash_weights_require_and_retain_the_actual_betting_abstraction(self):
        models = {name: ActionScorer(input_feature_count("nl25"), (4, 4), 2 if name == "value" else 1)
                  for name in ("advantage_p0", "advantage_p1", "value")}
        grid = {"open_sizes_bb": [2., 2.5, 3.]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            with self.assertRaisesRegex(ValueError, "action abstraction"):
                export_traversal_networks(models, path, 0., 20, "nl25")
            self.assertFalse(path.exists())
            export_traversal_networks(models, path, 0., 20, "nl25", grid)
            artifact = json.loads(path.read_text())
            self.assertEqual(artifact["cash_action_abstraction"], grid)
            self.assertNotIn("sampling_baseline", artifact)
            self.assertEqual(artifact["schema"], "hu-neural-own-payoff-training-networks-v3")
            self.assertEqual(artifact["input_size"],829)
            with self.assertRaisesRegex(ValueError,"features"):
                export_traversal_networks(models,path,.5,20)
            home_models = {name: ActionScorer(INPUT_FEATURE_COUNT,(4,4),2 if name == "value" else 1) for name in models}
            export_traversal_networks(home_models, path, .5, 20)
            self.assertNotIn("cash_action_abstraction", json.loads(path.read_text()))
            with self.assertRaisesRegex(ValueError,"features"):
                export_traversal_networks(home_models,path,0.,20,"nl25",grid)

    def test_digest_matches_shared_rust_and_typescript_fixtures(self):
        fixture = Path(__file__).resolve().parents[2] / "data/practice/cash-settlement-fixtures.json"
        self.assertEqual(rules_digest(profile_rules("nl25")), json.loads(fixture.read_text())["identities"]["nl25"]["sha256"])
        control = profile_rules("nl25-rake-off-control")
        self.assertEqual(control["blindsUnits"], [10, 25])
        self.assertNotEqual(rules_digest(control), rules_digest(profile_rules("nl25")))
        self.assertIsNone(profile_rules(None))

    def test_targets_cannot_cross_rules_or_payoff_contracts(self):
        rules = profile_rules("nl25")
        metadata = dict(schema=CASH_DATASET_SCHEMA, cash_rules=rules, rules_sha256=rules_digest(rules), payoff_contract=PAYOFF_CONTRACT,cash_terminal_action_integration=True)
        validate_cash_metadata(metadata, "nl25")
        for broken in [dict(metadata, rules_sha256="0" * 64), dict(metadata, schema="hu-neural-traversal-jsonl-v7"), dict(metadata, payoff_contract="zero-sum"),dict(metadata,cash_terminal_action_integration=False)]:
            with self.assertRaises(ValueError): validate_cash_metadata(broken, "nl25")
        with self.assertRaises(ValueError): validate_cash_metadata(metadata, "nl25-rake-off-control")
        altered = copy.deepcopy(rules); altered["blindsUnits"][0] = 12
        with self.assertRaises(ValueError): rules_digest(altered)
        with self.assertRaises(ValueError): validate_training_profile("nl25", 20, .5)
        validate_training_profile("nl25", 20, 0)

    def test_own_head_projection_does_not_couple_or_cancel_rake(self):
        values = np.zeros((1, 2, 1326)); values[:, 0] = 4.56; values[:, 1] = -5.
        legal = np.ones((1, 1326), bool); legal[0, 1] = False
        projection = OwnPayoffValueProjection(values, legal)
        self.assertEqual(projection.values[0, 0, 0], 4.56)
        self.assertEqual(projection.values[0, 1, 0], -5.)
        self.assertEqual(projection.values[0, 0, 1], 0.)
        upstream = np.zeros_like(values); upstream[:, 0] = 1.
        gradient = projection.vjp(upstream)
        self.assertEqual(gradient[:, 1].sum(), 0.)
        weights = np.ones_like(values); weights[:, :, 1] = 0
        self.assertLess(own_payoff_conservation_residual(projection.values, weights, np.array([.44]))[0], 1e-12)
        with self.assertRaises(ValueError): own_payoff_conservation_residual(values, weights * np.array([1, 2])[None, :, None], np.array([.44]))

    def test_own_projection_gradient_matches_finite_differences(self):
        random = np.random.default_rng(41)
        values = random.normal(size=(1, 2, 1326)); legal = np.ones((1, 1326), bool)
        upstream = random.normal(size=values.shape)
        projection = OwnPayoffValueProjection(values, legal, 1.)
        derivative = projection.vjp(upstream)
        for index in [(0,0,0),(0,1,5),(0,1,10)]:
            plus, minus = values.copy(), values.copy(); plus[index] += 1e-6; minus[index] -= 1e-6
            estimate = ((OwnPayoffValueProjection(plus, legal, 1.).values - OwnPayoffValueProjection(minus, legal, 1.).values) * upstream).sum() / 2e-6
            self.assertAlmostEqual(derivative[index], estimate, places=7)


if __name__ == "__main__": unittest.main()

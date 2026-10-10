import copy
import unittest
from unittest.mock import patch

import numpy as np

import cash_synthetic_ranges as synthetic
from native_value_dataset import board_family, compatible_masses, legal_combos
from test_cash_value_dataset import label_fixture


class CashSyntheticRangesTests(unittest.TestCase):
    def test_seeded_requests_preserve_rules_money_mask_and_excluded_families(self):
        label = label_fixture()
        original = copy.deepcopy(label)
        # Range-generation semantics only: existing exact-kernel tests cover
        # runout equity. No fabricated values go into a training corpus here.
        def equity(board, ranges, investments, rules):
            return np.zeros((2,1326)),np.tile(np.linspace(0,1,1326),(2,1))
        with patch.object(synthetic,"exact_cash_checkdown_features",side_effect=equity):
            first = synthetic.synthetic_cash_turn_requests(label,[25,25,125,475],4,1783)
            second = synthetic.synthetic_cash_turn_requests(label,[25,25,125,475],4,1783)
            self.assertEqual(first,second)
            family = tuple(first["cases"][0]["flop_family"])
            excluded = synthetic.synthetic_cash_turn_requests(label,[25,25,125,475],4,1783,[family])
        self.assertEqual(label,original)
        self.assertNotIn(family,[tuple(row["flop_family"]) for row in excluded["cases"]])
        self.assertEqual(len({tuple(row["flop_family"]) for row in first["cases"]}),4)
        self.assertFalse(first["own_values_generated"])
        self.assertFalse(first["policies_generated"])
        self.assertFalse(first["active"])
        for case in first["cases"]:
            config = case["solve_input"]
            state = config["state"]
            self.assertEqual(config["game"],label["input"]["game"])
            self.assertEqual(config["iterations"],64)
            self.assertEqual(state["street"],"turn")
            self.assertEqual(state["public_history"],["public_belief:turn_start"])
            self.assertEqual(state["invested_bb"],[case["investment_units"]/25]*2)
            self.assertEqual(board_family(state["board"]),tuple(case["flop_family"]))
            ranges = np.asarray(state["ranges"])
            legal = legal_combos(state["board"])
            self.assertTrue(np.isfinite(ranges).all())
            self.assertTrue(np.all(ranges[:,~legal] == 0))
            self.assertTrue(np.all(ranges[:,legal] >= .2/legal.sum()-1e-15))
            np.testing.assert_allclose(ranges.sum(axis=1),1,atol=2e-15)
            joint = np.sum(ranges*compatible_masses(ranges),axis=1)
            self.assertGreater(min(joint),0)
            self.assertAlmostEqual(joint[0],joint[1],places=13)
            self.assertGreater(float(np.max(np.abs(ranges[0]-ranges[1]))),.001)

    def test_invalid_budgets_fail_before_equity_generation(self):
        label = label_fixture()
        with patch.object(synthetic,"exact_cash_checkdown_features") as equity:
            for count in (0,65,True,1.5):
                with self.assertRaisesRegex(ValueError,"count"):
                    synthetic.synthetic_cash_turn_requests(label,[25],count,1783)
            for seed in (-1,2**32,True,1.5):
                with self.assertRaisesRegex(ValueError,"seed"):
                    synthetic.synthetic_cash_turn_requests(label,[25],1,seed)
            for amounts in ([],[24],[500],[25.], [True]):
                with self.assertRaisesRegex(ValueError,"commitments"):
                    synthetic.synthetic_cash_turn_requests(label,amounts,1,1783)
            equity.assert_not_called()

    def test_reference_rule_and_target_corruption_is_not_accepted_as_lineage(self):
        label = label_fixture()
        label["input"]["game"]["small_blind_bb"] = .5
        with self.assertRaisesRegex(ValueError,"pinned"):
            synthetic.synthetic_cash_turn_requests(label,[25],1,1783)

    def test_invalid_exclusion_and_strength_cannot_silently_contaminate_coverage(self):
        label = label_fixture()
        for families in ([[1,2,3]], [(True,4,8)], [(1,1,2)], [(52,53,54)]):
            with self.assertRaisesRegex(ValueError,"exclusions"):
                synthetic.synthetic_cash_turn_requests(label,[25],1,1783,families)
        for values in (np.zeros((2,10)), np.full((2,1326),np.nan),np.full((2,1326),1.01)):
            with patch.object(synthetic,"exact_cash_checkdown_features",return_value=(None,values)):
                with self.assertRaisesRegex(ValueError,"exact equity"):
                    synthetic.synthetic_cash_turn_requests(label,[25],1,1783)


if __name__ == "__main__":
    unittest.main()

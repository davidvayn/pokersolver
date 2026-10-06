import copy
import math
import unittest

from analyze_preflop_lineage import (
    CLASSES, ROOT_HISTORY, aggregate, continuation_route_comparison, indexed_roots, multiplicity,
    probabilities, root_comparison, served_roots, value_comparison,
)

LABELS = ["fold", "limp", "raise_to_2.000bb", "raise_all_in_to_20.000bb"]


def rows():
    return [dict(key=f"p0|{hand}|blinds:0.500/1.000", actor=0,
                 hand_class=hand, public_history=ROOT_HISTORY,
                 action_labels=LABELS, probabilities=[0.1, 0.2, 0.6, 0.1])
            for hand in sorted(CLASSES)]


def value_artifact(roots, evs):
    return dict(players=[[dict(**row, player=0, policy_probabilities=row["probabilities"],
                               action_values_bb=evs) for row in roots.values()], []])


class PreflopLineageTests(unittest.TestCase):
    def test_canonical_classes_and_combo_weights(self):
        self.assertEqual(len(CLASSES), 169)
        self.assertEqual(sum(multiplicity(h) for h in CLASSES), 1326)
        self.assertEqual([multiplicity(h) for h in ["AA", "AKs", "AKo"]], [6, 4, 12])
        with self.assertRaises(ValueError):
            multiplicity("2Ko")

    def test_missing_duplicate_and_hidden_root_errors_fail(self):
        fixture = rows()
        for broken in [fixture[:-1], fixture + [fixture[0]]]:
            with self.assertRaises(ValueError):
                indexed_roots(broken)
        broken = copy.deepcopy(fixture)
        broken[0]["key"] = "wrong"
        with self.assertRaises(ValueError):
            indexed_roots(broken)

    def test_invalid_probabilities_are_not_normalized_away(self):
        for vector in [[0.8, 0.3], [1.01, -0.01], [math.nan, 0], [True, 0], [0, 0], []]:
            with self.assertRaises(ValueError):
                probabilities(vector, 2)

    def test_root_comparison_uses_combo_not_class_or_visit_weight(self):
        first = indexed_roots(rows())
        second = copy.deepcopy(first)
        second["AKo"]["probabilities"] = [0.2, 0.1, 0.6, 0.1]
        comparison = root_comparison(first, second)
        self.assertAlmostEqual(comparison["comboWeightedTotalVariation"], 0.1 * 12 / 1326)
        self.assertAlmostEqual(comparison["comboWeightedActionMae"], 0.05 * 12 / 1326)

    def test_action_reordering_is_not_silently_accepted(self):
        first = indexed_roots(rows())
        second = copy.deepcopy(first)
        second["AA"]["action_labels"] = list(reversed(LABELS))
        with self.assertRaises(ValueError):
            root_comparison(first, second)

    def test_open_sizes_collapse_only_in_external_aggregate(self):
        roots = indexed_roots(rows())
        for row in roots.values():
            row["action_labels"] = LABELS[:-1] + ["raise_to_3.000bb", LABELS[-1]]
            row["probabilities"] = [0.1, 0.2, 0.3, 0.3, 0.1]
        result = aggregate(roots)
        self.assertAlmostEqual(result["raise"], 0.6)
        self.assertAlmostEqual(result["shove"], 0.1)

    def test_native_transport_preserves_labels_and_complete_multiplicities(self):
        student = indexed_roots(rows())
        context = dict(context="sb-first-in", queriedCombos=1326,
                       maximumWithinClassProbabilitySpread=0,
                       actionNames=["fold:", "call:", "raise:2", "all_in:20"],
                       hands=[dict(hand=h, combos=multiplicity(h),
                                   probabilities=row["probabilities"]) for h, row in student.items()])
        receipt = dict(schema="served-preflop-range-audit-v1", summaries=[context],
                       assumptions=dict(depthBb=20, blindsBb=[0.5, 1], anteBb=0,
                                        rake="none", cardCombos=1326))
        self.assertEqual(root_comparison(student, served_roots(receipt, student))["maximumActionDelta"], 0)
        context["hands"][0]["combos"] += 1
        with self.assertRaises(ValueError):
            served_roots(receipt, student)

    def test_native_suit_asymmetry_cannot_hide_in_class_averages(self):
        receipt = dict(schema="served-preflop-range-audit-v1",
                       assumptions=dict(depthBb=20, blindsBb=[0.5, 1], anteBb=0,
                                        rake="none", cardCombos=1326),
                       summaries=[dict(context="sb-first-in", queriedCombos=1326,
                                       maximumWithinClassProbabilitySpread=0.1)])
        with self.assertRaisesRegex(ValueError, "suit invariant"):
            served_roots(receipt, indexed_roots(rows()))

    def test_different_value_games_are_not_certified_as_matched(self):
        roots = indexed_roots(rows())
        result = value_comparison(value_artifact(roots, [0, -1, -2, -3]),
                                  value_artifact(roots, [0, 1, -2, -3]), roots, roots)
        self.assertEqual(result["oldFoldNowContinueClasses"], 169)
        self.assertEqual(result["oldFoldNowContinueComboFraction"], 1)
        self.assertFalse(result["identicalContinuationGameEstablished"])

    def test_values_require_their_actual_source_probabilities(self):
        roots = indexed_roots(rows())
        values = value_artifact(roots, [0, 1, 2, 3])
        values["players"][0][0]["policy_probabilities"] = [0.25] * 4
        with self.assertRaisesRegex(ValueError, "source policy"):
            value_comparison(values, values, roots, roots)

    def test_same_iterations_do_not_establish_same_continuation_game(self):
        result = continuation_route_comparison(
            dict(iterations=2, dcfr=dict(positive_regret_exponent=1.5,
                                        negative_regret_exponent=0, strategy_exponent=2)),
            dict(resolver=dict(flopIterations=2, flopResolvedActor=1),
                 dcfr=dict(positiveRegretExponent=1.5, negativeRegretExponent=0,
                           strategyExponent=0)))
        self.assertTrue(result["checks"]["iterationsMatch"])
        self.assertFalse(result["checks"]["actorRoutingMatch"])
        self.assertFalse(result["checks"]["dcfrMatch"])
        self.assertFalse(result["canUseAsMatchedServingEvaluation"])


if __name__ == "__main__":
    unittest.main()

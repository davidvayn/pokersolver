import math
import unittest

from run_accurate_full_hand_pair import compare, validate_cluster
from run_native_full_hand_preflight import expected_route_sha256


def row(iterations, values=(1, 2)):
    """Synthetic validator fixture only; never scored as a trained model."""
    return dict(schema="accurate-native-full-hand-lbr-cluster-v1", releaseAccepted=False,
        cohort="screening", index=0, chanceSeed=10, deal=dict(holes=[[0,1],[2,3]], board=[4,5,6,7,8]),
        diagnostics=dict(preflopSha256="a"*64, preflopRounds=32, learnedLeafModelSha256=None,
            completeRootSupport=True, rootRealizationTurnAverages=False, safeResolving=False,
            releaseQualified=False, executionLeafWorkers=4, flopIterations=iterations,
            trainingTurnIterations=64, responseTurnIterations=64,
            routeSha256=expected_route_sha256("a"*64, iterations)),
        report=dict(schema="paired-frozen-lbr-hand-v1", lbrSeed=90001, earlyRunoutsPerCombo=16,
            actionSeed=20, seatAttackerUtilityBb=list(values), pairedTotalResponseGainBb=sum(values),
            attacks=[dict(utility=v, decisions=[1,0,0,0], actions=[dict(action="Call",legalActions=["Call","Fold"])])
                     for v in values]))


class AccurateFullHandPairTest(unittest.TestCase):
    def test_matched_total_scale_and_no_single_deal_promotion(self):
        value = compare([row(32), row(64, (-1,0))], "a"*64)
        self.assertEqual(value["pairedImprovementBbPerHand"], 4)
        self.assertEqual(value["dealClusters"], 1)
        self.assertFalse(value["releaseAccepted"])
        self.assertFalse(value["fullGameGateEvaluated"])
        self.assertEqual(value["native64"]["postflopAttackerDecisions"], 0)
        with self.assertRaises(ValueError): compare([row(32)], "a"*64)

    def test_different_cards_seeds_or_policy_budgets_are_not_matched(self):
        for key, replacement in (("chanceSeed",11),("index",1),("cohort","holdout"),
                ("deal",dict(holes=[[9,1],[2,3]],board=[4,5,6,7,8]))):
            b = row(64); b[key] = replacement
            with self.assertRaises(ValueError): compare([row(32),b],"a"*64)
        b = row(64); b["report"]["actionSeed"] = 21
        with self.assertRaises(ValueError): compare([row(32),b],"a"*64)
        with self.assertRaises(ValueError): compare([row(64),row(64)],"a"*64)
        with self.assertRaises(ValueError): validate_cluster(row(32),"b"*64,32)

    def test_invalid_or_incomplete_attack_and_accounting_rejected(self):
        for values in ((21,0),(math.nan,0),(math.inf,0)):
            with self.assertRaises(ValueError): validate_cluster(row(32,values),"a"*64,32)
        a = row(32); a["report"]["pairedTotalResponseGainBb"] = 1
        with self.assertRaises(ValueError): validate_cluster(a,"a"*64,32)
        for key, replacement in (("utility",math.nan),("utility",4),("decisions",[2,0,0,0]),("actions",[])):
            a = row(32); a["report"]["attacks"][0][key] = replacement
            with self.assertRaises(ValueError): validate_cluster(a,"a"*64,32)
        a = row(32); a["deal"]["holes"][0][0] = 4
        with self.assertRaises(ValueError): validate_cluster(a,"a"*64,32)

    def test_a_preflop_terminal_hand_is_kept_not_replaced(self):
        a = row(32); b = row(64)
        for value in (a,b):
            value["report"]["attacks"][1]["decisions"] = [0,0,0,0]
            value["report"]["attacks"][1]["actions"] = []
        result = compare([a,b],"a"*64)
        self.assertEqual(result["pairedImprovementBbPerHand"],0)
        self.assertEqual(result["native32"]["postflopAttackerDecisions"],0)


if __name__ == "__main__": unittest.main()

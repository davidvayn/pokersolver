import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from run_accurate_full_hand_pair import compare, expansion, pin_first_pair, validate_cluster
from run_native_full_hand_preflight import expected_route_sha256


def row(iterations, values=(1, 2), index=0, seed=100101):
    """Synthetic validator fixture only; never scored as a trained model."""
    return dict(schema="accurate-native-full-hand-lbr-cluster-v1", releaseAccepted=False,
        cohort="screening", index=index, chanceSeed=10+index, deal=dict(holes=[[0,1],[2,3]], board=[4,5,6,7,8]),
        diagnostics=dict(preflopSha256="a"*64, preflopRounds=32, learnedLeafModelSha256=None,
            completeRootSupport=True, rootRealizationTurnAverages=False, safeResolving=False,
            releaseQualified=False, executionLeafWorkers=4, flopIterations=iterations,
            trainingTurnIterations=64, responseTurnIterations=64,
            routeSha256=expected_route_sha256("a"*64, iterations, seed)),
        report=dict(schema="paired-frozen-lbr-hand-v1", lbrSeed=90001, earlyRunoutsPerCombo=16,
            actionSeed=20+index, seatAttackerUtilityBb=list(values), pairedTotalResponseGainBb=sum(values),
            attacks=[dict(utility=v, decisions=[1,0,0,0], actions=[dict(action="Call",legalActions=["Call","Fold"])])
                     for v in values]))


class AccurateFullHandPairTest(unittest.TestCase):
    def test_expansion_is_preregistered_and_not_selected_by_payoff(self):
        for index, seed in ((0,100101),(1,100102),(2,100101),(3,100102)):
            self.assertEqual(expansion(index,index > 0),seed)
            value = compare([row(32,index=index,seed=seed),row(64,index=index,seed=seed)],"a"*64,index,seed)
            self.assertEqual(value["pairedImprovementBbPerHand"],0)
        for index, present in ((0,True),(1,False),(4,True),(-1,False)):
            with self.assertRaises(ValueError): expansion(index,present)

    def test_completed_first_admission_requires_both_real_reports_and_resource_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); binary = path/"binary"; preflop = path/"preflop"
            pinned = {str(binary):"b"*64,str(preflop):"a"*64}; jobs=[]
            rows = [row(32),row(64,(-1,0))]
            for value,iterations in zip(rows,(32,64)):
                report = path/f"native{iterations}.json"; report.write_text(json.dumps(value))
                jobs.append(dict(iterations=iterations,output=str(report),outputSha256="c"*64,
                    summary=validate_cluster(value,"a"*64,iterations),worker=dict(status="complete",
                        exitCode=0,resourceStopReason=None,sampledPeakMemoryBytes=1400000000)))
            prior = dict(schema="accurate-native-full-hand-pair-v1",status="complete",cohort="screening",
                index=0,policySeed=100101,pinnedInputs=dict(pinned),jobs=jobs,elapsedSeconds=3600,
                systemMemoryGuard=dict(stopReason=None),summary=compare(rows,"a"*64))
            pin_first_pair(path/"manifest.json",prior,pinned,binary,preflop)
            self.assertIn(str(path/"native64.json"),pinned)
            for changed in ({**prior,"status":"running"},{**prior,"jobs":jobs[:1]},
                    {**prior,"elapsedSeconds":7201},{**prior,"policySeed":100102}):
                with self.assertRaises(ValueError): pin_first_pair(path/"manifest.json",changed,pinned,binary,preflop)
            for field,value in (("exitCode",1),("sampledPeakMemoryBytes",3*1024**3),("resourceStopReason","memory")):
                changed=copy.deepcopy(prior); changed["jobs"][1]["worker"][field]=value
                with self.assertRaises(ValueError): pin_first_pair(path/"manifest.json",changed,pinned,binary,preflop)
            changed=copy.deepcopy(prior); changed["summary"]["pairedImprovementBbPerHand"] = 0
            with self.assertRaises(ValueError): pin_first_pair(path/"manifest.json",changed,pinned,binary,preflop)

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

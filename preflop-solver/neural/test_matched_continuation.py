import copy
import unittest
from run_matched_continuation import summarize


def fixtures():
    outputs = []
    for strength in (1, 2):
        for board in range(4 if strength == 1 else 8):
            # Equal class values, legitimate Hold'em class multiplicities.
            counts = [6]*13 + [4]*78 + [12]*78
            gap = [4, -6, 4, 4, 3, 3, 3, 3][board]
            row = dict(history=["root"], seat=0, foldCfvBb=[0.0]*169,
                callCfvBb=[gap]*169, ownPrefixReach=[0.5]*169,
                foldProbability=[0.2]*169, callProbability=[0.3]*169)
            second = copy.deepcopy(row); second.update(history=["other"], seat=1, ownPrefixReach=[0.0]*169)
            outputs.append(dict(schema="matched-call-fold-capture-v1", releaseAccepted=False,
                fullGameGateEvaluated=False, strength=strength, boardIndex=board,
                preflopSha256="p", modelSha256="m", kernelSha256="k", chanceSeed=71201,
                classes=list(range(169)), multiplicities=counts, board=[0, 1, board+2], turn=20,
                records=[row, second]))
    return outputs


class MatchedTests(unittest.TestCase):
    def test_select_after_chance_average_and_preserve_other_action_mass(self):
        cells = summarize(fixtures())["cells"]
        self.assertEqual(cells[0]["choices"][0], [0]*169)  # (4-6)/2 < 0
        self.assertEqual(cells[1]["choices"][0], [1]*169)
        self.assertAlmostEqual(cells[0]["meanEvaluationGainBb"], -0.45)
        self.assertAlmostEqual(cells[1]["meanEvaluationGainBb"], 0.30)

    def test_evaluation_values_never_choose_actions(self):
        values = fixtures(); before = summarize(values)
        for c in values:
            if c["boardIndex"] >= 4:
                for row in c["records"]: row["callCfvBb"] = [-100.0]*169
        after = summarize(values)
        self.assertEqual([c["choices"] for c in before["cells"]], [c["choices"] for c in after["cells"]])
        self.assertLess(after["cells"][1]["meanEvaluationGainBb"], 0)

    def test_refuse_unmatched_chance_or_incomplete_data(self):
        values = fixtures(); values[-1]["modelSha256"] = "other"
        with self.assertRaises(ValueError): summarize(values)
        with self.assertRaises(ValueError): summarize(fixtures()[:-1])
        values = fixtures(); values[4]["turn"] = 21
        with self.assertRaises(ValueError): summarize(values)


if __name__ == "__main__": unittest.main()

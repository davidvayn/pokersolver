import copy
import unittest
from run_compact_continuation_noise import summarize


class FixedNoiseTests(unittest.TestCase):
    def fixture(self,index):
        return dict(schema="frozen-preflop-continuation-noise-v1",releaseAccepted=False,
            preflopSha256="a"*64,modelSha256="b"*64,kernelSha256="c"*64,history=["open","call"],
            classes=list(range(169)),classReachWeights=[[1.0]*169]*2,boardIndex=index,board=[0,1,2],
            turnSampleCount=2,records=[dict(turn=3+i,conditionalStrategicResidualBb=[[float(index+i)]*169]*2) for i in range(2)])

    def test_cluster_variance_and_finite_population_correction(self):
        rows=[self.fixture(i) for i in range(4)]
        result=summarize(rows)
        self.assertEqual(result["independentBoardClusters"],4)
        self.assertAlmostEqual(result["reachWeightedWithinTurnVarianceBb2"],0.5)
        self.assertAlmostEqual(result["withinTurnContributionToBoardMeanVarianceBb2"],0.25*(1-2/49))
        self.assertAlmostEqual(result["observedBoardMeanVarianceBb2"],5/3)
        self.assertIsNone(summarize(rows[:1])["classMeanErrorRmsBb"])
        self.assertIsNone(result["fullHandExploitability"])
        self.assertFalse(result["releaseAccepted"])
        flat=[self.fixture(0) for _ in range(4)]
        for i,row in enumerate(flat): row["boardIndex"]=i
        self.assertLess(summarize(flat)["estimatedFlopVarianceComponentBb2"],0.0)
        for error in ("identity","turn","nan","duplicate"):
            bad=copy.deepcopy(rows)
            if error=="identity": bad[1]["preflopSha256"]="d"*64
            elif error=="turn": bad[1]["records"][1]["turn"]=3
            elif error=="nan": bad[1]["records"][1]["conditionalStrategicResidualBb"][0][0]=float("nan")
            else: bad[1]["boardIndex"]=0
            with self.assertRaises(ValueError): summarize(bad)


if __name__=="__main__": unittest.main()

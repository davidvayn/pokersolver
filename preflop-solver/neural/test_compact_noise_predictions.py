import copy
import unittest
from run_compact_noise_predictions import corrected_observations
import test_compact_continuation_noise as fixtures


class PredictionCorrectionTests(unittest.TestCase):
    def test_complete_chance_mean_removes_variation_even_with_biased_predictor(self):
        native=fixtures.FixedNoiseTests().fixture(0); native["policySha256"]="d"*64
        native["turnSampleCount"]=49
        native["records"]=[dict(turn=t,rawCfvBb=[[float(t)]*169]*2,
            sampledCheckdownCfvBb=[[0.0]*169]*2,conditionalStrategicResidualBb=[[float(t)]*169]*2) for t in range(3,52)]
        pred={k:copy.deepcopy(native[k]) for k in ("boardIndex","board","policySha256",
            "preflopSha256","modelSha256","kernelSha256","classes","classReachWeights")}
        pred.update(schema="fixed-continuation-turn-predictions-v1",releaseAccepted=False,
            classOpponentMass=[[1.0]*169]*2,predictions=[dict(turn=t,predictedRawCfvBb=[[float(t)-7.0]*169]*2) for t in range(3,52)])
        result=corrected_observations(native,pred)
        self.assertTrue(all(r["conditionalStrategicResidualBb"][0][0]==27.0 for r in result["records"]))
        self.assertEqual(native["records"][0]["conditionalStrategicResidualBb"][0][0],3.0)
        for error in ("missing","duplicate","identity"):
            bad=copy.deepcopy(pred)
            if error=="missing": bad["predictions"].pop()
            elif error=="duplicate": bad["predictions"][0]["turn"]=4
            else: bad["policySha256"]="e"*64
            with self.assertRaises(ValueError): corrected_observations(native,bad)


if __name__=="__main__": unittest.main()

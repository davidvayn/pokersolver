import unittest
import numpy as np

from action_contrast_dataset import contrast_loss_and_q_gradient
from decision_gradient_allocation import contrast_allocation, margin_loss_and_q_gradient, ranking_pilot_supported


class DecisionGradientAllocationTests(unittest.TestCase):
    def test_decomposition_reconstructs_real_unique_pair_huber(self):
        q=np.array([[4.,-2.,0.],[1.,4.,.2],[2.,3.,.3]])
        target=np.array([[2.,3.,.01],[0.,1.,.02],[1.,0.,.03]])
        weights=np.array([.2,.8,0.])
        pieces,report=contrast_allocation(q,target,weights)
        _,original=contrast_loss_and_q_gradient(q,target,weights)
        np.testing.assert_allclose(sum(pieces.values()),original,atol=1e-15)
        self.assertAlmostEqual(sum(report["pairDerivativeShares"].values()),1.)
        self.assertGreater(report["pairDerivativeShares"]["correct"],0.)
        self.assertGreater(report["pairDerivativeShares"]["inverted"],0.)
        self.assertAlmostEqual(report["nativeRankingLossBb"],1.6)

    def test_near_ties_not_misreported_as_wrong_rankings(self):
        pieces,report=contrast_allocation(np.array([[0.],[1.]]),np.array([[.02],[0.]]),np.ones(1))
        self.assertEqual(report["pairDerivativeShares"]["nearTie"],1.)
        np.testing.assert_array_equal(pieces["inverted"],0.)

    def test_margin_ignores_correct_large_gaps_and_near_ties(self):
        q=np.array([[2.,0.],[0.,1.]])
        target=np.array([[2.,.02],[0.,0.]])
        loss,gradient=margin_loss_and_q_gradient(q,target,np.ones(2))
        self.assertEqual(loss,0.)
        np.testing.assert_array_equal(gradient,0.)

    def test_actual_margin_gradient_matches_finite_differences(self):
        q=np.array([[-.8,1.],[.3,-.5],[.9,.2]])
        target=np.array([[1.,.1],[.2,.8],[0.,.3]])
        weights=np.array([.6,.4])
        _,gradient=margin_loss_and_q_gradient(q,target,weights)
        for index in np.ndindex(q.shape):
            high,low=q.copy(),q.copy();high[index]+=1e-6;low[index]-=1e-6
            numerical=(margin_loss_and_q_gradient(high,target,weights)[0]-margin_loss_and_q_gradient(low,target,weights)[0])/2e-6
            self.assertAlmostEqual(gradient[index],numerical,places=10)
        np.testing.assert_allclose(gradient.sum(axis=0),0.,atol=1e-15)

    def test_invalid_or_partial_inputs_fail_closed(self):
        rows=[dict(seed=s,step=t,meanNativeRankingLossBb=.1,meanCorrectDerivativeShare=.8)
            for s in (10601,10602) for t in (0,600)]
        self.assertTrue(ranking_pilot_supported(rows))
        rows[-1]["meanCorrectDerivativeShare"]=.7
        self.assertFalse(ranking_pilot_supported(rows))
        with self.assertRaises(ValueError): ranking_pilot_supported(rows[:-1])
        for weights in (np.zeros(1),np.array([-1.]),np.array([float("nan")])):
            with self.assertRaises(ValueError): contrast_allocation(np.zeros((2,1)),np.ones((2,1)),weights)


if __name__=="__main__":unittest.main()

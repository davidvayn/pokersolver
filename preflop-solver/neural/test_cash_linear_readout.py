import unittest
import numpy as np
from cash_linear_readout import fit_linear_delta


class CashLinearReadoutTests(unittest.TestCase):
    def test_exact_known_weighted_solution_and_weight_scaling(self):
        x=np.asarray([[1.,0.],[0.,1.],[1.,1.]])
        target=x@np.asarray([2.,-3.]); w=np.asarray([1.,2.,4.])
        delta,report=fit_linear_delta(x,target,w)
        self.assertTrue(np.allclose(delta,[2.,-3.]))
        self.assertLess(report['training_mse_after'],1e-20)
        scaled,_=fit_linear_delta(x,target,w*100.)
        self.assertTrue(np.allclose(delta,scaled))

    def test_nullspace_minimum_delta_leaves_unidentified_directions_unchanged(self):
        x=np.asarray([[1.,1.,0.],[2.,2.,0.]])
        delta,report=fit_linear_delta(x,np.asarray([2.,4.]),np.ones(2))
        self.assertTrue(np.allclose(delta,[1.,1.,0.]))
        self.assertEqual(report['effective_rank'],1)

    def test_cutoff_discards_ill_conditioned_directions(self):
        x=np.diag([1.,1e-9])
        delta,report=fit_linear_delta(x,np.asarray([1.,1.]),np.ones(2))
        self.assertTrue(np.allclose(delta,[1.,0.]))
        self.assertEqual(report['effective_rank'],1)
        self.assertLessEqual(report['effective_condition_number'],1e6)

    def test_zero_weight_row_cannot_move_the_fit(self):
        x=np.asarray([[1.],[1.]])
        delta,report=fit_linear_delta(x,np.asarray([2.,1000.]),np.asarray([1.,0.]))
        self.assertAlmostEqual(delta[0],2.)
        self.assertLess(report['training_mse_after'],1e-20)

    def test_invalid_axes_mass_dtype_and_cutoff_fail_closed(self):
        x=np.ones((2,1)); y=np.ones(2); w=np.ones(2)
        for invalid in (0.,-1.,1.,True,float('inf'),float('nan')):
            with self.assertRaises(ValueError): fit_linear_delta(x,y,w,rcond=invalid)
        for bad_x,bad_y,bad_w in ((x,np.ones(3),w),(x,y,-w),(x,y,w*0),
                (x.astype(int),y,w),(x,y,np.asarray([np.nan,1.])),
                (np.ones((2,34)),y,w)):
            with self.assertRaises(ValueError): fit_linear_delta(bad_x,bad_y,bad_w)


if __name__=='__main__': unittest.main()


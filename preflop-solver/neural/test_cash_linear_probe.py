"""Deterministic globally weighted frozen-feature probes, not poker labels."""
import unittest
import numpy as np

from cash_linear_probe import FrozenLinearProbe, frozen_cash_interaction_features


class CashLinearProbeTests(unittest.TestCase):
    def test_cross_features_preserve_seat_order_and_frozen_parent_inputs(self):
        combined=np.zeros((1,2,1326,256),np.float32)
        combined[:,:,:,64:128]=2.
        combined[:,:,:,128:192]=3.
        combined[:,0,:,192:]=4.
        combined[:,1,:,192:]=5.
        before=combined.copy(); crossed=frozen_cash_interaction_features(combined)
        self.assertEqual(crossed.shape,(1,2,1326,384))
        np.testing.assert_array_equal(crossed[:,:,:,:256],combined)
        np.testing.assert_array_equal(crossed[:,0,:,256:320],np.full((1,1326,64),12.))
        np.testing.assert_array_equal(crossed[:,1,:,256:320],np.full((1,1326,64),15.))
        np.testing.assert_array_equal(crossed[:,0,:,320:],np.full((1,1326,64),8.))
        np.testing.assert_array_equal(crossed[:,1,:,320:],np.full((1,1326,64),10.))
        crossed[:,:,:,:256]=0
        np.testing.assert_array_equal(combined,before)
        for bad in (combined[:,:,:,:255],combined[:,:1],combined*np.nan,
                    np.zeros((9,2,1326,256),np.float32),np.full_like(combined,1e30)):
            with self.assertRaises(ValueError): frozen_cash_interaction_features(bad)

    def fixture(self):
        x=np.array([[1.,0.,1.],[0.,1.,1.],[2.,1.,1.],[-1.,2.,1.]])
        target=np.array([1.,-2.,.4,3.])
        weight=np.array([.01,12.,.2,3.])
        return x,target,weight

    def test_streamed_batches_use_one_global_weight_mass(self):
        x,target,weight=self.fixture()
        whole=FrozenLinearProbe(3); whole.add(x,target,weight)
        split=FrozenLinearProbe(3)
        split.add(x[:2],target[:2],weight[:2]); split.add(x[2:],target[2:],weight[2:])
        np.testing.assert_allclose(split.solve(.01)['coefficients'],whole.solve(.01)['coefficients'],atol=1e-12)
        self.assertEqual(split.rows,4)
        self.assertAlmostEqual(split.mass,weight.sum())
        # Explicit independent weighted least-squares reference.
        expected=np.linalg.lstsq(x*np.sqrt(weight[:,None]),target*np.sqrt(weight),rcond=1e-10)[0]
        np.testing.assert_allclose(split.solve(0)['coefficients'],expected,atol=1e-10)

    def test_ridge_anchors_a_zero_displacement_and_reports_actual_objective(self):
        x,target,weight=self.fixture(); probe=FrozenLinearProbe(3); probe.add(x,target,weight)
        result=probe.solve(.2); delta=result['coefficients']
        mse=float(np.sum(weight*(x@delta-target)**2)/weight.sum())
        scale=np.sqrt(np.sum(weight[:,None]*x*x,axis=0)/weight.sum())
        self.assertAlmostEqual(result['data_mse'],mse,places=11)
        self.assertAlmostEqual(result['regularized_objective'],mse+.2*np.sum((scale*delta)**2),places=11)
        self.assertLessEqual(result['regularized_objective'],result['initial_mse']+1e-10)

    def test_global_mass_and_feature_scale_changes_do_not_change_predictions(self):
        x,target,weight=self.fixture(); control=FrozenLinearProbe(3); control.add(x,target,weight)
        expected=x@control.solve(.1)['coefficients']
        for scale in (1e-10,1e10):
            candidate=FrozenLinearProbe(3); candidate.add(x,target,weight*scale)
            np.testing.assert_allclose(x@candidate.solve(.1)['coefficients'],expected,atol=1e-10)
        factors=np.array([1e-5,1e5,3.])
        candidate=FrozenLinearProbe(3); candidate.add(x*factors,target,weight)
        np.testing.assert_allclose((x*factors)@candidate.solve(.1)['coefficients'],expected,atol=1e-10)

    def test_zero_weight_and_zero_feature_columns_do_not_invent_information(self):
        x=np.array([[1.,0.],[2.,0.],[3.,0.]])
        probe=FrozenLinearProbe(2); probe.add(x,np.array([2.,4.,1e9]),np.array([1.,1.,0.]))
        result=probe.solve(0)
        np.testing.assert_allclose(result['coefficients'],[2.,0.],atol=1e-10)
        self.assertAlmostEqual(result['data_mse'],0.,places=10)

    def test_rejected_add_is_atomic_and_does_not_mutate_caller_arrays(self):
        x,target,weight=self.fixture(); original=[a.copy() for a in (x,target,weight)]
        probe=FrozenLinearProbe(3); probe.add(x,target,weight)
        before=(probe.mass,probe.rows,probe.solve(.1)['coefficients'].copy())
        for bad in [(x[:,:2],target,weight),(x,target[:2],weight),(x,target,-weight),
                    (x*np.nan,target,weight),(x,target,np.zeros(4)),(x,target,np.full(4,np.inf))]:
            with self.assertRaises(ValueError): probe.add(*bad)
        self.assertEqual((probe.mass,probe.rows),before[:2])
        np.testing.assert_array_equal(probe.solve(.1)['coefficients'],before[2])
        for old,current in zip(original,(x,target,weight)): np.testing.assert_array_equal(old,current)

    def test_invalid_dimensions_uninitialized_solves_and_ridge_fail_closed(self):
        for invalid in (0,-1,True,1.5,513):
            with self.assertRaises(ValueError): FrozenLinearProbe(invalid)
        probe=FrozenLinearProbe(2)
        with self.assertRaises(ValueError): probe.solve(0)
        probe.add(np.eye(2),np.ones(2),np.ones(2))
        for invalid in (-1,True,np.nan,np.inf,'0'):
            with self.assertRaises(ValueError): probe.solve(invalid)

    def test_overflow_and_nonreal_inputs_fail_without_partial_statistics(self):
        probe=FrozenLinearProbe(2)
        for x in (np.array([[1e308,1e308]]),np.array([[1j,0j]]),np.array([[True,False]])):
            with self.assertRaises(ValueError): probe.add(x,np.array([1.]),np.array([1.]))
            self.assertEqual((probe.mass,probe.rows),(0.,0))
            np.testing.assert_array_equal(probe.normal,np.zeros((2,2)))
            np.testing.assert_array_equal(probe.rhs,np.zeros(2))


if __name__=='__main__': unittest.main()

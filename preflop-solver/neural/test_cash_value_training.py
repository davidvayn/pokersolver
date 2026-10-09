import unittest
import copy
from pathlib import Path
import mlx.core as mx
import numpy as np
from train_cash_value_network import cash_accounting_penalty,context_loss_multipliers,run,split_cash_families


class CashValueTrainingTests(unittest.TestCase):
    def test_known_house_regularizer_does_not_zero_sum_shift_outputs(self):
        with mx.stream(mx.cpu):
            own = np.zeros((1,2,1326),dtype=np.float32); own[:,0] = 4.56; own[:,1] = -5.
            prediction = mx.array(own.reshape(1,-1) / 10.)
            joint = mx.ones((1,2,1326)); scales = mx.array([10.]); house = mx.array([.44])
            before = np.array(prediction)
            self.assertLess(float(cash_accounting_penalty(prediction,scales,joint,house).item()),1e-10)
            self.assertGreater(float(cash_accounting_penalty(prediction,scales,joint,mx.array([0.])).item()),.19)
            np.testing.assert_array_equal(np.array(prediction),before)
            shifted = prediction + .1
            penalty = cash_accounting_penalty(shifted,scales,joint,house)
            self.assertAlmostEqual(float(penalty.item()),4.,places=4)
            gradient = mx.grad(lambda x: cash_accounting_penalty(x,scales,joint,house))(shifted)
            self.assertTrue(np.all(np.array(gradient) > 0))
            self.assertAlmostEqual(float(np.sum(np.array(gradient))),80.,places=3)

    def test_invalid_regularization_budget_fails_before_dataset_io(self):
        for value in [-1.,float('nan'),float('inf'),101.]:
            with self.assertRaisesRegex(ValueError,'accounting loss weight'):
                run(Path('must-not-read'),Path('must-not-create'),1,2,None,value)

    def test_forced_leaf_curriculum_only_weights_captured_leaf_parents_without_changing_targets(self):
        source=dict(labels=[dict(index=i) for i in range(5)],source_datasets=dict(sources=[
            dict(selected_rows=[0,1,2],dataset=dict(labels=[{}, {}, {}])),
            dict(selected_rows=[0,1],dataset=dict(labels=[{}, {}],flop_leaf_provenance={}))]))
        before=copy.deepcopy(source)
        np.testing.assert_array_equal(context_loss_multipliers(source,.25),[1.,1.,1.,.25,.25])
        np.testing.assert_array_equal(context_loss_multipliers(source,1.),np.ones(5))
        self.assertEqual(source,before)
        with self.assertRaisesRegex(ValueError,'mixed frozen-flop'):
            context_loss_multipliers(dict(labels=[{}]),.25)
        for value in [0.,-1.,float('nan'),float('inf'),1.1]:
            with self.assertRaisesRegex(ValueError,'flop leaf loss weight'):
                run(Path('must-not-read'),Path('must-not-create'),1,2,flop_leaf_loss_weight=value)

    def test_shared_split_is_independent_of_network_seed_and_keeps_flop_families_together(self):
        source = {"game": {}, "labels": [{"input":{"state":{"board":board}}} for board in [
            [0,5,10,15], [0,5,10,19], [4,9,14,19], [8,13,18,23],
            [12,17,22,27], [16,21,26,31], [20,25,30,35], [24,29,34,39],
        ]]}
        first = split_cash_families(source,937)
        second = split_cash_families(source,937)
        for left,right in zip(first,second): np.testing.assert_array_equal(left,right)
        self.assertEqual(sorted(np.concatenate(first).tolist()),list(range(8)))
        groups = [set(rows.tolist()) for rows in first]
        self.assertTrue(any({0,1} <= rows for rows in groups))
        self.assertTrue(all(not (left & right) for i,left in enumerate(groups) for right in groups[i+1:]))

    def test_invalid_split_seed_fails_before_dataset_io(self):
        for seed in [-1,2**32,True,1.5]:
            with self.assertRaisesRegex(ValueError,'split seed'):
                run(Path('must-not-read'),Path('must-not-create'),1,2,split_seed=seed)


if __name__ == '__main__': unittest.main()

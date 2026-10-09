import unittest
import copy
import json
import tempfile
from pathlib import Path
import mlx.core as mx
import numpy as np
from train_cash_value_network import OwnComboValueNetwork,export_cash_model,cash_accounting_penalty,cash_profile_value_penalty,context_loss_multipliers,run,split_cash_families
from cash_profiles import profile_rules,rules_digest
from cash_value_dataset import NETWORK_SCHEMA,POOLED_NETWORK_SCHEMA,BLOCKER_POOLED_NETWORK_SCHEMA,BLOCKER_POOLED_CONTRACT


class CashValueTrainingTests(unittest.TestCase):
    def test_profile_supervision_detects_equal_and_opposite_errors_without_projection(self):
        with mx.stream(mx.cpu):
            own = np.zeros((1,2,1326),np.float32); own[:,0] = 4.56; own[:,1] = -5.
            shifted = own.copy(); shifted[:,0] += 1.; shifted[:,1] -= 1.
            scale = mx.array([2.]); reaches = mx.ones((1,2,1326)); targets = mx.array([[4.56,-5.]])
            correct = mx.array(own.reshape((1,2652))/2.)
            errors = mx.array(shifted.reshape((1,2652))/2.)
            self.assertAlmostEqual(float(cash_profile_value_penalty(correct,scale,reaches,targets).item()),0.,places=8)
            self.assertAlmostEqual(float(cash_accounting_penalty(errors,scale,reaches,mx.array([.44])).item()),0.,places=8)
            self.assertAlmostEqual(float(cash_profile_value_penalty(errors,scale,reaches,targets).item()),2.,places=5)
            np.testing.assert_array_equal(np.array(errors),shifted.reshape((1,2652))/2.)
        for invalid in (-1.,float('nan'),float('inf'),101.):
            with self.assertRaisesRegex(ValueError,'profile value loss weight'):
                run(Path('must-not-read'),Path('must-not-create'),7101,2,profile_value_loss_weight=invalid)

    def test_pooled_cash_export_has_versioned_layout_and_zero_correction_baseline(self):
        rules=profile_rules("nl25"); source=dict(game=dict(cash_rules=rules),rules_sha256=rules_digest(rules))
        with tempfile.TemporaryDirectory() as directory:
            for architecture, schema in [("compact",NETWORK_SCHEMA),("wide",NETWORK_SCHEMA),("wide-pooled",POOLED_NETWORK_SCHEMA),("wide-blocker-pooled",BLOCKER_POOLED_NETWORK_SCHEMA)]:
                model=OwnComboValueNetwork(architecture)
                self.assertTrue((np.array(model.head.layers[-1].weight) == 0).all())
                self.assertTrue((np.array(model.head.layers[-1].bias) == 0).all())
                path=Path(directory)/f"{architecture}.json"
                export_cash_model(model,path,7101,source,"a"*64)
                payload=json.loads(path.read_text())
                self.assertEqual(payload["schema"],schema)
                self.assertEqual(payload["projection"],"independent-full-stack-clip-and-board-mask-no-zero-sum")
                embeddings=payload["contextTower"][-1]["outputSize"]+payload["queryTower"][-1]["outputSize"]*(3 if architecture in ("wide-pooled","wide-blocker-pooled") else 1)
                self.assertEqual(payload["head"][0]["inputSize"],embeddings)
                if architecture == "wide-blocker-pooled":
                    self.assertEqual(payload["predictionContract"],BLOCKER_POOLED_CONTRACT)
        with self.assertRaisesRegex(ValueError,"architecture"):
            run(Path('must-not-read'),Path('must-not-create'),1,2,architecture="unknown")

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

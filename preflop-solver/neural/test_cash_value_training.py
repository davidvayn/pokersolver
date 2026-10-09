import unittest
import copy
import json
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from train_cash_value_network import OwnComboValueNetwork,export_cash_model,cash_accounting_penalty,cash_profile_value_penalty,cash_regression_losses,context_loss_multipliers,run,split_cash_families
from cash_profiles import profile_rules,rules_digest
from cash_value_dataset import NETWORK_SCHEMA,POOLED_NETWORK_SCHEMA,BLOCKER_POOLED_NETWORK_SCHEMA,BLOCKER_POOLED_CONTRACT
from train_public_value_network import FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT


class CashValueTrainingTests(unittest.TestCase):
    def test_huber_uses_net_bb_errors_and_retains_mse_metric_and_local_gradients(self):
        with mx.stream(mx.cpu):
            errors = mx.array([-2.,-.5,0.,.5,2.]); weights = mx.ones((5,))
            mse,objective = cash_regression_losses(errors,weights,"huber",1.)
            self.assertAlmostEqual(float(mse.item()),1.7,places=6)
            self.assertAlmostEqual(float(objective.item()),1.3,places=6)
            gradient = mx.grad(lambda values: cash_regression_losses(values,weights,"huber",1.)[1])(errors)
            np.testing.assert_allclose(np.array(gradient),[-.4,-.2,0.,.2,.4],atol=1e-7)
            small = mx.array([-.5,0.,.5])
            first,second = cash_regression_losses(small,mx.ones((3,)),"huber",1.)
            self.assertEqual(float(first.item()),float(second.item()))
            weighted_mse,weighted_huber = cash_regression_losses(mx.array([.5,2.]),mx.array([1.,3.]),"huber",1.)
            self.assertAlmostEqual(float(weighted_mse.item()),3.0625,places=6)
            self.assertAlmostEqual(float(weighted_huber.item()),2.3125,places=6)
            baseline,baseline_objective = cash_regression_losses(errors,weights)
            self.assertEqual(float(baseline.item()),float(mse.item()))
            self.assertEqual(float(baseline_objective.item()),float(mse.item()))

    def test_invalid_regression_options_fail_before_dataset_io(self):
        for invalid in (None,True,1,"unknown",["mse"]):
            with self.assertRaisesRegex(ValueError,"regression loss"):
                run(Path("must-not-read"),Path("must-not-create"),7101,2,regression_loss=invalid)
        for invalid in (None,True,0.,-.1,.009,20.1,float("nan"),float("inf"),"1"):
            with self.assertRaisesRegex(ValueError,"Huber threshold"):
                run(Path("must-not-read"),Path("must-not-create"),7101,2,huber_delta_bb=invalid)

    def test_frozen_cash_weights_record_loss_and_unchanged_checkpoint_criterion(self):
        rules = profile_rules("nl25"); source = dict(game=dict(cash_rules=rules),rules_sha256=rules_digest(rules))
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("mse","huber"):
                path = Path(directory)/f"{mode}.json"
                export_cash_model(OwnComboValueNetwork("wide-pooled"),path,7101,source,"a"*64,
                                  regression_loss=mode,huber_delta_bb=1.)
                payload = json.loads(path.read_text())
                self.assertEqual(payload["regressionLoss"],mode)
                self.assertEqual(payload["huberDeltaBb"],1.)
                self.assertEqual(payload["checkpointSelectionCriterion"],"tuning-own-payoff-mse-plus-explicit-auxiliary-penalties")
                self.assertEqual(payload["regressionLossNormalization"],
                    "twice-standard-huber-local-mse-match" if mode=="huber" else "squared-net-bb-error")

    def test_adam_moment_correction_is_an_explicit_matched_pilot_option(self):
        with mx.stream(mx.cpu):
            gradient = {"x":mx.array([1.])}; initial = {"x":mx.array([0.])}
            corrected = optim.Adam(learning_rate=.001,bias_correction=True).apply_gradients(gradient,initial)
            uncorrected = optim.Adam(learning_rate=.001,bias_correction=False).apply_gradients(gradient,initial)
            self.assertAlmostEqual(float(corrected["x"].item()),-.001,places=7)
            self.assertAlmostEqual(float(uncorrected["x"].item()),-.001 * .1 / np.sqrt(.001),places=7)
        for invalid in (1,0,None,"yes"):
            with self.assertRaisesRegex(ValueError,"Adam bias correction"):
                run(Path("must-not-read"),Path("must-not-create"),7101,2,adam_bias_correction=invalid)
        for invalid in (-1.,0.,True,float("nan"),float("inf"),.101,"yes"):
            with self.assertRaisesRegex(ValueError,"learning rate"):
                run(Path("must-not-read"),Path("must-not-create"),7101,2,learning_rate=invalid)

    def test_baseline_conditioned_head_retains_original_inputs_and_initial_parameters(self):
        mx.random.seed(7171); old = OwnComboValueNetwork("wide-pooled")
        mx.random.seed(7171); added = OwnComboValueNetwork("wide-baseline-conditioned")
        for left,right in ((old.context_tower,added.context_tower),(old.query_tower,added.query_tower)):
            for a,b in zip(left.layers,right.layers):
                if hasattr(a,"weight"):
                    np.testing.assert_array_equal(np.array(a.weight),np.array(b.weight))
                    np.testing.assert_array_equal(np.array(a.bias),np.array(b.bias))
        np.testing.assert_array_equal(np.array(old.head.layers[0].weight),np.array(added.head.layers[0].weight)[:,:-1])
        self.assertTrue((np.array(added.head.layers[0].weight)[:,-1] == 0).all())
        rules=profile_rules("nl25"); source=dict(game=dict(cash_rules=rules),rules_sha256=rules_digest(rules))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"network.json"; export_cash_model(added,path,7171,source,"a"*64)
            payload=json.loads(path.read_text())
            self.assertEqual(payload["schema"],"hu-cash-public-belief-combo-value-network-v5")
            self.assertEqual(payload["predictionContract"],"cash-turn-start-cfv-full-stack-baseline-conditioned-v1")
            self.assertEqual(payload["head"][0]["inputSize"],old.head.layers[0].weight.shape[-1]+1)

    def test_exact_runout_features_are_explicit_in_weights_and_reject_unknown_inputs(self):
        rules=profile_rules("nl25"); source=dict(game=dict(cash_rules=rules),rules_sha256=rules_digest(rules))
        with tempfile.TemporaryDirectory() as directory:
            for features in (FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT):
                model = OwnComboValueNetwork("wide-pooled", features)
                path = Path(directory) / f"{features}.json"
                export_cash_model(model, path, 7101, source, "a"*64)
                self.assertEqual(json.loads(path.read_text())["featureSchema"], features)
        with self.assertRaisesRegex(ValueError, "feature schema"):
            run(Path("must-not-read"), Path("must-not-create"), 7101, 2, feature_schema="unknown")

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
                self.assertIs(payload["adamBiasCorrection"],False)
                self.assertEqual(payload["learningRate"],.001)
                self.assertEqual(payload["projection"],"independent-full-stack-clip-and-board-mask-no-zero-sum")
                embeddings=payload["contextTower"][-1]["outputSize"]+payload["queryTower"][-1]["outputSize"]*(3 if architecture in ("wide-pooled","wide-blocker-pooled") else 1)
                self.assertEqual(payload["head"][0]["inputSize"],embeddings)
                if architecture == "wide-blocker-pooled":
                    self.assertEqual(payload["predictionContract"],BLOCKER_POOLED_CONTRACT)
            explicit=Path(directory)/"explicit-optimizer.json"
            export_cash_model(OwnComboValueNetwork("wide-pooled"),explicit,7101,source,"a"*64,
                              adam_bias_correction=True,learning_rate=.003445078064)
            payload=json.loads(explicit.read_text())
            self.assertIs(payload["adamBiasCorrection"],True)
            self.assertEqual(payload["learningRate"],.003445078064)
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

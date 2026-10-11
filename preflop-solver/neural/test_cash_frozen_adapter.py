import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from cash_frozen_adapter import (FrozenTailAdapterHead, frozen_adapter_gate,
    make_frozen_adapter_payload, validate_frozen_adapter)
from cash_value_dataset import KERNEL_POOLED_NETWORK_SCHEMA, KERNEL_POOLED_CONTRACT
import test_cash_value_initialization as fixtures
import test_cash_kernel_network as kernel_fixtures
from train_cash_value_network import OwnComboValueNetwork, export_cash_model
from train_public_value_network import tower_payload
import validate_cash_value_parity as parity


class CashFrozenAdapterTests(unittest.TestCase):
    def fixture(self, directory):
        _, network, _, source, _ = fixtures.CashValueInitializationTests().fixture(directory)
        model = OwnComboValueNetwork('wide-kernel-pooled')
        export_cash_model(model, network, 7101, source, 'a' * 64)
        parent = json.loads(network.read_bytes())
        head = FrozenTailAdapterHead()
        return model, source, parent, head

    def test_zero_cold_head_and_export_preserve_every_parent_tensor(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            _, _, parent, head = self.fixture(Path(temp))
            before = copy.deepcopy(parent)
            payload = make_frozen_adapter_payload(parent, 'b' * 64, 'c' * 64, tower_payload(head.head, 'relu', 'linear'))
            self.assertEqual(parent, before)
            for name in ('contextTower', 'queryTower', 'head', 'kernelQuery', 'kernelKey'):
                self.assertEqual(payload[name], parent[name])
            self.assertEqual(parity.cash_architecture(payload), 'wide-kernel-pooled')
            np.testing.assert_array_equal(np.array(head(mx.ones((1, 2, 1326, 256)))), np.zeros((1, 2, 1326)))

    def test_gate_uses_cent_units_at_both_sides_of_boundary(self):
        source = dict(game=dict(effective_stack_bb=20., cash_rules=dict(unitsPerBb=25)),
            labels=[dict(input=dict(state=dict(invested_bb=[amount, amount]))) for amount in (17.96, 18., 18.04)])
        np.testing.assert_array_equal(frozen_adapter_gate(source), [False, True, True])
        for amounts in ([18.01, 18.01], [18., 17.96], [20., 20.]):
            source['labels'] = [dict(input=dict(state=dict(invested_bb=amounts)))]
            with self.assertRaises(ValueError): frozen_adapter_gate(source)

    def test_net_bb_correction_is_before_clip_and_outside_predictions_are_exact(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            model, source, parent, head = self.fixture(Path(temp))
            arrays = kernel_fixtures.CashKernelNetworkTests().arrays(model)
            arrays = tuple(np.repeat(value, 3, axis=0) if value is not None else None for value in arrays)
            source['labels'] = [dict(input=dict(state=dict(invested_bb=[amount, amount], ranges=arrays[2][index].tolist())))
                                for index, amount in enumerate((17.96, 18., 18.04))]
            head.head.layers[-1].bias = mx.array([.5])
            payload = make_frozen_adapter_payload(parent, 'b' * 64, 'c' * 64, tower_payload(head.head, 'relu', 'linear'))
            with patch.object(parity, 'feature_arrays', return_value=arrays):
                original = parity.python_predictions(source, parent)
                candidate = parity.python_predictions(source, payload)
                cpu = parity.mlx_predictions(source, payload, mx.cpu)
                original_cpu = parity.mlx_predictions(source, parent, mx.cpu)
            np.testing.assert_array_equal(candidate[0], original[0])
            np.testing.assert_array_equal(cpu[0], original_cpu[0])
            np.testing.assert_allclose(candidate[1:, :, 1:], original[1:, :, 1:] + .5, atol=1e-7)
            np.testing.assert_array_equal(candidate[:, :, 0], np.zeros((3, 2)))
            np.testing.assert_allclose(cpu, candidate, atol=5e-6, rtol=0)
            head.head.layers[-1].bias = mx.array([100.])
            payload = make_frozen_adapter_payload(parent, 'b' * 64, 'c' * 64, tower_payload(head.head, 'relu', 'linear'))
            with patch.object(parity, 'feature_arrays', return_value=arrays):
                clipped = parity.python_predictions(source, payload)
            np.testing.assert_array_equal(clipped[1:, :, 1:], np.full((2, 2, 1325), 20.))

    def test_smooth_tail_operator_has_zero_endpoint_and_cent_aligned_fade(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            model, source, parent, head = self.fixture(Path(temp))
            amounts = (17.96, 18., 18.04, 18.48, 18.52, 19., 19.04)
            arrays = kernel_fixtures.CashKernelNetworkTests().arrays(model)
            arrays = tuple(np.repeat(value, len(amounts), axis=0) if value is not None else None for value in arrays)
            source['labels'] = [dict(input=dict(state=dict(invested_bb=[amount, amount], ranges=arrays[2][index].tolist())))
                                for index, amount in enumerate(amounts)]
            head.head.layers[-1].bias = mx.array([.5])
            payload = make_frozen_adapter_payload(parent, 'b'*64, 'c'*64, tower_payload(head.head,'relu','linear'))
            payload.update(schema='hu-cash-public-belief-combo-value-network-v9',
                architecture='wide-kernel-pooled-smooth-tail-adapter',
                predictionContract='cash-turn-start-cfv-full-stack-smooth-tail-adapter-v1')
            payload['frozenTailAdapter']['fadeWidthUnits'] = 25
            with patch.object(parity, 'feature_arrays', return_value=arrays):
                original = parity.python_predictions(source,parent)
                predicted = parity.python_predictions(source,payload)
                cpu = parity.mlx_predictions(source,payload,mx.cpu)
                original_cpu = parity.mlx_predictions(source,parent,mx.cpu)
            np.testing.assert_array_equal(predicted[:2],original[:2])
            np.testing.assert_array_equal(cpu[:2],original_cpu[:2])
            coefficients = np.asarray([0.,0.,.004672,.470016,.529984,1.,1.])
            np.testing.assert_allclose(predicted[:,:,1:]-original[:,:,1:],
                np.broadcast_to(.5*coefficients[:,None,None],(7,2,1325)),atol=1e-7,rtol=0)
            np.testing.assert_allclose(cpu,predicted,atol=5e-6,rtol=0)
            np.testing.assert_array_equal(predicted[:,:,0],np.zeros((7,2)))
            for width in (None,True,24,26):
                bad = copy.deepcopy(payload)
                if width is None: del bad['frozenTailAdapter']['fadeWidthUnits']
                else: bad['frozenTailAdapter']['fadeWidthUnits'] = width
                with self.assertRaises(ValueError): parity.cash_architecture(bad)
            bad = copy.deepcopy(payload)
            bad.update(schema='hu-cash-public-belief-combo-value-network-v8',
                architecture='wide-kernel-pooled-frozen-tail-adapter',
                predictionContract='cash-turn-start-cfv-full-stack-frozen-tail-adapter-v1')
            with self.assertRaises(ValueError): parity.cash_architecture(bad)

    def test_bad_metadata_tensors_and_silent_downgrades_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            _, _, parent, head = self.fixture(Path(temp))
            original = make_frozen_adapter_payload(parent, 'b' * 64, 'c' * 64, tower_payload(head.head, 'relu', 'linear'))
            changes = (lambda p:p['frozenTailAdapter'].update(maximumRemainingUnits=51),
                lambda p:p['frozenTailAdapter'].update(maximumRemainingUnits=True),
                lambda p:p['frozenTailAdapter'].update(correctionUnit='normalized_state_value_scale'),
                lambda p:p['frozenTailAdapter'].update(parentNetworkSha256='B' * 64),
                lambda p:p['frozenTailAdapter'].update(parentDatasetSha256='unknown'),
                lambda p:p['frozenTailAdapter'].update(head=[]),
                lambda p:p['frozenTailAdapter']['head'][0].update(weights=[0.]),
                lambda p:p['frozenTailAdapter']['head'][0].update(normalization='layer'),
                lambda p:p['frozenTailAdapter']['head'][1].update(biases=[float('nan')]),
                lambda p:p['frozenTailAdapter']['head'][1].update(biases=[True]),
                lambda p:p.update(schema=KERNEL_POOLED_NETWORK_SCHEMA, architecture='wide-kernel-pooled', predictionContract=KERNEL_POOLED_CONTRACT),
                lambda p:p.update(projection='zero-sum'))
            changes += (lambda p:p.update(sourceGame=None), lambda p:p['sourceGame'].update(cash_rules=None))
            for change in changes:
                bad = copy.deepcopy(original); change(bad)
                with self.assertRaises(ValueError): parity.cash_architecture(bad)
            missing = copy.deepcopy(original); del missing['frozenTailAdapter']
            with self.assertRaises(ValueError): validate_frozen_adapter(missing)


if __name__ == '__main__': unittest.main()

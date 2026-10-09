import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from unittest.mock import patch

from cash_value_dataset import build_dataset
from cash_value_initialization import initialize_cash_from_frozen
from native_value_dataset import compatible_masses, legal_combos
from test_cash_value_dataset import label_fixture
from train_cash_value_network import OwnComboValueNetwork, export_cash_model, split_cash_families, run


class CashValueInitializationTests(unittest.TestCase):
    def fixture(self, directory):
        paths = []
        for index, board in enumerate(([8,13,22,31],[0,5,10,15],[4,9,18,27],[12,17,26,35])):
            label = label_fixture(); legal = legal_combos(board)
            ranges = np.tile(legal/legal.sum(), (2,1))
            label["input"]["state"].update(board=board, ranges=ranges.tolist())
            label["counterfactual_values_bb"] = np.tile(legal*-.86, (2,1)).tolist()
            label["opponent_compatible_mass"] = compatible_masses(ranges).tolist()
            path = directory/f"label-{index}.json"; path.write_text(json.dumps(label)); paths.append(path)
        source = build_dataset(paths)
        dataset = directory/"dataset.json"; dataset.write_text(json.dumps(source))
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        mx.random.seed(7101); model = OwnComboValueNetwork("wide-pooled")
        model.head.layers[-1].bias = mx.array([-.12])
        network = directory/"network.json"
        export_cash_model(model, network, 7101, source, digest)
        splits = split_cash_families(source,937)
        report = dict(schema="hu-cash-value-pilot-report-v1", status="research_only", active=False,
                      seed=7101, split_seed=937, steps=240, selected_step=230,
                      source_dataset_sha256=digest,
                      network_sha256=hashlib.sha256(network.read_bytes()).hexdigest(),
                      split_rows={key:rows.tolist() for key,rows in zip(("train","tuning","holdout"),splits)})
        (directory/"report.json").write_text(json.dumps(report))
        return model, network, dataset, source, splits

    def test_frozen_initialization_preserves_all_parameters_and_nonzero_residual(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            original, network, dataset, source, splits = self.fixture(Path(temp))
            mx.random.seed(17); target = OwnComboValueNetwork("wide-pooled")
            before = np.array(target.context_tower.layers[0].weight)
            metadata = initialize_cash_from_frozen(target, network, dataset, source, splits, 7101, 937)
            self.assertFalse(np.array_equal(before,np.array(target.context_tower.layers[0].weight)))
            for (name, expected),(actual_name,actual) in zip(tree_flatten(original.parameters()),tree_flatten(target.parameters())):
                self.assertEqual(name,actual_name)
                np.testing.assert_array_equal(np.array(expected),np.array(actual))
            self.assertAlmostEqual(float(target.head.layers[-1].bias.item()),-.12,places=7)
            self.assertEqual(metadata["network_sha256"],hashlib.sha256(network.read_bytes()).hexdigest())
            self.assertEqual(metadata["optimizer_initialization"],"fresh-adam-no-resumed-moments")
            self.assertEqual(metadata["source_selected_step"],230)

    def test_bad_contract_or_tensors_fail_before_any_parameter_is_loaded(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            _, network, dataset, source, splits = self.fixture(Path(temp))
            original = json.loads(network.read_text())
            changes = [lambda p:p.update(schema="hu-public-belief-combo-value-network-v5"),
                       lambda p:p.update(architecture="compact"),
                       lambda p:p.update(featureSchema="rank-suit-invariant-combo-query-v3"),
                       lambda p:p.update(targetScaleBb=40),
                       lambda p:p.update(seed=7102),
                       lambda p:p.update(rulesSha256="b"*64),
                       lambda p:p.update(projection="zero-sum"),
                       lambda p:p["head"][-1].update(activation="relu"),
                       lambda p:p["head"][-1].update(biases=[float("nan")]),
                       lambda p:p["queryTower"][0].update(weights=[0.]),
                       lambda p:p["head"][-1].update(biases=["0.1"])]
            for change in changes:
                payload = copy.deepcopy(original); change(payload)
                network.write_text(json.dumps(payload))
                report_path = Path(temp)/"report.json"; report = json.loads(report_path.read_text())
                report["network_sha256"] = hashlib.sha256(network.read_bytes()).hexdigest()
                report_path.write_text(json.dumps(report))
                target = OwnComboValueNetwork("wide-pooled")
                before = [(name,np.array(value)) for name,value in tree_flatten(target.parameters())]
                with self.assertRaises(ValueError):
                    initialize_cash_from_frozen(target,network,dataset,source,splits,7101,937)
                for (name,expected),(actual_name,actual) in zip(before,tree_flatten(target.parameters())):
                    self.assertEqual(name,actual_name)
                    np.testing.assert_array_equal(expected,np.array(actual))

    def test_initial_training_provenance_and_holdout_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, network, dataset, source, splits = self.fixture(Path(temp))
            report_path = Path(temp)/"report.json"; original = json.loads(report_path.read_text())
            for change in (lambda r:r.update(network_sha256="b"*64),
                           lambda r:r.update(source_dataset_sha256="b"*64),
                           lambda r:r.update(split_seed=938),
                           lambda r:r.update(active=True),
                           lambda r:r["split_rows"].update(train=[0,1,3],holdout=[])):
                report = copy.deepcopy(original); change(report); report_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    initialize_cash_from_frozen(OwnComboValueNetwork("wide-pooled"),network,dataset,source,splits,7101,937)
            report_path.write_text(json.dumps(original))
            altered = copy.deepcopy(source); altered["labels"] = altered["labels"][::-1]
            with self.assertRaises(ValueError):
                initialize_cash_from_frozen(OwnComboValueNetwork("wide-pooled"),network,dataset,altered,splits,7101,937)

    def test_warm_fit_can_keep_step_zero_and_exports_explicit_lineage(self):
        with tempfile.TemporaryDirectory() as temp, mx.stream(mx.cpu):
            _, network, dataset, source, _ = self.fixture(Path(temp))
            payload = json.loads(network.read_text()); count = len(source["labels"])
            arrays = (np.zeros((count,2,payload["contextSize"]),np.float32),
                      np.zeros((count,2,1326,payload["querySize"]),np.float32),
                      np.ones((count,2,1326),np.float32),np.ones(count,np.float32),
                      np.zeros((count,2,1326),np.float32),np.ones((count,1326),np.float32),
                      np.full((count,2652),-.12,np.float32),np.ones((count,2652),np.float32))
            output = Path(temp)/"fit"
            with patch("train_cash_value_network.feature_arrays",return_value=arrays):
                report = run(dataset,output,7101,1,split_seed=937,architecture="wide-pooled",
                             initial_value_network=network,initial_dataset=dataset)
            self.assertEqual(report["selected_step"],0)
            self.assertEqual(report["initial_value_network"]["network_sha256"],hashlib.sha256(network.read_bytes()).hexdigest())
            exported = json.loads((output/"value-network.json").read_text())
            self.assertTrue(exported["checkpointSelectionIncludesInitialModel"])
            self.assertEqual(exported["residualInitialization"],"frozen-cash-weights-with-fresh-adam")
            self.assertEqual(exported["initialModelSeed"],7101)
            self.assertFalse(report["active"])

    def test_initial_artifact_options_fail_before_dataset_io_when_incomplete(self):
        for options in (dict(initial_value_network=Path("missing")),dict(initial_dataset=Path("missing")),
                        dict(initial_value_network="missing",initial_dataset=Path("missing"))):
            with self.assertRaisesRegex(ValueError,"both network and dataset paths"):
                run(Path("must-not-read"),Path("must-not-create"),7101,1,**options)

    def test_warm_fit_cannot_overwrite_its_frozen_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"frozen"
            # Resolving aliases is required: ../frozen is the same directory.
            for directory in (output,output/".."/"frozen"):
                with self.assertRaisesRegex(ValueError,"frozen parent directory"):
                    run(Path("must-not-read"),directory,7101,1,
                        initial_value_network=output/"value-network.json",
                        initial_dataset=Path("must-not-read"))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

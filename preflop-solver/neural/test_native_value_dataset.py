import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

import numpy as np

import native_value_dataset as native
import run_native_value_preflight as runner
import train_public_value_network as training
import run_native_value_pilot as pilot
import validate_public_value_parity as parity
from run_native_family_extension import choose_new_families
from run_native_ace_coverage import select_ace_families
from run_native_training_refresh import capture_training_flags


def corpus(board=(0, 5, 10, 15), zero_player=False):
    legal = native.legal_combos(board)
    ranges = np.tile(legal / legal.sum(), (2, 1))
    absent = int(np.flatnonzero(legal)[-1])
    ranges[0, absent] = 0
    if zero_player:
        ranges[0] = 0
    else:
        ranges[0] /= ranges[0].sum()
    totals = np.array([0 if zero_player else 0.125, 0.25])
    raw_ranges = ranges * totals[:, None]
    masses = native.compatible_masses(ranges)
    best = np.zeros_like(ranges)
    best[0, absent] = 2 * masses[0, absent] * totals[1]
    profile = np.zeros_like(ranges)
    raw = np.where(ranges == 0, best, profile)
    raw_mass = masses * totals[::-1, None]
    conditional = np.divide(raw, raw_mass, out=np.zeros_like(raw), where=raw_mass > 0)
    target = dict(iteration=2, board=list(board), actor=1, invested_bb=[1.0, 1.0],
                  public_state=dict(board=list(board), actor=1, invested_bb=[1.0, 1.0],
                                    ranges=raw_ranges.tolist(), public_history=["test-only"], trajectory=[{"test": True}]),
                  ranges=ranges.tolist(), raw_reach_totals=totals.tolist(),
                  opponent_compatible_mass=masses.tolist(), counterfactual_values_bb=conditional.tolist(),
                  raw_counterfactual_bb=raw.tolist(), raw_profile_counterfactual_bb=profile.tolist(),
                  raw_best_response_counterfactual_bb=best.tolist(),
                  completed_zero_own_reach=((ranges == 0) & legal).sum(axis=1).tolist(),
                  conditional_response_gain_bb=None if zero_player else [0.0, 0.0],
                  policy_sha256="c" * 64, policy_rows=2, turn_iterations=64,
                  value_semantics=native.SEMANTICS)
    return dict(schema=native.SCHEMA, game=dict(effective_stack_bb=20.0),
                validation=dict(status="research_only"), source_public_input_sha256="a" * 64,
                source_policy_sha256="b" * 64, seed=100101, flop_iterations=2,
                turn_iterations=64, maximum_states=1, observed_queries=18, targets=[target])


class NativeValueDatasetTests(unittest.TestCase):
    def test_targeted_ace_roots_exclude_heldouts_suit_variants_and_duplicates(self):
        roots=[{"public":{"board":board}} for board in
               ([40,20,8],[48,20,8],[49,21,9],[48,24,12],[49,25,13],[48,28,16])]
        self.assertEqual(select_ace_families(roots,[[51,23,11]],2),[3,5])
        with self.assertRaises(ValueError): select_ace_families(roots,[[51,23,11]],3)

    def test_native_training_cli_preserves_reference_tuning_and_holdout(self):
        boards = [[0,5,10,15], [8,12,16,20], [32,36,40,44], [16,17,18,23]]
        reference = corpus()
        reference["targets"] = [corpus(board)["targets"][0] for board in boards]
        reference["maximum_states"] = reference["observed_queries"] = 4
        extended = copy.deepcopy(reference)
        extended["targets"].append(corpus([40,45,50,3])["targets"][0])
        extended["maximum_states"] = extended["observed_queries"] = 5
        expected = native.family_split(extended, 10601, .25, .25, reference=reference)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            reference_path, dataset_path = directory / "reference.json", directory / "extended.json"
            reference_path.write_text(json.dumps(reference))
            dataset_path.write_text(json.dumps(extended))
            digest = runner.sha256(reference_path)
            args = ["train_public_value_network.py", "--dataset", str(dataset_path),
                    "--native-split-reference", str(reference_path), "--native-split-reference-sha256", digest,
                    "--output-dir", str(directory / "models"), "--steps", "1", "--batch-size", "2",
                    "--seeds", "10601,10602", "--split-seed", "10601", "--validation-fraction", ".25",
                    "--tuning-fraction", ".25", "--variant-set", "range-only", "--evaluation-interval", "1"]
            with mock.patch.object(sys, "argv", args), mock.patch("builtins.print"):
                training.main()
            report = json.loads((directory / "models" / "turn-value-paired-report.json").read_text())
            self.assertEqual(report["nativeSplitReferenceSha256"], digest)
            self.assertNotEqual(report["validation"]["status"], "accepted")
            for name, indices in zip(("trainStates", "tuningStates", "validationStates"), expected):
                self.assertEqual(report[name], indices.tolist())
            # The refresh path changes only a training observation. The real CLI
            # must preserve held-out indices and record the opt-in contract.
            index = int(expected[0][0])
            extended["targets"][index]["iteration"] = 1
            dataset_path.write_text(json.dumps(extended))
            refresh_args = args + ["--native-training-refresh", "--architecture", "wide-pooled"]
            refresh_args[refresh_args.index("--output-dir")+1] = str(directory/"refreshed-models")
            with mock.patch.object(sys,"argv",refresh_args), mock.patch("builtins.print"):
                training.main()
            report = json.loads((directory/"refreshed-models"/"turn-value-paired-report.json").read_text())
            self.assertTrue(report["nativeTrainingRefresh"])
            self.assertEqual(report["networkSchema"],training.POOLED_NETWORK_SCHEMA)
            for name, indices in zip(("trainStates","tuningStates","validationStates"),expected):
                self.assertEqual(report[name],indices.tolist())

    def test_refresh_rejects_captures_crossing_heldout_boundary(self):
        source = dict(source_captures=[dict(states=2),dict(states=2)],
                      targets=[dict(source_capture_index=i//2) for i in range(4)])
        self.assertEqual(capture_training_flags(source,[0,1]),[True,False])
        with self.assertRaisesRegex(ValueError,"crosses"):
            capture_training_flags(source,[0,2])
        source["targets"][0]["source_capture_index"] = 1
        with self.assertRaisesRegex(ValueError,"order"):
            capture_training_flags(source,[0,1])

    def test_authentic_extension_requires_prefix_parity_and_skips_suit_duplicates(self):
        root = lambda board: {"public": {"board": board}}
        reference = [root([0,5,10]), root([8,12,16])]
        roots = reference + [root([1,4,11]), root([40,45,50]), root([16,17,18])]
        self.assertEqual(choose_new_families(roots, reference, 2), [3,4])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            choose_new_families(roots, reference, 3)
        changed = copy.deepcopy(roots)
        changed[0]["public"]["board"] = [1,2,3]
        with self.assertRaisesRegex(ValueError, "prefix"):
            choose_new_families(changed, reference, 2)

    def test_reference_split_preserves_old_targets_and_adds_only_training_families(self):
        boards = [[0,5,10,15], [8,12,16,20], [32,36,40,44], [16,17,18,23]]
        reference = dict(game={"test": True}, targets=[dict(board=b, value=i) for i,b in enumerate(boards)])
        extended = copy.deepcopy(reference)
        extended["targets"].append(dict(board=[40,45,50,3], value=5))
        baseline = native.family_split(reference, 10601, .25, .25)
        split = native.family_split(extended, 10601, .25, .25, reference=reference)
        np.testing.assert_array_equal(split[0], np.concatenate((baseline[0], [4])))
        for old, new in zip(baseline[1:], split[1:]):
            np.testing.assert_array_equal(old, new)
        extended["targets"][0]["value"] = -1
        with self.assertRaisesRegex(ValueError, "prefix"):
            native.family_split(extended, 10601, .25, .25, reference=reference)

    def test_reference_split_rejects_heldout_family_leakage_and_changed_game(self):
        reference = dict(game={"test": True}, targets=[dict(board=b) for b in
                         [[0,5,10,15], [8,12,16,20], [32,36,40,44], [16,17,18,23]]])
        baseline = native.family_split(reference, 10601, .25, .25)
        for index in np.concatenate(baseline[1:]):
            extended = copy.deepcopy(reference)
            row = copy.deepcopy(reference["targets"][index])
            # A different turn does not make a held-out flop a training family.
            row["board"][-1] = 51
            extended["targets"].append(row)
            with self.assertRaisesRegex(ValueError, "held-out"):
                native.family_split(extended, 10601, .25, .25, reference=reference)
        extended = copy.deepcopy(reference)
        extended["game"] = {"different": True}
        with self.assertRaisesRegex(ValueError, "game"):
            native.family_split(extended, 10601, .25, .25, reference=reference)

    def test_training_refresh_preserves_families_and_exact_heldout_labels(self):
        reference = dict(game={"test": True}, targets=[dict(board=b, value=i) for i,b in enumerate(
                         [[0,5,10,15], [8,12,16,20], [32,36,40,44], [16,17,18,23]])])
        split = native.family_split(reference, 10601, .25, .25)
        refreshed = copy.deepcopy(reference)
        index = int(split[0][0])
        refreshed["targets"][index]["value"] = -123
        refreshed["targets"][index]["board"][-1] = 51
        actual = native.family_split(refreshed, 10601, .25, .25, reference=reference,
                                     refresh_training=True)
        for expected, got in zip(split, actual): np.testing.assert_array_equal(expected, got)
        with self.assertRaisesRegex(ValueError, "prefix"):
            native.family_split(refreshed, 10601, .25, .25, reference=reference)
        for heldout in np.concatenate(split[1:]):
            changed = copy.deepcopy(refreshed)
            changed["targets"][heldout]["value"] = -1
            with self.assertRaisesRegex(ValueError, "held-out"):
                native.family_split(changed, 10601, .25, .25, reference=reference, refresh_training=True)
        refreshed["targets"][index]["board"] = [40,45,50,3]
        with self.assertRaisesRegex(ValueError, "family"):
            native.family_split(refreshed, 10601, .25, .25, reference=reference, refresh_training=True)
        with self.assertRaisesRegex(ValueError, "reference"):
            native.family_split(reference, 10601, .25, .25, refresh_training=True)

    def test_native_projection_uses_serving_epsilon_for_nearly_incompatible_ranges(self):
        board = [0, 5, 10, 15]
        ranges = np.zeros((2, native.COMBO_COUNT))
        key = lambda a, b: max(a, b) * (max(a, b) - 1) // 2 + min(a, b)
        tiny = 2.0 ** -30  # Exact binary weights avoid cancellation noise.
        ranges[0, key(51, 50)], ranges[0, key(49, 48)] = 1 - tiny, tiny
        ranges[1, key(51, 49)], ranges[1, key(50, 47)] = 1 - tiny, tiny
        joint = np.sum(ranges[0] * native.compatible_masses(ranges)[0])
        self.assertEqual(joint, tiny * tiny)
        values = np.full_like(ranges, .001)
        projected = native.project_native_predictions(values, board, ranges)
        # Native serving divides by max(joint, 1e-9), then skips a correction
        # <=1e-12. The matching Python check must take the same branch.
        np.testing.assert_allclose(projected[:, native.legal_combos(board)], .001, rtol=0, atol=1e-12)

    def test_native_prediction_projects_using_suit_augmented_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            path.write_text(json.dumps(corpus()))
            dataset = training.load_dataset(path, suit_augmentation_count=24)
        feature_schema = training.FEATURE_SCHEMA_BOARD_RELATIVE
        context_size, query_size = training.feature_sizes(feature_schema)
        def zero_layer(size):
            return dict(inputSize=size, outputSize=1, activation="linear",
                        weights=[0.0] * size, biases=[0.0])
        model = dict(schema=training.NETWORK_SCHEMA, usesExactRanges=True,
                     predictionContract="native-turn-cfv-full-stack-v1",
                     valueNormalization="payoff-exposure", featureSchema=feature_schema,
                     contextTower=[zero_layer(context_size)], queryTower=[zero_layer(query_size)],
                     head=[zero_layer(2)])
        original = parity.python_prediction(dataset, model, 0)
        index = 5
        mapping = training.combo_permutation(training.suit_permutations(24)[index])
        transformed = parity.python_prediction(dataset, model, index)
        np.testing.assert_allclose(transformed[:, mapping], original, rtol=0, atol=1e-6)

    def test_native_loader_features_preserve_sparse_blocker_cancellation_precision(self):
        source = corpus()
        row = source["targets"][0]
        legal = native.legal_combos(row["board"])
        rng = np.random.default_rng(45)
        ranges = np.exp(rng.uniform(-35, 0, (2, native.COMBO_COUNT))) * legal
        # Concentrated ranges amplify cancellation in compatible equity sums.
        ranges[:, -1] = 1e6
        ranges /= ranges.sum(axis=1, keepdims=True)
        masses = native.compatible_masses(ranges)
        totals = np.array([.125, .25])
        row["ranges"], row["opponent_compatible_mass"] = ranges.tolist(), masses.tolist()
        row["public_state"]["ranges"] = (ranges * totals[:,None]).tolist()
        row["completed_zero_own_reach"] = [0, 0]
        for name in ("counterfactual_values_bb", "raw_counterfactual_bb", "raw_profile_counterfactual_bb", "raw_best_response_counterfactual_bb"):
            row[name] = np.zeros_like(ranges).tolist()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.json"
            path.write_text(json.dumps(source))
            loaded = training.load_dataset(path)
        args = (np.array(row["board"]), row["actor"], np.array(row["invested_bb"]))
        _, expected = training.build_features(*args, ranges, masses, training.FEATURE_SCHEMA_BOARD_RELATIVE)
        _, actual = training.build_features(*args, loaded.ranges[0], loaded.masses[0], training.FEATURE_SCHEMA_BOARD_RELATIVE)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
        self.assertEqual(loaded.ranges.dtype, np.float64)
        self.assertEqual(loaded.projection_weights.dtype, np.float32)

    def test_full_stack_projection_retains_zero_own_and_preserves_zero_sum(self):
        source = corpus()
        target = source["targets"][0]
        ranges = np.asarray(target["ranges"])
        legal = native.legal_combos(target["board"])
        values = np.tile(np.linspace(-30, 30, native.COMBO_COUNT), (2, 1))
        projected = native.project_native_predictions(values, target["board"], ranges)
        self.assertTrue((np.abs(projected[:, legal]) > 1).any())
        self.assertTrue((np.abs(projected) <= 20).all())
        self.assertTrue((projected[:, ~legal] == 0).all())
        absent = int(np.flatnonzero((ranges[0] == 0) & legal)[0])
        self.assertNotEqual(projected[0, absent], 0)
        weights = ranges * native.compatible_masses(ranges)
        self.assertLess(abs(np.sum(projected * weights) / weights[0].sum()), 1e-10)

    def test_collection_pins_every_source_and_rejects_truncated_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, board in enumerate(((0,5,10,15), (8,12,16,20), (32,36,40,44), (16,17,18,23))):
                source = corpus(board=board)
                source["targets"] *= 64
                source["maximum_states"] = source["observed_queries"] = 64
                source["capture_selection"] = "all_intermediate_queries_bounded_feasibility"
                path = Path(directory) / (str(index) + ".json.gz")
                path.write_bytes(gzip.compress(json.dumps(source).encode(), mtime=0))
                paths.append(path)
            merged = pilot.merge_captures(paths)
            self.assertEqual(len(merged["targets"]), 256)
            self.assertEqual(len(merged["source_captures"]), 4)
            merged["source_captures"][0]["policy_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "provenance"):
                native.validate_dataset(merged)
            source["capture_selection"] = "first_n_in_public_leaf_chance_order_cost_preflight_only"
            paths[-1].write_bytes(gzip.compress(json.dumps(source).encode(), mtime=0))
            with self.assertRaisesRegex(ValueError, "truncated cost preflight"):
                pilot.merge_captures(paths)

    def test_search_beliefs_require_native_labels_frozen_proposer_and_every_stratum(self):
        tags = ["learned_flop_search_early_belief_native_label",
                "learned_flop_search_middle_belief_native_label",
                "learned_flop_search_late_belief_native_label",
                "learned_flop_final_average_belief_native_label"]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, board in enumerate(((0,5,10,15), (8,12,16,20), (32,36,40,44), (16,17,18,23))):
                source = corpus(board=board)
                source["targets"] = [{**source["targets"][0], "state_distribution":tags[j % 4]} for j in range(64)]
                source.update(maximum_states=64, observed_queries=1000, native_label_queries=64,
                              proposal_model_sha256="d"*64, sampling_seed=991+index,
                              proposal_policy_kind="frozen_learned_leaf_search_only",
                              capture_selection="stratified_learned_search_and_final_average_beliefs_with_native_labels")
                path = Path(directory) / (str(index) + ".json.gz")
                path.write_bytes(gzip.compress(json.dumps(source).encode(), mtime=0))
                paths.append(path)
            merged = pilot.merge_captures(paths, search_proposals=True)
            self.assertEqual(len(merged["targets"]), 256)
            self.assertEqual(merged["source_captures"][0]["proposal_model_sha256"], "d"*64)
            appended = pilot.merge_captures([paths[0]], search_proposals=True, reference=merged)
            self.assertEqual(appended["targets"][:256], merged["targets"])
            self.assertEqual(appended["source_captures"][:4], merged["source_captures"])
            self.assertEqual(len(appended["targets"]), 320)
            self.assertEqual({t["source_capture_index"] for t in appended["targets"][256:]}, {4})
            with self.assertRaises(ValueError):
                pilot.merge_captures(paths * 2, search_proposals=True, reference=merged)
            with self.assertRaisesRegex(ValueError, "truncated cost preflight"):
                pilot.merge_captures(paths)
            original = copy.deepcopy(source)
            for mutation in ("labels", "proposer", "distribution"):
                source = copy.deepcopy(original)
                if mutation == "labels": source["native_label_queries"] = 0
                if mutation == "proposer": source["proposal_model_sha256"] = "missing"
                if mutation == "distribution":
                    for row in source["targets"]: row["state_distribution"] = tags[0]
                paths[-1].write_bytes(gzip.compress(json.dumps(source).encode(), mtime=0))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    pilot.merge_captures(paths, search_proposals=True)

    def test_native_contract_accepts_zero_own_but_rejects_wrong_scaling_and_completion(self):
        source = corpus()
        native.validate_dataset(source)
        absent = int(np.flatnonzero(np.asarray(source["targets"][0]["counterfactual_values_bb"])[0])[-1])
        for mutation in ("range", "cfv", "completion", "mass", "residual", "policy", "accepted", "incomplete"):
            broken = copy.deepcopy(source)
            row = broken["targets"][0]
            if mutation == "range": row["ranges"][0][absent] = 0.1
            if mutation == "cfv": row["raw_counterfactual_bb"][0][absent] *= 0.125
            if mutation == "completion": row["raw_best_response_counterfactual_bb"][0][absent] = 0
            if mutation == "mass": row["opponent_compatible_mass"][0][absent] += 0.1
            if mutation == "residual": row["conditional_response_gain_bb"] = [0.1, 0.0]
            if mutation == "policy": row["policy_sha256"] = "missing"
            if mutation == "accepted": broken["validation"]["status"] = "accepted"
            if mutation == "incomplete": broken["maximum_states"] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.validate_dataset(broken)

    def test_zero_joint_reference_remains_undefined_not_passing_zero(self):
        source = corpus(zero_player=True)
        native.validate_dataset(source)
        source["targets"][0]["conditional_response_gain_bb"] = [0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "response residual"):
            native.validate_dataset(source)

    def test_loader_supervises_zero_own_holdings_without_changing_reach_or_projection(self):
        for zero_player in (False, True):
            source = corpus(zero_player=zero_player)
            payload = gzip.compress(json.dumps(source).encode(), mtime=0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "targets.json.gz"
                path.write_bytes(payload)
                dataset = training.load_dataset(path)
            self.assertEqual(dataset.source_sha256, hashlib.sha256(payload).hexdigest())
            ranges = dataset.ranges[0]
            legal = native.legal_combos(dataset.boards[0])
            zero_own = (ranges == 0) & legal
            weights = dataset.weights[0].reshape((2, native.COMBO_COUNT))
            supervised = zero_own & (dataset.masses[0] > 0)
            self.assertTrue(supervised.any())
            self.assertTrue((weights[supervised] > 0).all())
            self.assertTrue((dataset.projection_weights[0][zero_own] == 0).all())
            self.assertTrue((weights[:, ~legal] == 0).all())
            np.testing.assert_allclose(dataset.projection_weights[0], ranges * dataset.masses[0])
            self.assertIn("finite-budget", training.complete_turn_release_reasons(dataset.source)[0])

    def test_family_split_keeps_suits_turns_histories_and_iterations_together(self):
        boards = [[0, 5, 10, 15], [1, 4, 11, 18], [8, 12, 16, 20],
                  [8, 12, 16, 24], [32, 36, 40, 44], [16, 17, 18, 23]]
        source = dict(targets=[dict(board=b) for b in boards])
        a = native.family_split(source, 7, 0.25, 0.25)
        b = native.family_split(source, 7, 0.25, 0.25)
        families = [{native.board_family(boards[i]) for i in indices} for indices in a]
        self.assertTrue(all(families))
        self.assertFalse(families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
        self.assertEqual(sorted(np.concatenate(a).tolist()), list(range(len(boards))))
        for first, repeat in zip(a, b): np.testing.assert_array_equal(first, repeat)
        self.assertEqual(native.board_family(boards[0]), native.board_family(boards[1]))
        with self.assertRaisesRegex(ValueError, "cost preflight"):
            native.family_split(corpus(), 7, 0.25, 0.25)

    def test_preflight_refuses_disk_pressure_and_input_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, public = root / "binary", root / "input.json"
            binary.write_bytes(b"test-only")
            public.write_bytes(b"{}")
            arguments = (binary, runner.sha256(binary), public, runner.sha256(public), root / "run")
            with mock.patch.object(runner.shutil, "disk_usage", return_value=mock.Mock(free=runner.RESERVE_BYTES)):
                with self.assertRaisesRegex(ValueError, "disk reserve"):
                    runner.preflight(*arguments)
            with self.assertRaisesRegex(ValueError, "digest"):
                runner.preflight(binary, "bad", public, runner.sha256(public), root / "run")
            self.assertFalse((root / "run").exists())

    def test_cost_preflight_is_rejected_before_feature_allocation_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps(corpus()))
            output = Path(directory) / "models"
            with mock.patch.object(sys, "argv", ["train", "--dataset", str(path), "--output-dir", str(output)]), \
                 mock.patch.object(training, "feature_dataset_cached") as features:
                with self.assertRaisesRegex(ValueError, "cost preflight"):
                    training.main()
                features.assert_not_called()
            self.assertFalse(output.exists())

    def test_failed_worker_launch_is_recorded_as_failed_not_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, public, output = root / "binary", root / "input", root / "run"
            binary.write_bytes(b"invalid executable")
            public.write_bytes(b"{}")
            command = ["preflight", "--binary", str(binary), "--binary-sha256", runner.sha256(binary),
                       "--input", str(public), "--input-sha256", runner.sha256(public), "--output", str(output)]
            with mock.patch.object(sys, "argv", command), \
                 mock.patch.object(runner.shutil, "disk_usage", return_value=mock.Mock(free=runner.RESERVE_BYTES * 2)), \
                 mock.patch.object(runner.signal, "signal"), \
                 mock.patch.object(runner.subprocess, "Popen", side_effect=OSError("test launch failure")), \
                 mock.patch("builtins.print"):
                with self.assertRaises(SystemExit):
                    runner.main()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("test launch failure", manifest["failure"])
            self.assertFalse(manifest["releaseAccepted"])


if __name__ == "__main__":
    unittest.main()

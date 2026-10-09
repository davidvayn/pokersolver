import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import mlx.core as mx

from cash_profiles import profile_rules, rules_digest
from cash_value_dataset import LABEL_SCHEMA, build_dataset, validate_label, validate_dataset, extend_training_corpus
from native_value_dataset import compatible_masses, family_split, legal_combos, identity_hash
from train_cash_value_network import OwnComboValueNetwork, reference_predictions, split_cash_families
from cash_checkdown import exact_cash_checkdown


def label_fixture():
    rules = profile_rules("nl25"); board = [8,13,22,31]
    legal = legal_combos(board); ranges = np.tile(legal / legal.sum(), (2,1))
    values = np.tile(legal * -.86, (2,1))
    return {"schema":LABEL_SCHEMA,"validation":{"status":"research_only"},"joint_iterations":2,
            "input":{"iterations":2,"averaging_delay":0,"game":{"cash_rules":rules,"effective_stack_bb":20,"small_blind_bb":.4,"big_blind_bb":1},
                "state":{"street":"turn","board":board,"actor":1,"invested_bb":[19,19],"street_invested_bb":[0,0],
                         "last_full_raise_bb":1,"aggressions":0,"checks":0,"raise_reopened":True,"public_history":["public_belief:turn_start"],"ranges":ranges.tolist()}},
            "counterfactual_values_bb":values.tolist(),"opponent_compatible_mass":compatible_masses(ranges).tolist(),
            "metrics":{"exact_river_cards":48,"maximum_probability_sum_error":0,"exact_abstract_exploitability_bb_per_hand":.2,
                       "cash":{"rules_sha256":rules_digest(rules),"expected_house_rake_bb":1.72,"conservation_residual_bb":0,
                               "unilateral_gain_bb":[.1,.1],"nash_conv_bb_per_hand":.2}}}


class CashValueDatasetTests(unittest.TestCase):
    def test_training_extension_retains_parent_provenance_and_unchanged_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            def corpus(boards,prefix):
                paths = []
                for i,board in enumerate(boards):
                    label = label_fixture(); legal = legal_combos(board)
                    ranges = np.tile(legal/legal.sum(),(2,1))
                    label["input"]["state"].update(board=board,ranges=ranges.tolist())
                    label["counterfactual_values_bb"] = np.tile(legal*-.86,(2,1)).tolist()
                    label["opponent_compatible_mass"] = compatible_masses(ranges).tolist()
                    path = Path(directory)/f"{prefix}-{i}.json"; path.write_text(json.dumps(label)); paths.append(path)
                return build_dataset(paths)
            original = corpus([[8,13,22,31],[48,45,26,3],[32,29,18,7]],"original")
            _,_,holdout = split_cash_families(original,937)
            held_board = original["labels"][holdout[0]]["input"]["state"]["board"]
            addition = corpus([held_board,[40,45,2,7]],"extra")
            merged = extend_training_corpus(original,addition,937)
            validate_dataset(merged)
            self.assertEqual(len(merged["labels"]),4)
            self.assertEqual(merged["labels"][:3],original["labels"])
            baseline_splits = split_cash_families(original,937)
            new_splits = split_cash_families(merged,937,original)
            np.testing.assert_array_equal(baseline_splits[1],new_splits[1])
            np.testing.assert_array_equal(baseline_splits[2],new_splits[2])
            self.assertIn(3,new_splits[0])
            self.assertEqual(merged["source_datasets"]["sources"][1]["selected_rows"],[1])
            mutated = copy.deepcopy(merged)
            values = np.asarray(mutated["labels"][0]["counterfactual_values_bb"])
            legal = legal_combos(mutated["labels"][0]["input"]["state"]["board"])
            values[0,legal] += .1; values[1,legal] -= .1
            mutated["labels"][0]["counterfactual_values_bb"] = values.tolist()
            mutated["label_canonical_sha256"] = [identity_hash(l) for l in mutated["labels"]]
            with self.assertRaisesRegex(ValueError,"captured parent rows"): validate_dataset(mutated)
            changed = copy.deepcopy(original); changed["labels"][0]["input"]["iterations"] = 3
            with self.assertRaises(ValueError): split_cash_families(merged,937,changed)
            with self.assertRaisesRegex(ValueError,"no new unheldout"):
                extend_training_corpus(original,original,937)
            held_variant = [*held_board[:3],next(card for card in range(52) if card not in held_board)]
            second = corpus([held_variant,[4,9,30,35]],"second")
            with self.assertRaisesRegex(ValueError,"explicit original split reference"):
                extend_training_corpus(merged,second,937)
            expanded = extend_training_corpus(merged,second,937,split_reference=original)
            validate_dataset(expanded)
            self.assertEqual(len(expanded["labels"]),5)
            self.assertEqual(expanded["labels"][:4],merged["labels"])
            parents = expanded["source_datasets"]["sources"]
            self.assertEqual(len(parents),3)
            self.assertTrue(all("source_datasets" not in p["dataset"] for p in parents))
            np.testing.assert_array_equal(split_cash_families(expanded,937,original)[1],baseline_splits[1])
            np.testing.assert_array_equal(split_cash_families(expanded,937,original)[2],baseline_splits[2])
            with self.assertRaisesRegex(ValueError,"pinned split"):
                split_cash_families(expanded,938,original)
            with self.assertRaisesRegex(ValueError,"pinned split"):
                split_cash_families(expanded,937)
            third = corpus([[12,18,25,40]],"third")
            fourth = corpus([[0,16,35,45]],"fourth")
            four_parents = extend_training_corpus(expanded,third,937,split_reference=original)
            five_parents = extend_training_corpus(four_parents,fourth,937,split_reference=original)
            validate_dataset(five_parents)
            self.assertEqual(len(five_parents["source_datasets"]["sources"]),5)
            self.assertEqual(five_parents["labels"][:len(expanded["labels"])],expanded["labels"])
            np.testing.assert_array_equal(split_cash_families(five_parents,937,original)[1],baseline_splits[1])
            np.testing.assert_array_equal(split_cash_families(five_parents,937,original)[2],baseline_splits[2])
            with self.assertRaisesRegex(ValueError,"at most five flat sources"):
                extend_training_corpus(five_parents,third,937,split_reference=original)
            changed = copy.deepcopy(expanded)
            changed["source_datasets"]["split_reference_canonical_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError,"pinned split"): validate_dataset(changed)
            changed = copy.deepcopy(expanded)
            changed["source_datasets"]["sources"] *= 2
            with self.assertRaisesRegex(ValueError,"merge lineage"): validate_dataset(changed)
            changed = copy.deepcopy(expanded)
            changed["source_datasets"]["sources"][-1]["selected_rows"] = [0,1]
            changed["labels"].insert(4,copy.deepcopy(second["labels"][0]))
            changed["capture_sha256"].insert(4,second["capture_sha256"][0])
            changed["label_canonical_sha256"] = [identity_hash(l) for l in changed["labels"]]
            with self.assertRaisesRegex(ValueError,"pinned tuning/holdout"): validate_dataset(changed)

    def test_nested_merge_claims_are_rejected_without_recursive_work(self):
        label = label_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"label.json"; path.write_text(json.dumps(label))
            source = build_dataset([path])
            source["source_datasets"] = {"schema":"hu-cash-value-source-prefix-merge-v1","sources":[
                {"dataset":{"source_datasets":{}},"canonical_sha256":"a"*64,"selected_rows":[0]},
                {"dataset":{},"canonical_sha256":"b"*64,"selected_rows":[0]}]}
            with self.assertRaisesRegex(ValueError,"nested cash merges"): validate_dataset(source)

    def test_accepts_own_negative_house_share_not_zero_sum(self):
        label = label_fixture(); validate_label(label)
        corrupted = copy.deepcopy(label)
        values = np.asarray(corrupted["counterfactual_values_bb"])
        values[:, legal_combos(label["input"]["state"]["board"])] += .86
        corrupted["counterfactual_values_bb"] = values.tolist()
        with self.assertRaisesRegex(ValueError,"house ledger"): validate_label(corrupted)

    def test_rejects_relabelled_or_incomplete_target_contracts(self):
        label = label_fixture()
        for mutate in [lambda l: l.update(schema="hu-turn-public-belief-cfv-dataset-v2"),
                       lambda l: l["input"]["game"].update(small_blind_bb=.5),
                       lambda l: l["input"]["state"].update(checks=1),
                       lambda l: l["metrics"]["cash"].update(rules_sha256="0"*64),
                       lambda l: l["metrics"].update(exact_abstract_exploitability_bb_per_hand=.1),
                       lambda l: l["input"]["game"]["cash_rules"]["rake"].update(rateBasisPoints=0)]:
            corrupted = copy.deepcopy(label); mutate(corrupted)
            with self.assertRaises(ValueError): validate_label(corrupted)

    def test_blockers_and_cent_money_are_strict(self):
        label = label_fixture()
        corrupted = copy.deepcopy(label); corrupted["input"]["state"]["invested_bb"] = [19.01,19.01]
        with self.assertRaisesRegex(ValueError,"cent aligned"): validate_label(corrupted)
        corrupted = copy.deepcopy(label); corrupted["opponent_compatible_mass"][0][1000] += .01
        with self.assertRaisesRegex(ValueError,"blocker"): validate_label(corrupted)

    def test_equal_cent_native_float_roundtrip_is_accepted_without_relaxing_money(self):
        label = label_fixture()
        label["input"]["state"]["invested_bb"] = [19.,np.nextafter(19.,0.).item()]
        validate_label(label)
        label["input"]["state"]["invested_bb"] = [19.,18.96]
        with self.assertRaisesRegex(ValueError,"cent aligned"): validate_label(label)
        with self.assertRaisesRegex(ValueError,"cent aligned"):
            exact_cash_checkdown([0,5,10,15],np.zeros((2,1326)),[7.56,7.60],profile_rules("nl25"))

    def test_multiple_pots_keep_whole_board_families_out_of_training(self):
        labels = []
        for board in ([8,13,22,31], [48,45,26,3], [32,29,18,7]):
            for invested in (2,19):
                label = label_fixture()
                legal = legal_combos(board); ranges = np.tile(legal / legal.sum(), (2,1))
                label["input"]["state"].update(board=board,invested_bb=[invested,invested],ranges=ranges.tolist())
                house = .16 if invested == 2 else 1.72
                label["counterfactual_values_bb"] = np.tile(legal * (-house/2),(2,1)).tolist()
                label["opponent_compatible_mass"] = compatible_masses(ranges).tolist()
                label["metrics"]["cash"]["expected_house_rake_bb"] = house
                labels.append(label)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index,label in enumerate(labels):
                path = Path(directory)/f"label-{index}.json"
                path.write_text(json.dumps(label)); paths.append(path)
            source = build_dataset(paths)
            train,tuning,holdout = family_split({"game":source["game"],
                "targets":[{"board":l["input"]["state"]["board"]} for l in source["labels"]]},7101,.2,.2)
            for split in (train,tuning,holdout):
                self.assertEqual(len(split),2)
                self.assertEqual(labels[split[0]]["input"]["state"]["board"],labels[split[1]]["input"]["state"]["board"])
            with self.assertRaisesRegex(ValueError,"duplicate cash inputs"):
                build_dataset(paths + [paths[0]])
            relabeled = copy.deepcopy(labels[0]); relabeled["input"]["game"]["three_bet_sizes_bb"] = [6]
            paths[0].write_text(json.dumps(relabeled))
            with self.assertRaisesRegex(ValueError,"cannot mix"):
                build_dataset(paths)

    def test_mlx_own_projection_does_not_cancel_rake_or_couple_heads(self):
        model = OwnComboValueNetwork()
        raw = mx.full((1,2,1326), -.86)
        model.raw_values = lambda *args: raw
        weights = mx.ones_like(raw); legal = mx.ones((1,1326)); scale = mx.array([1.])
        values = np.array(model(None,None,weights,scale,raw,legal)).reshape((1,2,1326))
        self.assertAlmostEqual(values.sum(), -.86 * 2652, places=2)
        # This test isolates the exact differentiable clip/mask operation used
        # by both heads; altering p0 cannot feed a projection gradient into p1.
        gradient = mx.grad(lambda r: mx.sum(mx.clip(r,-20,20)[:,0]))(raw)
        self.assertEqual(float(mx.sum(gradient[:,1]).item()), 0)

    def test_reported_predictions_use_cpu_stream_and_exact_board_mask(self):
        model = OwnComboValueNetwork()
        model.raw_values = lambda *args: mx.full((1,2,1326), -.86)
        legal = np.ones((1,1326)); legal[0,5] = 0
        values = reference_predictions(model, [None]*5, np.ones(1), legal)
        self.assertEqual(values[0,0,5], 0)
        self.assertAlmostEqual(values[0,0,0] + values[0,1,0], -1.72, places=6)

    def test_exact_checkdown_retains_house_rake_under_compatible_joint_belief(self):
        label = label_fixture(); state = label["input"]["state"]
        ranges = np.asarray(state["ranges"])
        values = exact_cash_checkdown(state["board"],ranges,state["invested_bb"],profile_rules("nl25"))
        weights = ranges * compatible_masses(ranges)
        total = np.sum(weights*values,axis=1)/weights.sum(axis=1)
        self.assertAlmostEqual(total.sum(),-1.72,places=9)
        self.assertTrue(np.all(values[:,~legal_combos(state["board"])] == 0))

    def test_optional_authentic_reach_lineage_stays_bound_to_label_inputs(self):
        label = label_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"label.json"; path.write_text(json.dumps(label))
            source = build_dataset([path])
            source["reach_provenance"] = dict(schema="hu-cash-public-reach-lineage-v1",
                kind="authentic-frozen-average-turn-roots",rules_sha256=source["rules_sha256"],
                policy_sha256="a"*64,root_corpus_sha256="b"*64,root_sha256=["c"*64],
                label_input_sha256=[identity_hash(label["input"])])
            validate_dataset(source)
            for field,value in [("rules_sha256","0"*64),("root_sha256",[]),("policy_sha256","bad"),("label_input_sha256",["0"*64])]:
                altered = copy.deepcopy(source); altered["reach_provenance"][field] = value
                with self.assertRaises(ValueError): validate_dataset(altered)


if __name__ == "__main__": unittest.main()

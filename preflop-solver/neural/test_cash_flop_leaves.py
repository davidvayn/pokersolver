import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cash_flop_leaves import leaf_input
from cash_turn_roots import root_fingerprint
from cash_value_dataset import build_dataset, validate_dataset
from probe_cash_flop_leaves import check_check_input, reference, evaluate
from native_value_dataset import COMBOS
import test_cash_flop_pilot as pilot_tests
from test_cash_turn_roots import fixture
from test_cash_value_dataset import label_fixture, complete_fixture_targets
from native_value_dataset import compatible_masses, identity_hash, legal_combos


class CashFlopLeafTests(unittest.TestCase):
    def fixture(self):
        config,solution=pilot_tests.CashFlopPilotTests().fixture()
        source,_=fixture(); root=source["roots"][0]
        root["solve_input"]=config; root["source_public_actions"]=root["source_public_actions"][:2]
        second=copy.deepcopy(solution["root"])
        second.update(actor=0,public_history=["public_belief:flop_start","Flop:p1:check"])
        solution["strategies"].append(second)
        return root,solution

    def test_check_check_replays_both_ranges_before_masking_only_revealed_turn(self):
        root,solution=self.fixture(); original=copy.deepcopy(root)
        first=check_check_input(root,solution,31)
        second=check_check_input(root,solution,32)
        self.assertEqual(root,original)
        self.assertEqual(first["state"]["street"],"turn")
        self.assertEqual(first["state"]["public_history"],["public_belief:turn_start"])
        ranges=np.asarray(first["state"]["ranges"])
        self.assertTrue(np.allclose(ranges.sum(axis=1),1.))
        self.assertTrue((ranges[:,(COMBOS == 31).any(axis=1)] == 0).all())
        self.assertGreater(np.asarray(second["state"]["ranges"])[:,(COMBOS == 31).any(axis=1)].sum(),0.)
        self.assertEqual(first["iterations"],64)

    def test_nonuniform_action_likelihoods_condition_each_actors_own_range(self):
        root,solution=self.fixture()
        prior=np.asarray(root["solve_input"]["state"]["ranges"])
        expected=prior.copy()
        for row in solution["strategies"]:
            probabilities=np.asarray(row["probabilities"]).reshape(1326,2)
            legal=probabilities.sum(axis=1) > 0
            likelihood=np.linspace(.05,.95,1326)
            if row["actor"] == 0: likelihood=1-likelihood
            probabilities[legal,0]=likelihood[legal]
            probabilities[legal,1]=1-likelihood[legal]
            row["probabilities"]=probabilities.reshape(-1).tolist()
            expected[row["actor"]] *= probabilities[:,0]
        expected[:,(COMBOS == 31).any(axis=1)]=0
        expected /= expected.sum(axis=1)[:,None]
        actual=np.asarray(check_check_input(root,solution,31)["state"]["ranges"])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
        self.assertGreater(np.max(np.abs(actual[0]-actual[1])),1e-4)

    def test_live_bet_call_increases_both_cent_commitments_without_future_information(self):
        root,solution=self.fixture()
        root_row=solution["root"]
        root_row["action_labels"][1]="bet_to_1.000bb"
        called=copy.deepcopy(root_row)
        called.update(actor=0,public_history=["public_belief:flop_start","Flop:p1:bet_to_1.000bb"],
                      action_labels=["fold","call"])
        solution["strategies"].append(called)
        result=leaf_input(root,solution,31,["bet_to_1.000bb","call"])
        self.assertEqual(result["state"]["invested_bb"],[2.,2.])
        self.assertEqual(result["state"]["street_invested_bb"],[0,0])
        for branch in [["bet_to_1.000bb","fold"],["check","check","check"],
                       ["bet_to_1.000bb","check"],["bet_to_1.000bb","call","check"]]:
            with self.assertRaises(ValueError): leaf_input(root,solution,31,branch)

    def test_raise_call_conditions_three_actions_and_keeps_street_targets_distinct_from_total_commitments(self):
        root,solution=self.fixture()
        solution["root"]["action_labels"][1]="bet_to_1.000bb"
        legal=np.asarray(root["solve_input"]["state"]["ranges"])[0] > 0
        raised=dict(actor=0,public_history=["public_belief:flop_start","Flop:p1:bet_to_1.000bb"],
                    action_labels=["fold","call","raise_to_3.000bb"],
                    probabilities=(legal[:,None]*np.array([.2,.3,.5])).reshape(-1).tolist())
        called=copy.deepcopy(solution["root"])
        called.update(public_history=raised["public_history"]+["Flop:p0:raise_to_3.000bb"],
                      action_labels=["fold","call"])
        solution["strategies"].extend([raised,called])
        result=leaf_input(root,solution,31,["bet_to_1.000bb","raise_to_3.000bb","call"])
        self.assertEqual(result["state"]["invested_bb"],[4.,4.])
        self.assertEqual(result["state"]["street_invested_bb"],[0,0])

    def test_exact_reference_cache_is_independent_of_new_student_weights(self):
        root,solution=self.fixture(); config=check_check_input(root,solution,31)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); binary=output/"binary"; binary.write_bytes(b"native")
            first,second=output/"first-network",output/"second-network"
            first.write_bytes(b"first"); second.write_bytes(b"second")
            def native(command,**kwargs):
                if "turn-river-pbs-solve" in command:
                    label=dict(input=config,counterfactual_values_bb=np.zeros((2,1326)).tolist(),
                               metrics=dict(cash=dict(rules_sha256="a"*64,expected_house_rake_bb=.08)))
                    Path(command[command.index("--output")+1]).write_text(json.dumps(label))
                    return None
                prediction=dict(rules_sha256="a"*64,research_only=True,
                                counterfactual_values_bb=np.zeros((2,1326)).tolist())
                return type("Result",(),dict(stdout=json.dumps(prediction)))()
            with patch("probe_cash_flop_leaves.subprocess.run",side_effect=native) as call, \
                    patch("probe_cash_flop_leaves.validate_label"):
                provenance=dict(branch="check/check")
                evaluate(binary,output,config,[first],provenance)
                evaluate(binary,output,config,[second],provenance)
                self.assertEqual(sum("turn-river-pbs-solve" in c.args[0] for c in call.call_args_list),1)
                path,_=reference(binary,output,config,provenance)
                input_path=path.with_name(path.name.replace("reference-","input-"))
                input_path.write_text("{}")
                with self.assertRaisesRegex(ValueError,"pinned leaf"): reference(binary,output,config,provenance)

    def test_leaf_training_lineage_replays_ranges_instead_of_relabeling_arbitrary_inputs(self):
        root,solution=self.fixture()
        root["root_sha256"]=root_fingerprint("a"*64,root)
        config=check_check_input(root,solution,31,2)
        label=label_fixture(); legal=legal_combos(config["state"]["board"])
        ranges=np.asarray(config["state"]["ranges"])
        label.update(input=config,counterfactual_values_bb=np.tile(legal*-.04,(2,1)).tolist(),
                     opponent_compatible_mass=compatible_masses(ranges).tolist())
        label["metrics"]["cash"]["expected_house_rake_bb"]=.08
        complete_fixture_targets(label)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"label.json"; path.write_text(json.dumps(label))
            source=build_dataset([path])
            provenance=dict(schema="hu-cash-frozen-flop-leaf-lineage-v1",
                kind="forced-frozen-flop-average-leaves",rules_sha256=source["rules_sha256"],split_seed=937,
                rows=[dict(root=root,root_sha256=root["root_sha256"],flop_solution_sha256="b"*64,
                           branch_policy_rows=solution["strategies"],branch_labels=["check","check"],
                           public_turn_proposal=31,sampling_seed=971,label_input_sha256=identity_hash(config))])
            for key in ("source_policy_sha256","root_corpus_sha256","flop_pair_report_sha256",
                        "native_binary_sha256","source_flop_value_network_sha256",
                        "split_reference_sha256","excluded_roots_sha256"):
                provenance[key]="a"*64
            source["flop_leaf_provenance"]=provenance
            validate_dataset(source)
            for mutate in [lambda s:s["flop_leaf_provenance"]["rows"][0].update(public_turn_proposal=32),
                           lambda s:s["flop_leaf_provenance"].update(source_policy_sha256="b"*64),
                           lambda s:s["flop_leaf_provenance"]["rows"][0].update(branch_labels=["check","call"]),
                           lambda s:s["flop_leaf_provenance"]["rows"][0]["root"]["solve_input"]["state"].update(invested_bb=[2,2])]:
                bad=copy.deepcopy(source); mutate(bad)
                with self.assertRaises(ValueError): validate_dataset(bad)

    def test_missing_branch_future_card_and_invalid_budget_do_not_get_fallbacks(self):
        root,solution=self.fixture()
        for turn,updates in [(root["solve_input"]["state"]["board"][0],64),(31,129)]:
            with self.assertRaises(ValueError): check_check_input(root,solution,turn,updates)
        solution["strategies"].pop()
        with self.assertRaises(ValueError): check_check_input(root,solution,31)


if __name__ == "__main__": unittest.main()

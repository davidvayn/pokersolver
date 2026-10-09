import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile

import numpy as np

from cash_profiles import CASH_NETWORK_SCHEMA, CASH_STATE_FEATURE_COUNT, profile_rules, rules_digest
from cash_turn_roots import HASH_SCHEMA, root_fingerprint, validate_roots, validate_public_money
from native_value_dataset import compatible_masses, legal_combos
from run_cash_value_pilot import generate_root
from test_cash_value_dataset import label_fixture


def fixture():
    rules = profile_rules("nl25"); label = label_fixture()
    config = label["input"]; config["game"]["action_abstraction"] = {}
    config.update(river_refinement_iterations=0, regret_matching_plus=False)
    config["state"]["invested_bb"] = [1.,1.]
    actions = [dict(actor=actor,street=street,kind=kind,amount_bb=amount,amount_to_bb=None,pot_after_bb=2.)
               for actor,street,kind,amount in [(0,"preflop","call",.6),(1,"preflop","check",0.),(1,"flop","check",0.),(0,"flop","check",0.)]]
    ranges = np.asarray(config["state"]["ranges"])
    root = dict(solve_input=config,source_public_actions=actions,source_deal_index=1,
                compatible_joint_mass=float(np.sum(ranges*compatible_masses(ranges),axis=1)[0]))
    root["root_sha256"] = root_fingerprint("a"*64,root)
    source = dict(schema="hu-cash-authentic-turn-roots-v1",root_hash_schema=HASH_SCHEMA,validation_status="research_only",
                  rules_sha256=rules_digest(rules),policy_sha256="a"*64,sampling_seed=931,sampled_deals=2,roots=[root])
    model = dict(schema=CASH_NETWORK_SCHEMA,strategy_transform="softmax",input_size=CASH_STATE_FEATURE_COUNT+9,
                 cash_depth_bb=20,cash_rules=rules,cash_action_abstraction={})
    return source,model


class CashTurnRootTests(unittest.TestCase):
    def test_fresh_flop_capture_does_not_inherit_turn_card_conditioning(self):
        source,model = fixture(); root = source["roots"][0]
        source["schema"] = "hu-cash-authentic-flop-roots-v1"
        config = root["solve_input"]; state = config["state"]
        state.update(street="flop",board=state["board"][:3],public_history=["public_belief:flop_start"])
        legal = legal_combos(state["board"])
        state["ranges"] = np.broadcast_to(legal / legal.sum(),(2,1326)).tolist()
        config.pop("river_refinement_iterations"); config.pop("regret_matching_plus"); config["threads"] = 2
        root["source_public_actions"] = root["source_public_actions"][:2]
        ranges = np.asarray(state["ranges"])
        root["compatible_joint_mass"] = float(np.sum(ranges*compatible_masses(ranges),axis=1)[0])
        root["root_sha256"] = root_fingerprint("a"*64,root)
        validate_roots(source,model,"a"*64,"nl25",street="flop")
        self.assertEqual(np.count_nonzero(ranges[0]),1176)
        with self.assertRaises(ValueError): validate_roots(source,model,"a"*64,"nl25")
        config["iterations"] = 33
        root["root_sha256"] = root_fingerprint("a"*64,root)
        with self.assertRaises(ValueError): validate_roots(source,model,"a"*64,"nl25",street="flop")

    def test_validated_public_capture_is_not_conditioned_on_a_sampled_private_hand(self):
        source,model = fixture(); validate_roots(source,model,"a"*64,"nl25")
        ranges = np.asarray(source["roots"][0]["solve_input"]["state"]["ranges"])
        self.assertEqual(np.count_nonzero(ranges[0]),1128)
        self.assertEqual(np.count_nonzero(ranges[1]),1128)
        self.assertLess(abs(ranges.sum(axis=1)[0]-1),1e-10)

    def test_mutated_state_ranges_line_or_model_cannot_reuse_capture(self):
        mutations = [lambda s,m: s.update(policy_sha256="b"*64),
                     lambda s,m: m.update(strategy_transform="regret_matching"),
                     lambda s,m: s["roots"][0]["solve_input"]["state"].update(actor=0),
                     lambda s,m: s["roots"][0]["source_public_actions"][0].update(actor=1),
                     lambda s,m: s["roots"][0]["solve_input"]["state"]["ranges"][0].__setitem__(0,.1),
                     lambda s,m: s["roots"][0].update(root_sha256="b"*64),
                     lambda s,m: s.update(roots=s["roots"]*2)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                source,model = fixture(); mutate(source,model)
                with self.assertRaises(ValueError): validate_roots(source,model,"a"*64,"nl25")

    def test_hash_preserves_very_small_f64_reach_and_changes_with_public_state(self):
        source,_ = fixture(); root = source["roots"][0]
        first = root_fingerprint("a"*64,root)
        altered = copy.deepcopy(root); altered["solve_input"]["state"]["ranges"][0][0] = 1e-100
        self.assertNotEqual(first,root_fingerprint("a"*64,altered))
        self.assertEqual(root_fingerprint("a"*64,altered),root_fingerprint("a"*64,json.loads(json.dumps(altered))))

    def test_public_payments_must_replay_to_the_actual_committed_cents(self):
        source,_ = fixture(); actions = source["roots"][0]["source_public_actions"]
        validate_public_money(actions,[1.,1.])
        with self.assertRaises(ValueError): validate_public_money(actions,[19.,19.])
        altered = copy.deepcopy(actions); altered[0]["amount_bb"] = .64
        with self.assertRaises(ValueError): validate_public_money(altered,[1.,1.])
        altered = copy.deepcopy(actions); altered[2]["kind"] = "all_in"
        with self.assertRaises(ValueError): validate_public_money(altered,[1.,1.])

    def test_cent_equal_native_float_commitments_are_not_rejected_as_unequal_money(self):
        source,model = fixture(); root = source["roots"][0]
        # Native replay may represent the same paid cent as adjacent floats.
        root["solve_input"]["state"]["invested_bb"] = [1.,np.nextafter(1.,0.).item()]
        root["root_sha256"] = root_fingerprint("a"*64,root)
        validate_roots(source,model,"a"*64,"nl25")
        root["solve_input"]["state"]["invested_bb"] = [1.,1.04]
        root["root_sha256"] = root_fingerprint("a"*64,root)
        with self.assertRaisesRegex(ValueError,"cent commitments"):
            validate_roots(source,model,"a"*64,"nl25")

    def test_root_label_cache_binds_complete_input_and_rejects_changed_ranges(self):
        source,_ = fixture(); root = source["roots"][0]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory); binary = target/"binary"; binary.write_bytes(b"test-binary")
            def fake_native(command,**kwargs):
                input_path = Path(command[command.index("--input")+1])
                output_path = Path(command[command.index("--output")+1])
                label = label_fixture(); label["input"] = json.loads(input_path.read_text())
                label["joint_iterations"] = label["input"]["iterations"]
                output_path.write_text(json.dumps(label))
                return type("Result",(),dict(returncode=0))()
            with patch("run_cash_value_pilot.subprocess.run",side_effect=fake_native) as native, patch("run_cash_value_pilot.validate_label"):
                path,_ = generate_root(binary,target,"nl25",root,8,{"policy_sha256":"a"*64})
                generate_root(binary,target,"nl25",root,8,{"policy_sha256":"a"*64})
                self.assertEqual(native.call_count,1)
                altered = copy.deepcopy(root); altered["solve_input"]["state"]["ranges"][0][0] = .1
                other,_ = generate_root(binary,target,"nl25",altered,8,{"policy_sha256":"a"*64})
                self.assertNotEqual(path,other)
                changed = json.loads(path.read_text()); changed["input"]["state"]["ranges"][0][0] = .1
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): generate_root(binary,target,"nl25",root,8,{"policy_sha256":"a"*64})


if __name__ == "__main__": unittest.main()

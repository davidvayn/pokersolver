import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from native_value_dataset import legal_combos
from run_cash_flop_pilot import generate, run, sha, validate_solution
from test_cash_turn_roots import fixture


class CashFlopPilotTests(unittest.TestCase):
    def fixture(self):
        source,_ = fixture(); config=source["roots"][0]["solve_input"]
        config["threads"]=2; state=config["state"]
        state.update(street="flop",board=state["board"][:3],public_history=["public_belief:flop_start"])
        legal=legal_combos(state["board"])
        state["ranges"]=np.broadcast_to(legal/legal.sum(),(2,1326)).tolist()
        row=dict(actor=1,public_history=state["public_history"],action_labels=["check","bet_all_in_to_19.000bb"],
                 probabilities=np.broadcast_to([.5,.5],(1326,2)).__mul__(legal[:,None]).reshape(-1).tolist(),
                 action_values_bb=np.zeros(2652).tolist())
        result=dict(schema="hu-cash-depth-limited-flop-pilot-v1",validation=dict(status="research_only"),
                    full_game_exploitability="unmeasured",expected_house_rake_bb=None,
                    continuation_model_confidence="unqualified-research-only",value_network_sha256="a"*64,rules_sha256="b"*64,
                    iterations=config["iterations"],threads=2,game=config["game"],state=state,
                    action_value_method="exact-public-chance-and-terminals-with-learned-own-payoff-turn-leaves-v1",
                    profile_net_bb=[-.08,-.1],predicted_profile_payoff_sum_bb=-.18,root=row,strategies=[row])
        return config,result

    def test_frozen_cash_flop_rows_keep_own_payoffs_and_research_confidence(self):
        config,result=self.fixture()
        validate_solution(result,config,"a"*64,"b"*64)
        for key,value in [("rules_sha256","c"*64),("value_network_sha256","c"*64),
                          ("iterations",33),("full_game_exploitability","passed"),
                          ("expected_house_rake_bb",.18),("continuation_model_confidence","high")]:
            bad=copy.deepcopy(result); bad[key]=value
            with self.assertRaises(ValueError): validate_solution(bad,config,"a"*64,"b"*64)

    def test_bad_policy_values_and_capture_changes_are_rejected(self):
        config,result=self.fixture()
        for mutate in [lambda r:r["root"]["probabilities"].__setitem__(0,-.1),
                       lambda r:r["root"]["action_values_bb"].__setitem__(0,float("nan")),
                       lambda r:r["root"]["action_values_bb"].__setitem__(0,21.),
                       lambda r:r["state"]["ranges"][0].__setitem__(0,.1),
                       lambda r:r.update(predicted_profile_payoff_sum_bb=0.)]:
            bad=copy.deepcopy(result); mutate(bad)
            with self.assertRaises(ValueError): validate_solution(bad,config,"a"*64,"b"*64)

    def test_unbounded_process_counts_fail_before_artifact_io(self):
        for workers in (0,3):
            with self.assertRaises(ValueError): run(Path("missing"),Path("missing"),Path("missing"),
                [Path("missing"),Path("missing")],Path("must-not-create"),workers=workers)

    def test_cache_binds_binary_weights_input_and_frozen_solution_bytes(self):
        config,result=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); binary=output/"binary"; network=output/"weights"
            binary.write_bytes(b"native"); network.write_bytes(b"network")
            result["value_network_sha256"]=sha(network)
            root=dict(root_sha256="c"*64,solve_input=config)
            def native(command,**kwargs):
                Path(command[command.index("--output")+1]).write_text(json.dumps(result))
            with patch("run_cash_flop_pilot.subprocess.run",side_effect=native) as call:
                generated=generate(binary,output,root,network,"b"*64)
                self.assertFalse(generated["cache_hit"])
                self.assertTrue(generate(binary,output,root,network,"b"*64)["cache_hit"])
                self.assertEqual(call.call_count,1)
                path=Path(generated["path"])
                # Even a numerically valid alteration cannot reuse a capture.
                path.write_text(json.dumps(result,indent=2))
                with self.assertRaises(ValueError): generate(binary,output,root,network,"b"*64)


if __name__ == "__main__": unittest.main()

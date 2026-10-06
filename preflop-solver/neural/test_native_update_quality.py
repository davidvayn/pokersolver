import math
from pathlib import Path
import json
import tempfile
import unittest

from run_native_update_quality import first_verdict, native_environment, paired_verdict, pin_screen, progression, verify_cost
from run_native_value_preflight import sha256


class NativeUpdateQualityTest(unittest.TestCase):
    def test_completed_screen_seam_rejects_wrong_root_seed_partial_chance_and_failed_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path/"audit").mkdir()
            state = dict(board=[8,9,16],ranges=[[.5,.5],[.5,.5]])
            game = dict(effective_stack_bb=20)
            root = path/"root.json"; root.write_text(json.dumps(dict(public=state,game=game)))
            binary = path/"binary"; binary.write_text("fixture")
            candidate = dict(state=state,game=game,seed=100101,iterations=64,turn_iterations=64)
            (path/"candidate.json").write_text(json.dumps(candidate))
            candidate_sha = sha256(path/"candidate.json")
            response = dict(public_turns=49,candidate_sha256=candidate_sha,half_summed_gain_bb=.12)
            (path/"response.json").write_text(json.dumps(response))
            audit = path/"audit/completed.json"
            audit.write_text(json.dumps(dict(worker=dict(status="complete",exitCode=0,resourceStopReason=None))))
            record = dict(schema="native64-flop-update-first-screen-v1",status="complete",seed=100101,
                spot="limped-paired",flopIterations=64,turnIterations=64,controlGainBb=.188,gainBb=.12,
                candidateSha256=candidate_sha,responseSha256=sha256(path/"response.json"),
                pinnedInputs={str(binary):sha256(binary),str(root):sha256(root)},
                packets={str(t):dict(turn=t,sha256="b"*64) for t in range(52) if t not in state["board"]})
            control=dict(seed=100101,spot="limped-paired",gainBb=.188)
            pinned={str(binary):sha256(binary)}
            pin_screen(path/"manifest.json",record,pinned,binary,root,control)
            self.assertEqual(len([p for p in pinned if "/packets/" in p]),49)
            for bad in ({**record,"status":"running"},{**record,"seed":100102},
                        {**record,"packets":{}},{**record,"gainBb":.1}):
                with self.assertRaises(ValueError):
                    pin_screen(path/"manifest.json",bad,pinned,binary,root,control)
            changed=path/"different-root.json"
            changed.write_text(json.dumps(dict(public={**state,"board":[0,1,2]},game=game)))
            bad={**record,"pinnedInputs":{**record["pinnedInputs"],str(changed):sha256(changed)}}
            with self.assertRaises(ValueError): pin_screen(path/"manifest.json",bad,pinned,binary,changed,control)
            audit.write_text(json.dumps(dict(worker=dict(status="complete",exitCode=1))))
            with self.assertRaises(ValueError): pin_screen(path/"manifest.json",record,pinned,binary,root,control)

    def test_expansion_requires_its_completed_prior_cases(self):
        for row in (("limped-paired",100101,False,False),("limped-paired",100102,True,False),
                    ("single-raised-high-rainbow",100101,False,True),("single-raised-high-rainbow",100102,True,True)):
            progression(*row)
        for row in (("limped-paired",100102,False,False),("limped-paired",100101,True,False),
                    ("single-raised-high-rainbow",100101,False,False),("other",100101,False,True)):
            with self.assertRaises(ValueError): progression(*row)

    def test_real_policy_improvement_required(self):
        self.assertTrue(first_verdict(.16, .188)["promising"])
        self.assertFalse(first_verdict(.18, .188)["promising"])
        self.assertFalse(first_verdict(.3, .188)["promising"])
        self.assertFalse(first_verdict(.16, .188)["releaseAccepted"])
        for value in (math.nan, math.inf, -.1):
            with self.assertRaises(ValueError): first_verdict(value, .188)

    def test_only_flop_update_budget_changes(self):
        env = native_environment(Path("r"), "hash", Path("o"), {})
        self.assertEqual(env["POKER_NATIVE_FLOP_ITERATIONS"], "64")
        self.assertEqual(env["POKER_NATIVE_FLOP_TURN_ITERATIONS"], "64")
        self.assertEqual(env["POKER_NATIVE_FLOP_TURN_SAMPLES"], "1")
        self.assertFalse(any("MODEL" in k or "AVERAGING" in k or "TAIL" in k for k in env))
        with self.assertRaises(ValueError):
            native_environment(Path("r"), "hash", Path("o"), {"POKER_NATIVE_FLOP_VALUE_MODEL":"bad"})
        second = native_environment(Path("r"), "hash", Path("o"), {}, seed=100102)
        self.assertEqual(second["POKER_NATIVE_FLOP_SEED"], "100102")
        self.assertEqual({k:v for k,v in second.items() if not k.endswith("SEED")},
                         {k:v for k,v in env.items() if not k.endswith("SEED")})
        with self.assertRaises(ValueError): native_environment(Path("r"), "h", Path("o"), {}, seed=5)

    def test_pair_must_preserve_control_and_improve_mean(self):
        self.assertTrue(paired_verdict(.20, .246, .128, .188)["promising"])
        self.assertTrue(paired_verdict(.255, .246, .128, .188)["promising"])
        self.assertFalse(paired_verdict(.27, .246, .128, .188)["promising"])
        self.assertFalse(paired_verdict(.254, .246, .16, .188)["promising"])
        with self.assertRaises(ValueError): paired_verdict(.2, .246, .185, .188)

    def test_cost_preflight_requires_complete_exact_policy(self):
        value = dict(schema="native-parallel-construction-cost-v1", status="complete", leafWorkers=4,
            cases=[dict(iterations=8, parityPassed=True), dict(iterations=32, parityPassed=True,
                candidateSha256="abc", solveSeconds=1000)])
        self.assertEqual(verify_cost(value, {"candidateSha256":"abc"}), 1000)
        for bad in ({**value, "status":"running"}, {**value, "cases":value["cases"][:1]},
                    {**value, "leafWorkers":1}):
            with self.assertRaises(ValueError): verify_cost(bad, {"candidateSha256":"abc"})
        with self.assertRaises(ValueError): verify_cost(value, {"candidateSha256":"changed"})


if __name__ == "__main__": unittest.main()

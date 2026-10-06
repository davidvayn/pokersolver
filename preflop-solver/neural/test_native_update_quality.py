import math
from pathlib import Path
import unittest

from run_native_update_quality import first_verdict, native_environment, paired_verdict, verify_cost


class NativeUpdateQualityTest(unittest.TestCase):
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

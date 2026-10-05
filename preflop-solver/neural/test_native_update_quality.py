import math
from pathlib import Path
import unittest

from run_native_update_quality import first_verdict, native_environment, verify_cost


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

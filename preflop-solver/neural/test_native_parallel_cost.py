import math
from pathlib import Path
import unittest

from run_native_parallel_cost import construction_environment, projection_passes


class NativeParallelCostTest(unittest.TestCase):
    def test_only_parallelism_changes_native_configuration(self):
        env = construction_environment(Path("root"), "abc", Path("output"), 8, {})
        self.assertEqual(env["POKER_NATIVE_FLOP_LEAF_WORKERS"], "4")
        self.assertEqual(env["POKER_NATIVE_FLOP_TURN_ITERATIONS"], "64")
        self.assertEqual(env["POKER_NATIVE_FLOP_TURN_SAMPLES"], "1")
        self.assertEqual(env["POKER_NATIVE_FLOP_CHANCE_BASELINE"], "none")
        self.assertFalse(any("MODEL" in k or "TAIL" in k or "AVERAGING" in k for k in env))

    def test_rejects_stale_research_environment(self):
        for key in ("POKER_NATIVE_FLOP_VALUE_MODEL", "POKER_NATIVE_FLOP_NATIVE_TAIL_ITERATIONS",
                    "POKER_NATIVE_FLOP_AVERAGING_START", "POKER_NATIVE_FLOP_LEAF_WORKERS"):
            with self.assertRaises(ValueError):
                construction_environment(Path("root"), "abc", Path("output"), 8, {key: "1"})

    def test_fixed_cost_stage_budgets(self):
        for value in (0, 2, 16, 64, 128):
            with self.assertRaises(ValueError):
                construction_environment(Path("root"), "abc", Path("output"), value, {})
        self.assertEqual(construction_environment(Path("r"), "a", Path("o"), 32, {})["POKER_NATIVE_FLOP_ITERATIONS"], "32")

    def test_projection_includes_margin_and_rejects_invalid_timings(self):
        self.assertTrue(projection_passes(359))
        self.assertFalse(projection_passes(360))
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                projection_passes(value)


if __name__ == "__main__":
    unittest.main()

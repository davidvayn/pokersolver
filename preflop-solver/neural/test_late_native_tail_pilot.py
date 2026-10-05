import unittest

from run_late_native_tail_pilot import measured_packet_work, tail_environment, verdict


class LateNativeTailTests(unittest.TestCase):
    def test_cost_projection_requires_complete_actual_turn_timings(self):
        timing = dict(schema="native-inner-budget-first-quality-screen-v1", status="complete", cases=[
            dict(spot="limped-paired", seed=100101, packets={str(i):dict(turn=i, seconds=10.)for i in range(49)})])
        self.assertEqual(measured_packet_work(timing), 490.)
        for changed in (dict(timing, status="running"), dict(timing, cases=[])):
            with self.assertRaises(ValueError): measured_packet_work(changed)
        timing["cases"][0]["packets"]["48"]["turn"] = 0
        with self.assertRaises(ValueError): measured_packet_work(timing)

    def test_matched_environment_changes_only_tail_and_output(self):
        model = dict(path="frozen-model", sha256="a"*64)
        control = tail_environment("root", "b"*64, model, "control", 0, {})
        candidate = tail_environment("root", "b"*64, model, "candidate", 8, {})
        differences = {k for k in control if control[k] != candidate[k]}
        self.assertEqual(differences, {"POKER_NATIVE_FLOP_NATIVE_TAIL_ITERATIONS", "POKER_NATIVE_FLOP_OUTPUT"})
        self.assertEqual(control["POKER_NATIVE_FLOP_AVERAGING_START"], "25")
        self.assertEqual(candidate["POKER_NATIVE_FLOP_TURN_ITERATIONS"], "64")
        for tail, environment in ((4, {}), (8, {"POKER_NATIVE_FLOP_AVERAGING_START":"1"})):
            with self.assertRaises(ValueError): tail_environment("root", "b"*64, model, "output", tail, environment)

    def test_cost_and_actual_gain_both_required_without_release_acceptance(self):
        native, learned = dict(gainBb=.2, solveSeconds=1000), dict(gainBb=.5, solveSeconds=10)
        rows = [dict(arm="learned-tail-average", gainBb=.45, solveSeconds=10),
                dict(arm="native8-tail-average", gainBb=.3, solveSeconds=300)]
        result = verdict(rows, native, learned)
        self.assertEqual(result["status"], "promising")
        self.assertFalse(result["releaseAccepted"])
        self.assertFalse(result["generatorAccepted"])
        self.assertEqual(verdict([rows[0], {**rows[1], "gainBb":0.}], native, learned)["status"], "promising")
        for key, value in (("gainBb", .42), ("solveSeconds", 500)):
            changed = [rows[0], {**rows[1], key:value}]
            self.assertEqual(verdict(changed, native, learned)["status"], "not_promising")

    def test_partial_duplicate_nonfinite_or_invalid_reference_is_not_a_score(self):
        rows = [dict(arm="learned-tail-average", gainBb=.45, solveSeconds=10),
                dict(arm="native8-tail-average", gainBb=.3, solveSeconds=300)]
        native, learned = dict(gainBb=.2, solveSeconds=1000), dict(gainBb=.5, solveSeconds=10)
        for bad in (rows[:1], [rows[0], rows[0]], [rows[0], {**rows[1], "gainBb":float("nan")} ]):
            with self.assertRaises(ValueError): verdict(bad, native, learned)
        with self.assertRaises(ValueError): verdict(rows, learned, native)


if __name__ == "__main__": unittest.main()

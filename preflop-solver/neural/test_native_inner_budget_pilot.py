import copy
import math
import unittest
from run_native_inner_budget_pilot import case_rejection, construction_environment, controls, evaluation_candidate
from run_native_budget_quality_screen import verify_screen_inputs


class NativeInnerBudgetTest(unittest.TestCase):
    def test_native_only_launch_omits_model_path_and_rejects_inherited_models(self):
        env = construction_environment("input.json", "a" * 64, 100101, "output.json", environ={})
        self.assertNotIn("POKER_NATIVE_FLOP_VALUE_MODEL", env)
        self.assertNotIn("POKER_NATIVE_FLOP_VALUE_MODEL_SHA", env)
        self.assertEqual(env["POKER_NATIVE_FLOP_TURN_ITERATIONS"], "4")
        for inherited in ({"POKER_NATIVE_FLOP_VALUE_MODEL": ""},
                          {"POKER_NATIVE_FLOP_VALUE_MODEL": "model.json"},
                          {"POKER_NATIVE_FLOP_VALUE_MODEL_SHA": "a" * 64}):
            with self.subTest(inherited=inherited), self.assertRaises(ValueError):
                construction_environment("input.json", "a" * 64, 100101, "output.json", environ=inherited)

    def test_response_override_preserves_training_policy_and_raw_artifact(self):
        raw = dict(schema="hu-native-counterfactual-turn-flop-pilot-v1",
                   iterations=32, turn_iterations=4,
                   strategies=[dict(probabilities=[.4, .6])], turn_queries=288)
        original = copy.deepcopy(raw)
        result = evaluation_candidate(raw)
        self.assertEqual(raw, original)
        self.assertEqual(result.pop("response_turn_iterations"), 64)
        self.assertEqual(result, original)
        for field, value in (("turn_iterations", 64), ("iterations", 8),
                             ("learned_leaf_model_sha256", "a" * 64),
                             ("response_turn_iterations", 4), ("strategies", [])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluation_candidate({**raw, field: value})

    def test_explicit_null_response_override_is_accepted_without_other_changes(self):
        raw = dict(schema="hu-native-counterfactual-turn-flop-pilot-v1", iterations=32,
                   turn_iterations=4, response_turn_iterations=None, strategies=[{}])
        result = evaluation_candidate(raw)
        self.assertIsNone(raw["response_turn_iterations"])
        self.assertEqual(result["response_turn_iterations"], 64)

    def test_quality_and_cost_screen_is_independent_and_fail_closed(self):
        row = dict(gainBb=.22, native64GainBb=.20, learned32GainBb=.50,
                   solveSeconds=190, native64SolveSeconds=1000)
        self.assertIsNone(case_rejection(row))
        self.assertIn("0.05bb", case_rejection({**row, "gainBb": .251}))
        self.assertIn("half", case_rejection({**row, "learned32GainBb": .23}))
        self.assertIn("fivefold", case_rejection({**row, "solveSeconds": 201}))
        for field in row:
            with self.subTest(field=field), self.assertRaises(ValueError):
                case_rejection({**row, field: math.nan})
        with self.assertRaises(ValueError):
            case_rejection({**row, "learned32GainBb": .19})

    def test_controls_require_exact_complete_paired_measurements(self):
        cases = [(dict(id=spot), dict(seed=seed), {})
                 for spot in ("limped-paired", "single-raised-high-rainbow")
                 for seed in (100101, 100102)]
        reference = dict(schema="postflop-gap-matched-leaf-pilot-v1", status="complete",
                         phase="compare", iterations=32,
                         spots=["limped-paired", "single-raised-high-rainbow"],
                         cases=[dict(spot=s["id"], seed=m["seed"], arm=arm,
                                     gainBb=.2, solveSeconds=1000)
                                for s, m, _ in cases for arm in ("native", "learned")])
        self.assertEqual(len(controls(reference, cases)), 8)
        for change in (dict(status="running"), dict(cases=reference["cases"][:-1]),
                       dict(cases=reference["cases"] + [reference["cases"][0]])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                controls({**reference, **change}, cases)

    def test_staged_screen_cannot_accept_missing_or_failed_control_evidence(self):
        construction = dict(schema="native-inner-budget-pilot-v1", status="rejected", cases=[],
                            constructionTurnIterations=4, playedTurnIterations=64,
                            projectedTotalSeconds=13000, maximumSeconds=7200,
                            costRejection=dict(spot="limped-paired", seed=100101,
                                               solveSeconds=206, native64SolveSeconds=1956))
        preflight = dict(schema="native-streamed-policy-hash-preflight-v1", status="complete", exactPacketParity=True,
                         packets=[dict(arm=arm, sha256="a"*64) for arm in ("buffered", "streamed")])
        self.assertEqual(verify_screen_inputs(construction, preflight), construction["costRejection"])
        for change in (dict(status="failed"), dict(cases=[{}]), dict(projectedTotalSeconds=math.nan),
                       dict(costRejection={**construction["costRejection"], "solveSeconds": 400})):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_screen_inputs({**construction, **change}, preflight)
        for change in (dict(status="running"), dict(exactPacketParity=False), dict(packets=[]),
                       dict(packets=[*preflight["packets"][:1], dict(arm="streamed", sha256="b"*64)])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_screen_inputs(construction, {**preflight, **change})


if __name__ == "__main__":
    unittest.main()

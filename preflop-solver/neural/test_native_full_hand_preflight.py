import copy
import math
import unittest

from run_native_full_hand_preflight import expected_route_sha256, validate_probe


def fixture():
    """Synthetic interface-only output, not a policy or quality result."""
    return dict(schema="native-full-hand-integration-probe-v1", checkpointSha256="a"*64,
        cardsSeed=881901, solveBudget=dict(flopIterations=64, trainingTurnIterations=64,
            responseTurnIterations=64, leafWorkers=4),
        preflopQueries=20000, preflopMissing=0, preflopExhaustiveCoveragePass=True,
        maximumProbabilitySumError=1e-8,
        diagnostics=dict(preflopSha256="a"*64, learnedLeafModelSha256=None,
            completeRootSupport=True, rootRealizationTurnAverages=False,
            route="research-strict-preflop-native-flop-turn-river-v1", safeResolving=False,
            releaseQualified=False, flopIterations=64, trainingTurnIterations=64,
            responseTurnIterations=64, executionLeafWorkers=4, cachedTurnSolves=1,
            routeSha256=expected_route_sha256("a"*64)),
        decisions=[dict(street=street, actor=actor, actions=["Check", "Bet"],
            probabilities=[.3, .7], seconds=1.2, forcedAction="Check")
            for street in ("preflop", "flop", "turn", "river") for actor in (0, 1)])


class NativeFullHandPreflightTest(unittest.TestCase):
    def test_actual_output_contract_retains_all_streets_without_strength_claim(self):
        value = validate_probe(fixture(), "a"*64)
        self.assertEqual(len(value["streetSeconds"]), 8)
        self.assertEqual(value["routeSha256"], expected_route_sha256("a"*64))
        self.assertFalse(value["fullGameGateEvaluated"])
        self.assertFalse(value["fullHandCoverageQualified"])
        self.assertFalse(value["releaseAccepted"])

    def test_rejects_model_budget_or_support_substitution(self):
        for key, replacement in (("learnedLeafModelSha256", "c"*64),
                ("completeRootSupport", False), ("flopIterations", 32),
                ("responseTurnIterations", 4), ("safeResolving", True),
                ("releaseQualified", True), ("executionLeafWorkers", 1), ("routeSha256", "b"*64)):
            value = fixture(); value["diagnostics"][key] = replacement
            with self.assertRaises(ValueError): validate_probe(value, "a"*64)
        with self.assertRaises(ValueError): validate_probe(fixture(), "d"*64)

    def test_rejects_partial_hand_and_invalid_probabilities(self):
        for probabilities in ([.3, .3], [math.nan, .7], [-.1, 1.1], [math.inf, 0]):
            value = fixture(); value["decisions"][2]["probabilities"] = probabilities
            with self.assertRaises(ValueError): validate_probe(value, "a"*64)
        for key, replacement in (("seconds", -.1), ("forcedAction", "Fold"), ("actions", ["Check"])):
            value = fixture(); value["decisions"][2][key] = replacement
            with self.assertRaises(ValueError): validate_probe(value, "a"*64)
        value = fixture(); value["decisions"].pop()
        with self.assertRaises(ValueError): validate_probe(value, "a"*64)

    def test_exhaustive_preflop_threshold_is_not_full_hand_coverage(self):
        value = fixture(); value["preflopMissing"] = 2
        self.assertFalse(validate_probe(value, "a"*64)["fullHandCoverageQualified"])
        for missing in (3, -1, 20001, True):
            bad = copy.deepcopy(value); bad["preflopMissing"] = missing
            with self.assertRaises(ValueError): validate_probe(bad, "a"*64)
        for error in (math.nan, math.inf, -1, .001):
            bad = fixture(); bad["maximumProbabilitySumError"] = error
            with self.assertRaises(ValueError): validate_probe(bad, "a"*64)


if __name__ == "__main__": unittest.main()

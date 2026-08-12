import copy
import unittest

import blend_range_policy_updates as module


def policy(weight: float, parent: str | None = None) -> dict:
    layer = {
        "inputSize": 1,
        "outputSize": 1,
        "activation": "linear",
        "residual": False,
        "normalization": "none",
        "normalizationEpsilon": 1e-5,
        "weights": [weight],
        "biases": [weight + 1.0],
        "normalizationWeights": [],
        "normalizationBiases": [],
    }
    payload = {
        "schema": module.NETWORK_SCHEMA,
        "architecture": "test",
        "depthBb": 20.0,
        "usesExactRanges": True,
        "featureSchema": "feature",
        "contextSize": 1,
        "querySize": 1,
        "actionFeatureSchema": "action",
        "actionFeatureSize": 1,
        "rangeAggregation": "range",
        "policyComposition": "replace",
        "seed": 1,
    }
    for tower in module.TOWER_KEYS:
        payload[tower] = [copy.deepcopy(layer)]
    if parent is not None:
        payload["parentRangePolicySha256"] = parent
    return payload


class BlendRangePolicyUpdateTests(unittest.TestCase):
    def test_blends_equal_component_deltas_and_records_provenance(self):
        source_hash = "a" * 64
        first_hash = "b" * 64
        second_hash = "c" * 64
        blended = module.blend_policy(
            policy(2.0),
            policy(4.0, source_hash),
            policy(8.0, source_hash),
            source_hash,
            first_hash,
            second_hash,
            0.5,
            0.5,
            17,
        )
        for tower in module.TOWER_KEYS:
            self.assertEqual(blended[tower][0]["weights"], [6.0])
            self.assertEqual(blended[tower][0]["biases"], [7.0])
        self.assertEqual(blended["seed"], 17)
        self.assertEqual(blended["parentRangePolicySha256"], source_hash)
        self.assertEqual(
            blended["hybridUpdateComponentSha256s"],
            [first_hash, second_hash],
        )
        self.assertEqual(blended["hybridUpdateWeights"], [0.5, 0.5])

    def test_rejects_mismatched_parent(self):
        source_hash = "a" * 64
        with self.assertRaisesRegex(ValueError, "does not pin"):
            module.blend_policy(
                policy(2.0),
                policy(4.0, "d" * 64),
                policy(8.0, source_hash),
                source_hash,
                "b" * 64,
                "c" * 64,
                0.5,
                0.5,
                17,
            )

    def test_rejects_invalid_weights(self):
        source_hash = "a" * 64
        with self.assertRaisesRegex(ValueError, "sum to one"):
            module.blend_policy(
                policy(2.0),
                policy(4.0, source_hash),
                policy(8.0, source_hash),
                source_hash,
                "b" * 64,
                "c" * 64,
                0.6,
                0.5,
                17,
            )

    def test_preserves_uniformly_absent_optional_parameters(self):
        source_hash = "a" * 64
        policies = [
            policy(2.0),
            policy(4.0, source_hash),
            policy(8.0, source_hash),
        ]
        for payload in policies:
            payload["head"][0].pop("normalizationWeights")
            payload["head"][0].pop("normalizationBiases")
        blended = module.blend_policy(
            *policies,
            source_hash,
            "b" * 64,
            "c" * 64,
            0.5,
            0.5,
            17,
        )
        self.assertNotIn("normalizationWeights", blended["head"][0])
        self.assertNotIn("normalizationBiases", blended["head"][0])

    def test_rejects_inconsistent_parameter_presence(self):
        source_hash = "a" * 64
        source = policy(2.0)
        first = policy(4.0, source_hash)
        second = policy(8.0, source_hash)
        first["head"][0].pop("normalizationWeights")
        with self.assertRaisesRegex(ValueError, "parameter presence differs"):
            module.blend_policy(
                source,
                first,
                second,
                source_hash,
                "b" * 64,
                "c" * 64,
                0.5,
                0.5,
                17,
            )

    def test_rebases_a_donor_delta_onto_an_independent_parent(self):
        target_hash = "a" * 64
        donor_hash = "b" * 64
        candidate_hash = "c" * 64
        rebased = module.rebase_policy_update(
            policy(2.0),
            policy(4.0),
            policy(7.0, donor_hash),
            target_hash,
            donor_hash,
            candidate_hash,
            0.5,
            19,
        )
        for tower in module.TOWER_KEYS:
            self.assertEqual(rebased[tower][0]["weights"], [3.5])
            self.assertEqual(rebased[tower][0]["biases"], [4.5])
        self.assertEqual(rebased["parentRangePolicySha256"], target_hash)
        self.assertEqual(rebased["rebasedUpdateDonorSourceSha256"], donor_hash)
        self.assertEqual(rebased["rebasedUpdateDonorCandidateSha256"], candidate_hash)
        self.assertEqual(rebased["rebasedUpdateWeight"], 0.5)

    def test_rebase_rejects_an_unpinned_donor_candidate(self):
        with self.assertRaisesRegex(ValueError, "does not pin"):
            module.rebase_policy_update(
                policy(2.0),
                policy(4.0),
                policy(7.0, "d" * 64),
                "a" * 64,
                "b" * 64,
                "c" * 64,
                1.0,
                19,
            )


if __name__ == "__main__":
    unittest.main()

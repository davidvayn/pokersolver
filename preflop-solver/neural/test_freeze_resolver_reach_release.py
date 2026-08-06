import json
import tempfile
import unittest
from pathlib import Path

import freeze_resolver_reach_release as module


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "neural/20bb-v49-release-evaluation-protocol.json"


class FreezeResolverReachReleaseTests(unittest.TestCase):
    def test_real_protocol_is_frozen_and_fail_closed_across_lifecycle(self) -> None:
        raw_protocol = json.loads(PROTOCOL.read_text())
        holdout_outputs = [
            module.resolved(ROOT, shard["output"])
            for shard in raw_protocol["freshAuthenticHoldout"]["shards"]
        ]
        if any(path.exists() for path in holdout_outputs):
            with self.assertRaisesRegex(ValueError, "generated before release freeze"):
                module.validate_protocol(PROTOCOL, ROOT)
        else:
            module.validate_protocol(PROTOCOL, ROOT)
        protocol, experiment, corpus = module.validate_protocol(
            PROTOCOL, ROOT, require_unopened=False
        )
        self.assertFalse(protocol["activationAllowed"])
        self.assertEqual(experiment["postSelection"]["releaseTrainingSeeds"], [15301, 15302])
        self.assertEqual(
            [
                shard["seed"]
                for shard in protocol["freshAuthenticHoldout"]["shards"]
            ],
            [15401, 15402],
        )
        self.assertEqual(len(corpus["reservedEvaluationShards"]), 2)
        self.assertEqual(
            protocol["matchedContinualResolver"][
                "maximumExploitabilityBbPerHand"
            ],
            0.05,
        )
        self.assertTrue(
            protocol["matchedContinualResolver"]["crossEvaluateBothReleaseSeeds"]
        )

    def test_holdout_seed_overlap_is_rejected(self) -> None:
        payload = json.loads(PROTOCOL.read_text())
        payload["freshAuthenticHoldout"]["shards"][0]["seed"] = 15301
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "protocol.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "overlap"):
                module.validate_protocol(path, ROOT)

    def test_combined_folds_preserve_total_selected_resolver_weight(self) -> None:
        experiment = json.loads(
            (ROOT / "neural/20bb-v49-resolver-reach-experiments.json").read_text()
        )
        candidate = next(
            entry
            for entry in experiment["candidates"]
            if entry["name"] == "protected-expanded"
        )
        supplements, weights = module.release_supplements(experiment, candidate)
        self.assertEqual(len(supplements), 7)
        self.assertEqual(weights[-2:], [1.0, 1.0])
        self.assertEqual(sum(weights[-2:]), candidate["supplementalDatasetSamplingWeights"][-1])


if __name__ == "__main__":
    unittest.main()

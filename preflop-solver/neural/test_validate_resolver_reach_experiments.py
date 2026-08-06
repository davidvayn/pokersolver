import copy
import json
import unittest
from pathlib import Path

import validate_resolver_reach_experiments as module


class ResolverReachExperimentValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = Path(__file__).parent
        cls.payload = json.loads(
            (directory / "20bb-v49-resolver-reach-experiments.json").read_text()
        )
        cls.corpus = json.loads(
            (directory / "20bb-v49-resolver-reach-corpus.json").read_text()
        )

    def test_frozen_plan_has_four_configs_and_sixteen_unique_seeds(self) -> None:
        result = module.validate_plan_structure(self.payload, self.corpus)
        self.assertEqual(result["candidateCount"], 4)
        self.assertEqual(result["crossFitTrainingSeedCount"], 16)
        self.assertEqual(result["releaseTrainingSeeds"], [15301, 15302])

    def test_rejects_reused_seed_or_nonopposite_fold(self) -> None:
        reused = copy.deepcopy(self.payload)
        reused["candidates"][1]["folds"][0]["trainingSeeds"][0] = 15201
        with self.assertRaisesRegex(ValueError, "globally unique"):
            module.validate_plan_structure(reused, self.corpus)

        leaked = copy.deepcopy(self.payload)
        leaked["candidates"][0]["folds"][0]["evaluationFold"] = "seed15101"
        with self.assertRaisesRegex(ValueError, "not opposite"):
            module.validate_plan_structure(leaked, self.corpus)

    def test_rejects_release_holdout_reuse_or_activation(self) -> None:
        holdout = copy.deepcopy(self.payload)
        holdout["commonTrainer"]["holdoutStartIndex"] = 384
        with self.assertRaisesRegex(ValueError, "opened predecessor"):
            module.validate_plan_structure(holdout, self.corpus)

        active = copy.deepcopy(self.payload)
        active["activationAllowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot activate"):
            module.validate_plan_structure(active, self.corpus)


if __name__ == "__main__":
    unittest.main()

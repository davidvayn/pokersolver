import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import train_public_value_network as module


class PublicValueNetworkTests(unittest.TestCase):
    def test_combo_card_inverse_matches_triangular_keys(self) -> None:
        keys = []
        for high in range(1, 52):
            for low in range(high):
                key = high * (high - 1) // 2 + low
                self.assertEqual(module.combo_cards(key), (high, low))
                keys.append(key)
        self.assertEqual(keys, list(range(module.COMBO_COUNT)))

    def test_state_split_is_deterministic_and_disjoint(self) -> None:
        first = module.state_split(20, 7, 0.25)
        second = module.state_split(20, 7, 0.25)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(len(first[1]), 5)

    def test_suit_permutations_are_combo_bijections(self) -> None:
        permutations = module.suit_permutations(24)
        self.assertEqual(len(permutations), 24)
        for permutation in permutations:
            mapping = module.combo_permutation(permutation)
            self.assertEqual(len(np.unique(mapping)), module.COMBO_COUNT)

    def test_loader_rejects_board_blocked_private_targets(self) -> None:
        ranges = [[0.0] * module.COMBO_COUNT for _ in range(2)]
        values = [[0.0] * module.COMBO_COUNT for _ in range(2)]
        masses = [[0.0] * module.COMBO_COUNT for _ in range(2)]
        blocked_key = 1 * (1 - 1) // 2 + 0
        ranges[0][blocked_key] = 1.0
        payload = {
            "schema": "hu-turn-public-belief-cfv-dataset-v1",
            "targets": [{
                "board": [0, 5, 10, 15],
                "invested_bb": [1.0, 1.0],
                "actor": 1,
                "ranges": ranges,
                "counterfactual_values_bb": values,
                "opponent_compatible_mass": masses,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "board-blocked"):
                module.load_dataset(path)


if __name__ == "__main__":
    unittest.main()

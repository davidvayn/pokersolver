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

    def test_three_way_split_reserves_tuning_and_untouched_holdout(self) -> None:
        train, tuning, holdout = module.three_way_state_split(64, 7, 0.25, 0.15)
        self.assertEqual(len(train), 38)
        self.assertEqual(len(tuning), 10)
        self.assertEqual(len(holdout), 16)
        self.assertEqual(len(set(train) | set(tuning) | set(holdout)), 64)
        self.assertFalse(set(train) & set(tuning))
        self.assertFalse(set(train) & set(holdout))
        self.assertFalse(set(tuning) & set(holdout))

    def test_three_way_split_can_reserve_only_new_states_for_holdout(self) -> None:
        train, tuning, holdout = module.three_way_state_split(
            128, 7, 0.25, 0.15, holdout_start_index=64
        )
        self.assertEqual(len(train), 77)
        self.assertEqual(len(tuning), 19)
        self.assertEqual(len(holdout), 32)
        self.assertTrue(np.all(holdout >= 64))
        self.assertEqual(len(set(train) | set(tuning) | set(holdout)), 128)

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

    def test_hand_class_encoding_has_exactly_169_classes(self) -> None:
        self.assertEqual(set(module.HAND_CLASS_IDS), set(range(169)))

    def test_python_evaluator_matches_rust_reference_hands(self) -> None:
        straight_flush = module.evaluate_cards([51, 47, 43, 39, 35, 0, 1])
        quads = module.evaluate_cards([48, 49, 50, 51, 47, 0, 1])
        full_house = module.evaluate_cards([48, 49, 50, 44, 45, 0, 1])
        wheel = module.evaluate_cards([48, 0, 5, 10, 15])
        six_high = module.evaluate_cards([0, 5, 10, 15, 16])
        self.assertGreater(straight_flush, quads)
        self.assertGreater(quads, full_house)
        self.assertGreater(six_high, wheel)

    def test_shared_features_are_exactly_suit_equivariant(self) -> None:
        board = np.asarray([0, 5, 10, 15], dtype=np.int16)
        rng = np.random.default_rng(41)
        ranges = rng.random((2, module.COMBO_COUNT), dtype=np.float32)
        for player in range(2):
            for combo, (first, second) in enumerate(module.COMBO_CARDS):
                if first in board or second in board:
                    ranges[player, combo] = 0.0
            ranges[player] /= ranges[player].sum()

        def compatible_masses(values: np.ndarray) -> np.ndarray:
            result = np.zeros_like(values)
            card_mass = np.zeros((2, 52), dtype=np.float32)
            for player in range(2):
                for combo, (first, second) in enumerate(module.COMBO_CARDS):
                    card_mass[player, first] += values[player, combo]
                    card_mass[player, second] += values[player, combo]
            for player in range(2):
                opponent = 1 - player
                for combo, (first, second) in enumerate(module.COMBO_CARDS):
                    result[player, combo] = (
                        1.0
                        - card_mass[opponent, first]
                        - card_mass[opponent, second]
                        + values[opponent, combo]
                    )
            return result

        masses = compatible_masses(ranges)
        context, queries = module.build_features(
            board, 1, np.asarray([3.0, 4.0], dtype=np.float32), ranges, masses
        )
        permutation = (2, 0, 3, 1)
        mapping = module.combo_permutation(permutation)
        permuted_ranges = np.zeros_like(ranges)
        permuted_masses = np.zeros_like(masses)
        permuted_ranges[:, mapping] = ranges
        permuted_masses[:, mapping] = masses
        permuted_board = np.asarray(
            [module.permute_card(int(card), permutation) for card in board], dtype=np.int16
        )
        permuted_context, permuted_queries = module.build_features(
            permuted_board,
            1,
            np.asarray([3.0, 4.0], dtype=np.float32),
            permuted_ranges,
            permuted_masses,
        )
        np.testing.assert_allclose(permuted_context, context, atol=1e-6)
        np.testing.assert_allclose(permuted_queries[:, mapping], queries, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

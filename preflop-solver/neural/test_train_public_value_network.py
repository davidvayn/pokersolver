import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import train_public_value_network as module


class PublicValueNetworkTests(unittest.TestCase):
    @staticmethod
    def synthetic_dataset(
        board: list[int],
        source_hash: str,
        status: str = "accepted",
        resolver_leaf: bool = False,
        schema: str = module.COMPLETE_TURN_TARGET_SCHEMA,
    ) -> module.Dataset:
        target = {
            "board": board,
            "maximum_river_exploitability_bb_per_hand": 0.04,
            "turn_river_exploitability_bb_per_hand": 0.04,
            "current_turn_river_exploitability_bb_per_hand": 0.05,
            "turn_river_maximum_probability_sum_error": 1e-12,
            "turn_only_best_response_gain_bb_per_hand": 0.02,
            "river_only_best_response_gain_bb_per_hand": 0.03,
            "turn_river_solver_method": (
                "value_only_paired_alternating_vectorized_dcfr_exact_private_cards_"
                "observed_river_chance_and_complete_turn_river_betting"
            ),
            "turn_river_information_sets": 100,
            "turn_information_sets": 4,
            "river_information_sets": 96,
            "exact_river_cards": 48,
            "zero_sum_residual_bb": -1e-10,
            "range_particles": 4096,
            "range_replicates": 2,
            "range_effective_sample_size": 1000.0,
            "belief_method": "exact_per-player_reach_factors_test",
            "range_maximum_total_variation": 0.10,
        }
        if resolver_leaf:
            for field in (
                "range_particles",
                "range_replicates",
                "range_effective_sample_size",
                "range_maximum_total_variation",
            ):
                target.pop(field)
            target.update(
                {
                    "belief_method": "exact_resolver_average_strategy_counterfactual_reach",
                    "resolver_leaf_reach_probability": 0.25,
                    "resolver_root_board": board[:3],
                    "resolver_public_history": ["flop_start", "check", "check"],
                }
            )
        return module.Dataset(
            boards=np.asarray([board], dtype=np.int16),
            actors=np.asarray([0], dtype=np.int8),
            invested=np.asarray([[4.0, 4.0]], dtype=np.float32),
            ranges=np.zeros((1, 2, module.COMBO_COUNT), dtype=np.float32),
            masses=np.zeros((1, 2, module.COMBO_COUNT), dtype=np.float32),
            targets=np.zeros((1, 2 * module.COMBO_COUNT), dtype=np.float32),
            target_scales=np.ones(1, dtype=np.float32),
            weights=np.ones((1, 2 * module.COMBO_COUNT), dtype=np.float32),
            projection_weights=np.ones((1, 2, module.COMBO_COUNT), dtype=np.float32),
            groups=np.asarray([0], dtype=np.int32),
            source={
                "schema": schema,
                "source_policy_sha256": "f" * 64,
                "validation": {"status": status, "reasons": []},
                "targets": [target],
            },
            source_sha256=source_hash,
        )

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

    def test_three_way_split_stratifies_every_pot_band(self) -> None:
        strata = np.asarray([0] * 203 + [1] * 31 + [2] * 22, dtype=np.int8)
        first = module.three_way_state_split(
            256, 10901, 0.25, 0.10, holdout_start_index=128, strata=strata
        )
        second = module.three_way_state_split(
            256, 10901, 0.25, 0.10, holdout_start_index=128, strata=strata
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(set(strata[left]), {0, 1, 2})
        train, tuning, holdout = first
        self.assertEqual((len(train), len(tuning), len(holdout)), (166, 26, 64))
        self.assertTrue(np.all(holdout >= 128))
        self.assertEqual(len(set(train) | set(tuning) | set(holdout)), 256)

    def test_parallel_feature_construction_matches_single_worker(self) -> None:
        dataset = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        serial = module.feature_dataset(dataset, module.FEATURE_SCHEMA, 1)
        parallel = module.feature_dataset(dataset, module.FEATURE_SCHEMA, 2)
        np.testing.assert_array_equal(parallel[0], serial[0])
        np.testing.assert_array_equal(parallel[1], serial[1])

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
            "targets": [
                {
                    "board": [0, 5, 10, 15],
                    "invested_bb": [1.0, 1.0],
                    "actor": 1,
                    "ranges": ranges,
                    "counterfactual_values_bb": values,
                    "opponent_compatible_mass": masses,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            for version in ("v1", "v2"):
                payload["schema"] = f"hu-turn-public-belief-cfv-dataset-{version}"
                path = Path(directory) / f"targets-{version}.json"
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "board-blocked"):
                    module.load_dataset(path)

    def test_hand_class_encoding_has_exactly_169_classes(self) -> None:
        self.assertEqual(set(module.HAND_CLASS_IDS), set(range(169)))

    def test_value_scales_separate_pot_from_legal_payoff_exposure(self) -> None:
        self.assertEqual(module.value_scale_bb([2.0, 2.0], "pot"), 4.0)
        self.assertEqual(module.value_scale_bb([2.0, 2.0], "payoff-exposure"), 20.0)
        self.assertEqual(module.value_scale_bb([18.0, 18.0], "pot"), 36.0)
        self.assertEqual(module.value_scale_bb([18.0, 18.0], "payoff-exposure"), 20.0)

    def test_exact_turn_range_equity_matches_direct_runout(self) -> None:
        board = np.asarray([0, 5, 10, 15], dtype=np.int16)
        hero = 24 * 23 // 2 + 20
        opponent = 33 * 32 // 2 + 29
        ranges = np.zeros((2, module.COMBO_COUNT), dtype=np.float32)
        ranges[0, hero] = 1.0
        ranges[1, opponent] = 1.0
        masses = np.zeros_like(ranges)
        masses[0, hero] = 1.0
        masses[1, opponent] = 1.0
        result = module.exact_turn_range_equities(board, ranges, masses)
        hero_cards = [int(card) for card in module.COMBO_CARDS[hero]]
        opponent_cards = [int(card) for card in module.COMBO_CARDS[opponent]]
        wins = 0.0
        rivers = 0
        blocked = set(map(int, board)) | set(hero_cards) | set(opponent_cards)
        for river in range(52):
            if river in blocked:
                continue
            hero_strength = module.evaluate_cards(
                [*map(int, board), river, *hero_cards]
            )
            opponent_strength = module.evaluate_cards(
                [*map(int, board), river, *opponent_cards]
            )
            wins += float(hero_strength > opponent_strength)
            wins += 0.5 * float(hero_strength == opponent_strength)
            rivers += 1
        self.assertEqual(rivers, 44)
        self.assertAlmostEqual(float(result[0, hero]), wins / rivers, places=6)
        self.assertAlmostEqual(
            float(result[1, opponent]), 1.0 - wins / rivers, places=6
        )

    def test_stratified_batch_draws_every_available_pot_band(self) -> None:
        invested = np.asarray(
            [[2.0, 2.0], [2.5, 2.5], [5.0, 5.0], [7.0, 7.0], [12.0, 12.0]],
            dtype=np.float32,
        )
        rows = np.arange(len(invested))
        selected = module.stratified_batch_rows(
            np.random.default_rng(9), rows, invested, 6
        )
        bands = [module.pot_band(invested[row]) for row in selected]
        self.assertEqual({0, 1, 2}, set(bands))
        self.assertEqual([bands.count(index) for index in range(3)], [2, 2, 2])

    def test_stratified_batch_honors_supplemental_sampling_weight(self) -> None:
        invested = np.asarray([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        selected = module.stratified_batch_rows(
            np.random.default_rng(11),
            np.asarray([0, 1]),
            invested,
            1000,
            np.asarray([1.0, 0.01]),
        )
        self.assertGreater(int(np.sum(selected == 0)), 950)

    def test_primary_replay_guarantees_authentic_row_in_every_pot_band(self) -> None:
        invested = np.asarray(
            [
                [2.0, 2.0],
                [5.0, 5.0],
                [12.0, 12.0],
                [2.5, 2.5],
                [6.0, 6.0],
                [14.0, 14.0],
            ],
            dtype=np.float32,
        )
        selected = module.primary_replay_batch_rows(
            np.random.default_rng(13),
            np.asarray([0, 1, 2]),
            np.asarray([3, 4, 5]),
            invested,
            6,
            0.5,
        )
        self.assertEqual(int(np.sum(selected < 3)), 3)
        bands = [module.pot_band(invested[row]) for row in selected]
        self.assertEqual([bands.count(index) for index in range(3)], [2, 2, 2])
        for band in range(3):
            self.assertTrue(
                any(
                    row < 3 and module.pot_band(invested[row]) == band
                    for row in selected
                )
            )

    def test_weighted_metrics_report_strategic_signed_bias(self) -> None:
        truth = np.zeros((1, 2 * module.COMBO_COUNT), dtype=np.float32)
        prediction = np.zeros_like(truth)
        prediction[:, : module.COMBO_COUNT] = 0.5
        prediction[:, module.COMBO_COUNT :] = -0.25
        metrics = module.weighted_metrics(
            truth,
            prediction,
            np.ones_like(truth),
            np.asarray([2.0], dtype=np.float32),
        )
        self.assertAlmostEqual(metrics["weightedMeanErrorBb"], 0.25)
        self.assertEqual(metrics["playerWeightedMeanErrorBb"], [1.0, -0.5])
        self.assertAlmostEqual(metrics["maximumAbsolutePlayerWeightedMeanErrorBb"], 1.0)

    def test_absolute_rmse_gate_requires_every_seed(self) -> None:
        variants = [
            {"metrics": {"weightedRmseBb": 0.24}},
            {"metrics": {"weightedRmseBb": 0.26}},
        ]
        self.assertFalse(module.every_seed_within_rmse(variants, 0.25))
        variants[1]["metrics"]["weightedRmseBb"] = 0.25
        self.assertTrue(module.every_seed_within_rmse(variants, 0.25))

    def test_public_board_texture_is_suit_invariant(self) -> None:
        # 9h, Th, Jh, 2c and the same ranks under a global suit permutation.
        first = module.public_board_texture([30, 34, 38, 0])
        second = module.public_board_texture([29, 33, 37, 3])
        self.assertEqual(first, second)
        self.assertEqual(first["rank"], "unpaired")
        self.assertEqual(first["suit"], "three-flush")
        self.assertEqual(first["connectivity"], "connected")

    def test_public_board_texture_distinguishes_paired_two_tone_board(self) -> None:
        texture = module.public_board_texture([48, 49, 28, 29])
        self.assertEqual(texture["rank"], "two-pair")
        self.assertEqual(texture["suit"], "two-tone")

    def test_deep_gelu_export_preserves_every_dense_layer(self) -> None:
        model = module.SharedComboValueNetwork(True, "deep-gelu", "pot")
        context = module.tower_payload(model.context_tower, "gelu-fast", "gelu-fast")
        query = module.tower_payload(model.query_tower, "gelu-fast", "gelu-fast")
        head = module.tower_payload(model.head, "gelu-fast", "linear")
        self.assertEqual(len(context), 3)
        self.assertEqual(len(query), 3)
        self.assertEqual(len(head), 3)
        self.assertTrue(all(layer["activation"] == "gelu-fast" for layer in context))
        self.assertEqual(head[-1]["activation"], "linear")

    def test_supplemental_dataset_offsets_groups_and_preserves_component_hashes(
        self,
    ) -> None:
        primary = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        supplement = self.synthetic_dataset([1, 6, 11, 16], "b" * 64)
        combined = module.combine_training_datasets(primary, [supplement])
        np.testing.assert_array_equal(combined.groups, [0, 1])
        self.assertEqual(
            combined.source["component_dataset_sha256"], ["a" * 64, "b" * 64]
        )
        self.assertEqual(combined.source["validation"]["status"], "accepted")

    def test_legacy_target_schema_is_readable_but_release_rejected(self) -> None:
        primary = self.synthetic_dataset(
            [0, 5, 10, 15],
            "a" * 64,
            schema=module.LEGACY_TARGET_SCHEMA,
        )
        supplement = self.synthetic_dataset(
            [1, 6, 11, 16],
            "b" * 64,
            schema=module.LEGACY_TARGET_SCHEMA,
        )
        combined = module.combine_training_datasets(primary, [supplement])
        self.assertEqual(combined.source["validation"]["status"], "rejected")
        self.assertTrue(
            any(
                "omits complete turn betting" in reason
                for reason in combined.source["validation"]["reasons"]
            )
        )

    def test_v2_target_without_complete_solver_provenance_is_release_rejected(
        self,
    ) -> None:
        dataset = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        dataset.source["targets"][0].pop("turn_river_solver_method")
        reasons = module.complete_turn_release_reasons(dataset.source)
        self.assertTrue(any("solver provenance" in reason for reason in reasons))

    def test_v2_target_with_single_traversal_clock_is_release_rejected(self) -> None:
        dataset = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        dataset.source["targets"][0]["turn_river_solver_method"] = (
            "value_only_alternating_vectorized_dcfr_exact_private_cards_"
            "observed_river_chance_and_complete_turn_river_betting"
        )
        reasons = module.complete_turn_release_reasons(dataset.source)
        self.assertTrue(any("paired alternating" in reason for reason in reasons))

    def test_street_attribution_cannot_exceed_full_best_response(self) -> None:
        dataset = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        dataset.source["targets"][0][
            "river_only_best_response_gain_bb_per_hand"
        ] = 0.05
        reasons = module.complete_turn_release_reasons(dataset.source)
        self.assertTrue(
            any("river-only best-response attribution" in reason for reason in reasons)
        )

    def test_supplemental_dataset_rejects_duplicate_boards_but_revalidates_small_component(
        self,
    ) -> None:
        primary = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        duplicate = self.synthetic_dataset([0, 5, 10, 15], "b" * 64)
        combined = module.combine_training_datasets(primary, [duplicate])
        self.assertEqual(combined.source["validation"]["status"], "rejected")
        self.assertTrue(
            any(
                "95% distinct" in reason
                for reason in combined.source["validation"]["reasons"]
            )
        )
        rejected = self.synthetic_dataset([1, 6, 11, 16], "c" * 64, "rejected")
        combined = module.combine_training_datasets(primary, [rejected])
        self.assertEqual(combined.source["validation"]["status"], "accepted")

    def test_exact_resolver_leaf_supplement_does_not_require_particle_diagnostics(
        self,
    ) -> None:
        primary = self.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        resolver = self.synthetic_dataset([1, 6, 11, 16], "d" * 64, resolver_leaf=True)
        combined = module.combine_training_datasets(primary, [resolver])
        self.assertEqual(combined.source["validation"]["status"], "accepted")

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
        permutation = (2, 0, 3, 1)
        mapping = module.combo_permutation(permutation)
        permuted_ranges = np.zeros_like(ranges)
        permuted_masses = np.zeros_like(masses)
        permuted_ranges[:, mapping] = ranges
        permuted_masses[:, mapping] = masses
        permuted_board = np.asarray(
            [module.permute_card(int(card), permutation) for card in board],
            dtype=np.int16,
        )
        for schema in (
            module.FEATURE_SCHEMA,
            module.FEATURE_SCHEMA_BOARD_RELATIVE,
            module.FEATURE_SCHEMA_EXACT_RUNOUT,
        ):
            context, queries = module.build_features(
                board,
                1,
                np.asarray([3.0, 4.0], dtype=np.float32),
                ranges,
                masses,
                schema,
            )
            permuted_context, permuted_queries = module.build_features(
                permuted_board,
                1,
                np.asarray([3.0, 4.0], dtype=np.float32),
                permuted_ranges,
                permuted_masses,
                schema,
            )
            np.testing.assert_allclose(permuted_context, context, atol=1e-6)
            np.testing.assert_allclose(
                permuted_queries[:, mapping], queries, atol=1e-6
            )
        self.assertEqual(
            context.shape,
            (2, module.CONTEXT_COUNT + module.CONTEXT_BOARD_RELATIVE_COUNT),
        )
        self.assertEqual(
            queries.shape,
            (
                2,
                module.COMBO_COUNT,
                module.QUERY_COUNT + module.QUERY_BOARD_RELATIVE_COUNT,
            ),
        )
        legal = masses[0] > 1e-8
        np.testing.assert_allclose(
            queries[0, legal, module.QUERY_COUNT : module.QUERY_COUNT + 9].sum(
                axis=1
            ),
            1.0,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()

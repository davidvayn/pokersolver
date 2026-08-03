import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from train import (
    ACTION_FEATURE_COUNT,
    INPUT_FEATURE_COUNT,
    MAX_POLICY_ACTIONS,
    STATE_FEATURE_COUNT,
    TEXTURE_FEATURE_OFFSET,
    ActionScorer,
    DecisionReservoir,
    ReplayReservoir,
    RunConfig,
    bootstrap_dcfr_plus_targets,
    backfill_street_file,
    backfill_legacy_replay_streets,
    expand_state_action,
    export_teacher_snapshot,
    initialize_models,
    load_optimizer,
    linear_layers,
    make_compiled_step,
    migrate_legacy_resume_state,
    parse_args,
    save_optimizer,
    scheduled_learning_rate,
    scorer_json,
    set_optimizer_learning_rate,
    stratified_replay_indices,
)


def initial_state():
    return {
        "private_cards": [48, 49],
        "board": [],
        "street": "preflop",
        "actor": 0,
        "button": 0,
        "pot_bb": 0.0,
        "stacks_bb": [19.5, 19.0],
        "street_bets_bb": [0.5, 1.0],
        "total_committed_bb": [0.5, 1.0],
        "to_call_bb": 0.5,
        "last_full_raise_bb": 1.0,
        "raise_reopened": True,
        "trajectory": [],
    }


class NeuralTrainerTests(unittest.TestCase):
    def test_cli_defaults_match_leading_20bb_profile(self):
        args = parse_args(["--run-dir", "/tmp/leading-profile", "--seed", "4501"])
        self.assertEqual(args.traversals_per_round, 400)
        self.assertEqual(args.reservoir_capacity, 100_000)
        self.assertEqual(args.hidden_sizes, "256,128")
        self.assertEqual(args.batch_size, 1_024)
        self.assertEqual(args.steps_per_round, 100)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertIsNone(args.learning_rate_final)
        self.assertIsNone(args.learning_rate_decay_start_round)
        self.assertIsNone(args.learning_rate_decay_end_round)
        self.assertEqual(args.variance_baseline_scale, 0.5)
        self.assertEqual(args.replay_street_proposal, "authentic")
        self.assertEqual(args.value_rollouts_per_action, 4)
        self.assertFalse(args.sample_turn_rivers)

    def test_v9_resume_migration_adds_only_semantic_defaults(self):
        config = RunConfig(
            schema="hu-neural-mlx-run-v14",
            depth_bb=20,
            seed=4501,
            reservoir_capacity=100_000,
            hidden_sizes=(256, 128),
            batch_size=1_024,
            learning_rate=1e-3,
            learning_rate_final=None,
            learning_rate_decay_start_round=None,
            learning_rate_decay_end_round=None,
            traversals_per_round=400,
            steps_per_round=100,
            advantage_alpha=2,
            variance_baseline_scale=0.5,
            replay_street_proposal=None,
            value_rollouts_per_action=1,
            artifact_every=10,
            preflop_runout_samples=256,
            flop_runout_samples=128,
            exact_turn_rivers=True,
            compact_serving_grid=False,
        )
        legacy_config = json.loads(json.dumps(config.__dict__))
        legacy_config["schema"] = "hu-neural-mlx-run-v9"
        for key in (
            "learning_rate_final",
            "learning_rate_decay_start_round",
            "learning_rate_decay_end_round",
            "replay_street_proposal",
            "value_rollouts_per_action",
        ):
            legacy_config.pop(key)
        migrated, changed = migrate_legacy_resume_state(
            {
                "schema": "hu-neural-mlx-run-v9",
                "config": legacy_config,
                "completed_rounds": 50,
            },
            config,
            "new-hash",
        )
        self.assertTrue(changed)
        self.assertEqual(migrated["config"], json.loads(json.dumps(config.__dict__)))
        self.assertEqual(migrated["config_hash"], "new-hash")
        self.assertEqual(migrated["completed_rounds"], 50)

    def test_v9_street_backfill_recovers_pinned_one_hot_features(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "states.f16"
            street_path = Path(directory) / "street.u8"
            source = np.memmap(
                source_path,
                dtype=np.float16,
                mode="w+",
                shape=(4, STATE_FEATURE_COUNT),
            )
            source[:] = 0
            for row, street in enumerate((3, 1, 0)):
                source[row, 104 + street] = 1
            source.flush()
            del source
            self.assertTrue(
                backfill_street_file(
                    source_path,
                    street_path,
                    4,
                    3,
                    STATE_FEATURE_COUNT,
                )
            )
            recovered = np.memmap(street_path, dtype=np.uint8, mode="r", shape=(4,))
            np.testing.assert_array_equal(recovered[:3], np.asarray([3, 1, 0]))

    def test_fresh_run_does_not_require_legacy_replay_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            backfill_legacy_replay_streets(
                Path(directory),
                {"completed_rounds": 0, "config": {"reservoir_capacity": 100}},
            )

    def test_feature_expansion_matches_pinned_initial_layout(self):
        features = expand_state_action(initial_state(), {"kind": "fold", "amount_to_bb": None}, 20)
        self.assertEqual(len(features), STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT)
        self.assertEqual(len(features), INPUT_FEATURE_COUNT)
        self.assertEqual(features[48], 1)
        self.assertEqual(features[49], 1)
        self.assertEqual(features[104], 1)
        self.assertEqual(features[108], 1)
        self.assertEqual(features[110], 1)
        self.assertAlmostEqual(features[113], 19.5 / 20)
        self.assertEqual(features[STATE_FEATURE_COUNT], 1)

    def test_feature_expansion_is_invariant_to_global_suit_permutation(self):
        state = initial_state()
        state["private_cards"] = [48, 45]
        state["board"] = [0, 5, 10]
        permute = lambda card: (card // 4) * 4 + ((card % 4 + 1) % 4)
        permuted = {
            **state,
            "private_cards": [permute(card) for card in state["private_cards"]],
            "board": [permute(card) for card in state["board"]],
        }
        action = {"kind": "call", "amount_to_bb": None}
        np.testing.assert_array_equal(
            expand_state_action(state, action, 20),
            expand_state_action(permuted, action, 20),
        )

    def test_postflop_texture_features_capture_made_hand_draw_and_board(self):
        state = initial_state()
        state.update(
            {
                "private_cards": [48, 45],
                "board": [44, 40, 36],
                "street": "flop",
            }
        )
        features = expand_state_action(state, {"kind": "check", "amount_to_bb": None}, 20)
        texture = features[TEXTURE_FEATURE_OFFSET:STATE_FEATURE_COUNT]
        self.assertEqual(texture[0], 1)
        self.assertEqual(texture[2], 1)
        self.assertEqual(texture[16], 1)
        self.assertEqual(texture[21], 1)
        self.assertEqual(texture[28], 1)
        self.assertEqual(texture[32], 1)
        self.assertEqual(texture[36], 1)
        self.assertEqual(texture[48], 1)
        self.assertEqual(texture[51], 1)
        self.assertEqual(texture[54], 1)
        self.assertEqual(texture[58], 1)
        self.assertEqual(texture[59], 1)

    def test_traversal_baseline_export_selects_only_value_mean(self):
        model = ActionScorer(INPUT_FEATURE_COUNT, (8, 4), output_size=2)
        exported = scorer_json(model, output_index=0, output_scale=20)
        self.assertEqual(exported["layers"][-1]["output_size"], 1)
        self.assertEqual(len(exported["layers"][-1]["biases"]), 1)
        raw_weight = float(np.asarray(linear_layers(model)[-1].weight)[0, 0])
        self.assertAlmostEqual(exported["layers"][-1]["weights"][0], raw_weight * 20)

    def test_texture_columns_start_neutral_but_remain_trainable(self):
        config = RunConfig(
            schema="test",
            depth_bb=20,
            seed=17,
            reservoir_capacity=100,
            hidden_sizes=(8, 4),
            batch_size=8,
            learning_rate=1e-3,
            learning_rate_final=None,
            learning_rate_decay_start_round=None,
            learning_rate_decay_end_round=None,
            traversals_per_round=4,
            steps_per_round=1,
            advantage_alpha=2,
            variance_baseline_scale=0.5,
            replay_street_proposal=(0.25, 0.25, 0.25, 0.25),
            value_rollouts_per_action=1,
            artifact_every=1,
            preflop_runout_samples=4,
            flop_runout_samples=4,
            exact_turn_rivers=False,
            compact_serving_grid=False,
        )
        models, _ = initialize_models(config)
        for model in models.values():
            weights = np.asarray(linear_layers(model)[0].weight)
            self.assertTrue(np.all(weights[:, TEXTURE_FEATURE_OFFSET:STATE_FEATURE_COUNT] == 0))
            self.assertTrue(np.any(weights[:, :TEXTURE_FEATURE_OFFSET] != 0))

    def test_sparse_teacher_snapshot_is_hashed_at_artifact_rounds(self):
        config = RunConfig(
            schema="test",
            depth_bb=20,
            seed=17,
            reservoir_capacity=100,
            hidden_sizes=(8, 4),
            batch_size=8,
            learning_rate=1e-3,
            learning_rate_final=None,
            learning_rate_decay_start_round=None,
            learning_rate_decay_end_round=None,
            traversals_per_round=4,
            steps_per_round=1,
            advantage_alpha=2,
            variance_baseline_scale=0.5,
            replay_street_proposal=None,
            value_rollouts_per_action=2,
            artifact_every=1,
            preflop_runout_samples=4,
            flop_runout_samples=4,
            exact_turn_rivers=False,
            compact_serving_grid=False,
        )
        models, _ = initialize_models(config)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = export_teacher_snapshot(
                models, Path(directory), config, 3
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "hu-sparse-sd-cfr-teacher-v1")
            self.assertEqual(manifest["completedTraversals"], 12)
            self.assertEqual(manifest["strategyWeight"], 144.0)
            for descriptor in manifest["models"].values():
                payload = (Path(directory) / descriptor["file"]).read_bytes()
                self.assertEqual(descriptor["bytes"], len(payload))
                self.assertEqual(
                    descriptor["sha256"], hashlib.sha256(payload).hexdigest()
                )

    def test_learning_rate_decay_preserves_boundary_then_interpolates(self):
        config = RunConfig(
            schema="test",
            depth_bb=20,
            seed=17,
            reservoir_capacity=100,
            hidden_sizes=(8, 4),
            batch_size=8,
            learning_rate=1e-3,
            learning_rate_final=3e-4,
            learning_rate_decay_start_round=10,
            learning_rate_decay_end_round=25,
            traversals_per_round=4,
            steps_per_round=1,
            advantage_alpha=2,
            variance_baseline_scale=0.5,
            replay_street_proposal=None,
            value_rollouts_per_action=1,
            artifact_every=1,
            preflop_runout_samples=4,
            flop_runout_samples=4,
            exact_turn_rivers=False,
            compact_serving_grid=False,
        )
        self.assertEqual(scheduled_learning_rate(config, 10), 1e-3)
        self.assertAlmostEqual(scheduled_learning_rate(config, 11), 0.0009533333333333334)
        self.assertAlmostEqual(scheduled_learning_rate(config, 20), 0.0005333333333333334)
        self.assertEqual(scheduled_learning_rate(config, 25), 3e-4)

        _, optimizers = initialize_models(config)
        set_optimizer_learning_rate(optimizers, scheduled_learning_rate(config, 20))
        for optimizer in optimizers.values():
            self.assertAlmostEqual(float(optimizer.learning_rate), 0.0005333333)

    def test_reservoir_never_exceeds_capacity(self):
        import random

        with tempfile.TemporaryDirectory() as directory:
            reservoir = ReplayReservoir(Path(directory), "test", 4, 1)
            rng = random.Random(7)
            for index in range(100):
                reservoir.add(
                    np.full(INPUT_FEATURE_COUNT, index, dtype=np.float32),
                    np.asarray([index], dtype=np.float32),
                    1.0,
                    index % 4,
                    rng,
                )
            self.assertEqual(reservoir.size, 4)
            self.assertEqual(reservoir.seen, 100)

    def test_policy_reservoir_preserves_complete_decisions(self):
        import random

        with tempfile.TemporaryDirectory() as directory:
            reservoir = DecisionReservoir(
                Path(directory),
                "policy",
                4,
                normalize_targets=True,
            )
            state = np.zeros(STATE_FEATURE_COUNT, dtype=np.float32)
            actions = np.zeros((3, ACTION_FEATURE_COUNT), dtype=np.float32)
            actions[:, :3] = np.eye(3, dtype=np.float32)
            reservoir.add(
                state,
                actions,
                np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
                1.0,
                1,
                random.Random(9),
            )
            features, targets, masks, _ = reservoir.sample(
                2,
                np.random.default_rng(4),
                (0.25, 0.25, 0.25, 0.25),
            )
            self.assertEqual(features.shape, (2, MAX_POLICY_ACTIONS, INPUT_FEATURE_COUNT))
            self.assertTrue(np.allclose(np.asarray(targets)[:, :3], [0.2, 0.3, 0.5]))
            self.assertTrue(np.all(np.asarray(masks)[:, :3] == 1))
            self.assertTrue(np.all(np.asarray(masks)[:, 3:] == 0))

    def test_decision_reservoir_clear_reuses_storage_for_current_round(self):
        import random

        with tempfile.TemporaryDirectory() as directory:
            reservoir = DecisionReservoir(
                Path(directory),
                "advantage",
                4,
                normalize_targets=False,
            )
            state = np.zeros(STATE_FEATURE_COUNT, dtype=np.float32)
            actions = np.zeros((2, ACTION_FEATURE_COUNT), dtype=np.float32)
            reservoir.add(
                state,
                actions,
                np.asarray([1.0, -1.0]),
                1.0,
                0,
                random.Random(3),
            )
            reservoir.clear()
            self.assertEqual(reservoir.size, 0)
            self.assertEqual(reservoir.seen, 0)
            reservoir.add(
                state,
                actions,
                np.asarray([0.5, -0.5]),
                1.0,
                3,
                random.Random(4),
            )
            self.assertEqual(reservoir.size, 1)
            self.assertEqual(reservoir.seen, 1)

    def test_stratified_replay_is_deterministic_and_importance_correct(self):
        streets = np.asarray([0] * 40 + [1] * 30 + [2] * 20 + [3] * 10, dtype=np.uint8)
        proposal = (0.25, 0.25, 0.25, 0.25)
        first_indices, first_corrections = stratified_replay_indices(
            streets,
            len(streets),
            1000,
            proposal,
            np.random.default_rng(23),
        )
        second_indices, second_corrections = stratified_replay_indices(
            streets,
            len(streets),
            1000,
            proposal,
            np.random.default_rng(23),
        )
        np.testing.assert_array_equal(first_indices, second_indices)
        np.testing.assert_array_equal(first_corrections, second_corrections)
        sampled_streets = streets[first_indices]
        self.assertEqual(np.bincount(sampled_streets, minlength=4).tolist(), [250] * 4)
        for street, expected_probability in enumerate((0.4, 0.3, 0.2, 0.1)):
            estimate = np.mean((sampled_streets == street) * first_corrections)
            self.assertAlmostEqual(float(estimate), expected_probability)

    def test_deep_dcfr_plus_bootstrap_clips_and_discounts_prior_targets(self):
        prior = np.asarray([2.0, -3.0, 0.5], dtype=np.float32)
        instantaneous = np.asarray([-0.25, 0.75, 0.25], dtype=np.float32)
        np.testing.assert_allclose(
            bootstrap_dcfr_plus_targets(prior, instantaneous, 1, 2.0),
            instantaneous,
        )
        np.testing.assert_allclose(
            bootstrap_dcfr_plus_targets(prior, instantaneous, 3, 2.0),
            np.asarray([1.35, 0.75, 0.65], dtype=np.float32),
        )

    def test_compiled_step_and_optimizer_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            mx.random.seed(11)
            model = ActionScorer(INPUT_FEATURE_COUNT, (8, 4))
            optimizer = optim.AdamW(learning_rate=1e-3)
            optimizer.init(model.trainable_parameters())
            step = make_compiled_step(model, optimizer)
            x = mx.zeros((2, INPUT_FEATURE_COUNT))
            y = mx.ones((2, 1))
            w = mx.ones((2, 1))
            loss = step(x, y, w)
            mx.eval(loss, model.parameters(), optimizer.state)
            path = Path(directory) / "optimizer.npz"
            save_optimizer(optimizer, path)
            restored = optim.AdamW(learning_rate=1e-3)
            restored.init(model.trainable_parameters())
            load_optimizer(restored, path)
            self.assertEqual(int(restored.state["step"].item()), 1)


if __name__ == "__main__":
    unittest.main()

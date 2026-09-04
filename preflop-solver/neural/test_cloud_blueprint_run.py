import argparse
import gzip
import hashlib
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from cloud_blueprint_run import (
    SeedRun,
    aggregate_validated_summaries,
    artifact_prefix_identity,
    build_command,
    file_sha256,
    load_parent_stage,
    missing_run_outputs,
    parse_seeds,
    run_fingerprint,
    stale_generated_outputs,
    stream_gzip_and_sha256,
    validate_artifact_summary_identity,
    validate_numeric_options,
    validate_summary,
)


RANKS = "23456789TJQKA"


def root_strategies(fold_probability: float = 0.4) -> list[dict[str, object]]:
    rows = []
    for high, high_rank in enumerate(RANKS):
        rows.append(
            {
                "hand": high_rank * 2,
                "regretUpdates": 600,
                "averageVisits": 500,
                "trainedAverage": True,
                "actions": [
                    {"action": "fold", "probability": fold_probability},
                    {"action": "limp", "probability": 1.0 - fold_probability},
                ],
            }
        )
        for low_rank in RANKS[:high]:
            for suffix in ("s", "o"):
                rows.append(
                    {
                        "hand": f"{high_rank}{low_rank}{suffix}",
                        "regretUpdates": 600,
                        "averageVisits": 500,
                        "trainedAverage": True,
                        "actions": [
                            {"action": "fold", "probability": fold_probability},
                            {
                                "action": "limp",
                                "probability": 1.0 - fold_probability,
                            },
                        ],
                    }
                )
    return rows


def arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        binary=root / "preflop-solver",
        resume_from_dir=None,
        evaluation_only=False,
        depth=20.0,
        iterations=400_000,
        seeds=[26_001, 26_002],
        max_concurrent=2,
        max_information_sets=15_000_000,
        averaging_delay=40_000,
        checkpoint_every=100_000,
        held_out_deals=10_000,
        root_deviation_samples=100,
        action_value_deals=10_000,
        dcfr_alpha=1.5,
        dcfr_beta=0.0,
        dcfr_gamma=2.0,
        hs_dcfr_30_horizon=0,
        compact_serving_grid=False,
        public_chance_sampling=False,
        integrate_terminal_actions=False,
        opponent_checkdown_baseline=False,
        export_postflop_strategies=True,
        bytes_per_information_set=2_300,
        minimum_free_disk_gb=20.0,
    )


def seed_run(root: Path) -> SeedRun:
    return SeedRun(
        seed=26_001,
        held_out_seed=126_001,
        root_deviation_seed=226_001,
        action_value_seed=326_001,
        output=root / "artifact.json.gz",
        checkpoint=root / "checkpoint.json.gz",
        summary=root / "summary.json",
        log=root / "run.log",
    )


class CloudBlueprintRunTests(unittest.TestCase):
    def test_seed_parser_requires_independent_seeds(self) -> None:
        self.assertEqual(parse_seeds("26001, 26002"), [26_001, 26_002])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seeds("26001")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seeds("26001,26001")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seeds("-1,26002")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seeds("seed-a,seed-b")

    def test_numeric_preflight_rejects_invalid_cloud_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory))
            validate_numeric_options(args)
            for field, invalid, message in (
                ("depth", float("nan"), "depth"),
                ("depth", 1.0, "big blind"),
                ("iterations", 2**64, "u64 domain"),
                ("max_information_sets", 2**64, "usize domain"),
                ("averaging_delay", 400_000, "averaging delay"),
                ("checkpoint_every", 0, "checkpoint cadence"),
                ("checkpoint_every", 2**64, "u64 domain"),
                ("held_out_deals", 0, "evaluation counts"),
                ("held_out_deals", 2**64, "u64 domain"),
                ("bytes_per_information_set", 0, "evaluation counts"),
                ("minimum_free_disk_gb", -1.0, "minimum free disk"),
                ("dcfr_gamma", float("inf"), "DCFR exponents"),
                ("dcfr_beta", -1.0, "non-negative"),
            ):
                original = getattr(args, field)
                setattr(args, field, invalid)
                with self.assertRaisesRegex(SystemExit, message):
                    validate_numeric_options(args)
                setattr(args, field, original)
            args.hs_dcfr_30_horizon = 10_000_001
            with self.assertRaisesRegex(SystemExit, "may not exceed"):
                validate_numeric_options(args)
            args.hs_dcfr_30_horizon = 600_000
            args.dcfr_gamma = 1.0
            with self.assertRaisesRegex(SystemExit, "default base"):
                validate_numeric_options(args)

    def test_fingerprint_ignores_scheduling_but_pins_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory))
            expected = run_fingerprint(args, "a" * 64)
            args.max_concurrent = 1
            self.assertEqual(run_fingerprint(args, "a" * 64), expected)
            args.dcfr_gamma = 1.0
            self.assertNotEqual(run_fingerprint(args, "a" * 64), expected)
            args.dcfr_gamma = 2.0
            args.hs_dcfr_30_horizon = 600_000
            self.assertNotEqual(run_fingerprint(args, "a" * 64), expected)
            args.hs_dcfr_30_horizon = 0
            args.public_chance_sampling = True
            self.assertNotEqual(run_fingerprint(args, "a" * 64), expected)
            args.public_chance_sampling = False
            for field in ("integrate_terminal_actions", "opponent_checkdown_baseline"):
                setattr(args, field, True)
                self.assertNotEqual(run_fingerprint(args, "a" * 64), expected)
                setattr(args, field, False)
            args.evaluation_only = True
            self.assertNotEqual(run_fingerprint(args, "a" * 64), expected)
            args.evaluation_only = False
            self.assertNotEqual(run_fingerprint(args, "b" * 64), expected)
            self.assertNotEqual(run_fingerprint(args, "a" * 64, "p" * 64), expected)

    def test_resume_command_keeps_artifact_inputs_and_adds_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            run = seed_run(root)
            fresh = build_command(args, run)
            self.assertNotIn("--resume", fresh)
            self.assertIn("--export-postflop-strategies", fresh)
            run.checkpoint.write_bytes(b"checkpoint")
            resumed = build_command(args, run)
            index = resumed.index("--resume")
            self.assertEqual(resumed[index + 1], str(run.checkpoint))
            self.assertEqual(resumed[:index] + resumed[index + 2 :], fresh)
            args.hs_dcfr_30_horizon = 600_000
            scheduled = build_command(args, seed_run(root / "scheduled"))
            schedule_index = scheduled.index("--hs-dcfr-30-horizon")
            self.assertEqual(scheduled[schedule_index + 1], "600000")
            args.public_chance_sampling = True
            self.assertIn(
                "--public-chance-sampling",
                build_command(args, seed_run(root / "public-chance")),
            )
            for field, flag in (
                ("integrate_terminal_actions", "--integrate-terminal-actions"),
                ("opponent_checkdown_baseline", "--opponent-checkdown-baseline"),
            ):
                setattr(args, field, True)
                self.assertIn(flag, build_command(args, seed_run(root / field)))
                setattr(args, field, False)

    def test_opponent_variance_modes_require_pcs_and_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory))
            for field in ("integrate_terminal_actions", "opponent_checkdown_baseline"):
                setattr(args, field, True)
                with self.assertRaisesRegex(SystemExit, "requires public-chance"):
                    validate_numeric_options(args)
                args.public_chance_sampling = True
                validate_numeric_options(args)
                setattr(args, field, False)
                args.public_chance_sampling = False
            args.public_chance_sampling = True
            args.integrate_terminal_actions = args.opponent_checkdown_baseline = True
            with self.assertRaisesRegex(SystemExit, "choose only one"):
                validate_numeric_options(args)

    def test_extension_pins_parent_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            args = arguments(root / "next")
            args.resume_from_dir = parent
            args.binary.parent.mkdir(parents=True)
            args.binary.write_bytes(b"solver-binary")
            binary_hash = file_sha256(args.binary)
            commands = {}
            seed_records = {}
            for seed in args.seeds:
                prefix = f"hu-20bb-schema-v3-seed{seed}"
                checkpoint = parent / f"{prefix}.checkpoint.json.gz"
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
                commands[str(seed)] = [
                    str(args.binary),
                    "blueprint",
                    "--effective-stack-bb",
                    "20.0",
                    "--seed",
                    str(seed),
                    "--averaging-delay",
                    "40000",
                    "--dcfr-alpha",
                    "1.5",
                    "--dcfr-beta",
                    "0.0",
                    "--dcfr-gamma",
                    "2.0",
                ]
                seed_records[str(seed)] = {
                    "status": "complete",
                    "checkpointCompressedBytes": checkpoint.stat().st_size,
                    "checkpointSha256": file_sha256(checkpoint),
                }
            (parent / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "hu-schema-v3-cloud-blueprint-run-v1",
                        "status": "complete",
                        "binarySha256": binary_hash,
                        "depthBb": 20.0,
                        "iterations": 100_000,
                        "runFingerprint": "f" * 64,
                        "commands": commands,
                        "seeds": seed_records,
                    }
                )
            )
            fingerprint, checkpoints = load_parent_stage(args, binary_hash)
            self.assertEqual(fingerprint, "f" * 64)
            self.assertEqual(set(checkpoints), set(args.seeds))
            for field, flag in (
                ("integrate_terminal_actions", "--integrate-terminal-actions"),
                ("opponent_checkdown_baseline", "--opponent-checkdown-baseline"),
            ):
                setattr(args, field, True)
                with self.assertRaisesRegex(SystemExit, flag):
                    load_parent_stage(args, binary_hash)
                setattr(args, field, False)
            extension = replace(
                seed_run(root / "extension"), resume_source=checkpoints[26_001]
            )
            parent_command = build_command(args, extension)
            parent_index = parent_command.index("--resume")
            self.assertEqual(parent_command[parent_index + 1], str(checkpoints[26_001]))
            preferred = parent / "hu-20bb-schema-v3-seed26001.checkpoint.msgpack.gz"
            preferred.write_bytes(b"unrelated-new-format-checkpoint")
            _, coexistence_checkpoints = load_parent_stage(args, binary_hash)
            self.assertEqual(coexistence_checkpoints[26_001], checkpoints[26_001])
            extension.checkpoint.parent.mkdir()
            extension.checkpoint.write_bytes(b"newer-stage-checkpoint")
            local_command = build_command(args, extension)
            local_index = local_command.index("--resume")
            self.assertEqual(local_command[local_index + 1], str(extension.checkpoint))
            checkpoints[26_001].write_bytes(b"changed")
            with self.assertRaisesRegex(SystemExit, "checkpoint size changed"):
                load_parent_stage(args, binary_hash)

    def test_evaluation_only_accepts_same_iteration_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            args = arguments(root / "evaluation")
            args.resume_from_dir = parent
            args.evaluation_only = True
            args.binary.parent.mkdir(parents=True)
            args.binary.write_bytes(b"solver-binary")
            binary_hash = file_sha256(args.binary)
            commands = {}
            seed_records = {}
            for seed in args.seeds:
                prefix = f"hu-20bb-schema-v3-seed{seed}"
                checkpoint = parent / f"{prefix}.checkpoint.json.gz"
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
                commands[str(seed)] = [
                    str(args.binary),
                    "blueprint",
                    "--effective-stack-bb",
                    "20.0",
                    "--seed",
                    str(seed),
                    "--averaging-delay",
                    "40000",
                    "--dcfr-alpha",
                    "1.5",
                    "--dcfr-beta",
                    "0.0",
                    "--dcfr-gamma",
                    "2.0",
                ]
                seed_records[str(seed)] = {
                    "status": "complete",
                    "checkpointCompressedBytes": checkpoint.stat().st_size,
                    "checkpointSha256": file_sha256(checkpoint),
                }
            (parent / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "hu-schema-v3-cloud-blueprint-run-v1",
                        "status": "complete",
                        "binarySha256": binary_hash,
                        "depthBb": 20.0,
                        "iterations": args.iterations,
                        "runFingerprint": "f" * 64,
                        "commands": commands,
                        "seeds": seed_records,
                    }
                )
            )
            fingerprint, checkpoints = load_parent_stage(args, binary_hash)
            self.assertEqual(fingerprint, "f" * 64)
            self.assertEqual(set(checkpoints), set(args.seeds))
            args.evaluation_only = False
            with self.assertRaisesRegex(SystemExit, "target must exceed"):
                load_parent_stage(args, binary_hash)

    def test_streaming_integrity_hashes_decompressed_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json.gz"
            canonical = b'{"schema":1,"value":"deterministic"}'
            with gzip.open(path, "wb") as output:
                output.write(canonical)
            size, digest = stream_gzip_and_sha256(path)
            self.assertEqual(size, len(canonical))
            self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())

    def test_streaming_artifact_identity_is_bound_to_the_compact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json.gz"
            artifact = {
                "schema_version": 1,
                "solver_version": "0.1.0",
                "artifact_id": "hu-blueprint-20bb-i2-s1-deadbeefdeadbeef",
                "config_hash": "deadbeefdeadbeef",
                "training_config_hash": "0123456789abcdef",
                "model": "hu-abstracted-external-sampling-dcfr-trajectory-v3",
                "approximate": True,
                "strategies": [{"payload": "x" * 300_000}],
            }
            with gzip.open(path, "wt") as output:
                json.dump(artifact, output, separators=(",", ":"))
            summary = {
                "solverVersion": "0.1.0",
                "artifactId": artifact["artifact_id"],
                "configHash": artifact["config_hash"],
                "trainingConfigHash": artifact["training_config_hash"],
                "model": artifact["model"],
            }

            identity = artifact_prefix_identity(path)
            validate_artifact_summary_identity(identity, summary)
            summary["artifactId"] = "swapped-artifact"
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_artifact_summary_identity(identity, summary)

    def test_missing_outputs_are_reported_before_integrity_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = seed_run(root)
            self.assertEqual(
                missing_run_outputs(run),
                [run.output, run.checkpoint, run.summary],
            )
            run.output.write_bytes(b"artifact")
            run.checkpoint.write_bytes(b"checkpoint")
            run.summary.write_bytes(b"summary")
            self.assertEqual(missing_run_outputs(run), [])

    def test_generated_outputs_must_be_fresh_but_checkpoint_may_be_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = seed_run(Path(directory))
            for path in (run.output, run.checkpoint, run.summary):
                path.write_bytes(path.name.encode())
            old_ns = time.time_ns() - 2_000_000_000
            for path in (run.output, run.checkpoint, run.summary):
                os.utime(path, ns=(old_ns, old_ns))
            launch_ns = time.time_ns()

            self.assertEqual(
                stale_generated_outputs(run, launch_ns),
                [run.output, run.summary],
            )
            run.output.write_bytes(b"new artifact")
            run.summary.write_bytes(b"new summary")
            self.assertEqual(stale_generated_outputs(run, launch_ns), [])

    def test_summary_identity_and_quality_metrics_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory))
            summary = {
                "schema": "hu-blueprint-run-summary-v1",
                "traversal": "external_sampling",
                "model": "hu-abstracted-external-sampling-dcfr-trajectory-v3",
                "solverVersion": "0.1.0",
                "artifactId": "hu-blueprint-20bb-i400000-s26001-fixture",
                "seed": 26_001,
                "depthBb": 20.0,
                "averagingDelay": 40_000,
                "dcfr": {
                    "positive_regret_exponent": 1.5,
                    "negative_regret_exponent": 0.0,
                    "strategy_exponent": 2.0,
                },
                "dcfrSchedule": "fixed",
                "dcfrScheduleHorizon": 0,
                "requestedIterations": 400_000,
                "trainingIterations": 400_000,
                "stoppedEarly": False,
                "heldOutDeals": 10_000,
                "rootLocalDeviationSamplesPerClass": 100,
                "actionValueDeals": 10_000,
                "configHash": "0123456789abcdef",
                "trainingConfigHash": "fedcba9876543210",
                "heldOutButtonMeanNetBb": 0.0,
                "heldOutButtonNetStandardErrorBb": 0.01,
                "heldOutUnknownInformationSetFraction": 0.04,
                "heldOutUntrainedInformationSetFraction": 0.02,
                "rootLocalDeviationGainBb": 0.25,
                "rootLocalDeviationStandardErrorBb": 0.01,
                "rootLocalDeviation99PctLowerBoundBb": 0.22,
                "rootTrainedComboFraction": 1.0,
                "rootContinuationUnknownInformationSetFraction": 0.0,
                "rootContinuationUntrainedInformationSetFraction": 0.0,
                "actionValueStandardErrorCoverage": 0.95,
                "informationSets": 1_000_000,
                "preflopInformationSets": 100_000,
                "postflopInformationSets": 900_000,
                "trainedInformationSets": 750_000,
                "exportedInformationSets": 750_000,
                "actionValueEvaluatedInformationSets": 700_000,
                "actionValueExportedInformationSetCoverage": 0.95,
                "rootStrategies": root_strategies(),
                "validationStatus": "advisory_only",
                "validationReasons": ["independent validation still required"],
            }
            self.assertIs(validate_summary(summary, args, 26_001), summary)
            for field, summary_field in (
                ("integrate_terminal_actions", "integrateTerminalActions"),
                ("opponent_checkdown_baseline", "opponentCheckdownBaseline"),
            ):
                setattr(args, field, True)
                with self.assertRaisesRegex(ValueError, summary_field):
                    validate_summary(summary, args, 26_001)
                summary[summary_field] = True
                self.assertIs(validate_summary(summary, args, 26_001), summary)
                setattr(args, field, False)
                summary[summary_field] = False
            args.hs_dcfr_30_horizon = 600_000
            with self.assertRaisesRegex(ValueError, "dcfrSchedule"):
                validate_summary(summary, args, 26_001)
            summary["dcfrSchedule"] = "hs30"
            summary["dcfrScheduleHorizon"] = 600_000
            self.assertIs(validate_summary(summary, args, 26_001), summary)
            args.hs_dcfr_30_horizon = 0
            summary["dcfrSchedule"] = "fixed"
            summary["dcfrScheduleHorizon"] = 0
            summary["rootLocalDeviationGainBb"] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_summary(summary, args, 26_001)
            summary["rootLocalDeviationGainBb"] = 0.25
            summary["heldOutButtonNetStandardErrorBb"] = -0.01
            with self.assertRaisesRegex(ValueError, "negative"):
                validate_summary(summary, args, 26_001)
            summary["heldOutButtonNetStandardErrorBb"] = 0.01
            summary["actionValueStandardErrorCoverage"] = 1.01
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                validate_summary(summary, args, 26_001)
            summary["actionValueStandardErrorCoverage"] = 0.95
            summary["heldOutUnknownInformationSetFraction"] = 1.01
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                validate_summary(summary, args, 26_001)
            summary["heldOutUnknownInformationSetFraction"] = 0.04
            summary["rootStrategies"][0]["actions"][0]["probability"] = 0.5
            with self.assertRaisesRegex(ValueError, "probabilities sum"):
                validate_summary(summary, args, 26_001)

    def test_summary_fails_closed_on_incomplete_full_export_and_bad_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory))
            summary = {
                "schema": "hu-blueprint-run-summary-v1",
                "traversal": "external_sampling",
                "model": "hu-abstracted-external-sampling-dcfr-trajectory-v3",
                "solverVersion": "0.1.0",
                "artifactId": "hu-blueprint-20bb-i400000-s26001-fixture",
                "seed": 26_001,
                "depthBb": 20.0,
                "averagingDelay": 40_000,
                "dcfr": {
                    "positive_regret_exponent": 1.5,
                    "negative_regret_exponent": 0.0,
                    "strategy_exponent": 2.0,
                },
                "dcfrSchedule": "fixed",
                "dcfrScheduleHorizon": 0,
                "requestedIterations": 400_000,
                "trainingIterations": 400_000,
                "stoppedEarly": False,
                "heldOutDeals": 10_000,
                "rootLocalDeviationSamplesPerClass": 100,
                "actionValueDeals": 10_000,
                "configHash": "0123456789abcdef",
                "trainingConfigHash": "fedcba9876543210",
                "heldOutButtonMeanNetBb": 0.0,
                "heldOutButtonNetStandardErrorBb": 0.01,
                "heldOutUnknownInformationSetFraction": 0.04,
                "heldOutUntrainedInformationSetFraction": 0.02,
                "rootLocalDeviationGainBb": 0.25,
                "rootLocalDeviationStandardErrorBb": 0.01,
                "rootLocalDeviation99PctLowerBoundBb": 0.22,
                "rootTrainedComboFraction": 1.0,
                "rootContinuationUnknownInformationSetFraction": 0.0,
                "rootContinuationUntrainedInformationSetFraction": 0.0,
                "actionValueStandardErrorCoverage": 0.95,
                "actionValueExportedInformationSetCoverage": 0.95,
                "actionValueEvaluatedInformationSets": 700_000,
                "informationSets": 1_000_000,
                "preflopInformationSets": 100_000,
                "postflopInformationSets": 900_000,
                "trainedInformationSets": 750_000,
                "exportedInformationSets": 700_000,
                "rootStrategies": root_strategies(),
                "validationStatus": "advisory_only",
                "validationReasons": ["independent validation still required"],
            }

            with self.assertRaisesRegex(ValueError, "full postflop export"):
                validate_summary(summary, args, 26_001)
            summary["exportedInformationSets"] = 750_000
            summary["postflopInformationSets"] = 899_999
            with self.assertRaisesRegex(ValueError, "street counts"):
                validate_summary(summary, args, 26_001)
            summary["postflopInformationSets"] = 900_000
            summary["actionValueExportedInformationSetCoverage"] = 1.01
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                validate_summary(summary, args, 26_001)

    def test_aggregate_summary_requires_complete_validated_seed_records(self) -> None:
        first = {
            "rootLocalDeviationGainBb": 0.4,
            "actionValueStandardErrorCoverage": 0.96,
            "heldOutUnknownInformationSetFraction": 0.08,
            "heldOutUntrainedInformationSetFraction": 0.01,
            "rootContinuationUnknownInformationSetFraction": 0.1,
            "rootContinuationUntrainedInformationSetFraction": 0.02,
            "informationSets": 10,
            "rootStrategies": root_strategies(),
        }
        second = {
            "rootLocalDeviationGainBb": 0.5,
            "actionValueStandardErrorCoverage": 0.94,
            "heldOutUnknownInformationSetFraction": 0.12,
            "heldOutUntrainedInformationSetFraction": 0.02,
            "rootContinuationUnknownInformationSetFraction": 0.2,
            "rootContinuationUntrainedInformationSetFraction": 0.03,
            "informationSets": 12,
            "rootStrategies": root_strategies(),
        }
        records = {
            "1": {"status": "complete", "blueprintSummary": first},
            "2": {"status": "complete", "blueprintSummary": second},
        }
        self.assertEqual(
            aggregate_validated_summaries(records),
            {
                "seedCount": 2,
                "rootLocalDeviationGainMeanBb": 0.45,
                "rootLocalDeviationGainSpreadBb": 0.09999999999999998,
                "minimumActionValueStandardErrorCoverage": 0.94,
                "maximumHeldOutUnknownInformationSetFraction": 0.12,
                "maximumHeldOutUntrainedInformationSetFraction": 0.02,
                "maximumRootContinuationUnknownInformationSetFraction": 0.2,
                "maximumRootContinuationUntrainedInformationSetFraction": 0.03,
                "maximumInformationSets": 12,
                "crossSeedRootPolicyStability": {
                    "available": True,
                    "rootHandClassesPerSeed": [169, 169],
                    "maximumProbabilitySumError": 0.0,
                    "maximumAggregateActionFrequencyDelta": 0.0,
                    "maximumComboWeightedPerActionMae": 0.0,
                    "minimumPrimaryActionAgreement": 1.0,
                    "minimumComboWeightedPrimaryActionAgreement": 1.0,
                    "thresholds": {
                        "maximumAggregateActionFrequencyDelta": 0.03,
                        "maximumComboWeightedPerActionMae": 0.05,
                        "minimumPrimaryActionAgreement": 0.85,
                        "maximumProbabilitySumError": 1e-6,
                    },
                    "gates": {
                        "completeRootCoverage": True,
                        "compatibleActionSets": True,
                        "validProbabilitySums": True,
                        "aggregateActionFrequencyDelta": True,
                        "comboWeightedPerActionMae": True,
                        "primaryActionAgreement": True,
                    },
                    "passed": True,
                },
            },
        )
        records["2"]["status"] = "failed"
        self.assertIsNone(aggregate_validated_summaries(records))

    def test_cross_seed_policy_stability_enforces_action_frequency_gates(self) -> None:
        def summary(fold_probability: float) -> dict[str, object]:
            return {
                "rootLocalDeviationGainBb": 0.45,
                "actionValueStandardErrorCoverage": 0.95,
                "heldOutUnknownInformationSetFraction": 0.1,
                "heldOutUntrainedInformationSetFraction": 0.01,
                "rootContinuationUnknownInformationSetFraction": 0.1,
                "rootContinuationUntrainedInformationSetFraction": 0.01,
                "informationSets": 10,
                "rootStrategies": root_strategies(fold_probability),
            }

        records = {
            "1": {"status": "complete", "blueprintSummary": summary(0.4)},
            "2": {"status": "complete", "blueprintSummary": summary(0.47)},
        }
        aggregate = aggregate_validated_summaries(records)
        self.assertIsNotNone(aggregate)
        stability = aggregate["crossSeedRootPolicyStability"]

        self.assertTrue(stability["available"])
        self.assertAlmostEqual(stability["maximumAggregateActionFrequencyDelta"], 0.07)
        self.assertAlmostEqual(stability["maximumComboWeightedPerActionMae"], 0.07)
        self.assertFalse(stability["gates"]["aggregateActionFrequencyDelta"])
        self.assertFalse(stability["gates"]["comboWeightedPerActionMae"])
        self.assertTrue(stability["gates"]["primaryActionAgreement"])
        self.assertFalse(stability["passed"])


if __name__ == "__main__":
    unittest.main()

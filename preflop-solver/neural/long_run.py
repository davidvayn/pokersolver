#!/usr/bin/env python3
"""Preflight, launch, monitor, and safely resume the frozen 20bb long pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "hu-neural-long-run-plan-v1"
RUNS_DIRECTORY = Path(__file__).resolve().parent / "runs"
TRAINER = Path(__file__).resolve().parent / "train.py"
VALIDATOR = Path(__file__).resolve().parent / "validate_seeds.py"
ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(__file__).resolve().parents[1] / ".venv-neural/bin/python"
STOP_REQUESTED = False


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("long-run plan schema is incompatible")
    seeds = plan.get("seeds")
    stages = plan.get("stages")
    shared = plan.get("sharedTraining")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or not all(isinstance(seed, int) and seed > 0 for seed in seeds)
        or not isinstance(stages, list)
        or len(stages) != 2
        or not isinstance(shared, dict)
    ):
        raise ValueError("long-run plan requires two seeds, two stages, and shared settings")
    total_minutes = 0.0
    stage_names: set[str] = set()
    run_directories: set[str] = set()
    for stage in stages:
        name = stage.get("name")
        template = stage.get("runDirectory")
        minutes = stage.get("minutes")
        if (
            not isinstance(name, str)
            or name in stage_names
            or not isinstance(template, str)
            or "{seed}" not in template
            or template.startswith(("/", "~"))
            or not isinstance(minutes, (int, float))
            or minutes <= 0
        ):
            raise ValueError("long-run stage is invalid")
        stage_names.add(name)
        total_minutes += float(minutes)
        for seed in seeds:
            rendered = template.format(seed=seed)
            if rendered in run_directories or Path(rendered).name != rendered:
                raise ValueError("long-run directories must be unique local names")
            run_directories.add(rendered)
    target_hours = float(plan.get("targetTrainingHoursPerSeed", 0))
    if not 8 <= target_hours <= 12 or abs(total_minutes / 60 - target_hours) > 1e-9:
        raise ValueError("stage time must equal the declared 8–12 hour per-seed budget")
    if int(shared.get("valueRolloutsPerAction", 0)) < 2:
        raise ValueError("long-run uncertainty training requires at least two value rollouts")
    if int(plan.get("monitorIntervalSeconds", 0)) < 60:
        raise ValueError("monitor interval must be at least 60 seconds")
    validation = plan.get("postRunValidation")
    if not isinstance(validation, dict):
        raise ValueError("long-run plan requires post-run validation settings")
    confidence = float(validation.get("exploitabilityCertificateConfidence", 0))
    deals = int(validation.get("exploitabilityCertificateDeals", 0))
    selected_preflop_round = int(validation.get("selectedPreflopRound", 0))
    if not 0 < confidence < 1 or deals < 2:
        raise ValueError("exploitability certificate settings are invalid")
    if (
        selected_preflop_round <= 0
        or selected_preflop_round % int(shared["artifactEvery"]) != 0
        or validation.get("useLatestPostflopArtifact") is not True
    ):
        raise ValueError("routed validation artifact selection is invalid")
    per_seed_alpha = (1.0 - confidence) / len(seeds)
    certificate_margin = float(plan["depthBb"]) * math.sqrt(
        math.log(1.0 / per_seed_alpha) / (2.0 * deals)
    )
    if certificate_margin >= 0.10:
        raise ValueError(
            "certificate chance margin cannot clear the 0.10bb release gate"
        )
    return plan


def training_seconds(run_dir: Path) -> float:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return 0.0
    elapsed = 0.0
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            elapsed += float(json.loads(line)["elapsed_seconds"])
    return elapsed


def run_summary(run_dir: Path, target_seconds: float) -> dict[str, Any]:
    trained = training_seconds(run_dir)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    metrics = state.get("metrics") or []
    last = metrics[-1] if metrics else {}
    return {
        "runDirectory": str(run_dir),
        "completedRounds": int(state.get("completed_rounds", 0)),
        "completedTraversals": int(state.get("completed_traversals", 0)),
        "trainingSeconds": trained,
        "targetSeconds": target_seconds,
        "progress": min(trained / target_seconds, 1.0) if target_seconds else 1.0,
        "lastRoundSeconds": last.get("elapsed_seconds"),
        "peakResidentBytes": last.get("peak_resident_bytes"),
    }


def trainer_command(
    plan: dict[str, Any], stage: dict[str, Any], seed: int, remaining_seconds: float
) -> list[str]:
    shared = plan["sharedTraining"]
    run_dir = RUNS_DIRECTORY / stage["runDirectory"].format(seed=seed)
    command = [
        str(PYTHON),
        str(TRAINER),
        "--run-dir",
        str(run_dir),
        "--depth-bb",
        str(plan["depthBb"]),
        "--seed",
        str(seed),
        "--rounds",
        "1000000",
        "--target-round",
        "1000000",
        "--max-minutes",
        str(remaining_seconds / 60),
        "--traversals-per-round",
        str(shared["traversalsPerRound"]),
        "--reservoir-capacity",
        str(shared["reservoirCapacity"]),
        "--hidden-sizes",
        str(stage["hiddenSizes"]),
        "--batch-size",
        str(shared["batchSize"]),
        "--steps-per-round",
        str(shared["stepsPerRound"]),
        "--learning-rate",
        str(stage["learningRate"]),
        "--learning-rate-final",
        str(stage["learningRateFinal"]),
        "--learning-rate-decay-start-round",
        str(stage["learningRateDecayStartRound"]),
        "--learning-rate-decay-end-round",
        str(stage["learningRateDecayEndRound"]),
        "--advantage-alpha",
        str(shared["advantageAlpha"]),
        "--variance-baseline-scale",
        str(shared["varianceBaselineScale"]),
        "--value-rollouts-per-action",
        str(shared["valueRolloutsPerAction"]),
        "--artifact-every",
        str(shared["artifactEvery"]),
        "--preflop-runout-samples",
        str(shared["preflopRunoutSamples"]),
        "--flop-runout-samples",
        str(shared["flopRunoutSamples"]),
        "--replay-street-proposal",
        str(shared["replayStreetProposal"]),
    ]
    if not shared["exactTurnRivers"]:
        command.append("--sample-turn-rivers")
    if shared["compactServingGrid"]:
        command.append("--compact-serving-grid")
    return command


def validation_command(plan: dict[str, Any]) -> list[str]:
    validation = plan["postRunValidation"]
    preflop_stage, postflop_stage = plan["stages"]

    def run_directory(stage: dict[str, Any], seed: int) -> Path:
        return RUNS_DIRECTORY / stage["runDirectory"].format(seed=seed)

    first_seed, second_seed = plan["seeds"]
    return [
        str(PYTHON),
        str(VALIDATOR),
        str(run_directory(preflop_stage, first_seed)),
        str(run_directory(preflop_stage, second_seed)),
        "--postflop-run-a",
        str(run_directory(postflop_stage, first_seed)),
        "--postflop-run-b",
        str(run_directory(postflop_stage, second_seed)),
        "--round",
        str(validation["selectedPreflopRound"]),
        "--postflop-latest",
        "--traversals",
        str(validation["trajectoryTraversalsPerSeed"]),
        "--seed",
        str(validation["independentEvaluationSeed"]),
        "--action-value-rollouts-per-action",
        str(validation["actionValueRolloutsPerAction"]),
        "--exploitability-certificate-deals",
        str(validation["exploitabilityCertificateDeals"]),
        "--exploitability-certificate-seed",
        str(validation["certificateSeed"]),
        "--exploitability-certificate-threads",
        str(validation["exploitabilityCertificateThreads"]),
        "--output",
        str(RUNS_DIRECTORY / f"{plan['modelVersion']}-validation.json"),
    ]


def preflight(plan: dict[str, Any], build: bool = True) -> dict[str, Any]:
    if not PYTHON.is_file():
        raise RuntimeError(f"neural virtual environment is missing: {PYTHON}")
    if not TRAINER.is_file():
        raise RuntimeError(f"trainer is missing: {TRAINER}")
    if not VALIDATOR.is_file():
        raise RuntimeError(f"validator is missing: {VALIDATOR}")
    free_disk = shutil.disk_usage(ROOT).free
    required_disk = int(plan["minimumFreeDiskGiB"] * 1024**3)
    if free_disk < required_disk:
        raise RuntimeError(
            f"long run needs {required_disk} free bytes but only {free_disk} are available"
        )
    subprocess.run(
        [str(PYTHON), "-c", "import mlx.core, numpy"],
        cwd=ROOT,
        check=True,
    )
    if build:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", "preflop-solver/Cargo.toml"],
            cwd=ROOT,
            check=True,
        )
    return {
        "schema": PLAN_SCHEMA,
        "status": "ready",
        "freeDiskBytes": free_disk,
        "requiredDiskBytes": required_disk,
        "logicalCpus": os.cpu_count(),
        "modelVersion": plan["modelVersion"],
        "seeds": plan["seeds"],
        "trainingHoursPerSeed": plan["targetTrainingHoursPerSeed"],
        "totalSeedComputeHours": plan["targetTrainingHoursPerSeed"] * len(plan["seeds"]),
    }


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_stage(plan: dict[str, Any], stage: dict[str, Any], dry_run: bool) -> None:
    target_seconds = float(stage["minutes"]) * 60
    pending: list[tuple[int, Path, list[str]]] = []
    for seed in plan["seeds"]:
        run_dir = RUNS_DIRECTORY / stage["runDirectory"].format(seed=seed)
        remaining = max(target_seconds - training_seconds(run_dir), 0.0)
        if remaining > 0:
            pending.append((seed, run_dir, trainer_command(plan, stage, seed, remaining)))
    event = {
        "event": "stage_plan",
        "stage": stage["name"],
        "commands": [command for _, _, command in pending],
    }
    print(json.dumps(event), flush=True)
    if dry_run or not pending:
        return
    processes = [
        (seed, run_dir, subprocess.Popen(command, cwd=ROOT))
        for seed, run_dir, command in pending
    ]
    stop_signals_sent = False
    interval = float(plan["monitorIntervalSeconds"])
    next_update = time.monotonic() + interval
    while processes:
        if STOP_REQUESTED and not stop_signals_sent:
            for _, _, process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            stop_signals_sent = True
        failed: tuple[int, int] | None = None
        remaining_processes = []
        for seed, run_dir, process in processes:
            status = process.poll()
            if status is None:
                remaining_processes.append((seed, run_dir, process))
            elif status != 0:
                failed = (seed, status)
        processes = remaining_processes
        now = time.monotonic()
        if now >= next_update or not processes:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "stage": stage["name"],
                        "runs": [
                            run_summary(
                                RUNS_DIRECTORY
                                / stage["runDirectory"].format(seed=seed),
                                target_seconds,
                            )
                            for seed in plan["seeds"]
                        ],
                    }
                ),
                flush=True,
            )
            next_update = now + interval
        if failed is not None:
            for _, _, process in processes:
                process.send_signal(signal.SIGINT)
            for _, _, process in processes:
                process.wait()
            raise RuntimeError(f"seed {failed[0]} exited with status {failed[1]}")
        if processes:
            time.sleep(1)
    if STOP_REQUESTED:
        raise KeyboardInterrupt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().with_name("long-run-20bb-v1.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--stage", action="append", help="run only a named stage")
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args = parse_args()
    plan = load_plan(args.plan.resolve())
    selected = set(args.stage or [stage["name"] for stage in plan["stages"]])
    known = {stage["name"] for stage in plan["stages"]}
    if not selected <= known:
        raise ValueError(f"unknown stages: {sorted(selected - known)}")
    print(json.dumps(preflight(plan, build=not args.skip_build), sort_keys=True), flush=True)
    if args.preflight_only:
        return
    for stage in plan["stages"]:
        if stage["name"] in selected:
            run_stage(plan, stage, args.dry_run)
    print(
        json.dumps(
            {
                "event": "long_run_complete",
                "status": "paused_for_validation",
                "validationCommand": validation_command(plan),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(json.dumps({"event": "long_run_paused", "resumable": True}), flush=True)
        sys.exit(130)

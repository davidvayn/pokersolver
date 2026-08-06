#!/usr/bin/env python3
"""Plan or execute the frozen successor authentic-value recheck."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import freeze_resolver_reach_release as release_freeze
import run_resolver_reach_release as old_runner


PROTOCOL_SCHEMA = "hu-fresh-authentic-value-recheck-protocol-v1"
PLAN_SCHEMA = "hu-fresh-authentic-value-recheck-execution-plan-v1"
VALUE_SCHEMA = "hu-resolver-reach-value-release-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-plan", type=Path)
    return parser.parse_args()


def resolved(repository_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def checked_reference(repository_root: Path, reference: dict[str, Any]) -> Path:
    path = resolved(repository_root, reference.get("path", ""))
    if not path.is_file() or release_freeze.sha256_file(path) != reference.get("sha256"):
        raise ValueError(f"fresh authentic recheck artifact is missing or changed: {path}")
    return path


def validate_protocol(
    protocol_path: Path, repository_root: Path, require_unopened: bool = False
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen-before-successor-authentic-recheck"
        or protocol.get("activationAllowed") is not False
    ):
        raise ValueError("fresh authentic recheck protocol is not fail-closed")
    predecessor = protocol.get("predecessor", {})
    release_path = checked_reference(repository_root, predecessor["releaseFreeze"])
    value_path = checked_reference(
        repository_root, predecessor["acceptedValueValidation"]
    )
    failure_path = checked_reference(
        repository_root, predecessor["rangeResponseFailure"]
    )
    release = json.loads(release_path.read_text())
    value = json.loads(value_path.read_text())
    failure = json.loads(failure_path.read_text())
    if (
        release.get("activationAllowed") is not False
        or value.get("schema") != VALUE_SCHEMA
        or value.get("status") != "accepted-awaiting-strategy-and-full-game-gates"
        or value.get("activationAllowed") is not False
        or not value.get("gates")
        or not all(gate is True for gate in value["gates"].values())
        or failure.get("status") != "rejected"
        or failure.get("activationAllowed") is not False
        or failure.get("freeze", {}).get("entireRootSetBurned") is not True
    ):
        raise ValueError("fresh authentic recheck predecessor state is invalid")
    checked_reference(repository_root, protocol["sourcePolicy"])
    models = protocol.get("models", [])
    for reference in models:
        path = checked_reference(repository_root, reference)
        payload = json.loads(path.read_text())
        if (
            int(payload.get("seed", -1)) != int(reference["seed"])
            or payload.get("usesExactRanges") is not True
            or payload.get("sourceValidationStatus") != "accepted"
        ):
            raise ValueError(f"fresh authentic recheck model is ineligible: {path}")
    if len(models) != 2 or len({int(model["seed"]) for model in models}) != 2:
        raise ValueError("fresh authentic recheck requires exactly two models")

    for section_name in ("implementation", "orchestration"):
        section = protocol.get(section_name, {})
        if len(str(section.get("commit", ""))) != 40:
            raise ValueError(f"fresh authentic recheck {section_name} commit is not pinned")
        references = section.get("files", [])
        for reference in references:
            checked_reference(repository_root, reference)
        if len(references) < 3:
            raise ValueError(
                f"fresh authentic recheck {section_name} sources are not fully pinned"
            )

    generator = protocol.get("generator", {})
    expected_generator = {
        "command": "turn-pbs-self-play-targets",
        "effectiveStackBb": 20.0,
        "statesPerShard": 64,
        "rangeParticles": 4096,
        "riverIterations": 200,
        "riverAveragingDelay": 20,
        "threads": 10,
        "beliefReplicates": 2,
        "explorationProbability": 0.0,
        "minimumPotBb": 0.0,
    }
    if generator != expected_generator:
        raise ValueError("fresh authentic recheck generator controls changed")
    shards = protocol.get("shards", [])
    seeds = [int(shard["seed"]) for shard in shards]
    outputs = [resolved(repository_root, shard["output"]) for shard in shards]
    checkpoints = [
        resolved(repository_root, shard["checkpointDirectory"]) for shard in shards
    ]
    if (
        seeds != [15501, 15502]
        or len(set(outputs)) != 2
        or len(set(checkpoints)) != 2
        or set(seeds) & {int(model["seed"]) for model in models}
    ):
        raise ValueError("fresh authentic recheck shard freeze is invalid")
    if require_unopened and any(path.exists() for path in outputs):
        raise ValueError("fresh authentic recheck output existed before protocol execution")
    gates = protocol.get("gates", {})
    failure_policy = protocol.get("failurePolicy", {})
    if (
        float(gates.get("maximumPerSeedRmseBb", -1.0)) != 0.25
        or float(gates.get("minimumCrossSeedPredictionCorrelation", -1.0)) != 0.95
        or gates.get("requireUniqueFingerprints") is not True
        or gates.get("requireDisjointnessFromEveryPreviouslyOpenedState") is not True
        or failure_policy.get("useForModelSelection") is not False
        or failure_policy.get("missingOrInvalidEvidenceFailsClosed") is not True
        or failure_policy.get("rejectionRequiresNewHoldoutSeeds") is not True
        or failure_policy.get("activationRequiresEveryStrategyAndFullGameGate")
        is not True
    ):
        raise ValueError("fresh authentic recheck gates were weakened")
    return protocol


def holdout_command(protocol: dict[str, Any], shard: dict[str, Any]) -> list[str]:
    generator = protocol["generator"]
    return [
        "target/release/preflop-solver",
        generator["command"],
        "--effective-stack-bb",
        str(generator["effectiveStackBb"]),
        "--states",
        str(generator["statesPerShard"]),
        "--range-particles",
        str(generator["rangeParticles"]),
        "--river-iterations",
        str(generator["riverIterations"]),
        "--river-averaging-delay",
        str(generator["riverAveragingDelay"]),
        "--seed",
        str(shard["seed"]),
        "--threads",
        str(generator["threads"]),
        "--networks",
        protocol["sourcePolicy"]["path"],
        "--belief-replicates",
        str(generator["beliefReplicates"]),
        "--exploration",
        str(generator["explorationProbability"]),
        "--minimum-pot-bb",
        str(generator["minimumPotBb"]),
        "--checkpoint-dir",
        shard["checkpointDirectory"],
        "--output",
        shard["output"],
    ]


def build_plan(protocol: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    holdouts = [
        {
            "name": f"fresh-authentic-recheck-seed{shard['seed']}",
            "seed": int(shard["seed"]),
            "command": holdout_command(protocol, shard),
            "output": shard["output"],
        }
        for shard in protocol["shards"]
    ]
    diagnostics = []
    for model in protocol["models"]:
        for shard in protocol["shards"]:
            output = (
                f"{protocol['diagnosticOutputDirectory']}/holdout-seed{shard['seed']}"
                f"-model{model['seed']}.json"
            )
            diagnostics.append(
                {
                    "name": f"holdout-seed{shard['seed']}-model{model['seed']}",
                    "datasetSeed": int(shard["seed"]),
                    "modelSeed": int(model["seed"]),
                    "command": old_runner.diagnostic_command(
                        shard["output"],
                        model["path"],
                        int(protocol["generator"]["statesPerShard"]),
                        output,
                    ),
                    "output": output,
                }
            )
    return {
        "schema": PLAN_SCHEMA,
        "status": "frozen-for-successor-authentic-recheck-execution",
        "activationAllowed": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": release_freeze.sha256_file(protocol_path),
        },
        "models": protocol["models"],
        "gates": protocol["gates"],
        "holdoutJobs": holdouts,
        "diagnosticJobs": diagnostics,
    }


def run_job(job: dict[str, Any], repository_root: Path, resume: bool) -> None:
    output = resolved(repository_root, job["output"])
    if output.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite authentic recheck output: {output}")
        print(json.dumps({"event": "authentic-recheck-job-reused", "name": job["name"]}), flush=True)
        return
    print(json.dumps({"event": "authentic-recheck-job-start", "name": job["name"]}), flush=True)
    old_runner.run_command(job["command"], repository_root, output.with_suffix(".log"))
    if not output.is_file():
        raise ValueError(f"authentic recheck job created no output: {output}")
    print(json.dumps({"event": "authentic-recheck-job-complete", "name": job["name"]}), flush=True)


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.protocol.resolve().parent.parent
    protocol = validate_protocol(
        args.protocol, repository_root, require_unopened=not args.resume
    )
    plan = build_plan(protocol, args.protocol)
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_plan.with_suffix(args.output_plan.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output_plan)
    if not args.execute:
        print(encoded, end="")
        return
    subprocess.run(
        ["cargo", "build", "--release", "--locked"],
        cwd=repository_root,
        check=True,
    )
    for job in plan["holdoutJobs"]:
        run_job(job, repository_root, args.resume)
    for job in plan["diagnosticJobs"]:
        run_job(job, repository_root, args.resume)
        old_runner.validate_authentic_diagnostic(
            resolved(repository_root, job["output"]),
            release_freeze.sha256_file(
                resolved(
                    repository_root,
                    old_runner.crossfit.option_values(job["command"], "--dataset")[0],
                )
            ),
            int(job["modelSeed"]),
            release_freeze.sha256_file(
                resolved(
                    repository_root,
                    old_runner.crossfit.option_values(job["command"], "--model")[0],
                )
            ),
        )
    print(json.dumps({"status": "authentic-recheck-complete", "activationAllowed": False}))


if __name__ == "__main__":
    main()

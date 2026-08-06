#!/usr/bin/env python3
"""Plan or execute a frozen V49 release training and fresh evaluation run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import freeze_resolver_reach_release as release_freeze
import run_resolver_reach_crossfit as crossfit
import select_resolver_reach_value_config as selection


PLAN_SCHEMA = "hu-resolver-reach-release-execution-plan-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_freeze", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-plan", type=Path)
    return parser.parse_args()


def flag(command: list[str], name: str, value: Any) -> None:
    command.extend((name, str(value)))


def trainer_command(payload: dict[str, Any]) -> list[str]:
    trainer = payload["trainer"]
    command = [
        ".venv-neural/bin/python",
        "neural/train_public_value_network.py",
        "--dataset",
        payload["primaryDataset"]["path"],
    ]
    for source in payload["supplementalDatasets"]:
        command.extend(("--supplemental-dataset", source["path"]))
    for weight in payload["supplementalDatasetSamplingWeights"]:
        command.extend(("--supplemental-dataset-weight", str(weight)))
    options = (
        ("--minimum-primary-batch-fraction", payload["minimumPrimaryBatchFraction"]),
        ("--output-dir", trainer["outputDirectory"]),
        ("--architecture", trainer["architecture"]),
        ("--feature-schema", trainer["featureSchema"]),
        ("--feature-workers", trainer["featureWorkers"]),
        ("--feature-cache-dir", trainer["featureCacheDirectory"]),
        ("--value-normalization", trainer["valueNormalization"]),
        ("--variant-set", trainer["variantSet"]),
        ("--steps", trainer["steps"]),
        ("--batch-size", trainer["batchSize"]),
        ("--evaluation-interval", trainer["evaluationInterval"]),
        ("--learning-rate", trainer["learningRate"]),
        ("--early-stopping-patience", trainer["earlyStoppingPatience"]),
        ("--huber-delta", trainer["huberDelta"]),
        ("--raw-bb-auxiliary-weight", trainer["rawBbAuxiliaryWeight"]),
        ("--suit-augmentations", trainer["suitAugmentationsPerState"]),
        ("--split-seed", trainer["splitSeed"]),
        ("--validation-fraction", trainer["validationFraction"]),
        ("--tuning-fraction", trainer["tuningFraction"]),
        ("--seeds", ",".join(str(seed) for seed in trainer["trainingSeeds"])),
        (
            "--minimum-cross-seed-correlation",
            payload["releaseGates"]["minimumCrossSeedPredictionCorrelation"],
        ),
        (
            "--minimum-tuning-cross-seed-correlation",
            payload["releaseGates"]["minimumCrossSeedPredictionCorrelation"],
        ),
        (
            "--maximum-rmse-bb",
            payload["releaseGates"]["maximumPerSeedAuthenticFreshHoldoutRmseBb"],
        ),
    )
    for name, value in options:
        flag(command, name, value)
    if trainer.get("adamBiasCorrection"):
        command.append("--adam-bias-correction")
    return command


def resolver_evaluation_command(
    corpus: dict[str, Any], shard: dict[str, Any]
) -> list[str]:
    generator = corpus["generator"]
    sources = {
        int(source["trainingSeed"]): source for source in corpus["sourceValueNetworks"]
    }
    source = sources[int(shard["sourceTrainingSeed"])]
    return [
        "target/release/preflop-solver",
        "flop-pbs-leaf-targets",
        "--effective-stack-bb",
        str(generator["effectiveStackBb"]),
        "--boards",
        ";".join(shard["boards"]),
        "--states-per-board",
        str(generator["statesPerBoard"]),
        "--pot-bb",
        str(generator["rootPotBb"]),
        "--actor",
        str(generator["rootActor"]),
        "--resolver-iterations",
        str(generator["resolverIterations"]),
        "--resolver-averaging-delay",
        str(generator["resolverAveragingDelay"]),
        "--river-iterations",
        str(generator["riverIterations"]),
        "--river-averaging-delay",
        str(generator["riverAveragingDelay"]),
        "--seed",
        str(shard["seed"]),
        "--threads",
        str(generator["threads"]),
        "--value-network",
        source["path"],
        "--checkpoint-dir",
        shard["checkpointDirectory"],
        "--output",
        shard["output"],
    ]


def authentic_holdout_command(holdout: dict[str, Any], shard: dict[str, Any]) -> list[str]:
    generator = holdout["generator"]
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
        holdout["sourcePolicy"]["path"],
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


def diagnostic_command(dataset: str, model: str, states: int, output: str) -> list[str]:
    return [
        ".venv-neural/bin/python",
        "neural/diagnose_public_value_model.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--state-indices",
        ",".join(str(index) for index in range(states)),
        "--output",
        output,
    ]


def parity_command(dataset: str, model: str, output: str, threshold: float) -> list[str]:
    return [
        ".venv-neural/bin/python",
        "neural/validate_public_value_parity.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--state-indices",
        "0,1,2,3,4,5",
        "--maximum-absolute-error-bb",
        str(threshold),
        "--output",
        output,
    ]


def build_plan(payload: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    output_directory = payload["trainer"]["outputDirectory"]
    report = f"{output_directory}/turn-value-paired-report.json"
    models = [
        {
            "seed": int(seed),
            "path": f"{output_directory}/turn-value-range-seed{seed}.json",
        }
        for seed in payload["trainer"]["trainingSeeds"]
    ]
    resolver_jobs = [
        {
            "name": f"reserved-resolver-seed{shard['seed']}",
            "command": resolver_evaluation_command(corpus, shard),
            "output": shard["output"],
        }
        for shard in payload["reservedResolverEvaluation"]
    ]
    holdout_jobs = [
        {
            "name": f"fresh-authentic-seed{shard['seed']}",
            "command": authentic_holdout_command(
                payload["freshAuthenticHoldout"], shard
            ),
            "output": shard["output"],
        }
        for shard in payload["freshAuthenticHoldout"]["shards"]
    ]
    baseline_diagnostics = []
    for model in corpus["sourceValueNetworks"]:
        for shard in payload["reservedResolverEvaluation"]:
            output = (
                f"{output_directory}/baseline-resolver-eval-seed{shard['seed']}"
                f"-model{model['trainingSeed']}.json"
            )
            baseline_diagnostics.append(
                {
                    "name": (
                        f"baseline-resolver-seed{shard['seed']}"
                        f"-model{model['trainingSeed']}"
                    ),
                    "command": diagnostic_command(
                        shard["output"],
                        model["path"],
                        int(shard["expectedStateCount"]),
                        output,
                    ),
                    "output": output,
                }
            )
    diagnostics = []
    for model in models:
        for shard in payload["reservedResolverEvaluation"]:
            output = (
                f"{output_directory}/resolver-eval-seed{shard['seed']}"
                f"-model{model['seed']}.json"
            )
            diagnostics.append(
                {
                    "name": f"resolver-seed{shard['seed']}-model{model['seed']}",
                    "command": diagnostic_command(
                        shard["output"],
                        model["path"],
                        int(shard["expectedStateCount"]),
                        output,
                    ),
                    "output": output,
                }
            )
        for shard in payload["freshAuthenticHoldout"]["shards"]:
            output = (
                f"{output_directory}/authentic-holdout-seed{shard['seed']}"
                f"-model{model['seed']}.json"
            )
            diagnostics.append(
                {
                    "name": f"authentic-seed{shard['seed']}-model{model['seed']}",
                    "command": diagnostic_command(
                        shard["output"],
                        model["path"],
                        int(payload["freshAuthenticHoldout"]["generator"]["statesPerShard"]),
                        output,
                    ),
                    "output": output,
                }
            )
    first_resolver = payload["reservedResolverEvaluation"][0]
    parity_jobs = [
        {
            "name": f"parity-model{model['seed']}",
            "command": parity_command(
                first_resolver["output"],
                model["path"],
                f"{output_directory}/parity-model{model['seed']}.json",
                float(payload["releaseGates"]["maximumRustPythonParityErrorBb"]),
            ),
            "output": f"{output_directory}/parity-model{model['seed']}.json",
        }
        for model in models
    ]
    return {
        "schema": PLAN_SCHEMA,
        "activationAllowed": False,
        "trainingJob": {
            "name": "release-paired-training",
            "command": trainer_command(payload),
            "report": report,
            "models": models,
        },
        "resolverEvaluationJobs": resolver_jobs,
        "freshAuthenticHoldoutJobs": holdout_jobs,
        "baselineResolverDiagnosticJobs": baseline_diagnostics,
        "diagnosticJobs": diagnostics,
        "parityJobs": parity_jobs,
    }


def run_command(command: list[str], repository_root: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as sink:
        subprocess.run(
            command,
            cwd=repository_root,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_output_job(
    job: dict[str, Any], repository_root: Path, resume: bool
) -> None:
    output = repository_root / job["output"]
    if output.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite existing output: {output}")
        print(json.dumps({"event": "release-job-reused", "name": job["name"]}), flush=True)
        return
    print(json.dumps({"event": "release-job-start", "name": job["name"]}), flush=True)
    run_command(job["command"], repository_root, output.with_suffix(".log"))
    if not output.is_file():
        raise ValueError(f"release job did not create its output: {output}")
    print(json.dumps({"event": "release-job-complete", "name": job["name"]}), flush=True)


def validate_training_job(plan: dict[str, Any], repository_root: Path) -> None:
    training = plan["trainingJob"]
    crossfit.validate_training_job(
        {
            "trainingCommand": training["command"],
            "trainingReport": training["report"],
        },
        repository_root,
    )
    report = json.loads((repository_root / training["report"]).read_text())
    ceiling = float(
        crossfit.option_values(training["command"], "--maximum-rmse-bb")[0]
    )
    metrics = [
        entry.get("metrics", {})
        for entry in report.get("variants", {}).get("range", [])
    ]
    tuning = [float(entry.get("bestTuningRmseBb", float("inf"))) for entry in metrics]
    validation = [float(entry.get("weightedRmseBb", float("inf"))) for entry in metrics]
    if (
        len(metrics) != 2
        or max(tuning, default=float("inf")) > ceiling
        or max(validation, default=float("inf")) > ceiling
    ):
        raise ValueError(
            "release pair failed opened-primary tuning/validation preconditions; "
            "fresh evaluation remains sealed"
        )


def validate_diagnostic_job(job: dict[str, Any], repository_root: Path) -> None:
    command = job["command"]
    dataset = repository_root / crossfit.option_values(command, "--dataset")[0]
    model = repository_root / crossfit.option_values(command, "--model")[0]
    output = repository_root / job["output"]
    model_payload = json.loads(model.read_text())
    selection.diagnostic_metric(
        output,
        release_freeze.sha256_file(dataset),
        {int(model_payload["seed"]): release_freeze.sha256_file(model)},
    )


def execute_plan(
    plan: dict[str, Any], repository_root: Path, resume: bool
) -> None:
    training = plan["trainingJob"]
    report = repository_root / training["report"]
    if report.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite existing training report: {report}")
        validate_training_job(plan, repository_root)
    else:
        print(json.dumps({"event": "release-job-start", "name": training["name"]}), flush=True)
        run_command(
            training["command"],
            repository_root,
            report.with_name("training.log"),
        )
        validate_training_job(plan, repository_root)
        print(json.dumps({"event": "release-job-complete", "name": training["name"]}), flush=True)
    for group in ("resolverEvaluationJobs", "freshAuthenticHoldoutJobs"):
        for job in plan[group]:
            run_output_job(job, repository_root, resume)
    for group in ("baselineResolverDiagnosticJobs", "diagnosticJobs"):
        for job in plan[group]:
            run_output_job(job, repository_root, resume)
            validate_diagnostic_job(job, repository_root)
    for job in plan["parityJobs"]:
        run_output_job(job, repository_root, resume)
        report_payload = json.loads((repository_root / job["output"]).read_text())
        if report_payload.get("validation", {}).get("status") != "accepted":
            raise ValueError(f"release parity failed: {job['output']}")


def validate_release_freeze(path: Path, repository_root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != release_freeze.RELEASE_SCHEMA
        or payload.get("status") != "frozen-for-fresh-evaluation"
        or payload.get("activationAllowed") is not False
    ):
        raise ValueError("release freeze is not eligible for execution")
    protocol_path = release_freeze.resolved(repository_root, payload["protocol"]["path"])
    selection_path = release_freeze.resolved(repository_root, payload["selection"]["path"])
    if release_freeze.sha256_file(protocol_path) != payload["protocol"]["sha256"]:
        raise ValueError("release protocol hash mismatch")
    if release_freeze.sha256_file(selection_path) != payload["selection"]["sha256"]:
        raise ValueError("release selection hash mismatch")
    expected = release_freeze.build_release_freeze(
        protocol_path,
        selection_path,
        repository_root,
        require_unopened=False,
    )
    # The stored references may be repository-relative while recomputation uses
    # resolved paths. Path spelling is not semantic; their exact hashes are.
    expected["protocol"]["path"] = payload["protocol"]["path"]
    expected["selection"]["path"] = payload["selection"]["path"]
    if expected != payload:
        raise ValueError(
            "release freeze does not reproduce from its pinned protocol and selection"
        )
    return payload


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.release_freeze.resolve().parent.parent
    payload = validate_release_freeze(args.release_freeze, repository_root)
    protocol_path = release_freeze.resolved(repository_root, payload["protocol"]["path"])
    _, _, corpus = release_freeze.validate_protocol(
        protocol_path, repository_root, require_unopened=not args.resume
    )
    plan = build_plan(payload, corpus)
    plan["releaseFreeze"] = {
        "path": str(args.release_freeze),
        "sha256": release_freeze.sha256_file(args.release_freeze),
    }
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        args.output_plan.write_text(encoded)
    if not args.execute:
        print(encoded, end="")
        return
    execute_plan(plan, repository_root, args.resume)
    print(json.dumps({"status": "evaluation-complete-activation-still-disabled"}))


if __name__ == "__main__":
    main()

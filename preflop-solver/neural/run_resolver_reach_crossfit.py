#!/usr/bin/env python3
"""Execute the frozen V49 resolver-reach cross-validation sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import select_resolver_reach_value_config as selection
import select_public_value_config as value_selection
import validate_resolver_reach_experiments as experiment_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="preflop-solver directory; defaults to the config's parent directory",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-plan", type=Path)
    return parser.parse_args()


def flag(command: list[str], name: str, value: Any) -> None:
    command.extend((name, str(value)))


def trainer_command(
    payload: dict[str, Any],
    candidate: dict[str, Any],
    fold: dict[str, Any],
    resolver_folds: dict[str, dict[str, Any]],
) -> list[str]:
    common = payload["commonTrainer"]
    supplements = [
        entry["path"] for entry in payload["baseSupplementalDatasets"]
    ] + [resolver_folds[fold["trainingFold"]]["dataset"]]
    command = [
        ".venv-neural/bin/python",
        "neural/train_public_value_network.py",
        "--dataset",
        payload["primaryDataset"]["path"],
    ]
    for path in supplements:
        command.extend(("--supplemental-dataset", str(path)))
    for weight in candidate["supplementalDatasetSamplingWeights"]:
        command.extend(("--supplemental-dataset-weight", str(weight)))
    options = (
        ("--minimum-primary-batch-fraction", candidate["minimumPrimaryBatchFraction"]),
        ("--output-dir", fold["outputDirectory"]),
        ("--architecture", common["architecture"]),
        ("--feature-schema", common["featureSchema"]),
        ("--feature-workers", common["featureWorkers"]),
        ("--feature-cache-dir", common["featureCacheDirectory"]),
        ("--value-normalization", common["valueNormalization"]),
        ("--variant-set", common["variantSet"]),
        ("--steps", common["steps"]),
        ("--batch-size", common["batchSize"]),
        ("--evaluation-interval", common["evaluationInterval"]),
        ("--learning-rate", common["learningRate"]),
        ("--early-stopping-patience", common["earlyStoppingPatience"]),
        ("--huber-delta", common["huberDelta"]),
        ("--raw-bb-auxiliary-weight", common["rawBbAuxiliaryWeight"]),
        ("--suit-augmentations", common["suitAugmentationsPerState"]),
        ("--split-seed", common["splitSeed"]),
        ("--validation-fraction", common["validationFraction"]),
        ("--tuning-fraction", common["tuningFraction"]),
        ("--seeds", ",".join(str(seed) for seed in fold["trainingSeeds"])),
        (
            "--minimum-cross-seed-correlation",
            payload["selectionGates"]["minimumCrossSeedPredictionCorrelation"],
        ),
        (
            "--minimum-tuning-cross-seed-correlation",
            payload["selectionGates"]["minimumCrossSeedPredictionCorrelation"],
        ),
        (
            "--maximum-rmse-bb",
            payload["selectionGates"]["maximumAuthenticTuningRmseBb"],
        ),
    )
    for name, value in options:
        flag(command, name, value)
    if common.get("adamBiasCorrection"):
        command.append("--adam-bias-correction")
    return command


def diagnostic_command(dataset: str, model: str, output: str, states: int) -> list[str]:
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


def build_plan(payload: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    resolver_folds = {entry["name"]: entry for entry in payload["resolverFolds"]}
    baseline_models = [
        {
            "seed": int(entry["trainingSeed"]),
            "path": entry["path"],
            "sha256": entry["sha256"],
        }
        for entry in corpus["sourceValueNetworks"]
    ]
    baseline_folds = []
    baseline_jobs = []
    baseline_directory = "neural/runs/v49-resolver-reach/crossfit/baseline"
    for fold_name, fold in resolver_folds.items():
        diagnostics = []
        for model in baseline_models:
            output = f"{baseline_directory}/{fold_name}-seed{model['seed']}.json"
            diagnostics.append(output)
            baseline_jobs.append(
                {
                    "name": f"baseline-{fold_name}-seed{model['seed']}",
                    "command": diagnostic_command(
                        fold["dataset"],
                        model["path"],
                        output,
                        int(fold["expectedStateCount"]),
                    ),
                    "output": output,
                }
            )
        baseline_folds.append(
            {
                "name": fold_name,
                "evaluationDataset": fold["dataset"],
                "diagnostics": diagnostics,
            }
        )

    candidate_jobs = []
    selector_candidates = []
    for candidate in payload["candidates"]:
        selector_folds = []
        for fold in candidate["folds"]:
            training_command = trainer_command(payload, candidate, fold, resolver_folds)
            report = f"{fold['outputDirectory']}/turn-value-paired-report.json"
            diagnostics = []
            diagnostic_jobs = []
            evaluation = resolver_folds[fold["evaluationFold"]]
            for seed in fold["trainingSeeds"]:
                model = f"{fold['outputDirectory']}/turn-value-range-seed{seed}.json"
                output = (
                    f"{fold['outputDirectory']}/crossfit-{fold['evaluationFold']}"
                    f"-seed{seed}.json"
                )
                diagnostics.append(output)
                diagnostic_jobs.append(
                    {
                        "name": f"{candidate['name']}-{fold['evaluationFold']}-seed{seed}",
                        "command": diagnostic_command(
                            evaluation["dataset"],
                            model,
                            output,
                            int(evaluation["expectedStateCount"]),
                        ),
                        "output": output,
                    }
                )
            candidate_jobs.append(
                {
                    "name": f"{candidate['name']}-{fold['trainingFold']}",
                    "trainingCommand": training_command,
                    "trainingReport": report,
                    "diagnosticJobs": diagnostic_jobs,
                }
            )
            selector_folds.append(
                {
                    "name": fold["evaluationFold"],
                    "trainingReport": report,
                    "evaluationDataset": evaluation["dataset"],
                    "diagnostics": diagnostics,
                }
            )
        selector_candidates.append(
            {"name": candidate["name"], "folds": selector_folds}
        )
    selector_spec = {
        "schema": selection.SPEC_SCHEMA,
        "gates": {
            "minimumCrossSeedPredictionCorrelation": payload["selectionGates"][
                "minimumCrossSeedPredictionCorrelation"
            ],
            "maximumAuthenticTuningRmseBb": payload["selectionGates"][
                "maximumAuthenticTuningRmseBb"
            ],
            "minimumMaximumResolverReachRmseImprovementFraction": payload[
                "selectionGates"
            ]["minimumMaximumResolverReachRmseImprovementFraction"],
        },
        "baseline": {"models": baseline_models, "folds": baseline_folds},
        "candidates": selector_candidates,
    }
    return {
        "schema": "hu-resolver-reach-crossfit-execution-plan-v1",
        "activationAllowed": False,
        "baselineJobs": baseline_jobs,
        "candidateJobs": candidate_jobs,
        "selectorSpec": selector_spec,
        "selectorSpecOutput": "neural/20bb-v49-crossfit-selection-spec.json",
        "selectionOutput": "neural/20bb-v49-crossfit-selection.json",
    }


def run_command(command: list[str], repository_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=repository_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def option_values(command: list[str], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def validate_diagnostic_job(job: dict[str, Any], repository_root: Path) -> None:
    command = job["command"]
    dataset = repository_root / option_values(command, "--dataset")[0]
    model = repository_root / option_values(command, "--model")[0]
    output = repository_root / job["output"]
    model_payload = json.loads(model.read_text())
    model_seed = int(model_payload["seed"])
    model_sha256 = selection.sha256_file(model)
    selection.diagnostic_metric(
        output,
        selection.sha256_file(dataset),
        {model_seed: model_sha256},
    )
    expected_states = len(option_values(command, "--state-indices")[0].split(","))
    if int(json.loads(output.read_text()).get("states", -1)) != expected_states:
        raise ValueError(f"diagnostic has the wrong state count: {output}")


def validate_training_job(job: dict[str, Any], repository_root: Path) -> None:
    report_path = repository_root / job["trainingReport"]
    report = json.loads(report_path.read_text())
    if report.get("holdoutStartIndex") is not None:
        raise ValueError(f"training report reused a predecessor holdout: {report_path}")
    expected_seeds = {
        int(seed)
        for seed in option_values(job["trainingCommand"], "--seeds")[0].split(",")
    }
    actual_seeds = {
        int(entry["seed"])
        for entry in report.get("variants", {}).get("range", [])
    }
    if actual_seeds != expected_seeds:
        raise ValueError(f"training report has the wrong seeds: {report_path}")
    expected_supplements = option_values(
        job["trainingCommand"], "--supplemental-dataset"
    )
    if report.get("supplementalDatasets") != expected_supplements:
        raise ValueError(f"training report has the wrong supplemental datasets: {report_path}")
    expected_weights = [
        float(value)
        for value in option_values(
            job["trainingCommand"], "--supplemental-dataset-weight"
        )
    ]
    if report.get("supplementalDatasetSamplingWeights") != expected_weights:
        raise ValueError(f"training report has the wrong replay weights: {report_path}")
    minimum_correlation = float(
        option_values(job["trainingCommand"], "--minimum-tuning-cross-seed-correlation")[0]
    )
    value_selection.summarize_candidate(report_path, minimum_correlation)


def execute_plan(
    plan: dict[str, Any], repository_root: Path, resume: bool = False
) -> dict[str, Any]:
    for job in plan["baselineJobs"]:
        output = repository_root / job["output"]
        if output.exists():
            if not resume:
                raise ValueError(f"refusing to overwrite an existing diagnostic: {output}")
            validate_diagnostic_job(job, repository_root)
            print(json.dumps({"event": "crossfit-job-reused", "name": job["name"]}), flush=True)
        else:
            print(json.dumps({"event": "crossfit-job-start", "name": job["name"]}), flush=True)
            run_command(
                job["command"], repository_root, output.with_suffix(".log")
            )
            validate_diagnostic_job(job, repository_root)
    for job in plan["candidateJobs"]:
        report = repository_root / job["trainingReport"]
        if report.exists():
            if not resume:
                raise ValueError(f"refusing to overwrite an existing training report: {report}")
            validate_training_job(job, repository_root)
            print(json.dumps({"event": "crossfit-job-reused", "name": job["name"]}), flush=True)
        else:
            print(json.dumps({"event": "crossfit-job-start", "name": job["name"]}), flush=True)
            run_command(
                job["trainingCommand"],
                repository_root,
                report.with_name("training.log"),
            )
            if not report.is_file():
                raise ValueError(f"training job did not create its report: {report}")
            validate_training_job(job, repository_root)
        for diagnostic in job["diagnosticJobs"]:
            output = repository_root / diagnostic["output"]
            if output.exists():
                if not resume:
                    raise ValueError(f"refusing to overwrite an existing diagnostic: {output}")
                validate_diagnostic_job(diagnostic, repository_root)
            else:
                run_command(
                    diagnostic["command"], repository_root, output.with_suffix(".log")
                )
                validate_diagnostic_job(diagnostic, repository_root)
        print(json.dumps({"event": "crossfit-job-complete", "name": job["name"]}), flush=True)

    spec_path = repository_root / plan["selectorSpecOutput"]
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(plan["selectorSpec"], indent=2, sort_keys=True) + "\n")
    result = selection.select(spec_path, repository_root)
    selection_path = repository_root / plan["selectionOutput"]
    selection_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.config.resolve().parent.parent
    validation = experiment_validation.validate_experiment(
        args.config,
        repository_root,
        require_completed_training_corpus=args.execute,
    )
    payload = json.loads(args.config.read_text())
    corpus_path = repository_root / payload["corpusFreeze"]["path"]
    corpus = json.loads(corpus_path.read_text())
    plan = build_plan(payload, corpus)
    plan["experimentValidation"] = validation
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        args.output_plan.write_text(encoded)
    if not args.execute:
        print(encoded, end="")
        return
    result = execute_plan(plan, repository_root, args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

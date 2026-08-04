#!/usr/bin/env python3
"""Resumable, fail-closed orchestration for the v28 overnight preflop solve."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOLVER_ROOT = ROOT / "preflop-solver"
RUN_ROOT = SOLVER_ROOT / "neural" / "runs" / "v28-overnight"
LOG_ROOT = RUN_ROOT / "logs"
SOLVER = SOLVER_ROOT / "target" / "release" / "preflop-solver"
PYTHON = SOLVER_ROOT / ".venv-neural" / "bin" / "python"
COMPARE = SOLVER_ROOT / "neural" / "compare_tabular_preflop.py"
GATES = SOLVER_ROOT / "neural" / "validate_v28_gates.py"
SEEDS = (8801, 8802)

CACHES = {
    "t1": RUN_ROOT / "cache-t1.json.gz",
    "t2": RUN_ROOT / "cache-t2.json.gz",
    "v": RUN_ROOT / "cache-v.json.gz",
    "h": RUN_ROOT / "cache-h.json.gz",
}


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with (LOG_ROOT / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def valid_json(path: Path) -> bool:
    try:
        return path.is_file() and bool(read_json(path))
    except (OSError, ValueError, RuntimeError):
        return False


def run_command(name: str, command: list[str], expected: Path | None = None) -> None:
    if expected is not None and (
        valid_json(expected) if expected.suffix == ".json" else expected.is_file()
    ):
        log(f"skip {name}; verified output already exists")
        return
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"start {name}")
    with (LOG_ROOT / f"{name}.log").open("wb") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with status {result.returncode}")
    if expected is not None and not expected.is_file():
        raise RuntimeError(f"{name} did not produce {expected}")
    log(f"complete {name}")


def run_parallel(specifications: list[tuple[str, list[str], Path]]) -> None:
    running: list[tuple[str, subprocess.Popen[bytes], Any, Path]] = []
    for name, command, expected in specifications:
        if valid_json(expected):
            log(f"skip {name}; verified output already exists")
            continue
        log(f"start {name}")
        handle = (LOG_ROOT / f"{name}.log").open("wb")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append((name, process, handle, expected))
    failures: list[str] = []
    for name, process, handle, expected in running:
        status = process.wait()
        handle.close()
        if status != 0 or not valid_json(expected):
            failures.append(f"{name}:{status}")
        else:
            log(f"complete {name}")
    if failures:
        raise RuntimeError("parallel stage failed: " + ", ".join(failures))


def capture_json(name: str, command: list[str], output: Path) -> None:
    if valid_json(output):
        log(f"skip {name}; verified output already exists")
        return
    log(f"start {name}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (LOG_ROOT / f"{name}.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with status {result.returncode}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} did not emit a JSON object")
    atomic_json(output, value)
    log(f"complete {name}")


def wait_for_caches(timeout_seconds: int = 8 * 60 * 60) -> None:
    started = time.monotonic()
    while True:
        missing = [name for name, path in CACHES.items() if not path.is_file()]
        temporary = [path for path in CACHES.values() if path.with_suffix(".tmp").exists()]
        if not missing and not temporary:
            log("all four atomic cache outputs are present; H remains unopened")
            return
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"cache wait timed out; missing={missing}")
        log(f"waiting for caches; missing={','.join(missing)}")
        time.sleep(60)


def inspect_cache(name: str) -> Path:
    output = RUN_ROOT / f"cache-{name}-summary.json"
    capture_json(
        f"inspect-cache-{name}",
        [str(SOLVER), "preflop-cache-inspect", "--cache", str(CACHES[name])],
        output,
    )
    return output


def merge_training_caches() -> Path:
    output = RUN_ROOT / "cache-t1-t2-merged.json.gz"
    run_command(
        "merge-cache-t1-t2",
        [
            str(SOLVER),
            "preflop-cache-merge",
            "--cache-a",
            str(CACHES["t1"]),
            "--cache-b",
            str(CACHES["t2"]),
            "--output",
            str(output),
        ],
        output,
    )
    return output


def policy_path(stage: str, seed: int, iterations: int) -> Path:
    return RUN_ROOT / f"{stage}-seed{seed}-r{iterations}.json"


def train_pair(stage: str, cache: Path, iterations: int) -> list[Path]:
    outputs = [policy_path(stage, seed, iterations) for seed in SEEDS]
    specifications = []
    for seed, output in zip(SEEDS, outputs):
        specifications.append(
            (
                f"train-{stage}-seed{seed}-r{iterations}",
                [
                    str(SOLVER),
                    "preflop-dcfr",
                    "--cache",
                    str(cache),
                    "--iterations",
                    str(iterations),
                    "--seed",
                    str(seed),
                    "--solver",
                    "dcfr",
                    "--model-version",
                    f"hu-20bb-v28-{stage}-seed{seed}-r{iterations}",
                    "--output",
                    str(output),
                ],
                output,
            )
        )
    run_parallel(specifications)
    return outputs


def train_split_pair(iterations: int = 10_000_000) -> list[Path]:
    outputs = [
        policy_path("t1", SEEDS[0], iterations),
        policy_path("t2", SEEDS[1], iterations),
    ]
    specifications = []
    for name, cache, seed, output in zip(
        ("t1", "t2"), (CACHES["t1"], CACHES["t2"]), SEEDS, outputs
    ):
        specifications.append(
            (
                f"train-{name}-seed{seed}-r{iterations}",
                [
                    str(SOLVER),
                    "preflop-dcfr",
                    "--cache",
                    str(cache),
                    "--iterations",
                    str(iterations),
                    "--seed",
                    str(seed),
                    "--solver",
                    "dcfr",
                    "--model-version",
                    f"hu-20bb-v28-{name}-seed{seed}-r{iterations}",
                    "--output",
                    str(output),
                ],
                output,
            )
        )
    run_parallel(specifications)
    return outputs


def evaluate_pair(
    label: str,
    policies: list[Path],
    cache: Path,
    seeds: tuple[int, int] = SEEDS,
) -> list[Path]:
    outputs = [RUN_ROOT / f"{label}-seed{seed}-evaluation.json" for seed in seeds]
    specifications = []
    for seed, policy, output in zip(seeds, policies, outputs):
        specifications.append(
            (
                f"evaluate-{label}-seed{seed}",
                [
                    str(SOLVER),
                    "preflop-evaluate",
                    "--cache",
                    str(cache),
                    "--policy",
                    str(policy),
                    "--output",
                    str(output),
                ],
                output,
            )
        )
    run_parallel(specifications)
    return outputs


def compare_pair(label: str, policies: list[Path]) -> Path:
    output = RUN_ROOT / f"{label}-cross-seed.json"
    capture_json(
        f"compare-{label}",
        [
            str(PYTHON),
            str(COMPARE),
            str(policies[0]),
            str(policies[1]),
        ],
        output,
    )
    return output


def candidate_summary(
    label: str, policies: list[Path], evaluations: list[Path], cross: Path
) -> dict[str, Any]:
    values = [read_json(path)["exploitability_bb_per_hand"] for path in evaluations]
    stability = read_json(cross)
    stable = (
        stability["reachWeightedActionFrequencyMae"] <= 0.05
        and stability["reachWeightedPrimaryAgreement"] >= 0.85
        and stability["maximumAggregateActionDelta"] <= 0.03
        and stability["lookupIntersectionCoverage"] >= 0.9999
    )
    return {
        "label": label,
        "policies": [str(path.relative_to(ROOT)) for path in policies],
        "evaluations": [str(path.relative_to(ROOT)) for path in evaluations],
        "crossSeed": str(cross.relative_to(ROOT)),
        "validationExploitabilityBbPerHand": values,
        "meanValidationExploitabilityBbPerHand": sum(values) / len(values),
        "worstValidationExploitabilityBbPerHand": max(values),
        "passesStability": stable,
    }


def should_extend(ten_million: dict[str, Any], hundred_million: dict[str, Any]) -> bool:
    improvement = (
        ten_million["meanValidationExploitabilityBbPerHand"]
        - hundred_million["meanValidationExploitabilityBbPerHand"]
    )
    return hundred_million["passesStability"] and (
        hundred_million["worstValidationExploitabilityBbPerHand"] <= 0.10
        or (
            hundred_million["worstValidationExploitabilityBbPerHand"] <= 0.20
            and improvement >= 0.005
        )
    )


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [candidate for candidate in candidates if candidate["passesStability"]]
    if not stable:
        raise RuntimeError("no paired candidate passed the stability gates")
    return min(
        stable,
        key=lambda candidate: (
            candidate["meanValidationExploitabilityBbPerHand"],
            candidate["worstValidationExploitabilityBbPerHand"],
            candidate["label"],
        ),
    )


def compact_pair(label: str, policies: list[Path]) -> list[Path]:
    outputs = [RUN_ROOT / f"{label}-seed{seed}.bin" for seed in SEEDS]
    for seed, policy, output in zip(SEEDS, policies, outputs):
        run_command(
            f"compact-{label}-seed{seed}",
            [
                str(SOLVER),
                "preflop-compact",
                "--policy",
                str(policy),
                "--output",
                str(output),
            ],
            output,
        )
    return outputs


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        wait_for_caches()
        # Inspecting T1/T2/V is permitted before selection. H is deliberately
        # not read, inspected, compared, or evaluated in this section.
        cache_summaries = [inspect_cache(name) for name in ("t1", "t2", "v")]
        merged = merge_training_caches()
        merged_summary = RUN_ROOT / "cache-t1-t2-merged-summary.json"
        capture_json(
            "inspect-cache-t1-t2-merged",
            [str(SOLVER), "preflop-cache-inspect", "--cache", str(merged)],
            merged_summary,
        )

        split_policies = train_split_pair()
        split_v = evaluate_pair("split-r10000000-v", split_policies, CACHES["v"])
        split_cross = compare_pair("split-r10000000", split_policies)
        baseline_policies = [
            SOLVER_ROOT
            / "neural"
            / "runs"
            / f"20bb-v27-mixed10-tabular-seed{seed}-r10000000.json"
            for seed in (7601, 7602)
        ]
        baseline_v = evaluate_pair(
            "v27-r10000000-v", baseline_policies, CACHES["v"], (7601, 7602)
        )

        candidates: list[dict[str, Any]] = []
        ten_policies = train_pair("merged", merged, 10_000_000)
        ten_v = evaluate_pair("merged-r10000000-v", ten_policies, CACHES["v"])
        ten_cross = compare_pair("merged-r10000000", ten_policies)
        ten_summary = candidate_summary("merged-r10000000", ten_policies, ten_v, ten_cross)
        candidates.append(ten_summary)

        hundred_policies = train_pair("merged", merged, 100_000_000)
        hundred_v = evaluate_pair("merged-r100000000-v", hundred_policies, CACHES["v"])
        hundred_cross = compare_pair("merged-r100000000", hundred_policies)
        hundred_summary = candidate_summary(
            "merged-r100000000", hundred_policies, hundred_v, hundred_cross
        )
        candidates.append(hundred_summary)

        if should_extend(ten_summary, hundred_summary):
            log("100M candidate is stable and improving; extend paired solve to 300M")
            long_policies = train_pair("merged", merged, 300_000_000)
            long_v = evaluate_pair("merged-r300000000-v", long_policies, CACHES["v"])
            long_cross = compare_pair("merged-r300000000", long_policies)
            candidates.append(
                candidate_summary(
                    "merged-r300000000", long_policies, long_v, long_cross
                )
            )
        else:
            log("100M validation does not justify a 300M extension")

        selected = select_candidate(candidates)
        selected_policies = [ROOT / path for path in selected["policies"]]
        log(f"selected on V: {selected['label']}; opening H exactly once for final evaluation")
        holdout_evaluations = evaluate_pair(
            f"{selected['label']}-h", selected_policies, CACHES["h"]
        )
        holdout_summary = inspect_cache("h")
        compact = compact_pair(selected["label"], selected_policies)
        projected_storage = sum(path.stat().st_size for path in compact)
        gate_output = RUN_ROOT / "v28-release-gates.json"
        selected_cross = ROOT / selected["crossSeed"]
        run_command(
            "validate-v28-release-gates",
            [
                str(PYTHON),
                str(GATES),
                "--policy-a",
                str(selected_policies[0]),
                "--policy-b",
                str(selected_policies[1]),
                "--evaluation-a",
                str(holdout_evaluations[0]),
                "--evaluation-b",
                str(holdout_evaluations[1]),
                "--cross-seed",
                str(selected_cross),
                "--projected-storage-bytes",
                str(projected_storage),
                "--output",
                str(gate_output),
            ],
            gate_output,
        )
        gates = read_json(gate_output)
        result = {
            "schema": "hu-v28-overnight-result-v1",
            "status": gates["status"],
            "activated": gates["allPassed"],
            "cacheSummaries": [str(path.relative_to(ROOT)) for path in cache_summaries],
            "mergedCacheSummary": str(merged_summary.relative_to(ROOT)),
            "holdoutCacheSummary": str(holdout_summary.relative_to(ROOT)),
            "splitTrainingDiagnostic": {
                "policies": [str(path.relative_to(ROOT)) for path in split_policies],
                "validation": [str(path.relative_to(ROOT)) for path in split_v],
                "crossSeed": str(split_cross.relative_to(ROOT)),
            },
            "v27BaselineValidation": [
                str(path.relative_to(ROOT)) for path in baseline_v
            ],
            "candidates": candidates,
            "selected": selected,
            "holdoutEvaluations": [
                str(path.relative_to(ROOT)) for path in holdout_evaluations
            ],
            "compactPolicies": [str(path.relative_to(ROOT)) for path in compact],
            "projectedStorageBytes": projected_storage,
            "releaseGates": str(gate_output.relative_to(ROOT)),
            "nextStep": None
            if gates["allPassed"]
            else "range_conditioned_postflop_value_oracle_pilot",
        }
        atomic_json(RUN_ROOT / "v28-overnight-result.json", result)
        log(f"pipeline complete with status={gates['status']}")
    except Exception as error:
        atomic_json(
            RUN_ROOT / "v28-pipeline-error.json",
            {
                "schema": "hu-v28-pipeline-error-v1",
                "error": type(error).__name__,
                "message": str(error),
            },
        )
        log(f"pipeline failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

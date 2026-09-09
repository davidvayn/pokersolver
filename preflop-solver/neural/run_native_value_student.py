"""Fit a short independent pair, then check actual native inference and held-out errors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

import native_value_dataset as native
from run_native_value_pilot import guarded, read_capture, test_command, PREDICT_TEST
from run_native_value_preflight import atomic_json, sha256
import train_public_value_network as training
from validate_public_value_parity import python_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-reference", type=Path)
    parser.add_argument("--split-reference-sha256")
    parser.add_argument("--refresh-training", action="store_true")
    parser.add_argument("--architecture", choices=("compact", "wide", "wide-pooled"), default="compact")
    parser.add_argument("--steps", choices=(600, 1200), type=int, default=600)
    parser.add_argument("--player-bias-weight", choices=(0.0, 1.0), type=float, default=0.0)
    parser.add_argument("--feature-schema", choices=(training.FEATURE_SCHEMA_BOARD_RELATIVE, training.FEATURE_SCHEMA_EXACT_RUNOUT), default=training.FEATURE_SCHEMA_BOARD_RELATIVE)
    args = parser.parse_args()
    for key in ("corpus", "binary", "output"): setattr(args, key, getattr(args, key).resolve())
    if args.output.exists() or sha256(args.corpus) != args.corpus_sha256 or sha256(args.binary) != args.binary_sha256:
        raise ValueError("existing output or pinned input mismatch")
    source = read_capture(args.corpus)
    # 508 retained states plus the bounded 128-label Path A extension. Existing
    # 256MiB decoded-input and 6GiB fitting guards remain unchanged.
    if not 256 <= len(source["targets"]) <= 640 or not source.get("source_captures"):
        raise ValueError("complete multi-family feasibility corpus required")
    reference = None
    if args.split_reference:
        args.split_reference = args.split_reference.resolve()
        if sha256(args.split_reference) != args.split_reference_sha256:
            raise ValueError("split reference hash mismatch")
        reference = read_capture(args.split_reference)
    elif args.split_reference_sha256:
        raise ValueError("split reference SHA has no input")
    expected_split = native.family_split(source, 10601, .25, .25, reference=reference,
                                         refresh_training=args.refresh_training)
    args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    neural = Path(__file__).resolve().parent
    code = [neural / name for name in ("train_public_value_network.py", "native_value_dataset.py", "validate_public_value_parity.py")]
    identities = {str(path): sha256(path) for path in code}
    models = args.output / "models"
    command = [sys.executable, str(neural / "train_public_value_network.py"), "--dataset", str(args.corpus),
               "--output-dir", str(models), "--steps", str(args.steps), "--batch-size", "8",
               "--seeds", "10601,10602", "--split-seed", "10601", "--validation-fraction", ".25",
               "--tuning-fraction", ".25", "--architecture", args.architecture, "--variant-set", "range-only",
               "--feature-schema", args.feature_schema, "--feature-workers", "1",
               "--value-normalization", "payoff-exposure", "--learning-rate", ".0003",
               "--learning-rate-final", ".00003", "--adam-bias-correction",
               "--evaluation-interval", "50", "--early-stopping-patience", "6"]
    if reference is not None:
        command.extend(["--native-split-reference", str(args.split_reference),
                        "--native-split-reference-sha256", args.split_reference_sha256])
    if args.refresh_training: command.append("--native-training-refresh")
    if args.player_bias_weight:
        command.extend(["--player-bias-weight", str(args.player_bias_weight)])
    record = dict(schema="native-value-student-pair-controller-v1", status="running", releaseAccepted=False,
                  corpusSha256=args.corpus_sha256, binarySha256=args.binary_sha256,
                  codeSha256=identities, runnerSha256=sha256(Path(__file__)), command=command,
                  startedAtUnix=time.time(), predictions=[])
    record["splitReferenceSha256"] = args.split_reference_sha256
    record["trainingRefresh"] = args.refresh_training
    record["split"] = {k: v.tolist() for k, v in zip(("train", "tuning", "holdout"), expected_split)}
    atomic_json(args.output / "manifest.json", record)
    try:
        record["training"] = guarded(command, {}, args.output / "training", 1800, 6 * 1024**3, stop)
        report = json.loads((models / "turn-value-paired-report.json").read_text())
        if report["loss"].get("playerBiasWeight", 0.0) != args.player_bias_weight:
            raise ValueError("training changed the declared value-bias objective")
        if report["datasetSha256"] != args.corpus_sha256 or report["splitUnit"] != "suit_canonical_flop_family_all_turns_histories_iterations":
            raise ValueError("training input or split contract drift")
        if (report.get("nativeSplitReferenceSha256") != args.split_reference_sha256
                or report.get("nativeTrainingRefresh",False) != args.refresh_training
                or any(report[name] != indices.tolist() for name, indices in
                       zip(("trainStates", "tuningStates", "validationStates"), expected_split))):
            raise ValueError("training changed the pinned family split")
        if report["validation"]["status"] == "accepted":
            raise ValueError("research source was incorrectly release-accepted")
        dataset = training.load_dataset(args.corpus, 1, "payoff-exposure")
        for entry in report["variants"]["range"]:
            if stop.is_set(): raise ValueError("operator stopped student verification")
            model_path = models / entry["weights"]
            model_sha = sha256(model_path)
            model = json.loads(model_path.read_text())
            if model.get("predictionContract") != "native-turn-cfv-full-stack-v1" or model["sourceDatasetSha256"] != args.corpus_sha256:
                raise ValueError("export prediction contract or corpus identity drift")
            output = args.output / ("predictions-%d.json" % entry["seed"])
            env = {"POKER_NATIVE_PREDICT_DATASET": str(args.corpus), "POKER_NATIVE_PREDICT_DATASET_SHA": args.corpus_sha256,
                   "POKER_NATIVE_PREDICT_MODEL": str(model_path), "POKER_NATIVE_PREDICT_MODEL_SHA": model_sha,
                   "POKER_NATIVE_PREDICT_OUTPUT": str(output)}
            worker = guarded(test_command(args.binary, PREDICT_TEST), env, args.output / ("predict-%d" % entry["seed"]), 900, stop=stop)
            prediction = json.loads(output.read_text())
            values = np.asarray(prediction["predictions"])
            if prediction["modelSha256"] != model_sha or values.shape != (len(source["targets"]), 2, native.COMBO_COUNT) or not np.isfinite(values).all():
                raise ValueError("invalid native inference output")
            # Every board, range and zero-own-reach holding in this corpus; not
            # just a favourable first state or only positive-own-reach queries.
            maximum_error = max(float(np.max(np.abs(python_prediction(dataset, model, i) - values[i]))) for i in range(len(values)))
            if maximum_error > 1e-4:
                raise ValueError("Python/Rust native prediction mismatch: %g bb" % maximum_error)
            metrics = {}
            for name, indices in (("tuning", report["tuningStates"]), ("holdout", report["validationStates"])):
                indices = np.asarray(indices)
                truth = dataset.targets[indices] * dataset.target_scales[indices, None]
                predicted = values[indices].reshape((-1, 2 * native.COMBO_COUNT))
                authentic = dataset.projection_weights[indices].reshape(predicted.shape)
                metrics[name] = {
                    "counterfactual": training.weighted_metrics(truth, predicted, dataset.weights[indices], np.ones(len(indices))),
                    "authentic": training.weighted_metrics(truth, predicted, authentic, np.ones(len(indices))),
                }
            row = dict(seed=entry["seed"], model=str(model_path), modelSha256=model_sha,
                       predictionsSha256=sha256(output),
                       maximumParityErrorBb=maximum_error, metrics=metrics,
                       featureAndInferenceSeconds=prediction["seconds"], states=len(values), worker=worker)
            record["predictions"].append(row)
            atomic_json(args.output / "manifest.json", record)
            print(json.dumps(dict(event="native-student-verified", seed=entry["seed"], parityErrorBb=maximum_error,
                                  holdoutRmseBb=metrics["holdout"]["authentic"]["weightedRmseBb"])), flush=True)
        if (sha256(args.binary) != args.binary_sha256 or sha256(args.corpus) != args.corpus_sha256
                or (reference is not None and sha256(args.split_reference) != args.split_reference_sha256)
                or any(sha256(Path(p)) != h for p,h in identities.items())):
            raise ValueError("pinned inputs/code changed during student pilot")
        record["status"] = "complete"
        record["interpretation"] = "Fitted-artifact parity and value diagnostics only; independent policy response and serving acceptance still required."
    except (OSError, ValueError, KeyError) as error:
        record["status"], record["failure"] = "failed", str(error)
    record["elapsedSeconds"] = time.time() - record["startedAtUnix"]
    atomic_json(args.output / "manifest.json", record)
    print(json.dumps({k: record[k] for k in ("status", "elapsedSeconds", "releaseAccepted")} | {"failure":record.get("failure")}), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()

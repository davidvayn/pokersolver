"""All-state native parity and unchanged-function proof before pooled fitting."""
import gc
import json
from pathlib import Path
import signal
import threading

import mlx.core as mx
import numpy as np

import native_value_dataset as native
import train_public_value_network as training
from retained_initialization import RANGE_AUGMENTATION, import_retained_weights
from run_native_value_pilot import PREDICT_TEST, guarded, test_command
from run_native_value_preflight import atomic_json, sha256
from validate_public_value_parity import python_prediction


def preflight(args):
    if args.initialization_transform != RANGE_AUGMENTATION or args.architecture != "wide-pooled":
        raise ValueError("explicit function-preserving pooled transfer required")
    dataset = training.load_dataset(args.corpus, 1, "payoff-exposure")
    contexts, queries, _ = training.feature_dataset_cached(dataset,
        training.FEATURE_SCHEMA_EXACT_RUNOUT, 1, args.feature_cache)
    work = args.output / "initial-native-parity"; work.mkdir()
    rows = []
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    for seed in (10601, 10602):
        receipt = args.initial_models[seed]
        path = Path(receipt["path"])
        if sha256(path) != receipt["sha256"]:
            raise ValueError("retained initialization changed")
        source = json.loads(path.read_text())
        model = training.SharedComboValueNetwork(True, "wide-pooled", "payoff-exposure",
            training.FEATURE_SCHEMA_EXACT_RUNOUT)
        import_retained_weights(model, source, seed, RANGE_AUGMENTATION)
        exported = work / f"model-{seed}.json"
        training.export_model(model, exported, seed, dataset.source_sha256, native.SCHEMA,
            "research_only", dataset.source["source_policy_sha256"], "payoff-exposure")
        payload = json.loads(exported.read_text())
        del model; gc.collect(); mx.clear_cache()
        output = work / f"predictions-{seed}.json"
        worker = guarded(test_command(args.binary, PREDICT_TEST), dict(
            POKER_NATIVE_PREDICT_DATASET=str(args.corpus), POKER_NATIVE_PREDICT_DATASET_SHA=args.corpus_sha256,
            POKER_NATIVE_PREDICT_MODEL=str(exported), POKER_NATIVE_PREDICT_MODEL_SHA=sha256(exported),
            POKER_NATIVE_PREDICT_OUTPUT=str(output)), work / f"predict-{seed}", 900,
            2*1024**3, stop)
        values = np.asarray(json.loads(output.read_text())["predictions"])
        if values.shape != (615, 2, 1326) or not np.isfinite(values).all():
            raise ValueError("invalid full native initialization predictions")
        parity, preservation = 0., 0.
        for index in range(615):
            features = contexts[index], queries[index]
            pooled = python_prediction(dataset, payload, index, features)
            original = python_prediction(dataset, source, index, features)
            parity = max(parity, float(abs(pooled-values[index]).max()))
            preservation = max(preservation, float(abs(pooled-original).max()))
        if not np.isfinite([parity, preservation]).all() or max(parity, preservation) > .0001:
            raise ValueError("native parity or initial function preservation failed")
        if stop.is_set(): raise ValueError("initial parity stopped")
        rows.append(dict(seed=seed, states=615, maximumParityErrorBb=parity,
            maximumInitialFunctionDifferenceBb=preservation, modelSha256=sha256(exported),
            predictionsSha256=sha256(output), worker=worker))
        print(json.dumps(dict(event="pooled-initialization-verified", **{k: v for k, v in rows[-1].items()
            if k not in ("worker", "modelSha256", "predictionsSha256")})), flush=True)
        del values
    atomic_json(args.output / "initial-native-parity.json", dict(status="complete", variants=rows,
        transform=RANGE_AUGMENTATION, releaseAccepted=False))

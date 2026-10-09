"""Strict frozen-weight initialization, not optimizer/checkpoint resume.

Keep dataset prefix, seed lineage and family holdouts intact when extending a
cash fit. Validate every tensor before loading any parameter into the model.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import cash_value_dataset as datasets
from cash_profiles import PAYOFF_CONTRACT
from train_public_value_network import tower_payload


def _splits(rows, count):
    if not isinstance(rows, dict) or set(rows) != {"train", "tuning", "holdout"}:
        raise ValueError("initial cash training split is missing")
    flat = []
    for group in rows.values():
        if (not isinstance(group, list) or not group
                or any(type(index) is not int or not 0 <= index < count for index in group)):
            raise ValueError("initial cash training split has invalid rows")
        flat.extend(group)
    if sorted(flat) != list(range(count)):
        raise ValueError("initial cash training split does not partition its corpus")
    return rows


def initialize_cash_from_frozen(model, network: Path, dataset: Path,
                                current_source: dict, current_splits, seed: int, split_seed: int):
    if type(seed) is not int or not 0 <= seed < 2**32 or len(current_splits) != 3:
        raise ValueError("initial cash seed/split inputs are invalid")
    network_bytes = network.read_bytes(); dataset_bytes = dataset.read_bytes()
    report_path = network.parent / "report.json"; report_bytes = report_path.read_bytes()
    payload = json.loads(network_bytes); initial_source = json.loads(dataset_bytes)
    report = json.loads(report_bytes)
    datasets.validate_dataset(initial_source); datasets.validate_dataset(current_source)
    network_sha = hashlib.sha256(network_bytes).hexdigest()
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    if (report.get("schema") != "hu-cash-value-pilot-report-v1"
            or report.get("status") != "research_only" or report.get("active") is not False
            or report.get("network_sha256") != network_sha
            or report.get("source_dataset_sha256") != dataset_sha
            or type(report.get("seed")) is not int or type(report.get("split_seed")) is not int
            or report.get("seed") != seed or report.get("split_seed") != split_seed
            or type(report.get("steps")) is not int or not 0 < report["steps"] <= 10000
            or type(report.get("selected_step")) is not int
            or not 0 <= report["selected_step"] <= report["steps"]):
        raise ValueError("initial cash weights lack matching training provenance")
    if (initial_source["game"] != current_source["game"]
            or current_source["labels"][:len(initial_source["labels"])] != initial_source["labels"]
            or current_source["label_canonical_sha256"][:len(initial_source["labels"])]
                != initial_source["label_canonical_sha256"]):
        raise ValueError("initial cash corpus is not the unchanged current prefix")
    previous = _splits(report.get("split_rows"), len(initial_source["labels"]))
    current = _splits({key:np.asarray(rows).tolist() for key,rows in
                      zip(("train", "tuning", "holdout"),current_splits)},len(current_source["labels"]))
    if (previous["tuning"] != current["tuning"] or previous["holdout"] != current["holdout"]
            or not set(previous["train"]) <= set(current["train"])):
        raise ValueError("initial cash fit would cross the current held-out split")

    architecture = model.architecture
    schema = {"compact":datasets.NETWORK_SCHEMA, "wide":datasets.NETWORK_SCHEMA,
              "wide-pooled":datasets.POOLED_NETWORK_SCHEMA,
              "wide-blocker-pooled":datasets.BLOCKER_POOLED_NETWORK_SCHEMA,
              "wide-baseline-conditioned":datasets.BASELINE_CONDITIONED_NETWORK_SCHEMA}[architecture]
    contract = (datasets.BLOCKER_POOLED_CONTRACT if architecture == "wide-blocker-pooled" else
                datasets.BASELINE_CONDITIONED_CONTRACT if architecture == "wide-baseline-conditioned" else
                datasets.PREDICTION_CONTRACT)
    required = dict(schema=schema,architecture=architecture,featureSchema=model.feature_schema,
                    predictionContract=contract,payoffContract=PAYOFF_CONTRACT,
                    sourceDatasetSchema=datasets.SCHEMA,sourceDatasetSha256=dataset_sha,
                    sourceValidationStatus="research_only",sourceGame=current_source["game"],
                    rulesSha256=current_source["rules_sha256"],cashRules=current_source["game"]["cash_rules"],
                    usesExactRanges=True,targetScaleBb=20.,valueNormalization="pot",seed=seed,
                    baseline="exact-own-payoff-forced-turn-checkdown-44-compatible-rivers",
                    residualUnit="normalized_state_value_scale",rangeScale=1326,
                    projection="independent-full-stack-clip-and-board-mask-no-zero-sum")
    if (any(payload.get(key) != value for key,value in required.items())
            or payload.get("usesExactRanges") is not True or type(payload.get("seed")) is not int):
        raise ValueError("initial cash value contract/architecture/rules do not match")
    staged = []
    for name, exported, final_activation in (("context_tower", "contextTower", "relu"),
                                             ("query_tower", "queryTower", "relu"),
                                             ("head", "head", "linear")):
        expected = tower_payload(getattr(model,name),"relu",final_activation)
        actual = payload.get(exported)
        if not isinstance(actual,list) or len(actual) != len(expected):
            raise ValueError("initial cash tower length does not match")
        for index,(layer,shape) in enumerate(zip(actual,expected)):
            if (not isinstance(layer,dict) or any(type(layer.get(key)) is not int for key in ("inputSize","outputSize"))
                    or any(layer.get(key) != shape[key]
                    for key in ("inputSize", "outputSize", "activation"))):
                raise ValueError("initial cash tensor layout/activation does not match")
            for field,suffix,dimensions in (("weights","weight",(shape["outputSize"],shape["inputSize"])),
                                            ("biases","bias",(shape["outputSize"],))):
                values = layer.get(field)
                if (not isinstance(values,list) or len(values) != int(np.prod(dimensions))
                        or any(type(value) not in (int,float) for value in values)):
                    raise ValueError("initial cash tensor length/type does not match")
                array = np.asarray(values,dtype=np.float64)
                if not np.isfinite(array).all() or (np.abs(array)>np.finfo(np.float32).max).any():
                    raise ValueError("initial cash tensor is not finite FP32")
                staged.append((f"{name}.layers.{index*2}.{suffix}",mx.array(array.astype(np.float32).reshape(dimensions))))
    model.load_weights(staged)
    return dict(network_sha256=network_sha,dataset_sha256=dataset_sha,
                training_report_sha256=hashlib.sha256(report_bytes).hexdigest(),parent_seed=seed,
                source_training_steps=report["steps"],source_selected_step=report["selected_step"],
                optimizer_initialization="fresh-adam-no-resumed-moments")

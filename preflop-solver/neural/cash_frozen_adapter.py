"""Strict opt-in frozen-backbone cash adapter; never a release qualification.

The original v7 encoder/head remains embedded without tensor changes. The
small additional head is in net bb (not pot units), before independent clipping.
Its exact-money gate is fixed to the validated NL25 20bb research experiment.
"""
from __future__ import annotations
import copy
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cash_profiles import equal_cash_investment_units, rules_digest
from cash_profiles import PAYOFF_CONTRACT
from cash_value_dataset import (FROZEN_TAIL_NETWORK_SCHEMA, FROZEN_TAIL_CONTRACT,
    KERNEL_POOLED_NETWORK_SCHEMA)

ARCHITECTURE = "wide-kernel-pooled-frozen-tail-adapter"


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("frozen cash adapter requires lowercase SHA256 identities")
    return value


def validate_frozen_adapter(payload):
    adapter = payload.get("frozenTailAdapter")
    if payload.get("schema") != FROZEN_TAIL_NETWORK_SCHEMA:
        if adapter is not None:
            raise ValueError("legacy cash values cannot silently discard a frozen adapter")
        return None
    source = payload.get("sourceGame", {})
    if not isinstance(source, dict) or not isinstance(source.get("cash_rules"), dict):
        raise ValueError("frozen cash adapter requires a rules-pinned source game")
    rules = source["cash_rules"]
    if (payload.get("architecture") != ARCHITECTURE
            or payload.get("predictionContract") != FROZEN_TAIL_CONTRACT
            or payload.get("targetScaleBb") != 20.
            or source.get("effective_stack_bb") != 20.
            or rules.get("unitsPerBb") != 25
            or payload.get("cashRules") != rules
            or payload.get("valueNormalization") != "pot"
            or payload.get("featureSchema") != "rank-suit-invariant-combo-query-v2"
            or payload.get("sourceValidationStatus") != "research_only"
            or payload.get("payoffContract") != PAYOFF_CONTRACT
            or payload.get("usesExactRanges") is not True
            or payload.get("projection") != "independent-full-stack-clip-and-board-mask-no-zero-sum"
            or payload.get("baseline") != "exact-own-payoff-forced-turn-checkdown-44-compatible-rivers"
            or payload.get("rulesSha256") != rules_digest(rules)
            or not isinstance(adapter, dict)
            or set(adapter) != {"parentNetworkSha256", "parentDatasetSha256", "maximumRemainingUnits", "correctionUnit", "head"}
            or type(adapter.get("maximumRemainingUnits")) is not int
            or adapter.get("maximumRemainingUnits") != 50
            or adapter.get("correctionUnit") != "own-player-net-bb"):
        raise ValueError("frozen cash adapter metadata or money gate is incompatible")
    _digest(adapter["parentNetworkSha256"])
    _digest(adapter["parentDatasetSha256"])
    _digest(payload.get("sourceDatasetSha256"))
    head = adapter["head"]
    if not isinstance(head, list) or len(head) != 2:
        raise ValueError("frozen cash adapter requires a two-layer shared head")
    for layer, inputs, outputs, activation in zip(head, (256, 32), (32, 1), ("relu", "linear")):
        if (not isinstance(layer, dict) or type(layer.get("inputSize")) is not int
                or type(layer.get("outputSize")) is not int
                or layer.get("inputSize") != inputs or layer.get("outputSize") != outputs
                or layer.get("activation") != activation or layer.get("normalization") is not None
                or layer.get("residual", False) is not False
                or layer.get("normalizationWeights", []) or layer.get("normalizationBiases", [])):
            raise ValueError("frozen cash adapter tensor layout is incompatible")
        for field, count in (("weights", inputs * outputs), ("biases", outputs)):
            values = layer.get(field)
            if (not isinstance(values, list) or len(values) != count
                    or any(type(value) not in (int, float) for value in values)):
                raise ValueError("frozen cash adapter tensor type/length is invalid")
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all() or (np.abs(array) > np.finfo(np.float32).max).any():
                raise ValueError("frozen cash adapter tensor must be finite FP32")
    return adapter


def frozen_adapter_gate(source):
    rules = source["game"]["cash_rules"]
    if rules["unitsPerBb"] != 25 or source["game"]["effective_stack_bb"] != 20:
        raise ValueError("frozen cash adapter only supports the pinned 20bb money scale")
    result = []
    for label in source["labels"]:
        invested = equal_cash_investment_units(label["input"]["state"]["invested_bb"], 25)
        if invested >= 500:
            raise ValueError("all-in cash continuations require exact settlement, not a frozen adapter")
        result.append(500 - invested <= 50)
    return np.asarray(result, dtype=bool)


def make_frozen_adapter_payload(parent, parent_sha256, dataset_sha256, head):
    # Import at the exporter seam: the decoder imports this metadata module.
    from validate_cash_value_parity import cash_architecture
    if parent.get("schema") != KERNEL_POOLED_NETWORK_SCHEMA or cash_architecture(parent) != "wide-kernel-pooled":
        raise ValueError("frozen cash adapter requires an unmodified v7 cash parent")
    payload = copy.deepcopy(parent)
    payload.update(schema=FROZEN_TAIL_NETWORK_SCHEMA, architecture=ARCHITECTURE,
        predictionContract=FROZEN_TAIL_CONTRACT, sourceDatasetSha256=_digest(dataset_sha256),
        frozenTailAdapter=dict(parentNetworkSha256=_digest(parent_sha256),
            parentDatasetSha256=_digest(parent.get("sourceDatasetSha256")),
            maximumRemainingUnits=50, correctionUnit="own-player-net-bb", head=copy.deepcopy(head)))
    provenance_keys = ("checkpointSelectionCriterion", "residualInitialization", "gradientBatchSize",
        "gradientAccumulation", "cardPoolGradientContract")
    payload["parentTrainingProvenance"] = {key: payload.pop(key) for key in provenance_keys if key in payload}
    payload.update(checkpointSelectionCriterion="immutable-parent-plus-preregistered-final-adapter",
        residualInitialization="frozen-cash-parent-plus-zero-cold-adapter",
        gradientAccumulation="frozen-features-full-pass-adapter-only-one-adam-update")
    validate_frozen_adapter(payload)
    return payload


class FrozenTailAdapterHead(nn.Module):
    """Only these parameters are trainable in a frozen-feature pilot."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head.layers[-1].weight = mx.zeros_like(self.head.layers[-1].weight)
        self.head.layers[-1].bias = mx.zeros_like(self.head.layers[-1].bias)

    def __call__(self, values):
        return self.head(values).reshape((-1, 2, 1326))

#!/usr/bin/env python3
"""Small rule-pinned own-payoff continuation pilot; never release activation.

Reuses the audited Home feature/tower implementation, not its zero-sum
projection. Whole flop families are held out, including suit variants/turns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

import cash_value_dataset as cash_values
from cash_profiles import PAYOFF_CONTRACT
from cash_checkdown import exact_cash_checkdown_features
from cash_range_pooling import card_removed_opponent_pool
from native_value_dataset import family_split, legal_combos, training_weights
from serving_value_projection import OwnPayoffValueProjection, own_payoff_conservation_residual
from train_public_value_network import (SharedComboValueNetwork, build_features,
    export_model, FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT, value_scale_bb)

CASH_FEATURE_SCHEMAS = (FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT)


class OwnComboValueNetwork(SharedComboValueNetwork):
    def __init__(self, architecture="compact", feature_schema=FEATURE_SCHEMA_BOARD_RELATIVE):
        if architecture not in ("compact", "wide", "wide-pooled", "wide-blocker-pooled"):
            raise ValueError("cash pilot architecture must be compact, wide, wide-pooled, or wide-blocker-pooled")
        if feature_schema not in CASH_FEATURE_SCHEMAS:
            raise ValueError("cash pilot feature schema must be board-relative or exact-turn-runout")
        super().__init__(True, "wide-pooled" if architecture == "wide-blocker-pooled" else architecture, "pot", feature_schema)
        self.architecture = architecture
        # The exact checkdown is already a meaningful reference. Start the
        # learned future-betting correction at zero, not random multi-bb EVs.
        self.head.layers[-1].weight = mx.zeros_like(self.head.layers[-1].weight)
        self.head.layers[-1].bias = mx.zeros_like(self.head.layers[-1].bias)

    def raw_values(self, context, queries, joint_weights, scales, own_baseline, ranges=None):
        if self.architecture == "wide-blocker-pooled":
            if ranges is None:
                raise ValueError("blocker-conditioned cash pooling requires raw exact ranges")
            context_embedding = self.context_tower(context)
            query_embedding = self.query_tower(queries)
            reach = joint_weights / mx.maximum(mx.sum(joint_weights,axis=2,keepdims=True),1e-9)
            own = mx.sum(query_embedding*reach[:,:,:,None],axis=2)
            combined = mx.concatenate((mx.broadcast_to(context_embedding[:,:,None,:],query_embedding.shape),
                        mx.broadcast_to(own[:,:,None,:],query_embedding.shape),
                        card_removed_opponent_pool(query_embedding,ranges),query_embedding),axis=-1)
            correction = self.head(combined).reshape((-1,2,1326))
            return correction + own_baseline / scales[:,None,None]
        old_raw = super().raw_values(context, queries, joint_weights, scales)
        equity = queries[:,:,:,94]
        old_baseline = (equity * context[:,:,20,None] - (1-equity) * context[:,:,19,None]) * (20 / scales[:,None,None])
        return old_raw - old_baseline + own_baseline / scales[:,None,None]

    def __call__(self, context, queries, joint_weights, scales, own_baseline, legal, ranges=None):
        raw = self.raw_values(context, queries, joint_weights, scales, own_baseline, ranges)
        limit = 20.0 / scales[:, None, None]
        return (mx.clip(raw, -limit, limit) * legal[:, None, :]).reshape((-1, 2652))


def reference_predictions(model, tensors, scales, legal):
    """CPU FP32 inference matches native serving; GPU matmul is not this oracle.

    On the pilot host identical GPU weights differed by .0185bb, whereas CPU,
    NumPy and native agreed within 6e-6bb. Keep training/evaluation distinct.
    """
    with mx.stream(mx.cpu):
        raw = model.raw_values(*tensors[:5],tensors[6] if len(tensors)>6 else None)
        mx.eval(raw)
        raw_bb = np.array(raw) * scales[:, None, None]
    return OwnPayoffValueProjection(raw_bb, legal.astype(bool)).values


def feature_arrays(source: dict, feature_schema=FEATURE_SCHEMA_BOARD_RELATIVE):
    if feature_schema not in CASH_FEATURE_SCHEMAS:
        raise ValueError("cash pilot feature schema must be board-relative or exact-turn-runout")
    cash_values.validate_dataset(source)
    contexts, queries, weights, loss_weights, scales, baselines, legal, targets = [], [], [], [], [], [], [], []
    for label in source["labels"]:
        state = label["input"]["state"]
        board = np.asarray(state["board"], dtype=np.int16)
        ranges = np.asarray(state["ranges"], dtype=np.float64)
        masses = np.asarray(label["opponent_compatible_mass"], dtype=np.float64)
        context, query = build_features(board, state["actor"], np.asarray(state["invested_bb"]), ranges, masses, FEATURE_SCHEMA_BOARD_RELATIVE)
        baseline, equity = exact_cash_checkdown_features(board,ranges,state["invested_bb"],source["game"]["cash_rules"])
        if feature_schema == FEATURE_SCHEMA_EXACT_RUNOUT:
            # Shared v2/v3 features differ only at this equity slot. Reuse the
            # baseline's exact 44-river win/tie traversal instead of evaluating
            # every showdown a second time through the generic Home builder.
            query[:,:,94] = equity
        scale = value_scale_bb(state["invested_bb"], "pot")
        contexts.append(context); queries.append(query); weights.append(ranges * masses)
        loss_weights.append(training_weights(board, ranges, masses, .1).reshape(-1))
        scales.append(scale); legal.append(legal_combos(board))
        baselines.append(baseline)
        targets.append(np.asarray(label["counterfactual_values_bb"]).reshape(-1) / scale)
    return tuple(np.asarray(v, dtype=np.float32) for v in (contexts, queries, weights, scales, baselines, legal, targets, loss_weights))


def export_cash_model(model, path: Path, seed: int, source: dict, digest: str, accounting_loss_weight: float = 0., flop_leaf_loss_weight: float = 1., profile_value_loss_weight: float = 0.):
    export_model(model, path, seed, digest, cash_values.SCHEMA, "research_only", None, "pot")
    payload = json.loads(path.read_text())
    blocker_pooled = model.architecture == "wide-blocker-pooled"
    payload.update(schema=(cash_values.BLOCKER_POOLED_NETWORK_SCHEMA if blocker_pooled else
                          cash_values.POOLED_NETWORK_SCHEMA if model.pools_exact_ranges else cash_values.NETWORK_SCHEMA),
                   cashRules=source["game"]["cash_rules"],
                   sourceGame=source["game"], baseline="exact-own-payoff-forced-turn-checkdown-44-compatible-rivers",
                   rulesSha256=source["rules_sha256"], payoffContract=PAYOFF_CONTRACT,
                   predictionContract=cash_values.BLOCKER_POOLED_CONTRACT if blocker_pooled else cash_values.PREDICTION_CONTRACT,
                   residualInitialization="zero-final-linear-layer",
                   accountingLossWeight=accounting_loss_weight,
                   profileValueLossWeight=profile_value_loss_weight,
                   flopLeafLossWeight=flop_leaf_loss_weight,
                   projection="independent-full-stack-clip-and-board-mask-no-zero-sum",
                   limitations=["finite-budget conditioned turn reference; not full-game exploitability", "only fresh equal-investment turn roots; not yet serving"])
    if blocker_pooled:
        payload["rangeAggregation"] = "per-query-card-removed-opponent-and-joint-own-pooling"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def native_parity(binary: Path, network: Path, source: dict, expected: np.ndarray, directory: Path):
    maximum = 0.0
    for index, label in enumerate(source["labels"]):
        request = directory / f"parity-input-{index:04}.json"
        request.write_text(json.dumps(label["input"]) + "\n")
        result = subprocess.run([str(binary.resolve()), "cash-turn-value-predict", "--input", str(request), "--value-network", str(network)], check=True, text=True, capture_output=True, timeout=120)
        native = json.loads(result.stdout)
        if native["rules_sha256"] != source["rules_sha256"]:
            raise ValueError("native parity crossed cash rules")
        difference = float(np.max(np.abs(np.asarray(native["counterfactual_values_bb"]) - expected[index])))
        maximum = max(maximum, difference)
    if maximum > 1e-3:
        raise ValueError(f"cash native/Python prediction mismatch: {maximum}bb")
    return maximum


def cash_accounting_penalty(prediction, scales, joint_weights, house_target):
    """Known joint-belief house target supervises means; predictions stay raw."""
    own_bb = prediction.reshape((-1,2,1326)) * scales[:,None,None]
    means = mx.sum(own_bb * joint_weights,axis=2) / mx.maximum(mx.sum(joint_weights,axis=2),1e-9)
    residual = mx.sum(means,axis=1) + house_target
    return mx.mean(residual**2)


def cash_profile_value_penalty(prediction, scales, joint_weights, own_targets):
    """Supervise each known own-player mean, not just their sum or a projection.

    Equal-and-opposite mean errors can pass a house-sum diagnostic while
    allocating the players' values incorrectly. Targets come from the exact
    paired CFVs under the same compatible joint belief, never half the rake.
    """
    own_bb = prediction.reshape((-1,2,1326)) * scales[:,None,None]
    means = mx.sum(own_bb * joint_weights,axis=2) / mx.maximum(mx.sum(joint_weights,axis=2),1e-9)
    return mx.mean(mx.sum((means-own_targets)**2,axis=1))


def split_cash_families(source: dict, seed: int, reference: dict | None = None):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("cash split seed must be an integer in 0..2^32-1")
    merged = source.get("source_datasets")
    if merged is not None and merged.get("schema") == "hu-cash-value-source-prefix-merge-v2":
        if (reference is None or seed != merged.get("split_seed")
                or cash_values.identity_hash(reference) != merged.get("split_reference_canonical_sha256")):
            raise ValueError("cash training must retain its pinned split seed and original reference")
    split_source = {"game": source["game"], "targets": [{"board": l["input"]["state"]["board"]} for l in source["labels"]]}
    pinned = None
    if reference is not None:
        cash_values.validate_dataset(reference)
        if source["game"] != reference["game"] or source["labels"][:len(reference["labels"])] != reference["labels"]:
            raise ValueError("cash split reference must be the unchanged target prefix")
        pinned = {"game":reference["game"],"targets":[{"board":l["input"]["state"]["board"]} for l in reference["labels"]]}
    return family_split(split_source, seed, .2, .2,reference=pinned)


def context_loss_multipliers(source: dict, flop_leaf_loss_weight: float):
    """Retain authentic supervision while testing forced-coverage curricula.

Only captured, replayable frozen-flop leaf parents receive the multiplier.
It never alters reaches, native values, or tuning/holdout membership.
"""
    if not np.isfinite(flop_leaf_loss_weight) or not 0 < flop_leaf_loss_weight <= 1:
        raise ValueError("forced flop leaf loss weight must be finite in (0,1]")
    result = np.ones(len(source["labels"]), dtype=np.float32)
    parents = source.get("source_datasets", {}).get("sources", [])
    cursor = 0; leaf_rows = 0
    for parent in parents:
        size = len(parent["selected_rows"])
        if parent["dataset"].get("flop_leaf_provenance") is not None:
            result[cursor:cursor+size] = flop_leaf_loss_weight
            leaf_rows += size
        cursor += size
    if flop_leaf_loss_weight != 1 and (not parents or not leaf_rows or cursor != len(result)):
        raise ValueError("weighted coverage requires a validated mixed frozen-flop leaf corpus")
    return result


def run(dataset: Path, output: Path, seed: int, steps: int, binary: Path | None = None, accounting_loss_weight: float = 0., split_seed: int | None = None, split_reference: Path | None = None, flop_leaf_loss_weight: float = 1., architecture: str = "compact", profile_value_loss_weight: float = 0., feature_schema: str = FEATURE_SCHEMA_BOARD_RELATIVE):
    if not 0 < steps <= 10000:
        raise ValueError("cash pilot step budget must be 1..10000")
    if not np.isfinite(accounting_loss_weight) or not 0 <= accounting_loss_weight <= 100:
        raise ValueError("cash accounting loss weight must be finite and in 0..100")
    if not np.isfinite(profile_value_loss_weight) or not 0 <= profile_value_loss_weight <= 100:
        raise ValueError("cash profile value loss weight must be finite and in 0..100")
    if not np.isfinite(flop_leaf_loss_weight) or not 0 < flop_leaf_loss_weight <= 1:
        raise ValueError("forced flop leaf loss weight must be finite in (0,1]")
    if architecture not in ("compact", "wide", "wide-pooled", "wide-blocker-pooled"):
        raise ValueError("cash pilot architecture must be compact, wide, wide-pooled, or wide-blocker-pooled")
    if feature_schema not in CASH_FEATURE_SCHEMAS:
        raise ValueError("cash pilot feature schema must be board-relative or exact-turn-runout")
    split_seed = seed if split_seed is None else split_seed
    if type(split_seed) is not int or not 0 <= split_seed < 2**32:
        raise ValueError("cash split seed must be an integer in 0..2^32-1")
    data = dataset.read_bytes(); source = json.loads(data)
    # Matched pilots pin this seed independently of network initialization.
    # Earlier reports used different holdouts for each training seed and must
    # not be interpreted as same-board paired generalization comparisons.
    reference_bytes = split_reference.read_bytes() if split_reference else None
    reference = json.loads(reference_bytes) if reference_bytes else None
    train, tuning, holdout = split_cash_families(source, split_seed,reference)
    context_weights = context_loss_multipliers(source, flop_leaf_loss_weight)
    contexts, queries, weights, scales, baselines, legal, targets, loss_weights = feature_arrays(source,feature_schema)
    mx.random.seed(seed); model = OwnComboValueNetwork(architecture,feature_schema); optimizer = optim.Adam(learning_rate=1e-3)
    tensors = [mx.array(v) for v in (contexts, queries, weights, scales, baselines, legal)]
    tensors.append(mx.array(np.asarray([label["input"]["state"]["ranges"] for label in source["labels"]],dtype=np.float32)))
    target = mx.array(targets); loss_weight = mx.array(loss_weights)
    context_weight = mx.array(context_weights)
    # Supervised regularization against an independently traversed target
    # house ledger. Never infer rake by negating the predicted own values, add
    # rake as an inference feature, or project a failing prediction to pass.
    house = np.asarray([l["metrics"]["cash"]["expected_house_rake_bb"] for l in source["labels"]])
    house_target = mx.array(house.astype(np.float32))
    # Own means retain the teacher's actual allocation between the seats.
    # They are supervision only, never input features or inference repairs.
    from native_value_dataset import compatible_masses
    own_targets = []
    for label in source["labels"]:
        ranges = np.asarray(label["input"]["state"]["ranges"])
        joint = ranges * compatible_masses(ranges)
        own_targets.append(np.sum(joint*np.asarray(label["counterfactual_values_bb"]),axis=1)/joint.sum(axis=1))
    own_target = mx.array(np.asarray(own_targets,dtype=np.float32))

    def loss_components(current, rows, weight_training_contexts=True):
        prediction = current(*(v[rows] for v in tensors))
        errors = (prediction - target[rows]) * tensors[3][rows, None]
        weighted = loss_weight[rows]
        if weight_training_contexts:
            weighted = weighted * context_weight[rows, None]
        regression = mx.sum(weighted * errors**2) / mx.maximum(mx.sum(weighted), 1e-9)
        if accounting_loss_weight == 0 and profile_value_loss_weight == 0:
            return regression, mx.array(0.)
        penalty = mx.array(0.)
        if accounting_loss_weight:
            penalty += accounting_loss_weight * cash_accounting_penalty(prediction,tensors[3][rows],tensors[2][rows],house_target[rows])
        if profile_value_loss_weight:
            penalty += profile_value_loss_weight * cash_profile_value_penalty(prediction,tensors[3][rows],tensors[2][rows],own_target[rows])
        return regression, penalty

    def loss_fn(current, rows, weight_training_contexts=True):
        regression, penalty = loss_components(current,rows,weight_training_contexts)
        return regression + penalty

    loss_grad = nn.value_and_grad(model, loss_fn); train_rows = mx.array(train)
    initial_loss = float(loss_fn(model, train_rows).item())
    initial_mse = float(loss_components(model,train_rows)[0].item())
    best_parameters, best_tuning, best_step = None, float("inf"), 0
    for step in range(1, steps + 1):
        loss, gradients = loss_grad(model, train_rows); optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % 10 == 0 or step == steps:
            with mx.stream(mx.cpu):
                value = float(loss_fn(model, mx.array(tuning),False).item())
            if value < best_tuning:
                best_tuning, best_step = value, step
                best_parameters = [(key, np.array(value)) for key, value in tree_flatten(model.parameters())]
    model.load_weights([(key, mx.array(value)) for key, value in best_parameters])
    predicted = reference_predictions(model, tensors, scales, legal)
    native_weights = weights.astype(np.float64)
    # Recompute masses in f64 for the identical compatible joint distribution.
    for row, label in enumerate(source["labels"]):
        ranges = np.asarray(label["input"]["state"]["ranges"])
        native_weights[row] = ranges * compatible_masses(ranges)
    residuals = own_payoff_conservation_residual(predicted, native_weights, house)
    errors = predicted.reshape((-1,2652)) - targets * scales[:, None]
    held_loss = loss_weights[holdout]
    rmse = float(np.sqrt(np.sum(held_loss * errors[holdout]**2) / held_loss.sum()))
    baseline_errors = baselines.reshape((-1,2652)) - targets * scales[:,None]
    baseline_rmse = float(np.sqrt(np.sum(held_loss * baseline_errors[holdout]**2) / held_loss.sum()))
    output.mkdir(parents=True, exist_ok=True)
    network_path = output / "value-network.json"
    export_cash_model(model, network_path, seed, source, hashlib.sha256(data).hexdigest(), accounting_loss_weight, flop_leaf_loss_weight,profile_value_loss_weight)
    parity = native_parity(binary, network_path, source, predicted, output) if binary else None
    report = {"schema": "hu-cash-value-pilot-report-v1", "status": "research_only", "seed":seed, "architecture":architecture,"feature_schema":feature_schema,
              "rules_sha256": source["rules_sha256"], "source_dataset_sha256":hashlib.sha256(data).hexdigest(),
              "steps":steps,"selected_step":best_step,"training_initial_mse_bb":initial_mse,
              "split_seed":split_seed,
              "split_reference_sha256":hashlib.sha256(reference_bytes).hexdigest() if reference_bytes else None,
              "training_initial_objective_bb_squared":initial_loss,
              "accounting_loss_weight":accounting_loss_weight,
              "profile_value_loss_weight":profile_value_loss_weight,
              "flop_leaf_loss_weight":flop_leaf_loss_weight,
              "heldout_rmse_bb":rmse,"maximum_own_payoff_accounting_bias_bb":float(residuals.max()),
              "exact_checkdown_heldout_rmse_bb":baseline_rmse,"beats_checkdown_on_holdout":rmse < baseline_rmse,
              "native_python_maximum_difference_bb":parity,
              "native_binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest() if binary else None,
              "network_sha256":hashlib.sha256(network_path.read_bytes()).hexdigest(),
              "evaluation_backend":"mlx-cpu-fp32-native-parity", "training_backend":str(mx.default_device()),
              "split_rows":{"train":train.tolist(),"tuning":tuning.tolist(),"holdout":holdout.tolist()},
              "full_game_exploitability":"unmeasured", "active":False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--steps", type=int, default=50); parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--accounting-loss-weight", type=float, default=0.)
    parser.add_argument("--profile-value-loss-weight", type=float, default=0.,help="Supervise each seat's exact joint-belief profile mean; does not project inference values")
    parser.add_argument("--split-seed", type=int, help="Pin identical flop-family splits across independent training seeds")
    parser.add_argument("--split-reference",type=Path,help="Keep tuning/holdout families unchanged when adding training contexts")
    parser.add_argument("--flop-leaf-loss-weight",type=float,default=1.,help="Training weight for captured forced-coverage leaf parents; does not change beliefs or targets")
    parser.add_argument("--architecture",choices=("compact","wide","wide-pooled","wide-blocker-pooled"),default="compact")
    parser.add_argument("--feature-schema",choices=CASH_FEATURE_SCHEMAS,default=FEATURE_SCHEMA_BOARD_RELATIVE)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.seed, args.steps, args.native_binary, args.accounting_loss_weight, args.split_seed,args.split_reference,args.flop_leaf_loss_weight,args.architecture,args.profile_value_loss_weight,args.feature_schema), indent=2))


if __name__ == "__main__": main()

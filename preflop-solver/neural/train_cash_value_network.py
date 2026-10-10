#!/usr/bin/env python3
"""Small rule-pinned own-payoff continuation pilot; never release activation.

Reuses the audited Home feature/tower implementation, not its zero-sum
projection. Whole flop families are held out, including suit variants/turns.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

import cash_value_dataset as cash_values
from cash_profiles import PAYOFF_CONTRACT
from cash_checkdown import exact_cash_checkdown_features
from cash_feature_cache import CashFeatureCache
from cash_range_pooling import card_removed_opponent_pool
from cash_value_initialization import initialize_cash_from_frozen
from native_value_dataset import family_split, legal_combos, training_weights
from serving_value_projection import OwnPayoffValueProjection, own_payoff_conservation_residual
from train_public_value_network import (SharedComboValueNetwork, build_features,
    export_model, FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT, value_scale_bb)

CASH_FEATURE_SCHEMAS = (FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT)
CASH_ARCHITECTURES = ("compact", "wide", "wide-pooled", "wide-blocker-pooled", "wide-baseline-conditioned")
CASH_TRAINING_OUTPUT_MODES = ("serving-clipped", "raw")
_FEATURE_CACHE = CashFeatureCache()


class OwnComboValueNetwork(SharedComboValueNetwork):
    def __init__(self, architecture="compact", feature_schema=FEATURE_SCHEMA_BOARD_RELATIVE):
        if architecture not in CASH_ARCHITECTURES:
            raise ValueError("cash pilot architecture must be a supported own-payoff network")
        if feature_schema not in CASH_FEATURE_SCHEMAS:
            raise ValueError("cash pilot feature schema must be board-relative or exact-turn-runout")
        super().__init__(True, "wide-pooled" if architecture in ("wide-blocker-pooled","wide-baseline-conditioned") else architecture, "pot", feature_schema)
        self.architecture = architecture
        if architecture == "wide-baseline-conditioned":
            weight = self.head.layers[0].weight
            # Preserve every original feature and initialized parameter. The
            # appended exact-payoff feature initially has no effect, but its
            # coefficient can learn without removing current-board equity.
            self.head.layers[0].weight = mx.concatenate((weight,mx.zeros((weight.shape[0],1))),axis=1)
        # The exact checkdown is already a meaningful reference. Start the
        # learned future-betting correction at zero, not random multi-bb EVs.
        self.head.layers[-1].weight = mx.zeros_like(self.head.layers[-1].weight)
        self.head.layers[-1].bias = mx.zeros_like(self.head.layers[-1].bias)

    def raw_values(self, context, queries, joint_weights, scales, own_baseline, ranges=None):
        if self.architecture in ("wide-blocker-pooled","wide-baseline-conditioned"):
            if self.architecture == "wide-blocker-pooled" and ranges is None:
                raise ValueError("blocker-conditioned cash pooling requires raw exact ranges")
            context_embedding = self.context_tower(context)
            query_embedding = self.query_tower(queries)
            reach = joint_weights / mx.maximum(mx.sum(joint_weights,axis=2,keepdims=True),1e-9)
            own = mx.sum(query_embedding*reach[:,:,:,None],axis=2)
            opponent = (card_removed_opponent_pool(query_embedding,ranges) if self.architecture == "wide-blocker-pooled"
                        else mx.broadcast_to(own[:,::-1,None,:],query_embedding.shape))
            combined = mx.concatenate((mx.broadcast_to(context_embedding[:,:,None,:],query_embedding.shape),
                        mx.broadcast_to(own[:,:,None,:],query_embedding.shape),
                        opponent,query_embedding),axis=-1)
            if self.architecture == "wide-baseline-conditioned":
                combined = mx.concatenate((combined,own_baseline[:,:,:,None]/20.),axis=-1)
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


def export_cash_model(model, path: Path, seed: int, source: dict, digest: str, accounting_loss_weight: float = 0., flop_leaf_loss_weight: float = 1., profile_value_loss_weight: float = 0., adam_bias_correction: bool = False, learning_rate: float = 1e-3, *, regression_loss: str = "mse", huber_delta_bb: float = 1., initialization: dict | None = None, training_output_mode: str = "serving-clipped", gradient_batch_size: int | None = None):
    validate_cash_regression_options(regression_loss,huber_delta_bb)
    validate_cash_training_output_mode(training_output_mode)
    validate_cash_gradient_batch_size(gradient_batch_size)
    precision = cash_training_precision()
    export_model(model, path, seed, digest, cash_values.SCHEMA, "research_only", None, "pot")
    payload = json.loads(path.read_text())
    blocker_pooled = model.architecture == "wide-blocker-pooled"
    baseline_conditioned = model.architecture == "wide-baseline-conditioned"
    payload.update(schema=(cash_values.BASELINE_CONDITIONED_NETWORK_SCHEMA if baseline_conditioned else cash_values.BLOCKER_POOLED_NETWORK_SCHEMA if blocker_pooled else
                          cash_values.POOLED_NETWORK_SCHEMA if model.pools_exact_ranges else cash_values.NETWORK_SCHEMA),
                   cashRules=source["game"]["cash_rules"],
                   sourceGame=source["game"], baseline="exact-own-payoff-forced-turn-checkdown-44-compatible-rivers",
                   rulesSha256=source["rules_sha256"], payoffContract=PAYOFF_CONTRACT,
                   trainingTargetSemantics=cash_values.TARGET_SEMANTICS,
                   predictionContract=(cash_values.BASELINE_CONDITIONED_CONTRACT if baseline_conditioned else cash_values.BLOCKER_POOLED_CONTRACT if blocker_pooled else cash_values.PREDICTION_CONTRACT),
                   residualInitialization="zero-final-linear-layer",
                   accountingLossWeight=accounting_loss_weight,
                   profileValueLossWeight=profile_value_loss_weight,
                   adamBiasCorrection=adam_bias_correction,
                   learningRate=learning_rate,
                   regressionLoss=regression_loss, huberDeltaBb=huber_delta_bb,
                   regressionLossNormalization=("twice-standard-huber-local-mse-match" if regression_loss=="huber" else "squared-net-bb-error"),
                   checkpointSelectionCriterion="tuning-own-payoff-mse-plus-explicit-auxiliary-penalties",
                   flopLeafLossWeight=flop_leaf_loss_weight,
                   projection="independent-full-stack-clip-and-board-mask-no-zero-sum",
                   limitations=["finite-budget conditioned turn reference; not full-game exploitability", "only fresh equal-investment turn roots; not yet serving"])
    if training_output_mode != "serving-clipped":
        # Preserve complete legacy export bytes for the unchanged default.
        # This records a fit objective, never a different serving projection.
        payload["trainingOutputMode"] = training_output_mode
    if gradient_batch_size is not None:
        payload.update(gradientBatchSize=gradient_batch_size,
                       gradientAccumulation="full-pass-fixed-global-weight-normalization-one-adam-update")
    if precision["mlx_enable_tf32"] == "0":
        # This is a launch-time fit request, not a changed serving contract or
        # a claim that a warm parent's entire history used this precision.
        payload.update(trainingMatmulPrecisionRequest="full-float32",
                       trainingMlxVersion=precision["mlx_version"])
    if blocker_pooled:
        payload["rangeAggregation"] = "per-query-card-removed-opponent-and-joint-own-pooling"
    if baseline_conditioned:
        payload["headAdditionalInput"] = "exact-own-cash-checkdown-net-bb-divided-by-full-stack"
    if initialization is not None:
        payload.update(residualInitialization="frozen-cash-weights-with-fresh-adam",
                       initialValueNetworkSha256=initialization["network_sha256"],
                       initialTrainingDatasetSha256=initialization["dataset_sha256"],
                       initialTrainingReportSha256=initialization["training_report_sha256"],
                       initialModelSeed=initialization["parent_seed"],
                       initialModelTrainingSteps=initialization["source_training_steps"],
                       initialModelSelectedStep=initialization["source_selected_step"],
                       optimizerInitialization=initialization["optimizer_initialization"],
                       checkpointSelectionIncludesInitialModel=True)
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


def validate_cash_regression_options(regression_loss: str, huber_delta_bb: float):
    if type(regression_loss) is not str or regression_loss not in ("mse","huber"):
        raise ValueError("cash regression loss must be mse or huber")
    if (type(huber_delta_bb) not in (int,float) or not np.isfinite(huber_delta_bb)
            or not .01 <= huber_delta_bb <= 20):
        raise ValueError("cash Huber threshold must be finite in [0.01,20] net bb")


def cash_regression_losses(errors_bb, weights, regression_loss: str = "mse", huber_delta_bb: float = 1., *, normalization_mass: float | None = None):
    """Return unchanged weighted MSE and a separately selected fit objective.

    Errors arrive in net bb, not pot-scaled network units. Twice the standard
    Huber preserves MSE's local value/gradient; only its outlier gradients are
    bounded. No target/output clipping or inference projection is introduced.
    """
    validate_cash_regression_options(regression_loss,huber_delta_bb)
    squared = errors_bb**2
    if normalization_mass is not None and (type(normalization_mass) not in (int,float)
            or not np.isfinite(normalization_mass) or normalization_mass <= 0):
        raise ValueError("cash regression normalization mass must be finite and positive")
    # Accumulated batches use the WHOLE pass's weight mass. Independently
    # normalizing each batch changes the reach/curriculum-weighted objective.
    denominator = mx.maximum(mx.sum(weights),1e-9) if normalization_mass is None else normalization_mass
    mse = mx.sum(weights*squared)/denominator
    if regression_loss == "mse":
        return mse,mse
    absolute = mx.abs(errors_bb)
    terms = mx.where(absolute <= huber_delta_bb,squared,
                     huber_delta_bb*(2*absolute-huber_delta_bb))
    return mse,mx.sum(weights*terms)/denominator


def cash_loss_predictions(model, inputs, *, raw_output=False):
    """Prediction seam for fitting; serving always keeps independent clipping."""
    if raw_output:
        raw = model.raw_values(*inputs[:5], inputs[6] if len(inputs) > 6 else None)
        return (raw * inputs[5][:, None, :]).reshape((-1, 2652))
    return model(*inputs)


def validate_cash_training_output_mode(mode):
    if type(mode) is not str or mode not in CASH_TRAINING_OUTPUT_MODES:
        raise ValueError("cash training output mode must be serving-clipped or raw")


def validate_cash_gradient_batch_size(size):
    if size is not None and (type(size) is not int or not 1 <= size <= 10000):
        raise ValueError("cash gradient batch size must be an integer in 1..10000 or None")


def cash_training_precision():
    """Record the launch environment, never silently toggle a cached backend.

    MLX may use reduced-precision matrix units even with FP32 array dtypes.
    Set MLX_ENABLE_TF32=0 BEFORE starting Python for a full-FP32 fit request.
    CPU/native serving parity remains independently checked in either case.
    """
    flag = os.environ.get("MLX_ENABLE_TF32")
    if flag not in (None,"0","1"):
        raise ValueError("cash MLX_ENABLE_TF32 must be unset, 0 or 1 at process launch")
    return dict(mlx_version=version("mlx"), mlx_enable_tf32=flag,
                matmul_precision_request="full-float32" if flag == "0" else "library-default-or-tf32-enabled",
                environment_must_be_set_before_import=True)


def cash_accumulated_value_and_grad(model, loss_grad, rows, batch_size):
    """Sum additive globally normalized losses at ONE unchanged parameter set.

    Evaluate every gradient before advancing so activation graphs are released.
    The caller performs one optimizer update after the whole pass, never one
    per batch. FP32 reduction order differs from an unbatched gradient.
    """
    validate_cash_gradient_batch_size(batch_size)
    if batch_size is None or len(rows) == 0:
        raise ValueError("cash accumulation requires a batch size and nonempty rows")
    total, accumulated = mx.array(0.), None
    for start in range(0, len(rows), batch_size):
        loss, gradient = loss_grad(model, rows[start:start+batch_size])
        total = total + loss
        accumulated = gradient if accumulated is None else tree_map(
            lambda left,right: left+right, accumulated, gradient)
        mx.eval(total, accumulated)
    return total, accumulated


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


def run(dataset: Path, output: Path, seed: int, steps: int, binary: Path | None = None, accounting_loss_weight: float = 0., split_seed: int | None = None, split_reference: Path | None = None, flop_leaf_loss_weight: float = 1., architecture: str = "compact", profile_value_loss_weight: float = 0., feature_schema: str = FEATURE_SCHEMA_BOARD_RELATIVE, adam_bias_correction: bool = False, learning_rate: float = 1e-3, *, regression_loss: str = "mse", huber_delta_bb: float = 1., initial_value_network: Path | None = None, initial_dataset: Path | None = None, training_output_mode: str = "serving-clipped", gradient_batch_size: int | None = None):
    if ((initial_value_network is None) != (initial_dataset is None)
            or any(value is not None and not isinstance(value,Path)
                   for value in (initial_value_network,initial_dataset))):
        raise ValueError("initial cash weights require both network and dataset paths")
    if initial_value_network is not None and output.resolve() == initial_value_network.parent.resolve():
        raise ValueError("cash fit output cannot overwrite its frozen parent directory")
    if not 0 < steps <= 10000:
        raise ValueError("cash pilot step budget must be 1..10000")
    if not np.isfinite(accounting_loss_weight) or not 0 <= accounting_loss_weight <= 100:
        raise ValueError("cash accounting loss weight must be finite and in 0..100")
    if not np.isfinite(profile_value_loss_weight) or not 0 <= profile_value_loss_weight <= 100:
        raise ValueError("cash profile value loss weight must be finite and in 0..100")
    if not np.isfinite(flop_leaf_loss_weight) or not 0 < flop_leaf_loss_weight <= 1:
        raise ValueError("forced flop leaf loss weight must be finite in (0,1]")
    if architecture not in CASH_ARCHITECTURES:
        raise ValueError("cash pilot architecture must be a supported own-payoff network")
    if feature_schema not in CASH_FEATURE_SCHEMAS:
        raise ValueError("cash pilot feature schema must be board-relative or exact-turn-runout")
    if type(adam_bias_correction) is not bool:
        raise ValueError("cash Adam bias correction option must be a boolean")
    if (type(learning_rate) not in (int,float) or not np.isfinite(learning_rate)
            or not 0 < learning_rate <= .1):
        raise ValueError("cash learning rate must be finite in (0,0.1]")
    validate_cash_regression_options(regression_loss,huber_delta_bb)
    validate_cash_training_output_mode(training_output_mode)
    validate_cash_gradient_batch_size(gradient_batch_size)
    training_precision = cash_training_precision()
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
    dataset_digest = hashlib.sha256(data).hexdigest()
    arrays, feature_preparation = _FEATURE_CACHE.get(dataset_digest, feature_schema,
                                                    lambda: feature_arrays(source,feature_schema))
    contexts, queries, weights, scales, baselines, legal, targets, loss_weights = arrays
    mx.random.seed(seed); model = OwnComboValueNetwork(architecture,feature_schema)
    initialization = (initialize_cash_from_frozen(model,initial_value_network,initial_dataset,
                      source,(train,tuning,holdout),seed,split_seed)
                      if initial_value_network is not None else None)
    optimizer = optim.Adam(learning_rate=learning_rate,bias_correction=adam_bias_correction)
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
    train_rows = mx.array(train)
    normalization_mass = tuning_normalization_mass = None
    if gradient_batch_size is not None:
        # Preserve the existing global regression denominator and unweighted
        # context means for BOTH explicit auxiliary losses. A final short
        # batch must not acquire the weight of a full batch.
        normalization_mass = float(mx.sum(loss_weight[train_rows]*context_weight[train_rows,None]).item())
        tuning_normalization_mass = float(mx.sum(loss_weight[mx.array(tuning)]).item())
        if not np.isfinite(normalization_mass) or normalization_mass <= 0:
            raise ValueError("cash accumulated training requires positive finite total loss weight")
        if not np.isfinite(tuning_normalization_mass) or tuning_normalization_mass <= 0:
            raise ValueError("cash accumulated tuning requires positive finite total loss weight")

    def loss_components(current, rows, weight_training_contexts=True):
        # An overshot raw value receives zero corrective gradient through a
        # hard serving clip. The explicit raw pilot supervises that value;
        # tuning/checkpoint selection still uses the actual serving output.
        prediction = cash_loss_predictions(current, tuple(v[rows] for v in tensors),
            raw_output=training_output_mode == "raw" and weight_training_contexts)
        errors = (prediction - target[rows]) * tensors[3][rows, None]
        weighted = loss_weight[rows]
        if weight_training_contexts:
            weighted = weighted * context_weight[rows, None]
        mse,regression = cash_regression_losses(errors,weighted,regression_loss,huber_delta_bb,
            normalization_mass=normalization_mass if weight_training_contexts else tuning_normalization_mass)
        if accounting_loss_weight == 0 and profile_value_loss_weight == 0:
            return mse,regression,mx.array(0.)
        penalty = mx.array(0.)
        if accounting_loss_weight:
            penalty += accounting_loss_weight * cash_accounting_penalty(prediction,tensors[3][rows],tensors[2][rows],house_target[rows])
        if profile_value_loss_weight:
            penalty += profile_value_loss_weight * cash_profile_value_penalty(prediction,tensors[3][rows],tensors[2][rows],own_target[rows])
        if normalization_mass is not None:
            penalty *= len(rows)/len(train if weight_training_contexts else tuning)
        return mse,regression,penalty

    def loss_fn(current, rows, weight_training_contexts=True):
        _,regression,penalty = loss_components(current,rows,weight_training_contexts)
        return regression + penalty

    def tuning_scores(current):
        with mx.stream(mx.cpu):
            if gradient_batch_size is None:
                mse,_,penalty = loss_components(current,mx.array(tuning),False)
                return mse,penalty
            mse,penalty = mx.array(0.),mx.array(0.)
            rows = mx.array(tuning)
            for start in range(0,len(tuning),gradient_batch_size):
                partial_mse,_,partial_penalty = loss_components(current,rows[start:start+gradient_batch_size],False)
                mse,penalty = mse+partial_mse,penalty+partial_penalty
                mx.eval(mse,penalty)
            return mse,penalty

    loss_grad = nn.value_and_grad(model, loss_fn)
    if gradient_batch_size is None:
        initial_loss = float(loss_fn(model, train_rows).item())
        initial_mse = float(loss_components(model,train_rows)[0].item())
    else:
        initial_components = [mx.array(0.) for _ in range(3)]
        for start in range(0, len(train), gradient_batch_size):
            components = loss_components(model,train_rows[start:start+gradient_batch_size])
            initial_components = [a+b for a,b in zip(initial_components,components)]
            mx.eval(initial_components)
        initial_mse = float(initial_components[0].item())
        initial_loss = float((initial_components[1]+initial_components[2]).item())
    best_parameters, best_tuning, best_step = None, float("inf"), 0
    checkpoint_history = []
    if initialization is not None:
        # Do not replace a working frozen model if every attempted update
        # worsens the same tuning criterion. No holdout-based selection.
        mse,penalty = tuning_scores(model)
        with mx.stream(mx.cpu):
            best_tuning = float((mse+penalty).item())
        checkpoint_history.append(dict(step=0, pre_update_training_objective_bb_squared=None,
            tuning_mse_bb_squared=float(mse.item()), tuning_objective_bb_squared=best_tuning,
            improved_tuning=True))
        best_parameters = [(key,np.array(value)) for key,value in tree_flatten(model.parameters())]
    for step in range(1, steps + 1):
        if gradient_batch_size is None:
            loss, gradients = loss_grad(model, train_rows)
        else:
            loss, gradients = cash_accumulated_value_and_grad(model,loss_grad,train_rows,gradient_batch_size)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % 10 == 0 or step == steps:
            # Isolate the fit objective: checkpoint selection retains the
            # original MSE plus any explicitly requested auxiliary terms.
            tuning_mse,tuning_penalty = tuning_scores(model)
            with mx.stream(mx.cpu):
                value = float((tuning_mse+tuning_penalty).item())
            # Report already-evaluated scores, without extra forwards or a
            # changed criterion. The training loss is from BEFORE this update;
            # the tuning score is the serving output AFTER this update.
            checkpoint_history.append(dict(step=step,
                pre_update_training_objective_bb_squared=float(loss.item()),
                tuning_mse_bb_squared=float(tuning_mse.item()),
                tuning_objective_bb_squared=value, improved_tuning=value < best_tuning))
            if value < best_tuning:
                best_tuning, best_step = value, step
                best_parameters = [(key, np.array(value)) for key, value in tree_flatten(model.parameters())]
    model.load_weights([(key, mx.array(value)) for key, value in best_parameters])
    if gradient_batch_size is None:
        predicted = reference_predictions(model, tensors, scales, legal)
    else:
        predicted = np.concatenate([reference_predictions(model,
            [v[start:start+gradient_batch_size] for v in tensors],
            scales[start:start+gradient_batch_size],legal[start:start+gradient_batch_size])
            for start in range(0,len(source["labels"]),gradient_batch_size)])
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
    export_cash_model(model, network_path, seed, source, dataset_digest, accounting_loss_weight, flop_leaf_loss_weight,profile_value_loss_weight,adam_bias_correction,learning_rate,regression_loss=regression_loss,huber_delta_bb=huber_delta_bb,initialization=initialization,training_output_mode=training_output_mode,gradient_batch_size=gradient_batch_size)
    parity = native_parity(binary, network_path, source, predicted, output) if binary else None
    report = {"schema": "hu-cash-value-pilot-report-v1", "status": "research_only", "seed":seed, "architecture":architecture,"feature_schema":feature_schema,
              "rules_sha256": source["rules_sha256"], "source_dataset_sha256":dataset_digest,
              "feature_preparation":feature_preparation,
              "steps":steps,"selected_step":best_step,"training_initial_mse_bb":initial_mse,
              "adam_bias_correction":adam_bias_correction,"learning_rate":learning_rate,
              "regression_loss":regression_loss,"huber_delta_bb":huber_delta_bb,
              "training_output_mode":training_output_mode,
              "checkpoint_selection_criterion":"tuning-own-payoff-mse-plus-explicit-auxiliary-penalties",
              "checkpoint_history":checkpoint_history,
              "split_seed":split_seed,
              "initial_value_network":initialization,
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
              "training_precision":training_precision,
              "split_rows":{"train":train.tolist(),"tuning":tuning.tolist(),"holdout":holdout.tolist()},
              "full_game_exploitability":"unmeasured", "active":False}
    if gradient_batch_size is not None:
        report["gradient_batching"] = dict(batch_size=gradient_batch_size,
            microbatches_per_update=(len(train)+gradient_batch_size-1)//gradient_batch_size,
            contexts_per_optimizer_update=len(train), normalization_mass=normalization_mass,
            order="frozen-split-row-order", sampling="complete-pass-no-subsampling",
            optimizer_updates=steps, floating_point_reduction_order_changed=True)
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
    parser.add_argument("--architecture",choices=CASH_ARCHITECTURES,default="compact")
    parser.add_argument("--feature-schema",choices=CASH_FEATURE_SCHEMAS,default=FEATURE_SCHEMA_BOARD_RELATIVE)
    parser.add_argument("--adam-bias-correction",action="store_true",help="Explicit matched-pilot comparison; legacy default remains uncorrected")
    parser.add_argument("--learning-rate",type=float,default=1e-3,help="Explicit bounded pilot rate; original default remains unchanged")
    parser.add_argument("--regression-loss",choices=("mse","huber"),default="mse",help="Matched cash fit objective; checkpoint selection and reporting retain MSE")
    parser.add_argument("--huber-delta-bb",type=float,default=1.,help="Huber fit threshold in net bb; small-error gradients match original squared loss")
    parser.add_argument("--training-output-mode",choices=CASH_TRAINING_OUTPUT_MODES,default="serving-clipped",help="Explicit raw-value fit pilot; checkpoint selection and inference retain serving clipping")
    parser.add_argument("--gradient-batch-size",type=int,help="Accumulate the full globally weighted gradient in bounded context batches; one Adam update per complete pass")
    parser.add_argument("--initial-value-network",type=Path,help="Frozen cash weights with sibling report.json; same seed/architecture/rules/split required")
    parser.add_argument("--initial-dataset",type=Path,help="Byte-pinned initial training corpus; must remain an unchanged current prefix")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, args.seed, args.steps, args.native_binary, args.accounting_loss_weight, args.split_seed,args.split_reference,args.flop_leaf_loss_weight,args.architecture,args.profile_value_loss_weight,args.feature_schema,args.adam_bias_correction,args.learning_rate,regression_loss=args.regression_loss,huber_delta_bb=args.huber_delta_bb,initial_value_network=args.initial_value_network,initial_dataset=args.initial_dataset,training_output_mode=args.training_output_mode,gradient_batch_size=args.gradient_batch_size), indent=2))


if __name__ == "__main__": main()

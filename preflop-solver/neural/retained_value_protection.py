"""TRAIN-only frozen-value retention; a regularizer, never new poker truth."""
from __future__ import annotations

import math
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import train_public_value_network as training
from action_contrast_loss import BundleObjective, gradient_add, gradient_norm


def eligible_rows(groups, train_rows, projection_weights, boundary=508):
    rows = np.asarray(train_rows)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(groups)).any()):
        raise ValueError("invalid disjoint training row indices")
    joint = np.asarray(projection_weights)[rows, 0].sum(axis=1)
    result = rows[(np.asarray(groups)[rows] < boundary) & (joint > 1e-9)]
    if not len(result): raise ValueError("no eligible original positive-joint TRAIN states")
    return result


def retention_loss(prediction, reference, weights, scales):
    error = prediction - mx.stop_gradient(reference)
    def huber(values):
        absolute = mx.abs(values); quadratic = mx.minimum(absolute, .05)
        return .5 * quadratic * quadratic + .05 * (absolute - quadratic)
    return (mx.sum(weights * huber(error))
        + .25 * mx.sum(weights * huber(error * scales[:, None] / 20.))) / mx.maximum(mx.sum(weights), 1e-8)


def conditioned_weight(learning_norm, protection_norm):
    if not all(math.isfinite(value) and value > 0 for value in (learning_norm, protection_norm)):
        raise ValueError("positive finite TRAIN gradient norms required")
    return min(32., .5 * learning_norm / protection_norm)


class RetainedValueProtection:
    def __init__(self, objective, dataset, contexts, queries, train_rows, reference_model,
                 coefficient, seed, cadence=4, batch_size=8):
        if not math.isfinite(coefficient) or not 0 <= coefficient <= 32 or cadence < 1 or batch_size < 1:
            raise ValueError("invalid bounded retained-value protection")
        self.objective, self.dataset = objective, dataset
        self.contexts, self.queries = contexts, queries
        self.rows = eligible_rows(dataset.groups, train_rows, dataset.projection_weights)
        self.coefficient, self.cadence, self.batch_size = coefficient, cadence, batch_size
        self.rng = np.random.default_rng(seed ^ 0xA6C40)
        self.reference = np.zeros_like(dataset.targets)
        # All anchor inputs belong to the original TRAIN prefix. No control,
        # tuning, holdout or newly registered board can become an anchor.
        for start in range(0, len(self.rows), batch_size):
            rows = self.rows[start:start+batch_size]
            self.reference[rows] = np.asarray(reference_model(*self.inputs(rows)))
        self.reference.flags.writeable = False
        weights = dataset.projection_weights.reshape((-1, 2652)).copy()
        weights *= 2652 / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        self.weights = weights
        self.updates, self.draws, self.last = 0, [], None

    def inputs(self, rows):
        d = self.dataset
        return (mx.array(self.contexts[rows]), mx.array(self.queries[rows]),
                mx.array(d.projection_weights[rows]), mx.array(d.target_scales[rows]))

    def gradients(self, model, rows):
        args = self.inputs(rows)
        reference, weights = mx.array(self.reference[rows]), mx.array(self.weights[rows])
        def loss(current):
            return retention_loss(current(*args), reference, weights, args[-1])
        value, gradient = nn.value_and_grad(model, loss)(model)
        mx.eval(value, gradient)
        if not math.isfinite(float(value)):
            raise ValueError("nonfinite frozen-output protection")
        return gradient, dict(loss=float(value), gradientNorm=gradient_norm(gradient))

    def sample(self):
        return training.primary_replay_batch_rows(self.rng, self.rows, np.array([], dtype=np.int64),
            self.dataset.invested, self.batch_size, 0., np.ones(len(self.dataset.targets)))

    def accumulate(self, model, ordinary_gradients, value_loss_fn, step):
        gradient = self.objective.accumulate(model, ordinary_gradients, value_loss_fn, step)
        if not self.coefficient or step % self.cadence: return gradient
        rows = self.sample()
        protection, self.last = self.gradients(model, rows)
        self.updates += 1; self.draws.extend(rows.tolist())
        return gradient_add(gradient, protection, self.coefficient)

    def report(self):
        return dict(**self.objective.report(), retainedValueProtection=dict(
            coefficient=self.coefficient, cadence=self.cadence, batchSize=self.batch_size,
            updates=self.updates, anchorStates=len(self.rows), originalBoundary=508,
            trainingRows=self.rows.tolist(), draws=len(self.draws), uniqueDraws=len(set(self.draws)),
            positiveJointAuthenticReachOnly=True, target="frozen_retained_output_not_ground_truth",
            last=self.last, releaseAccepted=False))


def condition_protection(args):
    # Conditioning only on frozen TRAIN gradients at the predeclared step-200
    # full-precision checkpoint. Never choose lambda from a response score.
    import run_action_contrast_students as base
    from retained_initialization import import_retained_weights
    from run_native_value_preflight import atomic_json, sha256
    settings = json.loads((args.output / "fit-settings.json").read_text())
    dataset, contexts, queries, split, bundles, _ = base.prepare(args)
    train = np.flatnonzero(np.isin(dataset.groups, split[0]))
    reference = training.SharedComboValueNetwork(True, "wide", "payoff-exposure",
        training.FEATURE_SCHEMA_EXACT_RUNOUT)
    original = args.initial_models[10601]
    import_retained_weights(reference, json.loads(Path(original["path"]).read_text()), 10601)
    objective = BundleObjective(bundles, settings["contrastWeight"], settings["cadence"], settings["chunkSize"])
    protection = RetainedValueProtection(objective, dataset, contexts, queries, train, reference, 0., 10601)
    checkpoint = args.protection_checkpoint
    if sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("predeclared protection-conditioning checkpoint changed")
    current = training.SharedComboValueNetwork(True, "wide", "payoff-exposure",
        training.FEATURE_SCHEMA_EXACT_RUNOUT)
    import_retained_weights(current, json.loads(Path(checkpoint["path"]).read_text()), 10601)
    batch = training.primary_replay_batch_rows(np.random.default_rng(10601), train,
        np.array([], dtype=np.int64), dataset.invested, 8, 0., np.ones(len(dataset.targets)))
    args_batch = (mx.array(contexts[batch]), mx.array(queries[batch]),
        mx.array(dataset.projection_weights[batch]), mx.array(dataset.target_scales[batch]),
        mx.array(dataset.targets[batch]), mx.array(dataset.weights[batch]))
    _, learning = nn.value_and_grad(current, base.value_loss)(current, *args_batch)
    for bundle in bundles:
        calibration, contrast, _ = objective.gradients(current, bundle, base.value_loss)
        learning = gradient_add(learning, calibration, 1/len(bundles))
        learning = gradient_add(learning, contrast, settings["contrastWeight"]/len(bundles))
    rows = protection.sample()
    gradient, report = protection.gradients(current, rows)
    learning_norm = gradient_norm(learning)
    coefficient = conditioned_weight(learning_norm, report["gradientNorm"])
    result = dict(schema="retained-value-protection-settings-v1", status="complete",
        coefficient=coefficient, coefficientCap=32., requestedGradientFraction=.5,
        measuredGradientFraction=coefficient*report["gradientNorm"]/learning_norm,
        learningGradientNorm=learning_norm, protectionGradientNorm=report["gradientNorm"],
        conditioningSeed=10601, conditioningStep=200, checkpoint=checkpoint,
        conditioningRows=rows.tolist(), anchorRows=protection.rows.tolist(),
        cadence=4, batchSize=8, optimizerUpdatesPerStep=1, releaseAccepted=False)
    atomic_json(args.output / "protection-settings.json", result)
    print(json.dumps(dict(event="retained-protection-conditioned", coefficient=coefficient,
        anchorStates=len(protection.rows))), flush=True)

"""Exact two-pass gradients at unchanged weights; one optimizer update.

Each family/group is weighted once, independent of its number of live leaves.
The first pass integrates chance. The second accumulates calibrated values and
the affine contrast VJP, microbatching whole 2x1326 queries (never holdings).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from action_contrast_dataset import AffineGroup, contrast_loss_and_q_gradient, leaf_gradient


def gradient_add(first, second, scale=1.):
    return tree_map(lambda a, b: a + scale * b, first, second)


def gradient_norm(gradient):
    return math.sqrt(sum(float(mx.sum(v * v)) for _, v in tree_flatten(gradient)))


@dataclass
class TrainingBundle:
    dataset: object
    contexts: np.ndarray
    queries: np.ndarray
    groups: list[AffineGroup]
    family: tuple[int, ...]


class BundleObjective:
    def __init__(self, bundles, contrast_weight, cadence=4, chunk_size=4, serving_aligned=False):
        if (not bundles or not math.isfinite(contrast_weight) or contrast_weight < 0
                or contrast_weight > 10 or cadence < 1 or chunk_size < 1):
            raise ValueError("invalid bounded bundle objective")
        self.bundles, self.contrast_weight = bundles, contrast_weight
        self.cadence, self.chunk_size = cadence, chunk_size
        self.serving_aligned = serving_aligned
        self.updates = 0
        self.last = None
        self.scheduled_families = []

    def accumulate(self, model, ordinary_gradients, value_loss_fn, step):
        if step % self.cadence:
            return ordinary_gradients
        bundle = self.bundles[(step // self.cadence - 1) % len(self.bundles)]
        calibration, contrast, report = self.gradients(model, bundle, value_loss_fn)
        combined = gradient_add(ordinary_gradients, calibration)
        if self.contrast_weight: combined = gradient_add(combined, contrast, self.contrast_weight)
        self.last = report
        self.updates += 1
        self.scheduled_families.append(bundle.family)
        return combined

    def gradients(self, model, bundle, value_loss_fn):
        dataset = bundle.dataset
        count = len(dataset.targets)
        predictions = []
        for start in range(0, count, self.chunk_size):
            end = min(count, start + self.chunk_size)
            args = self.inputs(bundle, start, end)
            normalized = np.asarray(model.raw_values(*args) if self.serving_aligned else model(*args))
            predictions.append(normalized.reshape(-1, 2, 1326) * dataset.target_scales[start:end, None, None])
        predictions = np.concatenate(predictions)
        projection = None
        if self.serving_aligned:
            import native_value_dataset as native
            from serving_value_projection import BoundedValueProjection
            weights = np.asarray([ranges * native.compatible_masses(ranges) for ranges in dataset.ranges])
            legal = np.asarray([native.legal_combos(board) for board in dataset.boards])
            projection = BoundedValueProjection(predictions, weights, legal)
            predictions = projection.values
        # This derivative is in CONDITIONAL BB, and has already integrated all
        # common sampled turns. Leaf normalization is undone in the VJP below.
        ev_derivative = np.zeros_like(predictions, dtype=np.float64)
        contrast_loss = 0.
        for group in bundle.groups:
            weights = group.weights * group.support
            q = group.backup(predictions[:, group.actor])
            loss, dq = contrast_loss_and_q_gradient(q, group.target, weights)
            contrast_loss += loss / len(bundle.groups)
            ev_derivative[:, group.actor] += leaf_gradient(group, dq) / len(bundle.groups)
        if projection is not None:
            ev_derivative = projection.vjp(ev_derivative)
        calibration_grad = contrast_grad = None
        calibration_loss = 0.
        for start in range(0, count, self.chunk_size):
            end = min(count, start + self.chunk_size)
            args = self.inputs(bundle, start, end)
            targets, weights = mx.array(dataset.targets[start:end]), mx.array(dataset.weights[start:end])
            fraction = (end - start) / count
            def calibrated(current):
                return value_loss_fn(current, *args, targets, weights) * fraction
            derivative = mx.array((ev_derivative[start:end] * dataset.target_scales[start:end, None, None]).reshape(end-start, -1).astype(np.float32))
            def linear_vjp(current):
                output = current.raw_values(*args).reshape((end-start,-1)) if self.serving_aligned else current(*args)
                return mx.sum(output * derivative)
            cal_loss, cal_gradient = nn.value_and_grad(model, calibrated)(model)
            _, aux_gradient = nn.value_and_grad(model, linear_vjp)(model)
            mx.eval(cal_loss, cal_gradient, aux_gradient)
            calibration_loss += float(cal_loss)
            calibration_grad = cal_gradient if calibration_grad is None else gradient_add(calibration_grad, cal_gradient)
            contrast_grad = aux_gradient if contrast_grad is None else gradient_add(contrast_grad, aux_gradient)
            mx.eval(calibration_grad, contrast_grad)
        return calibration_grad, contrast_grad, dict(family=list(bundle.family),
            leaves=count, valueLoss=calibration_loss, contrastLoss=contrast_loss,
            valueGradientNorm=gradient_norm(calibration_grad), contrastGradientNorm=gradient_norm(contrast_grad))

    @staticmethod
    def inputs(bundle, start, end):
        d = bundle.dataset
        return (mx.array(bundle.contexts[start:end]), mx.array(bundle.queries[start:end]),
                mx.array(d.projection_weights[start:end]), mx.array(d.target_scales[start:end]))

    def report(self):
        return dict(contrastWeight=self.contrast_weight, cadence=self.cadence,
                    chunkSize=self.chunk_size, bundleUpdates=self.updates,
                    familyCounts={str(f): self.scheduled_families.count(f) for f in set(self.scheduled_families)},
                    last=self.last, optimizerUpdatesPerStep=1,
                    servingAlignedContrast=self.serving_aligned,
                    target="profile_on_support_consistent_holdings", offSupport="completed_value_calibration_only")

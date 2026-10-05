"""Finite, shape-checked TRAIN gradient diagnostics; not policy metrics."""
import math

import mlx.core as mx
from mlx.utils import tree_flatten


def gradient_dot(first, second):
    left, right = dict(tree_flatten(first)), dict(tree_flatten(second))
    if left.keys() != right.keys() or not left:
        raise ValueError("gradient parameter paths differ or are empty")
    total = 0.
    for key, value in left.items():
        other = right[key]
        if value.shape != other.shape:
            raise ValueError("gradient parameter shapes differ")
        if not bool(mx.all(mx.isfinite(value))) or not bool(mx.all(mx.isfinite(other))):
            raise ValueError("nonfinite gradient")
        total += float(mx.sum(value * other))
    if not math.isfinite(total): raise ValueError("nonfinite gradient dot product")
    return total


def gradient_relationship(reference, proposed):
    dot = gradient_dot(reference, proposed)
    reference_norm = math.sqrt(max(0., gradient_dot(reference, reference)))
    proposed_norm = math.sqrt(max(0., gradient_dot(proposed, proposed)))
    denominator = reference_norm * proposed_norm
    return dict(dot=dot, referenceNorm=reference_norm, proposedNorm=proposed_norm,
        cosine=max(-1., min(1., dot/denominator)) if denominator > 1e-16 else None,
        referenceLossDirectionalDerivativeUnderNegativeGradient=-dot,
        interpretation="First-order unpreconditioned gradient direction only; not an AdamW update or poker performance guarantee.")

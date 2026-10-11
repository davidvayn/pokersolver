"""Optional two-training-domain PCGrad; not a policy or safety guarantee.

Both gradients must already use the same global loss denominator. Project
against the original other-domain gradient, across all parameters together.
Adam preconditioning can change descent; validate actual updates separately.
"""
from __future__ import annotations

from mlx.utils import tree_flatten, tree_map
import numpy as np


def combine_cash_domain_gradients(previous, addition, *, project: bool = True):
    if type(project) is not bool:
        raise ValueError('cash gradient projection must be an explicit boolean')
    left, right = dict(tree_flatten(previous)), dict(tree_flatten(addition))
    if not left or left.keys() != right.keys():
        raise ValueError('cash gradient domains must contain identical parameters')
    product, left_squared, right_squared = 0., 0., 0.
    for name in left:
        # FP32 squared reductions can overflow or underflow for finite
        # gradients. Parameter vectors are small; scalar FP64 reductions
        # retain range while the actual projected tensors remain FP32.
        a, b = np.asarray(left[name],dtype=np.float64), np.asarray(right[name],dtype=np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('cash gradient domains have mismatched or nonfinite tensors')
        product += float(np.sum(a*b))
        left_squared += float(np.sum(a*a)); right_squared += float(np.sum(b*b))
    if not all(np.isfinite(v) for v in (product,left_squared,right_squared)):
        raise ValueError('cash gradient scalar reductions are not finite')
    left_norm, right_norm = float(np.sqrt(left_squared)), float(np.sqrt(right_squared))
    denominator = left_norm*right_norm
    cosine = float(np.clip(product/denominator,-1.,1.)) if denominator > 0 else None
    projected = project and product < 0
    if projected:
        # Negative dot implies both norms are positive. Do not sequentially
        # project against an already modified gradient or normalize domains.
        adjusted_left = tree_map(lambda a,b:a-(product/right_squared)*b,previous,addition)
        adjusted_right = tree_map(lambda a,b:b-(product/left_squared)*a,previous,addition)
        combined = tree_map(lambda a,b:a+b,adjusted_left,adjusted_right)
    else:
        combined = tree_map(lambda a,b:a+b,previous,addition)
    return combined, dict(dot=product,previous_norm=left_norm,addition_norm=right_norm,
                          cosine=cosine,projected=projected)

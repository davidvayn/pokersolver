"""Vectorized native clip/mask/bounded-zero-sum projection and implicit VJP.

Differentiate the weighted constraint, not the discrete bisection branches.
Weights/ranges are fixed inputs; gradients are with respect to raw values only.
"""
from __future__ import annotations

import numpy as np


class OwnPayoffValueProjection:
    """Online continuation values clip/mask ONLY; never manufacture zero sum.

    The two own-player heads can have different conditional values. Their
    accounting residual must be measured against independently generated house
    rake under a shared joint belief, not removed by shifting both heads.
    """
    def __init__(self, raw_bb, legal, depth=20.):
        raw, legal = np.asarray(raw_bb, dtype=np.float64), np.asarray(legal, dtype=bool)
        if (raw.ndim != 3 or raw.shape[1:] != (2, 1326)
                or legal.shape != (len(raw), 1326) or not legal.any(axis=1).all()
                or not np.isfinite(raw).all() or not np.isfinite(depth) or depth <= 0):
            raise ValueError("invalid own-payoff value projection")
        self.raw, self.legal, self.depth = raw, legal, depth
        self.values = np.where(legal[:, None], np.clip(raw, -depth, depth), 0.)

    def vjp(self, upstream):
        gradient = np.asarray(upstream, dtype=np.float64)
        if gradient.shape != self.raw.shape or not np.isfinite(gradient).all():
            raise ValueError("invalid own-payoff upstream gradient")
        return gradient * self.legal[:, None] * (abs(self.raw) < self.depth)


def own_payoff_conservation_residual(values, joint_weights, expected_house_rake):
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(joint_weights, dtype=np.float64)
    house = np.asarray(expected_house_rake, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (2, 1326) or weights.shape != values.shape
            or house.shape != (len(values),) or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isfinite(house).all() or (house < 0).any()):
        raise ValueError("invalid own-payoff joint-belief accounting")
    masses = weights.sum(axis=2)
    if (masses <= 0).any() or not np.allclose(masses[:, 0], masses[:, 1], rtol=1e-10, atol=1e-12):
        raise ValueError("both player heads must use the same compatible joint belief")
    aggregate = (values * weights).sum(axis=2) / masses
    return abs(aggregate.sum(axis=1) + house)


class BoundedValueProjection:
    def __init__(self, raw_bb, weights, legal, depth=20.):
        raw, weights, legal = np.asarray(raw_bb, dtype=np.float64), np.asarray(weights, dtype=np.float64), np.asarray(legal, dtype=bool)
        if (raw.ndim != 3 or raw.shape[1:] != (2, 1326) or weights.shape != raw.shape
                or legal.shape != (len(raw),1326) or not legal.any(axis=1).all()
                or not np.isfinite(raw).all() or not np.isfinite(weights).all()
                or (weights < 0).any() or (weights * ~legal[:,None]).any()
                or not np.isfinite(depth) or depth <= 0):
            raise ValueError("invalid bounded native projection")
        self.raw, self.weights, self.legal, self.depth = raw, weights, legal, depth
        mask = legal[:,None]
        clipped = np.where(mask, np.clip(raw,-depth,depth), 0.)
        joint = np.maximum(weights[:,0].sum(axis=1),1e-9)
        aggregate = (weights*clipped).sum(axis=2)/joint[:,None]
        midpoint = (aggregate[:,0]-aggregate[:,1])/2.
        target = np.stack((midpoint,-midpoint),axis=1)
        skip = abs(aggregate-target) <= 1e-12
        low = -depth-np.max(np.where(mask,clipped,-np.inf),axis=2)
        high = depth-np.min(np.where(mask,clipped,np.inf),axis=2)
        for _ in range(80):
            shift = (low+high)/2.
            shifted = (weights*np.clip(clipped+shift[:,:,None],-depth,depth)).sum(axis=2)/joint[:,None]
            below = shifted < target
            low, high = np.where(below,shift,low), np.where(below,high,shift)
        shift = np.where(skip,0.,(low+high)/2.)
        self.shifted = clipped+shift[:,:,None]
        self.values = np.where(mask,np.clip(self.shifted,-depth,depth),0.)

    def vjp(self, upstream):
        gradient = np.asarray(upstream,dtype=np.float64)
        if gradient.shape != self.raw.shape or not np.isfinite(gradient).all():
            raise ValueError("invalid native projection upstream gradient")
        active = self.legal[:,None] & (abs(self.shifted) < self.depth)
        first_clip = self.legal[:,None] & (abs(self.raw) < self.depth)
        total = (self.weights*active).sum(axis=2)
        summed = (gradient*active).sum(axis=2)
        multiplier = np.divide(summed,total,out=np.zeros_like(summed),where=total>0)
        # Each target is +/- (first player mean - second player mean)/2.
        # Its derivative couples both players, even if only one has EV loss.
        coupled = (multiplier[:,0]-multiplier[:,1])/2.
        target_gradient = np.stack((coupled,-coupled),axis=1)
        result = first_clip*(active*(gradient-multiplier[:,:,None]*self.weights)
            +target_gradient[:,:,None]*self.weights)
        if not np.isfinite(result).all(): raise ValueError("nonfinite implicit projection derivative")
        return result

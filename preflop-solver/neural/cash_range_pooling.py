"""Learned opponent embeddings conditioned on each query's exact card removal.

For query {a,b}, inclusion/exclusion removes all opponent hands containing
a or b, then restores {a,b}, which was subtracted twice. Raw opponent reaches
are necessary: global joint weights can censor hands for zero-own-reach CFVs.
"""
import mlx.core as mx
import numpy as np
from native_value_dataset import COMBOS
from train_public_value_network import RANGE_POOL_EPSILON

_incidence = np.zeros((1326, 52), np.float32)
_incidence[np.arange(1326), COMBOS[:, 0]] = 1.
_incidence[np.arange(1326), COMBOS[:, 1]] = 1.
INCIDENCE = mx.array(_incidence)
FIRST = mx.array(COMBOS[:, 0].astype(np.int32))
SECOND = mx.array(COMBOS[:, 1].astype(np.int32))
_first_incidence = np.zeros((1326,52),np.float32)
_second_incidence = np.zeros((1326,52),np.float32)
_first_incidence[np.arange(1326),COMBOS[:,0]] = 1.
_second_incidence[np.arange(1326),COMBOS[:,1]] = 1.
FIRST_INCIDENCE = mx.array(_first_incidence)
SECOND_INCIDENCE = mx.array(_second_incidence)
CARD_POOL_GRADIENT_CONTRACT = 'fixed-incidence-card-view-adjoint-v1'


@mx.custom_function
def _card_query_views(cards):
    """Unchanged indexed forwards; repeated-card gradients use a fixed sum.

    Each query selects its two distinct cards. The exact linear transpose
    accumulates first/second cotangents via their fixed one-hot incidence.
    Avoid GPU scatter-add's unordered float accumulation, not any gradient.
    Captured incidence arrays are constants, never learned parameters.
    """
    return cards[:,:,FIRST,:],cards[:,:,SECOND,:]


@_card_query_views.vjp
def _card_query_views_adjoint(primals,cotangents,outputs):
    first,second = cotangents
    adjoint = mx.matmul(mx.swapaxes(first,2,3),FIRST_INCIDENCE)
    adjoint += mx.matmul(mx.swapaxes(second,2,3),SECOND_INCIDENCE)
    return (mx.swapaxes(adjoint,2,3),)


def card_removed_opponent_pool(embeddings, ranges):
    weighted = embeddings * ranges[:, :, :, None]
    total = mx.sum(weighted, axis=2)
    cards = mx.swapaxes(mx.matmul(mx.swapaxes(weighted, 2, 3), INCIDENCE), 2, 3)
    first,second = _card_query_views(cards)
    numerator = total[:, :, None, :] - first - second + weighted
    card_mass = mx.matmul(ranges, INCIDENCE)
    first_mass,second_mass = _card_query_views(card_mass[:,:,:,None])
    mass = mx.sum(ranges, axis=2)[:, :, None] - first_mass[:,:,:,0] - second_mass[:,:,:,0] + ranges
    conditional = numerator / mx.maximum(mass[:, :, :, None], RANGE_POOL_EPSILON)
    conditional = mx.where(mass[:, :, :, None] > RANGE_POOL_EPSILON, conditional, 0.)
    return mx.stack((conditional[:, 1], conditional[:, 0]), axis=1)


def numpy_card_removed_opponent_pool(embeddings, ranges):
    """Independent dense decoder, using the same exact inclusion/exclusion."""
    weighted = embeddings * ranges[:, :, :, None]
    total = weighted.sum(axis=2)
    cards = np.swapaxes(np.swapaxes(weighted, 2, 3) @ _incidence, 2, 3)
    numerator = total[:, :, None, :] - cards[:, :, COMBOS[:, 0], :] - cards[:, :, COMBOS[:, 1], :] + weighted
    card_mass = ranges @ _incidence
    mass = ranges.sum(axis=2)[:, :, None] - card_mass[:, :, COMBOS[:, 0]] - card_mass[:, :, COMBOS[:, 1]] + ranges
    result = np.divide(numerator, np.maximum(mass[:, :, :, None], RANGE_POOL_EPSILON))
    result = np.where(mass[:, :, :, None] > RANGE_POOL_EPSILON, result, 0.)
    return result[:, ::-1]

"""Research-only positive-kernel opponent aggregation with exact card removal.

Inputs are finite nonnegative raw reaches and positive query/key features.
Values may be signed. For own query i, average opponent values with weights
range[j] * dot(query[i], key[j]) over card-compatible hands j only.

Reuse card-marginal inclusion/exclusion on key/value outer products: no
1326-by-1326 attention allocation. This is this finite positive kernel's exact
operator, not exact softmax attention, a poker oracle, or a serving fallback.
Compatible mass and the conditional kernel denominator must exceed the shared
1e-9 cutoff; undefined or below-cutoff queries return zero, as in native code.
"""
import mlx.core as mx
import numpy as np

from cash_range_pooling import (INCIDENCE,FIRST,SECOND,_incidence,
    card_removed_opponent_pool,numpy_card_removed_opponent_pool)
from train_public_value_network import RANGE_POOL_EPSILON


def _validate_shapes(values,ranges,queries,keys):
    if (len(values.shape)!=4 or values.shape[1:3]!=(2,1326) or values.shape[-1]<1
            or ranges.shape!=values.shape[:3] or len(queries.shape)!=4
            or queries.shape[:3]!=values.shape[:3] or queries.shape!=keys.shape
            or not 1<=queries.shape[-1]<=32):
        raise ValueError('cash kernel pooling requires paired1326-combo tensors and rank1..32')


def positive_kernel_opponent_pool(values,ranges,queries,keys):
    """Linear in combo count at fixed kernel/value widths; differentiable."""
    _validate_shapes(values,ranges,queries,keys)
    width=values.shape[-1]; rank=keys.shape[-1]
    augmented=mx.concatenate((values,mx.ones(values.shape[:3]+(1,))),axis=-1)
    products=(keys[:,:,:,:,None]*augmented[:,:,:,None,:]).reshape(values.shape[:3]+(rank*(width+1),))
    moments=card_removed_opponent_pool(products,ranges).reshape(values.shape[:3]+(rank,width+1))
    numerator=mx.sum(queries[:,:,:,:,None]*moments[:,:,:,:,:width],axis=3)
    denominator=mx.sum(queries*moments[:,:,:,:,-1],axis=3)
    # Count support with exact small integers. Floating-point subtraction of
    # probability sums can invent tiny compatible mass when every hand blocks
    # a query. Do not turn an undefined conditional distribution into values.
    support=(ranges>0).astype(mx.float32)
    cards=mx.matmul(support,INCIDENCE)
    compatible=mx.sum(support,axis=2)[:,:,None]-cards[:,:,FIRST]-cards[:,:,SECOND]+support
    defined=mx.stack((compatible[:,1],compatible[:,0]),axis=1)>0
    return mx.where((defined & (denominator>RANGE_POOL_EPSILON))[:,:,:,None],
        numerator/mx.maximum(denominator[:,:,:,None],RANGE_POOL_EPSILON),0.)


def numpy_positive_kernel_opponent_pool(values,ranges,queries,keys):
    """Independent decoder for the same finite-kernel/card-removal contract."""
    _validate_shapes(values,ranges,queries,keys)
    width=values.shape[-1]; rank=keys.shape[-1]
    augmented=np.concatenate((values,np.ones(values.shape[:3]+(1,),dtype=values.dtype)),axis=-1)
    products=(keys[:,:,:,:,None]*augmented[:,:,:,None,:]).reshape(values.shape[:3]+(rank*(width+1),))
    moments=numpy_card_removed_opponent_pool(products,ranges).reshape(values.shape[:3]+(rank,width+1))
    numerator=np.sum(queries[:,:,:,:,None]*moments[:,:,:,:,:width],axis=3)
    denominator=np.sum(queries*moments[:,:,:,:,-1],axis=3)
    support=(ranges>0).astype(np.float32); cards=support@_incidence
    compatible=support.sum(axis=2)[:,:,None]-cards[:,:,np.array(FIRST)]-cards[:,:,np.array(SECOND)]+support
    defined=compatible[:,::-1]>0
    return np.where((defined & (denominator>RANGE_POOL_EPSILON))[:,:,:,None],
        numerator/np.maximum(denominator[:,:,:,None],RANGE_POOL_EPSILON),0.)

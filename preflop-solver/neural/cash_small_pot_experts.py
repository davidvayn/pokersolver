"""Capacity-matched public-money routing probe, never a serving network.

Both arms have the same two 256 -> 16 -> 1 zero-output experts (8,258
parameters). The shared control averages both; the routed arm uses a fixed
smooth public-money partition. The parent, beliefs, targets and loss weights
are unchanged. Callers retain the outer paired-turn gate and own-value clip.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from cash_paired_regime_probe import paired_small_pot_coefficients
from cash_profiles import equal_cash_investment_units


def small_pot_expert_weights(source, mode):
    if type(mode) is not str or mode not in ('shared','routed'):
        raise ValueError('Small-pot expert probe requires shared or routed mode')
    # Reuse exact rule/card/money validation; do not change its outer gate.
    paired_small_pot_coefficients(source)
    if mode=='shared':
        return np.full((len(source['labels']),2),.5,np.float32)
    values=[]
    for label in source['labels']:
        cents=equal_cash_investment_units(label['input']['state']['invested_bb'],25)
        z=min(1.,max(0.,(cents-25)/25.)); bet=z*z*(3.-2.*z)
        values.append((1.-bet,bet))
    return np.asarray(values,np.float32).reshape((-1,2))


def _validated_heads(heads):
    if not isinstance(heads,list) or len(heads)!=2:
        raise ValueError('Small-pot expert probe requires exactly two heads')
    for head in heads:
        if not isinstance(head,list) or len(head)!=2:
            raise ValueError('Each small-pot expert requires two layers')
        for layer,inputs,outputs,activation in zip(head,(256,16),(16,1),('relu','linear')):
            if (not isinstance(layer,dict) or type(layer.get('inputSize')) is not int
                    or type(layer.get('outputSize')) is not int
                    or layer['inputSize']!=inputs or layer['outputSize']!=outputs
                    or layer.get('activation')!=activation or layer.get('normalization') is not None
                    or layer.get('residual',False) is not False
                    or layer.get('normalizationWeights',[]) or layer.get('normalizationBiases',[])):
                raise ValueError('Small-pot expert tensor layout is incompatible')
            for field,count in (('weights',inputs*outputs),('biases',outputs)):
                values=layer.get(field)
                if (not isinstance(values,list) or len(values)!=count
                        or any(type(v) not in (int,float) for v in values)):
                    raise ValueError('Small-pot expert tensor type or length is invalid')
                array=np.asarray(values,np.float64)
                if not np.isfinite(array).all() or (np.abs(array)>np.finfo(np.float32).max).any():
                    raise ValueError('Small-pot expert tensor must be finite FP32')
    return heads


def export_expert_heads(model):
    from train_public_value_network import tower_payload
    return [tower_payload(head,'relu','linear') for head in model.heads]


def restore_expert_heads(model,heads):
    _validated_heads(heads)
    for expert,payload in zip(model.heads,heads):
        for layer,row in zip((v for v in expert.layers if isinstance(v,nn.Linear)),payload):
            layer.weight=mx.array(np.asarray(row['weights'],np.float32).reshape(row['outputSize'],row['inputSize']))
            layer.bias=mx.array(np.asarray(row['biases'],np.float32))
    mx.eval(model.parameters())


def expert_dense_forward(inputs,heads,weights):
    """Independent FP32 NumPy audit; it does not export/qualify a native model."""
    from validate_cash_value_parity import dense_forward
    if (inputs.dtype!=np.float32 or inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)
            or not 1<=len(inputs)<=512 or not np.isfinite(inputs).all()
            or weights.dtype!=np.float32 or weights.shape!=(len(inputs),2)
            or not np.isfinite(weights).all() or (weights<0).any() or (weights>1).any()
            or not np.allclose(weights.sum(axis=1),1.,atol=np.finfo(np.float32).eps,rtol=0)):
        raise ValueError('Expected exact FP32 combo axes and convex public expert weights')
    _validated_heads(heads)
    values=[dense_forward(inputs,head).reshape((len(inputs),2,1326)) for head in heads]
    return values[0]*weights[:,0,None,None]+values[1]*weights[:,1,None,None]


class SmallPotExpertHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads=[nn.Sequential(nn.Linear(256,16),nn.ReLU(),nn.Linear(16,1)) for _ in range(2)]
        for head in self.heads:
            head.layers[-1].weight=mx.zeros_like(head.layers[-1].weight)
            head.layers[-1].bias=mx.zeros_like(head.layers[-1].bias)

    def __call__(self,inputs,weights):
        if (inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)
                or weights.shape!=(inputs.shape[0],2)
                or inputs.dtype!=mx.float32 or weights.dtype!=mx.float32):
            raise ValueError('Small-pot expert head requires pinned FP32 axes')
        values=[head(inputs).reshape((inputs.shape[0],2,1326)) for head in self.heads]
        return values[0]*weights[:,0,None,None]+values[1]*weights[:,1,None,None]

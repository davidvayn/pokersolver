"""Frozen-feature additive/FiLM contrast; never a serving or native schema.

All 32 hidden features remain active. Both arms have identical tensors and
neutral zero conditioning at initialization. The conditioner sees only the
existing broadcast public-context coordinates, not a query's private cards.
Callers preserve the outer public gate, exact beliefs and own-value settlement.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _mode(mode):
    if type(mode) is not str or mode not in ('additive','film'):
        raise ValueError('Context probe requires additive or film mode')


def _head(payload):
    shapes={'input_tower':(256,32),'context_tower':(64,64),'output_tower':(32,1)}
    if not isinstance(payload,dict) or set(payload)!=set(shapes):
        raise ValueError('Context probe requires exactly its three linear towers')
    for name,(inputs,outputs) in shapes.items():
        tower=payload[name]
        if not isinstance(tower,list) or len(tower)!=1 or not isinstance(tower[0],dict):
            raise ValueError('Context probe towers must be single linear layers')
        layer=tower[0]
        if (type(layer.get('inputSize')) is not int or type(layer.get('outputSize')) is not int
                or layer['inputSize']!=inputs or layer['outputSize']!=outputs
                or layer.get('activation')!='linear' or layer.get('normalization') is not None
                or layer.get('residual',False) is not False
                or layer.get('normalizationWeights',[]) or layer.get('normalizationBiases',[])):
            raise ValueError('Context probe tensor layout is incompatible')
        for field,count in (('weights',inputs*outputs),('biases',outputs)):
            values=layer.get(field)
            if (not isinstance(values,list) or len(values)!=count
                    or any(type(v) not in (int,float) for v in values)):
                raise ValueError('Context probe tensor type or length is invalid')
            array=np.asarray(values,np.float64)
            if not np.isfinite(array).all() or (np.abs(array)>np.finfo(np.float32).max).any():
                raise ValueError('Context probe tensors must be finite FP32')


def export_context_head(model):
    from train_public_value_network import tower_payload
    return {name:tower_payload(getattr(model,name),'linear','linear')
        for name in ('input_tower','context_tower','output_tower')}


def restore_context_head(model,payload):
    _head(payload)
    for name,tower in payload.items():
        row=tower[0]; layer=getattr(model,name).layers[0]
        layer.weight=mx.array(np.asarray(row['weights'],np.float32).reshape(row['outputSize'],row['inputSize']))
        layer.bias=mx.array(np.asarray(row['biases'],np.float32))
    mx.eval(model.parameters())


def context_dense_forward(inputs,payload,mode):
    """Independent CPU audit with an explicit public-context broadcast guard."""
    from validate_cash_value_parity import dense_forward
    _mode(mode); _head(payload)
    if (inputs.dtype!=np.float32 or inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)
            or not 1<=len(inputs)<=512 or not np.isfinite(inputs).all()
            or not np.array_equal(inputs[:,:,:,:64],
                np.broadcast_to(inputs[:,:,:1,:64],inputs[:,:,:,:64].shape))):
        raise ValueError('Context probe requires exact FP32 axes and broadcast public context')
    pre=dense_forward(inputs,payload['input_tower'])
    condition=dense_forward(inputs[:,:,:1,:64],payload['context_tower'])
    gamma,beta=np.split(condition,2,axis=-1)
    hidden=(np.maximum(pre+gamma,np.float32(0))+beta if mode=='additive' else
        np.maximum(pre,np.float32(0))*(np.float32(1)+gamma)+beta)
    return dense_forward(hidden,payload['output_tower']).reshape((len(inputs),2,1326))


class ContextConditionedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_tower=nn.Sequential(nn.Linear(256,32))
        self.context_tower=nn.Sequential(nn.Linear(64,64))
        self.output_tower=nn.Sequential(nn.Linear(32,1))
        for tower in (self.context_tower,self.output_tower):
            tower.layers[0].weight=mx.zeros_like(tower.layers[0].weight)
            tower.layers[0].bias=mx.zeros_like(tower.layers[0].bias)

    def __call__(self,inputs,mode):
        _mode(mode)
        if (inputs.dtype!=mx.float32 or inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)):
            raise ValueError('Context probe requires pinned FP32 axes')
        pre=self.input_tower(inputs)
        condition=self.context_tower(inputs[:,:,0,:64])[:,:,None,:]
        gamma,beta=mx.split(condition,2,axis=-1)
        hidden=(mx.maximum(pre+gamma,0)+beta if mode=='additive' else
            mx.maximum(pre,0)*(1+gamma)+beta)
        return self.output_tower(hidden).reshape((inputs.shape[0],2,1326))

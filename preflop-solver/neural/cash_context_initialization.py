"""One cold, training-only input-tower moment normalization experiment."""
import json
import numpy as np
from cash_context_conditioning import _head


def normalize_cold_context_input(payload,combined,weights):
    from validate_cash_value_parity import dense_forward
    _head(payload)
    if (combined.dtype!=np.float32 or combined.ndim!=4 or combined.shape[1:]!=(2,1326,256)
            or not 1<=len(combined)<=512 or weights.shape!=combined.shape[:-1]
            or weights.dtype!=np.float32 or not np.isfinite(combined).all()
            or not np.isfinite(weights).all() or (weights<0).any()
            or weights.sum(dtype=np.float64)<=0):
        raise ValueError('Expected authorized training features and finite positive reach mass')
    for name in ('context_tower','output_tower'):
        layer=payload[name][0]
        if any(value!=0 for key in ('weights','biases') for value in layer[key]):
            raise ValueError('Moment initialization requires a neutral cold readout and conditioner')
    pre=dense_forward(combined,payload['input_tower']).astype(np.float64)
    mass=weights.sum(dtype=np.float64); w=weights.astype(np.float64)/mass
    mean=np.sum(w[:,:,:,None]*pre,axis=(0,1,2))
    variance=np.sum(w[:,:,:,None]*(pre-mean)**2,axis=(0,1,2))
    std=np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=1e-8).any():
        raise ValueError('Cold input features lack sufficient variance for fixed normalization')
    result=json.loads(json.dumps(payload)); layer=result['input_tower'][0]
    matrix=np.asarray(layer['weights'],np.float64).reshape((32,256))/std[:,None]
    bias=(np.asarray(layer['biases'],np.float64)-mean)/std
    layer['weights']=matrix.astype(np.float32).ravel().tolist()
    layer['biases']=bias.astype(np.float32).tolist(); _head(result)
    normalized=dense_forward(combined,result['input_tower']).astype(np.float64)
    after_mean=np.sum(w[:,:,:,None]*normalized,axis=(0,1,2))
    after_std=np.sqrt(np.sum(w[:,:,:,None]*(normalized-after_mean)**2,axis=(0,1,2)))
    positive=np.sum(w[:,:,:,None]*(normalized>0),axis=(0,1,2))
    if ((np.abs(after_mean)>1e-4).any() or (np.abs(after_std-1.)>1e-4).any()
            or (positive<=0).any() or (positive>=1).any()):
        raise ArithmeticError('FP32 moment normalization did not meet its predeclared invariants')
    return result,dict(training_weight_mass=float(mass),preactivation_mean_before=mean.tolist(),
        preactivation_std_before=std.tolist(),preactivation_mean_after=after_mean.tolist(),
        preactivation_std_after=after_std.tolist(),positive_reach_fraction_after=positive.tolist(),
        target_values_used=False,development_rows_used=False,review_rows_used=False,
        updated_existing_tower='input_tower',other_towers_unchanged=True)


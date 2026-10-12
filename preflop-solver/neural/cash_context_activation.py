"""Non-serving FiLM activation repair; no change to frozen control modules.

Fixed negative slope0.01 removes the zero-readout/never-positive ReLU trap.
The three original towers and zero correction at initialization are unchanged.
Native inference requires a separately validated schema; never treat this as
the original ReLU head in a decoder or serving payload.
"""
import mlx.core as mx
import numpy as np
from cash_context_conditioning import ContextConditionedHead,_head

NEGATIVE_SLOPE=.01


class LeakyContextHead(ContextConditionedHead):
    def __call__(self,inputs,mode='film'):
        if (mode!='film' or inputs.dtype!=mx.float32 or inputs.ndim!=4
                or inputs.shape[1:]!=(2,1326,256)):
            raise ValueError('Leaky context probe requires original FP32 FiLM axes')
        pre=self.input_tower(inputs)
        condition=self.context_tower(inputs[:,:,0,:64])[:,:,None,:]
        gamma,beta=mx.split(condition,2,axis=-1)
        activated=mx.maximum(pre,0)+NEGATIVE_SLOPE*mx.minimum(pre,0)
        hidden=activated*(1+gamma)+beta
        return self.output_tower(hidden).reshape((inputs.shape[0],2,1326))


def leaky_context_dense_forward(inputs,payload):
    from validate_cash_value_parity import dense_forward
    _head(payload)
    if (inputs.dtype!=np.float32 or inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)
            or not 1<=len(inputs)<=512 or not np.isfinite(inputs).all()
            or not np.array_equal(inputs[:,:,:,:64],
                np.broadcast_to(inputs[:,:,:1,:64],inputs[:,:,:,:64].shape))):
        raise ValueError('Leaky context probe requires exact FP32 axes and public broadcast')
    pre=dense_forward(inputs,payload['input_tower'])
    condition=dense_forward(inputs[:,:,:1,:64],payload['context_tower'])
    gamma,beta=np.split(condition,2,axis=-1)
    activated=np.maximum(pre,np.float32(0))+np.float32(NEGATIVE_SLOPE)*np.minimum(pre,np.float32(0))
    hidden=activated*(np.float32(1)+gamma)+beta
    return dense_forward(hidden,payload['output_tower']).reshape((len(inputs),2,1326))


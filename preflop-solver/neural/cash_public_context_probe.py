"""Research-only public-context sketch contrast; no serving/native schema.

The full frozen256-vector still enters the same input tower. Only the
conditioner input varies. A sketch consumes one417-vector per public belief
and player, never a private-combo axis. A fixed seed and training-only gain are
caller-pinned; neither is an optimizer parameter or selected by label error.
"""
import mlx.core as mx
import numpy as np

from cash_context_conditioning import ContextConditionedHead, _head


def make_public_sketch(seed):
    if type(seed) is not int or not 0<=seed<2**64:
        raise ValueError('Public sketch requires a fixed nonnegative seed')
    signs=np.random.default_rng(seed).integers(0,2,(417,64),dtype=np.int8)
    return (signs.astype(np.float32)*2-1)*np.float32(.125)


def _context(context):
    if (not isinstance(context,np.ndarray) or context.dtype!=np.float32
            or context.ndim!=3 or context.shape[1:]!=(2,417)
            or not 1<=len(context)<=512 or not np.isfinite(context).all()):
        raise ValueError('Expected exact FP32 public context axes[N,2,417]')


def _projection(projection):
    if (not isinstance(projection,np.ndarray) or projection.dtype!=np.float32
            or projection.shape!=(417,64) or not np.isfinite(projection).all()
            or not np.all(np.abs(projection)==np.float32(.125))):
        raise ValueError('Expected fixed signed FP32 projection417-to64')


def sketch_public_context(context,projection,gain=1.):
    _context(context); _projection(projection)
    if type(gain) not in (int,float) or not np.isfinite(gain) or gain<=0:
        raise ValueError('Public sketch requires a finite positive fixed scalar')
    with np.errstate(over='ignore',invalid='ignore'):
        result=(context@projection)*np.float32(gain)
    if result.dtype!=np.float32 or not np.isfinite(result).all():
        raise ValueError('Public sketch overflow; no fallback')
    return result


def match_training_sketch_gain(context,encoded,projection):
    """One target-independent scalar, computed on authorized training rows."""
    _context(context)
    if (not isinstance(encoded,np.ndarray) or encoded.dtype!=np.float32
            or encoded.shape!=(len(context),2,64) or not np.isfinite(encoded).all()):
        raise ValueError('Expected matching FP32 encoded training contexts')
    sketched=sketch_public_context(context,projection)
    raw_rms=float(np.sqrt(np.mean(sketched.astype(np.float64)**2)))
    encoded_rms=float(np.sqrt(np.mean(encoded.astype(np.float64)**2)))
    if raw_rms<=0 or encoded_rms<=0:
        raise ValueError('Training RMS matching requires nonzero contexts')
    gain=float(np.float32(encoded_rms/raw_rms))
    sketch_public_context(context,projection,gain)
    return gain


def external_context_dense_forward(inputs,public_context,payload):
    """Independent CPU implementation, including exact broadcast-input guards."""
    from validate_cash_value_parity import dense_forward
    _head(payload)
    if (not isinstance(inputs,np.ndarray) or inputs.dtype!=np.float32
            or inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)
            or not 1<=len(inputs)<=512 or not np.isfinite(inputs).all()
            or not np.array_equal(inputs[:,:,:,:64],
                np.broadcast_to(inputs[:,:,:1,:64],inputs[:,:,:,:64].shape))
            or not isinstance(public_context,np.ndarray) or public_context.dtype!=np.float32
            or public_context.shape!=(len(inputs),2,64) or not np.isfinite(public_context).all()):
        raise ValueError('Expected frozen inputs and one independent public context per seat')
    pre=dense_forward(inputs,payload['input_tower'])
    condition=dense_forward(public_context,payload['context_tower'])[:,:,None,:]
    gamma,beta=np.split(condition,2,axis=-1)
    hidden=np.maximum(pre,np.float32(0))*(np.float32(1)+gamma)+beta
    return dense_forward(hidden,payload['output_tower']).reshape((len(inputs),2,1326))


class ExternalContextHead(ContextConditionedHead):
    def __call__(self,inputs,public_context=None):
        if public_context is None:
            # The encoded control uses the original graph, not an equivalent
            # reimplementation; its pinned training trace must reproduce.
            return super().__call__(inputs,'film')
        if (inputs.dtype!=mx.float32 or inputs.ndim!=4 or inputs.shape[1:]!=(2,1326,256)
                or public_context.dtype!=mx.float32
                or public_context.shape!=(inputs.shape[0],2,64)):
            raise ValueError('Expected FP32 frozen inputs and public context axes')
        pre=self.input_tower(inputs)
        condition=self.context_tower(public_context)[:,:,None,:]
        gamma,beta=mx.split(condition,2,axis=-1)
        hidden=mx.maximum(pre,0)*(1+gamma)+beta
        return self.output_tower(hidden).reshape((inputs.shape[0],2,1326))

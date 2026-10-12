"""Stable, research-only linear-readout delta fitting.

The caller supplies authorized training rows only. The minimum-norm correction
leaves discarded/null directions unchanged relative to its frozen readout.
This does not validate poker payoffs, choose evaluation rows or activate models.
"""
import math
import numpy as np


def fit_linear_delta(design,residual,weights,*,rcond=1e-6):
    if (not isinstance(design,np.ndarray) or not isinstance(residual,np.ndarray)
            or not isinstance(weights,np.ndarray) or design.ndim!=2
            or not 1<=design.shape[1]<=33 or not 1<=design.shape[0]<=512*2652
            or residual.shape!=(len(design),) or weights.shape!=(len(design),)
            or any(value.dtype.kind!='f' for value in (design,residual,weights))
            or not all(np.isfinite(value).all() for value in (design,residual,weights))
            or (weights<0).any() or weights.sum(dtype=np.float64)<=0
            or type(rcond) not in (int,float) or not math.isfinite(rcond) or not 0<rcond<1):
        raise ValueError('Expected finite floating-point design, residual and positive training mass')
    x=design.astype(np.float64,copy=False); y=residual.astype(np.float64,copy=False)
    w=weights.astype(np.float64,copy=False); mass=w.sum(dtype=np.float64)
    if not math.isfinite(float(mass)):
        raise ValueError('Training weight mass is not finite')
    scale=np.sqrt(w/mass)
    delta,_,rank,singular=np.linalg.lstsq(x*scale[:,None],y*scale,rcond=rcond)
    error=y-x@delta
    before=float(np.sum(w*y**2,dtype=np.float64)/mass)
    after=float(np.sum(w*error**2,dtype=np.float64)/mass)
    if (not np.isfinite(delta).all() or not math.isfinite(after)
            or after>before+1e-10*max(1.,before)):
        raise ArithmeticError('Stable linear refit did not reduce its training objective')
    rank=int(rank)
    return delta,dict(training_mse_before=before,training_mse_after=after,
        training_weight_mass=float(mass),effective_rank=rank,columns=design.shape[1],
        singular_values=singular.tolist(),rcond=float(rcond),
        effective_condition_number=(None if rank==0 else float(singular[0]/singular[rank-1])),
        correction_l2_norm=float(np.linalg.norm(delta)))


"""Bounded, research-only last-layer displacement diagnostic.

Callers supply frozen features in payoff units, target-minus-parent residuals,
and the original nonnegative loss weights. Each batch contributes sufficient
statistics, not an independently normalized loss. No labels, model loading,
checkpoint selection, exports or policy qualification occur in this module.
"""
import numpy as np


def frozen_cash_interaction_features(combined):
    """Diagnostic crosses of already frozen, same-basis hand/range features.

    Input order is context, own pool, opponent pool, queried hand (64 each).
    Keep the original256 columns; append hand*opponent then hand*own. No
    learned features, reach distributions, labels or native layout change.
    """
    values=np.asarray(combined)
    if (values.ndim!=4 or not 1<=len(values)<=8 or values.shape[1:]!=(2,1326,256)
            or values.dtype.kind!='f' or not np.isfinite(values).all()):
        raise ValueError('cash interaction probe requires at most8 paired frozen256-feature contexts')
    with np.errstate(over='ignore',invalid='ignore'):
        hand=values[:,:,:,192:256]
        result=np.concatenate((values,hand*values[:,:,:,128:192],hand*values[:,:,:,64:128]),axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('cash interaction probe features overflowed')
    return result


class FrozenLinearProbe:
    def __init__(self, dimensions: int):
        if type(dimensions) is not int or not 1<=dimensions<=512:
            raise ValueError('frozen linear probe dimensions must be1..512')
        self.dimensions=dimensions
        self.normal=np.zeros((dimensions,dimensions),np.float64)
        self.rhs=np.zeros(dimensions,np.float64)
        self.target_squared=0.
        self.mass=0.
        self.rows=0

    def add(self, features, residual_targets, weights):
        inputs=[np.asarray(value) for value in (features,residual_targets,weights)]
        if any(value.dtype.kind not in 'iuf' for value in inputs):
            raise ValueError('frozen linear probe requires real numeric arrays')
        x,y,w=[value.astype(np.float64,copy=False) for value in inputs]
        if (x.ndim!=2 or not 1<=len(x)<=100000 or x.shape[1]!=self.dimensions
                or y.shape!=(len(x),) or w.shape!=(len(x),)
                or not all(np.isfinite(value).all() for value in (x,y,w)) or (w<0).any()):
            raise ValueError('frozen linear probe has invalid shapes, values or weights')
        mass=float(w.sum())
        if not np.isfinite(mass) or mass<=0:
            raise ValueError('frozen linear probe requires positive finite batch mass')
        # Exclude zero-weight queries before squaring potentially large values.
        active=w>0; x,y,w=x[active],y[active],w[active]
        weighted=x*np.sqrt(w[:,None]); target=y*np.sqrt(w)
        with np.errstate(over='ignore',invalid='ignore'):
            normal=self.normal+weighted.T@weighted
            rhs=self.rhs+weighted.T@target
            squared=self.target_squared+float(target@target)
        if not (np.isfinite(normal).all() and np.isfinite(rhs).all()
                and np.isfinite(squared) and np.isfinite(self.mass+mass)):
            raise ValueError('frozen linear probe sufficient statistics overflowed')
        self.normal,self.rhs,self.target_squared=normal,rhs,squared
        self.mass+=mass; self.rows+=len(inputs[0])

    def solve(self, ridge: float):
        if (type(ridge) not in (int,float) or not np.isfinite(ridge) or not 0<=ridge<=1e6
                or self.mass<=0):
            raise ValueError('frozen linear probe requires data and finite ridge in0..1e6')
        gram=self.normal/self.mass; rhs=self.rhs/self.mass
        scale=np.sqrt(np.maximum(np.diag(gram),0.)); active=scale>0
        coefficients=np.zeros(self.dimensions,np.float64)
        rank=0; standardized=np.zeros(int(active.sum()),np.float64)
        if active.any():
            normalized=gram[np.ix_(active,active)]/np.outer(scale[active],scale[active])
            normalized=(normalized+normalized.T)/2
            system=normalized+ridge*np.eye(len(normalized))
            standardized,_,rank,_=np.linalg.lstsq(system,rhs[active]/scale[active],rcond=1e-10)
            coefficients[active]=standardized/scale[active]
        initial=self.target_squared/self.mass
        mse=float(initial-2*coefficients@rhs+coefficients@gram@coefficients)
        tolerance=1e-9*max(1.,initial)
        objective=mse+ridge*float(standardized@standardized)
        if (not np.isfinite(coefficients).all() or not np.isfinite(objective)
                or mse < -tolerance or objective>initial+tolerance):
            raise ValueError('frozen linear probe solve failed numerical/objective checks')
        return dict(coefficients=coefficients,data_mse=max(0.,mse),initial_mse=initial,
            regularized_objective=max(0.,objective),ridge=ridge,effective_rank=int(rank),
            relative_singular_cutoff=1e-10,rows=self.rows,normalization_mass=self.mass,
            status='non_serving_research',active=False)

"""Frozen profile-ranking diagnostics; no exploitability or equilibrium claim."""
from __future__ import annotations

import math
import numpy as np


def checked(q, target, weights, depth, delta, tie):
    q,target,weights=map(lambda value:np.asarray(value,dtype=np.float64),(q,target,weights))
    if (q.ndim!=2 or q.shape[0]<2 or target.shape!=q.shape or weights.shape!=q.shape[1:]
            or not all(np.isfinite(value).all() for value in (q,target,weights))
            or (weights<0).any() or weights.sum()<=0
            or not all(math.isfinite(value) and value>0 for value in (depth,delta,tie))):
        raise ValueError("invalid profile-ranking inputs")
    return q,target,weights/weights.sum()


def contrast_allocation(q,target,weights,depth=20.,delta=.05,tie=.05):
    """Decompose the exact existing Huber derivative; sums reconstruct it.

    Magnitude shares are before the affine/network Jacobians. They cannot be
    interpreted as parameter-gradient norms or independent policy gains.
    """
    q,target,weights=checked(q,target,weights,depth,delta,tie)
    pieces={name:np.zeros_like(q) for name in ("correct","inverted","nearTie")}
    mass={name:0. for name in pieces}
    pairs=len(q)*(len(q)-1)//2
    for a in range(len(q)):
        for b in range(a+1,len(q)):
            predicted=q[a]-q[b]; truth=target[a]-target[b]
            derivative=weights*np.clip((predicted-truth)/depth,-delta,delta)/pairs/depth
            tied=abs(truth)<tie
            correct=(predicted*truth>0)&~tied
            masks=dict(correct=correct,inverted=~correct&~tied,nearTie=tied)
            for name,mask in masks.items():
                selected=derivative*mask
                pieces[name][a]+=selected; pieces[name][b]-=selected
                mass[name]+=float(abs(selected).sum())
    total=sum(mass.values())
    best=np.argmax(q,axis=0)
    report=dict(nativeRankingLossBb=float(weights@(target.max(axis=0)-target[best,np.arange(len(weights))])),
        pairDerivativeMagnitude=mass,
        pairDerivativeShares={name:value/total if total>0 else 0. for name,value in mass.items()},
        nearTieThresholdBb=tie,
        interpretation="Unique-pair derivative magnitudes before Jacobians, not parameter-gradient norms or exploitability.")
    return pieces,report


def margin_loss_and_q_gradient(q,target,weights,depth=20.,delta=.05,tie=.05,margin=.25):
    """Supervised best-action margin, not a poker best-response gradient.

    Native gaps below tie provide no ordering signal. Larger gaps cap the
    required predicted margin at 0.25bb; ordinary value calibration must stay
    active to retain EV/indifference information. Every potential competitor
    has fixed 1/(A-1) normalization, including omitted near ties.
    """
    q,target,weights=checked(q,target,weights,depth,delta,tie)
    if not math.isfinite(margin) or margin<tie: raise ValueError("invalid ranking margin")
    best=np.argmax(target,axis=0); combos=np.arange(len(weights))
    best_truth=target[best,combos]; best_q=q[best,combos]
    gradient=np.zeros_like(q); loss=0.
    for action in range(len(q)):
        gap=best_truth-target[action]
        active=gap>=tie
        error=np.maximum(0.,(np.minimum(gap,margin)-(best_q-q[action]))/depth)*active
        quadratic=np.minimum(error,delta)
        loss+=float(weights@(.5*quadratic**2+delta*(error-quadratic)))/(len(q)-1)
        derivative=weights*np.clip(error,0.,delta)/depth/(len(q)-1)
        gradient[action]+=derivative
        gradient[best,combos]-=derivative
    return loss,gradient


def ranking_pilot_supported(rows):
    if len(rows)!=4 or {(r["seed"],r["step"]) for r in rows}!={(s,t) for s in (10601,10602) for t in (0,600)}:
        raise ValueError("both frozen seeds/start/endpoints required")
    endpoints=[r for r in rows if r["step"]==600]
    for row in rows:
        metrics=[row["meanNativeRankingLossBb"],row["meanCorrectDerivativeShare"]]
        if not np.isfinite(metrics).all() or metrics[0]<0 or not 0<=metrics[1]<=1:
            raise ValueError("nonfinite or invalid allocation metrics")
    return all(r["meanNativeRankingLossBb"]>=.03 and r["meanCorrectDerivativeShare"]>=.75 for r in endpoints)

"""Research-only compression bypass; no accepted network or serving schema.

Both arms have the same 380-input, 32-hidden shared correction head. The raw
arm exposes the existing v2 query features; the control appends exact zeros.
The caller keeps the parent frozen and applies its independently tested public
gate, own-net-bb baseline, board mask and clipping. This module does not infer
strategy, alter ranges, generate card features or qualify a cash model.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def raw_query_probe_features(combined,queries,mode):
    """Append the pinned 124 features, or zeros, without changing combo order."""
    combined=np.asarray(combined); queries=np.asarray(queries)
    if (type(mode) is not str or mode not in ('raw-query','zero-query') or
            combined.dtype!=np.float32 or queries.dtype!=np.float32 or
            combined.ndim!=4 or queries.ndim!=4 or
            combined.shape[1:]!=(2,1326,256) or queries.shape[1:]!=(2,1326,124) or
            combined.shape[0]!=queries.shape[0] or not 1<=combined.shape[0]<=512):
        raise ValueError('Expected matched bounded FP32 v7 embeddings/v2 query rows and a fixed probe mode')
    if not np.isfinite(combined).all() or not np.isfinite(queries).all():
        raise ValueError('Raw-query probes reject nonfinite parent/query features in both arms')
    extra=queries if mode=='raw-query' else np.zeros_like(queries)
    return np.concatenate((combined,extra),axis=-1)


class RawQueryProbeHead(nn.Module):
    """Zero-output matched-capacity readout; never a standalone value model."""
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(380,32),nn.ReLU(),nn.Linear(32,1))
        self.head.layers[-1].weight=mx.zeros_like(self.head.layers[-1].weight)
        self.head.layers[-1].bias=mx.zeros_like(self.head.layers[-1].bias)

    def __call__(self,features):
        return mx.squeeze(self.head(features),axis=-1)

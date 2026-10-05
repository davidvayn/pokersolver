"""Resource and forward-parity screen before a serving-aligned contrast fit."""
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

import native_value_dataset as native
import run_action_contrast_students as base
import train_public_value_network as training
from action_contrast_loss import BundleObjective
from retained_initialization import import_retained_weights, native_import_prediction
from run_native_value_preflight import atomic_json
from serving_value_projection import BoundedValueProjection


def preflight(args):
    dataset, contexts, queries, split, bundles, _ = base.prepare(args)
    settings = json.loads((args.output/"fit-settings.json").read_text())
    model = training.SharedComboValueNetwork(True,getattr(args,"architecture","wide"),
        "payoff-exposure",training.FEATURE_SCHEMA_EXACT_RUNOUT)
    import_retained_weights(model,json.loads(Path(args.initial_models[10601]["path"]).read_text()),10601,
        getattr(args,"initialization_transform",None))
    bundle = max(bundles,key=lambda b:len(b.dataset.targets))
    d = bundle.dataset
    raw = np.asarray(model.raw_values(*BundleObjective.inputs(bundle,0,8))) * d.target_scales[:8,None,None]
    weights = np.asarray([r*native.compatible_masses(r) for r in d.ranges[:8]])
    legal = np.asarray([native.legal_combos(b) for b in d.boards[:8]])
    projection = BoundedValueProjection(raw,weights,legal)
    expected = native_import_prediction(model,bundle.contexts[:8],bundle.queries[:8],
        d.target_scales[:8],d.boards[:8],d.ranges[:8])
    error = float(abs(projection.values-expected).max())
    if not np.isfinite(error) or error > .0001:
        raise ValueError("serving-aligned contrast forward differs from the serving predictor")
    objective = BundleObjective(bundles,settings["contrastWeight"],4,4,serving_aligned=True)
    start = time.monotonic()
    _, _, gradient_report = objective.gradients(model,bundle,base.value_loss)
    seconds = time.monotonic()-start
    projected = 427. + 1.5 * seconds * 300
    if not np.isfinite(projected) or projected > 7200:
        raise ValueError("serving-aligned fixed pair exceeds the two-hour cost cap")
    result = dict(schema="serving-aligned-contrast-preflight-v1",status="complete",releaseAccepted=False,
        maximumForwardDifferenceBb=error, parityStates=8, privateQueriesPerState=2652,
        largestBundleStates=len(d.targets), wholeFamilyGradientSeconds=seconds,
        projectedPairSeconds=projected, gradientReport=gradient_report,
        servingAlignedContrast=True,
        nativeCounterfactualFraction=getattr(args, "native_counterfactual_fraction", .1),
        ordinaryValueCalibrationUnchanged=getattr(args, "native_counterfactual_fraction", .1) == .1,
        fixedFinalStep=600,cadence=4,chunkSize=4)
    atomic_json(args.output/"serving-preflight.json",result)
    print(json.dumps(dict(event="serving-contrast-preflight",maximumForwardDifferenceBb=error,
        projectedPairSeconds=projected)),flush=True)

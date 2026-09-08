"""Exact conditional sampling variance from already captured frozen endpoints.

This screens sampling schemes, not policy strength. Native residuals and the
exact checkdown baseline both come from Rust; no poker value model is recreated.
"""
import numpy as np

from run_frozen_preflop_response import tree_values, validate_capture


def fit_proposal(energies):
    values = np.asarray(energies, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid fitting gradient energies")
    signal = np.sqrt(values.mean(axis=0))
    uniform = np.full(len(signal), 1/len(signal))
    return uniform if signal.sum() == 0 else .5*uniform+.5*signal/signal.sum()


def importance_variance(residuals, probabilities):
    values = np.asarray(residuals, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    if (values.ndim < 2 or q.shape != (len(values),) or not np.isfinite(values).all()
            or not np.isfinite(q).all() or (q <= 0).any() or abs(q.sum()-1) > 1e-12):
        raise ValueError("invalid full-support importance proposal")
    second = (values**2/q.reshape((-1,)+(1,)*(values.ndim-1))).sum(axis=0)
    variance = second-values.sum(axis=0)**2
    if (variance < -1e-10*(1+second)).any():
        raise ValueError("negative importance variance beyond roundoff")
    return np.maximum(variance, 0)


def conditional_variances(residuals, groups, batch_size):
    values = np.asarray(residuals, dtype=float)
    if (values.ndim < 2 or len(values) != len(groups) or not np.isfinite(values).all()
            or not 1 <= batch_size <= len(values) or len(values) < 2):
        raise ValueError("invalid endpoint sampling population")
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(np.unique(groups)) != batch_size:
        raise ValueError("batch must match the number of strata")
    count = len(values)
    one = count*np.square(values).sum(axis=0)-np.square(values.sum(axis=0))
    batch = one*(count-batch_size)/(batch_size*(count-1))
    stratified = np.zeros_like(one)
    for group in np.unique(groups):
        subset = values[groups == group]
        stratified += len(subset)*np.square(subset).sum(axis=0)-np.square(subset.sum(axis=0))
    tolerance = 1e-10*(1+count*np.square(values).sum(axis=0))
    result = dict(uniformOne=one, uniformBatch=batch, rootStratified=stratified)
    if any((value < -tolerance).any() for value in result.values()):
        raise ValueError("negative sampling variance beyond roundoff")
    return {key: np.maximum(value, 0) for key, value in result.items()}


def decision_residuals(capture, baseline, decision_history=None):
    rows, endpoints = validate_capture(capture, True)
    if (baseline["schema"] != "frozen-preflop-checkdown-baseline-v1"
            or baseline["nativeQueries"] != 0 or baseline["releaseAccepted"] is not False
            or any(capture[key] != baseline[key] for key in
                   ("preflopSha256", "kernelSha256", "rootHistory", "classes", "multiplicities"))
            or capture["continuationFunction"]["flopControl"] !=
                "exact_checkdown_mean_plus_native_minus_sampled_checkdown_scale_1"):
        raise ValueError("baseline does not match frozen capture")
    means = {tuple(row["history"]): np.asarray(row["cfvBb"], dtype=float)
             for row in baseline["endpoints"]}
    live = [tuple(h) for h in baseline["liveHistories"]]
    if (len(means) != len(baseline["endpoints"]) or means.keys() != endpoints.keys()
            or len(live) != 49 or len(set(live)) != 49
            or set(live) != {tuple(c["history"]) for c in capture["captures"]}
            or any(v.shape != (2, 169) or not np.isfinite(v).all() for v in means.values())):
        raise ValueError("incomplete or invalid baseline")
    for h in means.keys()-set(live):
        if not np.allclose(means[h], endpoints[h], atol=1e-12, rtol=0):
            raise ValueError("exact terminal changed between baseline and capture")
    sampling_root = tuple(capture["rootHistory"])
    root = sampling_root if decision_history is None else tuple(decision_history)
    if root not in rows:
        raise ValueError("decision is not in the captured public tree")
    row = rows[root]
    actor = row["actor"]
    mix = np.asarray(row["probabilities"])
    def advantages(values):
        q = np.array([tree_values(rows, values, tuple(h), actor) for h in row["children"]])
        return q-(q*mix).sum(axis=0)
    residuals = []
    for h in live:
        if h[:len(root)] != root:
            residuals.append(np.zeros_like(mix))
            continue
        isolated = {key: np.zeros_like(v) for key, v in means.items()}
        isolated[h] = endpoints[h]-means[h]
        residuals.append(advantages(isolated))
    residuals = np.asarray(residuals)
    if not np.allclose(residuals.sum(axis=0), advantages(endpoints)-advantages(means), atol=1e-10, rtol=0):
        raise ValueError("endpoint contributions do not conserve decision advantages")
    groups = [h[len(sampling_root)] for h in live]
    costs = {tuple(c["history"]): c["seconds"] for c in capture["captures"]}
    if any(not np.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("invalid observed endpoint cost")
    return residuals, groups, np.array([costs[h] for h in live])


def gradient_energies(capture, baseline):
    """All 100 public states, both actors; actual class-weighted CFR deltas."""
    weights = np.asarray(capture["multiplicities"])/1326
    energies = np.zeros((2, 49))
    total_squared = np.zeros(2)
    for row in capture["rows"]:
        residuals, _, costs = decision_residuals(capture, baseline, row["history"])
        gradients = residuals*weights
        actor = row["actor"]
        energies[actor] += np.square(gradients).sum(axis=(1, 2))
        total_squared[actor] += np.square(gradients.sum(axis=0)).sum()
    return energies, total_squared, costs


def analyze(capture, baseline, decision_history=None):
    residuals, groups, costs = decision_residuals(capture, baseline, decision_history)
    labels = np.unique(groups)
    variances = conditional_variances(residuals, groups, len(labels))
    weights = np.array(capture["multiplicities"])/1326
    group_array = np.array(groups)
    times = dict(uniformOne=float(costs.mean()), uniformBatch=float(len(labels)*costs.mean()),
                 rootStratified=float(sum(costs[group_array == group].mean() for group in labels)))
    result = {}
    for scheme, variance in variances.items():
        mean = float((variance*weights).sum(axis=-1).mean())
        result[scheme] = dict(meanActionVarianceBb2=mean,
            expectedNativeCaptureSeconds=times[scheme], varianceTimesNativeSeconds=mean*times[scheme])
    return dict(boardIndex=capture["boardIndex"],
        decisionHistory=capture["rootHistory"] if decision_history is None else list(decision_history),
        schemes=result,
        strataSizes={group: groups.count(group) for group in labels},
        interpretation="Exact endpoint-selection variance conditional on this frozen policy and observed flop/turn. Excludes chance variance, changing ranges, and convergence. Cost reuses observed native endpoint times; not a new live timing benchmark.")

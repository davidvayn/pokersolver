"""Attribute a frozen preflop attack's gain to actionable reference-policy errors.

The performance-difference sum uses RESPONDER own reaches and reference
continuation advantages. Opponent reaches are already contained in the CFVs.
This is a decomposition of the measured response, not a new strength metric.
"""
import numpy as np


def contributions(rows, endpoints, root, seat, choices, weights):
    reference = {}
    def profile(history):
        if history in reference:
            return reference[history]
        if history in endpoints:
            value = endpoints[history][seat]
        else:
            row = rows[history]
            children = np.asarray([profile(tuple(h)) for h in row["children"]])
            value = ((children*np.asarray(row["probabilities"])).sum(axis=0)
                     if row["actor"] == seat else children.sum(axis=0))
        reference[history] = value
        return value
    profile(root)
    result = {}
    def visit(history, own_reach):
        if history in endpoints:
            return
        row = rows[history]
        if row["actor"] == seat:
            chosen = choices[history]
            children = np.asarray([reference[tuple(h)] for h in row["children"]])
            advantage = children[chosen, np.arange(len(weights))]-reference[history]
            by_class = np.asarray(weights)*own_reach*advantage
            result[history] = dict(gainContributionBb=float(by_class.sum()),
                                   classContributionsBb=by_class.tolist())
        for a, h in enumerate(row["children"]):
            reach = own_reach*(choices[history] == a) if row["actor"] == seat else own_reach
            visit(tuple(h), reach)
    visit(root, np.ones(len(weights)))
    return result

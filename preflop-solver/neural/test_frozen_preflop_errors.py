import unittest
import numpy as np
from frozen_preflop_errors import contributions
from run_frozen_preflop_response import tree_values


class FrozenPreflopErrorTests(unittest.TestCase):
    def test_sequential_advantages_use_response_reach_and_telescope(self):
        rows = {(): dict(actor=0,children=[["a"],["b"]],probabilities=[[.8],[.2]]),
                ("b",): dict(actor=0,children=[["b","win"],["b","lose"]],probabilities=[[.5],[.5]])}
        endpoints = {("a",): np.array([[.2],[0.0]]),
                     ("b","win"): np.array([[1.0],[0.0]]),
                     ("b","lose"): np.array([[0.0],[0.0]])}
        choices = {():np.array([1]),("b",):np.array([0])}
        parts = contributions(rows,endpoints,(),0,choices,np.ones(1))
        self.assertAlmostEqual(parts[()]["gainContributionBb"],.24)
        self.assertAlmostEqual(parts[("b",)]["gainContributionBb"],.5)
        actual = tree_values(rows,endpoints,(),0,choices)-tree_values(rows,endpoints,(),0)
        self.assertAlmostEqual(sum(v["gainContributionBb"] for v in parts.values()),actual[0])


if __name__ == "__main__":
    unittest.main()

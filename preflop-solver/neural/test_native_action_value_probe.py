import copy
import unittest

from run_native_action_value_probe import benchmark_cases


class BenchmarkCaseSelectionTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "schema": "postflop-benchmark-protocol-v1", "sha256": "a" * 64,
            "spots": [{"id": "limped-paired", "board": [2, 12, 15]}],
            "models": [{"seed": 100101}, {"seed": 100102}],
        }
        self.completed = {
            "schema": "postflop-benchmark-results-v1", "status": "complete",
            "protocolSha256": "a" * 64,
            "rows": [{"spot": "limped-paired", "seed": seed,
                      "board": [2, 12, 15], "all49Turns": True,
                      "flopAccountingAuditPassed": True} for seed in (100101, 100102)],
        }

    def test_requires_both_complete_seed_cases(self):
        cases = benchmark_cases(self.protocol, self.completed, ["limped-paired"])
        self.assertEqual([row[2]["seed"] for row in cases], [100101, 100102])
        for change in ("missing", "duplicate", "incomplete", "board", "protocol"):
            completed = copy.deepcopy(self.completed)
            if change == "missing":
                completed["rows"].pop()
            elif change == "duplicate":
                completed["rows"].append(copy.deepcopy(completed["rows"][0]))
            elif change == "incomplete":
                completed["rows"][1]["all49Turns"] = False
            elif change == "board":
                completed["rows"][0]["board"][0] = 3
            else:
                completed["protocolSha256"] = "b" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                benchmark_cases(self.protocol, completed, ["limped-paired"])

    def test_rejects_unknown_or_repeated_spot(self):
        for spots in ([], ["unknown"], ["limped-paired", "limped-paired"]):
            with self.subTest(spots=spots), self.assertRaises(ValueError):
                benchmark_cases(self.protocol, self.completed, spots)


if __name__ == "__main__":
    unittest.main()

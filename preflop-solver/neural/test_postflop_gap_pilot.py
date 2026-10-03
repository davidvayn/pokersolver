import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from run_postflop_gap_pilot import (
    PilotMemoryGuard, effective_memory_pressure, estimate_work_seconds, select_cases,
    measured_native_projection, same_public_state,
)


class MatchedGapPilotTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {"schema": "postflop-benchmark-protocol-v1", "sha256": "a" * 64,
                         "spots": [{"id": "limped-paired", "board": [2, 12, 15]}],
                         "models": [{"seed": 100101}, {"seed": 100102}]}
        self.baseline = {"schema": "postflop-benchmark-results-v1", "status": "complete",
                         "protocolSha256": "a" * 64,
                         "rows": [{"spot": "limped-paired", "seed": seed,
                                   "board": [2, 12, 15], "all49Turns": True,
                                   "flopAccountingAuditPassed": True}
                                  for seed in (100101, 100102)]}

    def test_selects_both_seed_controls_and_rejects_missing_evidence(self):
        cases = select_cases(self.protocol, self.baseline, ["limped-paired"])
        self.assertEqual([case[1]["seed"] for case in cases], [100101, 100102])
        for invalid in ("missing", "duplicate", "missing_turn", "board", "wrong_hash"):
            value = copy.deepcopy(self.baseline)
            if invalid == "missing":
                value["rows"].pop()
            elif invalid == "duplicate":
                value["rows"].append(copy.deepcopy(value["rows"][0]))
            elif invalid == "missing_turn":
                value["rows"][0]["all49Turns"] = False
            elif invalid == "board":
                value["rows"][1]["board"][0] = 3
            else:
                value["protocolSha256"] = "b" * 64
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_cases(self.protocol, value, ["limped-paired"])

    def test_projection_counts_every_packet_and_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = select_cases(self.protocol, self.baseline, ["limped-paired"])
            for _, model, _ in cases:
                for turn in range(49):
                    path = root / "jobs" / "limped-paired" / str(model["seed"]) / f"turn-{turn}"
                    path.mkdir(parents=True)
                    (path / "completed.json").write_text(json.dumps({
                        "worker": {"workerElapsedSeconds": 10}}))
            projection = estimate_work_seconds(root, cases, workers=4)
            self.assertEqual(projection["observedPackets"], 98)
            self.assertEqual(projection["estimatedSeconds"], 4090)
            (root / "jobs/limped-paired/100101/turn-0/completed.json").unlink()
            with self.assertRaises(ValueError):
                estimate_work_seconds(root, cases, workers=4)

    def test_macos_warning_requires_low_free_memory_but_critical_always_stops(self):
        def reader(level, free):
            def command(args, **_):
                value = str(level) if args[0] == "sysctl" else (
                    f"System-wide memory free percentage: {free}%")
                return type("Result", (), {"stdout": value})()
            return effective_memory_pressure(system="Darwin", command=command)

        self.assertEqual(reader(2, 43), 1)
        self.assertEqual(reader(2, 24), 2)
        self.assertEqual(reader(4, 57), 4)
        with self.assertRaises(ValueError):
            reader(2, -1)

    def test_guard_stops_sustained_low_memory_or_critical_pressure(self):
        stop = threading.Event()
        guard = PilotMemoryGuard(stop, Path("/unused"), reader=lambda: 2)
        for _ in range(59):
            guard.sample()
        self.assertFalse(stop.is_set())
        guard.sample()
        self.assertTrue(stop.is_set())
        critical_stop = threading.Event()
        critical = PilotMemoryGuard(critical_stop, Path("/unused"), reader=lambda: 4)
        critical.sample()
        self.assertTrue(critical_stop.is_set())

    def test_native_preflight_cost_is_included_before_32_update_comparison(self):
        estimate = {"estimatedPacketSeconds": 8000, "estimatedSeconds": 11600}
        pilot = {"schema": "postflop-gap-matched-leaf-pilot-v1",
                 "phase": "preflight", "status": "complete", "iterations": 8,
                 "cases": [{"arm": "native", "solveSeconds": 400,
                            "peakMemoryBytes": 1600}]}
        projected = measured_native_projection(estimate, pilot, 4)
        self.assertEqual(projected["estimatedSeconds"], 21200)
        pilot["status"] = "failed"
        with self.assertRaises(ValueError):
            measured_native_projection(estimate, pilot, 4)

    def test_public_root_match_rejects_support_or_game_drift(self):
        root = {"board": [1, 2, 3], "ranges": [[.4, .6], [.7, .3]]}
        same = copy.deepcopy(root)
        same["ranges"][0][0] += 1e-15
        same["ranges"][0][1] -= 1e-15
        self.assertTrue(same_public_state(root, same))
        same["ranges"][0] = [0, 1]
        self.assertFalse(same_public_state(root, same))
        same = copy.deepcopy(root)
        same["board"][0] = 4
        self.assertFalse(same_public_state(root, same))


if __name__ == "__main__":
    unittest.main()

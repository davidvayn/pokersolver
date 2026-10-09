"""Serving fixture/worker checks; no poker training or model inference required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark_practice_latency import ROOT, VERSION, command, fixtures


class PracticeLatencyTests(unittest.TestCase):
    def test_optional_cold_pair_preserves_cross_batch_indices_and_exact_inputs(self):
        base = fixtures(VERSION, cross_batch=True)
        serial = fixtures(VERSION, cross_batch=True, cold_pair="serial")
        parallel = fixtures(VERSION, cross_batch=True, cold_pair="parallel")
        self.assertEqual(serial, parallel)
        self.assertEqual(serial[:9], base)
        self.assertEqual(len(serial), 11)
        names = [name for name, _ in serial]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-2:], ["cold-pair-a", "cold-pair-b"])
        previous_boards = {tuple(query["board"]) for _, query in base}
        for name, query in serial[-2:]:
            self.assertEqual(name, query["requestId"])
            self.assertNotIn(tuple(query["board"]), previous_boards)
            self.assertEqual(len(set(query["board"] + query["privateCards"])), 5)

    def test_parallel_pair_does_not_require_cross_batch_mode(self):
        base = fixtures(VERSION)
        paired = fixtures(VERSION, cold_pair="parallel")
        self.assertEqual(paired[:7], base)
        self.assertEqual(len(paired), 9)

    def test_higher_worker_counts_leave_model_and_budget_unchanged(self):
        manifests = json.loads((ROOT / "data/practice/full-hand-manifests.json").read_text())
        manifest = next(m for m in manifests if m["version"] == VERSION)
        eight = command(Path("/test/resolver"), manifest, 8)
        sixteen = command(Path("/test/resolver"), manifest, 16)
        self.assertEqual(len(eight), len(sixteen))
        changed = [i for i, (a, b) in enumerate(zip(eight, sixteen)) if a != b]
        self.assertEqual(len(changed), 2)
        for index in changed:
            self.assertIn(eight[index - 1], ["--flop-resolver-threads", "--turn-resolver-threads"])
            self.assertEqual(sixteen[index], "16")

    def test_invalid_worker_budgets_fail_before_creating_receipts(self):
        with tempfile.TemporaryDirectory(prefix="practice-latency-test-") as task_dir:
            for count in (0, 17):
                output = Path(task_dir) / f"invalid-{count}"
                result = subprocess.run(
                    [sys.executable, str(ROOT / "preflop-solver/neural/benchmark_practice_latency.py"),
                     "--threads", str(count), "--output", str(output)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("threads must be between 1 and 16", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

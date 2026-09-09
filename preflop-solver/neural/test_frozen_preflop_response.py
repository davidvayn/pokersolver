import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import run_frozen_preflop_response as runner
from run_native_value_preflight import atomic_json, sha256
from run_frozen_preflop_response import tree_values, summarize


class FrozenPreflopResponseTests(unittest.TestCase):
    def test_extended_time_does_not_silently_increase_concurrency_or_memory(self):
        argv=["runner","--maximum-worker-seconds","7200","--output","unused"]
        for name in ("binary","preflop","model","kernel"):
            argv += ["--"+name,"unused","--"+name+"-sha256","unused"]
        with patch("sys.argv",argv), patch.object(runner,"guarded") as worker:
            with self.assertRaisesRegex(ValueError,"single-worker low-memory"):
                runner.main()
            worker.assert_not_called()

    def test_endpoint_recovery_pins_completed_files_not_temporary_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"endpoints"
            identity=dict(policy="fixed",binary="fixed")
            pins=runner.prepare_endpoint_cache(root,identity,False)
            self.assertEqual(len(pins),1)
            board=root/"board0";board.mkdir()
            completed=board/("a"*64+".json");completed.write_text("fixture checkpoint")
            (board/("b"*64+".tmp")).write_text("interrupted write")
            pins=runner.prepare_endpoint_cache(root,identity,True)
            self.assertEqual(pins[str(completed)],sha256(completed))
            self.assertEqual(len(pins),2)
            with self.assertRaisesRegex(ValueError,"identity"):
                runner.prepare_endpoint_cache(root,dict(policy="changed"),True)
            with self.assertRaises(FileExistsError):
                runner.prepare_endpoint_cache(root,identity,False)

    def test_interrupted_capture_reuses_finished_boards_and_runs_only_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = {}
            argv = ["runner", "--complete", "--turn-root-averages", "--workers", "1",
                    "--maximum-worker-memory-mib", "1536"]
            for name in ("binary", "preflop", "model", "kernel"):
                path = root/name
                path.write_text(name)
                inputs[str(path)] = sha256(path)
                argv += ["--"+name, str(path), "--"+name+"-sha256", sha256(path)]
            def capture(index):
                return dict(boardIndex=index, seconds=1, preflopSha256=inputs[str(root/"preflop")],
                    modelSha256=inputs[str(root/"model")], kernelSha256=inputs[str(root/"kernel")],
                    chanceSeed=32001, classes=[], multiplicities=[], rootHistory=[], rows=[],
                    continuationFunction=dict(rootRealizationTurnAverages=True))
            preflight = root/"preflight.json"
            partial = root/"partial.json"
            atomic_json(partial, capture(0))
            atomic_json(preflight, dict(status="complete", completeCapture=False,
                rootRealizationTurnAverages=True, pinnedInputs=inputs,
                jobs=[dict(output=str(partial), outputSha256=sha256(partial))]))
            source = root/"interrupted"
            source.mkdir()
            atomic_json(source/"manifest.json", dict(schema="frozen-preflop-response-controller-v1",
                status="running", completeCapture=True, rootRealizationTurnAverages=True,
                pinnedInputs=inputs, jobs=[]))
            for index in (0, 1):
                atomic_json(source/f"board{index}.json", capture(index))
                worker = source/f"board{index}-worker"
                worker.mkdir()
                (worker/"worker.log").write_text("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 388 filtered out; finished in 1.0s\n")
            manifest = source/"manifest.json"
            original = manifest.read_bytes()
            with patch.object(runner, "validate_capture"):
                expected = capture(0)
                for field, bad in (("boardIndex", 1), ("preflopSha256", "wrong"),
                                   ("continuationFunction", dict(rootRealizationTurnAverages=False))):
                    value = capture(0)
                    value[field] = bad
                    atomic_json(source/"board0.json", value)
                    with self.assertRaisesRegex(ValueError, "identity"):
                        runner.recover_captures(manifest, sha256(manifest), dict(inputs), expected)
                atomic_json(source/"board0.json", capture(0))
                log = source/"board0-worker/worker.log"
                success = log.read_text()
                log.write_text("still running")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    runner.recover_captures(manifest, sha256(manifest), dict(inputs), expected)
                log.write_text(success)
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    runner.recover_captures(manifest, "wrong", dict(inputs), expected)
            called = []
            def guarded(command, env, *args):
                index = int(env["POKER_NOISE_BOARD_INDEX"])
                self.assertEqual(args[2],1536*1024**2)
                self.assertEqual(env["POKER_RESPONSE_BINARY_SHA"],inputs[str(root/"binary")])
                self.assertEqual(Path(env["POKER_RESPONSE_ENDPOINT_CACHE"]).name,f"board{index}")
                called.append(index)
                atomic_json(Path(env["POKER_COMPACT_OUTPUT"]), capture(index))
                return dict(status="complete")
            output = root/"recovered"
            argv += ["--preflight", str(preflight), "--preflight-sha256", sha256(preflight),
                "--recover-from", str(manifest), "--recover-sha256", sha256(manifest),
                "--output", str(output)]
            with patch("sys.argv", argv), patch.object(runner, "validate_capture"), \
                    patch.object(runner, "projected_board_seconds", return_value=2), \
                    patch.object(runner, "guarded", side_effect=guarded), \
                    patch.object(runner, "summarize", side_effect=lambda values: dict(boards=sorted(v["boardIndex"] for v in values))), \
                    patch.object(runner.signal, "signal"):
                runner.main()
            record = json.loads((output/"manifest.json").read_text())
            self.assertEqual(sorted(called), [2, 3])
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["maximumConcurrentWorkers"],1)
            self.assertEqual(record["maximumWorkerMemoryBytes"],1536*1024**2)
            self.assertEqual(record["summary"]["boards"], [0, 1, 2, 3])
            self.assertEqual(manifest.read_bytes(), original)
            for job in record["jobs"][:2]:
                self.assertTrue(job["recovered"])
                self.assertIsNone(job["worker"])
                self.assertIn("unavailable", job["telemetry"])

    def test_chance_is_averaged_before_selection_and_evaluation_never_refits(self):
        rows = {(): dict(actor=0, children=[["a"], ["b"]], probabilities=[[.5], [.5]])}
        first = {("a",): np.array([[2.0], [0.0]]), ("b",): np.array([[0.0], [0.0]])}
        second = {("a",): np.array([[-2.0], [0.0]]), ("b",): np.array([[1.0], [0.0]])}
        mean = {h: (first[h]+second[h])/2 for h in first}
        choices = {}
        fitted = tree_values(rows, mean, (), 0, choices, True)
        np.testing.assert_array_equal(fitted, [.5])
        np.testing.assert_array_equal(choices[()], [1])
        # Inspecting the held-out board must not change b to the clairvoyant a.
        np.testing.assert_array_equal(tree_values(rows, first, (), 0, choices), [0.0])
        np.testing.assert_array_equal(choices[()], [1])

    def test_opponent_reach_is_not_multiplied_twice(self):
        rows = {(): dict(actor=1, children=[["a"], ["b"]], probabilities=[[.25], [.75]])}
        end = {("a",): np.array([[1.0], [0.0]]), ("b",): np.array([[3.0], [0.0]])}
        np.testing.assert_array_equal(tree_values(rows, end, (), 0), [4.0])

    def test_partial_capture_cannot_be_a_response_result(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()

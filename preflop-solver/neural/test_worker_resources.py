import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from worker_resources import RusageInfoV0, WorkerResourceGuard, process_memory_bytes


class WorkerResourcesTests(unittest.TestCase):
    def test_exit_race_empty_read_does_not_mark_successful_worker_stopped(self):
        process = Mock(pid=12345)
        calls = 0
        def poll():
            nonlocal calls
            calls += 1
            return None if calls < 3 else 0
        process.poll.side_effect = poll
        with patch("worker_resources.process_memory_bytes", return_value=(0, "macos_phys_footprint")), \
                patch("worker_resources.os.killpg", side_effect=AssertionError("exited child must not be signalled")):
            guard = WorkerResourceGuard(process, Path('/tmp'), interval=0.001)
            guard._watch()
        self.assertIsNone(guard.stop_reason)
        self.assertFalse(guard.stop_event.is_set())

    def test_group_signal_exit_race_falls_back_to_owned_child(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch("worker_resources.os.killpg", side_effect=PermissionError("exiting group")):
            WorkerResourceGuard(process, Path('/tmp'))._signal(signal.SIGTERM)
        process.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_persistent_disk_measurement_failure_stops_despite_valid_memory(self):
        process = Mock(pid=12345)
        # Three failed measurement cycles, followed by child exit. A memory
        # success must not reset failures from the disk measurement each time.
        polls = iter([None] * 7)
        process.poll.side_effect = lambda: next(polls, 0)
        with patch("worker_resources.process_memory_bytes", return_value=(1024, "test")), \
                patch("worker_resources.shutil.disk_usage", side_effect=OSError("disk unavailable")), \
                patch("worker_resources.os.killpg") as killpg:
            guard = WorkerResourceGuard(
                process, Path('/tmp'), minimum_free_disk_bytes=1, interval=0.001,
            )
            guard._watch()
        self.assertIn("disk unavailable", guard.stop_reason or "")
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)

    def test_live_process_memory_is_measurable(self):
        memory, metric = process_memory_bytes(os.getpid())
        self.assertGreater(memory, 0)
        self.assertIn(metric, ("macos_phys_footprint", "linux_rss_plus_swap"))
        self.assertEqual(ctypes.sizeof(RusageInfoV0), 96)
        self.assertEqual(RusageInfoV0.phys_footprint.offset, 72)

    def test_linux_counts_swap_and_rejects_missing_measurements(self):
        with patch("worker_resources.platform.system", return_value="Linux"):
            with patch.object(Path, "read_text", return_value="VmRSS: 120 kB\nVmSwap: 80 kB\n"):
                self.assertEqual(process_memory_bytes(123), (200 * 1024, "linux_rss_plus_swap"))
            with patch.object(Path, "read_text", return_value="Name: missing\n"):
                with self.assertRaises(OSError):
                    process_memory_bytes(123)

    def launch(self, directory, **kwargs):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        guard = WorkerResourceGuard(process, Path(directory), interval=0.02, **kwargs).start()
        return process, guard

    def test_memory_stop_leaves_completed_checkpoint_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "previous.checkpoint.msgpack.gz"
            checkpoint.write_bytes(b"last complete checkpoint")
            process, guard = self.launch(directory, max_memory_bytes=1)
            process.wait(timeout=5)
            record = guard.finish()
            self.assertIn("worker memory", record["resourceStopReason"])
            self.assertGreater(record["sampledPeakMemoryBytes"], 0)
            self.assertEqual(checkpoint.read_bytes(), b"last complete checkpoint")

    def test_timeout_stops_peer_and_escalates_if_term_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(30)"],
                stdout=subprocess.PIPE, start_new_session=True,
            )
            self.addCleanup(lambda: process.poll() is None and process.kill())
            self.assertEqual(process.stdout.readline(), b"ready\n")
            event = threading.Event()
            guard = WorkerResourceGuard(
                process, Path(directory), max_seconds=0.1,
                interval=0.02, grace_seconds=0.05, stop_event=event,
            ).start()
            process.wait(timeout=5)
            process.stdout.close()
            self.assertEqual(process.returncode, -signal.SIGKILL)
            self.assertIn("time limit", guard.finish()["resourceStopReason"])
            peer, peer_guard = self.launch(directory, stop_event=event)
            peer.wait(timeout=5)
            self.assertIn("another worker", peer_guard.finish()["resourceStopReason"])

    def test_failed_measurement_and_disk_reserve_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("worker_resources.process_memory_bytes", side_effect=OSError("unavailable")):
                process, guard = self.launch(directory)
                process.wait(timeout=5)
                self.assertIn("measurement failed", guard.finish()["resourceStopReason"])
            process, guard = self.launch(directory, minimum_free_disk_bytes=2**63)
            process.wait(timeout=5)
            self.assertIn("disk reserve", guard.finish()["resourceStopReason"])


if __name__ == "__main__":
    unittest.main()

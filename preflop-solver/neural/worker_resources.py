"""Sample a solver worker's real memory, including compressed/swapped pages.

These are sampled safety stops, not kernel-enforced allocation limits. Workers
must be launched with start_new_session=True; only their owned process group is
signalled. A killed checkpoint write leaves the last atomically renamed file.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import threading
import time


class RusageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (field, ctypes.c_uint64) for field in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime",
        )
    ]


def process_memory_bytes(pid: int) -> tuple[int, str]:
    if platform.system() == "Darwin":
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        libproc.proc_pid_rusage.restype = ctypes.c_int
        usage = RusageInfoV0()
        if libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return int(usage.phys_footprint), "macos_phys_footprint"
    if platform.system() == "Linux":
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmSwap"):
                fields[key] = int(value.split()[0]) * 1024
        if "VmRSS" not in fields or "VmSwap" not in fields:
            raise OSError("worker RSS/swap accounting unavailable")
        return fields["VmRSS"] + fields["VmSwap"], "linux_rss_plus_swap"
    raise OSError("live worker memory guard supports macOS and Linux only")


class WorkerResourceGuard:
    def __init__(
        self, process: subprocess.Popen, output_dir: Path,
        max_memory_bytes: int = 0, max_seconds: float = 0,
        minimum_free_disk_bytes: int = 0,
        stop_event: threading.Event | None = None,
        interval: float = 0.25, grace_seconds: float = 3.0,
    ):
        self.process = process
        self.output_dir = output_dir
        self.max_memory_bytes = max_memory_bytes
        self.max_seconds = max_seconds
        self.minimum_free_disk_bytes = minimum_free_disk_bytes
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.interval = interval
        self.grace_seconds = grace_seconds
        self.peak_memory_bytes = 0
        self.memory_metric = None
        self.stop_reason = None
        self.signal_error = None
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def request_stop(self, reason: str):
        if self.stop_reason is None:
            self.stop_reason = reason
        self.stop_event.set()

    def _signal(self, value: int):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, value)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS may revoke access to a process group while its final
                # member exits. Popen rechecks/reaps its own child before a
                # PID signal, avoiding the group-exit race and PID reuse.
                try:
                    self.process.send_signal(value)
                except (ProcessLookupError, PermissionError) as error:
                    if self.process.poll() is None:
                        self.signal_error = str(error)

    def _watch(self):
        stopping_at = None
        measurement_failures = 0
        while self.process.poll() is None:
            if not self.stop_event.is_set():
                try:
                    memory, self.memory_metric = process_memory_bytes(self.process.pid)
                    if memory <= 0:
                        raise OSError("empty worker memory reading")
                    self.peak_memory_bytes = max(self.peak_memory_bytes, memory)
                    if self.max_memory_bytes and memory >= self.max_memory_bytes:
                        self.request_stop(f"worker memory {memory} >= {self.max_memory_bytes} bytes")
                    if self.max_seconds and time.monotonic() - self.started >= self.max_seconds:
                        self.request_stop(f"worker time limit {self.max_seconds:g}s reached")
                    if (self.minimum_free_disk_bytes and
                            shutil.disk_usage(self.output_dir).free < self.minimum_free_disk_bytes):
                        self.request_stop("minimum free disk reserve reached")
                    measurement_failures = 0
                except (OSError, ValueError) as error:
                    if self.process.poll() is None:
                        measurement_failures += 1
                        # A dying process may return one empty/error reading
                        # just before waitpid observes completion. Persistently
                        # missing telemetry still fails closed within 3 polls.
                        if measurement_failures >= 3:
                            self.request_stop(f"resource measurement failed: {error}")
            if self.stop_event.is_set():
                if self.stop_reason is None:
                    self.stop_reason = "another worker or the operator stopped the stage"
                if stopping_at is None:
                    stopping_at = time.monotonic()
                    self._signal(signal.SIGTERM)
                elif time.monotonic() - stopping_at >= self.grace_seconds:
                    self._signal(signal.SIGKILL)
            time.sleep(self.interval)

    def finish(self) -> dict[str, object]:
        self.thread.join(timeout=self.interval + 1)
        return {
            "sampledPeakMemoryBytes": self.peak_memory_bytes,
            "memoryMetric": self.memory_metric,
            "resourceStopReason": self.stop_reason,
            "signalError": self.signal_error,
            "workerElapsedSeconds": round(time.monotonic() - self.started, 3),
        }

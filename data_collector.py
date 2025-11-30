"""Kernel scheduling data collector.

This module collects real scheduling data from the running system using ps/top
commands. Data is parsed and converted into our standardised TaskFeatures format.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import json
import re
import subprocess
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

# Project/local
from constants import (
    CPU_HISTORY_WINDOW_SIZE,
    DEFAULT_SAMPLE_INTERVAL_MS,
    MAX_TASK_HISTORY,
    SnapshotField,
    TaskField,
)
from scheduler_types import TaskFeatures, TaskType

# =============================================================================
# 2. TYPES & CONSTANTS
# =============================================================================

# Time format patterns in ps output
TIME_PATTERN_HMS = re.compile(r"(\d+):(\d+):(\d+\.\d+)")  # HH:MM:SS.ss
TIME_PATTERN_MS = re.compile(r"(\d+):(\d+\.\d+)")  # MM:SS.ss

# =============================================================================
# 3. PUBLIC API
# =============================================================================


def collect_scheduling_trace(
    duration_s: float, interval_ms: float = DEFAULT_SAMPLE_INTERVAL_MS
) -> list["SchedulingSnapshot"]:
    """Collect scheduling data trace from running system.

    Args:
        duration_s: How long to collect data (seconds).
        interval_ms: Sampling interval (milliseconds).

    Returns:
        List of scheduling snapshots over time.
    """
    collector = SystemDataCollector(sample_interval_ms=interval_ms)
    return collector.collect_trace(duration_s=duration_s)


def save_trace_to_file(trace: list["SchedulingSnapshot"], filepath: Path) -> None:
    """Save scheduling trace to JSON file.

    Args:
        trace: Scheduling snapshots to save.
        filepath: Where to save the data.
    """
    data = [snapshot.to_dict() for snapshot in trace]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# 4. CORE LOGIC & COMPONENTS
# =============================================================================


@dataclass
class SchedulingSnapshot:
    """Snapshot of all runnable tasks at a point in time.

    Attributes:
        timestamp: When snapshot was taken (seconds since epoch).
        tasks: List of task features at this moment.
        num_cores: Number of CPU cores.
        load_avg: System load average (1-minute).
    """

    timestamp: float
    tasks: list[TaskFeatures]
    num_cores: int
    load_avg: float

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for serialisation.

        Returns:
            Dictionary representation.
        """
        return {
            SnapshotField.TIMESTAMP: self.timestamp,
            SnapshotField.NUM_CORES: self.num_cores,
            SnapshotField.LOAD_AVG: self.load_avg,
            SnapshotField.TASKS: [self._task_to_dict(task) for task in self.tasks],
        }

    @staticmethod
    def _task_to_dict(task: TaskFeatures) -> dict[str, t.Any]:
        """Convert task features to dictionary."""
        return {
            TaskField.TASK_ID: task.task_id,
            TaskField.PARENT_ID: task.parent_id,
            TaskField.PRIORITY: task.priority,
            TaskField.NICE: task.nice,
            TaskField.CPU_USAGE_HIST: task.cpu_usage_hist,
            TaskField.TOTAL_CPU_TIME: task.total_cpu_time,
            TaskField.VIRTUAL_SIZE: task.virtual_size,
            TaskField.RESIDENT_SIZE: task.resident_size,
            TaskField.TASK_TYPE: task.task_type.value,
        }


class SystemDataCollector:
    """Collects scheduling data from the running system.

    Uses ps command to periodically sample process information and convert
    it to our TaskFeatures format.
    """

    def __init__(self, sample_interval_ms: float = DEFAULT_SAMPLE_INTERVAL_MS) -> None:
        """Initialise data collector.

        Args:
            sample_interval_ms: How often to sample (milliseconds).
        """
        self._sample_interval_ms = sample_interval_ms
        self._num_cores = self._detect_num_cores()
        self._task_history: dict[int, list[TaskFeatures]] = {}

    def collect_trace(self, duration_s: float) -> list[SchedulingSnapshot]:
        """Collect scheduling trace for specified duration.

        Args:
            duration_s: How long to collect (seconds).

        Returns:
            List of scheduling snapshots.
        """
        snapshots: list[SchedulingSnapshot] = []
        start_time = time.time()
        interval_s = self._sample_interval_ms / 1000.0

        while (time.time() - start_time) < duration_s:
            snapshot = self._take_snapshot()
            snapshots.append(snapshot)
            time.sleep(interval_s)

        return snapshots

    def _take_snapshot(self) -> SchedulingSnapshot:
        """Take a single snapshot of current system state.

        Returns:
            Scheduling snapshot at this moment.
        """
        timestamp = time.time()
        ps_output = self._run_ps_command()
        load_avg = self._get_load_average()

        tasks: list[TaskFeatures] = []
        for line in ps_output.split("\n")[1:]:  # Skip header
            if not line.strip():
                continue
            task = self._parse_ps_line(line, timestamp)
            if task:
                tasks.append(task)
                self._update_task_history(task)

        return SchedulingSnapshot(
            timestamp=timestamp,
            tasks=tasks,
            num_cores=self._num_cores,
            load_avg=load_avg,
        )

    # =============================================================================
    # 5. PRIVATE HELPERS & UTILITIES
    # =============================================================================

    def _run_ps_command(self) -> str:
        """Execute ps command to get process information.

        Returns:
            Raw ps output as string.
        """
        # macOS ps command format
        cmd = [
            "ps",
            "-eo",
            "pid,ppid,pri,nice,vsz,rss,pcpu,pmem,time,comm",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=5.0,
            )
            return result.stdout
        except subprocess.SubprocessError:
            # Fallback to empty output if ps fails
            return ""

    def _parse_ps_line(self, line: str, timestamp: float) -> TaskFeatures | None:
        """Parse a single line of ps output into TaskFeatures.

        Args:
            line: One line from ps output.
            timestamp: Current timestamp.

        Returns:
            Parsed task features or None if parsing fails.
        """
        fields = line.split()
        if len(fields) < 10:
            return None

        try:
            task_id = int(fields[0])
            parent_id = int(fields[1])
            priority = int(fields[2])
            nice = int(fields[3])
            virtual_size = int(fields[4]) * 1024  # KB to bytes
            resident_size = int(fields[5]) * 1024  # KB to bytes
            cpu_percent = float(fields[6])
            # mem_percent = float(fields[7])  # Not currently used
            time_str = fields[8]
            # command = " ".join(fields[9:])  # Not currently used

            # Parse total CPU time
            total_cpu_time = self._parse_time_string(time_str)

            # Get history if available
            cpu_usage_hist = self._get_cpu_history(task_id, cpu_percent)

            # Estimate task type based on CPU usage pattern
            task_type = self._estimate_task_type(cpu_usage_hist)

            return TaskFeatures(
                task_id=task_id,
                parent_id=parent_id,
                priority=priority,
                nice=nice,
                cpu_usage_hist=cpu_usage_hist,
                total_cpu_time=total_cpu_time,
                virtual_size=virtual_size,
                resident_size=resident_size,
                vruntime=0.0,  # Not available from ps
                time_since_scheduled=0.0,  # Not available
                sleep_time=0.0,  # Not available
                run_time=0.0,  # Not available
                io_bound_score=0.5,  # Computed later from history
                interactive_score=0.5,  # Computed later from history
                num_threads=1,  # Not easily available from ps
                last_core=0,  # Not available
                arrival_time=timestamp,
                task_type=task_type,
            )
        except (ValueError, IndexError):
            return None

    def _parse_time_string(self, time_str: str) -> float:
        """Parse TIME field from ps output (HH:MM:SS.ss or MM:SS.ss).

        Args:
            time_str: Time string from ps output.

        Returns:
            Total seconds as float.
        """
        # Try HH:MM:SS.ss format
        match = TIME_PATTERN_HMS.match(time_str)
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        # Try MM:SS.ss format
        match = TIME_PATTERN_MS.match(time_str)
        if match:
            minutes, seconds = match.groups()
            return int(minutes) * 60 + float(seconds)

        # Fallback
        return 0.0

    def _get_cpu_history(self, task_id: int, current_cpu: float) -> list[float]:
        """Get CPU usage history for a task.

        Args:
            task_id: Task identifier.
            current_cpu: Current CPU percentage.

        Returns:
            List of recent CPU usage values [0.0, 1.0].
        """
        if task_id not in self._task_history:
            # New task, initialise with current value
            return [current_cpu / 100.0] * CPU_HISTORY_WINDOW_SIZE

        # Get previous task features
        history = self._task_history[task_id]
        if not history:
            return [current_cpu / 100.0] * CPU_HISTORY_WINDOW_SIZE

        # Take last task's history and append current
        prev_hist = history[-1].cpu_usage_hist
        new_hist = prev_hist[1:] + [current_cpu / 100.0]
        return new_hist

    def _update_task_history(self, task: TaskFeatures) -> None:
        """Update task history with new observation.

        Args:
            task: Task features to add to history.
        """
        if task.task_id not in self._task_history:
            self._task_history[task.task_id] = []

        self._task_history[task.task_id].append(task)

        # Keep only last N observations per task to avoid memory growth
        if len(self._task_history[task.task_id]) > MAX_TASK_HISTORY:
            self._task_history[task.task_id] = self._task_history[task.task_id][
                -MAX_TASK_HISTORY:
            ]

    def _estimate_task_type(self, cpu_hist: list[float]) -> TaskType:
        """Estimate task type from CPU usage history.

        Args:
            cpu_hist: Recent CPU usage history.

        Returns:
            Estimated task type.
        """
        avg_cpu = sum(cpu_hist) / len(cpu_hist)
        variance = sum((x - avg_cpu) ** 2 for x in cpu_hist) / len(cpu_hist)

        # High CPU, low variance = CPU bound
        if avg_cpu > 0.7 and variance < 0.05:
            return TaskType.CPU_BOUND

        # Low CPU, high variance = I/O bound
        if avg_cpu < 0.3 and variance > 0.1:
            return TaskType.IO_BOUND

        # Medium CPU, high variance = interactive
        if 0.2 < avg_cpu < 0.6 and variance > 0.1:
            return TaskType.INTERACTIVE

        # Low CPU, low variance = batch
        if avg_cpu < 0.4 and variance < 0.05:
            return TaskType.BATCH

        return TaskType.MIXED

    def _detect_num_cores(self) -> int:
        """Detect number of CPU cores.

        Returns:
            Number of CPU cores.
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                check=True,
                timeout=1.0,
            )
            return int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            return 4  # Fallback default

    def _get_load_average(self) -> float:
        """Get system 1-minute load average.

        Returns:
            Load average.
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "vm.loadavg"],
                capture_output=True,
                text=True,
                check=True,
                timeout=1.0,
            )
            # Output format: "{ 1.50 2.30 2.10 }"
            parts = result.stdout.strip().strip("{}").split()
            return float(parts[0])
        except (subprocess.SubprocessError, ValueError, IndexError):
            return 0.0

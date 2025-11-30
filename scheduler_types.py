"""Core data types for scheduler simulation.

This module defines all data structures used throughout the scheduler simulator,
including task features, system state, scheduling events, and performance metrics.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from __future__ import annotations

# Standard library
import typing as t
from dataclasses import dataclass, field
from enum import Enum

# Project/local
from constants import (
    CPU_HISTORY_WINDOW_SIZE,
    DEFAULT_NICE,
    DEFAULT_PRIORITY,
    GB,
    ArrivalPattern,
    MetricField,
    PowerModeValue,
    SchedulerName,
    TaskField,
    TaskTypeValue,
)

# =============================================================================
# 2. TYPES & CONSTANTS
# =============================================================================


class PowerMode(Enum):
    """System power mode settings."""

    PERFORMANCE = PowerModeValue.PERFORMANCE
    BALANCED = PowerModeValue.BALANCED
    POWERSAVE = PowerModeValue.POWERSAVE


class TaskType(Enum):
    """Task classification by behaviour."""

    CPU_BOUND = TaskTypeValue.CPU_BOUND
    IO_BOUND = TaskTypeValue.IO_BOUND
    INTERACTIVE = TaskTypeValue.INTERACTIVE
    BATCH = TaskTypeValue.BATCH
    MIXED = TaskTypeValue.MIXED


# =============================================================================
# 3. TASK FEATURES
# =============================================================================


@dataclass
class TaskFeatures:
    """Features for a single task at scheduling decision point.

    Attributes:
        task_id: Unique task identifier.
        parent_id: Parent task ID (0 for root tasks).
        priority: Scheduling priority (0-139, Linux CFS range).
        nice: Nice value (-20 to 19).
        cpu_usage_hist: Recent CPU usage in windows [0.0, 1.0].
        total_cpu_time: Cumulative CPU time consumed (seconds).
        virtual_size: Virtual memory size (bytes).
        resident_size: Resident set size / physical memory (bytes).
        vruntime: Virtual runtime for CFS (nanoseconds).
        time_since_scheduled: Time since last scheduled (milliseconds).
        sleep_time: Total time spent sleeping (milliseconds).
        run_time: Total time spent running (milliseconds).
        io_bound_score: I/O bound metric (0.0=CPU, 1.0=I/O).
        interactive_score: Interactivity metric (0.0=batch, 1.0=interactive).
        num_threads: Number of threads in this task.
        last_core: CPU core ID where last executed.
        arrival_time: When task first became runnable (milliseconds).
        task_type: Classification of task behaviour.
    """

    task_id: int
    parent_id: int = 0
    priority: int = DEFAULT_PRIORITY
    nice: int = DEFAULT_NICE
    cpu_usage_hist: list[float] = field(
        default_factory=lambda: [0.0] * CPU_HISTORY_WINDOW_SIZE
    )
    total_cpu_time: float = 0.0
    virtual_size: int = 0
    resident_size: int = 0
    vruntime: float = 0.0
    time_since_scheduled: float = 0.0
    sleep_time: float = 0.0
    run_time: float = 0.0
    io_bound_score: float = 0.5
    interactive_score: float = 0.5
    num_threads: int = 1
    last_core: int = 0
    arrival_time: float = 0.0
    task_type: TaskType = TaskType.MIXED

    def to_feature_vector(self) -> list[float]:
        """Convert task features to flat numerical vector for ML model.

        Returns:
            Flattened feature vector suitable for neural network input.
        """
        return [
            float(self.task_id),
            float(self.priority),
            float(self.nice),
            *self.cpu_usage_hist,
            self.total_cpu_time,
            float(self.virtual_size) / GB,  # Normalise to GB
            float(self.resident_size) / GB,
            self.vruntime / 1e9,  # Normalise nanoseconds
            self.time_since_scheduled,
            self.sleep_time,
            self.run_time,
            self.io_bound_score,
            self.interactive_score,
            float(self.num_threads),
            float(self.last_core),
        ]

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary using StrEnum field names.

        Returns:
            Dictionary representation.
        """
        return {
            TaskField.TASK_ID: self.task_id,
            TaskField.PARENT_ID: self.parent_id,
            TaskField.PRIORITY: self.priority,
            TaskField.NICE: self.nice,
            TaskField.CPU_USAGE_HIST: self.cpu_usage_hist,
            TaskField.TOTAL_CPU_TIME: self.total_cpu_time,
            TaskField.VIRTUAL_SIZE: self.virtual_size,
            TaskField.RESIDENT_SIZE: self.resident_size,
            TaskField.VRUNTIME: self.vruntime,
            TaskField.TIME_SINCE_SCHEDULED: self.time_since_scheduled,
            TaskField.SLEEP_TIME: self.sleep_time,
            TaskField.RUN_TIME: self.run_time,
            TaskField.IO_BOUND_SCORE: self.io_bound_score,
            TaskField.INTERACTIVE_SCORE: self.interactive_score,
            TaskField.NUM_THREADS: self.num_threads,
            TaskField.LAST_CORE: self.last_core,
            TaskField.ARRIVAL_TIME: self.arrival_time,
            TaskField.TASK_TYPE: self.task_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> TaskFeatures:
        """Create TaskFeatures from dictionary.

        Args:
            data: Dictionary with task feature data.

        Returns:
            TaskFeatures instance.
        """
        task_type_str = data.get(TaskField.TASK_TYPE, TaskTypeValue.MIXED)
        return cls(
            task_id=data[TaskField.TASK_ID],
            parent_id=data.get(TaskField.PARENT_ID, 0),
            priority=data.get(TaskField.PRIORITY, DEFAULT_PRIORITY),
            nice=data.get(TaskField.NICE, DEFAULT_NICE),
            cpu_usage_hist=data.get(
                TaskField.CPU_USAGE_HIST, [0.0] * CPU_HISTORY_WINDOW_SIZE
            ),
            total_cpu_time=data.get(TaskField.TOTAL_CPU_TIME, 0.0),
            virtual_size=data.get(TaskField.VIRTUAL_SIZE, 0),
            resident_size=data.get(TaskField.RESIDENT_SIZE, 0),
            vruntime=data.get(TaskField.VRUNTIME, 0.0),
            time_since_scheduled=data.get(TaskField.TIME_SINCE_SCHEDULED, 0.0),
            sleep_time=data.get(TaskField.SLEEP_TIME, 0.0),
            run_time=data.get(TaskField.RUN_TIME, 0.0),
            io_bound_score=data.get(TaskField.IO_BOUND_SCORE, 0.5),
            interactive_score=data.get(TaskField.INTERACTIVE_SCORE, 0.5),
            num_threads=data.get(TaskField.NUM_THREADS, 1),
            last_core=data.get(TaskField.LAST_CORE, 0),
            arrival_time=data.get(TaskField.ARRIVAL_TIME, 0.0),
            task_type=TaskType[task_type_str]
            if isinstance(task_type_str, str)
            else task_type_str,
        )


# =============================================================================
# 4. SYSTEM STATE
# =============================================================================


@dataclass
class SystemFeatures:
    """Global system state at scheduling decision point.

    Attributes:
        current_time: Simulation time (milliseconds).
        tick: Scheduler tick counter.
        num_cores: Number of CPU cores.
        core_loads: Per-core load average [0.0, 1.0].
        runqueue_lengths: Number of runnable tasks per core.
        total_memory: Total system memory (bytes).
        free_memory: Available memory (bytes).
        memory_pressure: Memory pressure metric (0.0=plenty, 1.0=swapping).
        temperature: CPU temperature (Celsius).
        thermal_throttled: Whether thermal throttling is active.
        power_mode: Current power management mode.
        battery_level: Battery charge level (0.0 to 1.0).
    """

    current_time: float
    tick: int
    num_cores: int
    core_loads: list[float] = field(default_factory=lambda: [])
    runqueue_lengths: list[int] = field(default_factory=lambda: [])
    total_memory: int = 16 * 1024**3  # 16GB default
    free_memory: int = 8 * 1024**3  # 8GB default
    memory_pressure: float = 0.0
    temperature: float = 50.0
    thermal_throttled: bool = False
    power_mode: PowerMode = PowerMode.BALANCED
    battery_level: float = 1.0

    def to_feature_vector(self) -> list[float]:
        """Convert system features to flat numerical vector for ML model.

        Returns:
            Flattened feature vector suitable for neural network input.
        """
        # Encode power mode as one-hot
        power_mode_encoding = [
            1.0 if self.power_mode == PowerMode.PERFORMANCE else 0.0,
            1.0 if self.power_mode == PowerMode.BALANCED else 0.0,
            1.0 if self.power_mode == PowerMode.POWERSAVE else 0.0,
        ]

        return [
            self.current_time / 1000.0,  # Normalise to seconds
            float(self.num_cores),
            sum(self.core_loads) / len(self.core_loads) if self.core_loads else 0.0,
            sum(self.runqueue_lengths) / len(self.runqueue_lengths)
            if self.runqueue_lengths
            else 0.0,
            float(self.free_memory) / float(self.total_memory),
            self.memory_pressure,
            self.temperature / 100.0,  # Normalise
            1.0 if self.thermal_throttled else 0.0,
            *power_mode_encoding,
            self.battery_level,
        ]


# =============================================================================
# 5. SCHEDULING EVENT
# =============================================================================


@dataclass
class SchedulingEvent:
    """A single scheduling decision point.

    Attributes:
        timestamp: When this event occurred (milliseconds).
        core_id: Which CPU core needs scheduling.
        runnable_tasks: Tasks available to schedule.
        system_state: Global system state.
        selected_task: Task ID chosen by scheduler (None if idle).
        time_quantum: How long selected task should run (milliseconds).
        scheduler_name: Which scheduler made this decision.
    """

    timestamp: float
    core_id: int
    runnable_tasks: list[TaskFeatures]
    system_state: SystemFeatures
    selected_task: int | None = None
    time_quantum: float = 0.0
    scheduler_name: str = SchedulerName.UNKNOWN


# =============================================================================
# 6. PERFORMANCE METRICS
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during simulation run.

    Attributes:
        scheduler_name: Name of the scheduler being evaluated.
        tasks_completed: Number of tasks that finished.
        total_runtime: Total simulation time (seconds).
        throughput: Tasks completed per second.
        mean_latency: Mean task completion time (milliseconds).
        p50_latency: Median task completion time.
        p95_latency: 95th percentile latency.
        p99_latency: 99th percentile latency.
        max_latency: Maximum observed latency.
        cpu_time_variance: Variance in CPU time allocation.
        starvation_events: Tasks starved >100ms.
        context_switches: Total context switch count.
        avg_time_quantum: Average time quantum used (milliseconds).
        cpu_utilisation: CPU utilisation fraction [0.0, 1.0].
        interactive_p99: 99th percentile latency for interactive tasks.
        total_energy: Total energy consumed (Joules).
        energy_per_task: Energy per task (Joules).
    """

    scheduler_name: str
    tasks_completed: int = 0
    total_runtime: float = 0.0
    throughput: float = 0.0
    mean_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    max_latency: float = 0.0
    cpu_time_variance: float = 0.0
    starvation_events: int = 0
    context_switches: int = 0
    avg_time_quantum: float = 0.0
    cpu_utilisation: float = 0.0
    interactive_p99: float = 0.0
    total_energy: float = 0.0
    energy_per_task: float = 0.0

    def to_summary_dict(self) -> dict[str, t.Any]:
        """Convert metrics to dictionary using StrEnum field names.

        Returns:
            Dictionary with all metric values.
        """
        return {
            MetricField.SCHEDULER_NAME: self.scheduler_name,
            MetricField.TASKS_COMPLETED: self.tasks_completed,
            MetricField.TOTAL_RUNTIME: round(self.total_runtime, 2),
            MetricField.THROUGHPUT: round(self.throughput, 2),
            MetricField.MEAN_LATENCY: round(self.mean_latency, 2),
            MetricField.P50_LATENCY: round(self.p50_latency, 2),
            MetricField.P95_LATENCY: round(self.p95_latency, 2),
            MetricField.P99_LATENCY: round(self.p99_latency, 2),
            MetricField.MAX_LATENCY: round(self.max_latency, 2),
            MetricField.STARVATION_EVENTS: self.starvation_events,
            MetricField.CONTEXT_SWITCHES: self.context_switches,
            MetricField.CPU_UTILISATION: round(self.cpu_utilisation, 3),
            MetricField.INTERACTIVE_P99: round(self.interactive_p99, 2),
            MetricField.TOTAL_ENERGY: round(self.total_energy, 2),
            MetricField.ENERGY_PER_TASK: round(self.energy_per_task, 4),
        }


# =============================================================================
# 7. WORKLOAD DEFINITION
# =============================================================================


@dataclass
class WorkloadConfig:
    """Configuration for synthetic workload generation.

    Attributes:
        num_tasks: Total number of tasks to generate.
        duration: Simulation duration (seconds).
        task_mix: Distribution of task types.
        arrival_pattern: "constant", "poisson", or "bursty".
        arrival_rate: Average task arrivals per second.
        cpu_intensity_range: Range of CPU intensities (min, max).
        io_frequency_range: Range of I/O frequencies (min, max).
    """

    num_tasks: int
    duration: float
    task_mix: dict[TaskType, float] = field(
        default_factory=lambda: {
            TaskType.CPU_BOUND: 0.3,
            TaskType.IO_BOUND: 0.3,
            TaskType.INTERACTIVE: 0.2,
            TaskType.BATCH: 0.2,
        }
    )
    arrival_pattern: str = ArrivalPattern.POISSON
    arrival_rate: float = 10.0  # Tasks per second
    cpu_intensity_range: tuple[float, float] = (0.1, 1.0)
    io_frequency_range: tuple[float, float] = (0.0, 100.0)  # ms between I/O


@dataclass
class Task:
    """A simulated task in the workload.

    Attributes:
        task_id: Unique identifier.
        task_type: Classification of task behaviour.
        arrival_time: When task becomes runnable (milliseconds).
        total_work: Total CPU time needed (milliseconds).
        cpu_bursts: List of (cpu_time_ms, io_wait_ms) tuples.
        priority: Task priority.
        nice: Nice value.
        completion_time: When task finished (None if running).
    """

    task_id: int
    task_type: TaskType
    arrival_time: float
    total_work: float
    cpu_bursts: list[tuple[float, float]] = field(default_factory=lambda: [])
    priority: int = DEFAULT_PRIORITY
    nice: int = DEFAULT_NICE
    completion_time: float | None = None
    work_remaining: float = 0.0

    def __post_init__(self) -> None:
        """Initialise work remaining after creation."""
        self.work_remaining = self.total_work

    def is_completed(self) -> bool:
        """Check if task has finished all work.

        Returns:
            True if task is complete.
        """
        return self.work_remaining <= 0.0

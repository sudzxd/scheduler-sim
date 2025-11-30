"""Constants and enumerations for scheduler simulator.

This module defines all string constants as StrEnum to ensure type safety
and prevent typos throughout the codebase.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
from enum import StrEnum

# =============================================================================
# 2. FIELD NAMES
# =============================================================================

class TaskField(StrEnum):
    """Field names for task data structures."""

    TASK_ID = "task_id"
    PARENT_ID = "parent_id"
    PRIORITY = "priority"
    NICE = "nice"
    CPU_USAGE_HIST = "cpu_usage_hist"
    TOTAL_CPU_TIME = "total_cpu_time"
    VIRTUAL_SIZE = "virtual_size"
    RESIDENT_SIZE = "resident_size"
    VRUNTIME = "vruntime"
    TIME_SINCE_SCHEDULED = "time_since_scheduled"
    SLEEP_TIME = "sleep_time"
    RUN_TIME = "run_time"
    IO_BOUND_SCORE = "io_bound_score"
    INTERACTIVE_SCORE = "interactive_score"
    NUM_THREADS = "num_threads"
    LAST_CORE = "last_core"
    ARRIVAL_TIME = "arrival_time"
    TASK_TYPE = "task_type"
    COMPLETION_TIME = "completion_time"
    WORK_REMAINING = "work_remaining"


class SystemField(StrEnum):
    """Field names for system state data."""

    CURRENT_TIME = "current_time"
    TICK = "tick"
    NUM_CORES = "num_cores"
    CORE_LOADS = "core_loads"
    RUNQUEUE_LENGTHS = "runqueue_lengths"
    TOTAL_MEMORY = "total_memory"
    FREE_MEMORY = "free_memory"
    MEMORY_PRESSURE = "memory_pressure"
    TEMPERATURE = "temperature"
    THERMAL_THROTTLED = "thermal_throttled"
    POWER_MODE = "power_mode"
    BATTERY_LEVEL = "battery_level"


class MetricField(StrEnum):
    """Field names for performance metrics."""

    SCHEDULER_NAME = "scheduler_name"
    TASKS_COMPLETED = "tasks_completed"
    TOTAL_RUNTIME = "total_runtime"
    THROUGHPUT = "throughput"
    MEAN_LATENCY = "mean_latency"
    P50_LATENCY = "p50_latency"
    P95_LATENCY = "p95_latency"
    P99_LATENCY = "p99_latency"
    MAX_LATENCY = "max_latency"
    CPU_TIME_VARIANCE = "cpu_time_variance"
    STARVATION_EVENTS = "starvation_events"
    CONTEXT_SWITCHES = "context_switches"
    AVG_TIME_QUANTUM = "avg_time_quantum"
    CPU_UTILISATION = "cpu_utilisation"
    INTERACTIVE_P99 = "interactive_p99"
    TOTAL_ENERGY = "total_energy"
    ENERGY_PER_TASK = "energy_per_task"


class SnapshotField(StrEnum):
    """Field names for scheduling snapshots."""

    TIMESTAMP = "timestamp"
    NUM_CORES = "num_cores"
    LOAD_AVG = "load_avg"
    TASKS = "tasks"


# =============================================================================
# 3. SCHEDULER TYPES
# =============================================================================


class SchedulerName(StrEnum):
    """Scheduler implementation names."""

    CFS = "cfs"
    STM = "stm"
    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    UNKNOWN = "unknown"


# =============================================================================
# 4. WORKLOAD PATTERNS
# =============================================================================


class ArrivalPattern(StrEnum):
    """Task arrival patterns for workload generation."""

    CONSTANT = "constant"
    POISSON = "poisson"
    BURSTY = "bursty"
    PERIODIC = "periodic"


# =============================================================================
# 5. POWER MODES
# =============================================================================


class PowerModeValue(StrEnum):
    """Power management mode values."""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWERSAVE = "powersave"


# =============================================================================
# 6. TASK TYPES
# =============================================================================


class TaskTypeValue(StrEnum):
    """Task behaviour classification values."""

    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    MIXED = "mixed"


# =============================================================================
# 7. FILE FORMATS
# =============================================================================


class FileFormat(StrEnum):
    """Supported file formats for data export."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"


# =============================================================================
# 8. PLOT TYPES
# =============================================================================


class PlotType(StrEnum):
    """Types of plots for visualisation."""

    LATENCY_CDF = "latency_cdf"
    THROUGHPUT_TIME = "throughput_time"
    CPU_UTILISATION = "cpu_utilisation"
    SCHEDULER_COMPARISON = "scheduler_comparison"
    TASK_TIMELINE = "task_timeline"
    ENERGY_CONSUMPTION = "energy_consumption"


# =============================================================================
# 9. NUMERIC CONSTANTS
# =============================================================================

# Default values
DEFAULT_SAMPLE_INTERVAL_MS = 100.0
DEFAULT_TIME_QUANTUM_MS = 6.0  # Linux CFS default
DEFAULT_MIN_GRANULARITY_MS = 0.75  # Linux CFS minimum
DEFAULT_PRIORITY = 120  # CFS default priority
DEFAULT_NICE = 0

# CFS constants
CFS_MIN_PRIORITY = 100
CFS_MAX_PRIORITY = 139
CFS_NICE_MIN = -20
CFS_NICE_MAX = 19

# Simulation constants
STARVATION_THRESHOLD_MS = 100.0
CPU_HISTORY_WINDOW_SIZE = 8
MAX_TASK_HISTORY = 100

# Memory constants
GB = 1024**3
MB = 1024**2
KB = 1024

# Time constants
MS_PER_SECOND = 1000.0
NS_PER_MS = 1_000_000

# Training constants
TASK_FEATURE_DIM = 22  # From TaskFeatures.to_feature_vector()
CONTEXT_FEATURE_DIM = 12  # From SystemFeatures.to_feature_vector()
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 100
DEFAULT_WARMUP_EPOCHS = 10
DEFAULT_EPISODES_PER_EPOCH = 100
DEFAULT_MAX_TASKS_PER_EPISODE = 20
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SAVE_EVERY = 10
DEFAULT_CHECKPOINT_DIR = "checkpoints"

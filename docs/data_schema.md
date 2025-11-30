# Scheduler Data Schema

## Real Kernel Data (Available via ps/top on macOS)

```bash
PID    - Process ID
PPID   - Parent process ID
PRI    - Priority (0-200, lower = higher priority)
NI     - Nice value (-20 to 19)
VSZ    - Virtual memory size (KB)
RSS    - Resident set size (physical memory, KB)
%CPU   - CPU usage percentage
%MEM   - Memory usage percentage
TIME   - Total CPU time consumed
COMM   - Command name
#TH    - Number of threads
```

## Our Simulation Schema

### Task Features (input to scheduler)

Based on real kernel data + additional computed features:

```python
@dataclass
class TaskFeatures:
    """Features for a single task at scheduling decision point."""

    # Identity
    task_id: int
    parent_id: int

    # Priority & Niceness
    priority: int           # 0-139 (Linux CFS range)
    nice: int              # -20 to 19

    # CPU History (recent windows)
    cpu_usage_hist: list[float]  # Last 8 time windows, [0.0, 1.0]
    total_cpu_time: float        # Cumulative CPU time (seconds)

    # Memory
    virtual_size: int      # Virtual memory (bytes)
    resident_size: int     # Physical memory (bytes)

    # Scheduling State
    vruntime: float        # Virtual runtime (for CFS)
    time_since_scheduled: float  # Time since last scheduled (ms)
    sleep_time: float      # Time spent sleeping (ms)
    run_time: float        # Time spent running (ms)

    # Patterns (computed)
    io_bound_score: float  # 0.0 = CPU bound, 1.0 = I/O bound
    interactive_score: float  # 0.0 = batch, 1.0 = interactive

    # Context
    num_threads: int
    last_core: int         # Which CPU core last ran on
```

### System Features (global context)

```python
@dataclass
class SystemFeatures:
    """Global system state at scheduling decision point."""

    # Time
    current_time: float    # Simulation time (ms)
    tick: int             # Scheduler tick counter

    # CPU State
    num_cores: int
    core_loads: list[float]        # Per-core load [0.0, 1.0]
    runqueue_lengths: list[int]    # Tasks waiting per core

    # Memory Pressure
    total_memory: int     # Total system memory (bytes)
    free_memory: int      # Free memory (bytes)
    memory_pressure: float  # 0.0 = plenty, 1.0 = swapping

    # Thermal (simulated)
    temperature: float    # CPU temp (Celsius)
    thermal_throttled: bool

    # Power (simulated)
    power_mode: str      # "performance" | "balanced" | "powersave"
    battery_level: float # 0.0 to 1.0 (1.0 = full)
```

### Scheduling Event

```python
@dataclass
class SchedulingEvent:
    """A single scheduling decision point."""

    timestamp: float          # When (ms)
    core_id: int             # Which core needs scheduling
    runnable_tasks: list[TaskFeatures]  # Tasks to choose from
    system_state: SystemFeatures

    # Decision (filled by scheduler)
    selected_task: int | None  # Task ID or None
    time_quantum: float       # How long to run (ms)
```

### Performance Metrics (output)

```python
@dataclass
class PerformanceMetrics:
    """Metrics collected during simulation run."""

    # Throughput
    tasks_completed: int
    total_runtime: float  # Simulation time (s)
    throughput: float     # Tasks/second

    # Latency
    mean_latency: float    # Mean task completion time (ms)
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float

    # Fairness
    cpu_time_variance: float    # Variance in CPU time allocated
    starvation_events: int      # Tasks starved >100ms

    # Efficiency
    context_switches: int
    avg_time_quantum: float
    cpu_utilisation: float    # 0.0 to 1.0

    # Responsiveness (for interactive tasks)
    interactive_p99: float    # p99 latency for interactive tasks

    # Energy (simulated)
    total_energy: float      # Joules
    energy_per_task: float
```

## Mapping Real Data to Simulation

### From ps/top Output

```python
def parse_process_data(ps_line: str) -> TaskFeatures:
    """Convert real ps output to TaskFeatures.

    Example ps line:
    "  PID  PPID PRI NI    VSZ    RSS  %CPU %MEM    TIME COMM"
    "  328     1  31  0  55008   1024   0.4  0.3 29:08.34 logd"
    """
    fields = ps_line.split()

    # Parse TIME field (HH:MM:SS.ss or MM:SS.ss)
    time_parts = fields[8].split(':')
    if len(time_parts) == 3:  # HH:MM:SS
        total_cpu_time = (
            int(time_parts[0]) * 3600 +
            int(time_parts[1]) * 60 +
            float(time_parts[2])
        )
    else:  # MM:SS
        total_cpu_time = int(time_parts[0]) * 60 + float(time_parts[1])

    return TaskFeatures(
        task_id=int(fields[0]),
        parent_id=int(fields[1]),
        priority=int(fields[2]),
        nice=int(fields[3]),
        virtual_size=int(fields[4]) * 1024,  # KB to bytes
        resident_size=int(fields[5]) * 1024,
        cpu_usage_hist=[float(fields[6]) / 100.0],  # % to [0,1]
        total_cpu_time=total_cpu_time,
        # ... rest filled with defaults for simulation
    )
```

### Feature Engineering for ML

Additional features we compute from raw data:

```python
def compute_derived_features(task: TaskFeatures, history: list[TaskFeatures]) -> None:
    """Compute derived features from task history."""

    # I/O bound score: more sleep time relative to run time = more I/O bound
    total_time = task.sleep_time + task.run_time
    if total_time > 0:
        task.io_bound_score = task.sleep_time / total_time
    else:
        task.io_bound_score = 0.5

    # Interactive score: frequent short bursts = interactive
    # (requires analysing history of sleep/run patterns)
    if len(history) > 5:
        recent_bursts = [h.run_time for h in history[-5:]]
        avg_burst = sum(recent_bursts) / len(recent_bursts)
        # Interactive tasks have short bursts (<10ms typically)
        task.interactive_score = max(0.0, min(1.0, 1.0 - (avg_burst / 100.0)))
    else:
        task.interactive_score = 0.5
``
```

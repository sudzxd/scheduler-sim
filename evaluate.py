"""Evaluate and compare scheduler performance."""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import json
from dataclasses import dataclass
from pathlib import Path

# Third-party
import torch

# Project/local
from logging_config import get_logger, setup_logging
from scheduler_types import TaskFeatures
from schedulers.cfs import CFSScheduler
from schedulers.stm_scheduler import STMScheduler
from workload_generator import WorkloadProfile, create_mixed_workload

logger = get_logger(__name__)

# =============================================================================
# 2. EVALUATION METRICS
# =============================================================================


@dataclass
class SchedulerMetrics:
    """Performance metrics for a scheduler."""

    scheduler_name: str
    total_decisions: int
    avg_decision_time: float  # Milliseconds
    total_runtime: float  # Milliseconds
    tasks_completed: int
    avg_completion_time: float  # Milliseconds
    p50_latency: float
    p95_latency: float
    p99_latency: float

    def print_summary(self) -> None:
        """Print metrics summary."""
        print(f"\n{self.scheduler_name} Performance:")
        print("-" * 50)
        print(f"  Total Decisions: {self.total_decisions}")
        print(f"  Tasks Completed: {self.tasks_completed}")
        print(f"  Avg Decision Time: {self.avg_decision_time:.4f} ms")
        print(f"  Total Runtime: {self.total_runtime:.2f} ms")
        print(f"  Avg Completion: {self.avg_completion_time:.2f} ms")
        print(f"  Latency P50: {self.p50_latency:.2f} ms")
        print(f"  Latency P95: {self.p95_latency:.2f} ms")
        print(f"  Latency P99: {self.p99_latency:.2f} ms")

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scheduler_name": self.scheduler_name,
            "total_decisions": self.total_decisions,
            "avg_decision_time": self.avg_decision_time,
            "total_runtime": self.total_runtime,
            "tasks_completed": self.tasks_completed,
            "avg_completion_time": self.avg_completion_time,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int | str]) -> "SchedulerMetrics":
        """Create from dictionary."""
        return cls(
            scheduler_name=str(data["scheduler_name"]),
            total_decisions=int(data["total_decisions"]),
            avg_decision_time=float(data["avg_decision_time"]),
            total_runtime=float(data["total_runtime"]),
            tasks_completed=int(data["tasks_completed"]),
            avg_completion_time=float(data["avg_completion_time"]),
            p50_latency=float(data["p50_latency"]),
            p95_latency=float(data["p95_latency"]),
            p99_latency=float(data["p99_latency"]),
        )


# =============================================================================
# 3. SIMULATOR
# =============================================================================


class SchedulerSimulator:
    """Simplified scheduler simulator for evaluation."""

    def __init__(
        self,
        scheduler: CFSScheduler | STMScheduler,
    ) -> None:
        """Initialise simulator.

        Args:
            scheduler: Scheduler to evaluate.
        """
        self.scheduler = scheduler
        self.decision_times: list[float] = []
        self.completion_times: list[float] = []

    def run_simulation(
        self,
        tasks: list[TaskFeatures],
        max_time: float = 50000.0,
    ) -> SchedulerMetrics:
        """Run scheduling simulation.

        Args:
            tasks: List of tasks to schedule.
            max_time: Maximum simulation time (ms).

        Returns:
            Performance metrics.
        """
        import time

        # Reset scheduler
        self.scheduler.reset()

        # Sort tasks by arrival time
        tasks = sorted(tasks, key=lambda t: t.arrival_time)

        current_time = 0.0
        runnable_tasks: list[TaskFeatures] = []
        completed_tasks: list[TaskFeatures] = []
        task_idx = 0

        self.decision_times = []
        self.completion_times = []

        while current_time < max_time:
            # Add newly arrived tasks
            while (
                task_idx < len(tasks)
                and tasks[task_idx].arrival_time <= current_time
            ):
                runnable_tasks.append(tasks[task_idx])
                task_idx += 1

            # Break if no more tasks to schedule
            if not runnable_tasks and task_idx >= len(tasks):
                break

            if not runnable_tasks:
                current_time += 1.0
                continue

            # Make scheduling decision
            start = time.perf_counter()
            event = self.scheduler.schedule(
                runnable_tasks=runnable_tasks,
                current_time=current_time,
                core_id=0,
            )
            decision_time = (time.perf_counter() - start) * 1000  # ms

            self.decision_times.append(decision_time)

            if event.selected_task is None:
                current_time += 1.0
                continue

            # Find and execute selected task
            selected_idx = next(
                i for i, t in enumerate(runnable_tasks)
                if t.task_id == event.selected_task
            )
            selected_task = runnable_tasks[selected_idx]

            # Simulate execution
            exec_time = event.time_quantum
            current_time += exec_time
            selected_task.run_time += exec_time

            # Update scheduler state
            if isinstance(self.scheduler, CFSScheduler):
                self.scheduler.update_vruntime(selected_task, exec_time)

            # Check if task completed
            # total_cpu_time is in seconds, run_time is in ms
            if selected_task.run_time >= (selected_task.total_cpu_time * 1000):
                completion_time = current_time - selected_task.arrival_time
                self.completion_times.append(completion_time)
                completed_tasks.append(selected_task)
                runnable_tasks.pop(selected_idx)
                self.scheduler.task_complete(selected_task, current_time)

        # Calculate metrics
        return self._calculate_metrics(current_time, completed_tasks)

    def _calculate_metrics(
        self,
        total_time: float,
        completed_tasks: list[TaskFeatures],
    ) -> SchedulerMetrics:
        """Calculate performance metrics.

        Args:
            total_time: Total simulation time.
            completed_tasks: List of completed tasks.

        Returns:
            Computed metrics.
        """
        if not self.completion_times:
            self.completion_times = [0.0]

        sorted_latencies = sorted(self.completion_times)
        n = len(sorted_latencies)

        return SchedulerMetrics(
            scheduler_name=str(self.scheduler.name),
            total_decisions=len(self.decision_times),
            avg_decision_time=sum(self.decision_times) / len(self.decision_times)
            if self.decision_times
            else 0.0,
            total_runtime=total_time,
            tasks_completed=len(completed_tasks),
            avg_completion_time=sum(sorted_latencies) / n if n > 0 else 0.0,
            p50_latency=sorted_latencies[n // 2] if n > 0 else 0.0,
            p95_latency=sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            p99_latency=sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        )


# =============================================================================
# 4. COMPARISON RUNNER
# =============================================================================


def compare_schedulers(
    workload_profile: WorkloadProfile,
    num_tasks: int = 50,
    model_path: Path = Path("checkpoints/final.pt"),
) -> tuple[SchedulerMetrics, SchedulerMetrics]:
    """Compare CFS and STM schedulers.

    Args:
        workload_profile: Workload to test on.
        num_tasks: Number of tasks to generate.
        model_path: Path to trained STM model.

    Returns:
        Tuple of (CFS metrics, STM metrics).
    """
    # Generate workload
    logger.info(f"Generating {num_tasks} tasks for {workload_profile.name}")
    tasks = create_mixed_workload(
        num_cpu_bound=num_tasks // 4,
        num_io_bound=num_tasks // 4,
        num_interactive=num_tasks // 4,
        num_batch=num_tasks // 4,
        seed=42,
    )

    # Create schedulers
    cfs = CFSScheduler()

    # Load trained STM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stm = STMScheduler(device=device, greedy=True)
    stm.load_weights(model_path)

    # Run simulations
    logger.info("Running CFS simulation...")
    cfs_sim = SchedulerSimulator(cfs)
    cfs_metrics = cfs_sim.run_simulation(tasks.copy())

    logger.info("Running STM simulation...")
    stm_sim = SchedulerSimulator(stm)
    stm_metrics = stm_sim.run_simulation(tasks.copy())

    return cfs_metrics, stm_metrics


# =============================================================================
# 5. MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Run scheduler comparison."""
    setup_logging(verbose=True)

    print("=" * 70)
    print("Scheduler Performance Comparison: CFS vs STM")
    print("=" * 70)

    # Check for trained model
    model_path = Path("checkpoints/final.pt")
    if not model_path.exists():
        print(f"\nError: Trained model not found at {model_path}")
        print("Please run train.py first to train the STM model.")
        return

    print(f"\nUsing trained model: {model_path}")
    print()

    # Run comparison on mixed workload
    from workload_generator import WorkloadProfile, WorkloadType

    mixed_profile = WorkloadProfile(
        name=WorkloadType.MIXED,
        num_tasks=100,
        nice_range=(-10, 10),
        io_bound_score_range=(0.2, 0.8),
        interactive_score_range=(0.2, 0.8),
        arrival_rate_ms=50.0,
        total_cpu_time_range_s=(1.0, 20.0),
        memory_range_gb=(0.1, 2.0),
    )

    print("Running comparison on 100-task mixed workload...")
    print()

    cfs_metrics, stm_metrics = compare_schedulers(
        workload_profile=mixed_profile,
        num_tasks=100,
        model_path=model_path,
    )

    # Print results
    cfs_metrics.print_summary()
    stm_metrics.print_summary()

    # Print comparison
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)

    # Check if both schedulers completed tasks
    if cfs_metrics.tasks_completed == 0 and stm_metrics.tasks_completed == 0:
        print("\nWarning: Neither scheduler completed any tasks!")
        print("Consider increasing simulation time or reducing task complexity.")
    elif cfs_metrics.tasks_completed == 0:
        print(f"\nSTM completed {stm_metrics.tasks_completed} tasks")
        print("CFS completed 0 tasks (cannot compare)")
    elif stm_metrics.tasks_completed == 0:
        print(f"\nCFS completed {cfs_metrics.tasks_completed} tasks")
        print("STM completed 0 tasks (cannot compare)")
    else:
        # Both completed tasks - compare performance
        speedup = cfs_metrics.avg_completion_time / stm_metrics.avg_completion_time
        print(f"\nCompletion Time Speedup: {speedup:.2f}x")

        if stm_metrics.avg_completion_time < cfs_metrics.avg_completion_time:
            improvement = (
                (cfs_metrics.avg_completion_time - stm_metrics.avg_completion_time)
                / cfs_metrics.avg_completion_time
                * 100
            )
            print(f"STM is {improvement:.1f}% faster than CFS")
        else:
            degradation = (
                (stm_metrics.avg_completion_time - cfs_metrics.avg_completion_time)
                / cfs_metrics.avg_completion_time
                * 100
            )
            print(f"STM is {degradation:.1f}% slower than CFS")

    # Save results for plotting
    results_path = Path("results.json")
    results = {
        "cfs": cfs_metrics.to_dict(),
        "stm": stm_metrics.to_dict(),
    }
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print()


if __name__ == "__main__":
    main()

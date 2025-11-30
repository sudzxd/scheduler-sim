"""Completely Fair Scheduler (CFS) implementation.

This module implements Linux's CFS algorithm, which ensures fairness
through virtual runtime tracking and priority-based weighting.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library

# Project/local
from constants import (
    DEFAULT_MIN_GRANULARITY_MS,
    DEFAULT_TIME_QUANTUM_MS,
    NS_PER_MS,
    SchedulerName,
)
from logging_config import get_logger
from scheduler_types import SchedulingEvent, SystemFeatures, TaskFeatures
from schedulers.base import Scheduler

logger = get_logger(__name__)

# =============================================================================
# 2. CFS SCHEDULER
# =============================================================================


class CFSScheduler(Scheduler):
    """Completely Fair Scheduler implementation.

    CFS assigns tasks based on virtual runtime (vruntime), ensuring
    that all tasks get fair CPU time weighted by their priority.
    """

    def __init__(self) -> None:
        """Initialise CFS scheduler."""
        super().__init__(name=SchedulerName.CFS)

        # Track minimum vruntime across all tasks
        self._min_vruntime: float = 0.0

        # Nice value to weight mapping (Linux CFS formula)
        self._nice_to_weight = self._build_nice_weight_table()

    def schedule(
        self,
        runnable_tasks: list[TaskFeatures],
        current_time: float,
        core_id: int,
    ) -> SchedulingEvent:
        """Select next task to run using CFS algorithm.

        Args:
            runnable_tasks: Tasks ready to run.
            current_time: Current simulation time (milliseconds).
            core_id: CPU core making this decision.

        Returns:
            Scheduling event with selected task.
        """
        if not runnable_tasks:
            # No runnable tasks, return idle event
            system_state = self._create_system_state(current_time, core_id)
            return SchedulingEvent(
                timestamp=current_time,
                core_id=core_id,
                runnable_tasks=[],
                system_state=system_state,
                selected_task=None,
                time_quantum=0.0,
                scheduler_name=self.name,
            )

        # Select task with minimum vruntime
        selected_task = min(runnable_tasks, key=lambda t: t.vruntime)

        # Calculate time quantum based on number of runnable tasks
        time_quantum = self._calculate_time_quantum(len(runnable_tasks))

        system_state = self._create_system_state(current_time, core_id)

        logger.debug(
            f"CFS selected task {selected_task.task_id} "
            f"(vruntime={selected_task.vruntime:.2f}ns, "
            f"priority={selected_task.priority}) "
            f"for {time_quantum:.2f}ms"
        )

        return SchedulingEvent(
            timestamp=current_time,
            core_id=core_id,
            runnable_tasks=runnable_tasks,
            system_state=system_state,
            selected_task=selected_task.task_id,
            time_quantum=time_quantum,
            scheduler_name=self.name,
        )

    def task_wakeup(self, task: TaskFeatures, current_time: float) -> None:
        """Handle task waking up.

        Args:
            task: Task that woke up.
            current_time: Current time (milliseconds).
        """
        # Place task at current min_vruntime to avoid unfair advantage
        if task.vruntime < self._min_vruntime:
            task.vruntime = self._min_vruntime

        logger.debug(
            f"Task {task.task_id} woke up at t={current_time:.2f}ms, "
            f"vruntime={task.vruntime:.2f}ns"
        )

    def task_sleep(self, task: TaskFeatures, current_time: float) -> None:
        """Handle task going to sleep.

        Args:
            task: Task going to sleep.
            current_time: Current time (milliseconds).
        """
        logger.debug(
            f"Task {task.task_id} sleeping at t={current_time:.2f}ms, "
            f"vruntime={task.vruntime:.2f}ns"
        )

    def task_complete(self, task: TaskFeatures, current_time: float) -> None:
        """Handle task completion.

        Args:
            task: Task that completed.
            current_time: Current time (milliseconds).
        """
        logger.debug(
            f"Task {task.task_id} completed at t={current_time:.2f}ms, "
            f"final vruntime={task.vruntime:.2f}ns"
        )

    def update_vruntime(self, task: TaskFeatures, runtime_ms: float) -> None:
        """Update task's virtual runtime after execution.

        Args:
            task: Task that just ran.
            runtime_ms: How long it ran (milliseconds).
        """
        # Get weight for this task's nice value
        weight = self._nice_to_weight[task.nice]

        # Convert runtime to nanoseconds
        runtime_ns = runtime_ms * NS_PER_MS

        # Virtual runtime = physical runtime / weight
        # Higher priority (lower nice) → higher weight → slower vruntime growth
        vruntime_delta = runtime_ns / weight

        task.vruntime += vruntime_delta

        # Update minimum vruntime
        self._min_vruntime = min(self._min_vruntime, task.vruntime)

        logger.debug(
            f"Task {task.task_id} ran {runtime_ms:.2f}ms, "
            f"vruntime increased by {vruntime_delta:.2f}ns "
            f"to {task.vruntime:.2f}ns"
        )

    # =============================================================================
    # PRIVATE HELPERS
    # =============================================================================

    def _calculate_time_quantum(self, num_tasks: int) -> float:
        """Calculate time quantum based on number of runnable tasks.

        Args:
            num_tasks: Number of runnable tasks.

        Returns:
            Time quantum in milliseconds.
        """
        # CFS uses: quantum = sched_period / num_tasks
        # But enforce minimum granularity
        quantum = DEFAULT_TIME_QUANTUM_MS / num_tasks
        return max(quantum, DEFAULT_MIN_GRANULARITY_MS)

    def _build_nice_weight_table(self) -> dict[int, float]:
        """Build nice value to weight mapping.

        Uses Linux CFS formula: weight = 1024 / 1.25^nice

        Returns:
            Dictionary mapping nice value to weight.
        """
        weights = {}
        for nice in range(-20, 20):  # Nice ranges from -20 to 19
            weights[nice] = 1024.0 / (1.25**nice)
        return weights

    def _create_system_state(
        self, current_time: float, core_id: int
    ) -> SystemFeatures:
        """Create system state snapshot.

        Args:
            current_time: Current time (milliseconds).
            core_id: Current core ID.

        Returns:
            System state features.
        """
        # Simplified system state for now
        return SystemFeatures(
            current_time=current_time,
            tick=int(current_time),
            num_cores=1,  # Will be updated by simulator
            core_loads=[],
            runqueue_lengths=[],
        )

"""Base scheduler interface.

This module defines the abstract base class that all scheduler implementations
must inherit from.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
from abc import ABC, abstractmethod

# Project/local
from constants import SchedulerName
from scheduler_types import SchedulingEvent, TaskFeatures

# =============================================================================
# 2. BASE SCHEDULER
# =============================================================================


class Scheduler(ABC):
    """Abstract base class for all scheduler implementations."""

    def __init__(self, name: str = SchedulerName.UNKNOWN) -> None:
        """Initialise scheduler.

        Args:
            name: Scheduler name for identification.
        """
        self.name = name

    @abstractmethod
    def schedule(
        self,
        runnable_tasks: list[TaskFeatures],
        current_time: float,
        core_id: int,
    ) -> SchedulingEvent:
        """Make a scheduling decision.

        Args:
            runnable_tasks: List of tasks that are ready to run.
            current_time: Current simulation time (milliseconds).
            core_id: Which CPU core is making this decision.

        Returns:
            Scheduling event with selected task and time quantum.
        """
        raise NotImplementedError

    @abstractmethod
    def task_wakeup(self, task: TaskFeatures, current_time: float) -> None:
        """Notify scheduler that a task has become runnable.

        Args:
            task: Task that just woke up.
            current_time: Current simulation time (milliseconds).
        """
        raise NotImplementedError

    @abstractmethod
    def task_sleep(self, task: TaskFeatures, current_time: float) -> None:
        """Notify scheduler that a task is going to sleep.

        Args:
            task: Task that is sleeping.
            current_time: Current simulation time (milliseconds).
        """
        raise NotImplementedError

    @abstractmethod
    def task_complete(self, task: TaskFeatures, current_time: float) -> None:
        """Notify scheduler that a task has completed.

        Args:
            task: Task that finished.
            current_time: Current simulation time (milliseconds).
        """
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027
        """Reset scheduler state.

        Subclasses can override this to clear internal state.
        """
        pass  # Default implementation does nothing

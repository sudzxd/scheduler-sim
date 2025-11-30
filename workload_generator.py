"""Synthetic workload generators for scheduler testing.

This module provides utilities to generate various types of synthetic workloads
with different characteristics (CPU-bound, I/O-bound, interactive, etc.) for
evaluating scheduler performance.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

# Project/local
from constants import DEFAULT_PRIORITY
from logging_config import get_logger
from scheduler_types import TaskFeatures, TaskType

logger = get_logger(__name__)

# =============================================================================
# 2. CONSTANTS
# =============================================================================


class WorkloadType(StrEnum):
    """Types of synthetic workloads."""

    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    MIXED = "mixed"


# Task generation parameters
MIN_TASK_ID = 1000
MAX_TASK_ID = 9999
MIN_ARRIVAL_TIME = 0.0
MAX_ARRIVAL_TIME = 1000.0


# =============================================================================
# 3. WORKLOAD PROFILES
# =============================================================================


@dataclass
class WorkloadProfile:
    """Configuration for a specific workload type."""

    name: WorkloadType
    num_tasks: int
    nice_range: tuple[int, int]
    io_bound_score_range: tuple[float, float]
    interactive_score_range: tuple[float, float]
    arrival_rate_ms: float
    total_cpu_time_range_s: tuple[float, float]
    memory_range_gb: tuple[float, float]


# Predefined workload profiles
CPU_BOUND_PROFILE = WorkloadProfile(
    name=WorkloadType.CPU_BOUND,
    num_tasks=20,
    nice_range=(0, 5),
    io_bound_score_range=(0.0, 0.2),
    interactive_score_range=(0.2, 0.5),
    arrival_rate_ms=100.0,
    total_cpu_time_range_s=(5.0, 20.0),
    memory_range_gb=(0.1, 2.0),
)

IO_BOUND_PROFILE = WorkloadProfile(
    name=WorkloadType.IO_BOUND,
    num_tasks=30,
    nice_range=(-5, 0),
    io_bound_score_range=(0.7, 1.0),
    interactive_score_range=(0.3, 0.7),
    arrival_rate_ms=50.0,
    total_cpu_time_range_s=(1.0, 10.0),
    memory_range_gb=(0.05, 1.0),
)

INTERACTIVE_PROFILE = WorkloadProfile(
    name=WorkloadType.INTERACTIVE,
    num_tasks=15,
    nice_range=(-10, -5),
    io_bound_score_range=(0.3, 0.6),
    interactive_score_range=(0.8, 1.0),
    arrival_rate_ms=200.0,
    total_cpu_time_range_s=(0.5, 5.0),
    memory_range_gb=(0.05, 0.5),
)

BATCH_PROFILE = WorkloadProfile(
    name=WorkloadType.BATCH,
    num_tasks=10,
    nice_range=(5, 10),
    io_bound_score_range=(0.1, 0.3),
    interactive_score_range=(0.0, 0.2),
    arrival_rate_ms=500.0,
    total_cpu_time_range_s=(20.0, 100.0),
    memory_range_gb=(1.0, 10.0),
)


# =============================================================================
# 4. WORKLOAD GENERATOR
# =============================================================================


class WorkloadGenerator:
    """Generate synthetic task workloads for scheduler testing."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialise workload generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        logger.info(f"Initialised workload generator (seed={seed})")

    def generate_workload(
        self,
        profile: WorkloadProfile,
        start_time: float = 0.0,
    ) -> list[TaskFeatures]:
        """Generate a workload based on the given profile.

        Args:
            profile: Workload configuration.
            start_time: Starting time for task arrivals (milliseconds).

        Returns:
            List of tasks with synthetic characteristics.
        """
        tasks: list[TaskFeatures] = []
        current_time = start_time

        for _ in range(profile.num_tasks):
            task_id = random.randint(MIN_TASK_ID, MAX_TASK_ID)
            nice = random.randint(*profile.nice_range)
            priority = DEFAULT_PRIORITY - nice

            io_bound_score = random.uniform(*profile.io_bound_score_range)
            interactive_score = random.uniform(*profile.interactive_score_range)
            total_cpu_time = random.uniform(*profile.total_cpu_time_range_s)
            memory_gb = random.uniform(*profile.memory_range_gb)

            # Convert GB to bytes
            memory_bytes = int(memory_gb * 1024 * 1024 * 1024)

            # Map WorkloadType to TaskType
            task_type_map = {
                WorkloadType.CPU_BOUND: TaskType.CPU_BOUND,
                WorkloadType.IO_BOUND: TaskType.IO_BOUND,
                WorkloadType.INTERACTIVE: TaskType.INTERACTIVE,
                WorkloadType.BATCH: TaskType.BATCH,
                WorkloadType.MIXED: TaskType.MIXED,
            }

            task = TaskFeatures(
                task_id=task_id,
                priority=priority,
                nice=nice,
                arrival_time=current_time,
                total_cpu_time=total_cpu_time,
                io_bound_score=io_bound_score,
                interactive_score=interactive_score,
                virtual_size=memory_bytes,
                resident_size=int(memory_bytes * 0.7),  # ~70% resident
                vruntime=0.0,
                cpu_usage_hist=[0.0] * 8,
                time_since_scheduled=0.0,
                sleep_time=0.0,
                run_time=0.0,
                num_threads=1,
                last_core=0,
                task_type=task_type_map[profile.name],
            )

            tasks.append(task)
            current_time += profile.arrival_rate_ms

        logger.info(f"Generated {len(tasks)} tasks for {profile.name} workload")
        return tasks

    def generate_mixed_workload(
        self,
        num_cpu_bound: int = 10,
        num_io_bound: int = 15,
        num_interactive: int = 8,
        num_batch: int = 5,
        start_time: float = 0.0,
    ) -> list[TaskFeatures]:
        """Generate a mixed workload with different task types.

        Args:
            num_cpu_bound: Number of CPU-bound tasks.
            num_io_bound: Number of I/O-bound tasks.
            num_interactive: Number of interactive tasks.
            num_batch: Number of batch tasks.
            start_time: Starting time (milliseconds).

        Returns:
            Combined list of tasks from all types.
        """
        tasks: list[TaskFeatures] = []

        # Generate each workload type
        if num_cpu_bound > 0:
            profile = replace(CPU_BOUND_PROFILE, num_tasks=num_cpu_bound)
            tasks.extend(self.generate_workload(profile, start_time))

        if num_io_bound > 0:
            profile = replace(IO_BOUND_PROFILE, num_tasks=num_io_bound)
            tasks.extend(self.generate_workload(profile, start_time))

        if num_interactive > 0:
            profile = replace(INTERACTIVE_PROFILE, num_tasks=num_interactive)
            tasks.extend(self.generate_workload(profile, start_time))

        if num_batch > 0:
            profile = replace(BATCH_PROFILE, num_tasks=num_batch)
            tasks.extend(self.generate_workload(profile, start_time))

        # Shuffle to interleave different task types
        random.shuffle(tasks)

        logger.info(
            f"Generated mixed workload: {num_cpu_bound} CPU-bound, "
            f"{num_io_bound} I/O-bound, {num_interactive} interactive, "
            f"{num_batch} batch (total: {len(tasks)} tasks)"
        )
        return tasks

    def generate_custom_workload(
        self,
        num_tasks: int,
        task_generator: Callable[[int], TaskFeatures],
    ) -> list[TaskFeatures]:
        """Generate a custom workload using a user-provided generator function.

        Args:
            num_tasks: Number of tasks to generate.
            task_generator: Function that takes task index and returns TaskFeatures.

        Returns:
            List of generated tasks.
        """
        tasks = [task_generator(i) for i in range(num_tasks)]
        logger.info(f"Generated {len(tasks)} custom tasks")
        return tasks


# =============================================================================
# 5. CONVENIENCE FUNCTIONS
# =============================================================================


def create_cpu_bound_workload(
    num_tasks: int = 20,
    seed: int | None = None,
) -> list[TaskFeatures]:
    """Create a CPU-bound workload.

    Args:
        num_tasks: Number of tasks.
        seed: Random seed.

    Returns:
        List of CPU-bound tasks.
    """
    generator = WorkloadGenerator(seed=seed)
    profile = replace(CPU_BOUND_PROFILE, num_tasks=num_tasks)
    return generator.generate_workload(profile)


def create_io_bound_workload(
    num_tasks: int = 30,
    seed: int | None = None,
) -> list[TaskFeatures]:
    """Create an I/O-bound workload.

    Args:
        num_tasks: Number of tasks.
        seed: Random seed.

    Returns:
        List of I/O-bound tasks.
    """
    generator = WorkloadGenerator(seed=seed)
    profile = replace(IO_BOUND_PROFILE, num_tasks=num_tasks)
    return generator.generate_workload(profile)


def create_interactive_workload(
    num_tasks: int = 15,
    seed: int | None = None,
) -> list[TaskFeatures]:
    """Create an interactive workload.

    Args:
        num_tasks: Number of tasks.
        seed: Random seed.

    Returns:
        List of interactive tasks.
    """
    generator = WorkloadGenerator(seed=seed)
    profile = replace(INTERACTIVE_PROFILE, num_tasks=num_tasks)
    return generator.generate_workload(profile)


def create_batch_workload(
    num_tasks: int = 10,
    seed: int | None = None,
) -> list[TaskFeatures]:
    """Create a batch workload.

    Args:
        num_tasks: Number of tasks.
        seed: Random seed.

    Returns:
        List of batch tasks.
    """
    generator = WorkloadGenerator(seed=seed)
    profile = replace(BATCH_PROFILE, num_tasks=num_tasks)
    return generator.generate_workload(profile)


def create_mixed_workload(
    num_cpu_bound: int = 10,
    num_io_bound: int = 15,
    num_interactive: int = 8,
    num_batch: int = 5,
    seed: int | None = None,
) -> list[TaskFeatures]:
    """Create a mixed workload with different task types.

    Args:
        num_cpu_bound: Number of CPU-bound tasks.
        num_io_bound: Number of I/O-bound tasks.
        num_interactive: Number of interactive tasks.
        num_batch: Number of batch tasks.
        seed: Random seed.

    Returns:
        Combined list of tasks.
    """
    generator = WorkloadGenerator(seed=seed)
    return generator.generate_mixed_workload(
        num_cpu_bound=num_cpu_bound,
        num_io_bound=num_io_bound,
        num_interactive=num_interactive,
        num_batch=num_batch,
    )

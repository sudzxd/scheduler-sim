"""Example: Generate synthetic workloads for scheduler testing."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workload_generator import (
    create_batch_workload,
    create_cpu_bound_workload,
    create_interactive_workload,
    create_io_bound_workload,
    create_mixed_workload,
)

# Test each workload type
print("Testing workload generators...")
print()

cpu_tasks = create_cpu_bound_workload(num_tasks=5, seed=42)
print(f"CPU-bound: {len(cpu_tasks)} tasks")
print(f"  Sample: task_id={cpu_tasks[0].task_id}, nice={cpu_tasks[0].nice}")
print(f"  io_bound_score={cpu_tasks[0].io_bound_score:.2f}")
print()

io_tasks = create_io_bound_workload(num_tasks=5, seed=42)
print(f"I/O-bound: {len(io_tasks)} tasks")
print(f"  Sample: task_id={io_tasks[0].task_id}, nice={io_tasks[0].nice}")
print(f"  io_bound_score={io_tasks[0].io_bound_score:.2f}")
print()

interactive_tasks = create_interactive_workload(num_tasks=5, seed=42)
print(f"Interactive: {len(interactive_tasks)} tasks")
print(
    f"  Sample: task_id={interactive_tasks[0].task_id}, "
    f"nice={interactive_tasks[0].nice}"
)
print(f"  interactive_score={interactive_tasks[0].interactive_score:.2f}")
print()

batch_tasks = create_batch_workload(num_tasks=5, seed=42)
print(f"Batch: {len(batch_tasks)} tasks")
print(f"  Sample: task_id={batch_tasks[0].task_id}, nice={batch_tasks[0].nice}")
print(f"  total_cpu_time={batch_tasks[0].total_cpu_time:.2f}s")
print()

mixed_tasks = create_mixed_workload(
    num_cpu_bound=3,
    num_io_bound=3,
    num_interactive=2,
    num_batch=2,
    seed=42,
)
print(f"Mixed: {len(mixed_tasks)} tasks")
print(f"  Task types: {[t.task_type.value for t in mixed_tasks[:5]]}")
print()

print("All workload generators working correctly!")

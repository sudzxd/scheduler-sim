"""Example: Collect real scheduling data from the system."""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project/local
from data_collector import collect_scheduling_trace, save_trace_to_file

# =============================================================================
# 2. MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Run data collection test."""
    print("Starting data collection test...")
    print("Collecting 5 seconds of scheduling data...")

    # Collect trace
    trace = collect_scheduling_trace(duration_s=5.0, interval_ms=500.0)

    print(f"\nCollected {len(trace)} snapshots")

    # Print first snapshot summary
    if trace:
        first_snapshot = trace[0]
        print("\nFirst snapshot:")
        print(f"  Timestamp: {first_snapshot.timestamp:.2f}")
        print(f"  Num cores: {first_snapshot.num_cores}")
        print(f"  Load avg: {first_snapshot.load_avg:.2f}")
        print(f"  Num tasks: {len(first_snapshot.tasks)}")

        if first_snapshot.tasks:
            print("\nSample task (first one):")
            task = first_snapshot.tasks[0]
            print(f"  PID: {task.task_id}")
            print(f"  Priority: {task.priority}")
            print(f"  Nice: {task.nice}")
            print(f"  Virtual size: {task.virtual_size / 1024 / 1024:.1f} MB")
            print(f"  Resident size: {task.resident_size / 1024 / 1024:.1f} MB")
            print(f"  CPU usage hist: {[f'{x:.2f}' for x in task.cpu_usage_hist]}")
            print(f"  Total CPU time: {task.total_cpu_time:.2f}s")
            print(f"  Task type: {task.task_type.value}")

        # Show feature vector
        print("\nFeature vector (first task):")
        vec = first_snapshot.tasks[0].to_feature_vector()
        print(f"  Dimension: {len(vec)}")
        print(f"  Values: {[f'{v:.3f}' for v in vec[:10]]}... (showing first 10)")

    # Save to file
    output_path = Path("test_trace.json")
    save_trace_to_file(trace, output_path)
    print(f"\nSaved trace to {output_path}")

    print("\n✓ Data collection test completed successfully!")


if __name__ == "__main__":
    main()

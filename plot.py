"""Generate publication-quality plots for research paper."""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import json
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Project/local
from evaluate import SchedulerMetrics
from logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# =============================================================================
# 2. PLOT CONFIGURATION
# =============================================================================

# Set publication style
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Colour scheme
COLOURS = {
    "cfs": "#2E86AB",  # Blue
    "stm": "#A23B72",  # Purple/Pink
    "highlight": "#F18F01",  # Orange
}

# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================


def plot_performance_comparison(
    cfs_metrics: SchedulerMetrics,
    stm_metrics: SchedulerMetrics,
    output_path: Path,
) -> None:
    """Create performance comparison bar chart.

    Args:
        cfs_metrics: CFS performance metrics.
        stm_metrics: STM performance metrics.
        output_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    metrics_data = [
        {
            "title": "Average Completion Time",
            "ylabel": "Time (seconds)",
            "cfs": cfs_metrics.avg_completion_time / 1000,
            "stm": stm_metrics.avg_completion_time / 1000,
            "lower_is_better": True,
        },
        {
            "title": "Tasks Completed",
            "ylabel": "Count",
            "cfs": cfs_metrics.tasks_completed,
            "stm": stm_metrics.tasks_completed,
            "lower_is_better": False,
        },
        {
            "title": "Median Latency (P50)",
            "ylabel": "Time (seconds)",
            "cfs": cfs_metrics.p50_latency / 1000,
            "stm": stm_metrics.p50_latency / 1000,
            "lower_is_better": True,
        },
    ]

    for ax, data in zip(axes, metrics_data, strict=True):
        x = np.arange(2)
        values = [data["cfs"], data["stm"]]
        bars = ax.bar(
            x,
            values,
            color=[COLOURS["cfs"], COLOURS["stm"]],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Calculate improvement
        if data["lower_is_better"]:
            improvement = ((data["cfs"] - data["stm"]) / data["cfs"]) * 100
            better = "↓" if improvement > 0 else "↑"
        else:
            improvement = ((data["stm"] - data["cfs"]) / data["cfs"]) * 100
            better = "↑" if improvement > 0 else "↓"

        ax.set_title(
            f"{data['title']}\n{better} {abs(improvement):.1f}% improvement",
            fontweight="bold",
        )
        ax.set_ylabel(data["ylabel"])
        ax.set_xticks(x)
        ax.set_xticklabels(["CFS", "STM"], fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved performance comparison to {output_path}")
    plt.close()


def plot_latency_distribution(
    cfs_metrics: SchedulerMetrics,
    stm_metrics: SchedulerMetrics,
    output_path: Path,
) -> None:
    """Create latency distribution plot.

    Args:
        cfs_metrics: CFS performance metrics.
        stm_metrics: STM performance metrics.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Data for plotting
    schedulers = ["CFS", "STM"]
    p50 = [cfs_metrics.p50_latency / 1000, stm_metrics.p50_latency / 1000]
    p95 = [cfs_metrics.p95_latency / 1000, stm_metrics.p95_latency / 1000]
    p99 = [cfs_metrics.p99_latency / 1000, stm_metrics.p99_latency / 1000]

    x = np.arange(len(schedulers))
    width = 0.25

    # Create grouped bars
    bars1 = ax.bar(
        x - width,
        p50,
        width,
        label="P50",
        color=COLOURS["cfs"],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        p95,
        width,
        label="P95",
        color=COLOURS["stm"],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        p99,
        width,
        label="P99",
        color=COLOURS["highlight"],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Scheduler", fontweight="bold")
    ax.set_ylabel("Latency (seconds)", fontweight="bold")
    ax.set_title(
        "Task Completion Latency Distribution", fontweight="bold", fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved latency distribution to {output_path}")
    plt.close()


def plot_decision_overhead(
    cfs_metrics: SchedulerMetrics,
    stm_metrics: SchedulerMetrics,
    output_path: Path,
) -> None:
    """Create scheduler decision overhead comparison.

    Args:
        cfs_metrics: CFS performance metrics.
        stm_metrics: STM performance metrics.
        output_path: Path to save figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Decision time comparison
    schedulers = ["CFS", "STM"]
    decision_times = [
        cfs_metrics.avg_decision_time,
        stm_metrics.avg_decision_time,
    ]
    colours = [COLOURS["cfs"], COLOURS["stm"]]

    bars = ax1.bar(
        schedulers,
        decision_times,
        color=colours,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax1.set_ylabel("Time (milliseconds)", fontweight="bold")
    ax1.set_title("Average Decision Time", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Total decisions comparison
    total_decisions = [
        cfs_metrics.total_decisions,
        stm_metrics.total_decisions,
    ]

    bars = ax2.bar(
        schedulers,
        total_decisions,
        color=colours,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax2.set_ylabel("Count", fontweight="bold")
    ax2.set_title("Total Scheduling Decisions", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved decision overhead plot to {output_path}")
    plt.close()


def plot_training_curve(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Plot training loss curve from checkpoint.

    Args:
        checkpoint_path: Path to training checkpoint.
        output_path: Path to save figure.
    """
    import torch

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    train_losses = checkpoint.get("train_losses", [])

    if not train_losses:
        logger.warning("No training losses found in checkpoint")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(
        epochs,
        train_losses,
        color=COLOURS["stm"],
        linewidth=2,
        marker="o",
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax.set_title("STM Training Progress", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    # Add final loss annotation
    final_loss = train_losses[-1]
    ax.annotate(
        f"Final: {final_loss:.4f}",
        xy=(len(train_losses), final_loss),
        xytext=(10, -10),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "black"},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0"},
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved training curve to {output_path}")
    plt.close()


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Generate all plots for research paper."""
    setup_logging(verbose=True)

    print("=" * 70)
    print("Generating Publication-Quality Plots")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Load evaluation results
    results_path = Path("results.json")
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        print("Please run evaluate.py first to generate results.")
        return

    print(f"Loading results from {results_path}...")
    with results_path.open() as f:
        results = json.load(f)

    cfs_metrics = SchedulerMetrics.from_dict(results["cfs"])
    stm_metrics = SchedulerMetrics.from_dict(results["stm"])
    print("Results loaded successfully!")
    print()

    # Generate plots
    print("Generating plots...")
    print()

    plot_performance_comparison(
        cfs_metrics,
        stm_metrics,
        output_dir / "performance_comparison.png",
    )

    plot_latency_distribution(
        cfs_metrics,
        stm_metrics,
        output_dir / "latency_distribution.png",
    )

    plot_decision_overhead(
        cfs_metrics,
        stm_metrics,
        output_dir / "decision_overhead.png",
    )

    # Plot training curve if checkpoint exists
    checkpoint_path = Path("checkpoints/final.pt")
    if checkpoint_path.exists():
        plot_training_curve(
            checkpoint_path,
            output_dir / "training_curve.png",
        )
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Skipping training curve plot.")

    print()
    print("=" * 70)
    print("All Plots Generated!")
    print("=" * 70)
    print()
    print(f"Figures saved to: {output_dir.absolute()}/")
    print()
    print("Generated files:")
    for plot_file in output_dir.glob("*.png"):
        print(f"  - {plot_file.name}")
    print()


if __name__ == "__main__":
    main()

"""Train the STM scheduler model."""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
from pathlib import Path

# Third-party
import torch

# Project/local
from logging_config import setup_logging
from trainer import STMTrainer, TrainingConfig
from workload_generator import (
    BATCH_PROFILE,
    CPU_BOUND_PROFILE,
    INTERACTIVE_PROFILE,
    IO_BOUND_PROFILE,
)

# =============================================================================
# 2. MAIN TRAINING SCRIPT
# =============================================================================


def main() -> None:
    """Train the STM model on synthetic workloads."""
    # Setup logging
    setup_logging(verbose=True)

    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("STM Scheduler Training")
    print("=" * 70)
    print()
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Configure training
    config = TrainingConfig(
        # Model architecture
        embed_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        # Training hyperparameters
        learning_rate=1e-3,
        num_epochs=100,
        episodes_per_epoch=100,
        max_tasks_per_episode=20,
        # Optimization
        weight_decay=1e-4,
        grad_clip=1.0,
        # Device
        device=device,
        # Checkpointing
        save_every=10,
        checkpoint_dir=Path("checkpoints"),
    )

    print("Training Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Episodes per Epoch: {config.episodes_per_epoch}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Model: {config.embed_dim}D, {config.num_heads} heads")
    print()

    # Create trainer
    trainer = STMTrainer(config)

    # Define workload profiles to train on
    profiles = [
        CPU_BOUND_PROFILE,
        IO_BOUND_PROFILE,
        INTERACTIVE_PROFILE,
        BATCH_PROFILE,
    ]

    print(f"Training on {len(profiles)} workload types:")
    for profile in profiles:
        print(f"  - {profile.name.value}")
    print()

    # Start training
    print("Starting training...")
    print("-" * 70)

    try:
        trainer.train(profiles)

        print()
        print("=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print()
        print(f"Final model: {config.checkpoint_dir / 'final.pt'}")
        print(f"All checkpoints: {config.checkpoint_dir}")
        print()

    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
        print(f"Checkpoints saved in: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()

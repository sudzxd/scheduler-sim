"""Training infrastructure for STM scheduler model.

This module implements the training loop and loss computation for learning
scheduling policies using reinforcement learning and imitation learning.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
from dataclasses import dataclass
from pathlib import Path

# Third-party
import torch
import torch.nn as nn
import torch.optim as optim

# Project/local
from constants import (
    CONTEXT_FEATURE_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_EPISODES_PER_EPOCH,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_TASKS_PER_EPISODE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    TASK_FEATURE_DIM,
)
from logging_config import get_logger
from models.stm import SchedulingTransformer
from scheduler_types import TaskFeatures
from schedulers.cfs import CFSScheduler
from schedulers.stm_scheduler import STMScheduler
from workload_generator import WorkloadGenerator, WorkloadProfile

logger = get_logger(__name__)

# =============================================================================
# 2. TRAINING CONFIGURATION
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training the STM model."""

    # Model hyperparameters
    task_feature_dim: int = TASK_FEATURE_DIM
    context_feature_dim: int = CONTEXT_FEATURE_DIM
    embed_dim: int = DEFAULT_EMBED_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT

    # Training hyperparameters
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS

    # Data generation
    episodes_per_epoch: int = DEFAULT_EPISODES_PER_EPOCH
    max_tasks_per_episode: int = DEFAULT_MAX_TASKS_PER_EPISODE

    # Optimisation
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP

    # Checkpointing
    save_every: int = DEFAULT_SAVE_EVERY
    checkpoint_dir: Path = Path(DEFAULT_CHECKPOINT_DIR)

    # Device
    device: str = "cpu"


# =============================================================================
# 3. TRAINER
# =============================================================================


class STMTrainer:
    """Train the STM model using imitation learning from CFS."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialise trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        self.model = SchedulingTransformer(
            task_feature_dim=config.task_feature_dim,
            context_feature_dim=config.context_feature_dim,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)

        # Create optimiser
        self.optimiser = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser,
            T_max=config.num_epochs,
        )

        # Loss function (cross-entropy for task selection)
        self.criterion = nn.CrossEntropyLoss()

        # Expert scheduler (CFS) for imitation learning
        self.expert = CFSScheduler()

        # Workload generator
        self.workload_gen = WorkloadGenerator(seed=42)

        # Metrics
        self.train_losses: list[float] = []
        self.val_accuracies: list[float] = []

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialised STM trainer with {self.model.count_parameters():,} "
            f"parameters on {self.device}"
        )

    def generate_episode(
        self,
        profile: WorkloadProfile,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        """Generate a single training episode.

        Args:
            profile: Workload profile to use.

        Returns:
            Tuple of (task_features, context_features, expert_labels).
        """
        # Generate tasks
        tasks = self.workload_gen.generate_workload(profile)

        # Sort by arrival time
        tasks.sort(key=lambda t: t.arrival_time)

        task_features_list: list[torch.Tensor] = []
        context_features_list: list[torch.Tensor] = []
        expert_labels: list[int] = []

        current_time = 0.0
        runnable_tasks: list[TaskFeatures] = []

        # Simulate scheduling decisions
        for task in tasks:
            # Add newly arrived tasks
            if task.arrival_time <= current_time:
                runnable_tasks.append(task)

            if not runnable_tasks:
                current_time += 1.0
                continue

            # Get expert decision (CFS)
            event = self.expert.schedule(
                runnable_tasks=runnable_tasks,
                current_time=current_time,
                core_id=0,
            )

            if event.selected_task is None:
                continue

            # Find index of selected task
            selected_idx = next(
                i
                for i, t in enumerate(runnable_tasks)
                if t.task_id == event.selected_task
            )

            # Convert to tensors
            task_vecs = [t.to_feature_vector() for t in runnable_tasks]
            task_tensor = torch.tensor(
                task_vecs, dtype=torch.float32, device=self.device
            )

            # Simplified context
            context_vec = [
                current_time / 1000.0,  # Normalise time
                float(len(runnable_tasks)) / 100.0,  # Normalise count
            ] + [0.0] * 10  # Padding to match context_feature_dim

            context_tensor = torch.tensor(
                context_vec, dtype=torch.float32, device=self.device
            )

            task_features_list.append(task_tensor)
            context_features_list.append(context_tensor)
            expert_labels.append(selected_idx)

            # Simulate execution
            current_time += event.time_quantum

            # Update task state (simplified)
            selected_task = runnable_tasks[selected_idx]
            selected_task.run_time += event.time_quantum

            # Remove if completed (simplified)
            if selected_task.run_time >= selected_task.total_cpu_time * 1000:
                runnable_tasks.pop(selected_idx)

            # Stop if we have enough samples
            if len(expert_labels) >= self.config.max_tasks_per_episode:
                break

        return task_features_list, context_features_list, expert_labels

    def train_epoch(
        self,
        epoch: int,
        profiles: list[WorkloadProfile],
    ) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.
            profiles: List of workload profiles to sample from.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        epoch_losses: list[float] = []

        for episode_idx in range(self.config.episodes_per_epoch):
            # Sample random profile
            profile = profiles[episode_idx % len(profiles)]

            # Generate episode
            task_feats, ctx_feats, labels = self.generate_episode(profile)

            if not labels:
                continue

            # Process each decision
            for task_feat, ctx_feat, label in zip(
                task_feats, ctx_feats, labels, strict=True
            ):
                # Add batch dimension
                task_feat_batch = task_feat.unsqueeze(0)
                ctx_feat_batch = ctx_feat.unsqueeze(0)

                # Forward pass
                scores = self.model(task_feat_batch, ctx_feat_batch)

                # Compute loss
                target = torch.tensor([label], device=self.device)
                loss = self.criterion(scores, target)

                # Backward pass
                self.optimiser.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

                self.optimiser.step()

                epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        self.train_losses.append(avg_loss)

        logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}: loss={avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, epoch: int, path: Path | None = None) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            path: Path to save checkpoint (default: checkpoint_dir/epoch_N.pt).
        """
        if path is None:
            path = self.config.checkpoint_dir / f"epoch_{epoch + 1}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> int:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Epoch number from checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_accuracies = checkpoint["val_accuracies"]

        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {path} (epoch {epoch + 1})")

        return epoch

    def train(
        self,
        profiles: list[WorkloadProfile],
        resume_from: Path | None = None,
    ) -> None:
        """Train the model.

        Args:
            profiles: List of workload profiles to train on.
            resume_from: Path to checkpoint to resume from (optional).
        """
        start_epoch = 0

        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1

        logger.info(
            f"Starting training from epoch {start_epoch + 1} "
            f"for {self.config.num_epochs} epochs"
        )

        for epoch in range(start_epoch, self.config.num_epochs):
            # Train epoch
            _ = self.train_epoch(epoch, profiles)

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)

        # Save final model
        final_path = self.config.checkpoint_dir / "final.pt"
        self.save_checkpoint(self.config.num_epochs - 1, final_path)

        logger.info("Training complete!")

    def export_scheduler(self, checkpoint_path: Path, output_path: Path) -> None:
        """Export trained model as STMScheduler.

        Args:
            checkpoint_path: Path to model checkpoint.
            output_path: Path to save scheduler weights.
        """
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)

        # Create scheduler with trained model
        scheduler = STMScheduler(
            model=self.model,
            device=str(self.device),
            greedy=True,
        )

        # Save weights
        scheduler.save_weights(output_path)

        logger.info(f"Exported trained scheduler to {output_path}")

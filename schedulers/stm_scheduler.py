"""STM (Scheduling Transformer Model) scheduler implementation.

This module wraps the SchedulingTransformer model and implements the
Scheduler interface for use in simulations.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
from pathlib import Path

# Third-party
import torch

# Project/local
from constants import SchedulerName
from logging_config import get_logger
from models.stm import SchedulingTransformer
from scheduler_types import SchedulingEvent, SystemFeatures, TaskFeatures
from schedulers.base import Scheduler

logger = get_logger(__name__)

# =============================================================================
# 2. STM SCHEDULER
# =============================================================================


class STMScheduler(Scheduler):
    """Scheduling Transformer Model scheduler.

    Uses a learned transformer model to make scheduling decisions
    based on task features and system state.
    """

    def __init__(
        self,
        model: SchedulingTransformer | None = None,
        device: str = "cpu",
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> None:
        """Initialise STM scheduler.

        Args:
            model: Pre-trained model (if None, creates default).
            device: Device to run model on ('cpu' or 'cuda').
            temperature: Temperature for softmax sampling.
            greedy: If True, use greedy selection; else sample.
        """
        super().__init__(name=SchedulerName.STM)

        self.device = torch.device(device)
        self.temperature = temperature
        self.greedy = greedy

        # Create or use provided model
        if model is None:
            # Default model configuration
            # Task features: 22 dimensions (from TaskFeatures.to_feature_vector)
            # Context features: 12 dimensions (from SystemFeatures.to_feature_vector)
            self.model = SchedulingTransformer(
                task_feature_dim=22,
                context_feature_dim=12,
                embed_dim=64,
                num_heads=4,
                num_layers=1,
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()  # Start in evaluation mode

        logger.info(
            f"Initialised STM scheduler with "
            f"{self.model.count_parameters():,} parameters"
        )

    def schedule(
        self,
        runnable_tasks: list[TaskFeatures],
        current_time: float,
        core_id: int,
    ) -> SchedulingEvent:
        """Select next task using transformer model.

        Args:
            runnable_tasks: Tasks ready to run.
            current_time: Current simulation time (milliseconds).
            core_id: CPU core making this decision.

        Returns:
            Scheduling event with selected task.
        """
        if not runnable_tasks:
            # No runnable tasks
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

        # Convert tasks to feature tensors
        task_features = self._tasks_to_tensor(runnable_tasks)  # [1, N, 22]
        context_features = self._context_to_tensor(current_time, core_id)  # [1, 12]

        # Run model inference
        with torch.no_grad():
            selected_idx_tensor, scores = self.model.select_task(
                task_features,
                context_features,
                greedy=self.greedy,
                temperature=self.temperature,
            )

        selected_idx: int = int(selected_idx_tensor.item())
        selected_task = runnable_tasks[selected_idx]

        # Use fixed time quantum for now (can be learned later)
        time_quantum = 6.0  # milliseconds

        system_state = self._create_system_state(current_time, core_id)

        logger.debug(
            f"STM selected task {selected_task.task_id} "
            f"(score={scores[0, selected_idx].item():.3f}) "
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
        logger.debug(f"Task {task.task_id} woke up at t={current_time:.2f}ms")

    def task_sleep(self, task: TaskFeatures, current_time: float) -> None:
        """Handle task going to sleep.

        Args:
            task: Task going to sleep.
            current_time: Current time (milliseconds).
        """
        logger.debug(f"Task {task.task_id} sleeping at t={current_time:.2f}ms")

    def task_complete(self, task: TaskFeatures, current_time: float) -> None:
        """Handle task completion.

        Args:
            task: Task that completed.
            current_time: Current time (milliseconds).
        """
        logger.debug(f"Task {task.task_id} completed at t={current_time:.2f}ms")

    def load_weights(self, path: Path) -> None:
        """Load model weights from file.

        Args:
            path: Path to weights file.
        """
        # Load checkpoint (contains model_state_dict and training info)
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )

        # Extract model weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logger.info(f"Loaded STM weights from {path}")

    def save_weights(self, path: Path) -> None:
        """Save model weights to file.

        Args:
            path: Path to save weights.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved STM weights to {path}")

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    # =============================================================================
    # PRIVATE HELPERS
    # =============================================================================

    def _tasks_to_tensor(self, tasks: list[TaskFeatures]) -> torch.Tensor:
        """Convert task features to tensor.

        Args:
            tasks: List of task features.

        Returns:
            Tensor of shape [1, n_tasks, feature_dim].
        """
        feature_vectors = [task.to_feature_vector() for task in tasks]
        tensor = torch.tensor(feature_vectors, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)  # Add batch dimension

    def _context_to_tensor(self, current_time: float, core_id: int) -> torch.Tensor:
        """Convert system context to tensor.

        Args:
            current_time: Current time (milliseconds).
            core_id: Current core ID.

        Returns:
            Tensor of shape [1, context_dim].
        """
        # Create simplified system state
        system_state = SystemFeatures(
            current_time=current_time,
            tick=int(current_time),
            num_cores=1,
        )
        feature_vector = system_state.to_feature_vector()
        tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)  # Add batch dimension

    def _create_system_state(self, current_time: float, core_id: int) -> SystemFeatures:
        """Create system state snapshot.

        Args:
            current_time: Current time (milliseconds).
            core_id: Current core ID.

        Returns:
            System state features.
        """
        return SystemFeatures(
            current_time=current_time,
            tick=int(current_time),
            num_cores=1,
        )

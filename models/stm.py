"""Scheduling Transformer Model (STM).

This module implements the transformer-based scheduler as described in our
research architecture. It uses multi-head self-attention to reason over
sets of runnable tasks and make scheduling decisions.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library

# Third-party
import torch
import torch.nn as nn

# Project/local
from logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# 2. COMPONENTS
# =============================================================================


class TaskEncoder(nn.Module):
    """Encode task features into embedding space."""

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 128) -> None:
        """Initialise task encoder.

        Args:
            input_dim: Dimension of input task features.
            embed_dim: Output embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode task features.

        Args:
            x: Task features [batch, n_tasks, input_dim].

        Returns:
            Task embeddings [batch, n_tasks, embed_dim].
        """
        return self.net(x)


class ContextEncoder(nn.Module):
    """Encode global system context into embedding space."""

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 128) -> None:
        """Initialise context encoder.

        Args:
            input_dim: Dimension of input context features.
            embed_dim: Output embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode context features.

        Args:
            x: Context features [batch, input_dim].

        Returns:
            Context embedding [batch, embed_dim].
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and FFN."""

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int | None = None
    ) -> None:
        """Initialise transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward network dimension (default: 4 * embed_dim).
        """
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * embed_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim].
            mask: Attention mask [batch, seq_len] (optional).

        Returns:
            Output embeddings [batch, seq_len, embed_dim].
        """
        # Multi-head self-attention with residual
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward network with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PolicyHead(nn.Module):
    """Policy head that outputs priority scores for each task."""

    def __init__(self, embed_dim: int, hidden_dim: int = 32) -> None:
        """Initialise policy head.

        Args:
            embed_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Scalar score per task
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute priority scores.

        Args:
            x: Task embeddings [batch, n_tasks, embed_dim].

        Returns:
            Priority scores [batch, n_tasks].
        """
        return self.net(x).squeeze(-1)


# =============================================================================
# 3. MAIN MODEL
# =============================================================================


class SchedulingTransformer(nn.Module):
    """Transformer-based scheduling model.

    Uses multi-head self-attention to reason over sets of runnable tasks
    and predict which task should be scheduled next.
    """

    def __init__(
        self,
        task_feature_dim: int,
        context_feature_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialise scheduling transformer.

        Args:
            task_feature_dim: Dimension of task feature vectors.
            context_feature_dim: Dimension of global context features.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.task_feature_dim = task_feature_dim
        self.context_feature_dim = context_feature_dim
        self.embed_dim = embed_dim

        # Encoders
        self.task_encoder = TaskEncoder(task_feature_dim, embed_dim)
        self.context_encoder = ContextEncoder(context_feature_dim, embed_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        # Policy head
        self.policy_head = PolicyHead(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        logger.info(
            f"Initialised STM: embed_dim={embed_dim}, heads={num_heads}, "
            f"layers={num_layers}, params={self.count_parameters():,}"
        )

    def forward(
        self,
        task_features: torch.Tensor,
        context_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            task_features: Task feature vectors [batch, n_tasks, task_dim].
            context_features: Global context features [batch, context_dim].
            mask: Task mask [batch, n_tasks] where True = valid task.

        Returns:
            Priority scores [batch, n_tasks].
        """
        batch_size, n_tasks, _ = task_features.shape

        # Encode tasks and context
        task_embeddings = self.task_encoder(task_features)  # [B, N, D]
        context_embedding = self.context_encoder(context_features)  # [B, D]

        # Add context as special token
        context_embedding = context_embedding.unsqueeze(1)  # [B, 1, D]
        embeddings = torch.cat(
            [task_embeddings, context_embedding], dim=1
        )  # [B, N+1, D]

        # Create mask for context token (always attend)
        if mask is not None:
            # Invert mask: True = attend, False = ignore
            attn_mask = ~mask  # [B, N]
            context_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=mask.device
            )
            attn_mask = torch.cat([attn_mask, context_mask], dim=1)  # [B, N+1]
        else:
            attn_mask = None

        # Pass through transformer layers
        for layer in self.transformer_layers:
            embeddings = layer(embeddings, attn_mask)
            embeddings = self.dropout(embeddings)

        # Extract task embeddings (drop context token)
        task_embeddings = embeddings[:, :-1, :]  # [B, N, D]

        # Compute priority scores
        scores = self.policy_head(task_embeddings)  # [B, N]

        # Mask out invalid tasks (set to -inf so softmax ignores them)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        return scores

    def select_task(
        self,
        task_features: torch.Tensor,
        context_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select which task to schedule.

        Args:
            task_features: Task features [batch, n_tasks, task_dim].
            context_features: Context features [batch, context_dim].
            mask: Task validity mask [batch, n_tasks].
            greedy: If True, select argmax; else sample from softmax.
            temperature: Temperature for softmax sampling.

        Returns:
            Tuple of (selected_indices, scores):
                - selected_indices: [batch] task indices
                - scores: [batch, n_tasks] all scores
        """
        scores = self.forward(task_features, context_features, mask)  # [B, N]

        if greedy:
            # Select task with highest score
            selected = torch.argmax(scores, dim=-1)  # [B]
        else:
            # Sample from softmax distribution
            probs = torch.softmax(scores / temperature, dim=-1)  # [B, N]
            selected = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

        return selected, scores

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(
        self,
        task_features: torch.Tensor,
        context_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Get attention weights for visualisation.

        Args:
            task_features: Task features [batch, n_tasks, task_dim].
            context_features: Context features [batch, context_dim].
            mask: Task mask [batch, n_tasks].

        Returns:
            List of attention weight tensors, one per layer.
        """
        # This requires modifying transformer blocks to return attention weights
        # For now, return empty list
        logger.warning("Attention weight extraction not implemented yet")
        return []

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_trace: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        n_classes: int,
        num_heads: int = 1,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Dimension of model basis vectors (k).
            d_trace: Dimension of flattened trace embedding (C * (k+1)).
            hidden_dim: Dimension for attention mechanism.
            n_classes: Number of output classes.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.d_trace = d_trace
        self.hidden_dim = hidden_dim

        # Projections
        self.model_proj = nn.Linear(d_model, hidden_dim)
        self.trace_proj = nn.Linear(d_trace, hidden_dim)

        # Cross Attention
        # batch_first=True means input is (batch, seq, feature)
        self.attn_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self, model_basis: torch.Tensor, trace_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model_basis: [Batch, T, d_model]
            trace_embedding: [Batch, d_trace]

        Returns:
            logits: [Batch, n_classes]
        """
        # 1. Project inputs
        # model_basis: [B, T, d_model] -> [B, T, hidden_dim]
        k_v = self.model_proj(model_basis)

        # trace_embedding: [B, d_trace] -> [B, hidden_dim] -> [B, 1, hidden_dim]
        q = self.trace_proj(trace_embedding).unsqueeze(1)

        # 2. Cross Attention
        # Query: Trace, Key/Value: Model
        # attn_output: [B, 1, hidden_dim]
        for attn_block, mlp_block in zip(self.attn_blocks, self.mlp_blocks):
            attn_output, _ = attn_block(query=q, key=k_v, value=k_v)
            attn_output = self.attn_norm(attn_output)
            attn_output = self.attn_dropout(attn_output)
            attn_output = mlp_block(attn_output) + attn_output
            q = attn_output + q  # skip connection

        # 3. MLP Head
        # Squeeze sequence dim: [B, 1, hidden_dim] -> [B, hidden_dim]
        mlp_input = q.squeeze(1)
        logits = self.mlp_head(mlp_input)

        return logits

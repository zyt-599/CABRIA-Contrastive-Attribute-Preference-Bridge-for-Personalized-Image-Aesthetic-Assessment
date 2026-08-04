from __future__ import annotations

import torch
from torch import nn


class ScoreHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


class TokenScoreHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_weights = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.score_head = ScoreHead(embed_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = torch.softmax(self.token_weights(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        return {
            "score": self.score_head(pooled),
            "pooled_tokens": pooled,
            "token_weights": weights,
        }


class TokenResidualHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_weights = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.token_residuals = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = torch.softmax(self.token_weights(tokens).squeeze(-1), dim=1)
        residuals = self.token_residuals(tokens).squeeze(-1)
        score = torch.sum(weights * residuals, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * tokens, dim=1)
        return {
            "score": score,
            "token_weights": weights,
            "token_residuals": residuals,
            "pooled_tokens": pooled,
        }


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, projection_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.proj(tokens), dim=-1)

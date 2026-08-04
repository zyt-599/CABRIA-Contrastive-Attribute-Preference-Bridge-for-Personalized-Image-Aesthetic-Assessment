from __future__ import annotations

import torch
from torch import nn


class AttributeTokenExtractor(nn.Module):
    def __init__(self, embed_dim: int, num_attribute_tokens: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attribute_queries = nn.Parameter(torch.randn(num_attribute_tokens, embed_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = visual_tokens.size(0)
        queries = self.attribute_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attention_weights = self.cross_attention(queries, visual_tokens, visual_tokens, need_weights=True)
        tokens = self.output_norm(attended + self.ffn(attended))
        return {"attribute_tokens": tokens, "attention_weights": attention_weights}


def attribute_diversity_loss(attribute_tokens: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(attribute_tokens, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    identity = torch.eye(similarity.size(-1), device=similarity.device).unsqueeze(0)
    return ((similarity - identity) ** 2).mean()

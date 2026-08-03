from __future__ import annotations

import math

import torch
from torch import nn


class CrossAttentionBridge(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def _attend(self, queries: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = queries.shape
        num_memory_tokens = memory.size(1)
        projected_queries = self.query_proj(queries).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        projected_keys = self.key_proj(memory).view(batch_size, num_memory_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        projected_values = self.value_proj(memory).view(batch_size, num_memory_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_logits = torch.matmul(projected_queries, projected_keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, projected_values)
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        token_weights = attention_weights.mean(dim=1)
        return attended, token_weights

    def forward(self, positive_memory: torch.Tensor, negative_memory: torch.Tensor, attribute_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_tokens, _ = attribute_tokens.shape
        del batch_size
        positive_context, positive_attention = self._attend(attribute_tokens, positive_memory)
        negative_context, negative_attention = self._attend(attribute_tokens, negative_memory)
        fused_tokens = attribute_tokens + positive_context - negative_context
        conditioned_tokens = self.output_norm(fused_tokens + self.ffn(fused_tokens))
        personalized_representation = conditioned_tokens.mean(dim=1)
        return {
            "conditioned_tokens": conditioned_tokens,
            "positive_attention": positive_attention,
            "negative_attention": negative_attention,
            "personalized_representation": personalized_representation,
        }

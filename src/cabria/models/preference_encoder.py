from __future__ import annotations

import torch
from torch import nn


class PreferenceEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_preference_tokens: int,
        num_heads: int,
        dropout: float = 0.0,
        mode: str = "hard",
        temperature: float = 1.0,
        score_bin_temperature: float = 0.7,
    ) -> None:
        super().__init__()
        del num_heads, dropout
        self.num_preference_tokens = num_preference_tokens
        self.mode = mode.lower()
        self.temperature = max(float(temperature), 1e-6)
        self.score_bin_temperature = max(float(score_bin_temperature), 1e-6)
        self.positive_score_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.negative_score_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.support_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.eps = 1e-6

    def _select_memory(
        self,
        support_embeddings: torch.Tensor,
        normalized_scores: torch.Tensor,
        descending: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_support, embed_dim = support_embeddings.shape
        k = min(self.num_preference_tokens, num_support)
        sorted_indices = torch.argsort(normalized_scores, dim=1, descending=descending)
        selected_indices = sorted_indices[:, :k]
        gather_indices = selected_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        selected_embeddings = torch.gather(support_embeddings, 1, gather_indices)
        selected_scores = torch.gather(normalized_scores, 1, selected_indices)

        if k < self.num_preference_tokens:
            pad_count = self.num_preference_tokens - k
            pad_embeddings = selected_embeddings[:, -1:, :].expand(batch_size, pad_count, embed_dim)
            pad_scores = selected_scores[:, -1:].expand(batch_size, pad_count)
            selected_embeddings = torch.cat([selected_embeddings, pad_embeddings], dim=1)
            selected_scores = torch.cat([selected_scores, pad_scores], dim=1)

        return selected_embeddings, selected_scores

    def _soft_memory(
        self,
        support_embeddings: torch.Tensor,
        normalized_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_weights = torch.softmax(normalized_scores / self.temperature, dim=1)
        negative_weights = torch.softmax(-normalized_scores / self.temperature, dim=1)
        positive_embedding = torch.sum(positive_weights.unsqueeze(-1) * support_embeddings, dim=1, keepdim=True)
        negative_embedding = torch.sum(negative_weights.unsqueeze(-1) * support_embeddings, dim=1, keepdim=True)
        positive_score = torch.sum(positive_weights * torch.relu(normalized_scores), dim=1, keepdim=True)
        negative_score = -torch.sum(negative_weights * torch.relu(-normalized_scores), dim=1, keepdim=True)
        return positive_embedding, positive_score, negative_embedding, negative_score

    def _support_token_memory(
        self,
        support_embeddings: torch.Tensor,
        normalized_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            support_embeddings,
            normalized_scores,
            support_embeddings,
            normalized_scores,
        )

    def _score_bin_memory(
        self,
        support_embeddings: torch.Tensor,
        normalized_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, embed_dim = support_embeddings.shape
        bin_positions = torch.linspace(
            0.5,
            float(self.num_preference_tokens) - 0.5,
            self.num_preference_tokens,
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )
        bin_centers = bin_positions / float(self.num_preference_tokens)
        score_scale = torch.clamp(normalized_scores.detach().abs().amax(dim=1, keepdim=True), min=1.0)
        positive_centers = bin_centers.unsqueeze(0) * score_scale
        negative_centers = -positive_centers

        def pool_by_centers(centers: torch.Tensor) -> torch.Tensor:
            distance = normalized_scores.unsqueeze(1) - centers.unsqueeze(-1)
            logits = -(distance * distance) / self.score_bin_temperature
            weights = torch.softmax(logits, dim=-1).to(dtype=support_embeddings.dtype)
            return torch.bmm(weights, support_embeddings)

        positive_embeddings = pool_by_centers(positive_centers)
        negative_embeddings = pool_by_centers(negative_centers)
        return (
            positive_embeddings.reshape(batch_size, self.num_preference_tokens, embed_dim),
            positive_centers,
            negative_embeddings.reshape(batch_size, self.num_preference_tokens, embed_dim),
            negative_centers,
        )

    def forward(self, support_embeddings: torch.Tensor, support_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        if support_embeddings.dim() == 2:
            support_embeddings = support_embeddings.unsqueeze(0)
        if support_scores.dim() == 1:
            support_scores = support_scores.unsqueeze(0)

        centered_scores = support_scores - support_scores.mean(dim=1, keepdim=True)
        normalized_scores = centered_scores / (support_scores.std(dim=1, keepdim=True, unbiased=False) + self.eps)
        normalized_support = self.support_norm(support_embeddings)

        if self.mode == "soft":
            positive_selected, positive_selected_scores, negative_selected, negative_selected_scores = self._soft_memory(
                normalized_support,
                normalized_scores,
            )
        elif self.mode == "hard":
            positive_selected, positive_selected_scores = self._select_memory(
                normalized_support,
                normalized_scores,
                descending=True,
            )
            negative_selected, negative_selected_scores = self._select_memory(
                normalized_support,
                normalized_scores,
                descending=False,
            )
        elif self.mode == "support_tokens":
            positive_selected, positive_selected_scores, negative_selected, negative_selected_scores = self._support_token_memory(
                normalized_support,
                normalized_scores,
            )
        elif self.mode == "score_bins":
            positive_selected, positive_selected_scores, negative_selected, negative_selected_scores = self._score_bin_memory(
                normalized_support,
                normalized_scores,
            )
        else:
            raise ValueError(f"Unsupported preference encoder mode: {self.mode}")

        positive_memory = self.output_norm(
            positive_selected + self.positive_score_embed(torch.relu(positive_selected_scores).unsqueeze(-1))
        )
        negative_memory = self.output_norm(
            negative_selected + self.negative_score_embed(torch.relu(-negative_selected_scores).unsqueeze(-1))
        )
        return {
            "positive_memory": positive_memory,
            "negative_memory": negative_memory,
            "normalized_scores": normalized_scores,
            "support_embeddings": normalized_support,
            "positive_selected_scores": positive_selected_scores,
            "negative_selected_scores": negative_selected_scores,
        }

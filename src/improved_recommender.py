"""Hard-tag and historical-scene enhancement for the MSCRS recommender.

All recommendation scores in this module are defined over the *candidate movie*
axis, never over the full DBpedia entity vocabulary.  This makes the tensor
contract explicit and avoids the easy-to-miss global-id/local-id mismatch in
the original training script.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(1.0)


@dataclass
class EnhancedRecOutput:
    final_logits: torch.Tensor
    hard_logits: torch.Tensor
    scene_logits: torch.Tensor
    tag_logits: torch.Tensor
    tag_probabilities: torch.Tensor
    scene_gate: torch.Tensor
    scene_reliability: torch.Tensor
    gate_features: torch.Tensor
    retrieved_scene_ids: torch.Tensor
    retrieved_scene_weights: torch.Tensor
    entity_preference_logits: torch.Tensor
    predicted_positive_mask: torch.Tensor


class EntityPreferencePredictor(nn.Module):
    """Predict positive/negative/neutral status for entities visible now.

    Questionnaire labels supervise this module during training only.  Forward
    retrieval always uses its predictions, which preserves train/inference
    parity and keeps validation/test labels out of the model inputs.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dialogue_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3),
        )

    def forward(
        self,
        dialogue_rep: torch.Tensor,
        entity_vectors: torch.Tensor,
        entity_mask: torch.Tensor,
        positive_threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioned = entity_vectors + self.dialogue_proj(dialogue_rep).unsqueeze(1)
        logits = self.network(conditioned)
        probabilities = F.softmax(logits, dim=-1)
        mask = entity_mask.to(entity_vectors.dtype)
        positive_weights = probabilities[..., 0] * mask
        negative_weights = probabilities[..., 1] * mask

        def weighted_pool(weights: torch.Tensor) -> torch.Tensor:
            return (entity_vectors * weights.unsqueeze(-1)).sum(1) / weights.sum(
                1, keepdim=True
            ).clamp_min(1e-8)

        positive = weighted_pool(positive_weights)
        negative = weighted_pool(negative_weights)
        predicted_positive = (
            logits.argmax(dim=-1).eq(0)
            & probabilities[..., 0].ge(float(positive_threshold))
            & entity_mask
        )
        return logits, positive, negative, predicted_positive


class HardTagPreference(nn.Module):
    """Predict a dialogue tag profile and score every candidate movie.

    Tensor contract:
      dialogue_rep:       [B, D]
      positive_entities:  [B, D] or None
      negative_entities:  [B, D] or None
      item_tags:           [N, R] fixed multi-hot/soft hard-tag matrix
      tag_logits:          [B, R]
      hard_logits:         [B, N]
    """

    def __init__(self, hidden_size: int, num_tags: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_tags = int(num_tags)
        self.dialogue_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.positive_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.negative_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tag_head = nn.Linear(hidden_size, num_tags)

    def forward(
        self,
        dialogue_rep: torch.Tensor,
        item_tags: torch.Tensor,
        positive_entities: Optional[torch.Tensor] = None,
        negative_entities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dialogue_rep.ndim != 2:
            raise ValueError(f"dialogue_rep must be [B,D], got {tuple(dialogue_rep.shape)}")
        if item_tags.ndim != 2 or item_tags.shape[1] != self.num_tags:
            raise ValueError(
                f"item_tags must be [N,{self.num_tags}], got {tuple(item_tags.shape)}"
            )

        fused = self.dialogue_proj(dialogue_rep)
        if positive_entities is not None:
            fused = fused + self.positive_proj(positive_entities)
        if negative_entities is not None:
            fused = fused - self.negative_proj(negative_entities)

        tag_logits = self.tag_head(fused)
        tag_probabilities = torch.sigmoid(tag_logits)
        normalized_tags = item_tags.to(tag_probabilities.dtype)
        normalized_tags = normalized_tags / normalized_tags.sum(dim=-1, keepdim=True).clamp_min(1.0)
        hard_logits = tag_probabilities @ normalized_tags.T
        return tag_logits, tag_probabilities, hard_logits


class HistoricalSceneMemory(nn.Module):
    """Leakage-aware batched retrieval from a fixed training-scene memory.

    A scene is one historical conversation.  `scene_movie_incidence[m, n]` is
    non-zero when candidate movie n belongs to scene m.  Retrieval can be
    anchored to positive movie mentions; when there is no anchor it falls back
    to global semantic/tag retrieval.
    """

    def __init__(
        self,
        scene_embeddings: torch.Tensor,
        scene_tag_profiles: torch.Tensor,
        scene_movie_incidence: torch.Tensor,
        scene_conversation_ids: torch.Tensor,
        top_k: int = 8,
        semantic_weight: float = 0.7,
        tag_weight: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        if scene_embeddings.ndim != 2:
            raise ValueError("scene_embeddings must be [M,D]")
        m = scene_embeddings.shape[0]
        if scene_tag_profiles.shape[0] != m or scene_movie_incidence.shape[0] != m:
            raise ValueError("all scene assets must have the same M dimension")
        if scene_conversation_ids.shape != (m,):
            raise ValueError("scene_conversation_ids must be [M]")
        self.register_buffer("scene_embeddings", F.normalize(scene_embeddings.float(), dim=-1))
        self.register_buffer("scene_tag_profiles", scene_tag_profiles.float())
        incidence = scene_movie_incidence.float()
        if not incidence.is_sparse:
            incidence = incidence.to_sparse_coo()
        incidence = incidence.coalesce()
        self.register_buffer("scene_movie_incidence", incidence)
        scene_rows, movie_cols = incidence.indices()
        counts = torch.bincount(scene_rows, minlength=m)
        max_movies = int(counts.max().item())
        padded_movie_ids = torch.zeros(m, max_movies, dtype=torch.long)
        padded_movie_values = torch.zeros(m, max_movies, dtype=torch.float)
        starts = torch.cumsum(counts, dim=0) - counts
        offsets = torch.arange(scene_rows.numel()) - torch.repeat_interleave(starts, counts)
        padded_movie_ids[scene_rows, offsets] = movie_cols
        padded_movie_values[scene_rows, offsets] = incidence.values()
        self.register_buffer("scene_movie_ids", padded_movie_ids)
        self.register_buffer("scene_movie_values", padded_movie_values)
        self.register_buffer("scene_conversation_ids", scene_conversation_ids.long())
        self.top_k = int(top_k)
        self.semantic_weight = float(semantic_weight)
        self.tag_weight = float(tag_weight)
        self.temperature = float(temperature)

    def forward(
        self,
        dialogue_rep: torch.Tensor,
        tag_probabilities: torch.Tensor,
        anchor_candidate_mask: Optional[torch.Tensor] = None,
        conversation_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(dialogue_rep.float(), dim=-1)
        semantic_similarity = query @ self.scene_embeddings.T

        q_tag = F.normalize(tag_probabilities.float(), dim=-1)
        s_tag = F.normalize(self.scene_tag_profiles.float(), dim=-1)
        tag_similarity = q_tag @ s_tag.T
        retrieval_logits = (
            self.semantic_weight * semantic_similarity + self.tag_weight * tag_similarity
        )

        valid = torch.ones_like(retrieval_logits, dtype=torch.bool)
        if anchor_candidate_mask is not None:
            if anchor_candidate_mask.shape[1] != self.scene_movie_incidence.shape[1]:
                raise ValueError("anchor candidate dimension does not match scene incidence")
            # A float mask carries the model's P(positive) for each mentioned
            # movie.  Boolean masks remain supported for ablations/tests.
            anchored_strength = torch.sparse.mm(
                self.scene_movie_incidence, anchor_candidate_mask.float().T
            ).T
            anchored = anchored_strength > 0
            # The method is explicitly anchor-based: with no confidently
            # positive mentioned movie, it returns zero scene compensation.
            valid = anchored
            # Prefer scenes supported by stronger/multiple positive anchors,
            # while keeping semantic and tag similarity as the main ranker.
            retrieval_logits = retrieval_logits + 0.1 * torch.log1p(anchored_strength)
        if conversation_ids is not None:
            # Exclude scenes from the same dialogue during training and evaluation.
            valid = valid & conversation_ids[:, None].ne(self.scene_conversation_ids[None, :])

        retrieval_logits = retrieval_logits.masked_fill(~valid, torch.finfo(retrieval_logits.dtype).min)
        k = min(self.top_k, retrieval_logits.shape[1])
        top_values, top_ids = torch.topk(retrieval_logits, k=k, dim=-1)
        top_valid = torch.gather(valid, 1, top_ids)
        top_values = top_values.masked_fill(~top_valid, -1e4)
        weights = F.softmax(top_values / self.temperature, dim=-1) * top_valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Reliability is high only when retrieval is both confident (the best
        # scene is separated from the runner-up) and supported by a positive
        # anchor.  It lets the gate distinguish "the base model is uncertain"
        # from "the scene memory has trustworthy evidence".
        if k > 1:
            sorted_values = torch.sort(top_values, dim=-1, descending=True).values
            retrieval_margin = torch.sigmoid(
                (sorted_values[:, 0] - sorted_values[:, 1]) / self.temperature
            )
        else:
            retrieval_margin = top_valid[:, 0].to(weights.dtype)
        retrieval_reliability = retrieval_margin * top_valid.any(dim=-1).to(weights.dtype)

        # Scatter retrieved scene memberships into candidate scores without
        # materializing a [B,K,N] tensor.
        selected_movie_ids = self.scene_movie_ids[top_ids]
        selected_movie_values = self.scene_movie_values[top_ids]
        contributions = weights.unsqueeze(-1) * selected_movie_values
        scene_logits = weights.new_zeros(weights.shape[0], self.scene_movie_incidence.shape[1])
        scene_logits.scatter_add_(
            1,
            selected_movie_ids.flatten(1),
            contributions.flatten(1),
        )
        coverage = scene_logits.gt(0).float().mean(dim=-1)
        return scene_logits, top_ids, weights, coverage, retrieval_reliability


class SceneNeedGate(nn.Module):
    """Learn whether the current dialogue needs scene compensation.

    Features match the paper design: tag uncertainty, top1-top2 tag margin,
    base recommender uncertainty, and tag/scene candidate coverage.
    """

    FEATURE_NAMES = (
        "tag_entropy",
        "tag_margin",
        "base_entropy",
        "base_margin",
        "hard_coverage",
        "scene_coverage",
        "scene_reliability",
    )

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(self.FEATURE_NAMES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        p = probabilities.clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-1) / torch.log(
            p.new_tensor(float(max(2, p.shape[-1])))
        )

    def forward(
        self,
        tag_probabilities: torch.Tensor,
        base_logits: torch.Tensor,
        item_tags: torch.Tensor,
        scene_coverage: torch.Tensor,
        scene_reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tag_distribution = tag_probabilities / tag_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tag_top2 = torch.topk(tag_distribution, k=min(2, tag_distribution.shape[-1]), dim=-1).values
        tag_margin = tag_top2[:, 0] - (tag_top2[:, 1] if tag_top2.shape[1] > 1 else 0)

        base_distribution = F.softmax(base_logits, dim=-1)
        base_top2 = torch.topk(base_distribution, k=min(2, base_distribution.shape[-1]), dim=-1).values
        base_margin = base_top2[:, 0] - (base_top2[:, 1] if base_top2.shape[1] > 1 else 0)

        active_tags = tag_probabilities.ge(0.5).float()
        hard_coverage = (active_tags @ item_tags.float().T).gt(0).float().mean(dim=-1)
        features = torch.stack(
            [
                self._normalized_entropy(tag_distribution),
                tag_margin,
                self._normalized_entropy(base_distribution),
                base_margin,
                hard_coverage,
                scene_coverage,
                scene_reliability,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.network(features)).squeeze(-1), features


class EnhancedRecommender(nn.Module):
    """Final score: base + alpha * hard + gate * beta * scene."""

    def __init__(
        self,
        hidden_size: int,
        item_tags: torch.Tensor,
        scene_assets: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        scene_top_k: int = 8,
        hard_threshold: float = 0.8,
        use_hard: bool = True,
        use_scene: bool = True,
        positive_anchor_threshold: float = 0.0,
        score_normalization: bool = True,
    ):
        super().__init__()
        if item_tags.ndim != 2:
            raise ValueError("item_tags must be [N,R]")
        self.register_buffer("item_tags", item_tags.float())
        self.entity_preference = EntityPreferencePredictor(hidden_size)
        self.tag_head = HardTagPreference(hidden_size, item_tags.shape[1])
        self.gate = SceneNeedGate()
        self.scene_memory = None
        if scene_assets is not None:
            self.scene_memory = HistoricalSceneMemory(top_k=scene_top_k, **scene_assets)
        self.log_alpha = nn.Parameter(torch.tensor(float(alpha)).log())
        self.log_beta = nn.Parameter(torch.tensor(float(beta)).log())
        self.hard_threshold = float(hard_threshold)
        self.use_hard = bool(use_hard)
        self.use_scene = bool(use_scene)
        self.positive_anchor_threshold = float(positive_anchor_threshold)
        self.score_normalization = bool(score_normalization)

    @staticmethod
    def _align_component_scale(
        component: torch.Tensor, base_logits: torch.Tensor
    ) -> torch.Tensor:
        """Match each auxiliary branch to the per-dialogue base-logit scale."""
        centered = component - component.mean(dim=-1, keepdim=True)
        component_std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        base_std = base_logits.std(dim=-1, keepdim=True, unbiased=False).detach().clamp_min(1e-3)
        return centered / component_std * base_std

    def forward(
        self,
        base_logits: torch.Tensor,
        dialogue_rep: torch.Tensor,
        scene_query_rep: Optional[torch.Tensor] = None,
        entity_vectors: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
        entity_candidate_ids: Optional[torch.Tensor] = None,
        conversation_ids: Optional[torch.Tensor] = None,
        hard_inference: bool = False,
    ) -> EnhancedRecOutput:
        if entity_vectors is None or entity_mask is None or entity_candidate_ids is None:
            raise ValueError("entity vectors, mask, and candidate ids are required")
        preference_logits, positive_entities, negative_entities, predicted_positive = (
            self.entity_preference(
                dialogue_rep, entity_vectors, entity_mask,
                positive_threshold=self.positive_anchor_threshold,
            )
        )
        anchor_candidate_mask = torch.zeros(
            base_logits.shape[0], self.item_tags.shape[0], dtype=base_logits.dtype,
            device=base_logits.device,
        )
        positive_probabilities = F.softmax(preference_logits, dim=-1)[..., 0]
        valid_anchor = (
            entity_mask
            & entity_candidate_ids.ge(0)
            & positive_probabilities.ge(self.positive_anchor_threshold)
        )
        anchor_candidate_mask.scatter_add_(
            1,
            entity_candidate_ids.clamp_min(0),
            positive_probabilities * valid_anchor.to(positive_probabilities.dtype),
        )

        tag_logits, tag_probabilities, hard_logits = self.tag_head(
            dialogue_rep,
            self.item_tags,
            positive_entities=positive_entities,
            negative_entities=negative_entities,
        )

        if self.scene_memory is None:
            scene_logits = torch.zeros_like(base_logits)
            scene_ids = torch.empty(base_logits.shape[0], 0, dtype=torch.long, device=base_logits.device)
            scene_weights = torch.empty_like(scene_ids, dtype=base_logits.dtype)
            scene_coverage = torch.zeros(base_logits.shape[0], device=base_logits.device)
            scene_reliability = torch.zeros(base_logits.shape[0], device=base_logits.device)
        else:
            scene_logits, scene_ids, scene_weights, scene_coverage, scene_reliability = self.scene_memory(
                dialogue_rep if scene_query_rep is None else scene_query_rep,
                tag_probabilities,
                anchor_candidate_mask=anchor_candidate_mask,
                conversation_ids=conversation_ids,
            )

        scene_gate, gate_features = self.gate(
            tag_probabilities, base_logits, self.item_tags, scene_coverage, scene_reliability
        )
        if hard_inference:
            # High tag confidence means the tag branch is already sufficient.
            scene_gate = tag_probabilities.max(dim=-1).values.lt(self.hard_threshold).float()

        if self.score_normalization:
            hard_logits = self._align_component_scale(hard_logits, base_logits)
            scene_logits = self._align_component_scale(scene_logits, base_logits)
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        hard_term = alpha * hard_logits if self.use_hard else torch.zeros_like(hard_logits)
        scene_term = (
            scene_gate[:, None] * beta * scene_logits
            if self.use_scene else torch.zeros_like(scene_logits)
        )
        final_logits = base_logits + hard_term + scene_term
        return EnhancedRecOutput(
            final_logits=final_logits,
            hard_logits=hard_logits,
            scene_logits=scene_logits,
            tag_logits=tag_logits,
            tag_probabilities=tag_probabilities,
            scene_gate=scene_gate,
            scene_reliability=scene_reliability,
            gate_features=gate_features,
            retrieved_scene_ids=scene_ids,
            retrieved_scene_weights=scene_weights,
            entity_preference_logits=preference_logits,
            predicted_positive_mask=predicted_positive,
        )


def gather_entity_preferences(
    entity_table: torch.Tensor,
    entity_ids: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool positive/negative mentioned entities into [B,D] preference vectors."""
    entity_vectors = entity_table[entity_ids]
    positive = _masked_mean(entity_vectors, positive_mask, dim=1)
    negative = _masked_mean(entity_vectors, negative_mask, dim=1)
    return positive, negative

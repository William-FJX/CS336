from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F


class GraphTripletLoss(nn.Module):
    """Prepare once from the edge counts returned by either graph builder."""

    def __init__(self, edge_counts, margin_scale=10.0):
        super().__init__()
        if not edge_counts:
            raise ValueError("The graph needs at least one edge.")
        if margin_scale <= 0:
            raise ValueError("margin_scale must be positive.")
        self.margin_scale = margin_scale

        self.total_weight = sum(edge_counts.values())  # Count each undirected edge once.
        self.neighbors = defaultdict(dict)
        for (left, right), weight in edge_counts.items():
            if not 0 <= left < right or weight <= 0:
                raise ValueError("Expected edges (left, right) with 0 <= left < right and positive counts.")
            self.neighbors[left][right] = weight
            self.neighbors[right][left] = weight

        self.anchors = list(self.neighbors)
        self.max_token_id = max(self.anchors)
        self.strongest_neighbors = {}
        for anchor, neighbors in self.neighbors.items():
            strongest_weight = max(neighbors.values())
            self.strongest_neighbors[anchor] = [
                token for token, weight in neighbors.items() if weight == strongest_weight
            ]

    def sample_anchors(self, batch_size):
        rows = torch.randint(len(self.anchors), (batch_size,), device="cpu").tolist()
        return torch.tensor([self.anchors[i] for i in rows], device="cpu")

    def choose_strongest_neighbors(self, anchors):
        """Sample uniformly among neighbors tied for the largest edge count."""
        positives = []
        for anchor in anchors.tolist():
            candidates = self.strongest_neighbors[anchor]
            index = torch.randint(len(candidates), (), device="cpu").item()
            positives.append(candidates[index])
        return torch.tensor(positives, device="cpu")

    def sample_negatives(self, anchors, positives, vocab_size, num_negatives):
        """Complete each triplet with a uniform draw excluding the anchor and positive."""
        # Sample from a shortened vocabulary, then skip the two excluded IDs.
        negatives = torch.randint(vocab_size - 2, (len(anchors), num_negatives), device="cpu")
        negatives += (negatives >= torch.minimum(anchors, positives)[:, None]).long()
        negatives += (negatives >= torch.maximum(anchors, positives)[:, None]).long()
        return negatives

    def calculate_ratio_gap(self, anchor_ids, positive_ids, negative_ids, embedding_weights):
        """Return r_positive - r_negative for each triplet, using the whole graph's weight."""
        weight_gaps = []
        for anchor, positive, negatives in zip(anchor_ids.tolist(), positive_ids.tolist(), negative_ids.tolist()):
            neighbors = self.neighbors[anchor]
            weight_gaps.append([neighbors[positive] - neighbors.get(n, 0) for n in negatives])
        dtype = torch.promote_types(embedding_weights.dtype, torch.float32)
        return torch.tensor(weight_gaps, device=embedding_weights.device, dtype=dtype) / self.total_weight

    def calculate_triplet_loss(self, embedding_weights, anchor_ids, positive_ids, negative_ids, delta):
        """Penalize the part of the required margin missing from the cosine-distance gap."""
        margin = torch.log1p((delta / self.margin_scale).square())
        device, dtype = embedding_weights.device, delta.dtype
        anchor_vectors = embedding_weights[anchor_ids.to(device)].to(dtype)
        positive_vectors = embedding_weights[positive_ids.to(device)].to(dtype)
        negative_vectors = embedding_weights[negative_ids.to(device)].to(dtype)
        positive_cos = F.cosine_similarity(anchor_vectors, positive_vectors, dim=-1)
        negative_cos = F.cosine_similarity(anchor_vectors[:, None, :], negative_vectors, dim=-1)

        # With cosine distance, d(anchor, neg) - d(anchor, pos) = cos(pos) - cos(neg).
        actual_gap = positive_cos[:, None] - negative_cos
        return (margin - actual_gap).relu().mean()

    def forward(self, embedding_weights, batch_size=64, num_negatives=8):
        vocab_size = embedding_weights.shape[0]
        if vocab_size < 3 or self.max_token_id >= vocab_size:
            raise ValueError("The embedding table needs at least three tokens and must cover every graph ID.")
        if batch_size < 1 or num_negatives < 1:
            raise ValueError("batch_size and num_negatives must be positive.")

        anchors = self.sample_anchors(batch_size)
        positives = self.choose_strongest_neighbors(anchors)
        negatives = self.sample_negatives(anchors, positives, vocab_size, num_negatives)
        delta = self.calculate_ratio_gap(anchors, positives, negatives, embedding_weights)
        return self.calculate_triplet_loss(embedding_weights, anchors, positives, negatives, delta)

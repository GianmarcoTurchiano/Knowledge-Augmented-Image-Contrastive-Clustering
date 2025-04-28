import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_wise_entropy(probs):
    probs_avg = probs.mean(dim=0)
    probs_avg_log = torch.log(probs_avg + 1e-10)
    entropy = -torch.sum(probs_avg * probs_avg_log)

    return entropy

class NTXentLoss(nn.Module):
    def __init__(self, temperature, batch_size):
        super().__init__()
        self._temperature = temperature
        self._N = 2 * batch_size

        # create pos‐mask
        pos = torch.zeros(self._N, self._N, dtype=torch.bool)
        idx = torch.arange(batch_size)
        pos[idx, idx + batch_size] = True
        pos[idx + batch_size, idx] = True

        # create neg‐mask = everything except diag & positives
        eye = torch.eye(self._N, dtype=torch.bool)
        neg = ~eye & ~pos

        # register as buffers so they move with .to(device), .eval(), etc.
        self.register_buffer('pos_mask', pos)
        self.register_buffer('neg_mask', neg)

        # create labels = zeros of length N
        labels = torch.zeros(self._N, dtype=torch.long)
        self.register_buffer('labels', labels)

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim = (z @ z.T) / self._temperature
        sim = sim - sim.max(dim=1, keepdim=True).values

        positives = sim[self.pos_mask].view(self._N, 1)
        negatives = sim[self.neg_mask].view(self._N, -1)
        logits = torch.cat([positives, negatives], dim=1)
        
        return F.cross_entropy(logits, self.labels)
    
class ContrastiveClusteringLoss(nn.Module):
    def __init__(
        self,
        batch_size=256,
        clusters_count=10,
        temperature_embeddings=0.5,
        temperature_clusters=1.0,
        regularization_strength=1.0,
    ):
        super().__init__()
        self.loss_embeddings_fn = NTXentLoss(temperature_embeddings, batch_size)
        self.loss_clusters_fn = NTXentLoss(temperature_clusters, clusters_count)
        self._regularization_strength = regularization_strength
        clusters_count = torch.Tensor([clusters_count])
        uniform_clusters_entropy = 2 * torch.log(clusters_count)
        self.register_buffer('uniform_clusters_entropy', uniform_clusters_entropy)

    def forward(self, embeddings_1, embeddings_2, logits_1, logits_2):
        loss_embeddings = self.loss_embeddings_fn(embeddings_1, embeddings_2)
        
        probs_1 = F.softmax(logits_1, dim=1)
        probs_2 = F.softmax(logits_2, dim=1)
        loss_clusters = self.loss_clusters_fn(probs_1.T, probs_2.T)
        batch_wise_clusters_entropy = batch_wise_entropy(probs_1) + batch_wise_entropy(probs_2)
        regularization = self._regularization_strength * (self.uniform_clusters_entropy - batch_wise_clusters_entropy)
        loss_clusters_regularized = loss_clusters + regularization
        
        loss_total = loss_embeddings + loss_clusters_regularized
        
        return loss_embeddings, loss_clusters, batch_wise_clusters_entropy, loss_total

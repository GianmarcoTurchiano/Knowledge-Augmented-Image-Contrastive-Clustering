import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torch.utils.data import DataLoader, Dataset
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os
from sklearn.manifold import TSNE


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)
    

normalize = transforms.Normalize(
    mean=[0.49139968, 0.48215827, 0.44653124],
    std=[0.24703233, 0.24348505, 0.26158768]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    # For small images such as CIFAR-10, one might leave out GaussianBlur.
    transforms.ToTensor(),
    normalize
])

class ContrastiveCIFAR10(Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                    download=True,
                                                    transform=TwoCropsTransform(train_transform))
    def __getitem__(self, index):
        (xi, xj), label = self.dataset[index]
        return xi, xj, label

    def __len__(self):
        return len(self.dataset)
    
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
        self.uniform_clusters_entropy = 2 * torch.log(clusters_count)
        #self.register_buffer('uniform_clusters_entropy', uniform_clusters_entropy)

    def forward(self, embeddings_1, embeddings_2, logits_1, logits_2):
        loss_embeddings = self.loss_embeddings_fn(embeddings_1, embeddings_2)
        
        probs_1 = F.softmax(logits_1, dim=1)
        probs_2 = F.softmax(logits_2, dim=1)
        loss_clusters = self.loss_clusters_fn(probs_1.T, probs_2.T)
        batch_wise_clusters_entropy = batch_wise_entropy(probs_1) + batch_wise_entropy(probs_2)
        regularization = self._regularization_strength * batch_wise_clusters_entropy
        loss_clusters_regularized = loss_clusters - regularization
        
        loss_total = loss_embeddings + loss_clusters_regularized
        
        return loss_embeddings, loss_clusters, batch_wise_clusters_entropy, loss_total
    
class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Head, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, x):
        y = self.net(x)
        return y

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = resnet34(num_classes=512)
        self.emb_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)
    
class ContrastiveClusteringModel(nn.Module):
    def __init__(self, num_clusters, emb_dim):
        super(ContrastiveClusteringModel, self).__init__()
        
        self.backbone = Backbone()
        self.inst_head = Head(in_dim=self.backbone.emb_dim, out_dim=emb_dim)
        self.clust_head = Head(in_dim=self.backbone.emb_dim, out_dim=num_clusters)

    def forward(self, x):
        in_emb = self.backbone(x)
        out_emb = self.inst_head(in_emb)
        logits = self.clust_head(in_emb)
        return out_emb, logits
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clusters_count = 10
embeddings_dimension = 128
batch_size = 128
epochs_count = 200
temperature_clusters = 1.0
temperatures_embeddings = 0.5
learning_rate = 3e-4
regularization_strength = 1.0

dataset = ContrastiveCIFAR10(train=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

loss_fn = ContrastiveClusteringLoss(
    regularization_strength=regularization_strength,
    temperature_embeddings=temperatures_embeddings,
    temperature_clusters=temperature_clusters,
    batch_size=batch_size,
    clusters_count=clusters_count
).to(device)

model = ContrastiveClusteringModel(clusters_count, embeddings_dimension).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
epochs_no_improve = 0

n_samples = len(loader)

for epoch in tqdm(range(epochs_count), desc="Epochs"):
    epoch_loss_total = 0.0
    epoch_loss_embeddings = 0.0
    epoch_loss_clusters = 0.0
    epoch_batch_wise_clusters_entropy = 0.0

    labels = []
    predictions_a = []
    predictions_b = []

    model.train()
    for pic_a, pic_b, label in tqdm(loader, desc="Training", leave=False):
        pic_a, pic_b = pic_a.to(device), pic_b.to(device)

        optimizer.zero_grad()

        emb_a, logits_a = model(pic_a)
        emb_b, logits_b = model(pic_b)
        
        (
            loss_embeddings,
            loss_clusters,
            batch_wise_clusters_entropy,
            loss_total
        ) = loss_fn(emb_a, emb_b, logits_a, logits_b)

        loss_total.backward()
        optimizer.step()

        epoch_loss_embeddings += loss_embeddings.item()
        epoch_loss_clusters += loss_clusters.item()
        epoch_loss_total += loss_total.item()
        epoch_batch_wise_clusters_entropy += batch_wise_clusters_entropy.item()

        pred_a = torch.argmax(logits_a, dim=1)
        predictions_a.append(pred_a.cpu())
        pred_b = torch.argmax(logits_b, dim=1)
        predictions_b.append(pred_b.cpu())
        labels.append(label)

    epoch_loss_total_avg = epoch_loss_total / n_samples
    epoch_loss_embeddings_avg = epoch_loss_embeddings / n_samples
    epoch_loss_clusters_avg = epoch_loss_clusters / n_samples
    epoch_batch_wise_clusters_entropy_avg = epoch_batch_wise_clusters_entropy / n_samples

    predictions_a = torch.cat(predictions_a, dim=0).numpy()
    predictions_b = torch.cat(predictions_b, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    nmi_a = normalized_mutual_info_score(labels, predictions_a)
    nmi_b = normalized_mutual_info_score(labels, predictions_b)
    nmi_avg = (nmi_a + nmi_b) / 2
    ari_a = adjusted_rand_score(labels, predictions_a)
    ari_b = adjusted_rand_score(labels, predictions_b)
    ari_avg = (ari_a + ari_b) / 2

    tqdm.write(f"EPOCH {epoch+1}\n"
        f"Loss: {epoch_loss_total_avg:.4f}\n"
            f"\tEmbeddings: {epoch_loss_embeddings_avg:.4f}\n"
            f"\tClusters: {epoch_loss_clusters_avg:.4f}\n"
                f"\t\tH[E(p)]: {epoch_batch_wise_clusters_entropy_avg:.4f} / {loss_fn.uniform_clusters_entropy.item():.4f} nats\n"
        f"Metrics:\n"
            f"\tNormalized Mutual Information (avg. {nmi_avg})\n"
                f"\t\tView A: {nmi_a:.4f}\n"
                f"\t\tView B: {nmi_b:.4f}\n"
            f"\tAdjusted Rand Score (avg. {ari_avg})\n"
                f"\t\tView A: {ari_a:.4f}\n"
                f"\t\tView B: {ari_b:.4f}\n")
    
torch.save(model.state_dict(), "best_model.pth")
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
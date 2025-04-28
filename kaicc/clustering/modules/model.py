import torch.nn as nn
from torchvision.models import resnet34


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
        
        self.clusters_count = num_clusters
        self.backbone = Backbone()
        self.inst_head = Head(in_dim=self.backbone.emb_dim, out_dim=emb_dim)
        self.clust_head = Head(in_dim=self.backbone.emb_dim, out_dim=num_clusters)

    def forward(self, x):
        in_emb = self.backbone(x)
        out_emb = self.inst_head(in_emb)
        logits = self.clust_head(in_emb)
        return out_emb, logits

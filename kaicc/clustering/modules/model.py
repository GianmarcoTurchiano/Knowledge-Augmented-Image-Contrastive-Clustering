import torch.nn as nn
from transformers import CLIPModel


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


class _CLIPEmbedder(nn.Module):
    def __init__(self, base_model_name):
        super(_CLIPEmbedder, self).__init__()
        self.model = CLIPModel.from_pretrained(base_model_name)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_vision_layer(self):
        for param in self.model.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        for param in self.model.visual_projection.parameters():
            param.requires_grad = True

    def unfreeze_last_text_layer(self):
        for param in self.model.text_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        for param in self.model.text_projection.parameters():
            param.requires_grad = True
    
    def forward_image(self, inputs):
        pass

    def forward_text(self, inputs):
        pass
    
    def get_image_embeddings_dimension(self):
        pass
    
    def get_text_embeddings_dimension(self):
        pass


class CLIPEmbedderRaw(_CLIPEmbedder):
    def forward_image(self, inputs):
        embedding = self.model.vision_model(**inputs)
        return embedding.pooler_output
    
    def forward_text(self, inputs):
        embedding = self.model.text_model(**inputs)
        return embedding.pooler_output

    def get_image_embeddings_dimension(self):
        return self.model.visual_projection.in_features
    
    def get_text_embeddings_dimension(self):
        return self.model.text_projection.in_features


class CLIPEmbedderProjected(_CLIPEmbedder):
    def forward_image(self, inputs):
        embedding = self.model.get_image_features(**inputs)
        return embedding
    
    def forward_text(self, inputs):
        embedding = self.model.get_text_features(**inputs)
        return embedding
    
    def get_image_embeddings_dimension(self):
        return self.model.visual_projection.out_features
    
    def get_text_embeddings_dimension(self):
        return self.model.text_projection.out_features


class _CLIPMainToAuxWrapper(nn.Module):
    def __init__(self, clip: _CLIPEmbedder):
        super(_CLIPMainToAuxWrapper, self).__init__()
        self.clip = clip

    def get_main_embeddings_dimension(self):
        pass
    
    def get_auxiliary_embeddings_dimension(self):
        pass

    def main_forward(self, inputs):
        pass
    
    def auxiliary_forward(self, inputs):
        pass


class CLIPImageMainToTextAuxWrapper(_CLIPMainToAuxWrapper):
    def get_main_embeddings_dimension(self):
        return self.clip.get_image_embeddings_dimension()
    
    def get_auxiliary_embeddings_dimension(self):
        return self.clip.get_text_embeddings_dimension()

    def main_forward(self, inputs):
        return self.clip.forward_image(inputs)
    
    def auxiliary_forward(self, inputs):
        return self.clip.forward_text(inputs)


class _CLIPBackbone(nn.Module):
    def __init__(self, clip: _CLIPMainToAuxWrapper):
        super(_CLIPBackbone, self).__init__()
        self.clip = clip


class CLIPMainVsAuxBackbone(_CLIPBackbone):
    def forward(self, inputs_main, inputs_auxiliary):
        embedding_main = self.clip.main_forward(inputs_main)
        embedding_auxiliary = self.clip.auxiliary_forward(inputs_auxiliary)
        return embedding_main, embedding_auxiliary


class CLIPMainVsMainBackbone(_CLIPBackbone):
    def forward(self, inputs_a, inputs_b):
        embedding_a = self.clip.main_forward(inputs_a)
        embedding_b = self.clip.main_forward(inputs_b)
        return embedding_a, embedding_b


class ContrastiveClusteringModel(nn.Module):
    def __init__(
        self,
        backbone: _CLIPBackbone,
        num_clusters: int,
        embeddings_dimension: int
    ):
        super(ContrastiveClusteringModel, self).__init__()
        
        self.clusters_count = num_clusters
        self.backbone = backbone
        backbone_embeddings_dimension = self.backbone.clip.get_main_embeddings_dimension()

        self.inst_head_main = Head(in_dim=backbone_embeddings_dimension, out_dim=embeddings_dimension)
        self.clust_head_main = Head(in_dim=backbone_embeddings_dimension, out_dim=num_clusters)

    def _main_projection(self, in_emb):
        out_emb = self.inst_head_main(in_emb)
        logits = self.clust_head_main(in_emb)
        return out_emb, logits

    def _auxiliary_projection(self, in_emb):
        return self._main_projection(in_emb)   

    def main_forward(self, inputs):
        in_emb = self.backbone.clip.main_forward(inputs)
        out_emb, logits = self._main_projection(in_emb)
        return out_emb, logits

    def auxiliary_forward(self, inputs):
        in_emb = self.backbone.clip.auxiliary_forward(inputs)
        out_emb, logits = self._auxiliary_projection(in_emb)
        return out_emb, logits

    def forward(self, inputs):
        return self.main_forward(inputs)

    def contrastive_forward(self, inputs_a, inputs_b):
        in_emb_a, in_emb_b = self.backbone(inputs_a, inputs_b)

        out_emb_a, logits_a = self._main_projection(in_emb_a)
        out_emb_b, logits_b = self._auxiliary_projection(in_emb_b)

        return out_emb_a, logits_a, out_emb_b, logits_b


class ContrastiveClusteringModelAux(ContrastiveClusteringModel):
    def __init__(
        self,
        backbone: _CLIPMainToAuxWrapper,
        num_clusters: int,
        embeddings_dimension: int
    ):
        super(ContrastiveClusteringModelAux, self).__init__(backbone, num_clusters, embeddings_dimension)

        backbone_auxiliary_embeddings_dimension = self.backbone.clip.get_auxiliary_embeddings_dimension()        

        self.inst_head_auxiliary = Head(in_dim=backbone_auxiliary_embeddings_dimension, out_dim=embeddings_dimension)
        self.clust_head_auxiliary = Head(in_dim=backbone_auxiliary_embeddings_dimension, out_dim=num_clusters)

    def _auxiliary_projection(self, in_emb):
        out_emb = self.inst_head_auxiliary(in_emb)
        logits = self.clust_head_auxiliary(in_emb)
        return out_emb, logits

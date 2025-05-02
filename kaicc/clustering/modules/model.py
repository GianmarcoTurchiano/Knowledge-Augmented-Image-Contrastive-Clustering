import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


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


class CLIPWrapper(nn.Module):
    def __init__(self, base_model_name):
        super(CLIPWrapper, self).__init__()
        self.model = CLIPModel.from_pretrained(base_model_name)
        self.processor = CLIPProcessor.from_pretrained(base_model_name)

    def preprocess_image(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True,
            do_rescale=False
        )
        return inputs
    
    def preprocess_text(self, text):
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            do_rescale=False
        )
        return inputs
    
    def forward_image(self, inputs):
        pass

    def forward_text(self, inputs):
        pass
    
    def get_image_embeddings_dimension(self):
        pass
    
    def get_text_embeddings_dimension(self):
        pass


class CLIPWrapperRaw(CLIPWrapper):
    def forward_image(self, inputs):
        embedding = self.model.vision_model(**inputs)
        return embedding
    
    def forward_text(self, inputs):
        embedding = self.model.text_model(**inputs)
        return embedding

    def get_image_embeddings_dimension(self):
        return self.model.visual_projection.in_features
    
    def get_text_embeddings_dimension(self):
        return self.model.text_projection.in_features


class CLIPWrapperProjected(CLIPWrapper):
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


class CLIPMainToAuxWrapper(nn.Module):
    def __init__(self, clip: CLIPWrapper):
        super(CLIPMainToAuxWrapper, self).__init__()
        self.clip = clip

    def get_main_embeddings_dimension(self):
        pass
    
    def get_auxiliary_embeddings_dimension(self):
        pass

    def main_forward(self, inputs):
        pass
    
    def auxiliary_forward(self, inputs):
        pass
    
    def main_preprocess(self, data):
        pass
    
    def auxiliary_preprocess(self, data):
        pass


class CLIPImageToTextWrapper(CLIPMainToAuxWrapper):
    def get_main_embeddings_dimension(self):
        return self.clip.get_image_embeddings_dimension()
    
    def get_auxiliary_embeddings_dimension(self):
        return self.clip.get_text_embeddings_dimension()

    def main_forward(self, inputs):
        return self.clip.forward_image(inputs)
    
    def auxiliary_forward(self, inputs):
        return self.clip.forward_text(inputs)
    
    def main_preprocess(self, data):
        return self.clip.preprocess_image(data)
    
    def auxiliary_preprocess(self, data):
        return self.clip.preprocess_text(data)


class CLIPBackbone(nn.Module):
    def __init__(self, clip: CLIPMainToAuxWrapper):
        super(CLIPBackbone, self).__init__()
        self.clip = clip

    def preprocess(self, data_a, data_b):
        pass


class CLIPMainVsAuxBackbone(CLIPBackbone):
    def forward(self, inputs_main, inputs_auxiliary):
        embedding_main = self.clip.main_forward(inputs_main)
        embedding_auxiliary = self.clip.auxiliary_forward(inputs_auxiliary)
        return embedding_main, embedding_auxiliary

    def preprocess(self, data_main, data_auxiliary):
        inputs_main = self.clip.main_preprocess(data_main)
        inputs_auxiliary = self.clip.auxiliary_preprocess(data_auxiliary)
        return inputs_main, inputs_auxiliary


class CLIPMainVsMainBackbone(CLIPBackbone):
    def forward(self, inputs_a, inputs_b):
        embedding_a = self.clip.main_forward(inputs_a)
        embedding_b = self.clip.main_forward(inputs_b)
        return embedding_a, embedding_b

    def preprocess(self, data_a, data_b):
        inputs_data_a = self.clip.main_preprocess(data_a)
        inputs_data_b = self.clip.main_preprocess(data_b)
        return inputs_data_a, inputs_data_b


class ContrastiveClusteringModel(nn.Module):
    def __init__(
        self,
        backbone: CLIPBackbone,
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
        backbone: CLIPMainToAuxWrapper,
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

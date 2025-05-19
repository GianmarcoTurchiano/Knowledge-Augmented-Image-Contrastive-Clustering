import random

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.tokenization_utils_base import BatchEncoding


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
    def __init__(self, base_model_name, random_text_slicing=False):
        super(_CLIPEmbedder, self).__init__()
        self.clip = CLIPModel.from_pretrained(base_model_name)
        self.processor = CLIPProcessor.from_pretrained(base_model_name, use_fast=False)
        self.random_text_slicing = random_text_slicing

    def freeze(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    def unfreeze_last_vision_layer(self):
        for param in self.clip.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True

    def unfreeze_last_text_layer(self):
        for param in self.clip.text_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        for param in self.clip.text_projection.parameters():
            param.requires_grad = True
    
    def process_image(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True,
            do_rescale=True
        )

        return inputs
    
    def process_text(self, text):
        if not self.random_text_slicing:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
            return inputs

        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
        )

        n_inputs = inputs['input_ids'].size(0)
        max_len = self.processor.tokenizer.model_max_length

        input_ids = []
        attention_masks = []

        for i in range(n_inputs):
            seq = inputs['input_ids'][i]
            mask = inputs['attention_mask'][i]
            seq_len = mask.sum().item()
            print(seq_len)

            if seq_len < max_len:
                input_ids.append(seq[:max_len])
                attention_masks.append(mask[:max_len])
            else:
                start = random.randint(0, seq_len - max_len)    
                input_ids.append(seq[start : start + max_len])
                attention_masks.append(mask[start : start + max_len])

        inputs_new = BatchEncoding({
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        })

        return inputs_new


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
        embedding = self.clip.vision_model(**inputs)
        return embedding.pooler_output
    
    def forward_text(self, inputs):
        embedding = self.clip.text_model(**inputs)
        return embedding.pooler_output

    def get_image_embeddings_dimension(self):
        return self.clip.visual_projection.in_features
    
    def get_text_embeddings_dimension(self):
        return self.clip.text_projection.in_features


class CLIPEmbedderProjected(_CLIPEmbedder):
    def forward_image(self, inputs):
        embedding = self.clip.get_image_features(**inputs)
        return embedding
    
    def forward_text(self, inputs):
        embedding = self.clip.get_text_features(**inputs)
        return embedding
    
    def get_image_embeddings_dimension(self):
        return self.clip.visual_projection.out_features
    
    def get_text_embeddings_dimension(self):
        return self.clip.text_projection.out_features


class _CLIPMainToAuxWrapper(nn.Module):
    def __init__(self, embedder: _CLIPEmbedder):
        super(_CLIPMainToAuxWrapper, self).__init__()
        self.embedder = embedder

    def get_main_embeddings_dimension(self):
        pass
    
    def get_auxiliary_embeddings_dimension(self):
        pass

    def main_forward(self, inputs):
        pass
    
    def auxiliary_forward(self, inputs):
        pass

    def main_process(self, data):
        pass
    
    def auxiliary_process(self, data):
        pass


class CLIPImageMainToTextAuxWrapper(_CLIPMainToAuxWrapper):
    def get_main_embeddings_dimension(self):
        return self.embedder.get_image_embeddings_dimension()
    
    def get_auxiliary_embeddings_dimension(self):
        return self.embedder.get_text_embeddings_dimension()

    def main_forward(self, inputs):
        return self.embedder.forward_image(inputs)
    
    def auxiliary_forward(self, inputs):
        return self.embedder.forward_text(inputs)

    def main_process(self, data):
        return self.embedder.process_image(data)
    
    def auxiliary_process(self, data):
        return self.embedder.process_text(data)


class _CLIPBackbone(nn.Module):
    def __init__(self, wrapper: _CLIPMainToAuxWrapper):
        super(_CLIPBackbone, self).__init__()
        self.wrapper = wrapper

    def process(self, data_a, data_b):
        pass


class CLIPMainVsAuxBackbone(_CLIPBackbone):
    def forward(self, inputs_main, inputs_auxiliary):
        embedding_main = self.wrapper.main_forward(inputs_main)
        embedding_auxiliary = self.wrapper.auxiliary_forward(inputs_auxiliary)
        return embedding_main, embedding_auxiliary

    def process(self, data_a, data_b):
        inputs_a = self.wrapper.main_process(data_a)
        inputs_b = self.wrapper.auxiliary_process(data_b)

        return inputs_a, inputs_b


class CLIPMainVsMainBackbone(_CLIPBackbone):
    def forward(self, inputs_a, inputs_b):
        embedding_a = self.wrapper.main_forward(inputs_a)
        embedding_b = self.wrapper.main_forward(inputs_b)
        return embedding_a, embedding_b

    def process(self, data_a, data_b):
        inputs_a = self.wrapper.main_process(data_a)
        inputs_b = self.wrapper.main_process(data_b)

        return inputs_a, inputs_b


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
        backbone_embeddings_dimension = self.backbone.wrapper.get_main_embeddings_dimension()

        self.inst_head_main = Head(in_dim=backbone_embeddings_dimension, out_dim=embeddings_dimension)
        self.clust_head_main = Head(in_dim=backbone_embeddings_dimension, out_dim=num_clusters)

    def _main_projection(self, in_emb):
        out_emb = self.inst_head_main(in_emb)
        logits = self.clust_head_main(in_emb)
        return out_emb, logits

    def _auxiliary_projection(self, in_emb):
        return self._main_projection(in_emb)   

    def main_forward(self, inputs):
        in_emb = self.backbone.wrapper.main_forward(inputs)
        out_emb, logits = self._main_projection(in_emb)
        return in_emb, out_emb, logits

    def auxiliary_forward(self, inputs):
        in_emb = self.backbone.wrapper.auxiliary_forward(inputs)
        out_emb, logits = self._auxiliary_projection(in_emb)
        return in_emb, out_emb, logits

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

        backbone_auxiliary_embeddings_dimension = self.backbone.wrapper.get_auxiliary_embeddings_dimension()        

        self.inst_head_auxiliary = Head(in_dim=backbone_auxiliary_embeddings_dimension, out_dim=embeddings_dimension)
        self.clust_head_auxiliary = Head(in_dim=backbone_auxiliary_embeddings_dimension, out_dim=num_clusters)

    def _auxiliary_projection(self, in_emb):
        out_emb = self.inst_head_auxiliary(in_emb)
        logits = self.clust_head_auxiliary(in_emb)
        return out_emb, logits

import argparse
from tqdm.autonotebook import tqdm
import random

import umap
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dotenv import load_dotenv
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from transformers import CLIPProcessor
import seaborn as sns

from kaicc.clustering.modules.model import ContrastiveClusteringModel, ContrastiveClusteringModelAux
from kaicc.clustering.modules.dataset import ArtworkDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id_path', type=str)
    parser.add_argument('--random_seed', type=int)

    parser.add_argument('--size', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--image_archive_path', type=str)
    parser.add_argument('--image_directory_path', type=str)
    parser.add_argument('--labels_file_path', type=str)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    load_dotenv()

    with open(args.run_id_path, 'r') as file:
        parent_run_id = file.read()

    try:
        model_uri = f'runs:/{parent_run_id}/{ContrastiveClusteringModel.__name__}'
        model = mlflow.pytorch.load_model(model_uri)
    except:
        model_uri = f'runs:/{parent_run_id}/{ContrastiveClusteringModelAux.__name__}'
        model = mlflow.pytorch.load_model(model_uri)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    base_model_name = model.backbone.clip.clip.model.config.name_or_path
    processor = CLIPProcessor.from_pretrained(base_model_name, use_fast=False)

    train_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.PILToTensor()
    ])

    dataset = ArtworkDataset(
        args.image_archive_path,
        args.image_directory_path,
        args.labels_file_path,
        train_transform
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    all_styles = []
    all_genres = []
    all_embeddings = []
    all_predictions = []

    for image, style, genre in tqdm(loader):
        inputs = processor(
            images=image,
            return_tensors="pt",
            padding=True,
            do_rescale=True
        )

        inputs = inputs.to(device)

        with torch.no_grad():
            embedding = model.backbone.clip.main_forward(inputs)
            _, logits = model._main_projection(embedding)

        prediction = torch.argmax(logits, dim=1)

        all_styles.extend(style)
        all_genres.extend(genre)
        all_embeddings.append(embedding.detach().cpu())
        all_predictions.append(prediction.detach().cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0) 

    with mlflow.start_run(run_name="Evaluation", parent_run_id=parent_run_id) as run:
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("size", args.size)

        ari_style = adjusted_rand_score(all_styles, all_predictions)
        ari_genre = adjusted_rand_score(all_genres, all_predictions)

        nmi_style = normalized_mutual_info_score(all_styles, all_predictions)
        nmi_genre = normalized_mutual_info_score(all_genres, all_predictions)

        print(f'Style ARI: {ari_style}')
        print(f'Genre ARI: {ari_genre}')
        print(f'Style NMI: {nmi_style}')
        print(f'Genre NMI: {nmi_genre}')

        mlflow.log_metric("Style ARI", ari_style)
        mlflow.log_metric("Genre ARI", ari_genre)
        mlflow.log_metric("Style NMI", nmi_style)
        mlflow.log_metric("Genre NMI", nmi_genre)

        sil = silhouette_score(all_embeddings, all_predictions)

        mlflow.log_metric("Silhouette", sil)

        print(f'Silhouette: {sil}')

        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(all_embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=all_predictions,
            palette="tab10",
            s=10,
            linewidth=0,
            ax=ax
        )
        
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        mlflow.log_figure(fig, "clusters.png")

        plt.close(fig)

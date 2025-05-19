import argparse
from tqdm.autonotebook import tqdm
import random
import os

import umap
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dotenv import load_dotenv
import mlflow
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    contingency_matrix
)
from transformers import CLIPProcessor
import seaborn as sns

from kaicc.clustering.modules.dataset import ArtworkDataset, get_transform

def clustering_accuracy(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    total_correct = cm[row_ind, col_ind].sum()
    return total_correct / np.sum(cm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id_path', type=str)
    parser.add_argument('--random_seed', type=int)

    parser.add_argument('--size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_samples', type=int)

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

    model_uri = f'runs:/{parent_run_id}/model'
    model = mlflow.pytorch.load_model(model_uri)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_transform = get_transform(args.size)

    dataset = ArtworkDataset(
        args.image_archive_path,
        args.image_directory_path,
        args.labels_file_path,
        test_transform
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    all_styles = []
    all_genres = []
    all_embeddings = []
    all_predictions = []

    for image, style, genre in tqdm(loader):
        inputs = model.backbone.wrapper.embedder.process_image(image)
        inputs = inputs.to(device)

        with torch.no_grad():
            embedding, _, logits = model(inputs)

        prediction = torch.argmax(logits, dim=1)

        all_styles.append(style)
        all_genres.append(genre)
        all_embeddings.append(embedding.detach().cpu())
        all_predictions.append(prediction.detach().cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_styles = torch.cat(all_styles, dim=0)
    all_genres = torch.cat(all_genres, dim=0)

    with mlflow.start_run(run_name="Evaluation", parent_run_id=parent_run_id) as run:
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("size", args.size)
        mlflow.log_param("n_samples", args.n_samples)

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

        n_clusters = len(all_predictions.unique())
        n_genres = len(all_genres.unique())
        n_styles = len(all_styles.unique())

        print(f'# Clusters: {n_clusters}')
        print(f'# Genres: {n_genres}')
        print(f'# Styles: {n_styles}')

        if n_clusters == n_styles:
            acc_style = clustering_accuracy(all_styles, all_predictions)
            print(f'Style ACC: {acc_style}')
            mlflow.log_metric("Style ACC", acc_style)

        if n_clusters == n_genres:
            acc_genre = clustering_accuracy(all_genres, all_predictions)
            print(f'Genre ACC: {acc_genre}')
            mlflow.log_metric("Genre ACC", acc_genre)

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

        df = dataset.df.copy()
        df['Cluster'] = all_predictions.numpy().flatten()
        samples = df.groupby('Cluster').apply(lambda x: x.sample(args.n_samples, random_state=args.random_seed))
        samples = samples.reset_index(drop=True)

        fig, axes = plt.subplots(n_clusters, args.n_samples, figsize=(args.n_samples * 3, n_clusters * 3))

        for i in range(n_clusters):
            filenames = samples[samples['Cluster'] == i]['Artwork'].tolist()
            for j, filename in enumerate(filenames):
                ax = axes[i, j]
        
                filepath = os.path.join(args.image_directory_path, filename)

                img = Image.open(filepath)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                if j == 0:
                    ax.set_ylabel(f"Cluster {i}", rotation=0, labelpad=40, va='center')

        plt.tight_layout()
        mlflow.log_figure(fig, "samples.png")
        plt.close(fig)
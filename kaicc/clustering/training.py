import random
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import mlflow

from kaicc.clustering.modules.loss import ContrastiveClusteringLoss
from kaicc.clustering.modules.model import ContrastiveClusteringModel


def train(
    model: ContrastiveClusteringModel,
    dataset: Dataset,
    epochs_count: int,
    learning_rate: float,
    batch_size: int,
    regularization_strength: float,
    temperature_embeddings: float,
    temperature_clusters: float,
    random_seed: int
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    loss_fn = ContrastiveClusteringLoss(
        regularization_strength=regularization_strength,
        temperature_embeddings=temperature_embeddings,
        temperature_clusters=temperature_clusters,
        batch_size=batch_size,
        clusters_count=model.clusters_count
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    n_samples = len(loader)

    model.train()

    for epoch in tqdm(range(epochs_count), desc="Epochs"):
        epoch_loss_total = 0.0
        epoch_loss_embeddings = 0.0
        epoch_loss_clusters = 0.0
        epoch_batch_wise_clusters_entropy = 0.0

        styles = []
        genres = []
        predictions_a = []
        predictions_b = []

        for inputs_a, inputs_b, style, genre in tqdm(loader, desc="Training", leave=False):
            inputs_a, inputs_b = inputs_a.to(device), inputs_b.to(device)

            optimizer.zero_grad()

            emb_a, logits_a, emb_b, logits_b = model.contrastive_forward(inputs_a, inputs_b)
            
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
            styles.append(style)
            genres.append(genre)

        epoch_loss_total_avg = epoch_loss_total / n_samples
        epoch_loss_embeddings_avg = epoch_loss_embeddings / n_samples
        epoch_loss_clusters_avg = epoch_loss_clusters / n_samples
        epoch_batch_wise_clusters_entropy_avg = epoch_batch_wise_clusters_entropy / n_samples

        predictions_a = torch.cat(predictions_a, dim=0).numpy()
        predictions_b = torch.cat(predictions_b, dim=0).numpy()
        styles = torch.cat(styles, dim=0).numpy()
        genres = torch.cat(genres, dim=0).numpy()

        nmi_style_a = normalized_mutual_info_score(styles, predictions_a)
        nmi_style_b = normalized_mutual_info_score(styles, predictions_b)
        nmi_style_avg = (nmi_style_a + nmi_style_b) / 2
        ari_style_a = adjusted_rand_score(styles, predictions_a)
        ari_style_b = adjusted_rand_score(styles, predictions_b)
        ari_style_avg = (ari_style_a + ari_style_b) / 2

        nmi_genre_a = normalized_mutual_info_score(genres, predictions_a)
        nmi_genre_b = normalized_mutual_info_score(genres, predictions_b)
        nmi_genre_avg = (nmi_genre_a + nmi_genre_b) / 2
        ari_genre_a = adjusted_rand_score(genres, predictions_a)
        ari_genre_b = adjusted_rand_score(genres, predictions_b)
        ari_genre_avg = (ari_genre_a + ari_genre_b) / 2

        tqdm.write(f"EPOCH {epoch+1}\n"
            f"Loss: {epoch_loss_total_avg:.4f}\n"
                f"\tEmbeddings: {epoch_loss_embeddings_avg:.4f}\n"
                f"\tClusters: {epoch_loss_clusters_avg:.4f}\n"
                    f"\t\tH[E(p)]: {epoch_batch_wise_clusters_entropy_avg:.4f} / {loss_fn.uniform_clusters_entropy.item():.4f} nats\n"
            f"Metrics:\n"
                f"\tStyle\n"                
                    f"\t\tNormalized Mutual Information (avg. {nmi_style_avg})\n"
                        f"\t\t\tView A: {nmi_style_a:.4f}\n"
                        f"\t\t\tView B: {nmi_style_b:.4f}\n"
                    f"\t\tAdjusted Rand Score (avg. {ari_style_avg})\n"
                        f"\t\t\tView A: {ari_style_a:.4f}\n"
                        f"\t\t\tView B: {ari_style_b:.4f}\n"
                f"\tGenre\n"                
                    f"\t\tNormalized Mutual Information (avg. {nmi_genre_avg})\n"
                        f"\t\t\tView A: {nmi_genre_a:.4f}\n"
                        f"\t\t\tView B: {nmi_genre_b:.4f}\n"
                    f"\t\tAdjusted Rand Score (avg. {ari_genre_avg})\n"
                        f"\t\t\tView A: {ari_genre_a:.4f}\n"
                        f"\t\t\tView B: {ari_genre_b:.4f}\n")

        mlflow.log_metric("Loss Total", epoch_loss_total_avg, step=epoch)
        mlflow.log_metric("Loss Embeddings", epoch_loss_embeddings_avg, step=epoch)
        mlflow.log_metric("Loss Clusters", epoch_loss_clusters_avg, step=epoch)
        mlflow.log_metric("Batch-wise Entropy", epoch_batch_wise_clusters_entropy_avg, step=epoch)

        mlflow.log_metric("Style NMI Avg.", nmi_style_avg, step=epoch)
        mlflow.log_metric("Style NMI View A", nmi_style_a, step=epoch)
        mlflow.log_metric("Style NMI View B", nmi_style_b, step=epoch)

        mlflow.log_metric("Style ARI Avg.", ari_style_avg, step=epoch)
        mlflow.log_metric("Style ARI View A", ari_style_a, step=epoch)
        mlflow.log_metric("Style ARI View B", ari_style_b, step=epoch)

        mlflow.log_metric("Genre NMI Avg.", nmi_genre_avg, step=epoch)
        mlflow.log_metric("Genre NMI View A", nmi_genre_a, step=epoch)
        mlflow.log_metric("Genre NMI View B", nmi_genre_b, step=epoch)

        mlflow.log_metric("Genre ARI Avg.", ari_genre_avg, step=epoch)
        mlflow.log_metric("Genre ARI View A", ari_genre_a, step=epoch)
        mlflow.log_metric("Genre ARI View B", ari_genre_b, step=epoch)

    model = model.to('cpu')
    mlflow.pytorch.log_model(model, model.__name__)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from kaicc.clustering.modules.loss import ContrastiveClusteringLoss
from kaicc.clustering.modules.dataset import ContrastiveDataset
from kaicc.clustering.modules.model import ContrastiveClusteringModel


def train(
    model: ContrastiveClusteringModel,
    dataset: ContrastiveDataset,
    epochs_count: int,
    learning_rate: float,
    batch_size: int,
    regularization_strength: float,
    temperature_embeddings: float,
    temperature_clusters: float
):
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

        labels = []
        predictions_a = []
        predictions_b = []

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

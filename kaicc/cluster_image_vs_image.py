import argparse

from dotenv import load_dotenv
from kaicc.clustering.training import train
from kaicc.clustering.modules.dataset import (
    ArtworkVsArtworkDataset,
    get_augmented_transform
)
from kaicc.clustering.modules.model import (
    ContrastiveClusteringModel,
    CLIPMainVsMainBackbone,
    CLIPImageMainToTextAuxWrapper,
    CLIPEmbedderProjected
)
import mlflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clip_base_model_name', type=str)
    parser.add_argument('--image_directory_path', type=str)
    parser.add_argument('--image_archive_path', type=str)
    parser.add_argument('--labels_file_path', type=str)
    parser.add_argument('--output_run_id_path', type=str)

    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--patience', type=int)
    
    parser.add_argument('--clusters_count', type=int)
    parser.add_argument('--embeddings_dimension', type=int)

    parser.add_argument('--epochs_count', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--regularization_strength', type=float)
    parser.add_argument('--temperature_embeddings', type=float)
    parser.add_argument('--temperature_clusters', type=float)

    parser.add_argument('--size', type=int)
    parser.add_argument('--scale', nargs=2, type=float)
    parser.add_argument('--brightness', type=float)
    parser.add_argument('--contrast', type=float)
    parser.add_argument('--saturation', type=float)
    parser.add_argument('--hue', type=float)
    parser.add_argument('--p_color_jitter', type=float)
    parser.add_argument('--p_gray_scale', type=float)
    parser.add_argument('--p_gaussian_blur', type=float)
    parser.add_argument('--sigma', nargs=2, type=float)

    args = parser.parse_args()

    load_dotenv()

    embedder = CLIPEmbedderProjected(args.clip_base_model_name)
    embedder.freeze()
    embedder.unfreeze_last_vision_layer()
    wrapper = CLIPImageMainToTextAuxWrapper(embedder)
    backbone = CLIPMainVsMainBackbone(wrapper)
    model = ContrastiveClusteringModel(
        backbone,
        args.clusters_count,
        args.embeddings_dimension
    )

    train_transform = get_augmented_transform(
        size=args.size,
        scale=tuple(args.scale),
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        p_color_jitter=args.p_color_jitter,
        p_gray_scale=args.p_gray_scale,
        p_gaussian_blur=args.p_gaussian_blur,
        sigma=tuple(args.sigma),
    )
    dataset = ArtworkVsArtworkDataset(
        embedder.processor,
        args.image_archive_path,
        args.image_directory_path,
        args.labels_file_path,
        train_transform
    )

    with mlflow.start_run(run_name="Image Vs. Image") as run:
        mlflow.log_param("clip_base_model_name", args.clip_base_model_name)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("clusters_count", args.clusters_count)
        mlflow.log_param("embeddings_dimension", args.embeddings_dimension)
        mlflow.log_param("epochs_count", args.epochs_count)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("regularization_strength", args.regularization_strength)
        mlflow.log_param("temperature_embeddings", args.temperature_embeddings)
        mlflow.log_param("temperature_clusters", args.temperature_clusters)
        mlflow.log_param("size", args.size)
        mlflow.log_param("scale", str(args.scale))
        mlflow.log_param("brightness", args.brightness)
        mlflow.log_param("contrast", args.contrast)
        mlflow.log_param("saturation", args.saturation)
        mlflow.log_param("hue", args.hue)
        mlflow.log_param("p_color_jitter", args.p_color_jitter)
        mlflow.log_param("p_gray_scale", args.p_gray_scale)
        mlflow.log_param("p_gaussian_blur", args.p_gaussian_blur)
        mlflow.log_param("sigma", str(args.sigma))

        train(
            model,
            dataset,
            args.epochs_count,
            args.learning_rate,
            args.batch_size,
            args.regularization_strength,
            args.temperature_embeddings,
            args.temperature_clusters,
            args.patience,
            args.random_seed
        )

        with open(args.output_run_id_path, 'w') as file:
            file.write(run.info.run_id)

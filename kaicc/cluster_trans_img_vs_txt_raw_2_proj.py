import argparse
import ast

from dotenv import load_dotenv
from kaicc.clustering.training import train
from kaicc.clustering.modules.dataset import (
    ArtworkVsCaptionDataset,
    get_augmented_transform
)
from kaicc.clustering.modules.model import (
    ContrastiveClusteringModelAux,
    CLIPMainVsAuxBackbone,
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
    parser.add_argument('--captions_file_path', type=str)
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

    parser.add_argument('--freeze_temperature_embeddings', type=ast.literal_eval)
    parser.add_argument('--freeze_temperature_clusters', type=ast.literal_eval)
    parser.add_argument('--random_text_slicing', type=ast.literal_eval)

    args = parser.parse_args()

    load_dotenv()

    embedder = CLIPEmbedderProjected(args.clip_base_model_name, args.random_text_slicing)
    embedder.freeze()
    embedder.unfreeze_last_vision_layer()
    embedder.unfreeze_last_text_layer()
    wrapper = CLIPImageMainToTextAuxWrapper(embedder)
    backbone = CLIPMainVsAuxBackbone(wrapper)
    model = ContrastiveClusteringModelAux(backbone, args.clusters_count, args.embeddings_dimension)

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

    dataset = ArtworkVsCaptionDataset(
        args.image_archive_path,
        args.image_directory_path,
        args.labels_file_path,
        args.captions_file_path,
        train_transform
    )

    with mlflow.start_run(run_name="Transformed Image Vs. Text (Raw, Double Projection)") as run:
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
        mlflow.log_param("freeze_temperature_embeddings", args.freeze_temperature_embeddings)
        mlflow.log_param("freeze_temperature_clusters", args.freeze_temperature_clusters)
        mlflow.log_param("random_text_slicing", args.random_text_slicing)
        mlflow.log_param("captions_file_path", args.captions_file_path)

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
            args.random_seed,
            args.freeze_temperature_embeddings,
            args.freeze_temperature_clusters
        )

        with open(args.output_run_id_path, 'w') as file:
            file.write(run.info.run_id)

import argparse

from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_labels_file_path', type=str)
    parser.add_argument('--output_pretraining_labels_file_path', type=str)
    parser.add_argument('--output_clustering_labels_file_path', type=str)
    parser.add_argument('--pretraining_labels_ratio', type=float)
    parser.add_argument('--random_seed', type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.input_labels_file_path)

    clustering, pretraining = train_test_split(
        df,
        test_size=args.pretraining_labels_ratio,
        random_state=args.random_seed
    )

    clustering.to_csv(args.output_clustering_labels_file_path, index=False)
    pretraining.to_csv(args.output_pretraining_labels_file_path, index=False)

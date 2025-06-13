import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_triples_file_path', type=str)
    parser.add_argument('--output_triples_file_path', type=str)

    args = parser.parse_args()
    
    df = pd.read_csv(args.input_triples_file_path)

    all_labels = pd.unique(df[['subject', 'property']].values.ravel())

    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    df['subject_idx']  = df['subject'].map(label_to_idx)
    df['property_idx'] = df['property'].map(label_to_idx)

    df.to_csv(args.output_triples_file_path, index=False)
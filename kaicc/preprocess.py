import argparse
import tarfile

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_triples_file_path', type=str)
    parser.add_argument('--output_triples_file_path', type=str)
    parser.add_argument('--output_labels_file_path', type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.input_triples_file_path)

    subject_filter = (df['subject_type'] == 'Artwork')
    property_filter = (df['property_type'].isin(['Style', 'Genre']))

    df_labels = df[subject_filter & property_filter]

    df_triples = df.drop(df_labels.index)
    location_filter = (df_triples['relation'].isin(['inCity', 'inCountry', 'locatedIn']))
    df_triples = df_triples[~location_filter]
    

    df_triples = df_triples.reset_index(drop=True)
    
    df_triples['property_type'] = df_triples['property_type'].replace({
        'Training': 'School',
        'People': 'Patron',
        'Serie': 'Series'
    })

    df_triples['relation_idx'] = pd.factorize(df_triples['relation'])[0]

    df_triples.to_csv(args.output_triples_file_path, index=False)

    df_labels = df_labels[['subject', 'property_type', 'property']]
    df_labels = df_labels.pivot_table(index='subject', 
                                      columns='property_type', 
                                      values='property', 
                                      aggfunc='first').reset_index()
    df_labels.rename(columns={'subject': 'Artwork'}, inplace=True)

    df_labels['Style_idx'] = pd.factorize(df_labels['Style'])[0]
    df_labels['Genre_idx'] = pd.factorize(df_labels['Genre'])[0]

    df_labels.to_csv(args.output_labels_file_path, index=False)

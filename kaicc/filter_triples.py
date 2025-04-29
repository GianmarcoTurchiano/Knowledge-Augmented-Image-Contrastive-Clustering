import pandas as pd

def f(pic_filename):
    filter_pics = (df_triples['subject'] == pic_filename)
    df_pics = df_triples[filter_pics]

    filter_artists = (df_pics['property_type'] == 'Artist')
    artists = df_pics.loc[filter_artists, 'property'].unique()

    df_artists = df_triples[df_triples['subject'].isin(artists)]
    df_combined = pd.concat([df_pics, df_artists], ignore_index=True)

    return df_combined

if __name__ == '__main__':
    df_labels = pd.read_csv('data/processed/labels.csv')
    df_triples = pd.read_csv('data/processed/triples.csv')

    for pic_filename in df_labels['subject']:
        df_combined = f(pic_filename)
        print(df_combined)
        print()

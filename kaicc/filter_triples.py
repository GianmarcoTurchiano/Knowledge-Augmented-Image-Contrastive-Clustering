import pandas as pd

def f(df_triples, pic_filename):
    filter_pics = (df_triples['subject'] == pic_filename)
    df_pics = df_triples[filter_pics]

    filter_artists = (df_pics['property_type'] == 'Artist')
    artists = df_pics.loc[filter_artists, 'property'].unique()

    df_artists = df_triples[df_triples['subject'].isin(artists)]

    return df_pics, df_artists

if __name__ == '__main__':
    df_labels = pd.read_csv('data/processed/labels.csv')
    df_triples = pd.read_csv('data/processed/triples.csv')
    df_texts = pd.read_csv('data/processed/node_texts.csv')

    _, relations_unique = pd.factorize(df_triples['relation'])
    relations_to_idx = {rel: i for i, rel in enumerate(relations_unique)}

    for pic_filename in df_labels['subject']:
        df_pics, df_artists = f(df_triples, pic_filename)

        for _, row in df_pics.iterrows():
            relation = row['relation']
            relation_idx = relations_to_idx[relation]

            subject = row['subject']
            property = row['property']

            subject_filter = df_texts['property'] == subject
            property_filter = df_texts['property'] == property

            subject_idx = df_texts.index[subject_filter]
            property_idx = df_texts.index[property_filter]

            print(property)
            print(property_idx)

            if not subject_idx.empty:
                print(subject)
                print(subject_idx)

        break

        #df_combined = pd.concat([df_pics, df_artists], ignore_index=True)
        #print(df_combined)
        #print()


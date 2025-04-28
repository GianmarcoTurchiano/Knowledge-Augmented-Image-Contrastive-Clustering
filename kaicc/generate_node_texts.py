import argparse
import pandas as pd
import re

def generate_property_string(row):
    property_value = row['property']
    property_type = row['property_type']
    subject_type = row['subject_type']
    relation = row['relation']
    relation_tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', relation).split()

    if relation_tokens[-1] == property_type:
        relation_tokens.pop(-1)

    if relation_tokens[0] in ['has', 'belongs']:
        relation = 'related to'
    else:
        relation = ' '.join(relation_tokens)

    article = 'an' if property_type.startswith('A') else 'a'

    result = f"{property_value} is {article} {property_type} that {subject_type}s can be {relation}"
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_triples_file', type=str)
    parser.add_argument('--output_node_texts_file', type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.input_triples_file)
    df = df.drop('subject', axis=1)
    df = df.drop_duplicates()
    df['property_txt'] = df.apply(generate_property_string, axis=1)

    df = df.drop(['relation', 'subject_type', 'property_type', 'relation_idx'], axis=1)
    df.to_csv(args.output_node_texts_file, index=False)

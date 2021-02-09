import pandas as pd

from predict_sp.settings import FEATURES_AMINO_ACIDS, FEATURE_LENGTH, \
    FEATURES_NAMES


def read_fasta(file):
    seqs = []
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                header = line.strip()
                classification = header.split("|")[2]
                seq = lines[i + 1].strip()[:FEATURE_LENGTH]
                seqs.append((seq, classification))
    return seqs


def one_hot_encode(aa):
    feature = [0] * len(FEATURES_AMINO_ACIDS)
    if not aa:
        return feature

    feature[FEATURES_AMINO_ACIDS[aa]] = 1
    return feature


def init_columns(seqs):
    df = pd.DataFrame(seqs, columns=["Seq", "Label"])
    for feature_name in FEATURES_NAMES:
        df[feature_name] = ""
    return df


def create_df(seqs):
    all_features = {feature_name: [] for feature_name in FEATURES_NAMES}
    df = init_columns(seqs)
    for _, row in df.iterrows():
        seq = row.Seq
        for i in range(FEATURE_LENGTH):
            try:
                aa = seq[i]
            except IndexError:
                aa = ""
            feature = one_hot_encode(aa)
            all_features[FEATURES_NAMES[i]].append(feature)

    for feature_name in FEATURES_NAMES:
        df[feature_name] = all_features[feature_name]

    return df


if __name__ == "__main__":
    seqs = read_fasta("./input/train_set.fasta")
    df = create_df(seqs)
    print(df.head())
    df.to_csv("./output/test.tsv", sep="\t")

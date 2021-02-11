import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics

from predict_sp.settings import FEATURES_AMINO_ACIDS, FEATURE_LENGTH


# TODO: make nice class so different classifiers can be used

def read_fasta(file):
    print("Reading fasta file")
    seqs = []
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                header = line.strip()
                classification = header.split("|")[2]
                seq = lines[i + 1].strip()[:FEATURE_LENGTH]

                # only get the sequences that meet the length requirement
                if len(seq) == FEATURE_LENGTH:
                    seqs.append((seq, classification))
    return seqs


def one_hot_encode(aa):
    feature = [0] * len(FEATURES_AMINO_ACIDS)
    if not aa:
        return np.array(feature)

    feature[FEATURES_AMINO_ACIDS[aa]] = 1
    return np.array(feature)


def get_data(seqs):
    print("Getting the data")
    features = []
    labels = []
    for info in seqs:
        seq = info[0]
        labels.append(info[1])
        feature = []
        if len(seq) >= FEATURE_LENGTH:
            for i in range(FEATURE_LENGTH):
                feature.extend(one_hot_encode(seq[i]))

            features.append(feature)
        else:
            print(len(seq))

    return features, labels


def save_classifier(svc):
    print("Saving classifier")
    file_name = f"./output/classifier_{FEATURE_LENGTH}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(svc, f)
    return file_name


def load_classifier(file_name):
    print("Loading classifier")
    with open(file_name, "rb") as f:
        classifier = pickle.load(f)
    return classifier


def train(x, y):
    print("Training the model")
    svc = svm.SVC(kernel="linear")
    svc.fit(x, y)
    return svc


def predict(svc, x):
    print("Predicting")
    return svc.predict(x)


def print_performance(svc, features_benchmark, predicted_labels,
                      actual_labels):
    score = svc.score(features_benchmark, actual_labels)
    print("=" * 60)
    print("\nScore:", score)

    print("\nResult overview:\n")
    classification_report = metrics.classification_report(actual_labels,
                                                          predicted_labels)
    print(classification_report)
    with open(f"./output/classification_report_{FEATURE_LENGTH}.txt",
              "w") as f:
        f.write(classification_report)

    disp = metrics.plot_confusion_matrix(svc, features_benchmark,
                                         actual_labels)

    print("\nConfusion matrix:\n")
    print(disp.confusion_matrix)

    disp.ax_.set_title(
        f"Confusion matrix with sequence length {FEATURE_LENGTH}")
    plt.savefig(f"./output/confusion_matrix_{FEATURE_LENGTH}.png",
                bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # seqs_train = read_fasta("./input/train_set.fasta")
    # features_train, labels_train = get_data(seqs_train)
    # svc = train(features_train, labels_train)
    #
    # file_name = save_classifier(svc)

    file_name = f"./output/classifier_{FEATURE_LENGTH}.pkl"
    svc = load_classifier(file_name)

    seqs_bechmark = read_fasta("./input/benchmark_set.fasta")
    features_benchmark, labels_benchmark = get_data(seqs_bechmark)
    predictions = predict(svc, features_benchmark)

    print_performance(svc, features_benchmark, predictions, labels_benchmark)

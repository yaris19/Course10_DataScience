import argparse
import logging
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from prediction.classifier import Classifier
from prediction.custom_formatter import CustomFormatter
from prediction.keras_classifier import KerasClassifier
from prediction.settings import LOGGING_LEVEL, POSITIONS_AMINO_ACIDS, \
    FEATURES_AMINO_ACIDS, FEATURES_AMINO_ACIDS_LENGTH

parser = argparse.ArgumentParser(
    description="Predict singnal peptides for protein sequences")
parser.add_argument("--train-set", dest="train_set",
                    type=str, required=True,
                    help="path to training data set")
parser.add_argument("--out-dir", dest="out_dir", type=str, required=True,
                    help="path to output dir")

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

CLASSIFIERS = {
    # "svc": svm.SVC(kernel="linear"),
    "rfc": RandomForestClassifier(n_jobs=-1,
                                  random_state=0,
                                  max_depth=2),
    "knc": KNeighborsClassifier(n_neighbors=3,
                                n_jobs=-1),
    "gnb": GaussianNB()
}

FEATURES_LENGTHS = [
    30,
    50
]


def parse_args(parser):
    args = parser.parse_args()

    if not os.path.exists(args.train_set):
        raise FileNotFoundError(f"{args.train_set} does not exist")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        logger.info(f"Created directory: {args.out_dir}")

    return args


def read_fasta(file, feature_length):
    logger.debug("Reading fasta file")
    seqs = []
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                header = line.strip()
                classification = header.split("|")[2]
                if classification in ["NO_SP", "SP"]:
                    seq = lines[i + 1].strip()[:feature_length]

                    # only get the sequences that meet the length requirement
                    if len(seq) == feature_length:
                        seqs.append((seq, classification))
    return seqs


def one_hot_encode_position(aa):
    feature = [0] * len(POSITIONS_AMINO_ACIDS)
    if not aa:
        return np.array(feature)

    feature[POSITIONS_AMINO_ACIDS[aa]] = 1
    return np.array(feature)


def one_hot_encode_feature(aa):
    feature = [0] * FEATURES_AMINO_ACIDS_LENGTH
    if not aa:
        return np.array(feature)

    feature[FEATURES_AMINO_ACIDS[aa]] = 1
    return np.array(feature)


def get_data(seqs, feature_length, classify_features):
    logger.debug("Getting the data")
    features = []
    labels = []
    for info in seqs:
        seq = info[0]
        labels.append(info[1])
        feature = []
        if len(seq) >= feature_length:
            for i in range(feature_length):
                if classify_features:
                    feature.extend(one_hot_encode_feature(seq[i]))
                else:
                    feature.extend(one_hot_encode_position(seq[i]))
            features.append(feature)

    return features, labels


def get_classifier(features_train, labels_train, features_benchmark,
                   labels_benchmark, out_dir, cfr, feature_length,
                   classify_features=False):
    return Classifier(features_train, labels_train,
                      features_benchmark,
                      labels_benchmark, out_dir,
                      CLASSIFIERS.get(cfr.lower()),
                      classify_features,
                      feature_length=feature_length)


def get_keras_classifier(features_train, labels_train, features_benchmark,
                         labels_benchmark, out_dir, feature_length,
                         classify_features=False):
    return KerasClassifier(features_train, labels_train,
                           features_benchmark,
                           labels_benchmark, out_dir,
                           classify_features,
                           feature_length=feature_length)


def models_positions(input_data, out_dir):
    for cfr in CLASSIFIERS.keys():
        for feature_length in FEATURES_LENGTHS:
            classifier = get_classifier(input_data[feature_length]["False"][0],
                                        input_data[feature_length]["False"][2],
                                        input_data[feature_length]["False"][1],
                                        input_data[feature_length]["False"][3],
                                        out_dir, cfr, feature_length)

            classifier.train()
            classifier.save_classifier()
            classifier.predict()

            classifier.print_performance_and_save()
            classifier.plot_confusion_matrix_and_save(plot_for="train",
                                                      plot_img=False)
            classifier.plot_confusion_matrix_and_save(plot_for="test",
                                                      plot_img=False)

    for feature_length in FEATURES_LENGTHS:
        keras_classifier = get_keras_classifier(
            input_data[feature_length]["False"][0],
            input_data[feature_length]["False"][2],
            input_data[feature_length]["False"][1],
            input_data[feature_length]["False"][3],
            out_dir, feature_length,
            classify_features=False)

        keras_classifier.train()
        keras_classifier.save_classifier()
        keras_classifier.predict()

        keras_classifier.save_performance()
        keras_classifier.plot_confusion_matrix_and_save(plot_img=False)


def model_features(input_data, out_dir):
    for cfr in CLASSIFIERS.keys():
        for feature_length in FEATURES_LENGTHS:
            classifier = get_classifier(input_data[feature_length]["True"][0],
                                        input_data[feature_length]["True"][2],
                                        input_data[feature_length]["True"][1],
                                        input_data[feature_length]["True"][3],
                                        out_dir, cfr, feature_length,
                                        classify_features=True)

            classifier.train()
            classifier.save_classifier()
            classifier.predict()
            classifier.print_performance_and_save()

            classifier.plot_confusion_matrix_and_save(plot_for="train",
                                                      plot_img=False)
            classifier.plot_confusion_matrix_and_save(plot_for="test",
                                                      plot_img=False)

    for feature_length in FEATURES_LENGTHS:
        keras_classifier = get_keras_classifier(
            input_data[feature_length]["True"][0],
            input_data[feature_length]["True"][2],
            input_data[feature_length]["True"][1],
            input_data[feature_length]["True"][3],
            out_dir, feature_length,
            classify_features=True)

        keras_classifier.train()
        keras_classifier.save_classifier()
        keras_classifier.predict()

        keras_classifier.save_performance()
        keras_classifier.plot_confusion_matrix_and_save(plot_img=False)


def main():
    args = parse_args(parser)

    input_data = {}
    for feature_length in FEATURES_LENGTHS:
        input_data[feature_length] = {}
        for boolean in [False, True]:
            seqs_train = read_fasta(args.train_set, feature_length)
            features, labels = get_data(seqs_train, feature_length,
                                        boolean)

            features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size=.2)
            input_data[feature_length][str(boolean)] = [features_train,
                                                        features_test,
                                                        labels_train,
                                                        labels_test]

    # use positions
    models_positions(input_data, args.out_dir)

    # use features of amino acids
    model_features(input_data, args.out_dir)


if __name__ == "__main__":
    main()

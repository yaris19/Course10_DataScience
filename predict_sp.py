import argparse
import logging
import os

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from prediction.classifier import Classifier
from prediction.custom_formatter import CustomFormatter
from prediction.settings import LOGGING_LEVEL, POSITIONS_AMINO_ACIDS, \
    FEATURES_AMINO_ACIDS, FEATURES_AMINO_ACIDS_LENGTH

parser = argparse.ArgumentParser(
    description="Predict singnal peptides for protein sequences")
parser.add_argument("--train-set", dest="train_set",
                    type=str, required=True,
                    help="path to training data set")
parser.add_argument("--out-dir", dest="out_dir", type=str, required=True,
                    help="path to output dir")
parser.add_argument("--classifier", dest="classifier", type=str, required=True,
                    help="sklearn classifier")
parser.add_argument("--trained-model", dest="trained_model",
                    type=str,
                    help="path to a trained model set")
parser.add_argument("--number-features", dest="feature_length", required=True,
                    type=int, help="number of features that need to be used")
parser.add_argument("--classify-features", dest="classify_features",
                    action="store_true", default=False,
                    help="train based on the features of an amino acid")

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

CLASSIFIERS = {
    "svc": svm.SVC(kernel="linear"),
    "rfc": RandomForestClassifier(n_jobs=-1,
                                  random_state=0,
                                  max_depth=2),
    "knc": KNeighborsClassifier(n_neighbors=3,
                                n_jobs=-1),
    "gnb": GaussianNB()
}

AVAIL_CLASSIFIERS = [
    "'svc' (C-Support Vector Classification)",
    "'rfc' (RandomForestClassifier)",
    "'knc' (KNeighborsClassifier)",
    "'gnb' (GaussianNB)"
]


def parse_args(parser):
    args = parser.parse_args()

    if not os.path.exists(args.train_set):
        raise FileNotFoundError(f"{args.train_set} does not exist")

    if not args.classifier and not os.path.exists(args.trained_model):
        raise FileNotFoundError(f"{args.trained_model} does not exist")

    if not args.trained_model and args.classifier.lower() not in CLASSIFIERS:
        avail_classifiers = ', '.join(AVAIL_CLASSIFIERS)
        raise ValueError(
            f"Choose one of the following classifiers: {avail_classifiers}")

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
                   labels_benchmark, args):
    return Classifier(features_train, labels_train,
                      features_benchmark,
                      labels_benchmark, args.out_dir,
                      CLASSIFIERS.get(args.classifier.lower()),
                      args.classify_features,
                      feature_length=args.feature_length)


if __name__ == "__main__":
    args = parse_args(parser)

    seqs_train = read_fasta(args.train_set, args.feature_length)
    features, labels = get_data(seqs_train, args.feature_length,
                                args.classify_features)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=.2)

    classifier = get_classifier(features_train, labels_train,
                                features_test, labels_test, args)

    if args.trained_model:
        classifier.load_classifier(args.trained_model)
    else:
        classifier.train()
        classifier.save_classifier()

    classifier.predict()
    classifier.print_performance_and_save()
    classifier.plot_confusion_matrix_and_save(plot_for="train")
    classifier.plot_confusion_matrix_and_save(plot_for="test")

import argparse
import logging
import os

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from prediction.classifier import Classifier
from prediction.custom_formatter import CustomFormatter
from prediction.settings import FEATURES_AMINO_ACIDS

parser = argparse.ArgumentParser(
    description="Predict singnal peptides on protein sequences")
parser.add_argument("--train-set", dest="train_set",
                    type=str, required=True,
                    help="path to training data set")
parser.add_argument("--test-set", dest="test_set", type=str,
                    required=True, help="path to test data set")
parser.add_argument("--out-dir", dest="out_dir", type=str, required=True,
                    help="path to output dir")
parser.add_argument("--classifier", dest="classifier", type=str, required=True,
                    help="sklearn classifier")
parser.add_argument("--trained-model", dest="trained_model",
                    type=str,
                    help="path to a trained model set")
parser.add_argument("--number-features", dest="feature_length", required=True,
                    type=int, help="number of features that need to be used")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

CLASSIFIERS = {
    "svc": svm.SVC(kernel="linear"),
    "randomforestclassifier": RandomForestClassifier(n_jobs=-1,
                                                     random_state=0,
                                                     max_depth=2),
    "rfc": RandomForestClassifier(n_jobs=-1,
                                  random_state=0,
                                  max_depth=2)
}


def parse_args(parser):
    args = parser.parse_args()

    if not os.path.exists(args.train_set):
        raise FileNotFoundError(f"{args.train_set} does not exist")

    if not os.path.exists(args.test_set):
        raise FileNotFoundError(f"{args.test_set} does not exist")

    if not args.classifier and not os.path.exists(args.trained_model):
        raise FileNotFoundError(f"{args.trained_model} does not exist")

    if not args.trained_model and args.classifier.lower() not in CLASSIFIERS:
        avail_classifiers = ', '.join(list(CLASSIFIERS.keys()))
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
                seq = lines[i + 1].strip()[:feature_length]

                # only get the sequences that meet the length requirement
                if len(seq) == feature_length:
                    seqs.append((seq, classification))
    return seqs


def one_hot_encode(aa):
    feature = [0] * len(FEATURES_AMINO_ACIDS)
    if not aa:
        return np.array(feature)

    feature[FEATURES_AMINO_ACIDS[aa]] = 1
    return np.array(feature)


def get_data(seqs, feature_length):
    logger.debug("Getting the data")
    features = []
    labels = []
    for info in seqs:
        seq = info[0]
        labels.append(info[1])
        feature = []
        if len(seq) >= feature_length:
            for i in range(feature_length):
                feature.extend(one_hot_encode(seq[i]))
            features.append(feature)

    return features, labels


def get_classifier(features_train, labels_train, features_benchmark,
                   labels_benchmark, args):
    if args.feature_length:
        classifier = Classifier(features_train, labels_train,
                                features_benchmark,
                                labels_benchmark, args.out_dir,
                                CLASSIFIERS.get(args.classifier.lower()),
                                feature_length=args.feature_length)
    else:
        classifier = Classifier(features_train, labels_train,
                                features_benchmark,
                                labels_benchmark, args.out_dir,
                                CLASSIFIERS.get(args.classifer.lower()))
    return classifier


if __name__ == "__main__":
    args = parse_args(parser)

    seqs_train = read_fasta(args.train_set, args.feature_length)
    features_train, labels_train = get_data(seqs_train, args.feature_length)
    seqs_bechmark = read_fasta(args.test_set, args.feature_length)
    features_benchmark, labels_benchmark = get_data(seqs_bechmark,
                                                    args.feature_length)

    classifier = get_classifier(features_train, labels_train,
                                features_benchmark, labels_benchmark, args)

    if args.trained_model:
        classifier.load_classifier(args.trained_model)
    else:
        classifier.train()

    classifier.predict()
    classifier.print_performance_and_save()
    classifier.plot_confusion_matrix_and_save()

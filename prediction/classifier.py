import logging
import os
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics

from .custom_formatter import CustomFormatter
from .settings import LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class Classifier:

    def __init__(self, train_data, train_labels, test_data, test_labels,
                 out_dir, classifier, classify_features, feature_length=50):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.classify_features = classify_features
        self.classifier = classifier
        self.feature_length = feature_length

        self.predictions = None
        self.classification_report = None

        self.subfolder = "features" if self.classify_features else "positions"

        self.classifier_name = self.classifier.__class__.__name__
        self.out_dir = os.path.join(out_dir,
                                    self.subfolder,
                                    self.classifier_name)
        self.models_dir = os.path.join("models",
                                       self.subfolder,
                                       self.classifier_name)

        self.init_dirs()

        logger.info(f"Using classifier {self.classifier_name}")

    def init_dirs(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train(self):
        logger.debug("Training the model")
        nr_samples = len(self.get_train_labels())
        logger.info(f"Number of samples in the training dataset: {nr_samples}")
        self.classifier.fit(self.train_data, self.train_labels)

    def predict(self):
        logger.debug("Predicting")
        nr_samples = len(self.get_test_labels())
        logger.info(f"Number of samples in the testing dataset: {nr_samples}")
        self.predictions = self.classifier.predict(self.test_data)

    def print_performance_and_save(self):
        logger.debug("Calculating performance")
        score = self.classifier.score(self.test_data, self.test_labels)
        logger.info(f"Score: {score}")

        self.classification_report = metrics.classification_report(
            self.test_labels,
            self.predictions,
            zero_division=0)

        logger.info("Classification report:")
        logger.info(f"\n{self.classification_report}")
        self.__save_classification_report()

    def plot_confusion_matrix_and_save(self):
        logger.debug("Calculating confusion matrix")
        disp = metrics.plot_confusion_matrix(self.classifier,
                                             self.test_data,
                                             self.test_labels)
        logger.info("Confusion matrix:")
        logger.info(f"\n{disp.confusion_matrix}")
        disp.ax_.set_title(
            f"Confusion matrix with sequence length {self.feature_length}")
        plt.savefig(os.path.join(
            self.out_dir,
            f"confusion_matrix_{self.feature_length}.png"),
            bbox_inches="tight")
        plt.show()

    def __save_classification_report(self):
        with open(os.path.join(
                self.out_dir,
                f"classification_report_{self.feature_length}.txt"),
                "w") as f:
            f.write(self.classification_report)

    def save_classifier(self):
        file_name = os.path.join(self.models_dir,
                                 f"classifier_{self.feature_length}.pkl")
        logger.info(f"Saving classifier {file_name}")
        with open(file_name, "wb") as f:
            pickle.dump(self.classifier, f)

    def load_classifier(self, file_name):
        logger.info(f"Loading classifier {file_name}")
        with open(file_name, "rb") as f:
            self.classifier = pickle.load(f)

    def get_predictions(self):
        return self.predictions

    def get_train_data(self):
        return self.train_data

    def get_train_labels(self):
        return self.train_labels

    def get_test_data(self):
        return self.test_data

    def get_test_labels(self):
        return self.test_labels

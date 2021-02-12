import logging
import os
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics

from .custom_formatter import CustomFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class Classifier:

    def __init__(self, train_data, train_labels, test_data, test_labels,
                 out_dir, classifier=None, feature_length=50):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.feature_length = feature_length
        self.classifier = classifier
        self.predictions = None
        self.classification_report = None
        self.name = self.classifier.__class__.__name__
        self.out_dir = os.path.join(out_dir, self.name)
        self.init_out_dir()

    def init_out_dir(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def train(self):
        logger.debug("Training the model")
        self.classifier.fit(self.train_data, self.train_labels)

    def predict(self):
        logger.debug("Predicting")
        self.predictions = self.classifier.predict(self.test_data)

    def print_performance_and_save(self):
        logger.debug("Calculating performance")
        score = self.classifier.score(self.test_data, self.test_labels)
        logger.info(f"Score: {score}")

        self.classification_report = metrics.classification_report(
            self.test_labels,
            self.predictions)
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
        logger.debug("Saving classifier")
        file_name = f"{self.out_dir}/classifier_{self.feature_length}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.classifier, f)

    def load_classifier(self, file_name):
        logger.debug("Loading classifier")
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
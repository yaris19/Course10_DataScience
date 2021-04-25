import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from .custom_formatter import CustomFormatter
from .settings import LOGGING_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)

ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


def create_model(features_len):
    model = Sequential()
    model.add(Input(shape=(features_len,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    return model


class KerasClassifier:

    def __init__(self, train_data, train_labels, test_data, test_labels,
                 out_dir, classify_features, feature_length=50):
        self.train_data = np.asarray(train_data)
        self.train_labels = np.asarray(
            [1 if label == "SP" else 0 for label in train_labels])
        self.test_data = np.asarray(test_data)
        self.test_labels = np.asarray(
            [1 if label == "SP" else 0 for label in test_labels])
        self.classify_features = classify_features
        self.model = create_model(len(self.train_data[0]))
        self.feature_length = feature_length

        self.predictions = None
        self.classification_report = None
        self.history = None

        self.subfolder = "features" if self.classify_features else "positions"

        self.classifier_name = "KerasClassifier"
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
        self.history = self.model.fit(self.train_data, self.train_labels,
                                      epochs=5,
                                      batch_size=128, shuffle=True,
                                      validation_data=(
                                          self.test_data, self.test_labels))

    def predict(self):
        logger.debug("Predicting")
        nr_samples = len(self.get_test_labels())
        logger.info(f"Number of samples in the testing dataset: {nr_samples}")
        self.predictions = self.model.predict(self.test_data)

    def save_performance(self):
        logger.debug("Saving classification report")
        with open(os.path.join(
                self.out_dir,
                f"classification_report_{self.feature_length}.json"),
                "w") as f:
            json.dump(self.history.history, f, ensure_ascii=False, indent=4)

    def plot_confusion_matrix_and_save(self, plot_img=True):
        conf_matrix = confusion_matrix(self.test_labels,
                                       [np.argmax(x) for x in
                                        self.predictions])
        sns.heatmap(conf_matrix, annot=True, cmap="rocket", fmt='d',
                    xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
        plt.title("Confusion matrix for test set")

        plt.savefig(os.path.join(
            self.out_dir,
            f"confusion_matrix_{self.feature_length}_keras.png"),
            bbox_inches="tight")
        if plot_img:
            plt.show()
        plt.clf()

    def save_classifier(self):
        file_name = os.path.join(self.models_dir,
                                 f"classifier_{self.feature_length}.h5")
        logger.info(f"Saving classifier {file_name}")
        self.model.save(file_name)

    def load_classifier(self, file_name):
        logger.info(f"Loading classifier {file_name}")
        self.model = load_model(file_name)

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

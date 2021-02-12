# Course 10 Data science code
Download input dataset: http://www.cbs.dtu.dk/services/SignalP/data.php

## Classifiers
The arguments used for the classifiers:
* [```SVC(kernel="linear")```](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [```RandomForestClassifier(n_jobs=-1, random_state=0, max_depth=2)```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


### Usage
```bash
usage: predict_sp.py [-h] --train-set TRAIN_SET --test-set TEST_SET --out-dir OUT_DIR --classifier CLASSIFIER [--trained-model TRAINED_MODEL]
                     --number-features FEATURE_LENGTH

Predict singnal peptides for protein sequences

optional arguments:
  -h, --help                        show this help message and exit
  --train-set TRAIN_SET             path to training data set
  --test-set TEST_SET               path to test data set
  --out-dir OUT_DIR                 path to output dir
  --classifier CLASSIFIER           sklearn classifier
  --trained-model TRAINED_MODEL     path to a trained model set
  --number-features FEATURE_LENGTH  number of features that need to be used
```
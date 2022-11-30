import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sys


RELEASE = False
# RELEASE = True
if RELEASE:
    # for kaggle testing
    path = "/kaggle/input/heart-desease/{}.csv"
else:
    # for testing in your PC
    path = "{}.csv"


def createSubmission(prediction):
    submission = pd.read_csv(path.format("sample"))
    submission['target'] = prediction
    submission.to_csv('submission.csv', index=False)


def pandasOutputInit():

    pd.options.display.expand_frame_repr = False
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None


def main():
    if not RELEASE:
        # output config
        sys.stdout = open('result', 'w')
        pandasOutputInit()

    # data import
    train_data = pd.read_csv(path.format("train"))
    test_data = pd.read_csv(path.format("test"))
    train_data_features, train_data_target = train_data.drop('target', axis=1), train_data['target']

    if not RELEASE:
        print(test_data)
        print(train_data)

    # model creation & training
    model = KNeighborsClassifier(n_neighbors=choose_by_your_own, weights='uniform', p=2)
    model.fit(train_data_features, train_data_target)


    # model result evaluation
    train_prediction = model.predict(train_data_features)
    print(recall_score(train_prediction, train_data_target))
    test_prediction = model.predict(test_data)
    if RELEASE:
        createSubmission(test_prediction)


if __name__ == '__main__':
    main()

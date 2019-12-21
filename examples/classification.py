"""
LDR - Classification Example
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KernelDensity

from ldr import LDR


class Classification:
    """
    LDR for classification example.
    """

    def __init__(self):
        self.load_data()
        self.preprocess()
        self.train_rf_classifier()
        self.train_if_classifier()

    def load_data(self):
        """
        Loads data in a pandas dataframe and targets into a pandas series.
        """
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["species"] = pd.Categorical.from_codes(
            data.target, data.target_names)
        self.targets = df["species"]
        self.df = df.drop(["species"], axis=1)

    def preprocess(self):
        """
        Scales data using LDR, then train/test splits.
        """
        self.ldr = LDR(self.df, self.targets)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.ldr.scaled, self.ldr.targets, test_size=0.3, random_state=42)

    def train_rf_classifier(self):
        """
        Trains a class classifier using random forest method.
        """
        self.rf_clf = RandomForestClassifier(
            n_estimators=100, random_state=42).fit(self.X_train, self.y_train)

    def rf_clf_func(self, df):
        """
        Converts RF classifier output to single numpy array of values.
        """
        return np.array([i[1] for i in self.rf_clf.predict_proba(df)])

    def train_if_classifier(self):
        """
        Trains an isolation forest for outlier detection.
        """
        self.if_clf = IsolationForest(
            n_estimators=100, behaviour="new", contamination="auto").fit(self.X_train)

    def if_clf_func(self, df):
        """
        Converts IF classifier output to single numpy array of values.
        """
        return [(i + 1) / 2 for i in self.if_clf.predict(df)]

    def oc_rf_clf_func(self, df):
        """
        Interpolates class certainty and outlier certainty.
        """
        oc_pred = self.if_clf_func(df)
        rf_pred = self.rf_clf_func(df)
        return [(rf_pred[i] - 0.5) * i_val + 0.5 for i, i_val in enumerate(oc_pred)]


if __name__ == "__main__":
    Classification()
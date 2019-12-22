# -*- coding: utf-8 -*-
import pytest
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from ldr import LDR


class TestEntry:
    """
    Tests module entry on different datasets.
    """
    @pytest.fixture(scope="module")
    def breast_cancer_data(self):
        """
        Loads the Wisconsin breast cancer dataset as a toy problem.
        """
        data = load_breast_cancer()
        data_df = pd.DataFrame(data.data, columns=data.feature_names)
        data_df["species"] = pd.Categorical.from_codes(data.target,
                                                       data.target_names)
        targets = data_df["species"]
        data_df = data_df.drop(["species"], axis=1)
        return data_df, targets

    def test_ldr_class_exception(self, breast_cancer_data):
        """
        Checks exception is raised if class order items do not match targets.

        If class order does not match targets then binning is impossible.
        """
        data_df, targets = breast_cancer_data
        with pytest.raises(Exception):
            # Actual classes are 'benign' and 'malignant'.
            LDR(data_df, targets, class_order=["hippo", "giraffe", "lion"])

    def test_ldr_breast_cancer(self, breast_cancer_data):
        """
        Tests that LDR works on the breast cancer dataset.
        """
        data_df, targets = breast_cancer_data
        ldr = LDR(data_df, targets)

        # Random forest works well as a basic classifier.
        x_train, x_test, y_train, y_test = train_test_split(
            ldr.scaled, ldr.targets, test_size=0.3, random_state=42)

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(x_train, y_train)
        preds = rf_clf.predict(x_test)

        # sanity check that F1 score is above 0.9.
        assert f1_score(y_test, preds, pos_label="malignant") > 0.9

        # Fewer samples used to prevent longer run time of test.
        ldr.density_estimate(rf_clf.predict_proba, rf_clf.classes_,
                             n_samples=10000)

        path = os.path.join(os.path.dirname(__file__), "output",
                            "breast_cancer_1d.png")
        ldr.vis_1d(save=path,
                   features=["mean area", "area error"])

        path = os.path.join(os.path.dirname(__file__), "output",
                            "breast_cancer_2d.png")
        ldr.vis_2d(save=path,
                   features=["mean area", "area error"])

    @pytest.fixture(scope="module")
    def wine_data(self):
        """
        Loads the wine dataset as a toy problem.

        Dataset found here:
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.
        load_wine.html#sklearn.datasets.load_wine
        """
        data = load_wine()
        data_df = pd.DataFrame(data.data, columns=data.feature_names)
        data_df["species"] = pd.Categorical.from_codes(data.target,
                                                       data.target_names)
        targets = data_df["species"]
        data_df = data_df.drop(["species"], axis=1)
        data_df = data_df[data_df.columns[:4]]
        return data_df, targets

    def test_ldr_wine(self, wine_data):
        """
        Tests that LDR works on the wine dataset.
        """
        data_df, targets = wine_data
        ldr = LDR(data_df, targets)

        # Random forest works well as a basic classifier.
        x_train, x_test, y_train, y_test = train_test_split(
            ldr.scaled, ldr.targets, test_size=0.3, random_state=42)

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(x_train, y_train)
        preds = rf_clf.predict(x_test)

        # sanity check that F1 score is above 0.9.
        assert f1_score(y_test, preds, average="weighted") > 0.8

        # Fewer samples than default used to prevent longer run time of test.
        ldr.density_estimate(rf_clf.predict_proba, rf_clf.classes_,
                             n_samples=10000)

        path = os.path.join(os.path.dirname(__file__), "output", "wine_1d.png")
        ldr.vis_1d(title="Wine Classifier Certainty",
                   save=path)

        path = os.path.join(os.path.dirname(__file__), "output", "wine_2d.png")
        ldr.vis_2d(title="Wine Classifier Certainty Matrix",
                   save=path)

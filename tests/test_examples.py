# -*- coding: utf-8 -*-
import pytest
from sklearn.metrics import f1_score, confusion_matrix

from ldr import LDR
from examples.classification import Classification


class TestExamples:
    """
    Tests the classification example.
    """

    @pytest.fixture(scope="module")
    def cfn(self):
        return Classification()

    def test_classifiers_are_accurate(self, cfn):
        rf_preds = cfn.rf_clf.predict(cfn.X_test)
        score = f1_score(cfn.y_test, rf_preds, pos_label="malignant")
        assert(score > 0.9)

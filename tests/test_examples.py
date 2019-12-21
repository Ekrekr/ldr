# -*- coding: utf-8 -*-
import os
import pytest
from sklearn.metrics import f1_score, confusion_matrix

from ldr import LDR
from examples.classification import Classification


class TestExamples:
    """
    Tests the classification example.
    """

    @pytest.fixture(scope="module")
    def output_path(self):
        return "examples/output"

    @pytest.fixture(scope="module")
    def cfn(self, output_path):
        return Classification(output_path=output_path)

    def test_classifiers_are_accurate(self, cfn):
        rf_preds = cfn.rf_clf.predict(cfn.X_test)
        score = f1_score(cfn.y_test, rf_preds, pos_label="malignant")
        assert(score > 0.9)

    def test_correct_files_outputted(self, output_path):
        files_that_should_exist = [
            "classification_example_vis_1d.png",
            "classification_example_vis_2d.png"
        ]
        for f in files_that_should_exist:
            assert(os.path.exists(os.path.join(output_path, f)))

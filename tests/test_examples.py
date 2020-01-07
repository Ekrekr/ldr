# -*- coding: utf-8 -*-
import os
import pytest
from sklearn.metrics import f1_score

from ldr import LDR


class TestExamples:
    """
    Tests the classification example.
    """

    def test_classifiers_are_accurate(self, classification_example):
        rf_preds = classification_example.rf_clf.predict(
            classification_example.X_test)
        score = f1_score(classification_example.y_test,
                         rf_preds, pos_label="malignant")
        assert(score > 0.9)

    def test_correct_files_outputted(self, examples_output_path):
        files_that_should_exist = [
            "classification_example_vis_1d.png",
            "classification_example_vis_2d.png"
        ]
        for f in files_that_should_exist:
            assert(os.path.exists(os.path.join(examples_output_path, f)))

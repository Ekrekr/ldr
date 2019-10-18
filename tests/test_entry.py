# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np
from ldr import LDR
from sklearn.datasets import load_breast_cancer


class TestEntry:
    @pytest.fixture(scope="module")
    def toy_class_data(self):
        """
        Loads the Wisconsin breast cancer dataset as a toy problem.
        """        
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["species"] = pd.Categorical.from_codes(data.target,
                                                  data.target_names)
        targets = df["species"]
        df = df.drop(["species"], axis=1)
        return df, targets
    
    def test_ldr_class_entry(self, toy_class_data):
        df, targets = toy_class_data
        ldr = LDR(df, targets, "class", pos_val="benign", neg_val="malignant")
        

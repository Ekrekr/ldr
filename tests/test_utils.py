# -*- coding: utf-8 -*-
"""
LDR test_utils

Tests `ldr.utils`.
"""
import pytest
import pandas as pd
import numpy as np
from ldr import utils


class TestUtils:
    """
    Tests the utils module.
    """
    def test_np_numerical(self):
        """
        Tests `utils.is_np_numerical` function.
        """
        np_num_list = np.array([1, 2, 3.0, 4.0, 5.5, 3.2])
        assert utils.is_np_numerical(np_num_list)

        np_mixed_list = np.array([1, 2, 3.0, 4.5, "c", 3.2])
        assert utils.is_np_numerical(np_mixed_list) is False

        np_cat_list = np.array(["a", "b", "c", lambda x: x])
        assert utils.is_np_numerical(np_cat_list) is False

    def test_gen_colors(self):
        """
        Tests utils.gen_colors works, but does not check colors are correct.
        """
        few_cols = utils.gen_colors(4)
        assert len(few_cols) == 4

        many_cols = utils.gen_colors(50)
        assert len(many_cols) == 50

        assert max(max(many_cols)) < 1.0
        assert min(min(many_cols)) > 0.0
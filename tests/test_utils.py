# -*- coding: utf-8 -*-
"""
LDR test_utils

Tests `ldr.utils`.
"""
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

        assert max([max(i) for i in many_cols]) < 1.0
        assert min([min(i) for i in many_cols]) > 0.0

    def test_find_nearest(self):
        """
        Tests number found in array is indeed nearest.
        """
        test_arr = np.array([0.0, 1.0, 2.0, 3.0])
        print(utils.find_nearest(test_arr, 1.2))
        print(utils.find_nearest(test_arr, 1.5))
        assert utils.find_nearest(test_arr, 1.2) == 1.0
        assert utils.find_nearest(test_arr, 1.0) == 1.0
        assert utils.find_nearest(test_arr, 2.51) == 3.0
        assert utils.find_nearest(test_arr, 2.5) == 2.0

    def test_reduce_colors(self):
        """
        Tests elementwise weighting of colors.
        """
        colors = np.array([np.array([0.9, 0.3, 0.1]),
                           np.array([0.4, 0.8, 0.3]),
                           np.array([0.2, 0.3, 0.9]),
                           np.array([0.7, 0.1, 0.2])])

        arr = 

        arr = np.array([[[np.array([0, 1, 0]), np.array([0, 0, 1])],
                         [np.array([1, 0, 0])]],
                        [[[],
                         np.array([0, 0, 1])]],
                        [[[], []]],
                        [[[], []]]])

        print("Shape:", arr.shape)

        w_cols = utils.reduce_colors(arr, colors)

        print("arr:", w_cols)

"""
LDR - utils

Useful generic functions.
"""
import colorsys
import typing
import numpy as np


def is_np_numerical(inp_list) -> bool:
    """
    Checks if a list contains only np.numerical objects.

    Args:
        inpList: List to check.

    Returns:
        True if numerical, else false.
    """
    return all(isinstance(i, np.number) for i in inp_list)


def gen_colors(num, hue=.01, lightness=.6, saturation=.65) -> typing.List:
    """
    Generates diverging colors for plotting.

    Args:
        num: Number of colors to generate.
        hue: Base hue to use.
        lightness: Base lightness to use.
        saturation: Base saturation to use.

    Returns:
        Diverging colors in RGB format.
    """
    # Easiest to generate colors in HSL by spinning the wheel.
    hues = np.linspace(0, 1, num + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(int)
    return np.array(
        [colorsys.hls_to_rgb(i, lightness, saturation) for i in hues])


def find_nearest(array: np.array,
                 value: float) -> float:
    """
    Finds nearest point of value in an array.

    Args:
        array: Array to find nearest value in.
        value: Value to find nearest value in array to.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Take mean of of each prediction, and weight it to the color of the
# class.
def reduce_colors(arr: np.array,
                  colors: np.array):
    """

    arr must be of the shape (x, y, ):
    [
        [
            [
                [w_0_0_0, w_0_0_1],
                [],
                [w_2_0_0]
            ],
            [
                [w_0_1_0, w_1_1_1]
            ]
        ],
        [
            ...
        ]
    ]
    Where w represents a weight, which a prediction from a single sample. The
    mean value of the weights gives the total weighting to give to a color.

    Args:
        arr: Array to transform then reduce to colors.
    """
    weights = np.mean(np.array(arr), axis=0) if len(arr) > 0 else np.array([0.0, 0.0, 0.0])
    w_cols = colors * weights
    # print("arr:", arr)
    # print("colors:", colors)
    # print("weights:", weights)
    # print("w_cols:", w_cols)
    # print("w_cols2:", w_cols)
    # raise Exception()
    w_cols = [sum(i) for i in w_cols]
    return w_cols

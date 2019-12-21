# -*- coding: utf-8 -*-
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
def reduce_colors(weights: np.array,
                  colors: np.array):
    """
    Reduces class colors and respective weights to a single color.

    Let `n` be the number of features in the problem.

    * weights must be a collection of certainties...

    * colors must be of the form:

    ```python3
    np.array([
        np.array([c_0_R, c_0_G, c_0_B]),
        np.array([c_1_R, c_1_G, c_1_B]),
        ...,
        np.array([c_n_R, c_n_G, c_n_B])
    ])
    ```

    * What is returned is a color of the form:

    ```python3
    np.array([c_R, c_G, c_B])
    ```

    Args:
        weights: Array of weights of certainty of each feature.
        colors: Colors for weightings to apply to.
    Returns:
        Average color when reduced.
    """
    try:
        reduced_weights = np.mean(np.array(weights), axis=0) if len(
            weights) > 0 else np.array([1.0 for i in range(len(colors))])
        weighted_colors = [colors[
            i] * reduced_weights[i] for i, _ in enumerate(reduced_weights)]
        final_color = np.sum(np.array(weighted_colors), axis=0) if len(
            weights) > 0 else np.array([1.0, 1.0, 1.0])
    except Exception:
        print("colors:", colors)
        print("weights:", weights)
        print("reduced_weights:", reduced_weights)
        print("weighted_colors:", weighted_colors)
        print("final_color:", final_color)
        exit(0)
    return final_color

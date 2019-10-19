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


def gen_colors(num=6, hue=.01, lightness=.6, saturation=.65) -> typing.List:
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
    return [colorsys.hls_to_rgb(i, lightness, saturation) for i in hues]

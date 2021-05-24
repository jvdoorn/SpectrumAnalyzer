"""
This file contains some miscellaneous methods that
are used in other parts of the software.
"""

import time
from typing import List, Tuple

import numpy as np
from scipy import integrate


def integral(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the integral from x[0] to x[-1] of y and returns that value.
    :param x: x-array.
    :param y: y-array.
    :return: integral of y from x[0] to x[-1].
    """
    return integrate.cumtrapz(y, x)[-1]


def power(frequencies: np.ndarray, fft: np.ndarray, f: float, delta_f: float) -> float:
    """
    Calculates the power for a range of frequencies.
    :param frequencies: the frequency array.
    :param fft: the FFT of the signal.
    :param f: the frequency you want to analyze.
    :param delta_f: the area around the frequency that should be take into account.
    :return: the intensity of the given frequency range.
    """
    interval = (frequencies > f - delta_f) & (frequencies < f + delta_f)
    return integral(frequencies[interval], np.abs(fft[interval] ** 2))


def find_nearest_index(array: np.ndarray, value: float) -> int:
    """
    Returns the index of the array element that is closest
    to the given value.
    :param array: a 1D-array.
    :param value: the value you are looking for.
    :return: the index of the element closest to value.
    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def timestamp() -> str:
    """
    Gets the current time since the Epoch is
    seconds. Milliseconds are ignored.
    :return: a string with the current time since Epoch.
    """
    return str(time.time()).split(".")[0]


def relative_phase(input_phase: float, output_phase: float) -> float:
    """
    Calculates the relative phase between two phases.
    :param input_phase: the input phase.
    :param output_phase: the output phase.
    :return: the relative phase.
    """
    phi = output_phase - input_phase
    if phi < -np.pi:
        return phi + 2 * np.pi
    elif phi > np.pi:
        return phi - 2 * np.pi
    else:
        return phi


def calculate_q_factor(frequencies: np.ndarray, intensity_array: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculates the Q-factor for a set of frequencies and intensities.
    :param frequencies: the frequency array.
    :param intensity_array: the intensity array.
    :return: the Q-factor, the -3dB intensity, first frequency and second frequency.
    """
    argmax = np.argmax(intensity_array)
    target = 0.707946 * intensity_array[argmax]

    arg_1 = find_nearest_index(intensity_array[:argmax], target)
    arg_2 = argmax + find_nearest_index(intensity_array[argmax:], target)

    return frequencies[argmax] / (frequencies[arg_2] - frequencies[arg_1]), 20 * np.log10(target), frequencies[arg_1], \
           frequencies[arg_2]


def latex_float(number: float, precision=4) -> str:
    """
    Turns a float into a latex formatted nice number.
    :param number: the number to format.
    :param precision: the precision you want in the number.
    :return: the number formatted for latex.
    """
    float_str = f"{number:.{precision}g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def cached_property_wrapper(f):
    try:
        from functools import cached_property
        return cached_property(f)
    except ImportError:
        return property(f)


def is_list_of(lst: List, _type):
    if not isinstance(lst, list):
        return False
    return not any([not isinstance(_obj, _type) for _obj in lst])

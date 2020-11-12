"""
This file contains some miscellaneous methods that
are used in other parts of the software.
"""

import time

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
    return - ((np.pi - output_phase + input_phase) % np.pi)

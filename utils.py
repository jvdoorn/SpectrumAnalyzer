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

def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def timestamp():
    return str(time.time()).split(".")[0]
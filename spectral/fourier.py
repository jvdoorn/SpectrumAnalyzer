"""
This file contains various methods to apply and
filter a Fourier transform.
"""

from typing import Tuple

import numpy as np


def fourier(signal: np.ndarray, sample_rate: int, filter: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a Fourier transform to the supplied signal. If filter is set
    to true the returned frequencies will contain only positive frequencies.
    A filter will only be applied when the signal is a 1D-array, else the
    shape of the array is lost.

    Please note that the transform is *not* normalized.
    :param signal: the signal to apply to transform to.
    :param sample_rate: the sample rate of the signal.
    :param filter: whether to return only positive frequencies.
    :return: a frequency array and the transformed signal.
    """
    assert len(signal.shape) <= 2, "Expected 1D or 2D ndarray."
    fft = np.fft.fft(signal)

    if len(signal.shape) == 2:
        freq = np.fft.fftfreq(signal.shape[1], 1 / sample_rate)
    else:
        freq = np.fft.fftfreq(len(signal), 1 / sample_rate)

    return freq, fft if not filter and signal.shape != 1 else filter_positives(freq, fft)


def inverse_fourier(fft: np.ndarray) -> np.ndarray:
    """
    Applies an inverse Fourier transform to the supplied signal. It
    requires that the input does not contain solely the positive
    frequencies (which is the case if you filter your signal).

    :param fft: a Fourier transform.
    :return: the original signal
    """
    assert len(fft.shape) <= 2, "Expected 1D or 2D ndarray."
    return np.fft.ifft(fft)


def filter_positives(frequencies: np.ndarray, fft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    A Fourier transform contains negative Frequencies that might cause
    issue when plotting or finding minima. To avoid this issue we filter
    all negative frequencies from the transform.
    :param frequencies: the frequency array.
    :param fft: the array with the transform.
    :return: a sanitized frequency and fft array.
    """
    mask = frequencies >= 0
    return frequencies[mask], fft[mask]

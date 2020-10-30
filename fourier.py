from typing import Tuple

import numpy as np


def fourier(signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    assert len(signal.shape) <= 2, "Expected 1D or 2D ndarray."

    if len(signal.shape) == 2:
        fft = np.fft.fft(signal, axis=1) / signal.shape[1]
        freq = np.fft.fftfreq(signal.shape[1], 1 / sample_rate)
    else:
        fft = np.fft.fft(signal) / len(signal)
        freq = np.fft.fftfreq(len(signal), 1 / sample_rate)

    return freq, fft


def filter_positives(frequencies, fft):
    mask = frequencies >= 0
    return frequencies[mask], fft[mask]

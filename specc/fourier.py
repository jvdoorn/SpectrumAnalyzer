import numpy as np


def fourier_1d(data: np.ndarray) -> np.ndarray:
    assert isinstance(data, np.ndarray), "Expected a ndarray."
    assert len(data.shape) == 1, "Expected 1D-ndarray."

    transform = np.fft.fft(data)
    return transform


def frequencies_1d(samples: int, sample_rate: float) -> np.ndarray:
    frequencies = np.fft.fftfreq(samples, 1 / sample_rate)
    return frequencies


def inverse_fourier_1d(transform: np.ndarray) -> np.ndarray:
    assert isinstance(transform, np.ndarray), "Expected a ndarray."
    assert len(transform.shape) == 1, "Expected 1D-ndarray."

    data = np.fft.ifft(transform)
    return data


def fourier_2d(data: np.ndarray) -> np.ndarray:
    assert isinstance(data, np.ndarray), "Expected a ndarray."
    assert len(data.shape) == 2, "Expected a 2D-ndarray."

    shifted_data = np.fft.fftshift(data)
    shifted_transform = np.fft.fft2(shifted_data)
    transform = np.fft.ifftshift(shifted_transform)
    return transform


def inverse_fourier_2d(transform: np.ndarray) -> np.ndarray:
    assert isinstance(transform, np.ndarray), "Expected a ndarray."
    assert len(transform.shape) == 2, "Expected a 2D-ndarray."

    shifted_transform = np.fft.fftshift(transform)
    shifted_data = np.fft.ifft2(shifted_transform)
    data = np.fft.ifftshift(shifted_data)
    return data

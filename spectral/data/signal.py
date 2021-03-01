from typing import Union

import numpy as np

from spectral.fourier import fourier_1d, frequencies_1d
from spectral.utils import find_nearest_index, integral


def _validate_samples(samples: Union[np.ndarray, list]) -> np.ndarray:
    assert isinstance(samples, (np.ndarray, list)), "Expected samples to be a list or ndarray."
    if isinstance(samples, list):
        samples = np.asarray(samples)
    assert len(samples.shape) == 1, "Expected 1D-ndarray as input signal."
    return samples


def _validate_sample_rate(sample_rate: float):
    assert sample_rate > 0, "Expected a positive sample rate."
    return sample_rate


class Signal:
    # See https://stackoverflow.com/a/41948659, fixes multiplication with ndarray * Signal.
    __array_priority__ = 10000

    def __init__(self, sample_rate: float, samples: Union[np.ndarray, list]):
        self.sample_rate = _validate_sample_rate(sample_rate)
        self.samples = _validate_samples(samples)

        self._fft = None
        self._nfft = None
        self._frequencies = None

    @classmethod
    def generate(cls, sample_rate: int, samples: int, frequency: float, amplitude: float = 1, method=np.sin):
        samples = amplitude * method(2 * np.pi * frequency * np.linspace(0, samples / sample_rate, samples))
        return cls(sample_rate, samples)

    @classmethod
    def load(cls, file, sample_rate: int):
        samples = np.genfromtxt(file)
        return cls(sample_rate, samples)

    def save(self, file):
        np.savetxt(file, self.samples)

    @property
    def fft(self) -> np.ndarray:
        if self._fft is None:
            self._fft = fourier_1d(self.samples)
        return self._fft

    @property
    def nfft(self) -> np.ndarray:
        if self._nfft is None:
            self._nfft = fourier_1d(self.samples - self.samples.mean())
        return self._nfft

    @property
    def frequencies(self):
        if self._frequencies is None:
            self._frequencies = frequencies_1d(len(self), self.sample_rate)
        return self._frequencies

    @property
    def _frequency_mask(self):
        return self.frequencies >= 0

    @property
    def masked_fft(self) -> np.ndarray:
        return self.fft[self._frequency_mask]

    @property
    def masked_nfft(self) -> np.ndarray:
        return self.nfft[self._frequency_mask]

    @property
    def masked_frequencies(self) -> np.ndarray:
        return self.frequencies[self._frequency_mask]

    @property
    def timestamps(self) -> np.ndarray:
        return np.linspace(0, len(self) / self.sample_rate, len(self))

    def __len__(self):
        return len(self.samples)

    def __mul__(self, other):
        if isinstance(other, Signal):
            assert other.sample_rate == self.sample_rate, "Signals must have similar sample rates."
            return Signal(self.sample_rate, self.samples * other.samples)
        elif isinstance(other, np.ndarray):
            return Signal(self.sample_rate, self.samples * other)
        elif isinstance(other, (float, int)):
            return Signal(self.sample_rate, self.samples * other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def find_nearest_frequency_index(self, frequency: float) -> int:
        return find_nearest_index(self.frequencies, frequency)

    @property
    def phases(self) -> np.ndarray:
        return np.angle(self.fft)

    def get_phase(self, frequency: float) -> float:
        return self.phases[self.find_nearest_frequency_index(frequency)]

    def power(self, frequency: float, df: float):
        interval = (self.frequencies > frequency - df) & (self.frequencies < frequency + df) & (self.frequencies > 0)
        normalized_fft = self.fft / len(self)
        return integral(self.frequencies[interval], np.abs(normalized_fft[interval] ** 2))

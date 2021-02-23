import matplotlib.pyplot as plt
import numpy as np

from spectral.fourier import fourier_1d, frequencies_1d
from spectral.utils import find_nearest_index, integral


class Signal:
    # See https://stackoverflow.com/a/41948659, fixes multiplication with ndarray * Signal.
    __array_priority__ = 10000

    def __init__(self, sample_rate: float, samples: np.ndarray):
        assert len(samples.shape) == 1, "Expected 1D-ndarray as input signal."
        assert sample_rate > 0, "Expected a positive sample rate."

        self.sample_rate = sample_rate
        self.samples = samples

        self.fft = fourier_1d(samples)
        self.frequencies = frequencies_1d(len(self), self.sample_rate)
        self._fft_mask = self.frequencies >= 0

    @staticmethod
    def generate(sample_rate: int, samples: int, frequency: float, amplitude: float = 1, method=np.sin):
        samples = amplitude * method(2 * np.pi * frequency * np.linspace(0, samples / sample_rate, samples))
        return Signal(sample_rate, samples)

    @staticmethod
    def load(file, sample_rate: int):
        samples = np.genfromtxt(file)
        return Signal(sample_rate, samples)

    def save(self, file):
        np.savetxt(file, self.samples)

    @property
    def masked_fft(self) -> np.ndarray:
        return self.fft[self._fft_mask]

    @property
    def masked_frequencies(self) -> np.ndarray:
        return self.frequencies[self._fft_mask]

    @property
    def timestamps(self) -> np.ndarray:
        return np.linspace(0, len(self) / self.sample_rate, len(self))

    def __len__(self):
        return len(self.samples)

    def __mul__(self, other):
        if isinstance(other, Signal):
            assert len(other) == len(self), "Signals must have the same length."
            assert other.sample_rate == self.sample_rate, "Signals must have similar sample rates."
            return Signal(self.sample_rate, self.samples * other.samples)
        elif isinstance(other, np.ndarray):
            assert len(other.shape) == 1, "Expected a 1D-ndarray."
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

    def plot(self):
        plt.plot(self.timestamps, self.samples)
        plt.xlabel("Time [s]")
        plt.ylabel("Signal [V]")
        return plt

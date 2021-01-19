from numpy import angle, ndarray
from scipy.fft import fft, fftfreq

from spectral.utils import find_nearest_index, integral


class Signal:
    def __init__(self, sample_rate: float, data: ndarray):
        assert len(data.shape) == 1, "Expected 1D-ndarray as input signal."
        assert sample_rate > 0, "Expected a positive sample rate."

        self.sample_rate = sample_rate
        self._length = len(data)

        self._fft = fft(data)
        self._frequencies = fftfreq(len(self), 1 / self.sample_rate)
        self._mask = (self._frequencies >= 0)

    def __len__(self):
        return self._length

    @property
    def fft(self):
        return self._fft[self._mask]

    @property
    def frequencies(self):
        return self._frequencies[self._mask]

    def find_nearest_index(self, frequency: float) -> int:
        return find_nearest_index(self.frequencies, frequency)

    @property
    def phases(self) -> ndarray:
        return angle(self.fft)

    def get_phase(self, frequency: float) -> float:
        index = find_nearest_index(self.frequencies, frequency)
        return self.phases[index]

    def power(self, frequency, df: float):
        interval = (self.frequencies > frequency - df) & (self.frequencies < frequency + df)
        return integral(self.frequencies[interval], abs(self.fft[interval] ** 2))

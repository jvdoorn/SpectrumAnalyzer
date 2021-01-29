from numpy import angle, linspace, ndarray, pi, sin
from scipy import ifft
from scipy.fft import fft, fftfreq

from spectral.utils import find_nearest_index, integral


class Signal:
    def __init__(self, sample_rate: float, data: ndarray):
        assert len(data.shape) == 1, "Expected 1D-ndarray as input signal."
        assert sample_rate > 0, "Expected a positive sample rate."

        self.sample_rate = sample_rate
        self.data = data

        self.fft = fft(data)
        self.frequencies = fftfreq(len(self), 1 / self.sample_rate)

    def __len__(self):
        return len(self.data)

    def __mul__(self, other):
        assert len(other) == len(self), "Signals must have the same length."

        if isinstance(other, Signal):
            assert other.sample_rate == self.sample_rate, "Signals must have similar sample rates."
            return Signal(self.sample_rate, self.data * other.data)
        elif isinstance(other, ndarray):
            if len(other.shape) != 1:
                raise NotImplementedError
            return Signal(self.sample_rate, self.data * other)
        else:
            raise NotImplementedError

    def transfer(self, transfer: ndarray):
        assert len(transfer) == len(self), "Transfer array must have the same length."
        new_fft = transfer * self.fft
        new_data = ifft(new_fft)
        return Signal(self.sample_rate, new_data)

    def find_nearest_frequency_index(self, frequency: float) -> int:
        return find_nearest_index(self.frequencies, frequency)

    @property
    def phases(self) -> ndarray:
        return angle(self.fft)

    def get_phase(self, frequency: float) -> float:
        return self.phases[self.find_nearest_frequency_index(frequency)]

    def power(self, frequency, df: float):
        interval = (self.frequencies > frequency - df) & (self.frequencies < frequency + df)
        return integral(self.frequencies[interval], abs(self.fft[interval] ** 2))


class ArtificialSignal(Signal):
    def __init__(self, frequency: float, amplitude: float, sample_rate: int, samples: int, method=sin):
        super().__init__(sample_rate,
                         amplitude * method(2 * pi * frequency * linspace(0, samples / sample_rate, samples)))

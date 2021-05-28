from typing import Union
from warnings import warn

import numpy as np

from specc.analysis.converter import Converter, DEFAULT_CONVERTER
from specc.fourier import fourier_1d, frequencies_1d
from specc.utils import cached_property_wrapper as cached_property, find_nearest_index, integral


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

    def __init__(self, sample_rate: float, samples: Union[np.ndarray, list], converter: Converter = DEFAULT_CONVERTER):
        self.sample_rate = _validate_sample_rate(sample_rate)
        self.samples = _validate_samples(samples)
        self.converter = converter

    def __eq__(self, other):
        return isinstance(other, Signal) \
               and self.sample_rate == other.sample_rate \
               and np.array_equal(self.samples, other.samples)

    @classmethod
    def generate(cls, sample_rate: int, samples: int, frequency: float, amplitude: float = 1, method=np.sin,
                 converter: Converter = DEFAULT_CONVERTER):
        samples = amplitude * method(2 * np.pi * frequency * np.linspace(0, samples / sample_rate, samples))
        return cls(sample_rate, samples, converter)

    @classmethod
    def load_from_csv(cls, file: str, sample_rate: int, converter: Converter = DEFAULT_CONVERTER):
        warn("load_from_csv should only be used to convert legacy data. It is recommended to use load.")
        samples = np.genfromtxt(file)
        return cls(sample_rate, samples, converter)

    @classmethod
    def load(cls, file: str, converter: Converter = DEFAULT_CONVERTER):
        data = np.load(file)

        sample_rate = data['sample_rate']
        samples = data['samples']

        return cls(sample_rate, samples, converter)

    def save(self, file: str):
        np.savez_compressed(file, samples=self.samples, sample_rate=self.sample_rate)

    @cached_property
    def error(self) -> float:
        return self.converter.error(self.samples)

    @cached_property
    def csamples(self) -> np.ndarray:
        return self.converter.convert(self.samples)

    @cached_property
    def fft(self) -> np.ndarray:
        return fourier_1d(self.samples)

    @cached_property
    def nfft(self) -> np.ndarray:
        return fourier_1d(self.samples - self.samples.mean())

    @cached_property
    def frequencies(self):
        return frequencies_1d(len(self), self.sample_rate)

    @cached_property
    def _frequency_mask(self):
        return self.frequencies >= 0

    @cached_property
    def masked_fft(self) -> np.ndarray:
        return self.fft[self._frequency_mask]

    @cached_property
    def masked_nfft(self) -> np.ndarray:
        return self.nfft[self._frequency_mask]

    @cached_property
    def masked_frequencies(self) -> np.ndarray:
        return self.frequencies[self._frequency_mask]

    @cached_property
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

    def __truediv__(self, other):
        if isinstance(other, (np.ndarray, int, float)):
            return Signal(self.sample_rate, self.samples / other)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if isinstance(other, (np.ndarray, int, float)):
            return Signal(self.sample_rate, other / self.samples)
        else:
            raise NotImplementedError

    def find_nearest_frequency_index(self, frequency: float) -> int:
        return find_nearest_index(self.frequencies, frequency)

    @cached_property
    def phases(self) -> np.ndarray:
        return np.angle(self.fft)

    def get_phase(self, frequency: float) -> float:
        return self.phases[self.find_nearest_frequency_index(frequency)]

    def power(self, frequency: float, df: float):
        interval = (self.frequencies > frequency - df) & (self.frequencies < frequency + df) & (self.frequencies > 0)
        normalized_fft = self.fft / len(self)
        return np.sqrt(integral(self.frequencies[interval], np.abs(normalized_fft[interval] ** 2)))

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.stats import linregress

from spectral.data.signal import Signal
from spectral.utils import find_nearest_index, relative_phase


class SystemResponse:
    def __init__(self, input_signal: Signal, output_signal: Signal):
        assert len(input_signal) == len(output_signal), "Expected signals to have equal lengths."
        assert input_signal.sample_rate == output_signal.sample_rate, "Expected signals to have equal sample rates."

        self.input_signal = input_signal
        self.output_signal = output_signal

    def relative_intensity(self, frequency: float, df: float):
        return self.output_signal.power(frequency, df) / self.input_signal.power(frequency, df)

    def relative_phase(self, frequency: float):
        index = find_nearest_index(self.input_signal.frequencies, frequency)

        input_phase = self.input_signal.phases[index]
        output_phase = self.output_signal.phases[index]

        return relative_phase(input_phase, output_phase)


class FrequencyResponse:
    def __init__(self, intensity: float, phase: float):
        self.intensity = intensity
        self.phase = phase


class SystemBehaviour:
    def __init__(self):
        self._responses: Dict[float, FrequencyResponse] = {}

    def add_response(self, frequency: float, response: FrequencyResponse):
        self._responses[frequency] = response

    @property
    def frequencies(self):
        return np.asarray([frequency for frequency in sorted(self._responses.keys())])

    @property
    def intensities(self):
        return np.asarray([response.intensity for _, response in sorted(self._responses.items())])

    @property
    def phases(self):
        return np.asarray([response.phase for _, response in sorted(self._responses.items())])

    def fit_n20db_line(self, order: int, delta: float = 1) -> Tuple[float, float, float]:
        assert order != 0, AssertionError("order must be non-zero.")
        assert delta >= 0, AssertionError("delta must be positive.")

        decibels = 20 * np.log10(self.intensities)
        frequencies = np.log10(self.frequencies)

        intensity_gradient = np.gradient(decibels, frequencies)
        n20db_mask = (order * 20 - delta <= intensity_gradient) & (intensity_gradient <= order * 20 + delta)

        decibels = decibels[n20db_mask]
        frequencies = frequencies[n20db_mask]

        slope, intercept, r_value, p_value, std_err = linregress(frequencies, decibels)
        return slope, intercept, std_err


class TransferFunctionBehaviour(SystemBehaviour):
    def __init__(self, frequencies: np.ndarray, transfer_function: Callable[[np.ndarray], np.ndarray]):
        super().__init__()

        self._frequencies = frequencies
        self._responses = transfer_function(frequencies)

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def intensities(self):
        return np.abs(self._responses)

    @property
    def phases(self):
        return np.angle(self._responses)

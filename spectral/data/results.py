from typing import Callable, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from spectral.data.signal import Signal
from spectral.utils import find_nearest_index, relative_phase

PHASE_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PHASE_LABELS = ["$-\\pi$", "$-\\frac{1}{2}\\pi$", "0", "$\\frac{1}{2}\\pi$", "$\\pi$"]


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

    def polar_plot(self, title: str):
        plt.polar(self.phases, self.intensities)
        plt.title(title)
        return plt

    def plot(self, title: str, intensity_markers: Union[list, None] = None, phase_markers: Union[list, None] = None):
        """
        Creates a bode plot of the frequencies, intensities and phases.
        :param title: the title of the plot.
        :param intensity_markers: markers for specific intensities [dB].
        :param phase_markers: markers for specific phases [rad].
        """

        if phase_markers is None:
            phase_markers = []
        if intensity_markers is None:
            intensity_markers = []

        fig = plt.figure(figsize=(6, 4), dpi=400)
        fig.suptitle(title)

        intensity_axis = plt.subplot2grid((2, 1), (0, 0))
        intensity_axis.set_ylabel("$20\\log|H(f)|$ [dB]")

        phase_axis = plt.subplot2grid((2, 1), (1, 0))
        phase_axis.set_ylabel("Phase [rad]")
        phase_axis.set_xlabel("Frequency [Hz]")
        phase_axis.set_yticks(PHASE_TICKS)
        phase_axis.set_yticklabels(PHASE_LABELS)
        phase_axis.set_ylim(-np.pi, np.pi)

        decibels = 20 * np.log10(self.intensities)

        intensity_axis.semilogx(self.frequencies, decibels)
        for marker in intensity_markers:
            intensity_axis.axhline(marker, linestyle='--', color='r', alpha=0.5)

        phase_axis.semilogx(self.frequencies, self.phases)
        for marker in phase_markers:
            phase_axis.axhline(marker, linestyle='--', color='r', alpha=0.5)

        return plt


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

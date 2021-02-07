"""
This is the main class containing the Analyzer base class
and two sub classes.
"""
import multiprocessing as mp
from os import listdir, makedirs
from typing import Dict, Tuple, Type, Union

import matplotlib.pyplot as plt
from numpy import argmax, genfromtxt, gradient, log10, max, ndarray, pi, savetxt
from scipy.stats import linregress
from tqdm import tqdm

from spectral.aquisition.daq import DataAcquisitionInterface
from spectral.data.results import FrequencyResponse, SystemBehaviour, SystemResponse, TransferFunctionBehaviour
from spectral.data.signal import ArtificialSignal, Signal
from spectral.utils import find_nearest_index, timestamp


class Analyzer:
    def __init__(self, df: int):
        self._df = df

    def analyze_directory(self, sample_rate: int, data_directory: str,
                          max_cpu_cores: int = mp.cpu_count()) -> SystemBehaviour:
        """
        This method can be used to analyze old data in the specified directory. It will
        analyze it in parallel using as many cores as possible. If the data files are large
        this can cause issues with memory. Hence it is sometimes faster to run it with less
        cores.
        :param sample_rate: the sample rate of the data in the directory.
        :param data_directory: the relative or absolute path of the directory.
        :param max_cpu_cores: limits the cpu core count.
        """
        behaviour = SystemBehaviour()
        input_files = _get_data_files(data_directory)

        with mp.Pool(max_cpu_cores) as pool:
            for frequency, file in iter(input_files):
                pool.apply_async(_process_file, args=(file, sample_rate),
                                 callback=lambda result: behaviour.add_response(frequency, result))

        return behaviour

    @staticmethod
    def fit_n20db_line(behaviour: SystemBehaviour, order: int = -1, delta: float = 1) -> \
            Tuple[float, float, float]:
        """
        Fits a line on all the points who's gradient matches
        the given order. Normally the order is a positive number
        however specifying a negative number will cause it to fit
        on a decreasing slope and a positive number on an increasing
        slope.
        :param behaviour: the behaviour of the system.
        :param order: the order you want to fit, the sign determines whether increasing or decreasing slopes.
        :param delta: the range around the gradient.
        :return: the calculated slope, intercept and standard error.
        """
        assert order != 0, AssertionError("order must be non-zero.")
        assert delta >= 0, AssertionError("delta must be positive.")

        decibels = 20 * log10(behaviour.intensities)
        intensity_gradient = gradient(decibels, log10(behaviour.frequencies))

        intensity_gradient_mask = (order * 20 - delta <= intensity_gradient) \
                                  & (intensity_gradient <= order * 20 + delta)
        slope, intercept, r_value, p_value, std_err = linregress(log10(behaviour.frequencies[intensity_gradient_mask]),
                                                                 decibels[intensity_gradient_mask])
        return slope, intercept, std_err

    @staticmethod
    def calculate_fit(frequencies, slope, intercept, intensities=None, trim: bool = False) -> \
            Tuple[ndarray, ndarray]:
        """
        Calculates a fit. It trim is True it will trim all points
        that are higher than the maximum value of intensities. This
        avoids graphs going absolutely crazy.
        :param frequencies: the frequency array.
        :param slope: the slope of the fit.
        :param intercept: the intercept of the fit.
        :param intensities: the intensities of the signal.
        :param trim: whether to trim the signal or not.
        :return: the frequencies and the corresponding fit values.
        """
        fit = slope * log10(frequencies) + intercept
        if trim:
            mask = fit <= max(intensities)
            return frequencies[mask], fit[mask]
        return frequencies, fit

    @staticmethod
    def plot(title: str, behaviour: Union[SystemBehaviour, TransferFunctionBehaviour], intensity_markers: list = [-3],
             phase_markers: list = [-pi / 4], mark_max=False, mark_min=False, mark_vertical: bool = True,
             plot_gradient: bool = False, fit: Tuple[ndarray, ndarray] = None, save: bool = True,
             directory: str = "figures/", filename: str = None):
        """
        Creates a bode plot of the frequencies, intensities and phases.
        :param title: the title of the plot.
        :param behaviour: the behaviour of the system.
        :param intensity_markers: markers for specific intensities [dB].
        :param phase_markers: markers for specific phases.
        :param mark_max: mark the maximum intensity.
        :param mark_min: mark the minimum intensity.
        :param mark_vertical: mark intensities/phases vertically as well.
        :param plot_gradient: plot the gradient of the phase/intensity as well.
        :param fit: an optional tuple of both frequencies and fitted intensities. See calculate_fit.
        :param save: whether to save this figure or not.
        :param directory: directory to save this figure to.
        :param filename: name of the file to save to (default title), do not use an extension.
        """
        # Try to apply a fancy style, since this is
        # not required don't complain if it fails.
        try:
            plt.style.use(['science', 'grid'])
        except IOError:
            pass

        # Create a new figure that's larger than default.
        fig = plt.figure(figsize=(6, 4), dpi=400)
        fig.suptitle(title)

        # Determine our axes.
        ax1 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection='polar')
        ax2 = plt.subplot2grid((2, 2), (0, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 0))

        # Create a polar plot.
        ax1.plot(behaviour.phases, behaviour.intensities)

        # Convert to decibels
        decibels = 20 * log10(behaviour.intensities)

        # Plot the intensities
        ax2.semilogx()
        ax2.plot(behaviour.frequencies, decibels)
        if plot_gradient:
            intensity_gradient = gradient(decibels, log10(behaviour.frequencies))
            ax2.plot(behaviour.frequencies, intensity_gradient, linestyle='--', color='g')
        if fit is not None:
            ax2.plot(*fit, linestyle='--', color='y')

        ax2.set_ylabel("$20\\log|H(f)|$ (dB)")

        if mark_max:
            marker_frequency = behaviour.frequencies[argmax(decibels)]
            ax2.axvline(x=marker_frequency, linestyle='--', color='r', alpha=0.5)
        if mark_min:
            marker_frequency = behaviour.frequencies[argmax(decibels)]
            ax2.axvline(x=marker_frequency, linestyle='--', color='r', alpha=0.5)

        for marker in intensity_markers:
            ax2.axhline(marker, linestyle='--', color='r', alpha=0.5)
            if mark_vertical:
                marker_frequency = behaviour.frequencies[find_nearest_index(decibels, marker)]
                ax2.axvline(marker_frequency, linestyle='--', color='r', alpha=0.5)

        # Plot the phases
        ax3.semilogx(behaviour.frequencies, behaviour.phases)
        if plot_gradient:
            phase_gradient = gradient(behaviour.phases, log10(behaviour.frequencies))
            ax3.semilogx(behaviour.frequencies, phase_gradient, linestyle='--', color='g')

        # Set some pretty yticks.
        ax3.set_yticks([-pi, -pi / 2, 0, pi / 2, pi])
        ax3.set_ylim(-pi, pi)
        ax3.set_yticklabels(["$-\\pi$", "$-\\frac{1}{2}\\pi$", "0", "$\\frac{1}{2}\\pi$", "$\\pi$"])

        ax3.set_ylabel("Phase")
        ax3.set_xlabel("Frequency [Hz]")

        for marker in phase_markers:
            ax3.axhline(marker, linestyle='--', color='r', alpha=0.5)
            if mark_vertical:
                marker_frequency = behaviour.frequencies[find_nearest_index(behaviour.phases, marker)]
                ax3.axvline(marker_frequency, linestyle='--', color='r', alpha=0.5)

        if save:
            makedirs(directory, exist_ok=True)
            plt.savefig(f"{directory}{title if not filename else filename}.png")
            plt.savefig(f"{directory}{title if not filename else filename}.svg")

        return plt


class DAQAnalyzer(Analyzer):
    """
    This class can be used to measure and analyze a system.
    """

    def __init__(self, daq: Type[DataAcquisitionInterface], df: int, base_directory: str = "data/",
                 write_channel: str = "myDAQ1/AO0", pre_system_channel: str = "myDAQ1/AI0",
                 post_system_channel: str = "myDAQ1/AI1"):
        super().__init__(df)

        self._daq = daq

        self._base_directory = base_directory

        self._write_channel = write_channel
        self._pre_system_channel = pre_system_channel
        self._post_system_channel = post_system_channel

    def measure_single(self, samples: int) -> SystemResponse:
        """
        Used to measure a signal before and after passing through a system. Useful when using other hardware to drive
        the system.
        :param samples: the amount of samples.
        :return: the response of the system.
        """
        data = self._daq.read([self._pre_system_channel, self._post_system_channel], samples)

        pre_system_signal = Signal(self._daq.sample_rate, data[0])
        post_system_signal = Signal(self._daq.sample_rate, data[1])

        return SystemResponse(pre_system_signal, post_system_signal)

    def drive_and_measure_single(self, frequency: float, data_directory: str, samples: int) -> SystemResponse:
        """
        Send a signal to a channel and measures the output.
        :param frequency: the frequency to measure.
        :param data_directory: the directory to save the data to.
        :param samples: the amount of samples.
        :return: the response of the system.
        """
        artificial_signal = ArtificialSignal(frequency, self._daq.sample_rate, samples)

        data = self._daq.read_write(artificial_signal.samples, [self._write_channel],
                                    [self._pre_system_channel, self._post_system_channel], samples)
        savetxt(f"{data_directory}{frequency}.csv", data)

        pre_system_signal = Signal(self._daq.sample_rate, data[0])
        post_system_signal = Signal(self._daq.sample_rate, data[1])

        return SystemResponse(pre_system_signal, post_system_signal)

    def drive_and_measure_multiple(self, frequencies: list, samples: int) -> SystemBehaviour:
        """
        Sends a series of signals to a channel and measures the output.
        :param frequencies: the frequencies to measure.
        :param samples: the amount of samples.
        :return: the frequencies, magnitudes and phases.
        """
        data_directory = f"{self._base_directory}{timestamp()}/"
        makedirs(data_directory)

        behaviour = SystemBehaviour()
        for frequency in tqdm(frequencies):
            response = self.drive_and_measure_single(frequency, data_directory, samples)
            response = FrequencyResponse(response.relative_intensity(frequency, self._df),
                                         response.relative_phase(frequency))

            behaviour.add_response(frequency, response)
        return behaviour


def _process_file(file: str, sample_rate: float) -> SystemResponse:
    """
    Processes a data file.
    :param file: the file with the data.
    :param sample_rate: the sample rate of the file.
    :return:
    """
    print(f"Processing {file}!")

    data = genfromtxt(file)
    input_signal = Signal(sample_rate, data[0])
    output_signal = Signal(sample_rate, data[1])

    return SystemResponse(input_signal, output_signal)


def _get_data_files(data_directory: str) -> Dict[float, str]:
    files = listdir(data_directory)
    return dict([(float(file.split(".csv")[0]), f"{data_directory}{file}") for file in files])

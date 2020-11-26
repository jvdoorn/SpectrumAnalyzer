"""
This is the main class containing the Analyzer base class
and two sub classes.
"""
import multiprocessing as mp
from os import listdir, makedirs
from typing import Callable, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

from spectral.aquisition.mydaq import MyDAQ
from spectral.fourier import filter_positives, fourier
from spectral.utils import find_nearest_index, power, relative_phase, timestamp

# The default sample rate, this is the maximum the
# Ni MyDAQ can handle.
DEFAULT_SAMPLE_RATE = int(2e5)
# The default sample size.
DEFAULT_SAMPLE_SIZE = 50000

# When generating a signal this is
# the default amplitude.
DEFAULT_AMPLITUDE = 5

# The default integration width.
DEFAULT_INTEGRATION_WIDTH = 20


class Analyzer:
    """
    The base Analyzer class, is not very useful on its own.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, df: int = DEFAULT_INTEGRATION_WIDTH):
        self._sample_rate = sample_rate
        self._df = df

    def generate_artificial_signal(self, frequency: float, amplitude: float = DEFAULT_AMPLITUDE,
                                   samples: int = DEFAULT_SAMPLE_SIZE) -> np.ndarray:
        """
        Generates an artificial signal (sinus) for the given frequency and amplitude of length samples.
        :param frequency: the frequency [Hz] of the signal.
        :param amplitude: the amplitude [V] of the signal.
        :param samples: the amount of samples.
        :return: an artificial signal.
        """
        return amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, samples / self._sample_rate, samples))

    def analyze_single(self, frequency: float, input_signal: np.ndarray, output_signal: np.ndarray) -> \
            Tuple[float, float]:
        """
        Analyzes a specific frequency given an input and output signal. It returns the
        relative intensity and phase of the signals.
        :param frequency: the specific frequency to analyze.
        :param input_signal: the input signal.
        :param output_signal: the output signal.
        :return: the intensity and phase.
        """
        # Apply a fourier transform and filter the signal
        input_frequencies, input_fft = fourier(input_signal, self._sample_rate, True)
        output_frequencies, output_fft = fourier(output_signal, self._sample_rate, True)

        # Determine the output power compared to the input power
        intensity = np.sqrt(power(output_frequencies, output_fft, frequency, self._df) / power(input_frequencies, input_fft,
                                                                                       frequency, self._df))

        # Determine the phases of the input and output signal
        input_phase = np.angle(input_fft)[find_nearest_index(input_frequencies, frequency)]
        output_phase = np.angle(output_fft)[find_nearest_index(output_frequencies, frequency)]

        # Determine the phase shift caused by the system.
        phase = relative_phase(input_phase, output_phase)

        return intensity, phase

    def analyze_directory(self, data_directory: str, max_cpu_cores: int = mp.cpu_count()) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method can be used to analyze old data in the specified directory. It will
        analyze it in parallel using as many cores as possible. If the data files are large
        this can cause issues with memory. Hence it is sometimes faster to run it with less
        cores.
        :param data_directory: the relative or absolute path of the directory.
        :param max_cpu_cores: limits the cpu core count.
        :return: the frequencies, magnitudes and phases.
        """
        # Get all the files in the directory
        files = listdir(data_directory)
        # Determine the frequencies and file paths.
        inputs = [(float(file.split(".csv")[0]), f"{data_directory}{file}") for file in files]

        # Create a new multiprocessing pool
        pool = mp.Pool(max_cpu_cores)

        # Since we run our program asynchronous we have to
        # sort our results when we are finished.
        results = []
        for i, (frequency, file) in enumerate(inputs):
            # Register all the asynchronous tasks.
            pool.apply_async(_process_file, args=(self, i, len(files), frequency, file),
                             callback=lambda result: results.append(result))
        # Run our tasks.
        pool.close()
        pool.join()

        # Sort our results and return them.
        return _sort_and_return_results(results)

    def are_frequencies_safe(self, frequencies: np.ndarray):
        """
        Checks if the given frequencies are higher than the recommended value
        sample_rate / 4. If frequencies exceed this value it can cause issues
        with when doing a Fourier transform.
        :param frequencies: the frequencies to check.
        :return: True if the frequencies exceed the recommended value.
        """
        return np.max(frequencies) > self._sample_rate / 4

    def warn_unsafe_frequencies(self, frequencies: np.ndarray):
        """
        Checks if the user should be warned for using unsafe frequencies and
        if so warns them.
        :param frequencies: the frequencies to check.
        """
        if self.are_frequencies_safe(frequencies):
            print(f"[WARNING] Your frequencies exceed the recommended value: {self._sample_rate / 4:2e} [Hz].")

    @staticmethod
    def predict(frequencies: np.ndarray, transfer_function: Callable[[np.ndarray], np.ndarray]) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts the phase and magnitude shift based on a transfer function. This method
        is different from the SimulationAnalyzer in that it doesn't simulate anything but
        merely evaluates the transfer function.
        :param frequencies: the frequencies to predict.
        :param transfer_function: the transfer function of the system.
        :return: the frequencies, magnitudes and phases.
        """
        transfer = transfer_function(frequencies)
        return frequencies, np.abs(transfer), np.angle(transfer)

    @staticmethod
    def plot(title: str, frequencies: np.ndarray, intensity_array: np.ndarray, phase_array: np.ndarray,
             intensity_markers: list = [-3], phase_markers: list = [-np.pi / 4], mark_max=False, mark_min=False,
             mark_vertical: bool = True, save: bool = True, directory: str = "figures/", filename: str = None):
        """
        Creates a bode plot of the frequencies, intensities and phases.
        :param title: the title of the plot.
        :param frequencies: the frequency array.
        :param intensity_array: the intensity array.
        :param phase_array: the phase array.
        :param intensity_markers: markers for specific intensities [dB].
        :param phase_markers: markers for specific phases.
        :param mark_max: mark the maximum intensity.
        :param mark_min: mark the minimum intensity.
        :param mark_vertical: mark intensities/phases vertically as well.
        :param save: whether to save this figure or not.
        :param directory: directory to save this figure to.
        :param filename: name of the file to save to (default title), do not use an extension.
        """
        # Make sure the inputs are a np.ndarray
        phase_array = np.asarray(phase_array)
        intensity_array = np.asarray(intensity_array)
        frequencies = np.asarray(frequencies)

        # Try to apply a fancy style, since this is
        # not required don't complain if it fails.
        try:
            plt.style.use(['science', 'grid'])
        except IOError:
            pass

        # Create a new figure that's larger than default.
        fig = plt.figure(figsize=(6, 4))
        fig.suptitle(title)

        # Determine our axes.
        ax1 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection='polar')
        ax2 = plt.subplot2grid((2, 2), (0, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 0))

        # Create a polar plot.
        ax1.plot(phase_array, intensity_array)

        # Convert to decibels
        intensity_array = 20 * np.log10(intensity_array)

        # Plot the intensities
        ax2.semilogx()
        ax2.plot(frequencies, intensity_array)

        ax2.set_ylabel("$20\\log|H(f)|$ (dB)")

        if mark_max:
            marker_frequency = frequencies[np.argmax(intensity_array)]
            ax2.axvline(x=marker_frequency, linestyle='--', color='r', alpha=0.5)
        if mark_min:
            marker_frequency = frequencies[np.argmax(intensity_array)]
            ax2.axvline(x=marker_frequency, linestyle='--', color='r', alpha=0.5)

        for marker in intensity_markers:
            ax2.axhline(marker, linestyle='--', color='r', alpha=0.5)
            if mark_vertical:
                marker_frequency = frequencies[find_nearest_index(intensity_array, marker)]
                ax2.axvline(marker_frequency, linestyle='--', color='r', alpha=0.5)

        # Plot the phases
        ax3.semilogx(frequencies, phase_array)

        # Set some pretty yticks.
        ax3.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax3.set_ylim(-np.pi, np.pi)
        ax3.set_yticklabels(["$-\\pi$", "$-\\frac{1}{2}\\pi$", "0", "$\\frac{1}{2}\\pi$", "$\\pi$"])

        ax3.set_ylabel("Phase")
        ax3.set_xlabel("Frequency [Hz]")

        for marker in phase_markers:
            ax3.axhline(marker, linestyle='--', color='r', alpha=0.5)
            if mark_vertical:
                marker_frequency = frequencies[find_nearest_index(phase_array, marker)]
                ax3.axvline(marker_frequency, linestyle='--', color='r', alpha=0.5)

        if save:
            makedirs(directory, exist_ok=True)
            plt.savefig(f"{directory}{title if not filename else filename}.png")
            plt.savefig(f"{directory}{title if not filename else filename}.svg")

        plt.show()


class SystemAnalyzer(Analyzer):
    """
    This class can be used to measure and analyze a system.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, df: int = DEFAULT_INTEGRATION_WIDTH,
                 base_directory: str = "data/", write_channel: str = "myDAQ1/AO0",
                 pre_system_channel: str = "myDAQ1/AI0", post_system_channel: str = "myDAQ1/AI1"):
        super().__init__(sample_rate, df)

        self._base_directory = base_directory
        self._write_channel = write_channel
        self._pre_system_channel = pre_system_channel
        self._post_system_channel = post_system_channel

    def measure_single(self, frequency: float, daq: MyDAQ, data_directory: str,
                       samples: int = DEFAULT_SAMPLE_SIZE, ) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Send a signal to a channel and measures the output.
        :param frequency: the frequency to measure.
        :param daq: the MyDAQ class.
        :param data_directory: the directory to save the data to.
        :param samples: the amount of samples.
        :return: the system before passing through the system and after passing through the system.
        """
        # Generate the artificial signal
        artificial_signal = self.generate_artificial_signal(frequency)

        # Write the artificial signal to the MyDAQ and read the input and output
        # voltage of the system.
        signal, time_array = daq.read_write(artificial_signal, np.asarray([self._write_channel]),
                                            np.asarray([self._pre_system_channel, self._post_system_channel]), samples)
        np.savetxt(f"{data_directory}{frequency}.csv", signal)

        return signal[0], signal[1]

    def measure_system_and_analyze(self, frequencies: np.ndarray, samples: int = DEFAULT_SAMPLE_SIZE) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sends a series of signals to a channel and measures the output.
        :param frequencies: the frequencies to measure.
        :param samples: the amount of samples.
        :return: the frequencies, magnitudes and phases.
        """
        # Optionally warn the user
        self.warn_unsafe_frequencies(frequencies)
        # Create a new MyDAQ interface.
        daq = MyDAQ(self._sample_rate)

        # Determine the data directory.
        data_directory = f"{self._base_directory}{timestamp()}/"
        makedirs(data_directory)

        # Initialize empty arrays.
        intensity_array = []
        phase_array = []

        for i, frequency in enumerate(frequencies, start=1):
            print(f"[{i}/{len(frequencies)}] Analyzing {frequency:.4e} Hz.")

            # Measure the signal.
            pre_system_signal, post_system_signal = self.measure_single(frequency, daq, data_directory, samples)

            # Analyze the signal.
            intensity, phase = self.analyze_single(frequency, pre_system_signal, post_system_signal)

            # Save the values.
            intensity_array.append(intensity)
            phase_array.append(phase)

        return frequencies, np.asarray(intensity_array), np.asarray(phase_array)


class SimulationAnalyzer(Analyzer):
    """
    This class can be used to analyze a system. It is useful when testing
    utility classes or predicting the outcome of your experiment.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, df: int = DEFAULT_INTEGRATION_WIDTH):
        super().__init__(sample_rate, df)

    def simulate_transfer_function(self, frequencies: np.ndarray, transfer_function: Callable[[np.ndarray], np.ndarray],
                                   samples: int = DEFAULT_SAMPLE_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates the specified transfer function.
        :param frequencies: the frequencies to predict.
        :param transfer_function: the transfer function.
        :param samples: the amount of samples in the signal.
        :return: the frequencies, magnitudes and phases.
        """
        # Optionally warn the user
        self.warn_unsafe_frequencies(frequencies)
        # Initialize empty arrays.
        intensity_array = []
        phase_array = []

        for i, frequency in enumerate(frequencies):
            print(f"[{i + 1}/{len(frequencies)}] Generating and analyzing {frequency:.4e} Hz.")

            # Generate an input signal.
            input_signal = self.generate_artificial_signal(frequency, samples=samples)

            # Apply a fourier transform
            input_frequencies, input_fft = filter_positives(*fourier(input_signal, self._sample_rate))
            # Calculate the output transform
            output_frequencies, output_fft = input_frequencies, transfer_function(frequency) * input_fft

            # Determine the output power compared to the input power
            intensity = np.sqrt(
                power(output_frequencies, output_fft, frequency, self._df) / power(input_frequencies, input_fft,
                                                                                   frequency, self._df))

            # Determine the phases of the input and output signal
            input_phase = np.angle(input_fft)[find_nearest_index(input_frequencies, frequency)]
            output_phase = np.angle(output_fft)[find_nearest_index(output_frequencies, frequency)]

            # Determine the phase shift caused by the system.
            phase = relative_phase(input_phase, output_phase)

            intensity_array.append(intensity)
            phase_array.append(phase)

        return frequencies, np.asarray(intensity_array), np.asarray(phase_array)


def _process_file(parent: Type[Analyzer], i: int, total: int, frequency: float, file: str) -> \
        Tuple[float, float, float]:
    """
    Processes a data file.
    :param parent: the parent class.
    :param i: which iteration.
    :param total: the total iteration count.
    :param frequency: the frequency to analyze.
    :param file: the file with the data.
    :return:
    """
    print(f"[{i + 1}/{total}] Analyzing {frequency:.4e} Hz.")

    # Read the signal
    signal = np.genfromtxt(file)
    # Get the arrays
    pre_system_signal, post_system_signal = signal[0], signal[1]

    return frequency, *parent.analyze_single(frequency, pre_system_signal, post_system_signal)


def _sort_and_return_results(results: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sorts a result array based on the first item. This should be the frequency.
    :param results: the result set.
    :return: sorted set of frequencies, intensities and phases.
    """
    results.sort(key=lambda r: r[0])
    results = np.asarray(results)

    frequency_array = results[:, 0]
    intensity_array = results[:, 1]
    phase_array = results[:, 2]

    return frequency_array, intensity_array, phase_array

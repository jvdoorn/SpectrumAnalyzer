import multiprocessing as mp
from os import listdir

import matplotlib.pyplot as plt
import numpy as np

from aquisition.mydaq import MyDAQ
from fourier import filter_positives, fourier
from utils import find_nearest_index, power, timestamp

DEFAULT_SAMPLE_RATE = int(2e5)
DEFAULT_SAMPLE_SIZE = 50000

DEFAULT_AMPLITUDE = 5

DEFAULT_INTEGRATION_WIDTH = 20


def _process_file(s, i, total, frequency, file):
    print(f"[{i + 1}/{total}] Analyzing {frequency:.4e} Hz.")

    signal = np.genfromtxt(file)
    pre_system_signal, post_system_signal = signal[0], signal[1]

    return frequency, *s.analyze_single(frequency, pre_system_signal, post_system_signal)


class Analyzer:
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, df=DEFAULT_INTEGRATION_WIDTH):
        self._sample_rate = sample_rate
        self._df = df

    def generate_artificial_signal(self, frequency, amplitude=DEFAULT_AMPLITUDE, samples=DEFAULT_SAMPLE_SIZE):
        return amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, samples / self._sample_rate, samples))

    def analyze_single(self, frequency, input_signal, output_signal):
        # Apply a fourier transform
        input_frequencies, input_fft = fourier(input_signal, self._sample_rate)
        output_frequencies, output_fft = fourier(output_signal, self._sample_rate)

        # Filter out the negative fourier frequencies
        input_frequencies, input_fft = filter_positives(input_frequencies, input_fft)
        output_frequencies, output_fft = filter_positives(output_frequencies, output_fft)

        # Determine the output power compared to the input power
        intensity = power(input_frequencies, input_fft, frequency, self._df) / power(input_frequencies, output_fft,
                                                                                     frequency, self._df)

        # Determine the phases of the input and output signal
        input_phase = np.angle(input_fft)[find_nearest_index(input_frequencies, frequency)]
        output_phase = np.angle(output_fft)[find_nearest_index(output_frequencies, frequency)]

        # Determine the phase shift caused by the system.
        phase = output_phase - input_phase

        return intensity, phase

    def plot(self, title, frequencies, intensity_array, phase_array):
        phase_array = np.asarray(phase_array)
        intensity_array = np.asarray(intensity_array)
        frequencies = np.asarray(frequencies)

        try:
            plt.style.use(['science', 'grid'])
        except IOError:
            pass

        fig = plt.figure(figsize=(6, 4))
        fig.suptitle(title)

        ax1 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, projection='polar')
        ax2 = plt.subplot2grid((2, 2), (0, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 0))

        ax1.plot(phase_array, intensity_array)

        ax2.loglog()
        ax2.plot(frequencies, intensity_array)

        ax2.set_ylabel("Intensity")

        ax3.semilogx(frequencies, phase_array)

        ax3.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax3.set_yticklabels(["$-\pi$", "$-\\frac{1}{2}\pi$", "0", "$\\frac{1}{2}\pi$", "$\pi$"])

        ax3.set_ylabel("Phase")
        ax3.set_xlabel("Frequency [Hz]")

        plt.show()


class FileAnalyzer(Analyzer):
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, df=DEFAULT_INTEGRATION_WIDTH):
        super().__init__(sample_rate, df)

    def analyze_directory(self, data_directory):
        files = listdir(data_directory)
        inputs = [(float(file.split(".csv")[0]), f"{data_directory}{file}") for file in files]

        pool = mp.Pool(mp.cpu_count())

        results = []
        for i, (frequency, file) in enumerate(inputs):
            pool.apply_async(_process_file, args=(self, i, len(files), frequency, file),
                             callback=lambda result: results.append(result))
        pool.close()
        pool.join()

        results.sort(key=lambda r: r[0])
        results = np.asarray(results)

        frequency_array = results[:, 0]
        intensity_array = results[:, 1]
        phase_array = results[:, 2]

        return frequency_array, intensity_array, phase_array


class SystemAnalyzer(Analyzer):
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, df=DEFAULT_INTEGRATION_WIDTH, base_directory="data/"):
        super().__init__(sample_rate, df)

        self._base_directory = base_directory

    def measure_single(self, frequency, daq, data_directory, samples=DEFAULT_SAMPLE_SIZE, write_channel="myDAQ1/AO0",
                       pre_system_channel="myDAQ1/AI0",
                       post_system_channel="myDAQ1/AI1"):
        # Generate the artificial signal
        artificial_signal = self.generate_artificial_signal(frequency)

        # Write the artificial signal to the MyDAQ and read the input and output
        # voltage of the system.
        signal, time_array = daq.read_write(artificial_signal, np.asarray([write_channel]),
                                            np.asarray[pre_system_channel, post_system_channel], samples)
        np.savetxt(f"{data_directory}{frequency}.csv", signal)

        return signal[0], signal[1]

    def measure_system_and_analyze(self, start_frequency, end_frequency, log_space=True, number=50,
                                   samples=DEFAULT_SAMPLE_SIZE, write_channel="myDAQ1/AO0",
                                   pre_system_channel="myDAQ1/AI0",
                                   post_system_channel="myDAQ1/AI1"):
        if log_space:
            frequencies = np.logspace(start_frequency, end_frequency, number)
        else:
            frequencies = np.linspace(start_frequency, end_frequency, number)

        daq = MyDAQ(self._sample_rate)

        data_directory = f"{self._base_directory}{timestamp()}/"

        intensity_array = []
        phase_array = []

        for i, frequency in enumerate(frequencies, start=1):
            print(f"[{i}/{len(frequencies)}] Analyzing {frequency:.4e} Hz.")

            pre_system_signal, post_system_signal = self.measure_single(frequency, daq, data_directory, samples,
                                                                        write_channel, pre_system_channel,
                                                                        post_system_channel)

            intensity, phase = self.analyze_single(frequency, pre_system_signal, post_system_signal)

            intensity_array.append(intensity)
            phase_array.append(phase)

        return frequencies, np.asarray(intensity_array), np.asarray(phase_array)

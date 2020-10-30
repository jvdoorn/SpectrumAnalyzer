from os import listdir

import numpy as np

from aquisition.mydaq import MyDAQ
from fourier import filter_positives, fourier
from utils import find_nearest_index, power, timestamp


class Analyzer:
    def __init__(self):
        self._sample_rate = int(2e5)
        self._samples = 50000

        self._amplitude = 5

        self._df = 20

        self._data_directory = "data/"

    def generate_artificial_signal(self, frequency):
        return self._amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, self._samples / self._sample_rate, self._samples))

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

    def measure_single(self, frequency, daq, data_directory, write_channel="myDAQ1/AO0",
                       pre_system_channel="myDAQ1/AI0",
                       post_system_channel="myDAQ1/AI1"):
        # Generate the artificial signal
        artificial_signal = self.generate_artificial_signal(frequency)

        # Write the artificial signal to the MyDAQ and read the input and output
        # voltage of the system.
        signal, time_array = daq.read_write(artificial_signal, np.asarray([write_channel]),
                                            np.asarray[pre_system_channel, post_system_channel], self._samples)
        np.savetxt(f"{data_directory}{frequency}.csv", signal)

        return signal[0], signal[1]

    def analyze_directory(self, subfolder):
        data_directory = f"{self._data_directory}{subfolder}/"

        frequencies = []
        intensity_array = []
        phase_array = []

        files = sorted(listdir(data_directory))
        for i, filename in enumerate(files):
            frequency = float(filename.split(".csv")[0])
            print(f"[{i}/{len(files)}] Analyzing {frequency:.4e} Hz.")

            signal = np.genfromtxt(f"{data_directory}{filename}")
            pre_system_signal, post_system_signal = signal[0], signal[1]

            intensity, phase = self.analyze_single(frequency, pre_system_signal, post_system_signal)

            frequencies.append(frequency)
            intensity_array.append(intensity)
            phase_array.append(phase)

        return np.asarray(frequencies), np.asarray(intensity_array), np.asarray(phase_array)

    def measure_system_and_analyze(self, start_frequency, end_frequency, log_space=True, number=50,
                                   write_channel="myDAQ1/AO0", pre_system_channel="myDAQ1/AI0",
                                   post_system_channel="myDAQ1/AI1"):
        if log_space:
            frequencies = np.logspace(start_frequency, end_frequency, number)
        else:
            frequencies = np.linspace(start_frequency, end_frequency, number)

        daq = MyDAQ(self._sample_rate)

        data_directory = f"{self._data_directory}{timestamp()}/"

        intensity_array = []
        phase_array = []

        for i, frequency in enumerate(frequencies, start=1):
            print(f"[{i}/{len(frequencies)}] Analyzing {frequency:.4e} Hz.")

            pre_system_signal, post_system_signal = self.measure_single(frequency, daq, data_directory, write_channel,
                                                                        pre_system_channel, post_system_channel)

            intensity, phase = self.analyze_single(frequency, pre_system_signal, post_system_signal)

            intensity_array.append(intensity)
            phase_array.append(phase)

        return frequencies, np.asarray(intensity_array), np.asarray(phase_array)

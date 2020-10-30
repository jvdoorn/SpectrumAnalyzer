import numpy as np

from aquisition.mydaq import MyDAQ
from fourier import fourier
from utils import find_nearest_index, power, timestamp


class Analyzer:
    def __init__(self, start_frequency, end_frequency, log_space=True, number=50):
        self._sample_rate = int(2e5)
        self._samples = 50000

        self._amplitude = 5

        self._df = 20

        self._data_directory = "data/"

        if log_space:
            self._frequencies = np.logspace(start_frequency, end_frequency, number)
        else:
            self._frequencies = np.linspace(start_frequency, end_frequency, number)

    def generate_artificial_signal(self, frequency):
        return self._amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, self._samples / self._sample_rate, self._samples))

    def analyze_single(self, frequency, input_signal, output_signal):
        # Apply a fourier transform
        input_frequencies, input_fft = fourier(input_signal, self._sample_rate)
        output_frequencies, output_fft = fourier(output_signal, self._sample_rate)

        # Filter out the negative fourier frequencies
        input_filter = input_frequencies >= 0
        output_filter = output_frequencies >= 0

        input_frequencies, input_fft = input_frequencies[input_filter], input_fft[input_filter]
        output_frequencies, output_fft = output_frequencies[output_filter], output_fft[output_filter]

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

    def measure_system_and_analyze(self, write_channel="myDAQ1/AO0", pre_system_channel="myDAQ1/AI0",
                                   post_system_channel="myDAQ1/AI1"):
        daq = MyDAQ(self._sample_rate)

        data_directory = f"{self._data_directory}{timestamp()}/"

        intensity_array = []
        phase_array = []

        for i, frequency in enumerate(self._frequencies, start=1):
            print(f"[{i}/{len(self._frequencies)}] Analyzing {frequency:.4e} Hz.")

            pre_system_signal, post_system_signal = self.measure_single(frequency, daq, data_directory, write_channel,
                                                                        pre_system_channel, post_system_channel)

            intensity, phase = self.analyze_single(frequency, pre_system_signal, post_system_signal)

            intensity_array.append(intensity)
            phase_array.append(phase)

        return np.asarray(intensity_array), np.asarray(phase_array)

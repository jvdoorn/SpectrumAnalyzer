import numpy as np

from aquisition.mydaq import MyDAQ
from fourier import fourier
from utils import find_nearest_index, power, timestamp


class Analyzer:
    def __init__(self, start_frequency, end_frequency, log_space=True, number=50,
                 read_channels=np.array(["myDAQ1/AI0", "myDAQ1/AI1"]), write_channels=np.array(["myDAQ1/AO0"])):

        self._sample_rate = int(2e5)
        self._samples = 50000

        self._amplitude = 5

        self._read_channels = np.asarray(read_channels)
        self._write_channels = np.asarray(write_channels)

        self._df = 20

        self._phase_array = []
        self._intensity_array = []

        self._data_directory = "data/"

        if log_space:
            self._frequencies = np.logspace(start_frequency, end_frequency, number)
        else:
            self._frequencies = np.linspace(start_frequency, end_frequency, number)

    def reset(self):
        self._phase_array = []
        self._intensity_array = []

    def generate_artificial_signal(self, frequency):
        return self._amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, self._samples / self._sample_rate, self._samples))

    def analyze(self):
        self.reset()

        daq = MyDAQ(self._sample_rate)

        data_directory = f"{self._data_directory}{timestamp()}/"

        for i, frequency in enumerate(self._frequencies):
            print(f"[{i}/{len(self._frequencies)}] Analyzing {frequency:.4e} Hz.")

            # Generate the artificial signal
            artificial_signal = self.generate_artificial_signal(frequency)

            # Write the artificial signal to the MyDAQ and read the input and output
            # voltage of the system.
            signal, time_array = daq.read_write(artificial_signal, self._write_channels, self._read_channels,
                                                self._samples)
            np.savetxt(f"{data_directory}{frequency}.csv", signal)

            # Get the signals
            input_signal = signal[0]  # Signal before passing through the system
            output_signal = signal[1]  # Signal after passing through the system

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

            self._intensity_array.append(intensity)
            self._phase_array.append(phase)

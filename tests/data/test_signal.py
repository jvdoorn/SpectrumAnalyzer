import shutil
import tempfile
import unittest

import numpy as np

from spectral.data.signal import Signal
from spectral.fourier import fourier_1d


class TestSignalGeneration(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency)

    def test_signal_length(self):
        expected_length = self.samples
        self.assertEqual(len(self.signal), expected_length)


class TestSignalTimestamps(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency)

    def testTimestampLength(self):
        self.assertEqual(self.samples, len(self.signal.timestamps))

    def testTimestampsStartAtZero(self):
        self.assertEqual(0, self.signal.timestamps[0])

    def testTimestampsEndAtRightTime(self):
        end_time = self.samples / self.sample_rate
        self.assertEqual(end_time, self.signal.timestamps[-1])


class TestSignalFFT(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5
        self.amplitude = 10
        self.df = 1

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency, self.amplitude)

    def test_nearest_frequency_finder(self):
        expected_index = int(self.frequency * self.samples / self.sample_rate)
        self.assertEqual(expected_index, self.signal.find_nearest_frequency_index(self.frequency))

    def test_fft_maximum_location(self):
        expected_index = self.signal.find_nearest_frequency_index(self.frequency)
        self.assertEqual(expected_index, np.argmax(self.signal.fft))

    def test_fft_maximum_value(self):
        ratio = 2
        signal2 = Signal.generate(self.sample_rate, self.samples, self.frequency, self.amplitude / ratio)

        signal_power = self.signal.power(self.frequency, 2)
        signal2_power = signal2.power(self.frequency, 2)
        power_ratio = signal_power / signal2_power
        self.assertEqual(ratio, power_ratio)

    def test_masked_minimum_zero(self):
        self.assertEqual(np.min(self.signal.masked_frequencies), 0)

    def test_masked_length(self):
        expected_length = len(self.signal.fft) // 2

        self.assertEqual(len(self.signal.masked_fft), expected_length)
        self.assertEqual(len(self.signal.masked_frequencies), expected_length)


class TestSignalMultiplication(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5
        self.amplitude = 5
        self.df = 0.01

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency, self.amplitude)

        self.window_width = 500
        self.window = np.zeros(self.samples)
        self.window[(self.samples - self.window_width) // 2: (self.samples + self.window_width) // 2] = 1
        self.window_signal = Signal(self.sample_rate, self.window)

        self.number = 5

    def test_multiplying_two_signals_linearity(self):
        windowed_signal = self.signal * self.window_signal
        windowed_signal2 = self.window_signal * self.signal

        self.assertTrue(np.array_equal(windowed_signal.samples, windowed_signal2.samples),
                        "Signal multiplication is not linear in the fft.")
        self.assertTrue(np.array_equal(windowed_signal.fft, windowed_signal2.fft),
                        "Signal multiplication is not linear in the samples.")

    def test_multiplying_signal_array_linearity(self):
        windowed_signal = self.signal * self.window
        windowed_signal2 = self.window * self.signal

        self.assertTrue(np.array_equal(windowed_signal.samples, windowed_signal2.samples),
                        "Signal multiplication is not linear in the samples.")
        self.assertTrue(np.array_equal(windowed_signal.fft, windowed_signal2.fft),
                        "Signal multiplication is not linear in the fft.")

    def test_multiplying_signal_float_linearity(self):
        modified_signal = self.signal * self.number
        modified_signal2 = self.number * self.signal

        self.assertTrue(np.array_equal(modified_signal.samples, modified_signal2.samples),
                        "Signal multiplication is not linear in the samples.")
        self.assertTrue(np.array_equal(modified_signal.fft, modified_signal2.fft),
                        "Signal multiplication is not linear in the fft.")

    def test_multiplying_two_signals_type(self):
        windowed_signal = self.signal * self.window_signal
        windowed_signal2 = self.window_signal * self.signal

        self.assertTrue(isinstance(windowed_signal, Signal), "Multiplying Signal with ndarray did not yield a Signal.")
        self.assertTrue(isinstance(windowed_signal2, Signal), "Multiplying Signal with ndarray did not yield a Signal.")

    def test_multiplying_signal_array_type(self):
        windowed_signal = self.signal * self.window
        windowed_signal2 = self.window * self.signal

        self.assertTrue(isinstance(windowed_signal, Signal), "Multiplying Signal with ndarray did not yield a Signal.")
        self.assertTrue(isinstance(windowed_signal2, Signal), "Multiplying Signal with ndarray did not yield a Signal.")

    def test_multiplying_signal_float_type(self):
        modified_signal = self.signal * self.number
        modified_signal2 = self.number * self.signal

        self.assertTrue(isinstance(modified_signal, Signal), "Multiplying Signal with float did not yield a Signal.")
        self.assertTrue(isinstance(modified_signal2, Signal), "Multiplying float with Signal did not yield a Signal.")

    def test_multiplying_two_signals_values(self):
        windowed_signal = self.signal * self.window_signal

        expected_samples = self.signal.samples * self.window_signal.samples
        expected_fft = fourier_1d(expected_samples)
        self.assertTrue(np.array_equal(expected_fft, windowed_signal.fft))

    def test_multiplying_signal_array_values(self):
        windowed_signal = self.signal * self.window

        expected_samples = self.signal.samples * self.window
        expected_fft = fourier_1d(expected_samples)
        self.assertTrue(np.array_equal(expected_fft, windowed_signal.fft))

    def test_multiplying_signal_float_values(self):
        modified_signal = self.signal * self.number

        expected_samples = self.signal.samples * self.number
        expected_fft = fourier_1d(expected_samples)
        self.assertTrue(np.array_equal(expected_fft, modified_signal.fft))

    def test_multiplying_two_signals_fft(self):
        windowed_signal = Signal(self.sample_rate, self.signal.samples * self.window)
        windowed_signal2 = self.signal * self.window_signal

        self.assertFalse(np.array_equal(windowed_signal2, self.signal),
                         "FFT was not transformed during signal multiplication.")
        self.assertTrue(np.array_equal(windowed_signal.fft, windowed_signal2.fft),
                        "FFT was not properly transformed during signal multiplication.")


class TestSaveSignalToFile(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.target_file = self.temporary_directory + '/signal.npz'
        self.master_file = 'tests/assets/signals/5hz_signal_measured_at_20000hz.npz'

        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5
        self.amplitude = 5
        self.df = 0.01

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency, self.amplitude)

    def test_save_signal_to_file(self):
        self.signal.save(self.master_file)

    def test_load_signal_from_file(self):
        loaded_signal = Signal.load(self.master_file)

        self.assertEqual(self.sample_rate, loaded_signal.sample_rate)
        self.assertEqual(len(self.signal), len(loaded_signal))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


if __name__ == '__main__':
    unittest.main()

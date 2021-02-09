import unittest

import numpy as np
from scipy.fft import fft

from spectral.data.signal import Signal


class TestSignalGeneration(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency)

    def test_signal_length(self):
        expected_length = self.samples
        self.assertEqual(len(self.signal), expected_length)


class TestSignalFFT(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5
        self.amplitude = 5
        self.df = 0.01

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency, self.amplitude)

    def test_nearest_frequency_finder(self):
        expected_index = int(self.frequency * self.samples / self.sample_rate)
        self.assertEqual(expected_index, self.signal.find_nearest_frequency_index(self.frequency))

    def test_fft_maximum_location(self):
        expected_index = self.signal.find_nearest_frequency_index(self.frequency)
        self.assertEqual(expected_index, np.argmax(self.signal.fft))

    def test_fft_maximum_value(self):
        expected_value = self.amplitude
        self.assertAlmostEqual(expected_value, self.signal.power(self.frequency, 2), delta=0.5)

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

    def test_multiplying_two_signals_values(self):
        windowed_signal = self.signal * self.window_signal

        expected_samples = self.signal.samples * self.window_signal.samples
        expected_fft = fft(expected_samples)
        self.assertTrue(np.array_equal(expected_fft, windowed_signal.fft))

    def test_multiplying_signal_array_values(self):
        windowed_signal = self.signal * self.window

        expected_samples = self.signal.samples * self.window
        expected_fft = fft(expected_samples)
        self.assertTrue(np.array_equal(expected_fft, windowed_signal.fft))

    def test_multiplying_two_signals_fft(self):
        windowed_signal = Signal(self.sample_rate, self.signal.samples * self.window)
        windowed_signal2 = self.signal * self.window_signal

        self.assertFalse(np.array_equal(windowed_signal2, self.signal),
                         "FFT was not transformed during signal multiplication.")
        self.assertTrue(np.array_equal(windowed_signal.fft, windowed_signal2.fft),
                        "FFT was not properly transformed during signal multiplication.")


if __name__ == '__main__':
    unittest.main()

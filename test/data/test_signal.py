import unittest

import numpy as np

from spectral.data.signal import Signal


class TestGeneratingSignal(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5
        self.amplitude = 5

        self.signal = Signal.generate(self.sample_rate, self.samples, self.frequency, self.amplitude)

    def test_signal_length(self):
        expected_length = self.samples
        self.assertEqual(len(self.signal), expected_length)


class TestSignal(unittest.TestCase):
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

    def test_nearest_frequency_finder(self):
        expected_index = int(self.frequency * self.samples / self.sample_rate)
        self.assertEqual(expected_index, self.signal.find_nearest_frequency_index(self.frequency))

    def test_fft_maximum_location(self):
        expected_index = self.signal.find_nearest_frequency_index(self.frequency)
        self.assertEqual(expected_index, np.argmax(self.signal.fft))

    def test_fft_maximum_value(self):
        expected_value = self.amplitude
        self.assertAlmostEqual(expected_value, self.signal.power(self.frequency, 2), delta=0.5)

    def test_multiplying_signals_linearity(self):
        window_signal = Signal(self.sample_rate, self.window)

        windowed_signal = window_signal * self.signal
        windowed_signal2 = self.signal * window_signal

        self.assertTrue(np.array_equal(windowed_signal.samples, windowed_signal2.samples),
                        "Signal multiplication is not linear.")

    def test_multiplying_signal_fft(self):
        window_signal = Signal(self.sample_rate, self.window)

        windowed_signal = Signal(self.sample_rate, self.signal.samples * self.window)
        windowed_signal2 = self.signal * window_signal

        self.assertFalse(np.array_equal(windowed_signal2, self.signal),
                         "FFT was not transformed during signal multiplication.")
        self.assertTrue(np.array_equal(windowed_signal.fft, windowed_signal2.fft),
                        "FFT was not properly transformed during signal multiplication.")


if __name__ == '__main__':
    unittest.main()

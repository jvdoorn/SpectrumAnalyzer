import shutil
import tempfile
import unittest

import numpy as np

from spectral.data.signal import Signal
from tests.utils import TEST_AMPLITUDE, TEST_DF, TEST_FREQUENCY, TEST_SAMPLES, TEST_SAMPLE_RATE, TEST_SIGNAL


class TestSignalGeneration(unittest.TestCase):
    def setUp(self):
        self.signal = TEST_SIGNAL

    def test_signal_length(self):
        expected_length = TEST_SAMPLES
        self.assertEqual(len(self.signal), expected_length)


class TestSignalTimestamps(unittest.TestCase):
    def setUp(self):
        self.signal = TEST_SIGNAL

    def testTimestampLength(self):
        self.assertEqual(TEST_SAMPLES, len(self.signal.timestamps))

    def testTimestampsStartAtZero(self):
        self.assertEqual(0, self.signal.timestamps[0])

    def testTimestampsEndAtRightTime(self):
        expected_end_time = TEST_SAMPLES / TEST_SAMPLE_RATE
        self.assertEqual(expected_end_time, self.signal.timestamps[-1])


class TestSignalFFT(unittest.TestCase):
    def setUp(self):
        self.signal = TEST_SIGNAL

    def test_nearest_frequency_finder(self):
        expected_index = int(TEST_FREQUENCY * TEST_SAMPLES / TEST_SAMPLE_RATE)
        self.assertEqual(expected_index, self.signal.find_nearest_frequency_index(TEST_FREQUENCY))

    def test_fft_maximum_location(self):
        expected_index = self.signal.find_nearest_frequency_index(TEST_FREQUENCY)
        self.assertEqual(expected_index, np.argmax(self.signal.fft))

    def test_fft_maximum_value(self):
        ratio = 2
        signal2 = Signal.generate(TEST_SAMPLE_RATE, TEST_SAMPLES, TEST_FREQUENCY, TEST_AMPLITUDE / ratio)

        signal_power = self.signal.power(TEST_FREQUENCY, TEST_DF)
        signal2_power = signal2.power(TEST_FREQUENCY, TEST_DF)
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
        self.signal = TEST_SIGNAL

        self.window_width = 500
        self.window = np.zeros(TEST_SAMPLES)
        self.window[(TEST_SAMPLES - self.window_width) // 2: (TEST_SAMPLES + self.window_width) // 2] = 1
        self.window_signal = Signal(TEST_SAMPLE_RATE, self.window)

        self.amplifier = 5

    def test_multiplying_two_signals_linearity(self):
        windowed_signal = self.signal * self.window_signal
        windowed_signal2 = self.window_signal * self.signal

        self.assertTrue(windowed_signal == windowed_signal2, "Signal multiplication is not linear.")

    def test_multiplying_signal_array_linearity(self):
        windowed_signal = self.signal * self.window
        windowed_signal2 = self.window * self.signal

        self.assertTrue(windowed_signal == windowed_signal2, "Signal ndarray multiplication is not linear.")

    def test_multiplying_signal_float_linearity(self):
        modified_signal = self.signal * self.amplifier
        modified_signal2 = self.amplifier * self.signal

        self.assertTrue(modified_signal == modified_signal2, "Signal with float multiplication is not linear.")

    def test_multiplying_two_signals_type(self):
        windowed_signal = self.signal * self.window_signal

        self.assertTrue(isinstance(windowed_signal, Signal), "Multiplying Signal did not yield a Signal.")

    def test_multiplying_signal_array_type(self):
        windowed_signal = self.signal * self.window
        windowed_signal2 = self.window * self.signal

        self.assertTrue(isinstance(windowed_signal, Signal), "Multiplying Signal with ndarray did not yield a Signal.")
        self.assertTrue(isinstance(windowed_signal2, Signal), "Multiplying ndarray with Signal did not yield a Signal.")

    def test_multiplying_signal_float_type(self):
        modified_signal = self.signal * self.amplifier
        modified_signal2 = self.amplifier * self.signal

        self.assertTrue(isinstance(modified_signal, Signal), "Multiplying Signal with float did not yield a Signal.")
        self.assertTrue(isinstance(modified_signal2, Signal), "Multiplying float with Signal did not yield a Signal.")

    def test_multiplying_two_signals_values(self):
        windowed_signal = self.signal * self.window_signal
        expected_signal = Signal(TEST_SAMPLE_RATE, self.signal.samples * self.window_signal.samples)

        self.assertTrue(windowed_signal, expected_signal)

    def test_multiplying_signal_array_values(self):
        windowed_signal = self.signal * self.window
        expected_signal = Signal(TEST_SAMPLE_RATE, self.signal.samples * self.window)

        self.assertTrue(windowed_signal == expected_signal)

    def test_multiplying_signal_float_values(self):
        amplified_signal = self.signal * self.amplifier
        expected_signal = Signal(TEST_SAMPLE_RATE, self.signal.samples * self.amplifier)

        self.assertTrue(amplified_signal == expected_signal)


class TestSaveSignalToFile(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.target_file = self.temporary_directory + '/5hz_signal_measured_at_20000hz.npz'
        self.master_file = 'tests/assets/signals/5hz_signal_measured_at_20000hz.npz'

        self.signal = TEST_SIGNAL

    def test_save_signal_to_file(self):
        self.signal.save(self.target_file)

        loaded_signal = Signal.load(self.target_file)
        self.assertEqual(TEST_SAMPLE_RATE, loaded_signal.sample_rate)
        self.assertEqual(len(self.signal), len(loaded_signal))

    def test_load_signal_from_file(self):
        loaded_signal = Signal.load(self.master_file)

        self.assertEqual(TEST_SAMPLE_RATE, loaded_signal.sample_rate)
        self.assertEqual(len(self.signal), len(loaded_signal))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


if __name__ == '__main__':
    unittest.main()

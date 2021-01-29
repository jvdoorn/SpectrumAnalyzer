import unittest

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

    def test_fft_maximum_location(self):
        expected_index = int(self.frequency * self.samples / self.sample_rate)
        self.assertAlmostEqual(expected_index, self.signal.find_nearest_frequency_index(self.frequency))

    def test_fft_maximum_value(self):
        expected_value = self.amplitude
        self.assertAlmostEqual(expected_value, self.signal.power(self.frequency, 2), delta=0.5)


if __name__ == '__main__':
    unittest.main()

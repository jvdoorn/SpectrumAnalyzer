import unittest

from numpy import linspace, pi, sin

from spectral.data.signal import Signal


class TestSignalPower(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.samples = 10000
        self.frequency = 5
        self.amplitude = 5

        self.t = linspace(0, self.samples / self.sample_rate, self.samples)

    def test_fft_maximum(self):
        data = self.amplitude * sin(2 * pi * self.frequency * self.t)
        signal = Signal(self.sample_rate, data)

        expected_index = int(self.frequency * self.samples / self.sample_rate)
        self.assertAlmostEqual(expected_index, signal.find_nearest_frequency_index(self.frequency))


if __name__ == '__main__':
    unittest.main()

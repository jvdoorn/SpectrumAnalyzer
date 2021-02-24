import unittest

import numpy as np

from spectral.analysis.analyzer import DAQAnalyzer
from spectral.aquisition.daq import DataAcquisitionInterface
from spectral.data.results import SystemResponse


class TestDAQSampleRate(unittest.TestCase):
    def test_sample_rate_limits(self):
        minimum_sample_rate = 10
        maximum_sample_rate = 100

        class DAQInterfaceMock(DataAcquisitionInterface):
            MINIMUM_SAMPLE_RATE = minimum_sample_rate
            MAXIMUM_SAMPLE_RATE = maximum_sample_rate

        self.assertRaises(AssertionError, lambda: DAQInterfaceMock(minimum_sample_rate - 1))
        self.assertRaises(AssertionError, lambda: DAQInterfaceMock(maximum_sample_rate + 1))

    def test_sample_rate_value(self):
        initalization_sample_rate = 10
        updated_sample_rate = 20

        daq = DataAcquisitionInterface(initalization_sample_rate)
        self.assertEqual(daq.sample_rate, initalization_sample_rate)

        daq.sample_rate = updated_sample_rate
        self.assertEqual(daq.sample_rate, updated_sample_rate)


class TestDAQTimeArray(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 100
        self.samples = 10000

        self.daq = DataAcquisitionInterface(self.sample_rate)
        self.time_array = self.daq.calculate_time_array(self.samples)

    def test_time_array_dimensions(self):
        self.assertEqual(1, len(self.time_array.shape), "The time array is not a 1D-ndarray.")
        self.assertEqual((self.samples,), self.time_array.shape)

    def test_time_array_values(self):
        expected_end_time = self.samples / self.sample_rate

        self.assertEqual(0, self.time_array[0], "Time array did not start at 0.")
        self.assertEqual(expected_end_time, self.time_array[-1], f"Time array did not end at {expected_end_time}.")


class TestDAQAnalyzerRead(unittest.TestCase):
    def setUp(self):
        class DAQMock(DataAcquisitionInterface):
            MOCK_FREQUENCY = 300
            MOCK_AMPLITUDE = 5

            def read(self, channels: np.ndarray, samples: int) -> np.ndarray:
                end_time = samples / self.sample_rate

                single_time_array = np.linspace(0, end_time, samples)
                time_array = np.tile(single_time_array, len(channels)).reshape((len(channels), samples))

                signal = self.MOCK_AMPLITUDE * np.sin(2 * np.pi * self.MOCK_FREQUENCY * time_array)
                return signal

        self.sample_rate = 5000
        self.samples = 20000

        self.df = 20

        self.daq = DAQMock(self.sample_rate)
        self.analyzer = DAQAnalyzer(self.daq, self.df)

    def test_measuring_single(self):
        response = self.analyzer.measure_single(self.samples)

        self.assertTrue(isinstance(response, SystemResponse))

        self.assertAlmostEqual(0, response.relative_phase(self.daq.MOCK_FREQUENCY))
        self.assertAlmostEqual(1, response.relative_intensity(self.daq.MOCK_FREQUENCY, 3))


if __name__ == '__main__':
    unittest.main()

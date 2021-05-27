import unittest

import numpy as np

from specc.analysis.analyzer import CircuitTester
from specc.aquisition.daq import DataAcquisitionInterface
from specc.data.results import SignalResponse
from tests.utils import ACCEPTABLE_ERROR, TEST_AMPLITUDE, TEST_DF, TEST_FREQUENCY, TEST_SAMPLES, TEST_SAMPLE_RATE


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
        self.daq = DataAcquisitionInterface(TEST_SAMPLE_RATE)
        self.time_array = self.daq.calculate_time_array(TEST_SAMPLES)

    def test_time_array_dimensions(self):
        self.assertEqual(1, len(self.time_array.shape), "The time array is not a 1D-ndarray.")
        self.assertEqual((TEST_SAMPLES,), self.time_array.shape)

    def test_time_array_values(self):
        expected_end_time = TEST_SAMPLES / TEST_SAMPLE_RATE

        self.assertEqual(0, self.time_array[0], "Time array did not start at 0.")
        self.assertEqual(expected_end_time, self.time_array[-1], f"Time array did not end at {expected_end_time}.")


class TestDAQAnalyzerRead(unittest.TestCase):
    def setUp(self):
        class DAQMock(DataAcquisitionInterface):
            MOCK_OUTPUT_PHASE = np.pi / 4

            def read(self, channels: np.ndarray, samples: int) -> np.ndarray:
                end_time = samples / self.sample_rate

                time_array = np.linspace(0, end_time, samples)

                if len(channels) == 1:
                    signal = TEST_AMPLITUDE * np.sin(2 * np.pi * TEST_FREQUENCY * time_array)
                elif len(channels) == 2:
                    signal = np.asarray([
                        TEST_AMPLITUDE * np.sin(2 * np.pi * TEST_FREQUENCY * time_array),
                        TEST_AMPLITUDE * np.sin(2 * np.pi * TEST_FREQUENCY * time_array + self.MOCK_OUTPUT_PHASE),
                    ])
                else:
                    raise NotImplementedError

                return signal

        self.daq = DAQMock(TEST_SAMPLE_RATE)
        self.analyzer = CircuitTester(self.daq)

    def test_measuring_single(self):
        response = self.analyzer.measure_single(TEST_SAMPLES)

        self.assertTrue(isinstance(response, SignalResponse))

        self.assertAlmostEqual(self.daq.MOCK_OUTPUT_PHASE, response.relative_phase(TEST_FREQUENCY),
                               delta=ACCEPTABLE_ERROR)
        self.assertAlmostEqual(1, response.relative_intensity(TEST_FREQUENCY, TEST_DF), delta=ACCEPTABLE_ERROR)


if __name__ == '__main__':
    unittest.main()

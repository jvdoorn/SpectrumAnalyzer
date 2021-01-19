import unittest

from spectral.aquisition.daq import DataAcquisitionInterface


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


if __name__ == '__main__':
    unittest.main()

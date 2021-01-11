import unittest

from spectral.measure.measurer import Measurer


class TestMeasurerSampleRate(unittest.TestCase):
    def test_sample_rate_limits(self):
        minimum_sample_rate = 10
        maximum_sample_rate = 100

        class MeasurerMock(Measurer):
            MINIMUM_SAMPLE_RATE = minimum_sample_rate
            MAXIMUM_SAMPLE_RATE = maximum_sample_rate

        self.assertRaises(AssertionError, lambda: MeasurerMock(minimum_sample_rate - 1))
        self.assertRaises(AssertionError, lambda: MeasurerMock(maximum_sample_rate + 1))

    def test_sample_rate_value(self):
        initalization_sample_rate = 10
        updated_sample_rate = 20

        measurer = Measurer(initalization_sample_rate)
        self.assertEqual(measurer.sample_rate, initalization_sample_rate)

        measurer.sample_rate = updated_sample_rate
        self.assertEqual(measurer.sample_rate, updated_sample_rate)


class TestMeasurerTimeArray(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 100
        self.samples = 10000

        self.time_array = Measurer(self.sample_rate).calculate_time_array(self.samples)

    def test_time_array_dimensions(self):
        self.assertEqual(1, len(self.time_array.shape), "The time array is not a 1D-ndarray.")
        self.assertEqual((self.samples,), self.time_array.shape)

    def test_time_array_values(self):
        self.assertEqual(0, self.time_array[0], "Time array did not start at 0.")
        self.assertEqual(self.samples / self.sample_rate, self.time_array[-1],
                         f"Time array did not end at {self.samples / self.sample_rate}.")


if __name__ == '__main__':
    unittest.main()

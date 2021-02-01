import unittest

import numpy as np

from spectral.analysis.analyzer import Analyzer, DAQAnalyzer
from spectral.aquisition.daq import DataAcquisitionInterface
from spectral.data.results import SystemResponse, TransferFunctionBehaviour
from spectral.utils import latex_float


class TestPlottingKnownTransferFunction(unittest.TestCase):
    def setUp(self):
        self.RC = 10 ** -3
        self.RC_neat = latex_float(self.RC)
        self.low_pass = lambda f: 1 / (1 + (1j * self.RC * 2 * np.pi * f))
        self.high_pass = lambda f: 1 / (1 + 1 / (1j * self.RC * 2 * np.pi * f))

        self.frequencies = np.logspace(0, 4, 8 * 12)

    def test_plot_high_pass(self):
        analyzer = Analyzer()

        high_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.high_pass)
        analyzer.plot(f"Prediction of high pass filter with $RC={self.RC_neat}$.", high_pass_behaviour, save=False)

    def test_plot_low_pass(self):
        analyzer = Analyzer()

        low_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.low_pass)
        analyzer.plot(f"Prediction of low pass filter with $RC={self.RC_neat}$.", low_pass_behaviour, save=False)


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

        self.daq = DAQMock(self.sample_rate)
        self.analyzer = DAQAnalyzer(daq=self.daq)

    def test_measuring_single(self):
        response = self.analyzer.measure_single(self.samples)

        self.assertTrue(isinstance(response, SystemResponse))

        self.assertAlmostEqual(0, response.relative_phase(self.daq.MOCK_FREQUENCY))
        self.assertAlmostEqual(1, response.relative_intensity(self.daq.MOCK_FREQUENCY, 3))


if __name__ == '__main__':
    unittest.main()

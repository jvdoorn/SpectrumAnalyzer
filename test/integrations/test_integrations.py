import unittest

import numpy as np

from spectral.analysis.analyzer import SimulationAnalyzer
from spectral.utils import latex_float


class TestPlottingKnownTransferFunction(unittest.TestCase):
    def setUp(self):
        self.RC = 10 ** -3
        self.RC_neat = latex_float(self.RC)
        self.low_pass = lambda f: 1 / (1 + (1j * self.RC * 2 * np.pi * f))
        self.high_pass = lambda f: 1 / (1 + 1 / (1j * self.RC * 2 * np.pi * f))

        self.frequencies = np.logspace(0, 4, 8 * 12)

    def test_plot_high_pass(self):
        analyzer = SimulationAnalyzer()

        analyzer.plot(f"Prediction of high pass filter with $RC={self.RC_neat}$.",
                      analyzer.predict(self.frequencies, self.high_pass), save=False)

    def test_plot_low_pass(self):
        analyzer = SimulationAnalyzer()
        analyzer.plot(f"Prediction of low pass filter with $RC={self.RC_neat}$.",
                      analyzer.predict(self.frequencies, self.low_pass), save=False)


if __name__ == '__main__':
    unittest.main()

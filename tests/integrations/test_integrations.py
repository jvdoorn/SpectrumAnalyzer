import unittest

import numpy as np

from spectral.data.results import TransferFunctionBehaviour
from spectral.plotting import plot
from spectral.utils import latex_float


class TestPlottingKnownTransferFunction(unittest.TestCase):
    def setUp(self):
        self.RC = 10 ** -3
        self.RC_neat = latex_float(self.RC)

        self.low_pass = lambda f: 1 / (1 + (1j * self.RC * 2 * np.pi * f))
        self.high_pass = lambda f: 1 / (1 + 1 / (1j * self.RC * 2 * np.pi * f))

        self.frequencies = np.logspace(0, 4, 8 * 12)

        self.df = 20

    def test_plot_high_pass(self):
        high_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.high_pass)
        plot(high_pass_behaviour, f"Prediction of high pass filter with $RC={self.RC_neat}$.")

    def test_plot_low_pass(self):
        low_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.low_pass)
        plot(low_pass_behaviour, f"Prediction of low pass filter with $RC={self.RC_neat}$.")


if __name__ == '__main__':
    unittest.main()

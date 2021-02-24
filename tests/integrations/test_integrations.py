import shutil
import tempfile
import unittest

import numpy as np

from spectral.data.results import TransferFunctionBehaviour
from spectral.plotting import plot
from spectral.utils import latex_float
from tests.utils import equal_file_hash


class TestPlottingKnownTransferFunction(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()

        self.RC = 10 ** -3
        self.RC_neat = latex_float(self.RC)

        self.low_pass = lambda f: 1 / (1 + (1j * self.RC * 2 * np.pi * f))
        self.high_pass = lambda f: 1 / (1 + 1 / (1j * self.RC * 2 * np.pi * f))

        self.frequencies = np.logspace(0, 4, 8 * 12)

        self.df = 20

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)

    def test_plot_high_pass(self):
        target_file = self.temporary_directory + "/high_pass.png"
        master_file = "tests/assets/images/high_pass.png"

        high_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.high_pass)
        plot(high_pass_behaviour, f"Prediction of high pass filter with $RC={self.RC_neat}$.").savefig(target_file)
        self.assertTrue(equal_file_hash(master_file, target_file))

    def test_plot_low_pass(self):
        target_file = self.temporary_directory + "/low_pass.png"
        master_file = "tests/assets/images/low_pass.png"

        low_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.low_pass)
        plot(low_pass_behaviour, f"Prediction of low pass filter with $RC={self.RC_neat}$.").savefig(target_file)
        self.assertTrue(equal_file_hash(master_file, target_file))


if __name__ == '__main__':
    unittest.main()

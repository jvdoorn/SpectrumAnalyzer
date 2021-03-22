import shutil
import tempfile
import unittest

import numpy as np

from spectral.data.results import TransferFunctionBehaviour
from spectral.data.signal import Signal
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

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


class TestLoadingAndPlottingSignalFromFile(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.signal_file = 'tests/assets/data/440hz_tuning_fork_measured_at_4000hz.csv'

        self.tuning_fork_frequency = 440
        self.sample_rate = 4000

        self.signal = Signal.load(self.signal_file, self.sample_rate)

    def test_plot_signal(self):
        target_file = self.temporary_directory + "/440hz_tuning_fork_measured_at_4000hz.png"
        master_file = "tests/assets/images/440hz_tuning_fork_measured_at_4000hz.png"

        plot(self.signal, '440Hz tuning fork measured at 4000Hz').savefig(target_file)
        self.assertTrue(equal_file_hash(master_file, target_file))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


if __name__ == '__main__':
    unittest.main()

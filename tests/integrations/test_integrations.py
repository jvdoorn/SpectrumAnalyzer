import shutil
import tempfile
import unittest
from typing import Union

import numpy as np

from specc.analysis.converter import Converter
from specc.data.results import TransferFunctionBehaviour
from specc.data.signal import Signal
from specc.plotting import plot
from specc.utils import latex_float
from tests.utils import equal_file_hash


class TestPlottingKnownTransferFunction(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()

        self.RC = 10 ** -3
        self.RC_neat = latex_float(self.RC)

        self.low_pass = lambda f: 1 / (1 + (1j * self.RC * 2 * np.pi * f))
        self.high_pass = lambda f: 1 / (1 + 1 / (1j * self.RC * 2 * np.pi * f))

        self.frequencies = np.logspace(0, 4, 8 * 12)

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

    def test_plot_low_and_high_pass(self):
        target_file = self.temporary_directory + "/low_high_pass.png"
        master_file = "tests/assets/images/low_high_pass.png"

        high_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.high_pass)
        low_pass_behaviour = TransferFunctionBehaviour(self.frequencies, self.low_pass)
        plot([low_pass_behaviour, high_pass_behaviour],
             f"Prediction of low and high pass filter with $RC={self.RC_neat}$.", ["Low pass", "High pass"]).savefig(
            target_file)
        self.assertTrue(equal_file_hash(master_file, target_file))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


class TestLoadingAndPlottingSignalFromFile(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.signal_file = 'tests/assets/signals/440hz_tuning_fork_measured_at_4000hz.npz'

        self.tuning_fork_frequency = 440
        self.sample_rate = 4000

        self.signal = Signal.load(self.signal_file)

    def test_plot_signal(self):
        target_file = self.temporary_directory + "/440hz_tuning_fork_measured_at_4000hz.png"
        master_file = "tests/assets/images/440hz_tuning_fork_measured_at_4000hz.png"

        plot(self.signal, '440Hz tuning fork measured at 4000Hz').savefig(target_file)
        self.assertTrue(equal_file_hash(master_file, target_file))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


class TestSignalWithConverter(unittest.TestCase):
    def setUp(self):
        class ConverterMock(Converter):
            def __init__(self):
                super().__init__('$\\degree$C')

                self.V_in = 5
                self.R1 = 5.08e3

                # TTC05682 properties
                self.R0 = 6800
                self.T0 = 25 + 273.25

                self.B = 4050

                self.sigma_T = 0.01

            def R(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
                return self.R1 * (self.V_in / V - 1)

            def convert(self, data: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
                return 1 / (1 / self.T0 + (1 / self.B) * np.log(self.R(data) / self.R0)) - 273.25

            def error(self, data: Union[np.ndarray, float]) -> float:
                return 3

        self.temporary_directory = tempfile.mkdtemp()
        self.signal_file = 'tests/assets/signals/ntc_voltage.npz'

        self.signal = Signal.load(self.signal_file, converter=ConverterMock())

    def test_plot_signal(self):
        target_file = self.temporary_directory + "/ntc_temperature.png"
        master_file = "tests/assets/images/ntc_temperature.png"
        plot(self.signal, 'NTC temperature').savefig(target_file)
        self.assertTrue(equal_file_hash(master_file, target_file))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


if __name__ == '__main__':
    unittest.main()

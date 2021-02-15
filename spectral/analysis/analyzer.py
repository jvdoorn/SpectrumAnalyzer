"""
This is the main class containing the Analyzer base class
and two sub classes.
"""
from os import makedirs
from typing import Type

import numpy as np
from tqdm import tqdm

from spectral.aquisition.daq import DataAcquisitionInterface
from spectral.data.results import FrequencyResponse, SystemBehaviour, SystemResponse
from spectral.data.signal import Signal
from spectral.utils import timestamp


class DAQAnalyzer:
    def __init__(self, daq: Type[DataAcquisitionInterface], df: int, base_directory: str = "data/",
                 write_channel: str = "myDAQ1/AO0", pre_system_channel: str = "myDAQ1/AI0",
                 post_system_channel: str = "myDAQ1/AI1"):
        self._df = df

        self._daq = daq

        self._base_directory = base_directory

        self._write_channel = write_channel
        self._pre_system_channel = pre_system_channel
        self._post_system_channel = post_system_channel

    def measure_single(self, samples: int) -> SystemResponse:
        """
        Used to measure a signal before and after passing through a system. Useful when using other hardware to drive
        the system.
        :param samples: the amount of samples.
        :return: the response of the system.
        """
        data = self._daq.read([self._pre_system_channel, self._post_system_channel], samples)

        pre_system_signal = Signal(self._daq.sample_rate, data[0])
        post_system_signal = Signal(self._daq.sample_rate, data[1])

        return SystemResponse(pre_system_signal, post_system_signal)

    def drive_and_measure_single(self, frequency: float, data_directory: str, samples: int) -> SystemResponse:
        """
        Send a signal to a channel and measures the output.
        :param frequency: the frequency to measure.
        :param data_directory: the directory to save the data to.
        :param samples: the amount of samples.
        :return: the response of the system.
        """
        artificial_signal = Signal.generate(self._daq.sample_rate, samples, frequency)

        data = self._daq.read_write(artificial_signal.samples, [self._write_channel],
                                    [self._pre_system_channel, self._post_system_channel], samples)
        np.savetxt(f"{data_directory}{frequency}.csv", data)

        pre_system_signal = Signal(self._daq.sample_rate, data[0])
        post_system_signal = Signal(self._daq.sample_rate, data[1])

        return SystemResponse(pre_system_signal, post_system_signal)

    def drive_and_measure_multiple(self, frequencies: list, samples: int) -> SystemBehaviour:
        """
        Sends a series of signals to a channel and measures the output.
        :param frequencies: the frequencies to measure.
        :param samples: the amount of samples.
        :return: the frequencies, magnitudes and phases.
        """
        data_directory = f"{self._base_directory}{timestamp()}/"
        makedirs(data_directory)

        behaviour = SystemBehaviour()
        for frequency in tqdm(frequencies):
            response = self.drive_and_measure_single(frequency, data_directory, samples)
            response = FrequencyResponse(response.relative_intensity(frequency, self._df),
                                         response.relative_phase(frequency))

            behaviour.add_response(frequency, response)
        return behaviour

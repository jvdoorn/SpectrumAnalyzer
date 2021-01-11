"""
This file contains a class for interfacing with a
NI MyDAQ. It has various methods to read and write
signals.
"""

import time
from typing import Tuple

import nidaqmx as dx
import numpy as np

from spectral.measure.measurer import Measurer


class MyDAQ(Measurer):
    def write(self, voltages: np.ndarray, channels: np.ndarray, samples: int):
        """
        Writes N signal arrays to N channels. It repeats the signal if the length
        of the individual signal arrays is smaller than the sample count.
        :param voltages: a 2D-ndarray, unless channels contains only one channel, then a 1D-ndarray.
        :param channels: a ndarray with channel names.
        :param samples: the number of samples.
        :return:
        """
        with dx.Task() as task:
            for channel in channels:
                task.ao_channels.add_ao_voltage_chan(channel)

            task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=dx.constants.AcquisitionType.FINITE,
                                            samps_per_chan=np.long(samples))
            task.write(voltages, auto_start=True)
            time.sleep(np.long(samples) / self.sample_rate + 0.0001)
            task.stop()

    def read(self, channels: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads N signals from N channels. It also calculates the corresponding timestamps.
        :param channels: a ndarray with channel names.
        :param samples: the number of samples.
        :return: a ndarray containing the signals and a ndarray with the timestamps.
        """
        with dx.Task() as task:
            for channel in channels:
                task.ai_channels.add_ai_voltage_chan(channel)

            task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=dx.constants.AcquisitionType.FINITE,
                                            samps_per_chan=np.long(samples))
            return task.read(number_of_samples_per_channel=samples,
                             timeout=dx.constants.WAIT_INFINITELY), self.calculate_time_array(samples)

    def read_write(self, voltages: np.ndarray, write_channels: np.ndarray, read_channel: np.ndarray, samples: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Write N signals to N channels and reads M signals from M channels.
        :param voltages: a 2D-ndarray, unless write_channels contains only one channel, then a 1D-ndarray.
        :param write_channels: a ndarray with channel names to write to.
        :param read_channel: a ndarray with channel names to read from.
        :param samples: the number of samples.
        :return: a ndarray containing the signals and a ndarray with the timestamps.
        """
        with dx.Task() as write_task, dx.Task() as read_task:
            for channel in write_channels:
                write_task.ao_channels.add_ao_voltage_chan(channel)

            for channel in read_channel:
                read_task.ai_channels.add_ai_voltage_chan(channel)

            read_task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=dx.constants.AcquisitionType.FINITE,
                                                 samps_per_chan=np.long(samples))
            write_task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=dx.constants.AcquisitionType.FINITE,
                                                  samps_per_chan=np.long(samples))

            write_task.write(voltages, auto_start=True)
            data = read_task.read(number_of_samples_per_channel=samples, timeout=dx.constants.WAIT_INFINITELY)
            time.sleep(np.long(samples) / self.sample_rate + 0.0001)
            write_task.stop()
            return data, self.calculate_time_array(samples)

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
    MAXIMUM_SAMPLE_RATE = int(2e5)

    @staticmethod
    def register_output_channels(task: dx.Task, channels: np.ndarray):
        for channel in channels:
            task.ao_channels.add_ao_voltage_chan(channel)

    @staticmethod
    def register_input_channels(task: dx.Task, channels: np.ndarray):
        for channel in channels:
            task.ai_channels.add_ai_voltage_chan(channel)

    def configure_timings(self, task: dx.Task, samples: int):
        task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=dx.constants.AcquisitionType.FINITE,
                                        samps_per_chan=np.long(samples))

    def write(self, voltages: np.ndarray, channels: np.ndarray, samples: int):
        with dx.Task() as task:
            self.register_output_channels(task, channels)
            self.configure_timings(task, samples)

            task.write(voltages, auto_start=True)
            time.sleep(np.long(samples) / self.sample_rate + 0.0001)
            task.stop()

    def read(self, channels: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        with dx.Task() as task:
            self.register_input_channels(task, channels)
            self.configure_timings(task, samples)

            data = task.read(number_of_samples_per_channel=samples, timeout=dx.constants.WAIT_INFINITELY)
            return data, self.calculate_time_array(samples)

    def read_write(self, voltages: np.ndarray, write_channels: np.ndarray, read_channels: np.ndarray, samples: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        with dx.Task() as write_task, dx.Task() as read_task:
            self.register_output_channels(write_task, write_channels)
            self.register_input_channels(read_task, read_channels)

            self.configure_timings(write_task, samples)
            self.configure_timings(read_task, samples)

            write_task.write(voltages, auto_start=True)
            data = read_task.read(number_of_samples_per_channel=samples, timeout=dx.constants.WAIT_INFINITELY)
            time.sleep(np.long(samples) / self.sample_rate + 0.0001)
            write_task.stop()
            return data, self.calculate_time_array(samples)

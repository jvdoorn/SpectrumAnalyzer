"""
This file contains a class for interfacing with a
NI MyDAQ. It has various methods to read and write
signals.
"""

import time

import nidaqmx as dx
import numpy as np

from specc.aquisition import CHANNEL_TYPE
from specc.aquisition.daq import DataAcquisitionInterface


class NIMyDAQInterface(DataAcquisitionInterface):
    MAXIMUM_SAMPLE_RATE = int(2e5)

    @staticmethod
    def _register_output_channels(task: dx.Task, channels: CHANNEL_TYPE):
        if isinstance(channels, str):
            NIMyDAQInterface._register_output_channels(task, [channels])
        else:
            for channel in channels:
                task.ao_channels.add_ao_voltage_chan(channel)

    @staticmethod
    def _register_input_channels(task: dx.Task, channels: CHANNEL_TYPE):
        if isinstance(channels, str):
            NIMyDAQInterface._register_input_channels(task, [channels])
        else:
            for channel in channels:
                task.ai_channels.add_ai_voltage_chan(channel)

    @staticmethod
    def _assert_channel_dimensions_match(write_channels: CHANNEL_TYPE, voltages: np.ndarray):
        channel_amount = 1 if isinstance(write_channels, str) else len(write_channels)
        if channel_amount == 1:
            assert len(voltages.shape) == 1, "Expected voltages to be a 1D-ndarray."
        else:
            assert channel_amount == voltages.shape[0], f"Expected voltages to have {channel_amount} rows."

    def _configure_timings(self, task: dx.Task, samples: int):
        task.timing.cfg_samp_clk_timing(self.sample_rate, sample_mode=dx.constants.AcquisitionType.FINITE,
                                        samps_per_chan=np.long(samples))

    def write(self, voltages: np.ndarray, channels: CHANNEL_TYPE, samples: int):
        self._assert_channel_dimensions_match(channels, voltages)

        with dx.Task() as task:
            self._register_output_channels(task, channels)
            self._configure_timings(task, samples)

            task.write(voltages, auto_start=True)
            time.sleep(np.long(samples) / self.sample_rate + 0.0001)
            task.stop()

    def read(self, channels: CHANNEL_TYPE, samples: int) -> np.ndarray:
        with dx.Task() as task:
            self._register_input_channels(task, channels)
            self._configure_timings(task, samples)

            data = task.read(number_of_samples_per_channel=samples, timeout=dx.constants.WAIT_INFINITELY)
            return np.asarray(data)

    def read_write(self, voltages: np.ndarray, write_channels: CHANNEL_TYPE, read_channels: CHANNEL_TYPE,
                   samples: int) -> np.ndarray:
        self._assert_channel_dimensions_match(write_channels, voltages)

        with dx.Task() as write_task, dx.Task() as read_task:
            self._register_output_channels(write_task, write_channels)
            self._register_input_channels(read_task, read_channels)

            self._configure_timings(write_task, samples)
            self._configure_timings(read_task, samples)

            write_task.write(voltages, auto_start=True)
            data = read_task.read(number_of_samples_per_channel=samples, timeout=dx.constants.WAIT_INFINITELY)
            time.sleep(np.long(samples) / self.sample_rate + 0.0001)
            write_task.stop()
            return np.asarray(data)

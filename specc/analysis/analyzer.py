from typing import Type

from tqdm import tqdm

from specc.aquisition.daq import DataAcquisitionInterface
from specc.data.results import FrequencyResponse, SignalResponse, SystemBehaviour
from specc.data.signal import Signal


class CircuitTester:
    """
    An easy to use class for testing a circuit. Provides a method to just analyze the behaviour based on other input and
    provides a method to supply the input and measure the response.
    """

    def __init__(self, daq: Type[DataAcquisitionInterface], pre_system_channel: str,
                 post_system_channel: str, write_channel: str = None):
        self._daq = daq

        self._pre_system_channel = pre_system_channel
        self._post_system_channel = post_system_channel
        self._write_channel = write_channel

    def measure_single(self, samples: int) -> SignalResponse:
        """
        Used to measure a signal before and after passing through a system. Useful when using other hardware to drive
        the system.
        :param samples: the amount of samples.
        :return: the response of the system.
        """
        data = self._daq.read([self._pre_system_channel, self._post_system_channel], samples)

        pre_system_signal = Signal(self._daq.sample_rate, data[0])
        post_system_signal = Signal(self._daq.sample_rate, data[1])

        return SignalResponse(pre_system_signal, post_system_signal)

    def drive_and_measure_single(self, frequency: float, samples: int) -> SignalResponse:
        """
        Send a signal to a channel and measures the output.
        :param frequency: the frequency to measure.
        :param samples: the amount of samples.
        :return: the response of the system.
        """
        artificial_signal = Signal.generate(self._daq.sample_rate, samples, frequency)

        data = self._daq.read_write(artificial_signal.samples, [self._write_channel],
                                    [self._pre_system_channel, self._post_system_channel], samples)

        pre_system_signal = Signal(self._daq.sample_rate, data[0])
        post_system_signal = Signal(self._daq.sample_rate, data[1])

        return SignalResponse(pre_system_signal, post_system_signal)

    def drive_and_measure_multiple(self, frequencies: list, samples: int, df: float) -> SystemBehaviour:
        """
        Sends a series of signals to a channel and measures the output.
        :param frequencies: the frequencies to measure.
        :param samples: the amount of samples.
        :return: the frequencies, magnitudes and phases.
        """
        behaviour = SystemBehaviour()
        for frequency in tqdm(frequencies):
            response = self.drive_and_measure_single(frequency, samples)
            response = FrequencyResponse.from_signal_response(response, frequency, df)

            behaviour.add_response(frequency, response)
        return behaviour

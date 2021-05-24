from numpy import linspace, ndarray

from specc.aquisition import CHANNEL_TYPE


class DataAcquisitionInterface:
    MAXIMUM_SAMPLE_RATE = float('inf')
    MINIMUM_SAMPLE_RATE = 0

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        assert sample_rate >= self.MINIMUM_SAMPLE_RATE, f"Sample rate must be bigger than {self.MINIMUM_SAMPLE_RATE}."
        assert sample_rate <= self.MAXIMUM_SAMPLE_RATE, f"Sample rate must be smaller than {self.MAXIMUM_SAMPLE_RATE}."

        self._sample_rate = sample_rate

    def calculate_time_array(self, samples: int) -> ndarray:
        """
        Calculates the timestamps that belong to a signal.
        :param samples: the number of samples.
        :return: a 1D-ndarray with the timestamps.
        """
        return linspace(0, samples / self.sample_rate, samples)

    def write(self, voltages: ndarray, channels: CHANNEL_TYPE, samples: int) -> None:
        """
        Writes N signal arrays to N channels. It repeats the signal if the length
        of the individual signal arrays is smaller than the sample count.
        :param voltages: a 2D-ndarray containing N signals.
        :param channels: a 1D-ndarray with N channel names.
        :param samples: the number of samples.
        """
        raise NotImplementedError

    def read(self, channels: CHANNEL_TYPE, samples: int) -> ndarray:
        """
        Reads N signals from N channels. It also calculates the corresponding timestamps.
        :param channels: a 1D-ndarray containing N channel names.
        :param samples: the number of samples.
        :return: a 2D-ndarray containing the signals and a 1D-ndarray with the timestamps.
        """
        raise NotImplementedError

    def read_write(self, voltages: ndarray, write_channels: CHANNEL_TYPE, read_channels: CHANNEL_TYPE,
                   samples: int) -> ndarray:
        """
        Write N signals to N channels and read M signals from M channels.
        :param voltages: a 2D-ndarray containing N signals.
        :param write_channels: a 1D-ndarray containing N channel names to write to.
        :param read_channels: a 1D-ndarray containing M channel names to read from.
        :param samples: the number of samples.
        :return: a 2D-ndarray containing the read signals and a 1D-ndarray with the timestamps.
        """
        raise NotImplementedError

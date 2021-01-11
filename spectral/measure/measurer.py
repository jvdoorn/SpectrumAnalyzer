from numpy import linspace, ndarray


class Measurer:
    MAXIMUM_SAMPLE_RATE = float('inf')
    MINIMUM_SAMPLE_RATE = 1

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        assert sample_rate >= self.MINIMUM_SAMPLE_RATE, f"Sample rate must be bigger than {self.MINIMUM_SAMPLE_RATE}."
        assert sample_rate <= self.MAXIMUM_SAMPLE_RATE, f"Sample rate must be smaller than {self.MINIMUM_SAMPLE_RATE}."

        self._sample_rate = sample_rate

    def calculate_time_array(self, samples: int) -> ndarray:
        """
        Calculates the timestamps that belong to a signal.
        :param samples: the number of samples.
        :return: a 1D-ndarray with the timestamps.
        """
        return linspace(0, samples / self.sample_rate, samples)

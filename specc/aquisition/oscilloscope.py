import numpy as np


class Oscilloscope:
    def snapshot(self, channel: int) -> (int, np.ndarray):
        """
        Takes a snapshot from the specified channel.
        :param channel: a integer specifying the channel.
        :return: sample rate and a 1D-ndarray containing the signal.
        """
        raise NotImplementedError

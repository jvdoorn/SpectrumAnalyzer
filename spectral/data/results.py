from spectral.data.signal import Signal
from spectral.utils import find_nearest_index, relative_phase


class SystemResponse:
    def __init__(self, input_signal: Signal, output_signal: Signal):
        assert len(input_signal) == len(output_signal), "Expected signals to have equal lengths."
        assert input_signal.sample_rate == output_signal.sample_rate, "Expected signals to have equal sample rates."

        self.input_signal = input_signal
        self.output_signal = output_signal

    def relative_intensity(self, frequency: float, df: float):
        return self.output_signal.power(frequency, df) / self.input_signal.power(frequency, df)

    def relative_phase(self, frequency: float):
        index = find_nearest_index(self.input_signal.frequencies, frequency)

        input_phase = self.input_signal.phases[index]
        output_phase = self.output_signal.phases[index]

        return relative_phase(input_phase, output_phase)

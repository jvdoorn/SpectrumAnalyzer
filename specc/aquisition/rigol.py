import numpy as np
import visa

from specc.aquisition.oscilloscope import Oscilloscope


class Rigol1000EChannelCalibration:
    VERTICAL_DIVISIONS = 9
    HORIZONTAL_DIVISIONS = 12

    SMALLEST_PIXEL = 11
    BIGGEST_PIXEL = 236
    TIME_PIXELS = 600

    def __init__(self, rigol: visa.Resource, zero_volt_pixel: float, channel: int):
        self.zero_volt_pixel = zero_volt_pixel

        self.volts_per_division = np.float64(rigol.query(f":CHAN{channel}:SCAL?"))
        self.time_per_division = np.float64(rigol.query(f":TIM:SCAL? CHAN{channel}"))
        self.volt_offset = np.float64(rigol.query(f":CHAN{channel}:OFFS?"))

    @property
    def volts_per_pixel(self) -> float:
        pixels = self.BIGGEST_PIXEL - self.SMALLEST_PIXEL
        volts = self.VERTICAL_DIVISIONS * self.volts_per_division
        return volts / pixels

    @property
    def sample_rate(self) -> float:
        time_per_pixel = (self.time_per_division * self.HORIZONTAL_DIVISIONS) / self.TIME_PIXELS
        return 1 / time_per_pixel


class Rigol1000ESeriesOscilloscope(Oscilloscope):
    def __init__(self):
        super().__init__()

        resource_manager = visa.ResourceManager()
        self._rigol = resource_manager.open_resource(resource_manager.list_resources()[0])

        self._zero_volt_pixel = 124.5

    def _read_channel_calibration(self, channel: int) -> Rigol1000EChannelCalibration:
        return Rigol1000EChannelCalibration(self._rigol, self._zero_volt_pixel, channel)

    def _take_snapshot(self, channel: int):
        self._rigol.write(f":WAVeform:DATA? CHAN{channel}")
        data = list(self._rigol.read_raw())
        return np.asarray(list(data))

    def _sanitize_data(self, data: np.ndarray, configuration: Rigol1000EChannelCalibration) -> np.ndarray:
        flipped_data = np.flip(data[10:])
        sanitized_samples = (self._zero_volt_pixel - flipped_data) * configuration.volts_per_pixel

        return sanitized_samples

    def snapshot(self, channel: int) -> (int, np.ndarray):
        channel_configuration = self._read_channel_calibration(channel)

        data = self._take_snapshot(channel)
        samples = self._sanitize_data(data, channel_configuration)

        return channel_configuration.sample_rate, samples

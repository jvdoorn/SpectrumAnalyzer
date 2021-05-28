from typing import Union

import numpy as np


class Converter:
    def __init__(self, unit='V'):
        self.unit = unit

    def convert(self, data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return data

    def error(self, data: Union[np.ndarray, float]) -> float:
        return 0


DEFAULT_CONVERTER = Converter()

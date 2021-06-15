from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Tuple, Union, overload
from warnings import warn

import numpy as np

if TYPE_CHECKING:
    from specc.data.signal import Signal


@overload
def validate_samples(samples: list) -> np.ndarray:
    return validate_samples(np.asarray(samples))


def validate_samples(samples: np.ndarray) -> np.ndarray:
    assert isinstance(samples, np.ndarray), "Expected samples to be a list or ndarray."
    assert len(samples.shape) == 1, "Expected 1D-ndarray as input signal."
    return samples


def validate_sample_rate(sample_rate: Real) -> int:
    if not isinstance(sample_rate, int):
        warn(f"Converting sample rate to int, was a {type(sample_rate)}.")
        sample_rate = int(sample_rate)
    assert sample_rate > 0, "Expected a positive sample rate."
    return sample_rate


@overload
def validate_compatible_array(signal: Signal, array: list) -> np.ndarray:
    return validate_compatible_array(signal, np.asarray(array))


def validate_compatible_array(signal: Signal, array: Union[list, np.ndarray]) -> np.ndarray:
    array = validate_samples(array)
    assert len(signal) == len(array), "Expected array to have equal length."
    return array


def validate_compatible_signals(*signals: Signal) -> Tuple[Signal]:
    sample_rates = set(map(lambda s: s.sample_rate, signals))
    assert len(sample_rates) <= 1, "Not all sample rates are equal."
    lengths = set(map(lambda s: len(s), signals))
    assert len(lengths) <= 1, "Not all signals are equally long."
    return signals

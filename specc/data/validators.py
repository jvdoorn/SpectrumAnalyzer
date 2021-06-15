from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from specc.data.signal import Signal


def validate_samples(samples: Union[np.ndarray, list]) -> np.ndarray:
    assert isinstance(samples, (np.ndarray, list)), "Expected samples to be a list or ndarray."
    if isinstance(samples, list):
        samples = np.asarray(samples)
    assert len(samples.shape) == 1, "Expected 1D-ndarray as input signal."
    return samples


def validate_sample_rate(sample_rate: float):
    assert sample_rate > 0, "Expected a positive sample rate."
    return sample_rate


def validate_compatible_array(signal: Signal, array: np.ndarray):
    assert len(array.shape) == 1, "Expected 1D array"
    assert len(signal) == len(array), "Expected array to have equal length."
    return array


def validate_compatible_signals(*signals: Signal):
    sample_rates = set(map(lambda s: s.sample_rate, signals))
    assert len(sample_rates) <= 1, "Not all sample rates are equal."
    lengths = set(map(lambda s: len(s), signals))
    assert len(lengths) <= 1, "Not all sample rates are equally long."
    return signals

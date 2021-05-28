from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from specc.data.results import SystemBehaviour
from specc.data.signal import Signal
from specc.utils import is_list_of

PHASE_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PHASE_LABELS = ["$-\\pi$", "$-\\frac{1}{2}\\pi$", "0", "$\\frac{1}{2}\\pi$", "$\\pi$"]

POLAR_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
POLAR_LABELS = ["$0$", "$\\frac{1}{4}\\pi$", "$\\frac{1}{2}\\pi$", "$\\frac{3}{4}\\pi$", "$\\pm\\pi$",
                "$-\\frac{3}{4}\\pi$", "$-\\frac{1}{2}\\pi$", "$-\\frac{1}{4}\\pi$"]


def plot_signal(signal: Signal, title: str = None):
    plt.clf()

    plt.plot(signal.timestamps, signal.csamples)
    if signal.error != 0:
        plt.fill_between(signal.timestamps, signal.csamples - signal.error, signal.csamples + signal.error, alpha=0.5)

    plt.xlabel("Time [s]")
    plt.ylabel(f"Signal [{signal.converter.unit}]")

    if title:
        plt.title(title)
    plt.tight_layout()
    return plt


def plot_behaviour(behaviours: Union[SystemBehaviour, List[SystemBehaviour]], title: str = None,
                   labels: Union[list, None] = None, intensity_markers: Union[list, None] = None,
                   phase_markers: Union[list, None] = None):
    plt.clf()
    if phase_markers is None:
        phase_markers = []
    if intensity_markers is None:
        intensity_markers = []
    if type(behaviours) is not list:
        behaviours = [behaviours]

    fig = plt.figure(figsize=(8, 4))
    if title:
        fig.suptitle(title)

    polar_axis = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection='polar')
    polar_axis.set_xticks(POLAR_TICKS)
    polar_axis.set_xticklabels(POLAR_LABELS)

    intensity_axis = plt.subplot2grid((2, 2), (0, 1))
    intensity_axis.set_ylabel("$20\\log|H(f)|$ [dB]")

    phase_axis = plt.subplot2grid((2, 2), (1, 1))
    phase_axis.set_ylabel("Phase [rad]")
    phase_axis.set_xlabel("Frequency [Hz]")
    phase_axis.set_yticks(PHASE_TICKS)
    phase_axis.set_yticklabels(PHASE_LABELS)
    phase_axis.set_ylim(-np.pi, np.pi)

    for behaviour in behaviours:
        polar_axis.plot(behaviour.phases, behaviour.intensities)
        intensity_axis.semilogx(behaviour.frequencies, behaviour.decibels)
        phase_axis.semilogx(behaviour.frequencies, behaviour.phases)

    if labels:
        intensity_axis.legend(labels)

    for marker in intensity_markers:
        intensity_axis.axhline(marker, linestyle='--', color='r', alpha=0.5)

    for marker in phase_markers:
        phase_axis.axhline(marker, linestyle='--', color='r', alpha=0.5)

    plt.tight_layout()
    return plt


def plot(obj: Union[SystemBehaviour, List[SystemBehaviour], Signal], *args, **kwargs):
    if isinstance(obj, SystemBehaviour) or is_list_of(obj, SystemBehaviour):
        return plot_behaviour(obj, *args, **kwargs)
    elif isinstance(obj, Signal):
        return plot_signal(obj, *args, **kwargs)

    raise NotImplementedError(f"{type(obj)} is not supported in the plot function.")

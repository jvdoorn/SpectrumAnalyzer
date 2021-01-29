"""
This file contains a demonstration of the SimulationAnalyzer
class. Other analyzers have a similar working but methods
might have a different name.

The demonstration is done on a low-pass RC circuit.
"""

import numpy as np

from spectral.analysis.analyzer import SimulationAnalyzer
from spectral.utils import latex_float

RC = 10 ** -3
RC_neat = latex_float(RC)
low_pass = lambda f: 1 / (1 + (1j * RC * 2 * np.pi * f))
high_pass = lambda f: 1 / (1 + 1 / (1j * RC * 2 * np.pi * f))

frequencies = np.logspace(0, 4, 8 * 12)

if __name__ == '__main__':
    analyzer = SimulationAnalyzer()
    plot = analyzer.plot(f"Prediction of low pass filter with $RC={RC_neat}$.", analyzer.predict(frequencies, low_pass))
    plot.show()
    plot = analyzer.plot(f"Prediction of high pass filter with $RC={RC_neat}$.",
                         analyzer.predict(frequencies, high_pass))
    plot.show()

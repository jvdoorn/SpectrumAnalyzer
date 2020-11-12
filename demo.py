"""
This file contains a demonstration of the SimulationAnalyzer
class. Other analyzers have a similar working but methods
might have a different name.

The demonstration is done on a low-pass RC circuit.
"""

import numpy as np

from spectral.analyzer import SimulationAnalyzer

RC_exponent = -3
RC = 10 ** RC_exponent
low_pass = lambda f: 1 / (1 + (1j * RC * 2 * np.pi * f))

frequencies = np.logspace(0, 4, 4 * 12)

if __name__ == '__main__':
    analyzer = SimulationAnalyzer()
    analyzer.plot("Simulation", *analyzer.simulate_transfer_function(frequencies, low_pass))
    analyzer.plot(f"Prediction of low pass filter with $RC=10^{{{RC_exponent}}}$.",
                  *analyzer.predict(frequencies, low_pass))

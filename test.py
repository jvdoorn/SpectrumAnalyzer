import numpy as np

from analyzer import TestAnalyzer

data_path = "test_data/"

RC_exponent = -3
RC = 10 ** RC_exponent
low_pass = lambda f: 1 / (1 + (1j * RC * 2 * np.pi * f))

frequencies = np.logspace(0, 4, 4 * 12)

if __name__ == '__main__':
    analyzer = TestAnalyzer()
    analyzer.plot("Simulation", *analyzer.simulate_transfer_function(frequencies, low_pass))
    analyzer.plot(f"Prediction of low pass filter with $RC=10^{{{RC_exponent}}}$.",
                  *analyzer.predict(frequencies, low_pass))

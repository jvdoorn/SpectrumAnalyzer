# spectral

This project is initially based on software I ([jvdoorn](https://github.com/jvdoorn)) wrote for an Experimental Physics course. It aims to provide everything you need for analyzing signals. It is able to generate, read and write signals. It analyzes signals using a discrete Fourier transform (provided by Numpy). It has built in functions to create eleborate Bode plots.

## Installing spectral
Spectral can be installed using pip: `pip install git+https://github.com/jvdoorn/SpectrumAnalyzer.git`. You can upgrade your installation by running the command again. You can then import any functions or classes with `import spectral.something`. 

## Contributing
You are welcome to contribute to spectral! Please fork the repository and create a PR!

## Overview
The file `demo.py` contains a simple demo to demonstrate the capabilities of spectral. It is not included when installing via pip.

* `spectral/analyzer.py` contains various Analyzers. They can be used to measure or simulate a system and provide methods to plot and analyze results.
* `spectral/fourier.py` contains various methods related to Fourier transforms. They are used internally but they can be used as standalone functions when needed.
* `spectral/utils.py` contains various utility methods. They are used internally but they can be used as standalone functions when needed.
* `spectral/aquisition/mydaq.py` contains an interface for the [Ni MyDAQ](http://ni.com/mydaq). You can read and write signals using it and is used by some Analyzer classes to obtain their data.

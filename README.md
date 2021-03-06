# spectral

This project is initially based on software I ([Julian van Doorn](https://github.com/jvdoorn)) wrote for an Experimental Physics course. It aims to provide everything you need for analyzing signals. It is able to generate, read and write signals. It analyzes signals using a discrete Fourier transform (provided by Numpy). It has built in functions to create eleborate Bode plots.

## Installing spectral
Spectral can be installed using pip: `pip install git+https://github.com/jvdoorn/SpectrumAnalyzer.git`. You can update your installation by running the command again. You can import any functions or classes with `import spectral.something`. 

## Contributing
You are welcome to contribute to spectral! Please fork the repository and create a PR!

## Overview
* `spectral/analysis/analyzer.py` contains various Analyzers. They can be used to measure or predict a system and provide methods to plot and analyze results.
* `spectral/aquisition/` contains various interfaces for data acquisition devices such as the [NI MyDAQ](http://ni.com/mydaq).
* `spectral/data/` contains various classes that are returned by the analyzers. 

import setuptools

setuptools.setup(
    name="spectral",
    version="0.2.9",
    author="Julian van Doorn",
    author_email="jvdoorn@antarc.com",
    url="https://github.com/jvdoorn/SpectrumAnalyzer",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/jvdoorn/SpectrumAnalyzer",
        "Tracker": "https://github.com/jvdoorn/SpectrumAnalyzer/issues",
    },
    install_requires=['numpy', 'matplotlib', 'scipy', 'nidaqmx', 'tqdm'],
)

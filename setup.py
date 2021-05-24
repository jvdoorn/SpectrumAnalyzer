import os

from setuptools import find_packages, setup

setup(
    name="specc",
    version=os.environ.get('PACKAGE_VERSION'),
    author="Julian van Doorn",
    author_email="jvdoorn@antarc.com",
    url="https://github.com/jvdoorn/SpectrumAnalyzer",
    packages=find_packages(exclude=("tests", "tests.*")),
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

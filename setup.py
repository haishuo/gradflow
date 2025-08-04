#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="gradflow",
    version="0.1.0",
    author="Your Name",
    description="Revolutionary Differentiable CFD - World's First Differentiable WENO",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "h5py>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
)

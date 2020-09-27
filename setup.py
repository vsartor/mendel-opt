#!/usr/bin/env python

import setuptools


setuptools.setup(
    name="mendel-opt",
    packages=["mendelopt"],
    version="0.1",
    license="MIT",
    description="Python library written as a case study in implementing pythonic Genetic Algorithms.",
    author="Victhor Sartório",
    author_email="victhor@vsartor.com",
    url="https://github.com/vsartor/mendel-opt",
    keywords=["optimizer", "genetic algorithm"],
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Typing :: Typed",
    ],
)

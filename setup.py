"""Pip installation script."""
from setuptools import find_packages, setup

setup(
    name='cslsearch',
    version="0.1",
    description="Brute-force search for CSL vectors in a given lattice system.",
    author='Adam J. Plowman',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ]
)

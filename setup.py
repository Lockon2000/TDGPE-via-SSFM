import os.path
import setuptools
import codecs


"""
The following section is needed to provide a single-source implementation for package meta data, e.g. version.
"""

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="TDGPEviaSSFM",
    version=get_version("TDGPEviaSSFM/__init__.py"),
    description="A python package to solve the time-dependent Gross–Pitaevskii equation numerically using the Split-Step Fourier method.",
    url="https://github.com/Lockon2000/TDGPE-via-SSFM",
    author="Mohamed Abdelwahab",
    author_email="m.abdelwahab2@gmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=["scipy", "numpy",],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)

"""
TDGPEviaSSFM

A python package to solve the time-dependent Gross–Pitaevskii equation numerically using the Split-Step Fourier method.
"""

__version__ = "0.3.0"


from . import configs

def configure(**options):
    for option, value in options.items():
        if option in [
            'dt',
            'N_t',
            'skippingFactor',
            'partFactor',
            'secsToMsecsConversionFactor',
            'savePath'
        ]:
            configs.__dict__[option] = value

from . import tools
from . import tdgpe
from . import interpolation
from . import plotting
from . import animation

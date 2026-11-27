"""Convolve in-situ hyperspectral reflectance to satellite sensor bands."""

from importlib.metadata import PackageNotFoundError, version

from .Convolution import Convolution, Convolved, convolve
from .core import SENSORS, band_weights

__all__ = ['SENSORS', 'Convolution', 'Convolved', 'band_weights', 'convolve']

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # package is not installed
    __version__ = 'unknown'

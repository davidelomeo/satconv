"""Convolve in-situ hyperspectral reflectance to satellite sensor bands."""

from importlib.metadata import PackageNotFoundError, version

from .Convolution import SENSORS, Convolution

__all__ = ['Convolution', 'SENSORS']

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # package is not installed
    __version__ = 'unknown'

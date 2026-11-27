"""NumPy-only core: sensor tables, band weights and the convolution itself.

A convolution is a matrix product: ``band_weights`` turns a wavelength grid and a sensor into
a weight matrix ``W``, and ``spectra @ W.T`` gives the band values. Nothing here needs pandas
or xarray, so this module (with the CSV tables) can be moved to any platform that has NumPy.
"""

import csv
from functools import cache
from importlib.resources import files
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = ['SENSORS', 'band_weights']

_DATA_DIR = files(__package__) / 'SRFs_and_bandpasses'

# Spectra are convolved in blocks of this many, to limit temporary memory when values are missing
_BLOCK_SIZE = 65_536

SENSORS = {
    'TM_L5': 'Landsat 5 Thematic Mapper',
    'ETM+_L7': 'Landsat 7 Enhanced Thematic Mapper Plus',
    'OLI_L8': 'Landsat 8 Operational Land Imager',
    'OLI_L9': 'Landsat 9 Operational Land Imager 2',
    'MSI_S2A': 'Sentinel-2A MultiSpectral Instrument',
    'MSI_S2B': 'Sentinel-2B MultiSpectral Instrument',
    'OLCI_S3A': 'Sentinel-3A Ocean and Land Colour Instrument',
    'OLCI_S3B': 'Sentinel-3B Ocean and Land Colour Instrument',
    'MERIS': 'Envisat Medium Resolution Imaging Spectrometer',
    'MODIS_AQUA': 'MODIS on Aqua',
    'MODIS_TERRA': 'MODIS on Terra',
    'VIIRS_SNPP': 'VIIRS on Suomi NPP',
    'VIIRS_NOAA20': 'VIIRS on NOAA-20',
    'VIIRS_NOAA21': 'VIIRS on NOAA-21',
    'GOCI': 'Geostationary Ocean Color Imager',
    'GOCI_II': 'Geostationary Ocean Color Imager II',
    'SeaWiFS': 'Sea-viewing Wide Field-of-view Sensor',
    'Superdove': 'PlanetScope SuperDove',
}


class SensorTable(NamedTuple):
    """Spectral description of one sensor, as read-only arrays."""

    centres: np.ndarray  # nominal band centres (nm), shape (n_bands,)
    fwhm: np.ndarray  # full width at half maximum of each band (nm), shape (n_bands,)
    wavelengths: np.ndarray  # wavelengths of the response curves (nm), shape (n_samples,)
    response: np.ndarray  # response curves with negatives set to zero, shape (n_samples, n_bands)


@cache
def sensor_table(sensor: str) -> SensorTable:
    """Read the band table and Spectral Response Functions (SRFs) of a sensor (cached)."""
    if sensor not in SENSORS:
        raise ValueError(f"Invalid sensor name {sensor!r}. Available sensors: {', '.join(SENSORS)}")

    with (_DATA_DIR / f'{sensor}_bandpass.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    centres = np.array([float(row['Nominal Center Wavelength']) for row in rows])
    fwhm = np.array([float(row['Width (FWHM)']) for row in rows])

    with (_DATA_DIR / f'{sensor}_SRF.csv').open(encoding='utf-8-sig') as f:
        header = f.readline().strip().split(',')  # 'wl', then one column per band
        data = np.loadtxt(f, delimiter=',', ndmin=2)
    column_of = {float(name): i for i, name in enumerate(header[1:], start=1)}
    # Negative values are measurement noise in some published tables
    response = np.clip(data[:, [column_of[centre] for centre in centres]], 0, None)

    if np.array_equal(centres, np.round(centres)):
        centres = centres.astype(np.int64)
    table = SensorTable(centres, fwhm, data[:, 0].copy(), response)
    for array in table:
        array.setflags(write=False)  # the cached arrays are shared between callers
    return table


def band_weights(wavelengths: ArrayLike, sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """Weight matrix that convolves spectra on a given wavelength grid to a sensor's bands.

    Each band is the integral of (spectrum x SRF) divided by the integral of the SRF, with
    the spectrum linearly interpolated onto the SRF wavelengths and integrated with the
    trapezoidal rule. Only the part of the SRF covered by the input wavelengths is used.

    Parameters
    ----------
    wavelengths : array-like
        Wavelengths (nm) of the input spectra: 1-D, finite and without duplicates. Any order
        and any spacing, including decimals.
    sensor : str
        One of the keys of ``convolution.SENSORS``.

    Returns
    -------
    weights : np.ndarray
        Shape (n_bands, n_wavelengths), so that ``spectra @ weights.T`` convolves spectra
        whose last axis is wavelength. Each row sums to 1, except for bands without any
        input wavelength inside their FWHM, whose row is all zeros.
    centres : np.ndarray
        Nominal band centre wavelengths (nm), shape (n_bands,).
    """
    return table_weights(wavelengths, sensor_table(sensor))


def table_weights(wavelengths: ArrayLike, table: SensorTable) -> tuple[np.ndarray, np.ndarray]:
    """``band_weights`` for a sensor description given as a ``SensorTable``."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1:
        raise ValueError(f'Wavelengths must be 1-D; got shape {wavelengths.shape}')
    if not np.isfinite(wavelengths).all():
        raise ValueError('Wavelengths must be finite numbers')
    order = np.argsort(wavelengths)
    x = wavelengths[order]
    if (np.diff(x) == 0).any():
        raise ValueError('Wavelengths must not contain duplicates')

    weights = np.zeros((table.centres.size, x.size))
    centres = table.centres.copy()
    if x.size < 2:
        return weights, centres

    # The response curves, limited to the range covered by the input
    keep = (table.wavelengths >= x[0]) & (table.wavelengths <= x[-1])
    s = table.wavelengths[keep]
    if s.size < 2:
        return weights, centres

    # Trapezoidal rule: the share of the integral belonging to each response sample
    step = np.diff(s)
    quadrature = np.concatenate(([step[0]], step[:-1] + step[1:], [step[-1]])) / 2
    area = quadrature[:, np.newaxis] * table.response[keep]
    total = area.sum(axis=0)

    # Linear interpolation: each response sample splits its share between its two neighbours
    left = np.clip(np.searchsorted(x, s, side='right') - 1, 0, x.size - 2)
    fraction = (s - x[left]) / (x[left + 1] - x[left])

    for band, (centre, width) in enumerate(zip(table.centres, table.fwhm, strict=True)):
        has_data = ((x >= centre - width / 2) & (x <= centre + width / 2)).any()
        if has_data and total[band] > 0:
            row = np.bincount(left, area[:, band] * (1 - fraction), minlength=x.size)
            row += np.bincount(left + 1, area[:, band] * fraction, minlength=x.size)
            weights[band, order] = row / total[band]
    return weights, centres


def apply_weights(spectra: ArrayLike, weights: np.ndarray, max_missing: float = 0.01) -> np.ndarray:
    """Convolve spectra of shape (..., n_wavelengths) with a matrix from ``band_weights``.

    NaN and infinite values count as missing. A band is NaN for a spectrum when more than
    ``max_missing`` of its response falls on missing values; otherwise it is computed from the
    values that are there. Bands with an all-zero row are always NaN.
    """
    if np.ma.isMaskedArray(spectra):
        spectra = spectra.astype(float).filled(np.nan)
    spectra = np.asarray(spectra)
    n_spectra = int(np.prod(spectra.shape[:-1]))
    flat = spectra.reshape(n_spectra, spectra.shape[-1])
    out = np.empty((n_spectra, weights.shape[0]))

    for start in range(0, n_spectra, _BLOCK_SIZE):
        block = flat[start:start + _BLOCK_SIZE].astype(float, copy=False)
        result = out[start:start + _BLOCK_SIZE]  # a view: filled in place
        finite = np.isfinite(block)
        if finite.all():
            result[:] = block @ weights.T
            continue
        missing = (~finite).astype(float) @ weights.T  # share of each band's response without data
        total = np.where(finite, block, 0.0) @ weights.T
        result[:] = np.divide(total, 1 - missing, out=np.full_like(total, np.nan),
                              where=missing <= max_missing)

    out[:, ~weights.any(axis=1)] = np.nan
    return out.reshape((*spectra.shape[:-1], weights.shape[0]))

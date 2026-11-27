"""Convolve in-situ hyperspectral reflectance to the bands of a target satellite sensor.

``convolve`` accepts NumPy arrays, xarray DataArrays and pandas DataFrames and returns the
same kind of object. The maths lives in ``convolution.core``; this module only translates
each input type to arrays and back. pandas and xarray are not imported unless used.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from .core import _DATA_DIR, SENSORS, SensorTable, apply_weights, sensor_table, table_weights

__all__ = ['SENSORS', 'Convolution', 'Convolved', 'convolve']


class Convolved(NamedTuple):
    """Result of convolving a NumPy array; unpacks as ``values, wavelengths``."""

    values: np.ndarray  # shape (..., n_bands)
    wavelengths: np.ndarray  # nominal band centres (nm), shape (n_bands,)


def convolve(
    spectra: Any,
    sensor: str,
    wavelengths: ArrayLike | None = None,
    *,
    dim: str | None = None,
    max_missing: float = 0.01,
) -> Any:
    """Convolve reflectance spectra to the bands of a satellite sensor.

    Parameters
    ----------
    spectra : array-like, xarray.DataArray or pandas.DataFrame
        * NumPy array (or array-like): wavelength is the **last** axis, e.g.
          (n_wavelengths,) or (n_observations, n_wavelengths) or (y, x, n_wavelengths).
        * xarray.DataArray: any shape, with one wavelength dimension.
        * pandas.DataFrame: index = wavelength (nm), one column per observation.
    sensor : str
        One of the keys of ``convolution.SENSORS``.
    wavelengths : array-like, optional
        Wavelengths (nm) of the input. Required for NumPy input. For xarray and pandas input
        they are read from the coordinate or index; give them to override.
    dim : str, optional
        Wavelength dimension of a DataArray. Found automatically when it is the only
        dimension whose name looks like a wavelength.
    max_missing : float, optional
        Largest share (0 to 1) of a band's response that may fall on NaN input values. Above
        it the band is NaN for that spectrum, otherwise it uses the values that are there.

    Returns
    -------
    Convolved, xarray.DataArray or pandas.DataFrame
        * NumPy input: ``Convolved(values, wavelengths)``, values of shape (..., n_bands).
        * DataArray input: DataArray with the wavelength dimension replaced by 'wavelength'
          (the band centres, last), lazy if the input is dask-backed.
        * DataFrame input: rows = observations, columns = band centres.

    Bands without any input wavelength inside their FWHM are NaN, and a warning is issued.
    A warning is also issued when missing values leave a band without a result for any
    spectrum (not checked for dask-backed input, which is not computed here).
    """
    return _convolve(spectra, sensor, sensor_table(sensor), wavelengths, dim, max_missing)


def _convolve(spectra, sensor, table, wavelengths=None, dim=None, max_missing=0.01):
    """``convolve`` with the sensor's description given as a ``SensorTable``.

    Warnings point at the caller of the function that calls this one.
    """
    if not 0 <= max_missing < 1:
        raise ValueError(f'max_missing must be at least 0 and below 1; got {max_missing}')

    is_dataarray = _is_instance(spectra, 'xarray', 'DataArray')
    is_dataframe = _is_instance(spectra, 'pandas', 'DataFrame')
    if _is_instance(spectra, 'xarray', 'Dataset'):
        raise TypeError("Pass one variable of the Dataset, e.g. convolve(ds['rrs'], ...)")
    if dim is not None and not is_dataarray:
        raise ValueError('dim only applies to xarray input')

    if is_dataarray:
        dim = wavelength_dim(spectra, dim)
        if wavelengths is None:
            wavelengths = _wavelength_coordinate(spectra, dim)
        n_input = spectra.sizes[dim]
    else:
        if is_dataframe:
            if wavelengths is None:
                wavelengths = spectra.index
            spectra_values = spectra.to_numpy(dtype=float, na_value=np.nan).T
        elif wavelengths is None:
            raise ValueError('Pass wavelengths= (in nm) for the last axis of the array')
        else:
            spectra_values = spectra if np.ma.isMaskedArray(spectra) else np.asarray(spectra)
        n_input = spectra_values.shape[-1] if spectra_values.ndim else 0

    weights, centres = table_weights(wavelengths, table)
    if weights.shape[1] != n_input:
        hint = '' if is_dataarray or is_dataframe else ' (wavelength must be the last axis; try spectra.T)'
        raise ValueError(f'The spectra have {n_input} values along wavelength but '
                         f'{weights.shape[1]} wavelengths were given{hint}')

    covered = weights.any(axis=1)
    if not covered.all():
        warnings.warn(f'{sensor}: no input data for bands {centres[~covered].tolist()} nm; '
                      'returned NaN.', stacklevel=3)

    if is_dataarray:
        result = _convolve_dataarray(spectra, dim, weights, centres, sensor, max_missing)
        values = None if result.chunks is not None else result.values  # leave dask input lazy
    else:
        values = apply_weights(spectra_values, weights, max_missing)
        if is_dataframe:
            index = spectra.columns
            if index.nlevels == 1 and index.name is None:
                index = index.rename('in_situ_obs')
            result = sys.modules['pandas'].DataFrame(values, index=index, columns=centres)
        else:
            result = Convolved(values, centres)

    if values is not None and values.size:
        spectrum_axes = tuple(range(values.ndim - 1))
        lost = covered & np.isnan(values).all(axis=spectrum_axes)
        if lost.any():
            warnings.warn(f'{sensor}: missing values leave bands {centres[lost].tolist()} nm without '
                          'a result for any spectrum (NaN).', stacklevel=3)
    return result


def wavelength_dim(data: Any, dim: str | None = None) -> str:
    """The wavelength dimension of a DataArray: ``dim`` if given, otherwise found by name."""
    if dim is not None:
        if dim not in data.dims:
            raise ValueError(f'No dimension {dim!r}; the dimensions are {list(data.dims)}')
        return dim
    if 'wavelength' in data.coords and data['wavelength'].ndim == 1:
        return data['wavelength'].dims[0]
    matches = [d for d in data.dims if looks_like_wavelength(d)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'Cannot tell which of the dimensions {list(data.dims)} is wavelength; pass dim=')


def looks_like_wavelength(name: object) -> bool:
    """Whether a dimension name refers to wavelength, e.g. 'wavelengths_rrs' or 'wl'."""
    name = str(name).lower()
    return 'wavelength' in name or name in ('wl', 'wvl', 'lambda')


def _wavelength_coordinate(data: Any, dim: str) -> np.ndarray:
    """Wavelength values along ``dim``: a 'wavelength' coordinate if present, else ``dim``'s own."""
    if 'wavelength' in data.coords and data['wavelength'].dims == (dim,):
        return data['wavelength'].values
    if dim in data.coords:
        return data[dim].values
    raise ValueError(f'Dimension {dim!r} has no wavelength coordinate; pass wavelengths=')


def _convolve_dataarray(data, dim, weights, centres, sensor, max_missing):
    xr = sys.modules['xarray']
    out = xr.apply_ufunc(
        apply_weights,
        data,
        kwargs={'weights': weights, 'max_missing': max_missing},
        input_core_dims=[[dim]],
        output_core_dims=[['wavelength']],
        exclude_dims={dim},
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'wavelength': centres.size}, 'allow_rechunk': True},
        keep_attrs=True,  # band averaging keeps the units
    )
    centre_attrs = {'units': 'nm', 'long_name': 'nominal band centre wavelength'}
    out = out.assign_coords(wavelength=('wavelength', centres, centre_attrs))
    out.attrs = {**data.attrs, 'sensor': sensor}
    out.name = data.name
    out.encoding = {}
    return out


def _is_instance(obj: Any, module: str, name: str) -> bool:
    """isinstance check that does not import the module: if it is not loaded, obj cannot be one."""
    loaded = sys.modules.get(module)
    return loaded is not None and isinstance(obj, getattr(loaded, name))


class Convolution:
    """Convolve in-situ reflectance spectra to the bands of a satellite sensor.

    The DataFrame interface of earlier versions, built on the same code as ``convolve``. The
    ``srf`` and ``bandpass`` tables of the instance are the ones used, so they can be edited,
    or supplied by overriding ``get_srf_and_bandpass``.

    Parameters
    ----------
    in_situ_rrs : pd.DataFrame
        Index = wavelength in nm, one column per observation.
    sensor_name : str
        One of the keys of ``convolution.SENSORS``.
    savefile : str, optional
        Folder where ``<sensor_name>_conv.csv`` is written by ``do_convolutions()``.
    """

    def __init__(self, in_situ_rrs, sensor_name, savefile=None):
        sensor_table(sensor_name)  # fails early on an unknown sensor

        self.sensor_name = sensor_name
        self.savefile = savefile
        self.in_situ_rrs = in_situ_rrs.copy()
        self.in_situ_rrs.columns = self.in_situ_rrs.columns.astype(str)
        self.srf, self.bandpass = self.get_srf_and_bandpass()
        self.convolved_bands = None

    def get_srf_and_bandpass(self):
        """Return the sensor's spectral response function and bandpass table as DataFrames."""
        import pandas as pd

        with (_DATA_DIR / f'{self.sensor_name}_SRF.csv').open(encoding='utf-8-sig') as f:
            srf = pd.read_csv(f, index_col='wl')
        with (_DATA_DIR / f'{self.sensor_name}_bandpass.csv').open(encoding='utf-8-sig') as f:
            bandpass = pd.read_csv(f)
        return srf, bandpass

    def do_convolutions(self):
        """Convolve every observation to every sensor band.

        Returns
        -------
        pd.DataFrame
            Rows = observations, columns = band nominal centre wavelengths (nm).
            Bands with no input data inside their FWHM are NaN.
        """
        self.convolved_bands = _convolve(self.in_situ_rrs, self.sensor_name, self._table())
        self.convolved_bands.index.name = 'in_situ_obs'

        if self.savefile:
            self.convolved_bands.to_csv(Path(self.savefile) / f'{self.sensor_name}_conv.csv')
        return self.convolved_bands

    def _table(self) -> SensorTable:
        """The ``srf`` and ``bandpass`` DataFrames as the arrays the convolution uses."""
        centres = self.bandpass['Nominal Center Wavelength'].to_numpy(dtype=float)
        fwhm = self.bandpass['Width (FWHM)'].to_numpy(dtype=float)
        srf = self.srf.sort_index()
        column_of = {float(name): name for name in srf.columns}
        response = srf[[column_of[centre] for centre in centres]].to_numpy(dtype=float)
        if np.array_equal(centres, np.round(centres)):
            centres = centres.astype(np.int64)
        # Negative values are measurement noise in some published tables
        return SensorTable(centres, fwhm, srf.index.to_numpy(dtype=float), np.clip(response, 0, None))

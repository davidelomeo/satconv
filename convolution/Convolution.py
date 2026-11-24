"""Convolve in-situ hyperspectral reflectance to the bands of a target satellite sensor."""

import os
import warnings
from importlib.resources import files

import numpy as np
import pandas as pd

__all__ = ['Convolution', 'SENSORS']

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

_DATA_DIR = files('convolution') / 'SRFs_and_bandpasses'

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


class Convolution:
    """Convolve in-situ reflectance spectra to the bands of a satellite sensor.

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
        if sensor_name not in SENSORS:
            raise ValueError(
                f"Invalid sensor name {sensor_name!r}. Available sensors: {', '.join(SENSORS)}")

        self.sensor_name = sensor_name
        self.savefile = savefile
        self.in_situ_rrs = in_situ_rrs.copy()
        self.in_situ_rrs.index = self.in_situ_rrs.index.astype(int)
        self.in_situ_rrs.columns = self.in_situ_rrs.columns.astype(str)
        self.srf, self.bandpass = self.get_srf_and_bandpass()
        self.convolved_bands = None

    def get_srf_and_bandpass(self):
        """Return the sensor's spectral response function and bandpass table as DataFrames."""
        srf = pd.read_csv(_DATA_DIR / f'{self.sensor_name}_SRF.csv', index_col='wl')
        bandpass = pd.read_csv(_DATA_DIR / f'{self.sensor_name}_bandpass.csv')
        return srf, bandpass

    def do_convolutions(self):
        """Convolve every observation to every sensor band.

        Returns
        -------
        pd.DataFrame
            Rows = observations, columns = band nominal centre wavelengths (nm).
            Bands with no input data inside their FWHM are NaN.
        """
        wl = self.in_situ_rrs.index.values
        rrs = self.in_situ_rrs.values.astype(float)
        result, missing = {}, []

        for center, fwhm in zip(self.bandpass['Nominal Center Wavelength'],
                                self.bandpass['Width (FWHM)']):
            # A band needs input data inside its FWHM; it is then integrated over the whole SRF
            has_data = ((wl >= center - fwhm / 2) & (wl <= center + fwhm / 2)).any()
            result[center] = (_convolve_band(wl, rrs, self.srf[str(int(center))]) if has_data
                              else np.full(rrs.shape[1], np.nan))
            if np.isnan(result[center]).all():
                missing.append(center)

        if missing:
            warnings.warn(f'{self.sensor_name}: no input data for bands {missing} nm; returned NaN.')

        self.convolved_bands = pd.DataFrame(result, index=self.in_situ_rrs.columns)
        self.convolved_bands.index.name = 'in_situ_obs'

        if self.savefile:
            self.convolved_bands.to_csv(os.path.join(self.savefile, f'{self.sensor_name}_conv.csv'))
        return self.convolved_bands


def _convolve_band(wl, rrs, srf):
    """SRF-weighted mean of each column of ``rrs``, using the whole SRF curve.

    The input spectra are linearly interpolated onto the SRF wavelength grid, limited to
    the range covered by the input data, then integrated with the trapezoidal rule.
    Negative SRF values (measurement noise in some tables) are treated as zero.
    """
    srf_wl = srf.index.values
    keep = (srf_wl >= wl.min()) & (srf_wl <= wl.max())
    srf_wl, response = srf_wl[keep], np.clip(srf.values[keep], 0, None)

    denominator = _trapezoid(response, srf_wl)
    if denominator == 0:
        return np.full(rrs.shape[1], np.nan)

    interp = np.column_stack([np.interp(srf_wl, wl, col) for col in rrs.T])
    return _trapezoid(interp * response[:, None], srf_wl, axis=0) / denominator

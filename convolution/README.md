# `convolution` package

Installation and usage are in the [main README](../README.md). This page describes how the package works.

## Contents

- `Convolution.py`
  - `SENSORS`: dict of available sensor names and their descriptions.
  - `Convolution(in_situ_rrs, sensor_name, savefile=None)`: loads the sensor's SRF and bandpass table.
    - `get_srf_and_bandpass()`: returns `(srf, bandpass)` as DataFrames.
    - `do_convolutions()`: returns the convolved bands (rows = observations, columns = nominal band centres in nm). If `savefile` was given, also writes `<savefile>/<sensor_name>_conv.csv`.
- `SRFs_and_bandpasses/`: one `<sensor>_SRF.csv` and one `<sensor>_bandpass.csv` per sensor.

## How a band is computed

For each band:

1. Input spectra are linearly interpolated onto the SRF wavelengths (only where input data exist).
2. Band value = ∫ rrs · SRF dλ / ∫ SRF dλ over the **whole** SRF curve (trapezoidal rule).
   Negative SRF values, which appear as noise in some published tables, are treated as zero.

A band with no input data inside its FWHM (nominal centre ± FWHM / 2) is returned as `NaN` and a warning is issued.

Input wavelengths are cast to whole nanometres before convolving.

## Adding a sensor

1. Add `<NAME>_SRF.csv`: first column `wl` (nm), then one column per band, named with the band's nominal centre wavelength.
2. Add `<NAME>_bandpass.csv` with the columns `Band Number, Nominal Center Wavelength, Center Wavelength, Width (FWHM), ...` and no units row. The `Nominal Center Wavelength` values must match the SRF column names.
3. Add `'<NAME>': '<description>'` to `SENSORS` in `Convolution.py` and to the table in the main README.

`tests/test_convolution.py` checks every sensor in `SENSORS` automatically.

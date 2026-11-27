# `convolution` package

Installation and usage are in the [main README](../README.md). This page describes how the package is put together.

## Contents

| File | What it holds |
|---|---|
| `core.py` | The maths, using NumPy only: `SENSORS`, `sensor_table` (reads and caches a sensor's CSV tables), `band_weights` (wavelengths and sensor → weight matrix; `table_weights` does the same for any `SensorTable`) and `apply_weights` (spectra and weight matrix → bands, including missing values). |
| `Convolution.py` | `convolve`, which turns NumPy, xarray and pandas input into arrays and back, and the `Convolution` class kept from earlier versions. |
| `cli.py` | The `satconv` command: reads files, calls `convolve`, writes files. Variables copied to the output are read a second time without decoding, so that they are written as stored. Output is written to a scratch folder first and then moved into place. `__main__.py` makes `python -m convolution` work. |
| `SRFs_and_bandpasses/` | One `<sensor>_SRF.csv` and one `<sensor>_bandpass.csv` per sensor. |

pandas and xarray are only imported when they are used, so `core.py` and the CSV tables can be moved to any platform that has NumPy.

## How a band is computed

For each band:

1. The input spectrum is linearly interpolated onto the SRF wavelengths (only where input data exist).
2. Band value = ∫ rrs · SRF dλ / ∫ SRF dλ over the whole SRF curve that the input covers (trapezoidal rule).
   Negative SRF values, which appear as noise in some published tables, are treated as zero.

Both steps are linear in the spectrum, so `band_weights` combines them into one matrix `W` of shape (n_bands, n_wavelengths), computed once per call. The convolution of any number of spectra is then `spectra @ W.T`.

A band with no input data inside its FWHM (nominal centre ± FWHM / 2) has a row of zeros in `W` and is returned as `NaN`.

Where a spectrum has missing values, `apply_weights` computes the share of each band's weight that falls on them. Above `max_missing` the band is `NaN`; otherwise the remaining weights are rescaled to sum to 1.

## Adding a sensor

1. Add `<NAME>_SRF.csv`: first column `wl` (nm), then one column per band, named with the band's nominal centre wavelength.
2. Add `<NAME>_bandpass.csv` with the columns `Band Number, Nominal Center Wavelength, Center Wavelength, Width (FWHM), ...` and no units row. The `Nominal Center Wavelength` values must match the SRF column names.
3. Add `'<NAME>': '<description>'` to `SENSORS` in `core.py` and to the table in the main README.

`tests/test_convolution.py` checks every sensor in `SENSORS` automatically.

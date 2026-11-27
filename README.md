# satconv: Satellite Sensor Convolutions

Convolve in-situ hyperspectral reflectance measurements to the bands of a target
satellite sensor, using each sensor's Spectral Response Function (SRF).

You can use it from the command line on netCDF, CSV and TSV files, or from Python on
NumPy arrays, xarray DataArrays and pandas DataFrames.

**Contents:** [Installation](#installation) · [Command line](#command-line) · [Python](#python) · [How bands are computed](#how-bands-are-computed) · [Upgrading from 0.3](#upgrading-from-03) · [Tests](#running-the-tests) · [Sensors](#available-sensors)

## Installation

Requires Python 3.11 or newer. Install directly from GitHub:

```bash
pip install "git+https://github.com/davidelomeo/satconv.git"
```

With [uv](https://docs.astral.sh/uv/), into the current environment:

```bash
uv pip install "git+https://github.com/davidelomeo/satconv.git"
```

Or as a dependency of a uv project:

```bash
uv add "git+https://github.com/davidelomeo/satconv.git"
```

With conda or mamba, activate your environment and use the `pip install` command above.

Or, from a local clone (use `-e` for an editable install while developing):

```bash
git clone https://github.com/davidelomeo/satconv.git
```

```bash
cd satconv && pip install -e .
```

The package is imported as `convolution`. Its dependencies (`numpy`, `pandas`, `xarray`,
`netCDF4`) are installed automatically. For lazy, chunked processing of large files, add dask:

```bash
pip install "convolution[dask] @ git+https://github.com/davidelomeo/satconv.git"
```

## Command line

```bash
satconv -i spectra.nc -o olci.nc -s OLCI_S3A -v rrs
```

The file extension sets the input and output types:

| Extension | Layout |
|---|---|
| `.nc`, `.nc4`, `.cdf` | netCDF with one or more variables that have a wavelength dimension |
| `.csv`, `.tsv` | Input: wavelength (nm) in the first column, one column per observation. Output: one row per observation, one column per band |

| Option | Description | Default |
|---|---|---|
| `-i`, `--input` | Input file | required |
| `-o`, `--output` | Output file (folders are created if needed) | required |
| `-s`, `--sensor` | Target sensor, case-insensitive ([list](#available-sensors)) | required |
| `-v`, `--variable` | netCDF variable(s) to convolve | the only one with wavelengths |
| `-d`, `--dim` | Name of the wavelength dimension | recognised by its name |
| `--range MIN MAX` | Only write bands with centres from MIN to MAX nm (inclusive) | all bands |
| `--max-missing` | Largest share of a band's response allowed on NaN input | 0.01 |
| `--list-sensors` | List the sensors and their bands | |
| `--version` | Print the version and exit | |

More examples:

```bash
# netCDF to a table, keeping only the bands from 412 to 779 nm
satconv -i spectra.nc -o olci.csv -s OLCI_S3A -v rrs --range 412 779

# two variables at once
satconv -i spectra.nc -o msi.nc -s MSI_S2A -v rrs irradiance

# CSV to CSV
satconv -i spectra.csv -o modis.csv -s MODIS_AQUA
```

netCDF output keeps the input's global attributes and every variable that does not depend on
wavelength (time, position, quality flags, …), copied with their stored types, attributes and
values. It adds the `satellite_sensor` and `history` attributes. Two limits: only the root
group of a netCDF-4 file is read and written, and variables of compound or variable-length
numeric types are left out, with a warning.

Text output holds a single variable.

The command exits with status 1 on an error. It never overwrites its input file, and an existing
output file is only replaced once the new one has been written completely.

`python -m convolution` runs the same command.

## Python

`convolve` returns the same kind of object that you give it.

### xarray

```python
import xarray as xr
from convolution import convolve

ds = xr.open_dataset("spectra.nc")

olci = convolve(
    ds["rrs"],
    sensor="OLCI_S3A",
)

olci = olci.sel(wavelength=slice(412, 779))
```

The result has a `wavelength` dimension (the band centres, last) in place of the input's
wavelength dimension, and keeps the other coordinates and the attributes.

The input's wavelength dimension is found through a coordinate called `wavelength` (the
command also accepts a data variable of that name), or through its name (anything containing `wavelength`, or `wl`, `wvl`, `lambda`). Otherwise, name
it with `dim=`. Dask-backed input stays lazy:

```python
ds = xr.open_dataset("big_file.nc", chunks={"time": 10_000})

olci = convolve(
    ds["rrs"],
    sensor="OLCI_S3A",
)  # nothing is computed until you write or load the result
```

### NumPy

Wavelength is the **last** axis; any number of axes can come before it.

```python
from convolution import convolve

values, centres = convolve(
    spectra,                  # (..., n_wavelengths)
    sensor="OLCI_S3A",
    wavelengths=wavelengths,  # nm, required for NumPy input
)
# values: (..., n_bands), centres: (n_bands,) nominal band centres in nm
```

### pandas

The index is wavelength in nm and each column is one observation:

```python
import pandas as pd
from convolution import convolve

rrs = pd.read_csv("my_spectra.csv", index_col=0)

bands = convolve(
    rrs,
    sensor="OLCI_S3A",
)  # rows = observations, columns = band centres
```

The `Convolution` class of earlier versions still works, and returns the same table:

```python
from convolution import Convolution

conv = Convolution(
    in_situ_rrs=rrs,
    sensor_name="OLCI_S3A",
    savefile=None,  # or a folder, to also write <folder>/OLCI_S3A_conv.csv
)

bands = conv.do_convolutions()
srf, bandpass = conv.get_srf_and_bandpass()
```

### The weight matrix

A convolution is a matrix product. `band_weights` returns the matrix, so that you can apply it
with any array library, or export it to another platform:

```python
from convolution import band_weights

weights, centres = band_weights(
    wavelengths,
    sensor="OLCI_S3A",
)  # weights: (n_bands, n_wavelengths)

values = spectra @ weights.T
```

The plain product does no checks on the data: bands that your wavelengths do not cover have a
row of zeros (`convolve` returns them as NaN), and any NaN in a spectrum spreads to every band.

## How bands are computed

Each band is ∫ rrs · SRF dλ / ∫ SRF dλ, with the spectrum linearly interpolated onto the SRF
wavelengths and integrated with the trapezoidal rule. The whole SRF curve is used, or the part
of it that your wavelengths cover.

- **Wavelengths** can be in any order and at any spacing, including decimals. Duplicates are an
  error.
- **Coverage:** a band with no input wavelength within its nominal centre ± FWHM / 2 is `NaN`,
  and a warning is issued.
- **Missing values:** NaN and infinite values are missing. A band is `NaN` for a spectrum if
  more than `max_missing` (default 1 %) of its response falls on missing values. Otherwise it
  is computed from the values that are there. A warning is issued if this leaves a band without
  a result for any spectrum (not checked for dask-backed input).

## Upgrading from 0.3

- Python 3.11+ and NumPy 2 are required. `xarray` and `netCDF4` are now dependencies.
- Decimal wavelengths are used as given. Version 0.3 cut them to whole nanometres, so results
  for non-integer wavelength grids differ slightly.
- A missing value now only affects the bands that rely on it (see `max_missing`).
- Unsorted wavelengths are accepted, and duplicated wavelengths raise an error.
- `Convolution(...).do_convolutions()` is otherwise unchanged. It still uses the instance's
  `srf` and `bandpass` tables, so edits to them, or an overridden `get_srf_and_bandpass`, apply.

## Running the tests

```bash
pip install -e ".[test]"
```

```bash
pytest
```

GitHub Actions runs the tests on Python 3.11 and 3.14, and lints with `ruff check`.

## Available sensors

| `sensor`       | Sensor                                   |
|----------------|------------------------------------------|
| `TM_L5`        | Landsat 5 Thematic Mapper                |
| `ETM+_L7`      | Landsat 7 Enhanced Thematic Mapper Plus  |
| `OLI_L8`       | Landsat 8 Operational Land Imager        |
| `OLI_L9`       | Landsat 9 Operational Land Imager 2      |
| `MSI_S2A`      | Sentinel-2A MultiSpectral Instrument     |
| `MSI_S2B`      | Sentinel-2B MultiSpectral Instrument     |
| `OLCI_S3A`     | Sentinel-3A Ocean and Land Colour Instrument |
| `OLCI_S3B`     | Sentinel-3B Ocean and Land Colour Instrument |
| `MERIS`        | Envisat MERIS                            |
| `MODIS_AQUA`   | MODIS on Aqua                            |
| `MODIS_TERRA`  | MODIS on Terra                           |
| `VIIRS_SNPP`   | VIIRS on Suomi NPP                       |
| `VIIRS_NOAA20` | VIIRS on NOAA-20                         |
| `VIIRS_NOAA21` | VIIRS on NOAA-21                         |
| `GOCI`         | Geostationary Ocean Color Imager         |
| `GOCI_II`      | Geostationary Ocean Color Imager II      |
| `SeaWiFS`      | Sea-viewing Wide Field-of-view Sensor    |
| `Superdove`    | PlanetScope SuperDove                    |

## Data sources

- SRFs and bandpasses: NASA Ocean Color RSR tables — https://oceancolor.gsfc.nasa.gov/resources/docs/rsr_tables/
- GOCI-II SRF: KIOST/NOSC — https://www.nosc.go.kr/eng/boardContents/actionBoardContentsCons0024.do

## License

MIT

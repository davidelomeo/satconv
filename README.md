# Satellite Sensors Convolutions

Convolve in-situ hyperspectral reflectance measurements to the bands of a target
satellite sensor, using each sensor's Spectral Response Function (SRF).

## Installation

Requires Python 3.9+. Install directly from GitHub:

```bash
pip install git+https://github.com/davidelomeo/satellite_sensors_convolutions.git
```

Or, from a local clone (use `-e` for an editable install while developing):

```bash
git clone https://github.com/davidelomeo/satellite_sensors_convolutions.git
cd satellite_sensors_convolutions
pip install -e .
```

The package is imported as `convolution`. Dependencies (`numpy`, `pandas`) are installed automatically.

## Usage

The input is a `pandas.DataFrame` whose **index is wavelength in nm** and where
**each column is one observation/spectrum**:

```python
import pandas as pd
from convolution import Convolution

# index = wavelength (nm), columns = observations
rrs = pd.read_csv("my_spectra.csv", index_col=0)

conv = Convolution(rrs, "OLCI_S3A")
bands = conv.do_convolutions()   # rows = observations, columns = band nominal wavelengths
print(bands)
```

Retrieve the SRF and bandpass table used for a sensor:

```python
srf, bandpass = conv.get_srf_and_bandpass()
```

To also write the result to `<folder>/<sensor_name>_conv.csv`, pass `savefile="<folder>"`.

Bands that fall outside the wavelength range of your input data are returned as `NaN`
and a warning is issued. The list of sensor names is also available as `convolution.SENSORS`.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

## Available sensors

| `sensor_name`  | Sensor                                   |
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

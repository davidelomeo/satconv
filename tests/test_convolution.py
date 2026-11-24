import numpy as np
import pandas as pd
import pytest

from convolution import SENSORS, Convolution
from convolution.Convolution import _trapezoid

WL = np.arange(350, 2501)


@pytest.fixture
def spectra():
    return pd.DataFrame({'flat': np.full(WL.size, 0.01), 'ramp': WL / 1e5}, index=WL)


@pytest.mark.parametrize('sensor', SENSORS)
def test_srf_and_bandpass_match(sensor, spectra):
    srf, bandpass = Convolution(spectra, sensor).get_srf_and_bandpass()
    assert [str(b) for b in bandpass['Nominal Center Wavelength']] == list(srf.columns)


@pytest.mark.parametrize('sensor', SENSORS)
def test_flat_spectrum_is_preserved(sensor, spectra):
    out = Convolution(spectra, sensor).do_convolutions()
    assert list(out.index) == ['flat', 'ramp']
    np.testing.assert_allclose(out.loc['flat'], 0.01)


@pytest.mark.parametrize('sensor', SENSORS)
def test_ramp_returns_srf_centroid(sensor, spectra):
    # For rrs = wavelength, the band value is the SRF-weighted mean wavelength of the whole curve
    conv = Convolution(spectra, sensor)
    out = conv.do_convolutions().loc['ramp'] * 1e5
    for band in out.index:
        srf = conv.srf[str(band)].clip(lower=0)
        srf = srf[(srf.index >= WL.min()) & (srf.index <= WL.max())]
        centroid = _trapezoid(srf.index * srf, srf.index) / _trapezoid(srf, srf.index)
        assert out[band] == pytest.approx(centroid, abs=0.05)


@pytest.mark.parametrize('sensor', SENSORS)
def test_band_centroid_near_published_centre(sensor, spectra):
    # Sanity check on the data files: most of each SRF's weight sits near the band centre
    conv = Convolution(spectra, sensor)
    out = conv.do_convolutions().loc['ramp'] * 1e5
    bp = conv.bandpass.set_index('Nominal Center Wavelength')
    assert (abs(out - bp['Center Wavelength']) <= bp['Width (FWHM)']).all()


def test_missing_bands_are_nan_and_warn(spectra):
    with pytest.warns(UserWarning, match='no input data'):
        out = Convolution(spectra.loc[400:700], 'SeaWiFS').do_convolutions()
    assert out[865].isna().all()
    assert out[443].notna().all()


def test_input_is_not_modified(spectra):
    spectra.index = spectra.index.astype(float)
    spectra.columns = [1, 2]
    before = spectra.copy()
    Convolution(spectra, 'MERIS').do_convolutions()
    pd.testing.assert_frame_equal(spectra, before)


def test_savefile_contains_results(spectra, tmp_path):
    out = Convolution(spectra, 'GOCI', savefile=str(tmp_path)).do_convolutions()
    saved = pd.read_csv(tmp_path / 'GOCI_conv.csv', index_col=0)
    np.testing.assert_allclose(saved.values, out.values)


def test_invalid_sensor():
    with pytest.raises(ValueError, match='Invalid sensor name'):
        Convolution(pd.DataFrame({'a': [0.0]}, index=[500]), 'NOPE')

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convolution import SENSORS, Convolution, band_weights, convolve

WL = np.arange(350, 2501)
SHORT_WL = np.arange(350, 1001)
GRIDS = {'integer': SHORT_WL, 'decimal': SHORT_WL + 0.7, 'coarse': np.arange(352, 1000, 7.3)}


@pytest.fixture
def spectra():
    return pd.DataFrame({'flat': np.full(WL.size, 0.01), 'ramp': WL / 1e5}, index=WL)


def random_spectra(wl, *shape):
    return np.random.default_rng(0).uniform(0.001, 0.05, (*shape, wl.size))


def plain_band(wl, spectrum, srf):
    """Plain implementation of one band of one spectrum, to check the matrix version."""
    srf = srf[(srf.index >= wl.min()) & (srf.index <= wl.max())].clip(lower=0)
    values = np.interp(srf.index, wl, spectrum)
    return np.trapezoid(values * srf, srf.index) / np.trapezoid(srf, srf.index)


@pytest.mark.parametrize('sensor', SENSORS)
def test_srf_and_bandpass_match(sensor, spectra):
    srf, bandpass = Convolution(spectra, sensor).get_srf_and_bandpass()
    assert [str(b) for b in bandpass['Nominal Center Wavelength']] == list(srf.columns)


@pytest.mark.parametrize('sensor', SENSORS)
@pytest.mark.parametrize('grid', GRIDS)
def test_matches_plain_implementation(sensor, grid):
    wl = GRIDS[grid]
    spectrum = random_spectra(wl)
    values, centres = convolve(spectrum, sensor, wavelengths=wl)
    srf, bandpass = Convolution(pd.DataFrame({'a': spectrum}, index=wl), sensor).get_srf_and_bandpass()
    for value, centre, width in zip(values, centres, bandpass['Width (FWHM)'], strict=True):
        if ((wl >= centre - width / 2) & (wl <= centre + width / 2)).any():
            assert value == pytest.approx(plain_band(wl, spectrum, srf[str(centre)]), rel=1e-12)
        else:
            assert np.isnan(value)


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
        centroid = np.trapezoid(srf.index * srf, srf.index) / np.trapezoid(srf, srf.index)
        assert out[band] == pytest.approx(centroid, abs=0.05)


@pytest.mark.parametrize('sensor', SENSORS)
def test_band_centroid_near_published_centre(sensor, spectra):
    # Sanity check on the data files: most of each SRF's weight sits near the band centre
    conv = Convolution(spectra, sensor)
    out = conv.do_convolutions().loc['ramp'] * 1e5
    bp = conv.bandpass.set_index('Nominal Center Wavelength')
    assert (abs(out - bp['Center Wavelength']) <= bp['Width (FWHM)']).all()


@pytest.mark.parametrize('sensor', SENSORS)
def test_weights_sum_to_one_or_zero(sensor):
    weights, centres = band_weights(SHORT_WL, sensor)
    assert weights.shape == (centres.size, SHORT_WL.size)
    assert (weights >= 0).all()
    totals = weights.sum(axis=1)
    np.testing.assert_allclose(totals[totals > 0], 1)


def test_decimal_wavelengths_are_used_as_given():
    # A ramp is interpolated exactly, so both grids must give the same band values
    wl = np.arange(350, 1101.0)
    expected, centres = convolve(wl, 'OLCI_S3A', wavelengths=wl)
    shifted = wl - 0.3
    got, _ = convolve(shifted, 'OLCI_S3A', wavelengths=shifted)
    inside = centres < 940  # bands whose whole response curve lies inside both grids
    assert inside.sum() > 15 and np.isfinite(expected[inside]).all()
    np.testing.assert_allclose(got[inside], expected[inside], rtol=1e-12)


def test_wavelength_order_does_not_matter():
    order = np.random.default_rng(1).permutation(SHORT_WL.size)
    spectrum = random_spectra(SHORT_WL)
    expected, _ = convolve(spectrum, 'MSI_S2A', wavelengths=SHORT_WL)
    got, _ = convolve(spectrum[order], 'MSI_S2A', wavelengths=SHORT_WL[order])
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize('wl, message', [
    ([400, 400, 401], 'duplicates'),
    ([400, np.nan, 402], 'finite'),
    ([[400, 401, 402]], '1-D'),
])
def test_bad_wavelengths(wl, message):
    with pytest.raises(ValueError, match=message):
        convolve(np.ones(3), 'OLCI_S3A', wavelengths=wl)


def test_numpy_input_of_any_shape():
    cube = random_spectra(SHORT_WL, 3, 4)
    result = convolve(cube, 'OLCI_S3A', wavelengths=SHORT_WL)
    values, centres = result
    assert values.shape == (3, 4, centres.size) and result.values is values
    one, _ = convolve(cube[1, 2], 'OLCI_S3A', wavelengths=SHORT_WL)
    assert one.shape == centres.shape
    np.testing.assert_allclose(values[1, 2], one)


def test_numpy_input_errors():
    with pytest.raises(ValueError, match='Pass wavelengths'):
        convolve(np.ones(10), 'OLCI_S3A')
    with pytest.raises(ValueError, match='last axis'):
        convolve(np.ones((SHORT_WL.size, 2)), 'OLCI_S3A', wavelengths=SHORT_WL)
    with pytest.raises(ValueError, match='max_missing'):
        convolve(np.ones(SHORT_WL.size), 'OLCI_S3A', wavelengths=SHORT_WL, max_missing=1)
    with pytest.raises(ValueError, match='only applies to xarray'):
        convolve(np.ones(SHORT_WL.size), 'OLCI_S3A', wavelengths=SHORT_WL, dim='wl')


def test_missing_values_only_affect_bands_that_use_them():
    weights, centres = band_weights(SHORT_WL, 'OLCI_S3A')
    band = list(centres).index(443)
    peak = weights[band].argmax()
    tail = np.flatnonzero((weights[band] > 0) & (weights[band] < 0.005))[0]
    spectra = random_spectra(SHORT_WL, 3)
    full, _ = convolve(spectra, 'OLCI_S3A', wavelengths=SHORT_WL)

    holed = spectra.copy()
    holed[1, peak] = np.nan
    holed[2, tail] = np.inf
    out, _ = convolve(holed, 'OLCI_S3A', wavelengths=SHORT_WL)

    np.testing.assert_array_equal(out[0], full[0])  # complete spectra are untouched
    assert np.isnan(out[1, band])
    assert out[2, band] == pytest.approx(full[2, band], rel=0.02)
    used = (weights[:, [peak, tail]] > 0).any(axis=1)
    np.testing.assert_allclose(out[:, ~used], full[:, ~used])
    strict, _ = convolve(holed, 'OLCI_S3A', wavelengths=SHORT_WL, max_missing=0)
    assert np.isnan(strict[2, band])


def test_partly_missing_band_uses_the_available_values():
    holed = np.full(SHORT_WL.size, 0.02)
    holed[SHORT_WL < 443] = np.nan
    out, centres = convolve(holed, 'OLCI_S3A', wavelengths=SHORT_WL, max_missing=0.9)
    assert out[list(centres).index(443)] == pytest.approx(0.02)


def test_masked_array_values_are_missing():
    data = np.ma.masked_array(np.full(SHORT_WL.size, 0.02), mask=SHORT_WL == 443)
    data.data[SHORT_WL == 443] = 1e30  # a fill value that must not leak into the result
    out, centres = convolve(data, 'OLCI_S3A', wavelengths=SHORT_WL)
    assert np.isnan(out[list(centres).index(443)])
    assert out[list(centres).index(560)] == pytest.approx(0.02)


def dataarray(dim='wavelengths_rrs'):
    return xr.DataArray(
        random_spectra(SHORT_WL, 5).T,
        dims=(dim, 'time'),
        coords={dim: SHORT_WL.astype(float), 'time': np.arange(5), 'depth': ('time', np.arange(5.0))},
        name='rrs', attrs={'units': 'sr-1'})


def test_dataarray_input():
    data = dataarray()
    out = convolve(data, 'OLCI_S3A')
    values, centres = convolve(data.values.T, 'OLCI_S3A', wavelengths=SHORT_WL)
    assert out.dims == ('time', 'wavelength') and out.name == 'rrs'
    assert out.attrs == {'units': 'sr-1', 'sensor': 'OLCI_S3A'}
    np.testing.assert_array_equal(out['wavelength'], centres)
    np.testing.assert_array_equal(out['depth'], data['depth'])
    np.testing.assert_allclose(out.values, values)


@pytest.mark.parametrize('dim', ['wavelength', 'wl', 'Wavelength_nm'])
def test_dataarray_wavelength_dimension_names(dim):
    xr.testing.assert_allclose(convolve(dataarray(dim), 'MERIS'), convolve(dataarray(), 'MERIS'))


def test_dataarray_explicit_and_coordinate_wavelengths():
    named = dataarray('x')
    with pytest.raises(ValueError, match='pass dim'):
        convolve(named, 'MERIS')
    expected = convolve(dataarray(), 'MERIS')
    xr.testing.assert_allclose(convolve(named, 'MERIS', dim='x'), expected)
    # Dimension 'band' (1, 2, ...) with the wavelengths in a separate coordinate
    banded = dataarray('band').assign_coords(band=np.arange(SHORT_WL.size),
                                             wavelength=('band', SHORT_WL))
    xr.testing.assert_allclose(convolve(banded, 'MERIS'), expected)
    xr.testing.assert_allclose(convolve(banded, 'MERIS', dim='band'), expected)
    with pytest.raises(ValueError, match='no wavelength coordinate'):
        convolve(named.drop_vars('x'), 'MERIS', dim='x')


def test_dataarray_with_dask_is_lazy():
    pytest.importorskip('dask')
    data = dataarray()
    lazy = convolve(data.chunk({'time': 2, 'wavelengths_rrs': 100}), 'OLCI_S3A')
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), convolve(data, 'OLCI_S3A'))


def test_dataset_input_is_rejected():
    with pytest.raises(TypeError, match='one variable'):
        convolve(dataarray().to_dataset(), 'MERIS')


def test_dataframe_input(spectra):
    out = convolve(spectra, 'MERIS')
    pd.testing.assert_frame_equal(out, Convolution(spectra, 'MERIS').do_convolutions())
    assert out.index.name == 'in_situ_obs'


def test_numpy_use_does_not_import_pandas_or_xarray():
    code = ('import sys, numpy, convolution; '
            'convolution.convolve(numpy.ones(600), "OLCI_S3A", wavelengths=numpy.arange(350, 950)); '
            'assert not {"pandas", "xarray"} & set(sys.modules)')
    subprocess.run([sys.executable, '-c', code], check=True)


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
    with pytest.raises(ValueError, match='Invalid sensor name'):
        convolve(np.zeros(2), 'NOPE', wavelengths=[500, 501])


def test_class_uses_the_tables_it_holds(spectra):
    conv = Convolution(spectra, 'OLCI_S3A')
    conv.bandpass = conv.bandpass.iloc[3:6]
    conv.srf = conv.srf.copy()
    conv.srf['490'] = conv.srf['510']  # a user-supplied response curve
    out = conv.do_convolutions()
    assert list(out.columns) == [490, 510, 560]
    np.testing.assert_allclose(out[490], out[510])
    expected = plain_band(WL, spectra['ramp'].values, conv.srf['510'])
    assert out.loc['ramp', 490] == pytest.approx(expected, rel=1e-12)


def test_class_uses_overridden_tables(spectra):
    class FirstBands(Convolution):
        def get_srf_and_bandpass(self):
            srf, bandpass = super().get_srf_and_bandpass()
            return srf, bandpass.head(2)

    assert list(FirstBands(spectra, 'MERIS').do_convolutions().columns) == [412, 442]


def test_dataframe_with_multiindex_columns(spectra):
    spectra.columns = pd.MultiIndex.from_tuples([('A', 1), ('B', 2)], names=['station', 'cast'])
    out = convolve(spectra, 'MERIS')
    pd.testing.assert_index_equal(out.index, spectra.columns)


@pytest.mark.parametrize('kind', ['numpy', 'dataframe', 'dataarray'])
def test_bands_lost_to_gaps_are_reported(kind):
    values = random_spectra(SHORT_WL, 3)
    values[:, (SHORT_WL > 540) & (SHORT_WL < 580)] = np.nan
    data = {'numpy': values,
            'dataframe': pd.DataFrame(values.T, index=SHORT_WL),
            'dataarray': xr.DataArray(values, dims=('time', 'wavelength'),
                                      coords={'wavelength': SHORT_WL})}[kind]
    wavelengths = SHORT_WL if kind == 'numpy' else None
    with pytest.warns(UserWarning, match=r'missing values leave bands \[560\]'):
        convolve(data, 'OLCI_S3A', wavelengths=wavelengths)


def test_no_gap_warning_when_some_spectrum_has_a_result_or_data_are_lazy():
    wl = np.arange(350, 1101)
    values = random_spectra(wl, 3)
    values[0, (wl > 540) & (wl < 580)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        convolve(values, 'OLCI_S3A', wavelengths=wl)
        convolve(np.empty((0, wl.size)), 'OLCI_S3A', wavelengths=wl)
        pytest.importorskip('dask')
        values[:] = np.nan
        data = xr.DataArray(values, dims=('time', 'wavelength'), coords={'wavelength': wl})
        assert convolve(data.chunk({'time': 1}), 'OLCI_S3A').chunks is not None

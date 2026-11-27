import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from convolution import Convolution, convolve
from convolution.cli import main

WL = np.arange(350.0, 900.0)


@pytest.fixture
def scene(tmp_path):
    """A netCDF file laid out like a field radiometer product, with two spectral variables."""
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {'rrs': (('time', 'wavelengths_rrs'), rng.uniform(0.001, 0.05, (4, WL.size))),
         'irradiance': (('time', 'wavelengths_spectra'), rng.uniform(0.1, 1, (4, WL.size))),
         'rrs_QC': ('time', np.array([1, 2, 3, 4], dtype=np.int16)),
         'station': ('time', ['a', 'b', 'c', 'd']),
         'latitude': ('time', np.linspace(54, 55, 4))},
        coords={'time': np.arange(4.0), 'wavelengths_rrs': WL, 'wavelengths_spectra': WL},
        attrs={'title': 'test', 'history': 'created'})
    path = tmp_path / 'scene.nc'
    ds.to_netcdf(path)
    return path


def run(*args):
    return main([str(a) for a in args])


def test_netcdf_to_netcdf(scene, tmp_path):
    out = tmp_path / 'new' / 'olci.nc'
    assert run('-i', scene, '-o', out, '-s', 'olci_s3a', '-v', 'rrs') == 0
    with xr.open_dataset(out) as result, xr.open_dataset(scene) as source:
        assert list(result.data_vars) == ['rrs', 'rrs_QC', 'station', 'latitude']
        assert result['rrs'].dims == ('time', 'wavelength')
        expected = convolve(source['rrs'], 'OLCI_S3A')
        np.testing.assert_allclose(result['rrs'], expected)
        np.testing.assert_array_equal(result['wavelength'], expected['wavelength'])
        assert result['rrs_QC'].dtype == np.int16
        assert result.attrs['title'] == 'test' and result.attrs['satellite_sensor'] == 'OLCI_S3A'
        assert result.attrs['history'].startswith('created\n')
        assert 'convolved rrs to OLCI_S3A' in result.attrs['history']


def test_several_variables_and_range(scene, tmp_path, capsys):
    out = tmp_path / 'both.nc'
    assert run('-i', scene, '-o', out, '-s', 'OLCI_S3A', '-v', 'rrs', 'irradiance',
               '--range', 412, 779) == 0
    with xr.open_dataset(out) as result:
        assert result['rrs'].dims == result['irradiance'].dims == ('time', 'wavelength')
        assert result['wavelength'].values[[0, -1]].tolist() == [412, 779]
    assert 'warning' not in capsys.readouterr().err  # uncovered bands lie outside the range


def test_uncovered_bands_are_reported(scene, tmp_path, capsys):
    assert run('-i', scene, '-o', tmp_path / 'a.nc', '-s', 'OLCI_S3A', '-v', 'rrs') == 0
    assert 'rrs: no result for any spectrum in bands 939, 1013 nm' in capsys.readouterr().err


def test_netcdf_to_csv(scene, tmp_path):
    out = tmp_path / 'olci.csv'
    assert run('-i', scene, '-o', out, '-s', 'OLCI_S3A', '-v', 'rrs') == 0
    table = pd.read_csv(out, index_col=0)
    with xr.open_dataset(scene) as source:
        expected = convolve(source['rrs'], 'OLCI_S3A')
    assert table.index.name == 'time'
    np.testing.assert_array_equal(table.columns.astype(int), expected['wavelength'])
    np.testing.assert_allclose(table.values, expected.values)


@pytest.mark.parametrize('suffix, separator', [('csv', ','), ('tsv', '\t')])
def test_text_to_text_matches_python(tmp_path, suffix, separator):
    spectra = pd.DataFrame({'a': WL / 1e5, '002': np.full(WL.size, 0.01)},
                           index=pd.Index(WL, name='wavelength'))
    source, out = tmp_path / f'in.{suffix}', tmp_path / f'out.{suffix}'
    spectra.to_csv(source, sep=separator)
    assert run('-i', source, '-o', out, '-s', 'MSI_S2A') == 0
    table = pd.read_csv(out, sep=separator, index_col=0)
    expected = Convolution(spectra, 'MSI_S2A').do_convolutions()
    assert table.index.name == 'in_situ_obs' and list(table.index) == ['a', '002']
    np.testing.assert_allclose(table.values, expected.values)


def test_text_to_netcdf(tmp_path):
    source, out = tmp_path / 'in.csv', tmp_path / 'out.nc'
    pd.DataFrame({'a': WL / 1e5}, index=pd.Index(WL, name='wl')).to_csv(source)
    assert run('-i', source, '-o', out, '-s', 'MERIS') == 0
    with xr.open_dataset(out) as result:
        assert result['rrs'].dims == ('in_situ_obs', 'wavelength')
        assert result['in_situ_obs'].values.tolist() == ['a']


def test_only_spectral_variable_is_found(scene, tmp_path):
    single = tmp_path / 'single.nc'
    with xr.open_dataset(scene) as source:
        source[['rrs', 'rrs_QC']].to_netcdf(single)
    assert run('-i', single, '-o', tmp_path / 'out.nc', '-s', 'MERIS') == 0


@pytest.mark.parametrize('output', ['one.csv', 'one.nc'])
def test_single_spectrum(tmp_path, capsys, output):
    source = tmp_path / 'one_in.nc'
    xr.Dataset({'rrs': ('wavelength', WL / 1e5)}, coords={'wavelength': WL}).to_netcdf(source)
    assert run('-i', source, '-o', tmp_path / output, '-s', 'OLCI_S3A') == 0
    assert 'bands 939, 1013 nm' in capsys.readouterr().err
    if output.endswith('.csv'):
        table = pd.read_csv(tmp_path / output, index_col=0)
        assert list(table.index) == ['rrs']
        values = table.to_numpy()[0]
    else:
        with xr.open_dataset(tmp_path / output) as result:
            values = result['rrs'].values
    np.testing.assert_allclose(values, convolve(WL / 1e5, 'OLCI_S3A', wavelengths=WL).values)


@pytest.mark.parametrize('extra, output, message', [
    ([], 'o.nc', 'choose with --variable: rrs, irradiance'),
    (['-v', 'nope'], 'o.nc', 'No variable nope'),
    (['-v', 'rrs_QC'], 'o.nc', 'rrs_QC: Cannot tell'),
    (['-v', 'rrs'], 'o.xlsx', 'Unknown output type'),
    (['-v', 'rrs', 'irradiance'], 'o.csv', 'holds one variable'),
    (['-v', 'rrs', '--range', '2000', '2100'], 'o.nc', 'No MERIS bands'),
    (['-v', 'rrs', '--max-missing', '2'], 'o.nc', 'max_missing'),
    (['-v', 'rrs'], 'scene.nc', 'must not be the input'),
])
def test_errors(scene, tmp_path, capsys, extra, output, message):
    before = scene.read_bytes()
    assert run('-i', scene, '-o', tmp_path / output, '-s', 'MERIS', *extra) == 1
    assert message in capsys.readouterr().err
    assert scene.read_bytes() == before


def test_missing_input_file(tmp_path, capsys):
    assert run('-i', tmp_path / 'no.nc', '-o', tmp_path / 'o.nc', '-s', 'MERIS') == 1
    assert 'satconv: error' in capsys.readouterr().err


@pytest.mark.parametrize('args', [['-i', 'a.nc'], ['-i', 'a.nc', '-o', 'b.nc', '-s', 'NOPE']])
def test_usage_errors(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_list_sensors(capsys):
    assert main(['--list-sensors']) == 0
    printed = capsys.readouterr().out
    assert 'OLCI_S3A' in printed and '412, 443' in printed


def test_module_can_be_run():
    done = subprocess.run([sys.executable, '-m', 'convolution', '--version'],
                          capture_output=True, text=True, check=True)
    assert done.stdout.startswith('satconv ')


def write_variable(nc, name, kind, dims, values, **attrs):
    fill = attrs.pop('_FillValue', None)
    variable = nc.createVariable(name, kind, dims, fill_value=fill)
    variable.set_auto_maskandscale(False)
    variable.set_auto_chartostring(False)
    variable.setncatts(attrs)
    variable[...] = values


def stored_contents(path):
    """Every variable of a netCDF file as it is stored: type, dimensions, attributes, values."""
    contents = {}
    with netCDF4.Dataset(path) as nc:
        for name, variable in nc.variables.items():
            variable.set_auto_maskandscale(False)
            variable.set_auto_chartostring(False)
            data = np.asarray(variable[...])
            values = data.tolist() if data.dtype.kind in 'OSU' else data.tobytes()
            attrs = {key: repr(variable.getncattr(key)) for key in variable.ncattrs()}
            contents[name] = (str(variable.dtype), variable.dimensions, attrs, values)
    return contents


@pytest.fixture
def awkward(tmp_path):
    """A netCDF file with variables stored in ways that decoding and re-encoding can change."""
    path = tmp_path / 'awkward.nc'
    with netCDF4.Dataset(path, 'w') as nc:
        nc.createDimension('time', None)
        nc.createDimension('wavelength', WL.size)
        nc.createDimension('length', 3)
        write_variable(nc, 'time', 'f8', ('time',), np.arange(4.0),
                       units='seconds since 2020-01-01', calendar='standard')
        write_variable(nc, 'wavelength', 'f8', ('wavelength',), WL)
        write_variable(nc, 'rrs', 'f4', ('time', 'wavelength'), np.full((4, WL.size), 0.01, 'f4'),
                       coordinates='lat gone', units='sr-1')
        write_variable(nc, 'lat', 'f4', ('time',), np.arange(4, dtype='f4'))
        write_variable(nc, 'sst', 'i2', ('time',), np.array([40, -1, -32768, -2], 'i2'),
                       _Unsigned='true', scale_factor=np.float32(0.5), _FillValue=np.int16(-2))
        write_variable(nc, 'flags', 'i4', ('time',), np.array([0, -1, -2147483648, 7], 'i4'),
                       _Unsigned='true')
        write_variable(nc, 'packed', 'i2', ('time',), np.array([1, 2, -999, 4], 'i2'),
                       scale_factor=0.001, add_offset=20.0, _FillValue=np.int16(-999))
        write_variable(nc, 'two_fills', 'f4', ('time',), np.array([1, -9, 3, -8], 'f4'),
                       _FillValue=np.float32(-9), missing_value=np.float32(-8))
        write_variable(nc, 'no_fill', 'f4', ('time',), np.array([1, 2, 3, np.nan], 'f4'))
        write_variable(nc, 'boolean', 'i1', ('time',), np.array([1, 0, 1, 1], 'i1'),
                       **{'dtype': 'bool'})
        write_variable(nc, 'text', str, ('time',), np.array(['a', 'bb', '', 'é'], object))
        letters = netCDF4.stringtochar(np.array([b'x', b'yy', b'zzz', b''], 'S3'))
        write_variable(nc, 'chars', 'S1', ('time', 'length'), letters, _Encoding='utf-8')
        write_variable(nc, 'letter', 'S1', ('time',), np.array([b'a', b'b', b'', b'd'], 'S1'))
        write_variable(nc, 'number', 'i4', (), 7, comment='scalar')
        flag = nc.createEnumType('u1', 'flag_t', {'bad': 0, 'good': 1})
        write_variable(nc, 'quality', flag, ('time',), np.array([0, 1, 1, 0], 'u1'))
    return path


def test_kept_variables_are_copied_as_stored(awkward, tmp_path):
    out = tmp_path / 'out.nc'
    assert run('-i', awkward, '-o', out, '-s', 'MERIS', '-v', 'rrs') == 0
    before, after = stored_contents(awkward), stored_contents(out)
    for name in set(before) - {'rrs', 'wavelength'}:
        assert after[name] == before[name], name
    assert set(after) == set(before)
    assert after['rrs'][2]['coordinates'] == repr('lat')
    with netCDF4.Dataset(out) as nc:
        assert nc.dimensions['time'].isunlimited()
        assert nc['sst'].filters()['zlib']
    with xr.open_dataset(out) as result:
        assert 'lat' in result['rrs'].coords
        assert result['sst'].values[:3].tolist() == [20.0, 32767.5, 16384.0]


def test_types_that_cannot_be_copied_are_reported(tmp_path, capsys):
    source, out = tmp_path / 'in.nc', tmp_path / 'out.nc'
    with netCDF4.Dataset(source, 'w') as nc:
        nc.createDimension('time', 2)
        nc.createDimension('wavelength', WL.size)
        write_variable(nc, 'wavelength', 'f8', ('wavelength',), WL)
        write_variable(nc, 'rrs', 'f4', ('time', 'wavelength'), np.full((2, WL.size), 0.01, 'f4'))
        write_variable(nc, 'kept', 'i4', ('time',), [1, 2])
        pair = nc.createCompoundType(np.dtype([('a', 'f4'), ('b', 'i4')]), 'pair_t')
        nc.createVariable('pair', pair, ('time',))[:] = np.array([(1, 2), (3, 4)], pair.dtype)
        ragged = nc.createVariable('ragged', nc.createVLType('i4', 'ragged_t'), ('time',))
        for i in range(2):
            ragged[i] = np.arange(i + 1, dtype='i4')
    assert run('-i', source, '-o', out, '-s', 'MERIS') == 0
    errors = capsys.readouterr().err
    assert 'pair not copied' in errors and 'ragged not copied' in errors
    assert set(stored_contents(out)) == {'rrs', 'wavelength', 'kept'}


@pytest.mark.parametrize('output', ['out.nc', 'out.csv'])
def test_failed_write_leaves_the_old_output(scene, tmp_path, capsys, monkeypatch, output):
    def fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b'partial')
        raise RuntimeError('NetCDF: HDF error')

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', fail)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail)
    target = tmp_path / output
    target.write_bytes(b'earlier result')
    files = sorted(tmp_path.iterdir())
    assert run('-i', scene, '-o', target, '-s', 'MERIS', '-v', 'rrs') == 1
    assert 'HDF error' in capsys.readouterr().err
    assert target.read_bytes() == b'earlier result' and sorted(tmp_path.iterdir()) == files


def test_output_through_a_link_updates_the_linked_file(scene, tmp_path):
    real, link = tmp_path / 'real.nc', tmp_path / 'link.nc'
    real.write_bytes(b'earlier result')
    link.symlink_to(real)
    assert run('-i', scene, '-o', link, '-s', 'MERIS', '-v', 'rrs') == 0
    assert link.is_symlink()
    with xr.open_dataset(real) as result:
        assert result.attrs['satellite_sensor'] == 'MERIS'


@pytest.mark.parametrize('suffix', ['csv', 'nc'])
def test_home_folder_paths(tmp_path, capsys, monkeypatch, suffix):
    home, elsewhere = tmp_path / 'home', tmp_path / 'elsewhere'
    home.mkdir()
    elsewhere.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(elsewhere)
    source = home / f'in.{suffix}'
    if suffix == 'csv':
        pd.DataFrame({'a': WL / 1e5}, index=pd.Index(WL, name='wl')).to_csv(source)
    else:
        xr.Dataset({'rrs': ('wavelength', WL / 1e5)}, coords={'wavelength': WL}).to_netcdf(source)
    before = source.read_bytes()
    # A shell leaves '~' in --option=~/file for the program to expand
    assert main([f'--input=~/in.{suffix}', f'--output=~/in.{suffix}', '-s', 'MERIS']) == 1
    assert 'must not be the input' in capsys.readouterr().err
    assert source.read_bytes() == before
    assert main([f'--input=~/in.{suffix}', f'--output=~/out.{suffix}', '-s', 'MERIS']) == 0
    assert (home / f'out.{suffix}').exists() and not list(elsewhere.iterdir())


def test_wavelengths_in_a_data_variable(tmp_path):
    source, out = tmp_path / 'in.nc', tmp_path / 'out.nc'
    xr.Dataset({'rrs': (('time', 'band'), np.tile(WL / 1e5, (2, 1))),
                'wavelength': ('band', WL), 'quality': ('time', [1, 2])}).to_netcdf(source)
    assert run('-i', source, '-o', out, '-s', 'MERIS') == 0
    with xr.open_dataset(out) as result:
        assert result['rrs'].dims == ('time', 'wavelength') and list(result.data_vars) == ['rrs', 'quality']
        np.testing.assert_allclose(result['rrs'][0], convolve(WL / 1e5, 'MERIS', wavelengths=WL).values)

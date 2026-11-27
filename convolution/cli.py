"""Command line interface: ``satconv -i spectra.nc -o bands.nc -s OLCI_S3A -v rrs``."""

import argparse
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .Convolution import convolve, looks_like_wavelength, wavelength_dim
from .core import SENSORS, sensor_table

_NETCDF_SUFFIXES = ('.nc', '.nc4', '.cdf')
_TEXT_SEPARATORS = {'.csv': ',', '.tsv': '\t'}

_EPILOG = """\
input and output types are chosen by file extension:
  netCDF (.nc, .nc4, .cdf)  one or more variables with a wavelength dimension
  text (.csv, .tsv)         wavelength (nm) in the first column, one column per observation;
                            text output has one row per observation, one column per band

examples:
  satconv -i spectra.nc -o olci.nc -s OLCI_S3A -v rrs
  satconv -i spectra.nc -o olci.csv -s OLCI_S3A -v rrs --range 412 779
  satconv -i spectra.csv -o msi.csv -s MSI_S2A
  satconv --list-sensors
"""


def _sensor_name(text: str) -> str:
    for name in SENSORS:
        if name.lower() == text.lower():
            return name
    raise argparse.ArgumentTypeError(f'unknown sensor {text!r}; see --list-sensors')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='satconv',
        description='Convolve hyperspectral reflectance to the bands of a satellite sensor.',
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('-i', '--input', help='input netCDF, CSV or TSV file')
    parser.add_argument('-o', '--output', type=Path, help='output netCDF, CSV or TSV file')
    parser.add_argument('-s', '--sensor', type=_sensor_name, help='target sensor, e.g. OLCI_S3A')
    parser.add_argument('-v', '--variable', nargs='+',
                        help='netCDF variable(s) to convolve; optional if only one has wavelengths')
    parser.add_argument('-d', '--dim', help='name of the wavelength dimension, if it cannot be guessed')
    parser.add_argument('--range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        help='only write bands with centres from MIN to MAX nm (inclusive)')
    parser.add_argument('--max-missing', type=float, default=0.01,
                        help='largest share of a band response allowed on NaN input (default: 0.01)')
    parser.add_argument('--list-sensors', action='store_true', help='list the sensors and exit')
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``satconv``; returns the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)

    if args.list_sensors:
        for name, description in SENSORS.items():
            centres = ', '.join(f'{centre:g}' for centre in sensor_table(name).centres)
            print(f'{name:14}{description}\n{"":14}bands (nm): {centres}')
        return 0

    absent = [flag for flag in ('input', 'output', 'sensor') if getattr(args, flag) is None]
    if absent:
        parser.error('the following arguments are required: ' + ', '.join(f'--{flag}' for flag in absent))

    # The shell does not expand '~' in --input=~/file
    args.input = os.path.expanduser(args.input)
    args.output = args.output.expanduser()

    try:
        _run(args)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f'satconv: error: {error}', file=sys.stderr)
        return 1
    return 0


def _run(args: argparse.Namespace) -> None:
    suffix = args.output.suffix.lower()
    if suffix not in (*_NETCDF_SUFFIXES, *_TEXT_SEPARATORS):
        raise ValueError(f'Unknown output type {suffix!r}; use .nc, .csv or .tsv')

    with _open(args.input) as opened:
        input_is_file = os.path.exists(args.input)  # it can also be a URL
        if input_is_file and args.output.exists() and args.output.samefile(args.input):
            raise ValueError('The output file must not be the input file')
        source = opened
        if 'wavelength' in source.data_vars and source['wavelength'].ndim == 1:
            source = source.set_coords('wavelength')
        names = _variables(source, args.variable, args.dim)
        converted = {name: _convolve_variable(source[name], args) for name in names}

        if suffix in _TEXT_SEPARATORS:
            with _replacing(args.output) as scratch:
                _write_text(converted, scratch, _TEXT_SEPARATORS[suffix])
        else:
            with _open(args.input, decode_cf=False) as stored, _replacing(args.output) as scratch:
                _write_netcdf(scratch, source, stored, converted, args)

    n_bands = next(iter(converted.values())).sizes['wavelength']
    print(f'satconv: wrote {args.output} ({", ".join(names)} at {n_bands} {args.sensor} bands)',
          file=sys.stderr)


@contextmanager
def _replacing(path: Path) -> Iterator[Path]:
    """A scratch path that takes the place of ``path`` only if the block finishes without error.

    A write that fails therefore leaves an existing file as it was, and no partial file.
    """
    target = path.resolve()  # writing to a link updates the file it points to
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.satconv-', dir=target.parent) as folder:
        scratch = Path(folder) / target.name
        yield scratch
        scratch.replace(target)


def _open(path: str, **options):
    """The input as an xarray Dataset. Text files become one variable, 'rrs'.

    ``options`` are passed to ``xarray.open_dataset`` for netCDF input.
    """
    import xarray as xr

    suffix = Path(path).suffix.lower()
    separator = _TEXT_SEPARATORS.get(suffix)
    if separator is None:
        return xr.open_dataset(path, **options)

    import pandas as pd

    table = pd.read_csv(path, sep=separator, index_col=0)
    try:
        wavelengths = table.index.to_numpy(dtype=float)
        values = table.to_numpy(dtype=float).T
    except ValueError:
        raise ValueError(f'{path}: expected wavelength (nm) in the first column and '
                         'one column of numbers per observation') from None
    return xr.Dataset(
        {'rrs': (('in_situ_obs', 'wavelength'), values)},
        coords={'in_situ_obs': np.asarray(table.columns, dtype=str), 'wavelength': wavelengths})


def _variables(source, requested: list[str] | None, dim: str | None) -> list[str]:
    """Names of the variables to convolve."""
    if requested:
        unknown = [name for name in requested if name not in source.data_vars]
        if unknown:
            raise ValueError(f'No variable {", ".join(unknown)} in the input; '
                             f'it has: {", ".join(map(str, source.data_vars))}')
        return list(dict.fromkeys(requested))

    found = []
    for name, data in source.data_vars.items():
        try:
            wavelength_dim(data, dim)
        except ValueError:
            continue
        found.append(name)
    if not found:
        raise ValueError('No variable with a wavelength dimension found; name the dimension with --dim')
    if len(found) > 1:
        raise ValueError(f'Several variables have wavelengths; choose with --variable: {", ".join(found)}')
    return found


def _convolve_variable(data, args: argparse.Namespace):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # reported below, for the requested bands only
            out = convolve(data, args.sensor, dim=args.dim, max_missing=args.max_missing)
    except ValueError as error:
        raise ValueError(f'{data.name}: {error}') from None

    if args.range:
        low, high = args.range
        selected = (out['wavelength'] >= low) & (out['wavelength'] <= high)
        if not selected.any():
            raise ValueError(f'No {args.sensor} bands between {low:g} and {high:g} nm')
        out = out.isel(wavelength=selected.values)
    out = out.load()

    all_nan = out.isnull().all([d for d in out.dims if d != 'wavelength'])
    if all_nan.any():
        bands = ', '.join(f'{band:g}' for band in out['wavelength'].values[all_nan.values])
        print(f'satconv: warning: {data.name}: no result for any spectrum in bands {bands} nm (NaN)',
              file=sys.stderr)
    return out


def _write_text(converted: dict, path: Path, separator: str) -> None:
    if len(converted) != 1:
        raise ValueError('Text output holds one variable; choose one with --variable, or write netCDF')
    [(name, data)] = converted.items()
    if data.ndim == 1:
        data = data.expand_dims(in_situ_obs=[name])
    if data.ndim != 2:
        raise ValueError(f'Text output needs (observation, wavelength) data; {name} has dimensions '
                         f'{list(data.dims)}. Write netCDF instead.')
    data.to_pandas().to_csv(path, sep=separator)


def _write_netcdf(path: Path, source, stored, converted: dict, args: argparse.Namespace) -> None:
    """Write the convolved variables, plus the input variables that do not depend on wavelength.

    The latter come from ``stored``, the input opened without decoding, so that their values
    and attributes are written exactly as they are in the input.
    """
    import xarray as xr

    used = {wavelength_dim(source[name], args.dim) for name in converted}
    kept = stored.drop_dims([d for d in source.dims if d in used or looks_like_wavelength(d)])

    variables = {}
    for name, data in converted.items():
        attrs = dict(data.attrs)
        if 'coordinates' in stored[name].attrs:
            attrs['coordinates'] = stored[name].attrs['coordinates']
        variables[name] = xr.Variable(data.dims, data.data, attrs)
    for name, variable in kept.variables.items():
        if name in variables or name == 'wavelength':
            continue
        if _can_copy(variable):
            variables[name] = _as_stored(variable)
        else:
            print(f'satconv: warning: {name} not copied: compound and variable-length types '
                  'are not supported', file=sys.stderr)
    variables['wavelength'] = next(iter(converted.values()))['wavelength'].variable

    now = datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    entry = f'{now} satconv {__version__}: convolved {", ".join(converted)} to {args.sensor} bands'
    history = stored.attrs.get('history')
    out = xr.Dataset(variables, attrs={**stored.attrs,
                                       'satellite_sensor': args.sensor,
                                       'history': f'{history}\n{entry}' if history else entry})
    for attrs in (out.attrs, *(variable.attrs for variable in out.variables.values())):
        _drop_absent_coordinates(attrs, out.variables)

    encoding = {name: {**data.encoding, 'zlib': True, 'complevel': 5}
                for name, data in out.data_vars.items() if data.dtype.kind in 'fiubS'}
    unlimited = [d for d in stored.encoding.get('unlimited_dims', ()) if d in out.dims]
    out.to_netcdf(path, encoding=encoding, unlimited_dims=unlimited)


def _can_copy(variable) -> bool:
    """Whether a variable's type can be written again (not compound or variable-length)."""
    if variable.dtype.names:
        return False
    # A variable-length type only shows once a value is read
    first = variable[tuple(slice(0, 1) for _ in variable.dims)].values
    return not (first.dtype == object and first.size and isinstance(first.flat[0], np.ndarray))


def _as_stored(variable):
    """An undecoded input variable, set up so that writing it reproduces what is in the file."""
    import xarray as xr

    dims, data = variable.dims, variable.data
    encoding = {key: value for key, value in variable.encoding.items() if key == 'dtype'}
    if variable.dtype == 'S1' and dims and variable.shape[-1]:
        # Join the characters into strings here: otherwise writing them adds a dimension
        characters = np.ascontiguousarray(variable.values)
        data = characters.view(f'S{characters.shape[-1]}').reshape(characters.shape[:-1])
        encoding, dims = {'char_dim_name': dims[-1]}, dims[:-1]
    if '_FillValue' not in variable.attrs:
        encoding['_FillValue'] = None  # otherwise xarray adds one to floating point variables
    return xr.Variable(dims, data, variable.attrs, encoding)


def _drop_absent_coordinates(attrs: dict, variables) -> None:
    """Remove names of variables that are not written from a CF 'coordinates' attribute."""
    if 'coordinates' not in attrs:
        return
    present = [name for name in attrs['coordinates'].split() if name in variables]
    if present:
        attrs['coordinates'] = ' '.join(present)
    else:
        del attrs['coordinates']

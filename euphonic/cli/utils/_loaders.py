from contextlib import suppress
import json
import os
from pathlib import Path
import re

from euphonic import (
    ForceConstants,
    QpointFrequencies,
    QpointPhononModes,
)
from euphonic.util import (
    format_error,
)


def _load_euphonic_json(filename: str | os.PathLike,
                        frequencies_only: bool = False,
) -> QpointPhononModes | QpointFrequencies | ForceConstants:
    with open(filename) as f:
        data = json.load(f)

    match data:
        case {'force_constants': fc}:
            return ForceConstants.from_json_file(filename)
        case {'frequencies': freq, 'eigenvectors': eig} if not frequencies_only:
            return QpointPhononModes.from_json_file(filename)
        case {'frequencies': freq}:
            return QpointFrequencies.from_json_file(filename)
        case _:
             msg = format_error(
                 f'Could not identify Euphonic data in JSON file ({filename}).',
                 fix='Ensure JSON file contains "force_constants" or "frequencies".',
             )
             raise ValueError(msg)


def _load_phonopy_file(filename: str | os.PathLike,
                       frequencies_only: bool = False,
) -> QpointPhononModes | QpointFrequencies | ForceConstants:
    path = Path(filename)
    loaded_data = None
    if not frequencies_only:
        with suppress(KeyError, RuntimeError):
            # KeyError will be raised if it is actually a force
            # constants file, RuntimeError will be raised if
            # it only contains q-point frequencies (no eigenvectors)

            loaded_data = QpointPhononModes.from_phonopy(
                path=path.parent, phonon_name=path.name)

    # Try to read QpointFrequencies if loading QpointPhononModes has
    # failed, or has been specifically requested with frequencies_only
    if frequencies_only or loaded_data is None:
        with suppress(KeyError):
            loaded_data = QpointFrequencies.from_phonopy(
                path=path.parent, phonon_name=path.name)

    if loaded_data is None:
        phonopy_kwargs: dict[str, str | os.PathLike] = {}
        phonopy_kwargs['path'] = path.parent
        if (path.parent / 'BORN').is_file():
            phonopy_kwargs['born_name'] = 'BORN'
        # Set summary_name and fc_name depending on input file
        if path.suffix == '.hdf5':
            if (path.parent / 'phonopy.yaml').is_file():
                phonopy_kwargs['summary_name'] = 'phonopy.yaml'
                phonopy_kwargs['fc_name'] = path.name
            else:
                msg = format_error(
                    'Missing phonopy.yaml.',
                    reason = (
                        'Phonopy force_constants.hdf5 file '
                        'must be accompanied by information '
                        'about atomic masses, supercell, etc.'
                    ),
                    fix='Ensure phonopy.yaml provided.',
                )
                raise ValueError(msg)
        elif path.suffix in ('.yaml', '.yml'):
            phonopy_kwargs['summary_name'] = path.name
            # Assume this is a (renamed?) phonopy.yaml file
            if (janus_fc := _janus_fc_filename(path)).is_file():
                phonopy_kwargs['fc_name'] = janus_fc.name
            elif path.with_name('force_constants.hdf5').is_file():
                phonopy_kwargs['fc_name'] = 'force_constants.hdf5'
            else:
                phonopy_kwargs['fc_name'] = 'FORCE_CONSTANTS'
        loaded_data = ForceConstants.from_phonopy(**phonopy_kwargs)

    return loaded_data


def _janus_fc_filename(phonopy_file: Path) -> Path:
    """Get corresponding force_constants filename following Janus convention

    If the filename follows the pattern "seedname-phonopy.yml" this will be
    "seedname-force_constants.hdf5" in the same directory.

    Otherwise, return Path.cwd(), which will fail an .is_file() check.
    """

    if re_match := re.match(r'(?P<seedname>.+)-phonopy\.(?P<ext>ya?ml)', phonopy_file.name):
        seedname = re_match.group('seedname')
        return Path(phonopy_file.parent / f'{seedname}-force_constants.hdf5')
    return Path.cwd()


def load_data_from_file(filename: str | os.PathLike,
                        frequencies_only: bool = False,
                        verbose: bool = False,
) -> QpointPhononModes | QpointFrequencies | ForceConstants:
    """
    Load phonon mode or force constants data from file

    Parameters
    ----------
    filename
        The file with a path
    frequencies_only
        If true only reads frequencies (not eigenvectors) from the
        file. Only applies if the file is not a force constants
        file.

    Returns
    -------
    file_data
    """
    castep_qpm_suffixes = ('.phonon',)
    castep_fc_suffixes = ('.castep_bin', '.check')
    phonopy_suffixes = ('.hdf5', '.yaml', '.yml')

    path = Path(filename)
    if path.suffix in castep_qpm_suffixes:
        if frequencies_only:
            data = QpointFrequencies.from_castep(path)
        else:
            data = QpointPhononModes.from_castep(path)
    elif path.suffix in castep_fc_suffixes:
        data = ForceConstants.from_castep(path)
    elif path.suffix == '.json':
        data = _load_euphonic_json(path, frequencies_only)
    elif path.suffix in phonopy_suffixes:
        data = _load_phonopy_file(path, frequencies_only)
    else:
        msg = format_error(
            f'File format ({path.suffix}) not recognised.',
            reason=f"""
            CASTEP force constants data for
            import should have extension from {castep_fc_suffixes}, CASTEP
            phonon mode data for import should have extension
            '{castep_qpm_suffixes}', data from Phonopy should have extension
            from {phonopy_suffixes}, data from Euphonic should have extension
            '.json'.""",
            fix='Ensure file format in known formats.',
        )
        raise ValueError(msg)
    if verbose:
        print(f'{data.__class__.__name__} data was loaded')
    return data

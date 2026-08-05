from contextlib import contextmanager
from pathlib import Path
import re
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from euphonic.util import convert_fc_phases, format_error

if TYPE_CHECKING:
    import h5py

BOUNDARY_TOLERANCE: float = 1e-12


class CrystalDict(TypedDict):
    cell_vectors: np.ndarray
    cell_vectors_unit: str
    atom_r: np.ndarray
    atom_type: np.ndarray
    atom_mass: np.ndarray
    atom_mass_unit: str


class PhononDataDict(TypedDict, total=False):
    crystal: CrystalDict
    qpts: np.ndarray
    frequencies: np.ndarray
    frequencies_unit: str
    weights: np.ndarray
    eigenvectors: np.ndarray


class InterpolationDataDict(TypedDict, total=False):
    crystal: CrystalDict
    force_constants: np.ndarray
    force_constants_unit: str
    sc_matrix: np.ndarray
    cell_origins: np.ndarray
    born: np.ndarray
    born_unit: str
    dielectric: np.ndarray
    dielectric_unit: str


class BornDict(TypedDict, total=False):
    born_raw: np.ndarray
    dielectric: np.ndarray


class ImportVaspReaderError(ModuleNotFoundError):
    """
    Error raised when h5py is required to read VASP HDF5 files but is missing.
    """

    def __init__(self) -> None:
        self.message = format_error(
            'Cannot import h5py to read VASP HDF5 files.',
            fix=(
                'To install optional HDF5 dependencies for Euphonic, try: '
                'pip install euphonic[phonopy-reader]'
            ),
        )

    def __str__(self) -> str:
        return self.message


class MissingPhononModesError(KeyError):
    """
    Error raised when precalculated phonon modes/frequencies are missing.
    """


class MissingPrimitiveCellError(KeyError):
    """
    Error raised when primitive cell structure is missing in VASP HDF5 file.
    """


def _normalize_fractional_coords(pos: np.ndarray) -> np.ndarray:
    """
    Normalizes fractional atomic positions to [0.0, 1.0), snapping boundary
    values within BOUNDARY_TOLERANCE of 1.0 or 0.0 to 0.0.
    """
    atom_r = pos % 1.0
    near_boundary = np.isclose(
        atom_r, 1.0, atol=BOUNDARY_TOLERANCE
    ) | np.isclose(atom_r, 0.0, atol=BOUNDARY_TOLERANCE)
    atom_r[near_boundary] = 0.0
    return atom_r


@contextmanager
def _open_vasp_h5(filename: Path):
    """
    Context manager to open a VASP HDF5 file with error handling for h5py.
    """
    try:
        import h5py
    except (ModuleNotFoundError, ImportError) as err:
        raise ImportVaspReaderError from err

    filepath = Path(filename)
    if not filepath.exists():
        msg = format_error(
            f'VASP file not found at {filepath}.',
            fix='Provide a valid path to an existing VASP HDF5 file.',
        )
        raise FileNotFoundError(msg)

    with h5py.File(filepath, 'r') as h5_file:
        yield h5_file


def _extract_pomass(h5_file: 'h5py.File') -> list[float]:
    """
    Extracts atomic masses (POMASS) per species from INCAR or POTCAR
    content datasets stored inside the VASP HDF5 file.

    Parameters
    ----------
    h5_file
        Opened h5py.File object representing the VASP HDF5 container

    Returns
    -------
    masses_per_type
        List of float atomic masses in amu per species

    Raises
    ------
    ValueError
        If atomic masses (POMASS) cannot be found in INCAR or POTCAR datasets.
    """
    filename = Path(h5_file.filename)

    # 1. Try active input/incar/POMASS
    if 'input/incar/POMASS' in h5_file:
        val = h5_file['input/incar/POMASS'].asstr()[()]
        raw_vals = re.findall(r'[0-9.]+', str(val))
        if not raw_vals:
            msg = format_error(
                f'POMASS found in input/incar/POMASS but could not parse '
                f'numeric values from: {val!r}',
                fix='Ensure POMASS contains valid numeric values.',
            )
            raise ValueError(msg)
        return [float(mass) for mass in raw_vals]

    # 2. Try original/incar/content
    if 'original/incar/content' in h5_file:
        incar_content = h5_file['original/incar/content'].asstr()[()]
        match = re.search(
            r'POMASS\s*=\s*(?P<masses>[0-9.\s,]+)', incar_content
        )
        if match:
            raw_vals = re.findall(r'[0-9.]+', match.group('masses'))
            if not raw_vals:
                msg = format_error(
                    f'POMASS found in original/incar/content but could not '
                    f'parse numeric values from: {match.group("masses")!r}',
                    fix='Ensure POMASS contains valid numeric values.',
                )
                raise ValueError(msg)
            return [float(mass) for mass in raw_vals]

    # 3. Try POTCAR content stored inside the HDF5 file
    if 'input/potcar/content' in h5_file:
        potcar_content = h5_file['input/potcar/content'].asstr()[()]
        matches = re.findall(r'POMASS\s*=\s*(?P<mass>[0-9.]+)', potcar_content)
        if matches:
            return [float(mass) for mass in matches]

    # 4. If missing from all, raise error
    msg = format_error(
        f'Could not find atomic masses (POMASS) in {filename}.',
        fix=(
            'Ensure POMASS is set in INCAR, or that POTCAR contains '
            'POMASS data.'
        ),
    )
    raise ValueError(msg)


def _read_cell_from_group(
    h5_file: 'h5py.File', group_path: str
) -> CrystalDict:
    """
    Helper function to parse crystal structure from a specific HDF5 group.
    """
    pos_group = h5_file[group_path]
    latt = pos_group['lattice_vectors'][()]
    pos = pos_group['position_ions'][()]
    species_counts = pos_group['number_ion_types'][()]
    species_types = pos_group['ion_types'].asstr()[()]

    species_masses = _extract_pomass(h5_file)

    atom_type = np.repeat(species_types, species_counts)
    atom_mass = np.repeat(species_masses, species_counts)

    atom_r = _normalize_fractional_coords(pos)

    return {
        'cell_vectors': latt,
        'cell_vectors_unit': 'angstrom',
        'atom_r': atom_r,
        'atom_type': atom_type,
        'atom_mass': atom_mass,
        'atom_mass_unit': 'amu',
    }


def read_cell(filename: Path) -> CrystalDict:
    """
    Reads calculation cell structure from input/poscar in VASP HDF5 file.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 output file

    Returns
    -------
    crystal_dict
        A CrystalDict for the calculation cell

    Raises
    ------
    FileNotFoundError
        If the file does not exist at filename.
    ImportVaspReaderError
        If h5py is not installed.
    KeyError
        If input/poscar group is missing from the file.
    ValueError
        If atomic masses (POMASS) cannot be found in INCAR or POTCAR.
    """
    with _open_vasp_h5(filename) as h5_file:
        if 'input/poscar' in h5_file:
            return _read_cell_from_group(h5_file, 'input/poscar')

        msg = format_error(
            f'Crystal position data not found in {filename}.',
            fix='Ensure the file contains input/poscar group.',
        )
        raise KeyError(msg)


def read_primitive_cell(filename: Path) -> CrystalDict:
    """
    Reads primitive cell structure from results/phonons/primitive.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 output file

    Returns
    -------
    crystal_dict
        A CrystalDict for the primitive cell

    Raises
    ------
    MissingPrimitiveCellError
        If primitive cell structure is not found in the file.
    FileNotFoundError
        If the file does not exist at filename.
    ImportVaspReaderError
        If h5py is not installed.
    ValueError
        If atomic masses (POMASS) cannot be found in INCAR or POTCAR.
    """
    with _open_vasp_h5(filename) as h5_file:
        if 'results/phonons/primitive' in h5_file:
            return _read_cell_from_group(h5_file, 'results/phonons/primitive')

        msg = format_error(
            f'Primitive cell structure not found in {filename}.',
            fix='Ensure the file contains results/phonons/primitive data.',
        )
        raise MissingPrimitiveCellError(msg)


def read_phonon_data(filename: Path) -> PhononDataDict:
    """
    Reads precalculated phonon mode/band data from a VASP HDF5 file in native
    THz units.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 file

    Returns
    -------
    data_dict
        A PhononDataDict with keys: 'crystal', 'qpts', 'frequencies',
        'frequencies_unit', 'eigenvectors', 'weights'

    Raises
    ------
    MissingPhononModesError
        If precalculated phonon mode/band data is not found in the file.
    FileNotFoundError
        If the file does not exist at filename.
    ImportVaspReaderError
        If h5py is not installed.
    """
    with _open_vasp_h5(filename) as h5_file:
        if 'results/phonons/frequencies' not in h5_file:
            msg = format_error(
                f'Pre-calculated phonon band data not found in {filename}.',
                fix='Use ForceConstants.from_vasp to read force constants.',
            )
            raise MissingPhononModesError(msg)

        phonon_group = h5_file['results/phonons']

        try:
            crystal_dict = read_primitive_cell(filename)
        except MissingPrimitiveCellError:
            crystal_dict = read_cell(filename)

        n_atoms = len(crystal_dict['atom_r'])

        qpts = phonon_group['qpoint_coords'][()]
        freqs = phonon_group['frequencies'][()]
        weights = phonon_group['qpoints_symmetry_weight'][()]
        evecs_raw = phonon_group['eigenvectors'][()]

        expected_shape = (len(qpts), 3 * n_atoms, n_atoms, 3, 2)
        if evecs_raw.shape != expected_shape:
            msg = format_error(
                f'Unexpected eigenvector array shape {evecs_raw.shape} '
                f'in {filename} (expected {expected_shape}).',
                fix='Ensure the file contains valid VASP 6 eigenvectors.',
            )
            raise ValueError(msg)

        evecs_complex = (
            np.ascontiguousarray(evecs_raw).view(dtype=complex).squeeze(axis=-1)
        )

        return {
            'crystal': crystal_dict,
            'qpts': qpts,
            'frequencies': freqs,
            'frequencies_unit': 'THz',
            'weights': weights,
            'eigenvectors': evecs_complex,
        }


def _find_fc_key(h5_file: 'h5py.File') -> str:
    """
    Finds dataset key for force constants or Hessian matrix in HDF5 container.

    Raises
    ------
    KeyError
        If results/linear_response force constants group is missing.
    """
    if 'results/linear_response/force_constants' in h5_file:
        return 'results/linear_response/force_constants'
    if 'results/linear_response/hessian' in h5_file:
        return 'results/linear_response/hessian'

    filename = Path(h5_file.filename)
    msg = format_error(
        f'Force constants not found in {filename}.',
        fix=(
            'Ensure the file contains results/linear_response '
            'force constants.'
        ),
    )
    raise KeyError(msg)


def _extract_born_and_dielectric(h5_file: 'h5py.File') -> BornDict:
    """
    Extracts Born charges and electronic dielectric tensor from HDF5 container.
    """
    born_dict = {}
    if 'results/linear_response/born_charges' in h5_file:
        born_dict['born_raw'] = h5_file[
            'results/linear_response/born_charges'
        ][()]

    if 'results/linear_response/electron_dielectric_tensor' in h5_file:
        born_dict['dielectric'] = h5_file[
            'results/linear_response/electron_dielectric_tensor'
        ][()]

    return born_dict


def _build_supercell_data(
    crystal_dict: CrystalDict,
    sc_hessian: np.ndarray,
    born_dict: BornDict,
) -> InterpolationDataDict:
    """
    Builds interpolation data dictionary for supercell-as-unit-cell fallback.

    Parameters
    ----------
    crystal_dict
        Crystal dictionary for the supercell (treated as unit cell)
    sc_hessian
        Raw supercell Hessian matrix of shape
        (3 * n_atoms, 3 * n_atoms) in eV/Angstrom^2
    born_dict
        Dictionary containing optional born charges and dielectric tensor
    """
    uc_n_atoms = len(crystal_dict['atom_r'])
    assert sc_hessian.shape == (3 * uc_n_atoms, 3 * uc_n_atoms), (
        f'Expected sc_hessian shape ({3 * uc_n_atoms}, {3 * uc_n_atoms}), '
        f'got {sc_hessian.shape}'
    )
    fc = -sc_hessian.reshape(1, 3 * uc_n_atoms, 3 * uc_n_atoms)

    result: InterpolationDataDict = {
        'crystal': crystal_dict,
        'force_constants': fc,
        'force_constants_unit': 'eV/angstrom**2',
        'sc_matrix': np.eye(3, dtype=int),
        'cell_origins': np.zeros((1, 3), dtype=int),
    }

    if 'born_raw' in born_dict:
        result['born'] = born_dict['born_raw']
        result['born_unit'] = 'e'

    if 'dielectric' in born_dict:
        result['dielectric'] = born_dict['dielectric']
        result['dielectric_unit'] = '(e**2)/(bohr*hartree)'

    return result


def _build_primitive_data(
    h5_file: 'h5py.File',
    crystal_dict: CrystalDict,
    sc_hessian: np.ndarray,
    born_dict: BornDict,
) -> InterpolationDataDict:
    """
    Builds interpolation data dictionary by transforming supercell
    force constants to primitive cell coordinates and cell origins.

    Parameters
    ----------
    h5_file
        Opened h5py.File object representing the VASP HDF5 container
    crystal_dict
        Crystal dictionary for the primitive cell
    sc_hessian
        Raw supercell Hessian matrix of shape
        (3 * sc_n_atoms, 3 * sc_n_atoms) in eV/Angstrom^2
    born_dict
        Dictionary containing optional born charges and dielectric tensor
    """
    uc_n_atoms = len(crystal_dict['atom_r'])

    prim_group = h5_file['results/phonons/primitive']
    prim_lat = prim_group['lattice_vectors'][()]
    prim_pos = prim_group['position_ions'][()]
    atom_r = _normalize_fractional_coords(prim_pos)

    # Use input/poscar for equilibrium structure (results/positions may contain
    # displaced positions from finite-difference Hessian calculation)
    pos_group = h5_file['input/poscar']
    sc_lat = pos_group['lattice_vectors'][()]
    sc_pos = pos_group['position_ions'][()]
    sc_n_atoms = len(sc_pos)

    assert sc_hessian.shape == (3 * sc_n_atoms, 3 * sc_n_atoms), (
        f'Expected sc_hessian shape ({3 * sc_n_atoms}, {3 * sc_n_atoms}), '
        f'got {sc_hessian.shape}'
    )

    sc_matrix = np.rint(sc_lat @ np.linalg.inv(prim_lat)).astype(int)
    sc_atom_r = (sc_pos @ sc_lat) @ np.linalg.inv(prim_lat)

    cell_origins_per_atom = np.floor(sc_atom_r + 1e-5).astype(int)
    r_in_p = _normalize_fractional_coords(
        sc_atom_r - cell_origins_per_atom
    )

    # Map each supercell atom index (0 to sc_n_atoms - 1) to its
    # equivalent primitive unit cell atom index (0 to uc_n_atoms - 1)
    sc_to_uc_atom_idx = np.zeros(sc_n_atoms, dtype=int)
    for i, pos in enumerate(r_in_p):
        diffs = np.linalg.norm(atom_r - pos, axis=1)
        sc_to_uc_atom_idx[i] = np.argmin(diffs)

    # Map each primitive unit cell atom index to one corresponding
    # supercell atom index (used for indexing Born charges)
    uc_to_sc_atom_idx = np.zeros(uc_n_atoms, dtype=int)
    for k in range(uc_n_atoms):
        uc_to_sc_atom_idx[k] = np.where(sc_to_uc_atom_idx == k)[0][0]

    fc_4d = -sc_hessian.reshape(sc_n_atoms, 3, sc_n_atoms, 3).transpose(
        0, 2, 1, 3
    )

    fc_converted, cell_origins = convert_fc_phases(
        fc_4d,
        atom_r,
        sc_atom_r,
        uc_to_sc_atom_idx,
        sc_to_uc_atom_idx,
        sc_matrix,
    )

    result: InterpolationDataDict = {
        'crystal': crystal_dict,
        'force_constants': fc_converted,
        'force_constants_unit': 'eV/angstrom**2',
        'sc_matrix': sc_matrix,
        'cell_origins': cell_origins,
    }

    if 'born_raw' in born_dict:
        result['born'] = born_dict['born_raw'][uc_to_sc_atom_idx]
        result['born_unit'] = 'e'

    if 'dielectric' in born_dict:
        result['dielectric'] = born_dict['dielectric']
        result['dielectric_unit'] = '(e**2)/(bohr*hartree)'

    return result


def read_interpolation_data(filename: Path) -> InterpolationDataDict:
    """
    Reads force constants, Born charges, dielectric tensor, and crystal
    structure data from a VASP HDF5 file in native VASP units.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 file

    Returns
    -------
    data_dict
        A dict with keys: 'crystal', 'force_constants', 'force_constants_unit',
        'sc_matrix', 'cell_origins'. Also optionally contains 'born',
        'born_unit', 'dielectric', and 'dielectric_unit' if present.

    Raises
    ------
    KeyError
        If results/linear_response force constants group is missing.
    FileNotFoundError
        If the file does not exist at filename.
    ImportVaspReaderError
        If h5py is not installed.
    """
    with _open_vasp_h5(filename) as h5_file:
        fc_key = _find_fc_key(h5_file)
        sc_hessian = h5_file[fc_key][()]
        born_dict = _extract_born_and_dielectric(h5_file)

        try:
            crystal_dict = read_primitive_cell(filename)
            return _build_primitive_data(
                h5_file, crystal_dict, sc_hessian, born_dict
            )
        except MissingPrimitiveCellError:
            crystal_dict = read_cell(filename)
            return _build_supercell_data(crystal_dict, sc_hessian, born_dict)

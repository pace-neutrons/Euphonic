import warnings
from pathlib import Path
import re
from typing import Any
import numpy as np

from euphonic.ureg import ureg
from euphonic.util import format_error

# Standard fallback atomic masses (amu) if POTCAR is stripped/sanitized
STANDARD_ATOMIC_MASSES: dict[str, float] = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.95, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91,
    'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71,
    'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91,
    'Ba': 137.33, 'La': 140.12, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97,
    'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23,
    'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38,
    'Pb': 207.2, 'Bi': 208.98, 'Th': 232.04, 'Pa': 231.04, 'U': 238.03,
}


class ImportVaspReaderError(ModuleNotFoundError):
    """
    Error raised when h5py is required to read VASP HDF5 files but is missing.
    """

    def __init__(self):
        self.message = (
            '\n\nCannot import h5py to read VASP HDF5 files, maybe '
            'it is not installed. To install the optional dependency '
            "for Euphonic's VASP reader, try:\n\n"
            'pip install euphonic[phonopy-reader]\n'
        )

    def __str__(self):
        return self.message


def _open_vasp_h5(filename: Path | str):
    """
    Helper function to open an HDF5 file with error handling for missing h5py.
    """
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise ImportVaspReaderError from e

    filepath = Path(filename)
    if not filepath.exists():
        msg = format_error(f'VASP file not found at {filepath}.')
        raise FileNotFoundError(msg)

    return h5py.File(filepath, 'r')


def _extract_pomass(f, filename: Path | str, n_species: int) -> list[float]:
    """
    Extracts atomic masses (POMASS) per species from POTCAR or INCAR in the HDF5 file.
    """
    # 1. Try POTCAR content
    if 'input/potcar/content' in f:
        content = f['input/potcar/content'][()].decode('utf-8', errors='ignore')
        matches = re.findall(r'POMASS\s*=\s*([0-9.]+)', content)
        if len(matches) == n_species:
            return [float(m) for m in matches]

    # 2. Try INCAR content or dataset
    incar_content = ''
    if 'original/incar/content' in f:
        incar_content = f['original/incar/content'][()].decode('utf-8', errors='ignore')
    elif 'input/incar/POMASS' in f:
        val = f['input/incar/POMASS'][()]
        incar_content = f'POMASS = {val}'

    if incar_content:
        match = re.search(r'POMASS\s*=\s*([0-9.\s,]+)', incar_content, re.IGNORECASE)
        if match:
            raw_vals = re.findall(r'[0-9.]+', match.group(1))
            if len(raw_vals) == n_species:
                return [float(m) for m in raw_vals]

    # 3. If missing from both, raise error
    msg = format_error(
        f'Could not find atomic masses (POMASS) in input/potcar/content or INCAR in {filename}.',
        fix='Ensure the file contains POMASS in POTCAR or INCAR.',
    )
    raise ValueError(msg)


def read_crystal(
    filename: Path | str,
    cell_vectors_unit: str = 'angstrom',
    atom_mass_unit: str = 'amu',
) -> dict[str, Any]:
    """
    Reads crystal structure information from a VASP HDF5 file.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 output file (e.g. vaspout.h5)
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in

    Returns
    -------
    crystal_dict
        A dict with keys: 'cell_vectors', 'cell_vectors_unit', 'atom_r',
        'atom_type', 'atom_mass', 'atom_mass_unit'
    """
    with _open_vasp_h5(filename) as f:
        if 'results/positions' not in f:
            msg = format_error(
                f'Crystal position data not found in {filename}.',
                fix='Ensure the file contains results/positions group.',
            )
            raise KeyError(msg)

        latt = f['results/positions/lattice_vectors'][()]
        pos = f['results/positions/position_ions'][()]
        types_count = f['results/positions/number_ion_types'][()]
        types_raw = f['results/positions/ion_types'][()]

        species_types = [
            r.decode('utf-8').strip() if isinstance(r, bytes) else str(r).strip()
            for r in types_raw
        ]

        # Read exact atomic mass (POMASS) from POTCAR or INCAR in HDF5 file
        masses_per_type = _extract_pomass(f, filename, len(species_types))

        atom_species = []
        atom_masses = []
        for s, c, m in zip(species_types, types_count, masses_per_type):
            atom_species.extend([s] * int(c))
            atom_masses.extend([m] * int(c))

        # Convert units if necessary (lattice vectors in VASP are in angstrom)
        cell_vectors = (
            latt * ureg('angstrom').to(cell_vectors_unit).magnitude
        )
        atom_masses_converted = (
            np.array(atom_masses) * ureg('amu').to(atom_mass_unit).magnitude
        )

        atom_r = pos - np.floor(pos)

        return {
            'cell_vectors': cell_vectors,
            'cell_vectors_unit': cell_vectors_unit,
            'atom_r': atom_r,
            'atom_type': np.array(atom_species),
            'atom_mass': atom_masses_converted,
            'atom_mass_unit': atom_mass_unit,
        }


def read_phonon_data(
    filename: Path | str,
    cell_vectors_unit: str = 'angstrom',
    atom_mass_unit: str = 'amu',
    frequencies_unit: str = 'meV',
) -> dict[str, Any]:
    """
    Reads precalculated or Gamma-point phonon mode data from a VASP HDF5 file.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 file
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    frequencies_unit
        The unit to return the frequencies in

    Returns
    -------
    data_dict
        A dict with keys: 'crystal', 'qpts', 'frequencies',
        'frequencies_unit', 'eigenvectors', 'weights'
    """
    crystal_dict = read_crystal(
        filename,
        cell_vectors_unit=cell_vectors_unit,
        atom_mass_unit=atom_mass_unit,
    )

    with _open_vasp_h5(filename) as f:
        n_atoms = len(crystal_dict['atom_r'])

        # Check if linear response Hessian/force_constants are present
        if 'results/linear_response/force_constants' in f or 'results/linear_response/hessian' in f:
            fc_key = (
                'results/linear_response/force_constants'
                if 'results/linear_response/force_constants' in f
                else 'results/linear_response/hessian'
            )
            fc = f[fc_key][()]

            masses = (
                crystal_dict['atom_mass']
                * ureg(atom_mass_unit).to('amu').magnitude
            )
            mass_matrix = np.outer(np.repeat(masses, 3), np.repeat(masses, 3))
            # VASP force constants / Hessian matrix convention is -D = FC / sqrt(m_i m_j)
            dyn_mat = -fc / np.sqrt(mass_matrix)
            dyn_mat = 0.5 * (dyn_mat + dyn_mat.T)

            evals, evecs = np.linalg.eigh(dyn_mat)

            # Conversion factor from eV/(A^2 * amu) to THz
            eV = 1.602176634e-19
            angstrom = 1e-10
            amu = 1.66053906660e-27
            factor_thz = (
                (eV / (angstrom**2 * amu)) ** 0.5 / (2 * np.pi * 1e12)
            )

            freqs_thz = np.sign(evals) * np.sqrt(np.abs(evals)) * factor_thz
            freqs_converted = (
                freqs_thz * ureg('THz').to(frequencies_unit).magnitude
            )

            evecs_reshaped = (
                evecs.T.reshape(1, 3 * n_atoms, n_atoms, 3).astype(
                    np.complex128
                )
            )

            return {
                'crystal': crystal_dict,
                'qpts': np.array([[0.0, 0.0, 0.0]]),
                'frequencies': freqs_converted.reshape(1, -1),
                'frequencies_unit': frequencies_unit,
                'eigenvectors': evecs_reshaped,
                'weights': np.array([1.0]),
            }

        msg = format_error(
            f'Phonon data not found in {filename}.',
            fix='Ensure the file contains results/linear_response force constants.',
        )
        raise KeyError(msg)


def read_interpolation_data(
    filename: Path | str,
    cell_vectors_unit: str = 'angstrom',
    atom_mass_unit: str = 'amu',
    force_constants_unit: str = 'hartree/bohr**2',
    born_unit: str = 'e',
    dielectric_unit: str = '(e**2)/(bohr*hartree)',
) -> dict[str, Any]:
    """
    Reads force constants, Born charges, dielectric tensor, and crystal
    structure data from a VASP HDF5 file.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 file
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    force_constants_unit
        The unit to return the force constants in
    born_unit
        The unit to return the Born charges in
    dielectric_unit
        The unit to return the dielectric permittivity tensor in

    Returns
    -------
    data_dict
        A dict with keys: 'crystal', 'force_constants', 'force_constants_unit',
        'sc_matrix', 'cell_origins'. Also optionally contains 'born',
        'born_unit', 'dielectric', and 'dielectric_unit' if present.
    """
    crystal_dict = read_crystal(
        filename,
        cell_vectors_unit=cell_vectors_unit,
        atom_mass_unit=atom_mass_unit,
    )

    with _open_vasp_h5(filename) as f:
        n_atoms = len(crystal_dict['atom_r'])

        if 'results/linear_response/force_constants' in f or 'results/linear_response/hessian' in f:
            fc_key = (
                'results/linear_response/force_constants'
                if 'results/linear_response/force_constants' in f
                else 'results/linear_response/hessian'
            )
            fc_raw = f[fc_key][()]

            # VASP force constants matrix is -Hessian
            fc = -fc_raw.reshape(1, 3 * n_atoms, 3 * n_atoms)
            fc_converted = (
                fc * ureg('eV/angstrom**2').to(force_constants_unit).magnitude
            )

            data_dict = {
                'crystal': crystal_dict,
                'force_constants': fc_converted,
                'force_constants_unit': force_constants_unit,
                'sc_matrix': np.eye(3, dtype=int),
                'cell_origins': np.zeros((1, 3), dtype=int),
            }

            if 'results/linear_response/born_charges' in f:
                born_raw = f['results/linear_response/born_charges'][()]
                data_dict['born'] = (
                    born_raw * ureg('e').to(born_unit).magnitude
                )
                data_dict['born_unit'] = born_unit

            if 'results/linear_response/electron_dielectric_tensor' in f:
                dielectric_raw = f[
                    'results/linear_response/electron_dielectric_tensor'
                ][()]
                data_dict['dielectric'] = (
                    dielectric_raw
                    * ureg('e**2/(hartree*bohr)').to(dielectric_unit).magnitude
                )
                data_dict['dielectric_unit'] = dielectric_unit

            return data_dict

        msg = format_error(
            f'Force constants not found in {filename}.',
            fix='Ensure the file contains results/linear_response force constants.',
        )
        raise KeyError(msg)

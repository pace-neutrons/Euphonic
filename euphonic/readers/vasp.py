from contextlib import contextmanager
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from euphonic.ureg import ureg
from euphonic.util import convert_fc_phases, format_error

if TYPE_CHECKING:
    import h5py


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


@contextmanager
def _open_vasp_h5(filename: Path | str):
    """
    Context manager to open a VASP HDF5 file with error handling for h5py.
    """
    try:
        import h5py
    except ModuleNotFoundError as err:
        raise ImportVaspReaderError from err

    filepath = Path(filename)
    if not filepath.exists():
        msg = format_error(f'VASP file not found at {filepath}.')
        raise FileNotFoundError(msg)

    with h5py.File(filepath, 'r') as h5_file:
        yield h5_file


def _extract_pomass(
    h5_file: 'h5py.File', filename: Path, n_species: int
) -> list[float]:
    """
    Extracts atomic masses (POMASS) per species from input/potcar/content or
    INCAR content datasets stored inside the VASP HDF5 file.

    Parameters
    ----------
    h5_file
        Opened h5py.File object representing the VASP HDF5 container
    filename
        Path to the VASP HDF5 file for error reporting
    n_species
        Number of atomic species expected

    Returns
    -------
    masses_per_type
        List of float atomic masses in amu per species
    """
    # 1. Try POTCAR content stored inside the HDF5 file
    if 'input/potcar/content' in h5_file:
        potcar_content = h5_file['input/potcar/content'][()].decode(
            'utf-8', errors='ignore'
        )
        matches = re.findall(r'POMASS\s*=\s*([0-9.]+)', potcar_content)
        if len(matches) == n_species:
            return [float(mass) for mass in matches]

    # 2. Try INCAR content stored inside the HDF5 file
    incar_content = ''
    if 'original/incar/content' in h5_file:
        incar_content = h5_file['original/incar/content'][()].decode(
            'utf-8', errors='ignore'
        )
    elif 'input/incar/POMASS' in h5_file:
        val = h5_file['input/incar/POMASS'][()]
        incar_content = f'POMASS = {val}'

    if incar_content:
        match = re.search(
            r'POMASS\s*=\s*([0-9.\s,]+)', incar_content, re.IGNORECASE
        )
        if match:
            raw_vals = re.findall(r'[0-9.]+', match.group(1))
            if len(raw_vals) == n_species:
                return [float(mass) for mass in raw_vals]

    # 3. If missing from both, raise error
    msg = format_error(
        f'Could not find atomic masses (POMASS) in {filename}.',
        fix='Ensure the file contains POMASS in POTCAR or INCAR datasets.',
    )
    raise ValueError(msg)


def read_crystal(
    filename: Path,
    *,
    cell_vectors_unit: str = 'angstrom',
    atom_mass_unit: str = 'amu',
    use_primitive: bool = False,
) -> dict[str, Any]:
    """
    Reads crystal structure information from a VASP HDF5 file.

    Parameters
    ----------
    filename
        Path to the VASP HDF5 output file
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    use_primitive
        Whether to attempt reading primitive cell structure if present

    Returns
    -------
    crystal_dict
        A dict with keys: 'cell_vectors', 'cell_vectors_unit', 'atom_r',
        'atom_type', 'atom_mass', 'atom_mass_unit'
    """
    with _open_vasp_h5(filename) as h5_file:
        pos_group = None
        if use_primitive:
            if 'results/phonons/primitive' in h5_file:
                pos_group = h5_file['results/phonons/primitive']
            elif 'results/phonon/primitive' in h5_file:
                pos_group = h5_file['results/phonon/primitive']

        if pos_group is None:
            if 'results/positions' not in h5_file:
                msg = format_error(
                    f'Crystal position data not found in {filename}.',
                    fix='Ensure the file contains results/positions group.',
                )
                raise KeyError(msg)
            pos_group = h5_file['results/positions']

        latt = pos_group['lattice_vectors'][()]
        pos = pos_group['position_ions'][()]
        types_count = pos_group['number_ion_types'][()]
        types_raw = pos_group['ion_types'][()]

        species_types = [
            raw.decode('utf-8').strip()
            if isinstance(raw, bytes)
            else str(raw).strip()
            for raw in types_raw
        ]

        # Read exact atomic mass (POMASS) from POTCAR or INCAR in HDF5 file
        masses_per_type = _extract_pomass(
            h5_file, filename, len(species_types)
        )

        atom_species = []
        atom_masses = []
        for species, count, mass in zip(
            species_types, types_count, masses_per_type, strict=False
        ):
            atom_species.extend([species] * int(count))
            atom_masses.extend([mass] * int(count))

        cell_vectors = latt * ureg('angstrom').to(cell_vectors_unit).magnitude
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
    filename: Path,
    *,
    cell_vectors_unit: str = 'angstrom',
    atom_mass_unit: str = 'amu',
    frequencies_unit: str = 'meV',
) -> dict[str, Any]:
    """
    Reads precalculated phonon mode/band data from a VASP HDF5 file.

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
        'frequencies_unit', 'eigenvectors' (optional), 'weights'
    """
    with _open_vasp_h5(filename) as h5_file:
        phonon_group = None
        if 'results/phonons' in h5_file:
            phonon_group = h5_file['results/phonons']
        elif 'results/phonon' in h5_file:
            phonon_group = h5_file['results/phonon']

        if phonon_group is None or (
            'frequencies' not in phonon_group
            and 'eigenvalues' not in phonon_group
        ):
            msg = format_error(
                f'Pre-calculated phonon band data not found in {filename}.',
                fix='Use ForceConstants.from_vasp to read force constants.',
            )
            raise MissingPhononModesError(msg)

        has_primitive = 'primitive' in phonon_group
        crystal_dict = read_crystal(
            filename,
            cell_vectors_unit=cell_vectors_unit,
            atom_mass_unit=atom_mass_unit,
            use_primitive=has_primitive,
        )

        n_atoms = len(crystal_dict['atom_r'])

        qpts_key = (
            'qpoint_coords'
            if 'qpoint_coords' in phonon_group
            else (
                'kpoint_coords'
                if 'kpoint_coords' in phonon_group
                else 'kpoints'
            )
        )
        freq_key = (
            'frequencies' if 'frequencies' in phonon_group else 'eigenvalues'
        )

        qpts = phonon_group[qpts_key][()]
        freqs_raw = phonon_group[freq_key][()]

        raw_unit = 'THz'
        if (
            'frequencies' in phonon_group
            and 'unit' in phonon_group['frequencies'].attrs
        ):
            attr_val = phonon_group['frequencies'].attrs['unit']
            raw_unit = (
                attr_val.decode('utf-8')
                if isinstance(attr_val, bytes)
                else str(attr_val)
            )

        freqs_converted = (
            freqs_raw * ureg(raw_unit).to(frequencies_unit).magnitude
        )

        res = {
            'crystal': crystal_dict,
            'qpts': qpts,
            'frequencies': freqs_converted,
            'frequencies_unit': frequencies_unit,
        }

        if 'qpoints_symmetry_weight' in phonon_group:
            res['weights'] = phonon_group['qpoints_symmetry_weight'][()]
        elif 'kpoint_weights' in phonon_group:
            res['weights'] = phonon_group['kpoint_weights'][()]
        else:
            res['weights'] = np.ones(len(qpts)) / len(qpts)

        if 'eigenvectors' in phonon_group:
            evecs_raw = phonon_group['eigenvectors'][()]
            if evecs_raw.ndim == 5 and evecs_raw.shape[-1] == 2:
                evecs_complex = evecs_raw[..., 0] + 1j * evecs_raw[..., 1]
            else:
                evecs_complex = evecs_raw.astype(np.complex128)

            if (
                evecs_complex.ndim == 3
                and evecs_complex.shape[1] == 3 * n_atoms
            ):
                evecs_complex = evecs_complex.reshape(
                    len(qpts), 3 * n_atoms, n_atoms, 3
                )
            res['eigenvectors'] = evecs_complex

        return res


def read_interpolation_data(
    filename: Path,
    *,
    cell_vectors_unit: str = 'angstrom',
    atom_mass_unit: str = 'amu',
    force_constants_unit: str = 'hartree/bohr**2',
    born_unit: str = 'e',
    dielectric_unit: str = '(e**2)/(bohr*hartree)',
    use_primitive: bool = True,
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
    use_primitive
        Whether to attempt converting force constants to the primitive cell
        if primitive cell structure is present in the file

    Returns
    -------
    data_dict
        A dict with keys: 'crystal', 'force_constants', 'force_constants_unit',
        'sc_matrix', 'cell_origins'. Also optionally contains 'born',
        'born_unit', 'dielectric', and 'dielectric_unit' if present.
    """
    with _open_vasp_h5(filename) as h5_file:
        has_primitive = use_primitive and (
            'results/phonons/primitive' in h5_file
            or 'results/phonon/primitive' in h5_file
        )

        crystal_dict = read_crystal(
            filename,
            cell_vectors_unit=cell_vectors_unit,
            atom_mass_unit=atom_mass_unit,
            use_primitive=has_primitive,
        )

        n_atoms_uc = len(crystal_dict['atom_r'])

        if (
            'results/linear_response/force_constants' in h5_file
            or 'results/linear_response/hessian' in h5_file
        ):
            fc_key = (
                'results/linear_response/force_constants'
                if 'results/linear_response/force_constants' in h5_file
                else 'results/linear_response/hessian'
            )
            fc_raw = h5_file[fc_key][()]

            if has_primitive:
                prim_group = (
                    h5_file['results/phonons/primitive']
                    if 'results/phonons/primitive' in h5_file
                    else h5_file['results/phonon/primitive']
                )
                l_p = prim_group['lattice_vectors'][()]
                r_p = prim_group['position_ions'][()]
                atom_r = r_p - np.floor(r_p)

                pos_group = h5_file['results/positions']
                l_sc = pos_group['lattice_vectors'][()]
                r_sc = pos_group['position_ions'][()]
                n_atoms_sc = len(r_sc)

                sc_matrix = np.rint(l_sc @ np.linalg.inv(l_p)).astype(int)
                sc_atom_r = (r_sc @ l_sc) @ np.linalg.inv(l_p)

                r_sc_pfrac = sc_atom_r
                cell_origins_per_atom = np.floor(r_sc_pfrac + 1e-5).astype(int)
                r_in_p = r_sc_pfrac - cell_origins_per_atom
                r_in_p = r_in_p - np.floor(r_in_p + 1e-5)

                sc_to_uc_atom_idx = np.zeros(n_atoms_sc, dtype=int)
                for i, pos in enumerate(r_in_p):
                    diffs = np.linalg.norm(atom_r - pos, axis=1)
                    sc_to_uc_atom_idx[i] = np.argmin(diffs)

                uc_to_sc_atom_idx = np.zeros(n_atoms_uc, dtype=int)
                for k in range(n_atoms_uc):
                    uc_to_sc_atom_idx[k] = np.where(sc_to_uc_atom_idx == k)[
                        0
                    ][0]

                fc_4d = -fc_raw.reshape(
                    n_atoms_sc, 3, n_atoms_sc, 3
                ).transpose(0, 2, 1, 3)

                fc_converted_raw, cell_origins = convert_fc_phases(
                    fc_4d,
                    atom_r,
                    sc_atom_r,
                    uc_to_sc_atom_idx,
                    sc_to_uc_atom_idx,
                    sc_matrix,
                )

                fc_converted = (
                    fc_converted_raw
                    * ureg('eV/angstrom**2').to(force_constants_unit).magnitude
                )

                data_dict = {
                    'crystal': crystal_dict,
                    'force_constants': fc_converted,
                    'force_constants_unit': force_constants_unit,
                    'sc_matrix': sc_matrix,
                    'cell_origins': cell_origins,
                }

                if 'results/linear_response/born_charges' in h5_file:
                    born_raw = h5_file['results/linear_response/born_charges'][
                        ()
                    ]
                    born_primitive = born_raw[uc_to_sc_atom_idx]
                    data_dict['born'] = (
                        born_primitive * ureg('e').to(born_unit).magnitude
                    )
                    data_dict['born_unit'] = born_unit

                if (
                    'results/linear_response/electron_dielectric_tensor'
                    in h5_file
                ):
                    dielectric_raw = h5_file[
                        'results/linear_response/electron_dielectric_tensor'
                    ][()]
                    data_dict['dielectric'] = (
                        dielectric_raw
                        * ureg('e**2/(hartree*bohr)')
                        .to(dielectric_unit)
                        .magnitude
                    )
                    data_dict['dielectric_unit'] = dielectric_unit

                return data_dict

            # Supercell as unit cell fallback
            fc = -fc_raw.reshape(1, 3 * n_atoms_uc, 3 * n_atoms_uc)
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

            if 'results/linear_response/born_charges' in h5_file:
                born_raw = h5_file['results/linear_response/born_charges'][()]
                data_dict['born'] = (
                    born_raw * ureg('e').to(born_unit).magnitude
                )
                data_dict['born_unit'] = born_unit

            if 'results/linear_response/electron_dielectric_tensor' in h5_file:
                dielectric_raw = h5_file[
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
            fix=(
                'Ensure the file contains results/linear_response '
                'force constants.'
            ),
        )
        raise KeyError(msg)

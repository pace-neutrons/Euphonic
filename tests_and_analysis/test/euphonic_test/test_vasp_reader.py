import builtins
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from euphonic import ForceConstants, QpointFrequencies, QpointPhononModes
from euphonic.readers.vasp import (
    ImportVaspReaderError,
    MissingPhononModesError,
    MissingPrimitiveCellError,
    read_cell,
    read_interpolation_data,
    read_phonon_data,
    read_primitive_cell,
)
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal,
    check_crystal,
)
from tests_and_analysis.test.utils import get_data_path

FC_NO_QPTS_H5 = get_data_path('vasp_files', 'vaspout_sanitized.h5')
ONLY_QPTS_H5 = get_data_path('vasp_files', 'vaspout_dos_sanitized.h5')
FC_AND_QPTS_H5 = get_data_path(
    'vasp_files', 'vaspout_dos_rerun_sanitized.h5'
)
# Non-diagonal supercell without Born charges (Al FCC, 32 primitive cells)
AL_NO_BORN_H5 = get_data_path('vasp_files', 'vaspout_al_no_born.h5')


def get_crystal_path(*subpaths):
    return get_data_path('crystal', *subpaths)


@pytest.mark.vasp_reader
class TestVaspReaderCell:

    def test_read_cell_fc_no_qpts(self):
        cell_dict = read_cell(FC_NO_QPTS_H5)
        cell_data = ExpectedCrystal(
            {**cell_dict, 'n_atoms': len(cell_dict['atom_r'])}
        )
        with open(get_crystal_path('crystal_vasp_fc_no_qpts.json')) as fp:
            expected = ExpectedCrystal(json.load(fp))
        check_crystal(cell_data, expected)

    def test_read_primitive_cell_only_qpts(self):
        prim_dict = read_primitive_cell(ONLY_QPTS_H5)
        prim_data = ExpectedCrystal(
            {**prim_dict, 'n_atoms': len(prim_dict['atom_r'])}
        )
        with open(get_crystal_path('crystal_vasp_only_qpts_prim.json')) as fp:
            expected = ExpectedCrystal(json.load(fp))
        check_crystal(prim_data, expected)

    def test_read_combined_cells(self):
        cell_dict = read_cell(FC_AND_QPTS_H5)
        cell_data = ExpectedCrystal(
            {**cell_dict, 'n_atoms': len(cell_dict['atom_r'])}
        )
        with open(
            get_crystal_path('crystal_vasp_fc_and_qpts_cell.json')
        ) as fp:
            exp_cell = ExpectedCrystal(json.load(fp))
        check_crystal(cell_data, exp_cell)

        prim_dict = read_primitive_cell(FC_AND_QPTS_H5)
        prim_data = ExpectedCrystal(
            {**prim_dict, 'n_atoms': len(prim_dict['atom_r'])}
        )
        with open(
            get_crystal_path('crystal_vasp_fc_and_qpts_prim.json')
        ) as fp:
            exp_prim = ExpectedCrystal(json.load(fp))
        check_crystal(prim_data, exp_prim)

    def test_read_cell_from_incar_override(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_vaspout_incar.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855\nISMEAR = 0')
            )

        data = read_cell(dummy_h5)
        assert_allclose(data['atom_mass'], 28.0855)

    def test_read_cell_incar_overrides_potcar(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_vaspout_priority.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 10.0')
            )

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        data = read_cell(dummy_h5)
        assert_allclose(data['atom_mass'], 28.0855)

    def test_read_cell_input_incar_overrides_original_incar(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_vaspout_input_incar.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 10.0')
            )

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 20.0')
            )

            input_incar = f.create_group('input/incar')
            input_incar.create_dataset('POMASS', data=np.bytes_(b'30.0'))

        data = read_cell(dummy_h5)
        assert_allclose(data['atom_mass'], 30.0)

    def test_read_cell_negative_positions(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_vaspout_neg.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset(
                'position_ions',
                data=np.array([[-0.1, -0.5, -1.0], [1.1, -1e-15, 0.25]]),
            )
            pos_group.create_dataset('number_ion_types', data=np.array([2]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        data = read_cell(dummy_h5)
        assert_allclose(data['atom_r'][0], [0.9, 0.5, 0.0])
        assert_allclose(data['atom_r'][1], [0.1, 0.0, 0.25])

    def test_read_primitive_cell_missing_raises_error(self):
        with pytest.raises(MissingPrimitiveCellError):
            read_primitive_cell(FC_NO_QPTS_H5)

    def test_read_cell_missing_pomass_raises_error(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_vaspout_empty.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

        with pytest.raises(ValueError, match='Could not find atomic masses'):
            read_cell(dummy_h5)


@pytest.mark.vasp_reader
class TestVaspReaderPhononData:

    def test_read_phonon_data_missing_precalculated_raises_error(self):
        with pytest.raises(MissingPhononModesError):
            read_phonon_data(FC_NO_QPTS_H5)

    def test_read_phonon_data_from_only_qpts(self):
        phonon_data = read_phonon_data(ONLY_QPTS_H5)
        assert 'crystal' in phonon_data
        assert phonon_data['crystal']['atom_r'].shape == (
            2,
            3,
        )  # Primitive cell
        assert phonon_data['qpts'].shape == (3, 3)
        assert phonon_data['frequencies'].shape == (3, 6)
        assert phonon_data['frequencies_unit'] == 'THz'
        assert phonon_data['eigenvectors'].shape == (3, 6, 2, 3)

        # Check Gamma-point optical max frequency ~ 7.6315 THz
        max_freq = np.max(phonon_data['frequencies'])
        assert_allclose(max_freq, 7.6315, rtol=1e-3)


@pytest.mark.vasp_reader
class TestQpointPhononModesFromVasp:

    def test_from_vasp_modes_missing_data_raises_error(self):
        with pytest.raises(MissingPhononModesError):
            QpointPhononModes.from_vasp(FC_NO_QPTS_H5)

    def test_from_vasp_modes_from_only_qpts(self):
        modes = QpointPhononModes.from_vasp(ONLY_QPTS_H5)
        assert modes.crystal.n_atoms == 2  # Primitive GaAs cell
        assert modes.frequencies.shape == (3, 6)
        assert modes.eigenvectors.shape == (3, 6, 2, 3)

        max_freq_mev = np.max(modes.frequencies.to('meV').magnitude)
        assert_allclose(max_freq_mev, 31.561, rtol=1e-3)

    def test_from_vasp_frequencies_from_only_qpts(self):
        freqs = QpointFrequencies.from_vasp(ONLY_QPTS_H5)
        assert freqs.crystal.n_atoms == 2
        assert freqs.frequencies.shape == (3, 6)

        max_freq_mev = np.max(freqs.frequencies.to('meV').magnitude)
        assert_allclose(max_freq_mev, 31.561, rtol=1e-3)


@pytest.mark.vasp_reader
class TestForceConstantsFromVasp:

    def test_read_interpolation_data(self):
        import h5py

        data = read_interpolation_data(FC_NO_QPTS_H5)
        assert 'crystal' in data
        assert data['force_constants'].shape == (1, 48, 48)
        assert data['sc_matrix'].shape == (3, 3)
        assert 'born' in data
        assert data['born'].shape == (16, 3, 3)
        assert 'dielectric' in data
        assert data['dielectric'].shape == (3, 3)

        # Assert raw force constants dimensions (3 * n_atoms_sc, 3 * n_atoms_sc)
        with h5py.File(FC_NO_QPTS_H5, 'r') as f:
            fc_raw = f['results/linear_response/force_constants'][()]
            n_atoms_sc = len(f['results/positions/position_ions'][()])
            assert fc_raw.shape == (3 * n_atoms_sc, 3 * n_atoms_sc)

    def test_fc_from_vasp_and_fallback_calculation(self):
        # 1. ForceConstants.from_vasp loads Hessian/force_constants
        fc = ForceConstants.from_vasp(FC_NO_QPTS_H5)
        assert fc.crystal.n_atoms == 16
        assert fc.n_cells_in_sc == 1
        assert fc.force_constants.shape == (1, 48, 48)

        # 2. Outer caller calculates modes explicitly from ForceConstants
        q_freqs = fc.calculate_qpoint_frequencies(np.array([[0.0, 0.0, 0.0]]))

        # 3. Compare with QpointFrequencies loaded from precalculated QPOINTS file
        precalc_freqs = QpointFrequencies.from_vasp(ONLY_QPTS_H5)

        fc_freqs_mev = q_freqs.frequencies.to('meV').magnitude[0]
        precalc_gamma_freqs_mev = (
            precalc_freqs.frequencies.to('meV').magnitude[0]
        )

        # Max optical frequency at Gamma (31.561 meV) must match
        assert_allclose(
            np.max(fc_freqs_mev), np.max(precalc_gamma_freqs_mev), rtol=1e-3
        )

    @pytest.mark.vasp_reader
    def test_fc_from_vasp_without_born(self):
        # Test with no Born charges/dielectric tensor
        # Also a non-diagonal supercell: FCC primitive in cubic supercell
        fc = ForceConstants.from_vasp(AL_NO_BORN_H5)

        # Primitive cell has 1 atom (FCC)
        assert fc.crystal.n_atoms == 1
        # Supercell contains 32 primitive cells
        assert fc.n_cells_in_sc == 32
        # Force constants shape: (n_cells_in_sc, 3*n_atoms, 3*n_atoms)
        assert fc.force_constants.shape == (32, 3, 3)

        # No Born charges or dielectric tensor
        assert fc.born is None
        assert fc.dielectric is None

    @pytest.mark.vasp_reader
    def test_read_interpolation_data_without_born(self):
        data = read_interpolation_data(AL_NO_BORN_H5)

        assert 'crystal' in data
        assert data['force_constants'].shape == (32, 3, 3)
        assert data['sc_matrix'].shape == (3, 3)
        # Non-diagonal supercell matrix for FCC primitive in cubic supercell
        # Determinant = 32 (32 primitive cells in supercell)
        assert_allclose(np.linalg.det(data['sc_matrix']), 32.0)

        # No Born charges or dielectric tensor
        assert 'born' not in data
        assert 'dielectric' not in data


@pytest.mark.vasp_reader
class TestVaspReaderCombined:

    def test_combined_fc_and_modes(self):
        # 1. ForceConstants reads primitive cell force constants (8, 6, 6)
        fc = ForceConstants.from_vasp(FC_AND_QPTS_H5)
        assert fc.crystal.n_atoms == 2
        assert fc.n_cells_in_sc == 8
        assert fc.force_constants.shape == (8, 6, 6)
        assert fc.born is not None
        assert fc.born.shape == (2, 3, 3)

        # 2. QpointPhononModes reads primitive precalculated modes (3 qpts, 6 branches)
        modes = QpointPhononModes.from_vasp(FC_AND_QPTS_H5)
        assert modes.crystal.n_atoms == 2
        assert modes.frequencies.shape == (3, 6)
        assert modes.eigenvectors.shape == (3, 6, 2, 3)

        # 3. Compare calculated frequencies from primitive FC vs precalculated modes
        q_freqs = fc.calculate_qpoint_frequencies(modes.qpts)
        assert_allclose(
            q_freqs.frequencies.to('meV').magnitude,
            modes.frequencies.to('meV').magnitude,
            rtol=1e-2,
            atol=1e-2,
        )


class TestVaspReaderEdgeCases:
    """Tests for edge cases and error handling.

    Note: test_missing_h5py_import_error does NOT have @pytest.mark.vasp_reader
    because it tests behavior when h5py is not installed.
    """

    def test_missing_h5py_import_error(self, mocker, tmp_path):
        """Test that a helpful error is raised when h5py is not installed."""
        dummy_h5 = tmp_path / 'dummy.h5'
        dummy_h5.write_text('dummy')

        real_import = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name == 'h5py':
                raise ModuleNotFoundError
            return real_import(name, *args, **kwargs)

        mocker.patch('builtins.__import__', side_effect=mocked_import)
        with pytest.raises(ImportVaspReaderError) as exc_info:
            read_cell(dummy_h5)
        assert 'Cannot import h5py' in str(exc_info.value)

    @pytest.mark.vasp_reader
    def test_unexpected_eigenvector_shape_raises_error(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_bad_evec.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

            ph_group = f.create_group('results/phonons')
            ph_group.create_dataset('qpoint_coords', data=np.zeros((1, 3)))
            ph_group.create_dataset('frequencies', data=np.zeros((1, 3)))
            ph_group.create_dataset('qpoints_symmetry_weight', data=np.ones(1))
            # Bad eigenvector shape
            ph_group.create_dataset('eigenvectors', data=np.zeros((1, 1, 1)))

        with pytest.raises(
            ValueError, match='Unexpected eigenvector array shape'
        ):
            read_phonon_data(dummy_h5)

    @pytest.mark.vasp_reader
    def test_read_cell_file_not_found_raises_error(self, tmp_path):
        non_existent = tmp_path / 'non_existent.h5'
        with pytest.raises(FileNotFoundError, match='VASP file not found'):
            read_cell(non_existent)

    @pytest.mark.vasp_reader
    def test_read_cell_missing_group_raises_key_error(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_nogroup.h5'
        with h5py.File(dummy_h5, 'w') as f:
            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        with pytest.raises(KeyError, match='Crystal position data not found'):
            read_cell(dummy_h5)

    @pytest.mark.vasp_reader
    def test_read_cell_falls_back_to_poscar(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_poscar.h5'
        with h5py.File(dummy_h5, 'w') as f:
            poscar_group = f.create_group('input/poscar')
            poscar_group.create_dataset('lattice_vectors', data=np.eye(3))
            poscar_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            poscar_group.create_dataset('number_ion_types', data=np.array([1]))
            poscar_group.create_dataset(
                'ion_types', data=np.array([b'Si'])
            )

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        data = read_cell(dummy_h5)
        assert data['cell_vectors'].shape == (3, 3)

    @pytest.mark.vasp_reader
    def test_read_cell_no_positions_or_poscar_raises_key_error(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_nopos.h5'
        with h5py.File(dummy_h5, 'w') as f:
            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        with pytest.raises(KeyError, match='Crystal position data not found'):
            read_cell(dummy_h5)

    @pytest.mark.vasp_reader
    def test_find_fc_key_missing_raises_key_error(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_nofc.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        with pytest.raises(KeyError, match='Force constants not found'):
            read_interpolation_data(dummy_h5)

    @pytest.mark.vasp_reader
    def test_find_fc_key_hessian(self, tmp_path):
        import h5py

        dummy_h5 = tmp_path / 'dummy_hessian.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('input/poscar')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

            lin_group = f.create_group('results/linear_response')
            lin_group.create_dataset('hessian', data=np.zeros((3, 3)))

        data = read_interpolation_data(dummy_h5)
        assert 'force_constants' in data

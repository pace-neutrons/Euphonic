import h5py
import numpy as np
import numpy.testing as npt
import pytest

from euphonic import ForceConstants, QpointFrequencies, QpointPhononModes
from euphonic.readers.vasp import (
    MissingPhononModesError,
    read_crystal,
    read_interpolation_data,
    read_phonon_data,
)
from tests_and_analysis.test.utils import get_data_path

VASPOUT_PATH = get_data_path('vasp_files', 'vaspout_sanitized.h5')
VASPOUT_DOS_PATH = get_data_path('vasp_files', 'vaspout_dos_sanitized.h5')
VASPOUT_DOS_RERUN_PATH = get_data_path(
    'vasp_files', 'vaspout_dos_rerun_sanitized.h5'
)


class TestVaspReaderCrystal:

    def test_read_crystal(self):
        crystal_data = read_crystal(VASPOUT_PATH)
        assert crystal_data['cell_vectors_unit'] == 'angstrom'
        assert crystal_data['cell_vectors'].shape == (3, 3)
        assert len(crystal_data['atom_r']) == 16
        assert len(crystal_data['atom_type']) == 16
        assert len(crystal_data['atom_mass']) == 16

        # Check Ga and As species counts
        types = list(crystal_data['atom_type'])
        assert types.count('Ga') == 8
        assert types.count('As') == 8

        # Check atomic masses from POTCAR (Ga ~69.723, As ~74.922)
        npt.assert_allclose(crystal_data['atom_mass'][:8], 69.723)
        npt.assert_allclose(crystal_data['atom_mass'][8:], 74.922)

    def test_read_crystal_from_incar_override(self, tmp_path):
        dummy_h5 = tmp_path / 'dummy_vaspout_incar.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('results/positions')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855\nISMEAR = 0')
            )

        data = read_crystal(dummy_h5)
        npt.assert_allclose(data['atom_mass'], 28.0855)

    def test_read_crystal_incar_overrides_potcar(self, tmp_path):
        dummy_h5 = tmp_path / 'dummy_vaspout_priority.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('results/positions')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset('content', data=np.bytes_(b'POMASS = 10.0'))

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset(
                'content', data=np.bytes_(b'POMASS = 28.0855')
            )

        data = read_crystal(dummy_h5)
        npt.assert_allclose(data['atom_mass'], 28.0855)

    def test_read_crystal_input_incar_overrides_original_incar(self, tmp_path):
        dummy_h5 = tmp_path / 'dummy_vaspout_input_incar.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('results/positions')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            potcar_group = f.create_group('input/potcar')
            potcar_group.create_dataset('content', data=np.bytes_(b'POMASS = 10.0'))

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset('content', data=np.bytes_(b'POMASS = 20.0'))

            input_incar = f.create_group('input/incar')
            input_incar.create_dataset('POMASS', data=np.bytes_(b'30.0'))

        data = read_crystal(dummy_h5)
        npt.assert_allclose(data['atom_mass'], 30.0)

    def test_read_crystal_negative_positions(self, tmp_path):
        dummy_h5 = tmp_path / 'dummy_vaspout_neg.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('results/positions')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.array([[-0.1, -0.5, -1.0], [1.1, -1e-15, 0.25]]))
            pos_group.create_dataset('number_ion_types', data=np.array([2]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

            incar_group = f.create_group('original/incar')
            incar_group.create_dataset('content', data=np.bytes_(b'POMASS = 28.0855'))

        data = read_crystal(dummy_h5)
        npt.assert_allclose(data['atom_r'][0], [0.9, 0.5, 0.0])
        npt.assert_allclose(data['atom_r'][1], [0.1, 0.0, 0.25])

    def test_read_crystal_missing_pomass_raises_error(self, tmp_path):
        dummy_h5 = tmp_path / 'dummy_vaspout_empty.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('results/positions')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))

        with pytest.raises(ValueError, match='Could not find atomic masses'):
            read_crystal(dummy_h5)


class TestVaspReaderPhononData:

    def test_read_phonon_data_missing_precalculated_raises_error(self):
        with pytest.raises(MissingPhononModesError):
            read_phonon_data(VASPOUT_PATH)

    def test_read_phonon_data_from_dos_vaspout(self):
        phonon_data = read_phonon_data(
            VASPOUT_DOS_PATH, frequencies_unit='meV'
        )
        assert 'crystal' in phonon_data
        assert phonon_data['crystal']['atom_r'].shape == (
            2,
            3,
        )  # Primitive cell
        assert phonon_data['qpts'].shape == (3, 3)
        assert phonon_data['frequencies'].shape == (3, 6)
        assert phonon_data['eigenvectors'].shape == (3, 6, 2, 3)

        # Check Gamma-point optical max frequency ~ 31.561 meV
        max_freq = np.max(phonon_data['frequencies'])
        npt.assert_allclose(max_freq, 31.561, rtol=1e-3)


class TestQpointPhononModesFromVasp:

    def test_from_vasp_modes_missing_data_raises_error(self):
        with pytest.raises(MissingPhononModesError):
            QpointPhononModes.from_vasp(VASPOUT_PATH)

    def test_from_vasp_modes_from_dos_file(self):
        modes = QpointPhononModes.from_vasp(VASPOUT_DOS_PATH)
        assert modes.crystal.n_atoms == 2  # Primitive GaAs cell
        assert modes.frequencies.shape == (3, 6)
        assert modes.eigenvectors.shape == (3, 6, 2, 3)

        max_freq = np.max(modes.frequencies.magnitude)
        npt.assert_allclose(max_freq, 31.561, rtol=1e-3)

    def test_from_vasp_frequencies_from_dos_file(self):
        freqs = QpointFrequencies.from_vasp(VASPOUT_DOS_PATH)
        assert freqs.crystal.n_atoms == 2
        assert freqs.frequencies.shape == (3, 6)

        max_freq = np.max(freqs.frequencies.magnitude)
        npt.assert_allclose(max_freq, 31.561, rtol=1e-3)


class TestForceConstantsFromVasp:

    def test_read_interpolation_data(self):
        data = read_interpolation_data(VASPOUT_PATH)
        assert 'crystal' in data
        assert data['force_constants'].shape == (1, 48, 48)
        assert data['sc_matrix'].shape == (3, 3)
        assert 'born' in data
        assert data['born'].shape == (16, 3, 3)
        assert 'dielectric' in data
        assert data['dielectric'].shape == (3, 3)

    def test_from_vasp_and_fallback_calculation(self):
        # 1. ForceConstants.from_vasp loads Hessian/force_constants
        fc = ForceConstants.from_vasp(VASPOUT_PATH)
        assert fc.crystal.n_atoms == 16
        assert fc.n_cells_in_sc == 1
        assert fc.force_constants.shape == (1, 48, 48)

        # 2. Outer caller calculates modes explicitly from ForceConstants
        q_freqs = fc.calculate_qpoint_frequencies(np.array([[0.0, 0.0, 0.0]]))

        # 3. Compare with QpointFrequencies loaded from precalculated QPOINTS interpolation file
        dos_freqs = QpointFrequencies.from_vasp(VASPOUT_DOS_PATH)

        fc_freqs_mev = q_freqs.frequencies.to('meV').magnitude[0]
        dos_gamma_freqs_mev = dos_freqs.frequencies.to('meV').magnitude[0]

        # Max optical frequency at Gamma (31.561 meV) must match
        npt.assert_allclose(
            np.max(fc_freqs_mev), np.max(dos_gamma_freqs_mev), rtol=1e-3
        )


class TestVaspReaderCombined:

    def test_combined_fc_and_modes(self):
        # 1. ForceConstants reads primitive cell force constants (8, 6, 6)
        fc = ForceConstants.from_vasp(VASPOUT_DOS_RERUN_PATH)
        assert fc.crystal.n_atoms == 2
        assert fc.n_cells_in_sc == 8
        assert fc.force_constants.shape == (8, 6, 6)
        assert fc.born is not None
        assert fc.born.shape == (2, 3, 3)

        # 2. QpointPhononModes reads primitive precalculated modes (3 qpts, 6 branches)
        modes = QpointPhononModes.from_vasp(VASPOUT_DOS_RERUN_PATH)
        assert modes.crystal.n_atoms == 2
        assert modes.frequencies.shape == (3, 6)
        assert modes.eigenvectors.shape == (3, 6, 2, 3)

        # 3. Compare calculated frequencies from primitive FC vs precalculated modes
        q_freqs = fc.calculate_qpoint_frequencies(modes.qpts)
        npt.assert_allclose(
            q_freqs.frequencies.to('meV').magnitude,
            modes.frequencies.to('meV').magnitude,
            rtol=1e-2,
            atol=1e-2,
        )

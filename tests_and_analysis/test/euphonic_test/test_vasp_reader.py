from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest

from euphonic import ForceConstants, QpointFrequencies, QpointPhononModes
from euphonic.readers.vasp import (
    read_crystal,
    read_interpolation_data,
    read_phonon_data,
)

from tests_and_analysis.test.utils import get_data_path

VASPOUT_PATH = get_data_path('vasp_files', 'vaspout_sanitized.h5')


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
        import h5py

        dummy_h5 = tmp_path / 'dummy_vaspout_incar.h5'
        with h5py.File(dummy_h5, 'w') as f:
            pos_group = f.create_group('results/positions')
            pos_group.create_dataset('lattice_vectors', data=np.eye(3))
            pos_group.create_dataset('position_ions', data=np.zeros((1, 3)))
            pos_group.create_dataset('number_ion_types', data=np.array([1]))
            pos_group.create_dataset('ion_types', data=np.array([b'Si']))
            
            incar_group = f.create_group('original/incar')
            incar_group.create_dataset('content', data=np.bytes_(b'POMASS = 28.0855\nISMEAR = 0'))

        data = read_crystal(dummy_h5)
        npt.assert_allclose(data['atom_mass'], 28.0855)

    def test_read_crystal_missing_pomass_raises_error(self, tmp_path):
        import h5py

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

    def test_read_phonon_data(self):
        phonon_data = read_phonon_data(VASPOUT_PATH, frequencies_unit='meV')
        assert 'crystal' in phonon_data
        assert phonon_data['qpts'].shape == (1, 3)
        assert phonon_data['frequencies'].shape == (1, 48)
        assert phonon_data['eigenvectors'].shape == (1, 48, 16, 3)

        # Check Gamma-point max frequency ~ 31.561 meV (7.6314 THz)
        max_freq = np.max(phonon_data['frequencies'])
        npt.assert_allclose(max_freq, 31.561, rtol=1e-3)


class TestQpointPhononModesFromVasp:

    def test_from_vasp_modes(self):
        modes = QpointPhononModes.from_vasp(VASPOUT_PATH)
        assert modes.crystal.n_atoms == 16
        assert modes.frequencies.shape == (1, 48)
        assert modes.eigenvectors.shape == (1, 48, 16, 3)

        max_freq = np.max(modes.frequencies.magnitude)
        npt.assert_allclose(max_freq, 31.561, rtol=1e-3)

    def test_from_vasp_frequencies(self):
        freqs = QpointFrequencies.from_vasp(VASPOUT_PATH)
        assert freqs.crystal.n_atoms == 16
        assert freqs.frequencies.shape == (1, 48)

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

    def test_from_vasp(self):
        fc = ForceConstants.from_vasp(VASPOUT_PATH)
        assert fc.crystal.n_atoms == 16
        assert fc.n_cells_in_sc == 1
        assert fc.force_constants.shape == (1, 48, 48)
        assert fc.born is not None
        assert fc.born.shape == (16, 3, 3)
        assert fc.dielectric is not None
        assert fc.dielectric.shape == (3, 3)

        # Cross-validation: calculate frequencies at Gamma and compare to QpointPhononModes
        q_freqs = fc.calculate_qpoint_frequencies(np.array([[0.0, 0.0, 0.0]]))
        modes = QpointPhononModes.from_vasp(VASPOUT_PATH)

        fc_freqs_meV = q_freqs.frequencies.to('meV').magnitude[0]
        modes_freqs_meV = modes.frequencies.to('meV').magnitude[0]

        # Both should match the sorted Gamma point frequencies
        npt.assert_allclose(np.sort(fc_freqs_meV), np.sort(modes_freqs_meV), rtol=1e-3)

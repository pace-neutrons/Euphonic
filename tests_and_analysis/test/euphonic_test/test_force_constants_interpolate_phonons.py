import pytest
import numpy as np
import numpy.testing as npt
import os
from tests_and_analysis.test.utils import get_data_path
from euphonic import ForceConstants


@pytest.mark.unit
class TestInterpolatePhonons:

    path = os.path.join(get_data_path(), 'interpolation')
    unique_sc_offsets = [[] for _ in range(3)]
    unique_cell_origins = [[] for _ in range(3)]

    graphite_unique_sc_i = np.loadtxt(os.path.join(
        path, 'graphite', 'graphite_unique_sc_i.txt'), dtype=np.int32)
    graphite_unique_cell_i = np.loadtxt(os.path.join(
        path, 'graphite', 'graphite_unique_cell_i.txt'), dtype=np.int32)
    lzo_unique_sc_i = np.loadtxt(os.path.join(
        path, 'LZO', 'lzo_unique_sc_i.txt'), dtype=np.int32)
    lzo_unique_cell_i = np.loadtxt(os.path.join(
        path, 'LZO', 'lzo_unique_cell_i.txt'), dtype=np.int32)

    @pytest.fixture(params=[
        (
            'LZO', 'La2Zr2O7.castep_bin',
            np.array([-1, 9.35, 3.35]),
            np.loadtxt(
                os.path.join(path, 'LZO', 'lzo_sc_phases.txt'),
                dtype=np.complex128
            ),
            np.loadtxt(
                os.path.join(path, 'LZO', 'lzo_cell_phases.txt'),
                dtype=np.complex128
            )
        ),
        (
            'LZO', 'La2Zr2O7.castep_bin',
            np.array([0.0, 0.0, 0.0]),  # At gamma point
            np.full(
                len(lzo_unique_sc_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            ),
            np.full(
                len(lzo_unique_cell_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            )
        ),
        (
            'graphite', 'graphite.castep_bin',
            np.array([0.001949, 0.001949, 0.0]),
            np.loadtxt(os.path.join(
                path, 'graphite', 'graphite_sc_phases.txt'
            ), dtype=np.complex128),
            np.loadtxt(os.path.join(
                path, 'graphite', 'graphite_cell_phases.txt'
            ), dtype=np.complex128)
        ),
        (
            'graphite', 'graphite.castep_bin',
            np.array([0.0, 0.0, 0.0]),  # At gamma point
            np.full(
                len(graphite_unique_sc_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            ),
            np.full(
                len(graphite_unique_cell_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            )
        )
    ])
    def calculate_phases(self, request):
        material_name, castep_bin_file, qpt, expected_sc_phases, \
            expected_cell_phases = request.param
        lower_material = material_name.lower()
        filename = os.path.join(self.path, material_name, castep_bin_file)
        self.unique_sc_i = np.loadtxt(
            os.path.join(
                self.path, material_name,
                '{}_unique_sc_i.txt'.format(lower_material)
            ), dtype=np.int32
        )
        self.unique_cell_i = np.loadtxt(
            os.path.join(
                self.path, material_name,
                '{}_unique_cell_i.txt'.format(lower_material)
            ), dtype=np.int32
        )
        unique_offsets_filepath = os.path.join(
            self.path, material_name,
            '{}_unique_sc_offsets.txt'.format(lower_material)
        )
        with open(unique_offsets_filepath) as f:
            for i in range(3):
                self.unique_sc_offsets[i] = [int(x)
                                             for x in f.readline().split()]
        unique_cell_origins_filepath = os.path.join(
            self.path, material_name,
            '{}_unique_cell_origins.txt'.format(lower_material)
        )
        with open(unique_cell_origins_filepath) as f:
            for i in range(3):
                self.unique_cell_origins[i] = [int(x)
                                               for x in f.readline().split()]
        return ForceConstants.from_castep(filename), qpt, \
            expected_sc_phases, expected_cell_phases

    def test_calculate_phases(self, calculate_phases):
        fc, qpt, expected_sc_phases, expected_cell_phases = calculate_phases
        sc_phases, cell_phases = fc._calculate_phases(
            qpt, self.unique_sc_offsets, self.unique_sc_i,
            self.unique_cell_origins, self.unique_cell_i
        )
        npt.assert_allclose(sc_phases, expected_sc_phases)
        npt.assert_allclose(cell_phases, expected_cell_phases)

    # Supercell image calculation limit - 2 supercells in each direction
    lim = 2
    n_atoms = 4
    n_cells_in_sc = 98

    def get_ijn_from_image_data(self, image_data):
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        return i, j, n

    @pytest.fixture
    def lzo_calculate_supercell_images_n_sc_images(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'LZO', 'lzo_n_sc_images.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        expected_n_images = np.zeros((22, 88))
        expected_n_images[i, j] = n
        # After refactoring where the shape of n_sc_images was changed from
        # (n_atoms, n_cells_in_sc*n_atoms) to (n_cells_in_sc, n_atoms, n_atoms),
        # expected_n_images must be reshaped to ensure tests still pass
        expected_n_images = np.transpose(np.reshape(
            expected_n_images, (22, 4, 22)), axes=[1, 0, 2]
        )
        filename = os.path.join(self.path, 'LZO', 'La2Zr2O7.castep_bin')
        return ForceConstants.from_castep(filename), "_n_sc_images", \
            expected_n_images

    @pytest.fixture
    def graphite_calculate_supercell_images_n_sc_images(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'graphite', 'graphite_n_sc_images.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        expected_n_images = np.zeros(
            (self.n_atoms, self.n_atoms * self.n_cells_in_sc))
        expected_n_images[i, j] = n
        # After refactoring where the shape of n_sc_images was changed from
        # (n_atoms, n_cells_in_sc*n_atoms) to (n_cells_in_sc, n_atoms, n_atoms),
        # expctc_n_images must be reshaped to ensure tests still pass
        expected_n_images = np.transpose(
            np.reshape(
                expected_n_images,
                (self.n_atoms, self.n_cells_in_sc, self.n_atoms)
            ), axes=[1, 0, 2]
        )
        filename = os.path.join(self.path, 'graphite', 'graphite.castep_bin')
        return ForceConstants.from_castep(filename), "_n_sc_images", \
            expected_n_images

    @pytest.fixture
    def lzo_calculate_supercell_images_sc_image_i(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'LZO', 'lzo_sc_image_i.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        sc_i = image_data[:, 3].astype(int)
        max_n = np.max(n) + 1
        # size = n_atoms X n_atoms*n_cells_in_sc X max supercell images
        expected_sc_image_i = np.full((22, 88, max_n), -1)
        expected_sc_image_i[i, j, n] = sc_i
        # After refactoring where the shape of sc_image_i was changed from
        # (n_atoms, n_cells_in_sc*n_atoms, (2*lim + 1)**3) to
        # (n_cells_in_sc, n_atoms, n_atoms, (2*lim + 1)**3),
        # expctd_image_i must be reshaped to ensure tests still pass
        expected_sc_image_i = np.transpose(
            np.reshape(expected_sc_image_i, (22, 4, 22, max_n)),
            axes=[1, 0, 2, 3]
        )
        filename = os.path.join(self.path, 'LZO', 'La2Zr2O7.castep_bin')
        return ForceConstants.from_castep(filename), "_sc_image_i", \
            expected_sc_image_i

    @pytest.fixture
    def graphite_calculate_supercell_images_sc_image_i(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'graphite', 'graphite_sc_image_i.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        sc_i = image_data[:, 3].astype(int)
        max_n = np.max(n) + 1
        # size = n_atoms X n_atoms*n_cells_in_sc X max supercell images
        expected_sc_image_i = np.full(
            (self.n_atoms, self.n_atoms * self.n_cells_in_sc, max_n), -1)
        expected_sc_image_i[i, j, n] = sc_i
        # After refactoring where the shape of sc_image_i was changed from
        # (n_atoms, n_cells_in_sc*n_atoms, (2*lim + 1)**3) to
        # (n_cells_in_sc, n_atoms, n_atoms, (2*lim + 1)**3),
        # expctc_image_i must be reshaped to ensure tests still pass
        expected_sc_image_i = np.transpose(
            np.reshape(
                expected_sc_image_i,
                (self.n_atoms, self.n_cells_in_sc,self.n_atoms, max_n)
            ), axes=[1, 0, 2, 3]
        )
        filename = os.path.join(self.path, 'graphite', 'graphite.castep_bin')
        return ForceConstants.from_castep(filename), "_sc_image_i", \
            expected_sc_image_i

    @pytest.mark.parametrize('calculate_supercell_images', [
        pytest.lazy_fixture('lzo_calculate_supercell_images_n_sc_images'),
        pytest.lazy_fixture('lzo_calculate_supercell_images_sc_image_i'),
        pytest.lazy_fixture('graphite_calculate_supercell_images_n_sc_images'),
        pytest.lazy_fixture('graphite_calculate_supercell_images_sc_image_i')
    ])
    def test_calculate_supercell_images(self, calculate_supercell_images):
        fc, testing_attribute, expected_attribute_data = \
            calculate_supercell_images
        fc._calculate_supercell_images(self.lim)
        npt.assert_equal(
            getattr(fc, testing_attribute), expected_attribute_data
        )

import numpy as np
import numpy.testing as npt
import pytest

from euphonic import ureg, Crystal
from euphonic.powder import (_get_qpts_sphere, _qpts_cart_to_frac,
                             sample_sphere_dos, sample_sphere_structure_factor)

sampling_functions = {
    'golden': 'euphonic.sampling.golden_sphere',
    'sphere-projected-grid': 'euphonic.sampling.sphere_from_square_grid',
    'spherical-polar-grid': 'euphonic.sampling.spherical_polar_grid',
    'spherical-polar-improved': 'euphonic.sampling.spherical_polar_improved',
    'random-sphere': 'euphonic.sampling.random_sphere'}


@pytest.fixture
def random_qpts_array():
    return np.random.random((4, 3))


@pytest.fixture(params=[True, False])
def jitter(request):
    return request.param


@pytest.mark.unit
@pytest.mark.parametrize('npts, sampling, sampling_args, sampling_kwargs',
                         [(10, 'golden', (10,), {'jitter': None}),
                          # 31 pts rounded up to 2N^2 -> 4 cols, 8 rows
                          (31, 'sphere-projected-grid',
                           (8, 4), {'jitter': None}),
                          (31, 'spherical-polar-grid',
                           (8, 4), {'jitter': None}),
                          (13, 'spherical-polar-improved',
                           (13,), {'jitter': None}),
                          (7, 'random-sphere', (7,), dict())]
                         )
def test_get_qpts_sphere(mocker, random_qpts_array, jitter,
                         npts, sampling, sampling_args, sampling_kwargs):
    """Check that functions from euphonic.sampling are called as expected"""
    mocked_sampling_function = mocker.patch(sampling_functions[sampling],
                                            return_value=random_qpts_array,
                                            autospec=True)

    if 'jitter' in sampling_kwargs:
        sampling_kwargs['jitter'] = jitter

    npt.assert_almost_equal(
        _get_qpts_sphere(npts, sampling=sampling, jitter=jitter),
        random_qpts_array)

    mocked_sampling_function.assert_called_with(*sampling_args,
                                                **sampling_kwargs)


class TestSphereSampledProperties:
    @staticmethod
    def mock_get_qpts_sphere(mocker, return_qpts):
        return mocker.patch('euphonic.powder._get_qpts_sphere',
                            return_value=return_qpts)

    @staticmethod
    def mock_get_default_bins(mocker, return_bins):
        return mocker.patch('euphonic.powder._get_default_bins',
                            return_value=return_bins)

    @pytest.fixture
    def mock_s(self, mocker):
        s = mocker.MagicMock()
        s.configure_mock(**{'calculate_1d_average.return_value':
                            'calculate_1d_average_return_value'})
        return s

    @pytest.fixture
    def mock_qpm(self, mocker, mock_s):
        qpm = mocker.MagicMock()
        qpm.configure_mock(
            **{'calculate_dos.return_value': 'calculate_dos_return_value',
               'calculate_structure_factor.return_value': mock_s})
        return qpm

    @pytest.fixture
    def mock_crystal(self, mocker):
        crystal = mocker.MagicMock()
        crystal.configure_mock(
            **{'reciprocal_cell.return_value': np.array([[1, 0, 0],
                                                         [0, 1, 0],
                                                         [0, 0, 1]])
               * ureg('1 / angstrom')})
        return crystal

    @pytest.fixture
    def mock_fc(self, mocker, mock_qpm, mock_crystal):
        fc = mocker.MagicMock()
        fc.configure_mock(
            **{'calculate_qpoint_phonon_modes.return_value': mock_qpm,
               'crystal': mock_crystal})
        return fc

    @pytest.mark.unit
    def test_sample_sphere_dos(self, mocker, mock_fc, mock_qpm,
                               random_qpts_array):
        mod_q = 1.2 * ureg('1 / angstrom')
        return_bins = np.linspace(1., 10., 5)

        # Dummy out functions called by sample_sphere_dos and tested elsewhere
        self.mock_get_default_bins(mocker, return_bins)
        self.mock_get_qpts_sphere(mocker, random_qpts_array)

        assert (sample_sphere_dos(mock_fc, mod_q)
                == 'calculate_dos_return_value')
        npt.assert_almost_equal(
            random_qpts_array * mod_q.magnitude,
            mock_fc.calculate_qpoint_phonon_modes.call_args[0][0])
        mock_qpm.calculate_dos.assert_called_with(return_bins)

    @pytest.mark.unit
    def test_sample_sphere_structure_factor(self, mocker, mock_fc, mock_qpm,
                                            random_qpts_array):
        mod_q = 1.2 * ureg('1 / angstrom')
        return_bins = np.linspace(1., 10., 5)

        # Dummy out functions called by sample_sphere_structure_factor
        # that are tested elsewhere
        self.mock_get_default_bins(mocker, return_bins)

        assert (sample_sphere_structure_factor(mock_fc, mod_q)
                == 'calculate_1d_average_return_value')


class TestQpointConversion:
    @pytest.fixture
    def trivial_crystal(self):
        return Crystal((np.array([[1., 0., 0.], [0., 2, 0.], [0., 0., 3]])
                        * ureg('angstrom')),
                       np.array([[0., 0., 0.]]),
                       np.array(['Si']), np.array([28.055]) * ureg('amu'))

    @pytest.fixture(params=[
        [[1, 1, 0], [0, -1, 0], [0, 0, 1]],
        [[0.1, 0.2, 0.3], [0.3, -0.2, 0.1], [0.2, 0., 0.6]],
        [[0., 0., 0.5], [1, 1, 0.1], [-1, 2., 0.0]]])
    def nontrivial_crystal(self, request):
        """Some arbitrary non-diagonal lattices"""
        return Crystal(np.asarray(request.param, dtype=float
                                  ) * ureg('angstrom'),
                       np.array([[0., 0., 0.]]),
                       np.array(['Si']), np.array([28.055]) * ureg('amu'))

    @pytest.fixture
    def trivial_qpts(self):
        return {'frac': [[1, 0, 0], [0, 1, 0],
                         [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5]],
                'cart': [[2 * np.pi, 0, 0], [0, np.pi, 0],
                         [0, 0, 2 * np.pi / 3], [np.pi, np.pi / 2, 0],
                         [0, np.pi / 2, np.pi / 3]]}

    @staticmethod
    @pytest.mark.unit
    def test_qpts_cart_to_frac_trivial(trivial_crystal, trivial_qpts):
        """Check internal method for q-point conversion with trivial example"""
        cart_qpts = np.asarray(trivial_qpts['cart'], dtype=float
                               ) * ureg('1 / angstrom')
        frac_qpts = np.asarray(trivial_qpts['frac'], dtype=float)
        calc_frac_qpts = _qpts_cart_to_frac(cart_qpts, trivial_crystal)

        npt.assert_almost_equal(calc_frac_qpts, frac_qpts)

    @staticmethod
    @pytest.mark.unit
    def test_qpts_cart_to_frac_roundtrip(random_qpts_array,
                                         nontrivial_crystal):
        frac_qpts = random_qpts_array
        cart_qpts = frac_qpts.dot(nontrivial_crystal.reciprocal_cell()
                                  .to('1 / angstrom').magnitude
                                  ) * ureg('1 / angstrom')
        calc_frac_qpts = _qpts_cart_to_frac(cart_qpts, nontrivial_crystal)

        npt.assert_almost_equal(calc_frac_qpts, frac_qpts)

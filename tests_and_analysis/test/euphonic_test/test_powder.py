import numpy as np
import numpy.testing as npt
import pytest

from euphonic.powder import _get_qpts_sphere

sampling_functions = {
    'golden': 'euphonic.sampling.golden_sphere',
    'sphere-projected-grid': 'euphonic.sampling.sphere_from_square_grid',
    'spherical-polar-grid': 'euphonic.sampling.spherical_polar_grid',
    'spherical-polar-improved': 'euphonic.sampling.spherical_polar_improved',
    'random-sphere': 'euphonic.sampling.random_sphere'}


@pytest.fixture
def random_qpts_list():
    return [tuple(row) for row in np.random.random((4, 3))]


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
def test_get_qpts_sphere(mocker, random_qpts_list, jitter,
                         npts, sampling, sampling_args, sampling_kwargs):
    """Check that functions from euphonic.sampling are called as expected"""
    mocked_sampling_function = mocker.patch(sampling_functions[sampling],
                                            return_value=random_qpts_list,
                                            autospec=True)

    if 'jitter' in sampling_kwargs:
        sampling_kwargs['jitter'] = jitter

    npt.assert_almost_equal(
        _get_qpts_sphere(npts, sampling=sampling, jitter=jitter),
        random_qpts_list)

    mocked_sampling_function.assert_called_with(*sampling_args,
                                                **sampling_kwargs)

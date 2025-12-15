from collections.abc import Iterator

import numpy as np
import pytest

import euphonic.sampling

# @pytest.fixture
# def fix_np_rng(monkeypatch):
#         monkeypatch.setattr(np.random, 'default_rng',
#                             partial(np.random.RandomState, seed=0))

def rng_seed_0():
    """Get a fresh random generator with fixed seed"""
    return np.random.RandomState(seed=0)


def check_sampling(func, params, ref_results):
    points_iter = func(**params)
    assert isinstance(points_iter, Iterator)
    assert np.allclose(list(points_iter), ref_results)


class TestSquareSampling:
    gold_square_ref_data = [({'npts': 3, 'offset': False, 'jitter': False},
                             [[0, 0],
                              [0.3333333333333333, 0.61803398874989480],
                              [0.6666666666666666, 0.23606797749978958]]),
                            ({'npts': 3, 'offset': True, 'jitter': False},
                             [[0.1666666666666666, 0],
                              [0.5, 0.61803398874989480],
                              [0.8333333333333333, 0.23606797749978958]]),
                            ({'npts': 3, 'offset': False, 'jitter': True, 'rng': rng_seed_0()},
                             [[0.0281824896325298, 0.12423963860186140],
                              [0.3926637961711317, 0.64394730653524050],
                              [0.6225887445136473, 0.32029998295200524]]),
                            ]

    @pytest.mark.parametrize('params, ref_results', gold_square_ref_data)
    def test_golden_square(self, params, ref_results):
        check_sampling(euphonic.sampling.golden_square,
                       params, ref_results)

    reg_square_ref_data = [({'n_rows': 2, 'n_cols': 4,
                             'offset': False, 'jitter': False},
                            [[0., 0.], [0., 0.5],
                             [0.25, 0.], [0.25, 0.5],
                             [0.5, 0.], [0.5, 0.5],
                             [0.75, 0.], [0.75, 0.5]]),
                           ({'n_rows': 2, 'n_cols': 4,
                             'offset': True, 'jitter': False},
                            [[0.125, 0.25], [0.125, 0.75],
                             [0.375, 0.25], [0.375, 0.75],
                             [0.625, 0.25], [0.625, 0.75],
                             [0.875, 0.25], [0.875, 0.75]]),
                           ({'n_rows': 1, 'n_cols': 2,
                             'offset': False, 'jitter': True, 'rng': rng_seed_0()},
                            [[0.024406751963662376, 0.21518936637241948],
                             [0.551381688035822, 0.044883182996896864]]),
                           ]

    @pytest.mark.parametrize('params, ref_results', reg_square_ref_data)
    def test_regular_square(self, params, ref_results):
        check_sampling(euphonic.sampling.regular_square, params, ref_results)


class TestSphereSampling:
    gold_sphere_ref = [({'npts': 4,
                         'cartesian': False, 'jitter': False},
                        [[1, 0.0, 2.4188584057763776],
                         [1, 3.8832220774509327, 1.8234765819369754],
                         [1, 1.483258847722279, 1.318116071652818],
                         [1, 5.366480925173213, 0.7227342478134157]]),
                       ({'npts': 4,
                         'cartesian': True, 'jitter': False},
                        [[0.6614378277661477, 0.0, -0.75],
                         [-0.7139543462022453, -0.6540406650499068, -0.25],
                         [0.08464959396472624, 0.9645384628108965, 0.25],
                         [0.4024444785343673, -0.5249175570479627, 0.75]])]

    @pytest.mark.parametrize('params, ref_results', gold_sphere_ref)
    def test_golden_sphere(self, params, ref_results):
        check_sampling(euphonic.sampling.golden_sphere,
                       params, ref_results)

    square_sphere_ref = [({'n_rows': 1, 'n_cols': 3,
                           'cartesian': False, 'jitter': False},
                          [[1.        , 3.14159265, 2.30052398],
                           [1.        , 3.14159265, 1.57079633],
                           [1.        , 3.14159265, 0.84106867]]),
                         ({'n_rows': 3, 'n_cols': 1,
                           'cartesian': True, 'jitter': False},
                          [(0.5, 0.8660254037844386, 0.),
                           (-1., 0., 0.),
                           (0.5, -0.866025403784439, 0.)])]

    @pytest.mark.parametrize('params, ref_results', square_sphere_ref)
    def test_sphere_from_square_grid(self, params, ref_results):
        check_sampling(euphonic.sampling.sphere_from_square_grid,
                       params, ref_results)

    polar_grid_ref = [({'n_phi': 2, 'n_theta': 2,
                           'cartesian': False, 'jitter': False},
                       [[1, -3.141592653589793, 0.7853981633974483],
                        [1, -3.141592653589793, 2.356194490192345],
                        [1, 0.0, 0.7853981633974483],
                        [1, 0.0, 2.356194490192345]]),
                         ({'n_phi': 2, 'n_theta': 2,
                           'cartesian': True, 'jitter': False},
                          [[-0.7071067811865475, -8.659560562354932e-17,
                            0.7071067811865476],
                           [-0.7071067811865476, -8.659560562354934e-17,
                            -0.7071067811865475],
                           [0.7071067811865475, 0.0, 0.7071067811865476],
                           [0.7071067811865476, 0.0, -0.7071067811865475]]),
                         ({'n_phi': 2, 'n_theta': 2,
                           'cartesian': True, 'jitter': True, 'rng': rng_seed_0()},
                          [[-0.8910033774558734, -0.1377185459691061, 0.4326044191387571],
                           [-0.6216723147673089, -0.2079773711885614, -0.7551615364445884],
                           [0.8249424691200341, -0.20174213857267537, 0.5279867727190379],
                           [0.16594308161020008, -0.03296086943464723, -0.9855843316286145]])]

    @pytest.mark.parametrize('params, ref_results', polar_grid_ref)
    def test_spherical_polar_grid(self, params, ref_results):
        check_sampling(
            euphonic.sampling.spherical_polar_grid, params, ref_results)

    polar_improved_ref = [({'npts': 6, 'cartesian': False, 'jitter': False},
                           [[1, -3.141592653589793, 0.7853981633974483],
                            [1, -1.0471975511965979, 0.7853981633974483],
                            [1, 1.0471975511965974, 0.7853981633974483],
                            [1, -3.141592653589793, 2.356194490192345],
                            [1, -1.0471975511965979, 2.356194490192345],
                            [1, 1.0471975511965974, 2.356194490192345]]),
                          ({'npts': 6, 'cartesian': True, 'jitter': False},
                           [[-0.7071067811865475, 0., 0.7071067811865476],
                            [0.3535533905932737, -0.6123724356957945,
                             0.7071067811865476],
                            [0.35355339059327395, 0.6123724356957944,
                             0.7071067811865476],
                            [-0.7071067811865476, 0., -0.7071067811865475],
                            [0.35355339059327373, -0.6123724356957946,
                             -0.7071067811865475],
                            [0.353553390593274, 0.6123724356957945,
                             -0.7071067811865475]]),
                          ({'npts': 6, 'cartesian': True, 'jitter': True, 'rng': rng_seed_0()},
                           [(-0.8968762869872088, -0.09201272945444275, 0.4326044191387571),
                            (0.5085411851719293, -0.5582605208227438, 0.6555387508566128),
                            (0.5363093035525994, 0.6584848508178914, 0.5279867727190379),
                            (-0.1677415154464929, 0.022052420363564928, -0.9855843316286145),
                            (0.8216447250550393, -0.06265189549804429, -0.5665462785860754),
                            (-0.058850836096272056, 0.6717223047726856, -0.7384617284339393)])]

    @pytest.mark.parametrize('params, ref_results', polar_improved_ref)
    def test_spherical_polar_improved(self, params, ref_results):
        check_sampling(euphonic.sampling.spherical_polar_improved,
                       params, ref_results)

    def test_spherical_polar_improved_npts_error(self):
        with pytest.raises(ValueError):
            next(euphonic.sampling.spherical_polar_improved(5))

    random_sphere_ref = [({'npts': 2, 'cartesian': False, 'rng': rng_seed_0()},
                          [[1, 4.493667318642264, 1.4730135689716053],
                           [1, 3.4236020095353483, 1.3637944071036738]]),
                         ({'npts': 2, 'cartesian': True, 'rng': rng_seed_0()},
                          [[-0.2159454115343286, -0.9715125045899397,
                            0.0976270078546495],
                           [-0.9399930037152809, -0.2723451984518093,
                            0.20552675214328772]])]

    @pytest.mark.parametrize('params, ref_results', random_sphere_ref)
    def test_random_sphere(self, params, ref_results):
        check_sampling(euphonic.sampling.random_sphere, params, ref_results)

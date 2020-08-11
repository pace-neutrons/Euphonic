import pytest
import numpy as np
from typing import Iterator

import euphonic.sampling


class TestSquareSampling:
    gold_square_ref_data = [({'npts': 3, 'offset': False, 'jitter': False},
                             [[0, 0],
                              [0.3333333333333333, 0.61803398874989480],
                              [0.6666666666666666, 0.23606797749978958]]),
                            ({'npts': 3, 'offset': True, 'jitter': False},
                             [[0.1666666666666666, 0],
                              [0.5, 0.61803398874989480],
                              [0.8333333333333333, 0.23606797749978958]]),
                            ({'npts': 3, 'offset': False, 'jitter': True},
                             [[0.0281824896325298, 0.12423963860186140],
                              [0.3926637961711317, 0.64394730653524050],
                              [0.6225887445136473, 0.32029998295200524]])
                            ]

    @pytest.mark.parametrize('params, ref_results', gold_square_ref_data)
    def test_golden_square(self, params, ref_results):
        np.random.seed(0)
        points_iter = euphonic.sampling.golden_square(**params)
        assert isinstance(points_iter, Iterator)
        assert np.allclose(list(points_iter), ref_results)

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
                             'offset': False, 'jitter': True},
                            [[0.024406751963662376, 0.21518936637241948],
                             [0.551381688035822, 0.044883182996896864]])
                           ]

    @pytest.mark.parametrize('params, ref_results', reg_square_ref_data)
    def test_regular_square(self, params, ref_results):
        np.random.seed(0)
        points_iter = euphonic.sampling.regular_square(**params)
        assert isinstance(points_iter, Iterator)
        assert np.allclose(list(points_iter), ref_results)

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
        points_iter = euphonic.sampling.golden_sphere(**params)
        assert isinstance(points_iter, Iterator)
        assert np.allclose(list(points_iter), ref_results)

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
        points_iter = euphonic.sampling.sphere_from_square_grid(**params)
        assert isinstance(points_iter, Iterator)
        assert np.allclose(list(points_iter), ref_results)

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

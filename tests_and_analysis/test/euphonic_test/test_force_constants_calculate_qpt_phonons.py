import pytest
import numpy as np
import numpy.testing as npt
from euphonic import ureg, ForceConstants
import os
from tests_and_analysis.test.utils import get_data_path
import json


class ExpectedFrequencies:

    def __init__(self, expected_freqs_json_file: str):
        self.data = json.load(open(expected_freqs_json_file))

    def get_expected_freqs(self, material: str, key: str):
        return np.array(self.data[material][key]) * \
            ureg(self.data[material][key + "_unit"])


# Load lzo and graphite materials test data
lzo_and_graphite_atol = 1e-10

lzo_and_graphite_materials = [
    {
        "material": "LZO",
        "json_file": os.path.join(
            "force_constants", "LZO", 'lzo_force_constants.json'
        ),
        "qpts": np.array([
            [-1.00, 9.35, 3.35],
            [-1.00, 9.00, 3.00]
        ]),
        "atol": lzo_and_graphite_atol
    },
    {
        "material": "graphite",
        "json_file": os.path.join(
            "force_constants", "graphite", "graphite_force_constants.json"
        ),
        "qpts": np.array([
            [0.00, 0.00, 0.00],
            [0.001949, 0.001949, 0.00],
            [0.50, 0.00, 0.00],
            [0.25, 0.00, 0.00],
            [0.00, 0.00, 0.50]
        ]),
        "atol": lzo_and_graphite_atol
    }
]

lzo_and_graphite_kwargs = [
    {
        "kwargs": {
            "use_c": False, "fall_back_on_python": True, "n_threads": 1
        },
        "key": "no_asr"
    },
    {
        "kwargs": {
            "use_c": True, "fall_back_on_python": False, "n_threads": 1
        },
        "key": "no_asr"
    },
    {
        "kwargs": {
            "use_c": True, "fall_back_on_python": False, "n_threads": 2
        },
        "key": "no_asr"
    },
    {
        "kwargs": {
            "use_c": False, "fall_back_on_python": True,
            "n_threads": 1, "asr": "realspace"
        },
        "key": "realspace"
    },
    {
        "kwargs": {
            "use_c": True, "fall_back_on_python": False,
            "n_threads": 1, "asr": "realspace"
        },
        "key": "realspace"
    },
    {
        "kwargs": {
            "use_c": True, "fall_back_on_python": False,
            "n_threads": 2, "asr": "realspace"
        },
        "key": "realspace"
    }
]

lzo_and_graphite_test_data = [
    {**material_json_qpts_atol, **kwargs_key}
    for material_json_qpts_atol in lzo_and_graphite_materials
    for kwargs_key in lzo_and_graphite_kwargs
]

# Load quartz material test data
quartz_qpts = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.50],
    [-0.25, 0.50, 0.50],
    [-0.151515, 0.575758, 0.5]
])
quartz_split_qpts = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.50],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [-0.25, 0.50, 0.50],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [-0.151515, 0.575758, 0.5],
    [0.00, 0.00, 0.00]
])
quartz_split_qpts_insert_gamma = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.50],
    [0.00, 0.00, 0.00],
    [-0.25, 0.50, 0.50],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [-0.151515, 0.575758, 0.5],
    [0.00, 0.00, 0.00]
])

quartz_kwargs_qpts_atol = [
    {
        "kwargs": {"dipole": True, "splitting": False},
        "qpts": quartz_qpts,
        "atol": 2e-6,
        "key": "no_asr"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False,
            "use_c": True, "fall_back_on_python": False
        },
        "qpts": quartz_qpts,
        "atol": 2e-6,
        "key": "no_asr"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False, "use_c": True,
            "n_threads": 2, "fall_back_on_python": False
        },
        "qpts": quartz_qpts,
        "atol": 8e-8,
        "key": "no_asr"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True, "splitting": False
        },
        "qpts": quartz_qpts,
        "atol": 5e-4,
        "key": "reciprocal"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True,
            "splitting": False, "use_c": True, "fall_back_on_python": False
        },
        "qpts": quartz_qpts,
        "atol": 5e-4,
        "key": "reciprocal"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True,
            "splitting": False, "use_c": True, "n_threads": 2,
            "fall_back_on_python": False
        },
        "qpts": quartz_qpts,
        "atol": 5e-4,
        "key": "reciprocal"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True, "splitting": True
        },
        "qpts": quartz_split_qpts,
        "atol": 5e-4,
        "key": "reciprocal_splitting"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True,
            "splitting": True, "use_c": True, "fall_back_on_python": False
        },
        "qpts": quartz_split_qpts,
        "atol": 5e-4,
        "key": "reciprocal_splitting"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True,
            "splitting": True, "use_c": True, "n_threads": 2,
            "fall_back_on_python": False
        },
        "qpts": quartz_split_qpts,
        "atol": 5e-4,
        "key": "reciprocal_splitting"
    },
    {
        "kwargs": {
            "asr": 'reciprocal', "dipole": True,
            "splitting": True, "insert_gamma": True
        },
        "qpts": quartz_split_qpts_insert_gamma,
        "atol": 5e-4,
        "key": "reciprocal_splitting"
    }
]

quartz_test_data = [
    {
        "material": "quartz",
        "json_file": os.path.join(
            "force_constants", "quartz", "quartz_force_constants.json"
        ),
        **data_dict
    } for data_dict in quartz_kwargs_qpts_atol
]

# Create test data for NaCl
nacl_kwargs_key = [
    {"kwargs": {"dipole": True, "splitting": False}, "key": "no_asr"},
    {
        "kwargs": {
            "dipole": True, "splitting": False,
            "use_c": True, "fall_back_on_python": False
        }, "key": "no_asr"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False,
            "use_c": True, "fall_back_on_python": False, "n_threads": 2
        }, "key": "no_asr"
    },
    {
        "kwargs": {"dipole": True, "splitting": False, "asr": 'reciprocal'},
        "key": "reciprocal"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False, "asr": 'reciprocal',
            "use_c": True, "fall_back_on_python": False
        }, "key": "reciprocal"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False, "asr": 'reciprocal',
            "use_c": True, "fall_back_on_python": False, "n_threads": 2
        }, "key": "reciprocal"
    },
    {
        "kwargs":  {"dipole": True, "splitting": False, "asr": 'realspace'},
        "key": "realspace"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False, "asr": 'realspace',
            "use_c": True, "fall_back_on_python": False
        }, "key": "realspace"
    },
    {
        "kwargs": {
            "dipole": True, "splitting": False, "asr": 'realspace',
            "use_c": True, "fall_back_on_python": False, "n_threads": 2
        }, "key": "realspace"
    }
]

nacl_test_data = [
    {
        "material": "NaCl",
        "json_file": os.path.join(
            "phonopy_data", "NaCl", "force_constants", "phonopy-yaml.json"
        ),
        "atol": 1e-8,
        "qpts": np.array([
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.50],
            [-0.25, 0.50, 0.50],
            [-0.151515, 0.575758, 0.5]
        ]),
        **data_dict
    } for data_dict in nacl_kwargs_key
]


@pytest.mark.integration
class TestCalculateQPointPhononModes:

    expected_freqs = ExpectedFrequencies(
        os.path.join(get_data_path(), "force_constants", "expected_freqs.json")
    )

    @pytest.fixture(
        params=lzo_and_graphite_test_data + quartz_test_data + nacl_test_data
    )
    def create_fc_and_formulate_args(self, request):
        test_data = request.param
        kwargs = test_data["kwargs"]
        kwargs["qpts"] = test_data["qpts"]
        filename = os.path.join(get_data_path(), test_data["json_file"])
        fc = ForceConstants.from_json_file(filename)
        return fc, kwargs, test_data["material"], \
            test_data["key"], test_data["atol"]

    def test_fc_calculate_qpoint_phonon_modes_expected_results(
            self, create_fc_and_formulate_args):
        fc, kwargs, material, key, atol = create_fc_and_formulate_args
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(**kwargs)
        test_expected_freqs = self.expected_freqs.\
            get_expected_freqs(material, key)
        npt.assert_allclose(
            qpoint_phonon_modes.frequencies.to('hartree').magnitude,
            test_expected_freqs.to('hartree').magnitude,
            atol=atol
        )

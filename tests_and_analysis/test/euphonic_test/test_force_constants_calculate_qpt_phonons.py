import pytest
import numpy as np
import numpy.testing as npt
from euphonic import ureg, ForceConstants
import os
from tests_and_analysis.test.utils import get_data_path
import itertools
import json


class ExpectedFrequencies:

    def __init__(self, expected_freqs_json_file: str):
        self.data = json.load(open(expected_freqs_json_file))

    def get_expected_freqs(self, material: str, key: str):
        return np.array(self.data[material][key]) * \
            ureg(self.data[material][key + "_unit"])


@pytest.mark.integration
class TestCalculateQPointPhononModes:

    expected_freqs = ExpectedFrequencies(
        os.path.join(get_data_path(), "force_constants", "expected_freqs.json")
    )

    path = os.path.join(get_data_path(), 'force_constants')

    materials = [
        (
            "LZO", 'La2Zr2O7.castep_bin',
            np.array([  # Qpoints
                [-1.00, 9.35, 3.35],
                [-1.00, 9.00, 3.00]
            ])
        ),
        (
            "graphite", "graphite.castep_bin",
            np.array([  # Qpoints
                [0.00, 0.00, 0.00],
                [0.001949, 0.001949, 0.00],
                [0.50, 0.00, 0.00],
                [0.25, 0.00, 0.00],
                [0.00, 0.00, 0.50]
            ])
        )
    ]

    kwargs = [
        {"use_c": False, "fall_back_on_python": True, "n_threads": 1},
        {"use_c": True, "fall_back_on_python": False, "n_threads": 1},
        {"use_c": True, "fall_back_on_python": False, "n_threads": 2},
        {
            "use_c": False, "fall_back_on_python": True,
            "n_threads": 1, "asr": "realspace"
        },
        {
            "use_c": True, "fall_back_on_python": False,
            "n_threads": 1, "asr": "realspace"
        },
        {
            "use_c": True, "fall_back_on_python": False,
            "n_threads": 2, "asr": "realspace"
        }
    ]

    @pytest.fixture(
        params=list(itertools.product(materials, kwargs))
    )
    def create_from_lzo_and_graphite(self, request):
        material, kwargs = request.param
        material_name, castep_bin_file, qpts = material
        kwargs["qpts"] = qpts
        filename = os.path.join(self.path, material_name, castep_bin_file)
        fc = ForceConstants.from_castep(filename)
        key = "asr" if "asr" in kwargs else "no_asr"
        atol = 1e-10
        return fc, kwargs, material_name, key, atol

    qpts = np.array([
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.50],
        [-0.25, 0.50, 0.50],
        [-0.151515, 0.575758, 0.5]
    ])
    split_qpts = np.array([
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
    split_qpts_insert_gamma = np.array([
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

    quartz_castep_bin_file = os.path.join(path, "quartz", "quartz.castep_bin")

    @pytest.fixture(params=[
        (
            {"qpts": qpts, "dipole": True, "splitting": False},
            2e-6
        ),
        (
            {
                "qpts": qpts, "dipole": True, "splitting": False,
                "use_c": True, "fall_back_on_python": False
            },
            2e-6
        ),
        (
            {
                "qpts": qpts, "dipole": True, "splitting": False, "use_c": True,
                "n_threads": 2, "fall_back_on_python": False
            },
            8e-8
        ),
        (
            {
                "qpts": qpts, "asr": 'reciprocal',
                "dipole": True, "splitting": False
            },
            5e-4
        ),
        (
            {
                "qpts": qpts, "asr": 'reciprocal', "dipole": True,
                "splitting": False, "use_c": True, "fall_back_on_python": False
            },
            5e-4
        ),
        (
            {
                "qpts": qpts, "asr": 'reciprocal', "dipole": True,
                "splitting": False, "use_c": True, "n_threads": 2,
                "fall_back_on_python": False
            },
            5e-4
        ),
        (
            {
                "qpts": split_qpts, "asr": 'reciprocal',
                "dipole": True, "splitting": True
            },
            5e-4
        ),
        (
            {
                "qpts": split_qpts, "asr": 'reciprocal', "dipole": True,
                "splitting": True, "use_c": True, "fall_back_on_python": False
            },
            5e-4
        ),
        (
            {
                "qpts": split_qpts, "asr": 'reciprocal', "dipole": True,
                "splitting": True, "use_c": True, "n_threads": 2,
                "fall_back_on_python": False
            },
            5e-4
        ),
        (
            {
                "qpts": split_qpts_insert_gamma, "asr": 'reciprocal',
                "dipole": True, "splitting": True, "insert_gamma": True
            },
            5e-4
        )
    ])
    def create_from_quartz(self, request):
        kwargs, atol = request.param
        fc = ForceConstants.from_castep(self.quartz_castep_bin_file)
        if "asr" in kwargs:
            if "splitting" in kwargs and kwargs["splitting"]:
                key = "asr_splitting"
            else:
                key = "asr"
        else:
            key = "no_asr"
        material_name = "quartz"
        return fc, kwargs, material_name, key, atol

    nacl_yaml_dir = os.path.join(
        get_data_path(), "phonopy_data", "NaCl", "force_constants"
    )

    @pytest.fixture(params=[
            {"dipole": True, "splitting": False},
            {
                "dipole": True, "splitting": False,
                "use_c": True, "fall_back_on_python": False
            },
            {
                "dipole": True, "splitting": False,
                "use_c": True, "fall_back_on_python": False, "n_threads": 2
            },
            {"dipole": True, "splitting": False, "asr": 'reciprocal'},
            {
                "dipole": True, "splitting": False, "asr": 'reciprocal',
                "use_c": True, "fall_back_on_python": False
            },
            {
                "dipole": True, "splitting": False, "asr": 'reciprocal',
                "use_c": True, "fall_back_on_python": False, "n_threads": 2
            },
            {"dipole": True, "splitting": False, "asr": 'realspace'},
            {
                "dipole": True, "splitting": False, "asr": 'realspace',
                "use_c": True, "fall_back_on_python": False
            },
            {
                "dipole": True, "splitting": False, "asr": 'realspace',
                "use_c": True, "fall_back_on_python": False, "n_threads": 2
            }
        ]
    )
    def create_from_nacl(self, request):
        fc = ForceConstants.from_phonopy(
            path=self.nacl_yaml_dir, summary_name="phonopy.yaml"
        )
        kwargs = request.param
        key = kwargs["asr"] if "asr" in kwargs else "no_asr"
        qpts = np.array([
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.50],
            [-0.25, 0.50, 0.50],
            [-0.151515, 0.575758, 0.5]
        ])
        kwargs["qpts"] = qpts
        atol = 1e-8
        material_name = "NaCl"
        return fc, kwargs, material_name, key, atol

    @pytest.mark.parametrize(("force_constants_creator"), [
        pytest.lazy_fixture("create_from_nacl"),
        pytest.lazy_fixture("create_from_quartz"),
        pytest.lazy_fixture("create_from_lzo_and_graphite")
    ])
    def test_fc_calculate_qpoint_phonon_modes_expected_results(
            self, force_constants_creator):
        fc, kwargs, material, key, atol = force_constants_creator
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(**kwargs)
        test_expected_freqs = self.expected_freqs.\
            get_expected_freqs(material, key)
        npt.assert_allclose(
            qpoint_phonon_modes.frequencies.to('hartree').magnitude,
            test_expected_freqs.to('hartree').magnitude,
            atol=atol
        )

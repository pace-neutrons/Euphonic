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
    def calculate_qpoint_phonon_modes(self, request):
        material, kwargs = request.param
        material_name, castep_bin_file, qpts = material
        filename = os.path.join(self.path, material_name, castep_bin_file)
        return ForceConstants.from_castep(filename), qpts, material_name, kwargs

    def test_calculate_qpoint_phonon_modes(self, calculate_qpoint_phonon_modes):
        fc, qpts, material_name, kwargs = calculate_qpoint_phonon_modes
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(
            qpts, **kwargs
        )
        if "asr" in kwargs:
            test_expected_freqs = self.expected_freqs.\
                get_expected_freqs(material_name, "asr")
        else:
            test_expected_freqs = self.expected_freqs.\
                get_expected_freqs(material_name, "no_asr")
        npt.assert_allclose(
            qpoint_phonon_modes.frequencies.to('hartree').magnitude,
            test_expected_freqs.to('hartree').magnitude,
            atol=1e-10
        )

    @pytest.mark.parametrize(
        ("material_name", "castep_bin_file", "fc_mat_asr_file"),
        [
            ('graphite', 'graphite.castep_bin', 'graphite_fc_mat_asr.npy'),
            ('LZO', 'La2Zr2O7.castep_bin', 'lzo_fc_mat_asr.npy')
        ]
    )
    def test_enforce_realspace_acoustic_sum_rule(
            self, material_name, castep_bin_file, fc_mat_asr_file):
        castep_bin_filepath = os.path.join(
            self.path, material_name, castep_bin_file
        )
        fc = ForceConstants.from_castep(castep_bin_filepath)
        expected_fc_mat = np.load(
            os.path.join(self.path, material_name, fc_mat_asr_file)
        )
        fc_mat = fc._enforce_realspace_asr()
        npt.assert_allclose(fc_mat, expected_fc_mat, atol=1e-18)

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

    quartz_test_data = [
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
    ]

    quartz_castep_bin_file = os.path.join(path, "quartz", "quartz.castep_bin")

    @pytest.mark.parametrize(("kwargs", "atol"), quartz_test_data)
    def test_quartz_calculate_qpoint_phonon_modes(self, kwargs, atol):
        fc = ForceConstants.from_castep(self.quartz_castep_bin_file)
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(**kwargs)
        if "asr" in kwargs:
            if "splitting" in kwargs and kwargs["splitting"]:
                test_expected_freqs = self.expected_freqs.\
                    get_expected_freqs("quartz", "asr_splitting")
            else:
                test_expected_freqs = self.expected_freqs.\
                    get_expected_freqs("quartz", "asr")
        else:
            test_expected_freqs = self.expected_freqs\
                .get_expected_freqs("quartz", "no_asr")
        npt.assert_allclose(
            qpoint_phonon_modes.frequencies.to('hartree').magnitude,
            test_expected_freqs.to('hartree').magnitude,
            atol=atol
        )

    phonopy_yaml_file = os.path.join(
        get_data_path(), "phonopy_data", "NaCl", "force_constants"
    )

    @pytest.mark.parametrize(
        ("kwargs"), [
            {"dipole": True, "splitting": False},
            {
                "dipole": True, "splitting": False, "asr": None,
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
    def test_phonopy_calculate_qpoint_phonon_modes(self, kwargs):
        fc = ForceConstants.from_phonopy(
            path=self.phonopy_yaml_file, summary_name="phonopy.yaml"
        )
        if "asr" in kwargs and kwargs["asr"] == 'realspace':
            expected_freqs = self.expected_freqs.\
                get_expected_freqs("NaCl", "realsp_asr")
        elif "asr" in kwargs and kwargs["asr"] == "reciprocal":
            expected_freqs = self.expected_freqs.\
                get_expected_freqs("NaCl", "recip_asr")
        else:
            expected_freqs = self.expected_freqs.\
                get_expected_freqs("NaCl", "no_asr")
        qpts = np.array([
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.50],
            [-0.25, 0.50, 0.50],
            [-0.151515, 0.575758, 0.5]
        ])
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(qpts, **kwargs)
        npt.assert_allclose(
            qpoint_phonon_modes.frequencies.to('hartree').magnitude,
            expected_freqs.to('hartree').magnitude,
            atol=1e-8
        )

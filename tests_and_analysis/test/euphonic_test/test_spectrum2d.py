import os
import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.spectra import Spectrum2D
from tests_and_analysis.test.utils import get_data_path

class ExpectedSpectrum2D:
    def __init__(self, spectrum2d_json_file: str):
        self.data = json.load(open(spectrum2d_json_file))

    @property
    def x_data(self):
        return np.array(self.data['x_data'])*ureg(
            self.data['x_data_unit'])

    @property
    def y_data(self):
        return np.array(self.data['y_data'])*ureg(
            self.data['y_data_unit'])

    @property
    def z_data(self):
        return np.array(self.data['z_data'])*ureg(
            self.data['z_data_unit'])

    @property
    def x_tick_labels(self):
        if 'x_tick_labels' in self.data.keys():
            return [tuple(x) for x in self.data['x_tick_labels']]

    def to_dict(self):
        d = {'x_data': self.x_data.magnitude,
             'x_data_unit': str(self.x_data.units),
             'y_data': self.y_data.magnitude,
             'y_data_unit': str(self.y_data.units),
             'z_data': self.z_data.magnitude,
             'z_data_unit': str(self.z_data.units)}
        if self.x_tick_labels is not None:
            d['x_tick_labels'] = self.x_tick_labels
        return d

    def to_constructor_args(self, x_data=None, y_data=None, z_data=None,
                            x_tick_labels=None):
        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if z_data is None:
            z_data = self.z_data
        if x_tick_labels is None:
            x_tick_labels = self.x_tick_labels

        kwargs = {}
        if x_tick_labels is not None:
            kwargs['x_tick_labels'] = x_tick_labels

        return (x_data, y_data, z_data), kwargs


def get_spectrum2d_dir():
    return os.path.join(get_data_path(), 'spectrum2d')


def get_json_file(json_filename):
    return os.path.join(get_spectrum2d_dir(), json_filename)


def get_spectrum2d(json_filename):
    return Spectrum2D.from_json_file(get_json_file(json_filename))


def get_expected_spectrum2d(json_filename):
    return ExpectedSpectrum2D(get_json_file(json_filename))


def check_spectrum2d(actual_spectrum2d, expected_spectrum2d):

    npt.assert_allclose(actual_spectrum2d.x_data.magnitude,
                        expected_spectrum2d.x_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum2d.x_data.units
            == expected_spectrum2d.x_data.units)

    npt.assert_allclose(actual_spectrum2d.y_data.magnitude,
                        expected_spectrum2d.y_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum2d.y_data.units
            == expected_spectrum2d.y_data.units)

    npt.assert_allclose(actual_spectrum2d.z_data.magnitude,
                        expected_spectrum2d.z_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum2d.z_data.units
            == expected_spectrum2d.z_data.units)

    if expected_spectrum2d.x_tick_labels is None:
        assert actual_spectrum2d.x_tick_labels is None
    else:
        assert (actual_spectrum2d.x_tick_labels
                == expected_spectrum2d.x_tick_labels)


@pytest.mark.unit
class TestSpectrum2DCreation:

    # As x_data and y_data can be either bin centres or edges, test all
    # possible cases with example_spectrum2d, example_xbin_edges_spectrum2d,
    # example_xybin_edges_spectrum2d. The case of x_data being centres
    # and y_data being edges is covered by quartz_bandstructure_sqw
    @pytest.fixture(params=[
        get_expected_spectrum2d('quartz_bandstructure_sqw.json'),
        get_expected_spectrum2d('example_spectrum2d.json'),
        get_expected_spectrum2d('example_xbin_edges_spectrum2d.json'),
        get_expected_spectrum2d('example_xybin_edges_spectrum2d.json')])
    def create_from_constructor(self, request):
        expected_spec2d = request.param
        args, kwargs = expected_spec2d.to_constructor_args()
        spec2d = Spectrum2D(*args, **kwargs)
        return spec2d, expected_spec2d

    @pytest.fixture(params=[
        'quartz_bandstructure_sqw.json',
        'example_spectrum2d.json',
        'example_xbin_edges_spectrum2d.json',
        'example_xybin_edges_spectrum2d.json'])
    def create_from_json(self, request):
        json_file = request.param
        expected_spec2d = get_expected_spectrum2d(json_file)
        spec2d = Spectrum2D.from_json_file(
            get_json_file(json_file))
        return spec2d, expected_spec2d

    @pytest.fixture(params=[
        'quartz_bandstructure_sqw.json',
        'example_spectrum2d.json',
        'example_xbin_edges_spectrum2d.json',
        'example_xybin_edges_spectrum2d.json'])
    def create_from_dict(self, request):
        json_file = request.param
        expected_spec2d = get_expected_spectrum2d(json_file)
        spec2d = Spectrum2D.from_dict(
            expected_spec2d.to_dict())
        return spec2d, expected_spec2d

    @pytest.mark.parametrize(('spec2d_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_dict')])
    def test_correct_object_creation(self, spec2d_creator):
        spec2d, expected_spec2d = spec2d_creator
        check_spectrum2d(spec2d, expected_spec2d)

    @pytest.fixture(params=[
        ('x_data',
         get_expected_spectrum2d('example_spectrum2d.json').x_data.magnitude,
         TypeError),
        ('x_data',
         get_expected_spectrum2d('example_spectrum2d.json').x_data[:-1],
         ValueError),
        ('y_data',
         get_expected_spectrum2d('example_spectrum2d.json').y_data.magnitude,
         TypeError),
        ('y_data',
         get_expected_spectrum2d('example_spectrum2d.json').y_data[:-1],
         ValueError),
        ('z_data',
         get_expected_spectrum2d('example_spectrum2d.json').z_data.magnitude,
         TypeError),
        ('z_data',
         get_expected_spectrum2d('example_spectrum2d.json').z_data[:-2],
         ValueError),
        ('x_tick_labels',
         get_expected_spectrum2d(
             'quartz_bandstructure_sqw.json').x_tick_labels[0],
         TypeError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_spec2d = get_expected_spectrum2d('example_spectrum2d.json')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_spec2d.to_constructor_args(
            **{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            Spectrum2D(*faulty_args, **faulty_kwargs)


@pytest.mark.unit
class TestSpectrum2DSerialisation:

    @pytest.fixture(params=[
        get_spectrum2d('quartz_bandstructure_sqw.json'),
        get_spectrum2d('example_spectrum2d.json'),
        get_spectrum2d('example_xbin_edges_spectrum2d.json'),
        get_spectrum2d('example_xybin_edges_spectrum2d.json')])
    def serialise_to_json_file(self, request, tmpdir):
        spec2d = request.param
        # Serialise
        output_file = str(tmpdir.join('tmp.test'))
        spec2d.to_json_file(output_file)
        # Deserialise
        deserialised_spec2d = Spectrum2D.from_json_file(output_file)
        return spec2d, deserialised_spec2d

    def test_serialise_to_file(self, serialise_to_json_file):
        spec2d, deserialised_spec2d = serialise_to_json_file
        check_spectrum2d(spec2d, deserialised_spec2d)

    @pytest.fixture(params=[
        'quartz_bandstructure_sqw.json',
        'example_spectrum2d.json',
        'example_xbin_edges_spectrum2d.json',
        'example_xybin_edges_spectrum2d.json'])
    def serialise_to_dict(self, request):
        json_file = request.param
        spec2d = get_spectrum2d(json_file)
        expected_spec2d = get_expected_spectrum2d(json_file)
        # Convert to dict, then back to object to test
        spec2d_dict = spec2d.to_dict()
        spec2d_from_dict = Spectrum2D.from_dict(spec2d_dict)
        return spec2d_from_dict, expected_spec2d

    def test_serialise_to_dict(self, serialise_to_dict):
        spec2d, expected_spec2d = serialise_to_dict
        check_spectrum2d(spec2d, expected_spec2d)

@pytest.mark.unit
class TestSpectrum2DMethods:

    @pytest.mark.parametrize(
        'args, spectrum2d_file, broadened_spectrum2d_file', [
        (({'x_width': 0.1*ureg('1/angstrom')}),
        'quartz_bandstructure_sqw.json',
        'quartz_bandstructure_0.1ang_xbroaden_sqw.json'),
        (({'y_width': 2*ureg('meV')}),
        'quartz_bandstructure_sqw.json',
        'quartz_bandstructure_2meV_ybroaden_sqw.json'),
        (({'x_width': 0.1*ureg('1/angstrom'), 'y_width': 2*ureg('meV')}),
        'quartz_bandstructure_sqw.json',
        'quartz_bandstructure_2meV_0.1ang_xybroaden_sqw.json'),
        (({'x_width': 0.1*ureg('1/angstrom'), 'y_width': 2*ureg('meV'),
           'shape': 'lorentz'}),
        'quartz_bandstructure_sqw.json',
        'quartz_bandstructure_xybroaden_lorentz_sqw.json')])
    def test_broaden(self, args, spectrum2d_file, broadened_spectrum2d_file):
        spec2d = get_spectrum2d(spectrum2d_file)
        expected_broadened_spec2d = get_spectrum2d(broadened_spectrum2d_file)
        broadened_spec2d = spec2d.broaden(**args)
        check_spectrum2d(broadened_spec2d, expected_broadened_spec2d)

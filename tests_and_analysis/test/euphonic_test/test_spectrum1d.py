import os
import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.spectra import Spectrum1D
from tests_and_analysis.test.utils import get_data_path, check_unit_conversion


class ExpectedSpectrum1D:
    def __init__(self, spectrum1d_json_file: str):
        with open(spectrum1d_json_file) as fd:
            self.data = json.load(fd)

    @property
    def x_data(self):
        return np.array(self.data['x_data'])*ureg(
            self.data['x_data_unit'])

    @property
    def y_data(self):
        return np.array(self.data['y_data'])*ureg(
            self.data['y_data_unit'])

    @property
    def x_tick_labels(self):
        if 'x_tick_labels' in self.data.keys():
            return [tuple(x) for x in self.data['x_tick_labels']]
        else:
            return None

    def to_dict(self):
        d = {'x_data': self.x_data.magnitude,
             'x_data_unit': str(self.x_data.units),
             'y_data': self.y_data.magnitude,
             'y_data_unit': str(self.y_data.units)}
        if self.x_tick_labels is not None:
            d['x_tick_labels'] = self.x_tick_labels
        return d

    def to_constructor_args(self, x_data=None, y_data=None,
                            x_tick_labels=None):
        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if x_tick_labels is None:
            x_tick_labels = self.x_tick_labels

        kwargs = {}
        if x_tick_labels is not None:
            kwargs['x_tick_labels'] = x_tick_labels

        return (x_data, y_data), kwargs


def get_spectrum1d_dir():
    return os.path.join(get_data_path(), 'spectrum1d')


def get_json_file(json_filename):
    return os.path.join(get_spectrum1d_dir(), json_filename)


def get_spectrum1d(json_filename):
    return Spectrum1D.from_json_file(get_json_file(json_filename))


def get_expected_spectrum1d(json_filename):
    return ExpectedSpectrum1D(get_json_file(json_filename))


def check_spectrum1d(actual_spectrum1d, expected_spectrum1d):

    npt.assert_allclose(actual_spectrum1d.x_data.magnitude,
                        expected_spectrum1d.x_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum1d.x_data.units
            == expected_spectrum1d.x_data.units)

    npt.assert_allclose(actual_spectrum1d.y_data.magnitude,
                        expected_spectrum1d.y_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum1d.y_data.units
            == expected_spectrum1d.y_data.units)

    if expected_spectrum1d.x_tick_labels is None:
        assert actual_spectrum1d.x_tick_labels is None
    else:
        assert (actual_spectrum1d.x_tick_labels
                == expected_spectrum1d.x_tick_labels)


@pytest.mark.unit
class TestSpectrum1DCreation:

    # As x_data can be either bin centres or edges, test both cases with
    # xsqw_spectrum1d and xsq_bin_edges_spectrum1d
    @pytest.fixture(params=[
        get_expected_spectrum1d('quartz_666_dos.json'),
        get_expected_spectrum1d('xsq_spectrum1d.json'),
        get_expected_spectrum1d('xsq_bin_edges_spectrum1d.json')])
    def create_from_constructor(self, request):
        expected_spec1d = request.param
        args, kwargs = expected_spec1d.to_constructor_args()
        spec1d = Spectrum1D(*args, **kwargs)
        return spec1d, expected_spec1d

    @pytest.fixture(params=[
        'quartz_666_dos.json',
        'xsq_spectrum1d.json',
        'xsq_bin_edges_spectrum1d.json'])
    def create_from_json(self, request):
        json_file = request.param
        expected_spec1d = get_expected_spectrum1d(json_file)
        spec1d = Spectrum1D.from_json_file(
            get_json_file(json_file))
        return spec1d, expected_spec1d

    @pytest.fixture(params=[
        'quartz_666_dos.json',
        'xsq_spectrum1d.json',
        'xsq_bin_edges_spectrum1d.json'])
    def create_from_dict(self, request):
        json_file = request.param
        expected_spec1d = get_expected_spectrum1d(json_file)
        spec1d = Spectrum1D.from_dict(
            expected_spec1d.to_dict())
        return spec1d, expected_spec1d

    @pytest.mark.parametrize(('spec1d_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_dict')])
    def test_correct_object_creation(self, spec1d_creator):
        spec1d, expected_spec1d = spec1d_creator
        check_spectrum1d(spec1d, expected_spec1d)

    @pytest.fixture(params=[
        ('x_data',
         get_expected_spectrum1d('xsq_spectrum1d.json').x_data.magnitude,
         TypeError),
        ('x_data',
         get_expected_spectrum1d('xsq_spectrum1d.json').x_data[:-1],
         ValueError),
        ('y_data',
         get_expected_spectrum1d('xsq_spectrum1d.json').y_data.magnitude,
         TypeError),
        ('y_data',
         get_expected_spectrum1d('xsq_spectrum1d.json').y_data[:-2],
         ValueError),
        ('x_tick_labels',
         get_expected_spectrum1d('xsq_spectrum1d.json').x_tick_labels[0],
         TypeError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_spec1d = get_expected_spectrum1d('xsq_spectrum1d.json')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_spec1d.to_constructor_args(
            **{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            Spectrum1D(*faulty_args, **faulty_kwargs)


@pytest.mark.unit
class TestSpectrum1DSerialisation:

    @pytest.fixture(params=[
        get_spectrum1d('quartz_666_dos.json'),
        get_spectrum1d('xsq_spectrum1d.json'),
        get_spectrum1d('xsq_bin_edges_spectrum1d.json')])
    def serialise_to_json_file(self, request, tmpdir):
        spec1d = request.param
        # Serialise
        output_file = str(tmpdir.join('tmp.test'))
        spec1d.to_json_file(output_file)
        # Deserialise
        deserialised_spec1d = Spectrum1D.from_json_file(output_file)
        return spec1d, deserialised_spec1d

    def test_serialise_to_file(self, serialise_to_json_file):
        spec1d, deserialised_spec1d = serialise_to_json_file
        check_spectrum1d(spec1d, deserialised_spec1d)

    @pytest.fixture(params=[
        'quartz_666_dos.json',
        'xsq_spectrum1d.json',
        'xsq_bin_edges_spectrum1d.json'])
    def serialise_to_dict(self, request):
        json_file = request.param
        spec1d = get_spectrum1d(json_file)
        expected_spec1d = get_expected_spectrum1d(json_file)
        # Convert to dict, then back to object to test
        spec1d_dict = spec1d.to_dict()
        spec1d_from_dict = Spectrum1D.from_dict(spec1d_dict)
        return spec1d_from_dict, expected_spec1d

    def test_serialise_to_dict(self, serialise_to_dict):
        spec1d, expected_spec1d = serialise_to_dict
        check_spectrum1d(spec1d, expected_spec1d)


@pytest.mark.unit
class TestSpectrum1DUnitConversion:

    @pytest.mark.parametrize('spectrum1d_file, attr, unit_val', [
        ('xsq_spectrum1d.json', 'x_data', '1/bohr'),
        ('xsq_spectrum1d.json', 'y_data', '1/cm')])
    def test_correct_unit_conversion(self, spectrum1d_file, attr, unit_val):
        spec1d = get_spectrum1d(spectrum1d_file)
        check_unit_conversion(spec1d, attr, unit_val)

    @pytest.mark.parametrize('spectrum1d_file, unit_attr, unit_val, err', [
        ('xsq_spectrum1d.json', 'x_data_unit', 'kg', ValueError),
        ('xsq_spectrum1d.json', 'y_data_unit', 'kg', ValueError)])
    def test_incorrect_unit_conversion(self, spectrum1d_file, unit_attr,
                                       unit_val, err):
        spec1d = get_spectrum1d(spectrum1d_file)
        with pytest.raises(err):
            setattr(spec1d, unit_attr, unit_val)


@pytest.mark.unit
class TestSpectrum1DSplitting:
    _sine_spectrum_x = np.linspace(0, 10, 21)
    _sine_spectrum = Spectrum1D(
        _sine_spectrum_x * ureg('1/bohr'),
        np.sin(_sine_spectrum_x) * ureg(None),
        x_tick_labels=[(i, f'{xi:.1f}') for i, xi in enumerate(_sine_spectrum_x)])

    sine_spectrum_data = {
            'spectrum': _sine_spectrum,
            'params': dict(indices=[2, 8]),
            'expected_split_x': [[0., 0.5],
                                 [1., 1.5, 2., 2.5, 3., 3.5],
                                 np.linspace(4., 10., 13)],
            'expected_split_y': [np.sin([0, 0.5]),
                                 np.sin(np.linspace(1, 3.5, 6)),
                                 np.sin(np.linspace(4., 10., 13))],
            'expected_split_labels': [[(0, '0.0'), (1, '0.5')],
                                      [(0, '1.0'), (1, '1.5'), (2, '2.0'),
                                       (3, '2.5'), (4, '3.0'), (5, '3.5')],
                                      [(i, f'{xi:.1f}')
                                       for i, xi
                                       in enumerate(np.linspace(4, 10, 13))]]
            }

    _band_split_x = [2., 3., 4., 24., 25., 26., 27., 50., 50.9]
    _band_split_y = np.random.random(9).tolist()

    _band_split_spectrum= Spectrum1D(_band_split_x * ureg('1/angstrom'),
                                     _band_split_y * ureg(None),
                                     x_tick_labels=[(0, 'A'), (2, 'B'),
                                                    (3, 'C'), (4, 'D'), (6, 'E'),
                                                    (7, 'F'), (8, 'G')])

    band_split_data = {
            'spectrum': _band_split_spectrum,
            'params': dict(btol=10.),
            'expected_split_x': [[2., 3., 4.],
                                 [24., 25., 26., 27.], [50., 50.9]],
            'expected_split_y': [_band_split_y[0:3],
                                 _band_split_y[3:7],
                                 _band_split_y[7:]],
            'expected_split_labels': [[(0, 'A'), (2, 'B')],
                                      [(0, 'C'), (1, 'D'), (3, 'E')],
                                      [(0, 'F'), (1, 'G')]]
         }

    @pytest.mark.parametrize('spectrum_data', [sine_spectrum_data, band_split_data])
    def test_split(self, spectrum_data):
        full_spectrum = spectrum_data['spectrum']
        spectra = full_spectrum.split(**spectrum_data['params'])
        for spectrum, expected_x in zip(spectra,
                                        spectrum_data['expected_split_x']):
            npt.assert_allclose(spectrum.x_data.magnitude, expected_x)
        for spectrum, expected_y in zip(spectra,
                                        spectrum_data['expected_split_y']):
            npt.assert_allclose(spectrum.y_data.magnitude, expected_y)
        for spectrum, expected_labels in zip(
                spectra, spectrum_data['expected_split_labels']):
            assert spectrum.x_tick_labels == expected_labels


@pytest.mark.unit
class TestSpectrum1DMethods:

    @pytest.mark.parametrize(
        'args, spectrum1d_file, broadened_spectrum1d_file', [
        ((1*ureg('meV'), {}),
        'quartz_666_dos.json',
        'quartz_666_1meV_gauss_broaden_dos.json'),
        ((1*ureg('meV'), {'shape':'gauss'}),
        'quartz_666_dos.json',
        'quartz_666_1meV_gauss_broaden_dos.json'),
        ((1*ureg('meV'), {'shape': 'lorentz'}),
        'quartz_666_dos.json',
        'quartz_666_1meV_lorentz_broaden_dos.json')])
    def test_broaden(self, args, spectrum1d_file, broadened_spectrum1d_file):
        spec1d = get_spectrum1d(spectrum1d_file)
        expected_broadened_spec1d = get_spectrum1d(broadened_spectrum1d_file)
        broadened_spec1d = spec1d.broaden(args[0], **args[1])
        check_spectrum1d(broadened_spec1d, expected_broadened_spec1d)

    def test_broaden_invalid_shape_raises_value_error(self):
        spec1d = get_spectrum1d('quartz_666_dos.json')
        with pytest.raises(ValueError):
            spec1d.broaden(1*ureg('meV'), shape='unknown')

    @pytest.mark.parametrize(
        'spectrum1d_file, expected_bin_edges', [
            ('xsq_spectrum1d.json',
             np.array([0.5, 1., 2., 3., 4., 5., 6.25, 8., 10., 12., 14., 16.5,
                       20., 24., 26.])*ureg('1/angstrom')),
            ('xsq_bin_edges_spectrum1d.json',
             np.array([0., 1., 2., 3., 4., 5., 6., 8., 10., 12., 14., 16., 20.,
                       24., 28.])*ureg('1/angstrom'))])
    def test_get_bin_edges(self, spectrum1d_file, expected_bin_edges):
        spec1d = get_spectrum1d(spectrum1d_file)
        bin_edges = spec1d._get_bin_edges()
        assert bin_edges.units == expected_bin_edges.units
        npt.assert_allclose(bin_edges.magnitude, expected_bin_edges.magnitude)

    @pytest.mark.parametrize(
        'spectrum1d_file, expected_bin_centres', [
            ('xsq_spectrum1d.json',
             np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7., 9., 11., 13., 15.,
                       18., 22., 26.])*ureg('1/angstrom')),
            ('xsq_bin_edges_spectrum1d.json',
             np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7., 9., 11., 13., 15.,
                       18., 22., 26.])*ureg('1/angstrom'))])
    def test_get_bin_centres(self, spectrum1d_file, expected_bin_centres):
        spec1d = get_spectrum1d(spectrum1d_file)
        bin_centres = spec1d._get_bin_centres()
        assert bin_centres.units == expected_bin_centres.units
        npt.assert_allclose(bin_centres.magnitude,
                            expected_bin_centres.magnitude)

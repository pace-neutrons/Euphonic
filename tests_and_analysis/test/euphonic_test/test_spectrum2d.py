import os
import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.spectra import Spectrum2D
from tests_and_analysis.test.utils import (
    get_data_path, check_unit_conversion, check_json_metadata,
    check_property_setters)

class ExpectedSpectrum2D:
    def __init__(self, spectrum2d_json_file: str):
        with open(spectrum2d_json_file) as fd:
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
    def z_data(self):
        return np.array(self.data['z_data'])*ureg(
            self.data['z_data_unit'])

    @property
    def x_tick_labels(self):
        if 'x_tick_labels' in self.data.keys():
            return [tuple(x) for x in self.data['x_tick_labels']]
        else:
            return None

    @property
    def metadata(self):
        if 'metadata' in self.data.keys():
            return self.data['metadata']
        else:
            return {}

    def to_dict(self):
        d = {'x_data': self.x_data.magnitude,
             'x_data_unit': str(self.x_data.units),
             'y_data': self.y_data.magnitude,
             'y_data_unit': str(self.y_data.units),
             'z_data': self.z_data.magnitude,
             'z_data_unit': str(self.z_data.units)}
        if self.x_tick_labels is not None:
            d['x_tick_labels'] = self.x_tick_labels
        if self.metadata is not None:
            d['metadata'] = self.metadata
        return d

    def to_constructor_args(self, x_data=None, y_data=None, z_data=None,
                            x_tick_labels=None, metadata=None):
        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if z_data is None:
            z_data = self.z_data
        if x_tick_labels is None:
            x_tick_labels = self.x_tick_labels
        if metadata is None:
            metadata = self.metadata

        kwargs = {}
        if x_tick_labels is not None:
            kwargs['x_tick_labels'] = x_tick_labels
        if metadata is not None:
            kwargs['metadata'] = metadata

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

    if expected_spectrum2d.metadata is None:
        assert actual_spectrum2d.metadata is None
    else:
        assert (actual_spectrum2d.metadata
                == expected_spectrum2d.metadata)


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
         TypeError),
        ('metadata',
         np.arange(10),
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

    @pytest.mark.parametrize('spec2d', [
        get_spectrum2d('quartz_bandstructure_sqw.json'),
        get_spectrum2d('example_spectrum2d.json'),
        get_spectrum2d('example_xbin_edges_spectrum2d.json'),
        get_spectrum2d('example_xybin_edges_spectrum2d.json')])
    def test_serialise_to_json_file(self, spec2d, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        spec2d.to_json_file(output_file)
        check_json_metadata(output_file, 'Spectrum2D')
        deserialised_spec2d = Spectrum2D.from_json_file(output_file)
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
class TestSpectrum2DUnitConversion:

    @pytest.mark.parametrize('spectrum2d_file, attr, unit_val', [
        ('example_spectrum2d.json', 'x_data', 'angstrom'),
        ('quartz_bandstructure_sqw.json', 'y_data', 'hartree'),
        ('example_spectrum2d.json', 'z_data', 'amu')])
    def test_correct_unit_conversion(self, spectrum2d_file, attr, unit_val):
        spec2d = get_spectrum2d(spectrum2d_file)
        check_unit_conversion(spec2d, attr, unit_val)

    @pytest.mark.parametrize('spectrum2d_file, unit_attr, unit_val, err', [
        ('example_spectrum2d.json', 'x_data_unit', 'N', ValueError),
        ('example_spectrum2d.json', 'y_data_unit', 'kg', ValueError),
        ('example_spectrum2d.json', 'z_data_unit', 'm', ValueError)])
    def test_incorrect_unit_conversion(self, spectrum2d_file, unit_attr,
                                       unit_val, err):
        spec2d = get_spectrum2d(spectrum2d_file)
        with pytest.raises(err):
            setattr(spec2d, unit_attr, unit_val)


@pytest.mark.unit
class TestSpectrum2DSetters:

    @pytest.mark.parametrize('spectrum2d_file, attr, unit, scale', [
        ('example_spectrum2d.json', 'x_data', 'm', 3.),
        ('example_spectrum2d.json', 'x_data', 'angstrom', 2.),
        ('example_spectrum2d.json', 'y_data', 'dimensionless', 3.),
        ('example_spectrum2d.json', 'z_data', 'kg', 2.),
        ('example_spectrum2d.json', 'z_data', 'm_e', 2.),
        ])
    def test_setter_correct_units(self, spectrum2d_file, attr,
                                  unit, scale):
        spec2d = get_spectrum2d(spectrum2d_file)
        check_property_setters(spec2d, attr, unit, scale)

    @pytest.mark.parametrize('spectrum2d_file, attr, unit, err', [
        ('example_spectrum2d.json', 'x_data', 'kg', ValueError),
        ('example_spectrum2d.json', 'y_data', 'mbarn', ValueError),
        ('example_spectrum2d.json', 'z_data', 'angstrom', ValueError)])
    def test_incorrect_unit_conversion(self, spectrum2d_file, attr,
                                       unit, err):
        spec2d = get_spectrum2d(spectrum2d_file)
        new_attr = getattr(spec2d, attr).magnitude*ureg(unit)
        with pytest.raises(err):
            setattr(spec2d, attr, new_attr)


@pytest.mark.unit
class TestSpectrum2DMethods:
    @pytest.mark.parametrize(
        'args, spectrum2d_file, split_spectrum_files',
        [({'indices': (4,)}, 'example_spectrum2d.json',
          [f'example_spectrum2d_split4_{i}.json' for i in range(2)])])
    def test_split(self, args, spectrum2d_file, split_spectrum_files):
        spec2d = get_spectrum2d(spectrum2d_file)
        split_spec2d = spec2d.split(**args)
        for spectrum, expected_file in zip(split_spec2d, split_spectrum_files):
            expected_spectrum = get_spectrum2d(expected_file)
            check_spectrum2d(spectrum, expected_spectrum)

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
             'quartz_bandstructure_xybroaden_lorentz_sqw.json'),
            (({'x_width': 0.2*ureg('1/angstrom'), 'y_width': 1.5*ureg('meV'),
               'shape': 'gauss'}),
             'lzo_57L_bragg_sqw.json',
             'lzo_57L_1.5meV_0.1ang_gauss_sqw.json'),
            (({'x_width': 0.2*ureg('1/angstrom'), 'y_width': 1.5*ureg('meV'),
               'shape': 'lorentz'}),
             'lzo_57L_bragg_sqw.json',
             'lzo_57L_1.5meV_0.1ang_lorentz_sqw.json')])
    def test_broaden(self, args, spectrum2d_file, broadened_spectrum2d_file):
        spec2d = get_spectrum2d(spectrum2d_file)
        expected_broadened_spec2d = get_spectrum2d(broadened_spectrum2d_file)
        broadened_spec2d = spec2d.broaden(**args)
        check_spectrum2d(broadened_spec2d, expected_broadened_spec2d)

    def test_broaden_invalid_shape_raises_value_error(self):
        spec2d = get_spectrum2d('quartz_bandstructure_sqw.json')
        with pytest.raises(ValueError):
            spec2d.broaden(y_width=1*ureg('meV'), shape='unknown')

    @pytest.mark.parametrize(
        'spectrum2d_file, ax, expected_bin_edges', [
            ('example_spectrum2d.json', 'x',
             np.array([0., 0.5, 1.5, 2.5, 3.5, 4.5, 6., 8., 10., 12.,
                       13.])*ureg('meter')),
            ('example_xbin_edges_spectrum2d.json', 'x',
             np.array([0., 1., 2., 3., 4., 5., 7., 9., 11.,
                       13.])*ureg('meter')),
            ('example_spectrum2d.json', 'y',
             np.array([-2.5, -2.25, -1.75, -1.25, -0.75, -0.25, 0.5, 1.5, 2.5,
                       3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
                       13.5, 14.5, 15.])*ureg('dimensionless')),
            ('example_xybin_edges_spectrum2d.json', 'y',
             np.array([-2.5, -2., -1.5, -1., -0.5, 0., 1., 2., 3., 4., 5., 6.,
                       7., 8., 9., 10., 11., 12., 13., 14.,
                       15.])*ureg('dimensionless'))])
    def test_get_bin_edges(self, spectrum2d_file, ax, expected_bin_edges):
        spec2d = get_spectrum2d(spectrum2d_file)
        bin_edges = spec2d.get_bin_edges(bin_ax=ax)
        assert bin_edges.units == expected_bin_edges.units
        npt.assert_allclose(bin_edges.magnitude, expected_bin_edges.magnitude)

    @pytest.mark.parametrize(
        'spectrum2d_file, ax, expected_bin_centres', [
            ('example_spectrum2d.json', 'x',
             np.array([0., 1., 2., 3., 4., 5., 7., 9., 11.,
                       13.])*ureg('meter')),
            ('example_xbin_edges_spectrum2d.json', 'x',
             np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6., 8., 10.,
                       12.])*ureg('meter')),
            ('example_spectrum2d.json', 'y',
             np.array([-2.5, -2., -1.5, -1., -0.5, 0., 1., 2., 3., 4., 5., 6.,
                       7., 8., 9., 10., 11., 12., 13., 14.,
                       15.])*ureg('dimensionless')),
            ('example_xybin_edges_spectrum2d.json', 'y',
             np.array([-2.25, -1.75, -1.25, -0.75, -0.25, 0.5, 1.5, 2.5, 3.5,
                       4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5,
                       14.5])*ureg('dimensionless'))])
    def test_get_bin_centres(self, spectrum2d_file, ax, expected_bin_centres):
        spec2d = get_spectrum2d(spectrum2d_file)
        bin_centres = spec2d.get_bin_centres(bin_ax=ax)
        assert bin_centres.units == expected_bin_centres.units
        npt.assert_allclose(bin_centres.magnitude,
                            expected_bin_centres.magnitude)

    @pytest.mark.parametrize(
        'bin_ax', ['x', 'y']
    )
    def test_get_bin_edges_with_invalid_data_shape_raises_value_error(
            self, bin_ax):
        spec2d = get_spectrum2d('example_xybin_edges_spectrum2d.json')
        spec2d._z_data = spec2d._z_data[:4, :6]
        with pytest.raises(ValueError):
            spec2d.get_bin_centres()

    @pytest.mark.parametrize(
        'bin_ax', ['x', 'y']
    )
    def test_get_bin_centres_with_invalid_data_shape_raises_value_error(
            self, bin_ax):
        spec2d = get_spectrum2d('example_spectrum2d.json')
        spec2d._z_data = spec2d._z_data[:3, :5]
        with pytest.raises(ValueError):
            spec2d.get_bin_centres()



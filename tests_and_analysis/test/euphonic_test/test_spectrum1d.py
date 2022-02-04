import os
import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.spectra import Spectrum1D
from tests_and_analysis.test.utils import (
    get_data_path, get_castep_path, check_unit_conversion,
    check_json_metadata, check_property_setters, get_spectrum_from_text,
    check_spectrum_text_header)


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
             'y_data_unit': str(self.y_data.units)}
        if self.x_tick_labels is not None:
            d['x_tick_labels'] = self.x_tick_labels
        if self.metadata is not None:
            d['metadata'] = self.metadata
        return d

    def to_constructor_args(self, x_data=None, y_data=None,
                            x_tick_labels=None, metadata=None):
        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if x_tick_labels is None:
            x_tick_labels = self.x_tick_labels
        if metadata is None:
            metadata = self.metadata

        kwargs = {}
        if x_tick_labels is not None:
            kwargs['x_tick_labels'] = x_tick_labels
        if metadata is not None:
            kwargs['metadata'] = metadata

        return (x_data, y_data), kwargs


def get_spectrum1d_path(*subpaths):
    return get_data_path('spectrum1d', *subpaths)


def get_spectrum1d(json_filename):
    return Spectrum1D.from_json_file(get_spectrum1d_path(json_filename))


def get_expected_spectrum1d(json_filename):
    return ExpectedSpectrum1D(get_spectrum1d_path(json_filename))


def check_spectrum1d(actual_spectrum1d, expected_spectrum1d, y_atol=None):

    if y_atol is None:
        y_atol = np.finfo(np.float64).eps

    assert (actual_spectrum1d.x_data.units
            == expected_spectrum1d.x_data.units)
    npt.assert_allclose(actual_spectrum1d.x_data.magnitude,
                        expected_spectrum1d.x_data.magnitude,
                        atol=np.finfo(np.float64).eps)

    assert (actual_spectrum1d.y_data.units
            == expected_spectrum1d.y_data.units)
    npt.assert_allclose(actual_spectrum1d.y_data.magnitude,
                        expected_spectrum1d.y_data.magnitude,
                        atol=y_atol)

    if expected_spectrum1d.x_tick_labels is None:
        assert actual_spectrum1d.x_tick_labels is None
    else:
        assert (actual_spectrum1d.x_tick_labels
                == expected_spectrum1d.x_tick_labels)

    if expected_spectrum1d.metadata is None:
        assert actual_spectrum1d.metadata is None
    else:
        assert (actual_spectrum1d.metadata
                == expected_spectrum1d.metadata)


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
            get_spectrum1d_path(json_file))
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

    @pytest.fixture(params=[
        ('quartz', 'quartz-554-full.phonon_dos', {},
         'quartz_554_full_castep_total_adaptive_dos.json'),
        ('quartz', 'quartz-554-full.phonon_dos', {'element': 'Si'},
         'quartz_554_full_castep_si_adaptive_dos.json'),
        ('LZO', 'La2Zr2O7-222-full.phonon_dos', {},
         'lzo_222_full_castep_total_adaptive_dos.json'),
        ('LZO', 'La2Zr2O7-222-full.phonon_dos', {'element': 'Zr'},
         'lzo_222_full_castep_zr_adaptive_dos.json')])
    def create_from_castep_phonon_dos(self, request):
        material, phonon_dos_file, kwargs, json_file = request.param
        expected_spec1d = get_expected_spectrum1d(json_file)
        spec1d = Spectrum1D.from_castep_phonon_dos(
            get_castep_path(material, phonon_dos_file), **kwargs)
        spec1d.y_data_unit = 'cm'
        return spec1d, expected_spec1d

    @pytest.mark.parametrize(('spec1d_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_castep_phonon_dos')])
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
         TypeError),
        ('x_tick_labels',
         [(0,), (1, 'one'), (2, 'two')],
         TypeError),
        ('x_tick_labels',
         [(0, 'zero'), (1,), (2,)],
         TypeError),
        ('x_tick_labels',
         [(0, 1), (2, 3), (4, 5)],
         TypeError),
        ('metadata',
         ['Not', 'a', 'dictionary'],
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


class TestSpectrum1DSerialisation:

    # Note that when writing .text there must be the same number of
    # x_data and y_data points so bin centres will be used, this and
    # using fmt may mean the output spectrum is slightly different.
    # x_tick_labels will also be lost
    @pytest.mark.parametrize('in_json, out_json', [
        ('quartz_666_dos.json', 'quartz_666_dos_from_text.json'),
        ('methane_pdos_index_1.json', 'methane_pdos_index_1.json')])
    def test_serialise_to_text_file(self, in_json, out_json, tmpdir):
        spec1d = get_spectrum1d(in_json)
        # Serialise
        output_file = str(tmpdir.join('tmp.test'))
        spec1d.to_text_file(output_file)
        # Deserialise
        deserialised_spec1d = get_spectrum_from_text(
            output_file, is_collection=False)
        expected_deserialised_spec1d = get_spectrum1d(out_json)
        check_spectrum_text_header(output_file)
        check_spectrum1d(deserialised_spec1d, expected_deserialised_spec1d)

    @pytest.mark.parametrize('spec1d', [
        get_spectrum1d('quartz_666_dos.json'),
        get_spectrum1d('xsq_spectrum1d.json'),
        get_spectrum1d('xsq_bin_edges_spectrum1d.json')])
    def test_serialise_to_json_file(self, spec1d, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        spec1d.to_json_file(output_file)
        check_json_metadata(output_file, 'Spectrum1D')
        deserialised_spec1d = Spectrum1D.from_json_file(output_file)
        check_spectrum1d(spec1d, deserialised_spec1d)

    @pytest.mark.parametrize('json_file', [
        'quartz_666_dos.json',
        'xsq_spectrum1d.json',
        'xsq_bin_edges_spectrum1d.json'])
    def test_serialise_to_dict(self, json_file):
        spec1d = get_spectrum1d(json_file)
        expected_spec1d = get_expected_spectrum1d(json_file)
        # Convert to dict, then back to object to test
        spec1d_dict = spec1d.to_dict()
        spec1d_from_dict = Spectrum1D.from_dict(spec1d_dict)
        check_spectrum1d(spec1d_from_dict, expected_spec1d)


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


class TestSpectrum1DSetters:

    @pytest.mark.parametrize('spectrum1d_file, attr, unit, scale', [
        ('xsq_spectrum1d.json', 'x_data', '1/cm', 3.),
        ('xsq_spectrum1d.json', 'x_data', '1/angstrom', 2.),
        ('xsq_spectrum1d.json', 'y_data', '1/bohr', 3.),
        ('xsq_spectrum1d.json', 'y_data', '1/cm', 2.),
        ])
    def test_setter_correct_units(self, spectrum1d_file, attr,
                                  unit, scale):
        spec1d = get_spectrum1d(spectrum1d_file)
        check_property_setters(spec1d, attr, unit, scale)

    @pytest.mark.parametrize('spectrum1d_file, attr, unit, err', [
        ('xsq_spectrum1d.json', 'x_data', 'kg', ValueError),
        ('xsq_spectrum1d.json', 'y_data', 'mbarn', ValueError)])
    def test_incorrect_unit_conversion(self, spectrum1d_file, attr,
                                       unit, err):
        spec1d = get_spectrum1d(spectrum1d_file)
        new_attr = getattr(spec1d, attr).magnitude*ureg(unit)
        with pytest.raises(err):
            setattr(spec1d, attr, new_attr)

    @pytest.mark.parametrize('value', [[(0, 'zero'), (1, 'one')]])
    def test_x_tick_labels_setter(self, value):
         spec1d = get_spectrum1d('xsq_spectrum1d.json')
         spec1d.x_tick_labels = value
         assert spec1d.x_tick_labels == value

    @pytest.mark.parametrize('value', [
        [(0,), (1, 'one')],
        [(0, 'zero'), ('one', 'one')]])
    def test_x_tick_labels_incorrect_setter(self, value):
         spec1d = get_spectrum1d('xsq_spectrum1d.json')
         with pytest.raises(TypeError):
             spec1d.x_tick_labels = value

    def test_x_tick_labels_converted_to_plain_int(self):
        # np.arange returns an array of type np.int32
        x_tick_labels = [(idx, 'label') for idx in np.arange(5)]
        spec1d = get_spectrum1d('xsq_spectrum1d.json')
        spec1d.x_tick_labels = x_tick_labels
        assert np.all([isinstance(x[0], int) for x in spec1d.x_tick_labels])


class TestSpectrum1DMethods:
    @pytest.mark.parametrize(
        'args, spectrum1d_file, split_spectrum_files', [
            # Multiple split by index, with x_tick_labels
            ({'indices': (3, 8)}, 'xsq_spectrum1d.json',
             [f'xsq_split38_{i}.json' for i in range(3)]),
            # Single split by index, no x_tick_labels
            ({'indices': (11,)}, 'quartz_666_dos.json',
             [f'quartz_666_split_11_{i}.json' for i in range(2)]),
            # No split
            ({'indices': ()}, 'xsq_spectrum1d.json', ('xsq_spectrum1d.json',)),
            # Default (btol=10) split of band data
            ({}, 'toy_band.json',
             [f'toy_band_btol10_split_{i}.json' for i in range(3)]),
            # Non-default btol split of band data
            ({'btol': 22.}, 'toy_band.json',
             [f'toy_band_btol22_split_{i}.json' for i in range(2)]),
            ])             
    def test_split(self, args, spectrum1d_file, split_spectrum_files):
        spec1d = get_spectrum1d(spectrum1d_file)
        split_spec1d = spec1d.split(**args)
        for spectrum, expected_file in zip(split_spec1d, split_spectrum_files):
            check_spectrum1d(spectrum, get_spectrum1d(expected_file))        

    @pytest.mark.parametrize(
        'args, spectrum1d_file, expected_error',
        [({'indices': (3, 4), 'btol': 4.}, 'xsq_spectrum1d.json', ValueError)])
    def test_split_errors(self, args, spectrum1d_file, expected_error):
        spec1d = get_spectrum1d(spectrum1d_file)
        with pytest.raises(expected_error):
            spec1d.split(**args)

    @pytest.mark.parametrize(
        'args, spectrum1d_file, broadened_spectrum1d_file', [
        ((1*ureg('meV'), {}),
        'methane_pdos_index_1.json',
        'methane_pdos_index_1_1meV_gauss_broaden.json'),
        ((1*ureg('meV'), {}),
        'quartz_666_dos.json',
        'quartz_666_1meV_gauss_broaden_dos.json'),
        ((1*ureg('meV'), {'shape':'gauss'}),
        'quartz_666_dos.json',
        'quartz_666_1meV_gauss_broaden_dos.json'),
        ((1*ureg('meV'), {'shape': 'lorentz'}),
        'quartz_666_dos.json',
        'quartz_666_1meV_lorentz_broaden_dos.json'),
        ((1*ureg('meV'), {'method': 'convolve'}),
         'toy_quartz_cropped_uneven_dos.json',
         'toy_quartz_cropped_uneven_broaden_dos.json')])
    def test_broaden(self, args, spectrum1d_file, broadened_spectrum1d_file):
        spec1d = get_spectrum1d(spectrum1d_file)
        expected_broadened_spec1d = get_spectrum1d(broadened_spectrum1d_file)
        broadened_spec1d = spec1d.broaden(args[0], **args[1])
        check_spectrum1d(broadened_spec1d, expected_broadened_spec1d)

    def test_broaden_invalid_shape_raises_value_error(self):
        spec1d = get_spectrum1d('quartz_666_dos.json')
        with pytest.raises(ValueError):
            spec1d.broaden(1*ureg('meV'), shape='unknown')

    def test_broaden_invalid_method_raises_value_error(self):
        spec1d = get_spectrum1d('quartz_666_dos.json')
        with pytest.raises(ValueError):
            spec1d.broaden(1*ureg('meV'), method='unknown')

    def test_broaden_uneven_bins_deprecation_warns(self):
        # In a future version this should raise a ValueError
        spec1d = get_spectrum1d('toy_quartz_cropped_uneven_dos.json')
        with pytest.warns(DeprecationWarning):
            spec1d.broaden(1*ureg('meV'))

    def test_broaden_uneven_bins_and_explicit_convolve_warns(self):
        spec1d = get_spectrum1d('toy_quartz_cropped_uneven_dos.json')
        with pytest.warns(UserWarning):
            spec1d.broaden(1*ureg('meV'), method='convolve')

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
        bin_edges = spec1d.get_bin_edges()
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
        bin_centres = spec1d.get_bin_centres()
        assert bin_centres.units == expected_bin_centres.units
        npt.assert_allclose(bin_centres.magnitude,
                            expected_bin_centres.magnitude)

    def test_get_bin_edges_with_invalid_data_shape_raises_value_error(self):
        spec1d = get_spectrum1d('xsq_spectrum1d.json')
        spec1d._x_data = spec1d._x_data[:4]
        with pytest.raises(ValueError):
            spec1d.get_bin_edges()

    def test_get_bin_centres_with_invalid_data_shape_raises_value_error(self):
        spec1d = get_spectrum1d('xsq_bin_edges_spectrum1d.json')
        spec1d._x_data = spec1d._x_data[:5]
        with pytest.raises(ValueError):
            spec1d.get_bin_centres()

    @pytest.mark.parametrize(
        'spectrum_file, other_spectrum_file, expected_spectrum_file',
        [('gan_bands_index_2.json', 'gan_bands_index_3.json',
          'gan_bands_index_2_3_add.json')])
    def test_add(self, spectrum_file, other_spectrum_file,
                 expected_spectrum_file):
        spec = get_spectrum1d(spectrum_file)
        other_spec = get_spectrum1d(other_spectrum_file)
        added_spectrum = spec + other_spec
        expected_spectrum = get_expected_spectrum1d(expected_spectrum_file)
        check_spectrum1d(added_spectrum, expected_spectrum)

    @pytest.mark.parametrize(
        'spectrum_files, metadata, expected_metadata',
        [(['gan_bands_index_2.json', 'gan_bands_index_3.json'],
          [{'a': 1, 'b': 'fizz', 'c': 'value'},
           {'a': 1, 'b': 'buzz'}],
          {'a': 1})])
    def test_add_metadata(self, spectrum_files, metadata,
                          expected_metadata):
        spec = get_spectrum1d(spectrum_files[0])
        spec.metadata = metadata[0]
        other_spec = get_spectrum1d(spectrum_files[1])
        other_spec.metadata = metadata[1]
        added_spectrum = spec + other_spec
        assert added_spectrum.metadata == expected_metadata

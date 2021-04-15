import os
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from euphonic import ureg
from euphonic.spectra import Spectrum1DCollection
from tests_and_analysis.test.utils import get_data_path, get_castep_path

from .test_spectrum1d import (get_spectrum1d, get_expected_spectrum1d,
                              check_spectrum1d, check_unit_conversion,
                              check_property_setters)


class ExpectedSpectrum1DCollection:
    def __init__(self, spectrum_json_file: str):
        with open(spectrum_json_file) as fd:
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


def get_spectrum_dir():
    return os.path.join(get_data_path(), 'spectrum1dcollection')


def get_json_file(json_filename):
    return os.path.join(get_spectrum_dir(), json_filename)


def get_spectrum1dcollection(json_filename):
    return Spectrum1DCollection.from_json_file(get_json_file(json_filename))


def get_expected_spectrum1dcollection(json_filename):
    return ExpectedSpectrum1DCollection(get_json_file(json_filename))


def check_spectrum1dcollection(actual_spectrum, expected_spectrum):

    assert (actual_spectrum.x_data.units
            == expected_spectrum.x_data.units)
    assert_allclose(actual_spectrum.x_data.magnitude,
                    expected_spectrum.x_data.magnitude,
                    atol=np.finfo(np.float64).eps)

    assert (actual_spectrum.y_data.units
            == expected_spectrum.y_data.units)
    assert_allclose(actual_spectrum.y_data.magnitude,
                    expected_spectrum.y_data.magnitude,
                    atol=np.finfo(np.float64).eps)

    if expected_spectrum.x_tick_labels is None:
        assert actual_spectrum.x_tick_labels is None
    else:
        assert (actual_spectrum.x_tick_labels
                == expected_spectrum.x_tick_labels)

    if expected_spectrum.metadata is None:
        assert actual_spectrum.metadata is None
    else:
        assert (actual_spectrum.metadata
                == expected_spectrum.metadata)


@pytest.mark.unit
class TestSpectrum1DCollectionCreation:
    @pytest.fixture(params=[
        get_expected_spectrum1dcollection('gan_bands.json'),
        get_expected_spectrum1dcollection('methane_pdos.json')])
    def create_from_constructor(self, request):
        expected_spectrum = request.param
        args, kwargs = expected_spectrum.to_constructor_args()
        spectrum = Spectrum1DCollection(*args, **kwargs)
        return spectrum, expected_spectrum

    @pytest.fixture(params=[
        'gan_bands.json',
        'methane_pdos.json'])
    def create_from_json(self, request):
        json_file = request.param
        expected_spectrum = get_expected_spectrum1dcollection(json_file)
        spectrum = Spectrum1DCollection.from_json_file(
            get_json_file(json_file))
        return spectrum, expected_spectrum

    @pytest.fixture(params=[
        'gan_bands.json',
        'methane_pdos.json'])
    def create_from_dict(self, request):
        json_file = request.param
        expected_spectrum = get_expected_spectrum1dcollection(json_file)
        spectrum = Spectrum1DCollection.from_dict(
            expected_spectrum.to_dict())
        return spectrum, expected_spectrum

    @pytest.fixture(params=[
        ('quartz', 'quartz-554-full.phonon_dos',
         'quartz_554_full_castep_adaptive_dos.json'),
        ('LZO', 'La2Zr2O7-222-full.phonon_dos',
         'lzo_222_full_castep_adaptive_dos.json')])
    def create_from_castep_phonon_dos(self, request):
        material, phonon_dos_file, json_file = request.param
        spec = Spectrum1DCollection.from_castep_phonon_dos(
            get_castep_path(material, phonon_dos_file))
        spec.y_data_unit = 'cm'
        expected_spec = get_expected_spectrum1dcollection(json_file)
        return spec, expected_spec

    @pytest.mark.parametrize(('spectrum_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_castep_phonon_dos')])
    def test_correct_object_creation(self, spectrum_creator):
        spectrum, expected_spectrum = spectrum_creator
        check_spectrum1dcollection(spectrum, expected_spectrum)

    @pytest.fixture(params=[
        ('x_data',
         get_expected_spectrum1dcollection('gan_bands.json').x_data.magnitude,
         TypeError),
        ('x_data',
         get_expected_spectrum1dcollection('gan_bands.json').x_data[:-1],
         ValueError),
        ('y_data',
         get_expected_spectrum1dcollection('gan_bands.json').y_data.magnitude,
         TypeError),
        ('y_data',
         get_expected_spectrum1dcollection('gan_bands.json').y_data[:, :-2],
         ValueError),
        ('y_data',
         get_expected_spectrum1dcollection('gan_bands.json').y_data[0, :].flatten(),
         ValueError),
        ('x_tick_labels',
         get_expected_spectrum1dcollection('gan_bands.json').x_tick_labels[0],
         TypeError),
        ('metadata',
         ['Not', 'a', 'dictionary'],
         TypeError),
        ('metadata',
         {'line_data': [{'label': 'Wrong number'}, {'label': 'Of Elements'}]},
         ValueError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_spectrum = get_expected_spectrum1dcollection('gan_bands.json')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_spectrum.to_constructor_args(
            **{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            Spectrum1DCollection(*faulty_args, **faulty_kwargs)

    @pytest.mark.parametrize(
        'input_spectra, expected_spectrum',
        [([get_spectrum1d(f'gan_bands_index_{i}.json') for i in range(2, 5)],
          get_spectrum1dcollection('gan_bands_index_2_5.json')),
         ([get_spectrum1d(f'methane_pdos_index_{i}.json') for i in range(1, 4)],
          get_spectrum1dcollection('methane_pdos_index_1_4.json'))
         ])
    def test_create_from_sequence(self, input_spectra, expected_spectrum):
        spectrum = Spectrum1DCollection.from_spectra(input_spectra)
        check_spectrum1dcollection(spectrum, expected_spectrum)

    @pytest.mark.parametrize(
        'input_metadata, expected_metadata',
        [([{},
           {'label': 'H3'},
           {'Another key': 'Anything'}
          ],
          {'line_data': [{}, {'label': 'H3'}, {'Another key': 'Anything'}]}
         ),
         ([{'desc': 'PDOS H2', 'int list': [1, 2, 3]},
           {'desc': 'PDOS H3', 'label': 'H3', 'int list': [1, 2, 3]},
           {'desc': 'PDOS', 'int list': [1, 2, 3]}
          ],
          {'int list': [1, 2, 3], 'line_data': [
               {'desc': 'PDOS H2'},
               {'desc': 'PDOS H3', 'label': 'H3'},
               {'desc': 'PDOS'}]}
         ),
         ([{'desc': 'methane PDOS'}, {}, {}],
          {'line_data': [{'desc': 'methane PDOS'}, {}, {}]}),
         ([{'desc': 'methane PDOS'},
           {'desc': 'methane PDOS'},
           {'desc': 'methane PDOS'}
          ],
          {'desc': 'methane PDOS'}
         )])
    def test_create_methane_pdos_from_sequence_metadata_handling(
            self, input_metadata, expected_metadata):
        spectra = [get_spectrum1d(
            f'methane_pdos_index_{i}.json') for i in range(1, 4)]
        for i, spec in enumerate(spectra):
            spec.metadata = input_metadata[i]
        expected_spectrum = get_spectrum1dcollection(
            'methane_pdos_index_1_4.json')
        expected_spectrum.metadata = expected_metadata
        spectrum = Spectrum1DCollection.from_spectra(spectra)
        check_spectrum1dcollection(spectrum, expected_spectrum)


    @pytest.mark.parametrize(
        'input_spectra, expected_error',
        [([], IndexError),
         ([get_spectrum1dcollection('gan_bands.json')], TypeError),
         ([f'gan_bands_index_{i}.json' for i in range(2, 5)], TypeError)])
    def test_faulty_create_from_sequence(self, input_spectra, expected_error):
        with pytest.raises(expected_error):
            Spectrum1DCollection.from_spectra(input_spectra)


@pytest.mark.unit
class TestSpectrum1DCollectionSerialisation:

    @pytest.fixture(params=[
        get_spectrum1dcollection('gan_bands.json'),
        get_spectrum1dcollection('methane_pdos.json')])
    def serialise_to_json_file(self, request, tmpdir):
        spectrum = request.param
        # Serialise
        output_file = str(tmpdir.join('tmp.test'))
        spectrum.to_json_file(output_file)
        # Deserialise
        deserialised_spectrum = (Spectrum1DCollection
                                 .from_json_file(output_file))
        return spectrum, deserialised_spectrum

    def test_serialise_to_file(self, serialise_to_json_file):
        spectrum, deserialised_spectrum = serialise_to_json_file
        check_spectrum1dcollection(spectrum, deserialised_spectrum)

    @pytest.fixture(params=[
        'gan_bands.json',
        'methane_pdos.json'])
    def serialise_to_dict(self, request):
        json_file = request.param
        spectrum = get_spectrum1dcollection(json_file)
        expected_spectrum = get_expected_spectrum1dcollection(json_file)
        # Convert to dict, then back to object to test
        spectrum_dict = spectrum.to_dict()
        spectrum_from_dict = Spectrum1DCollection.from_dict(spectrum_dict)
        return spectrum_from_dict, expected_spectrum

    def test_serialise_to_dict(self, serialise_to_dict):
        spectrum, expected_spectrum = serialise_to_dict
        check_spectrum1dcollection(spectrum, expected_spectrum)


@pytest.mark.unit
class TestSpectrum1DCollectionUnitConversion:

    @pytest.mark.parametrize('spectrum1d_file, attr, unit_val', [
        ('LZO_cut_dispersion.json', 'x_data', '1/bohr'),
        ('LZO_cut_dispersion.json', 'y_data', 'hartree')])
    def test_correct_unit_conversion(self, spectrum1d_file, attr, unit_val):
        spec1d = get_spectrum1dcollection(spectrum1d_file)
        check_unit_conversion(spec1d, attr, unit_val)

    @pytest.mark.parametrize('spectrum1d_file, unit_attr, unit_val, err', [
        ('LZO_cut_dispersion.json', 'x_data_unit', 'kg', ValueError),
        ('LZO_cut_dispersion.json', 'y_data_unit', 'mbarn', ValueError)])
    def test_incorrect_unit_conversion(self, spectrum1d_file, unit_attr,
                                       unit_val, err):
        spec1d = get_spectrum1dcollection(spectrum1d_file)
        with pytest.raises(err):
            setattr(spec1d, unit_attr, unit_val)


@pytest.mark.unit
class TestSpectrum1DCollectionSetters:

    @pytest.mark.parametrize('spectrum1d_file, attr, unit, scale', [
        ('LZO_cut_dispersion.json', 'x_data', '1/cm', 3.),
        ('LZO_cut_dispersion.json', 'x_data', '1/angstrom', 2.),
        ('LZO_cut_dispersion.json', 'y_data', 'THz', 3.),
        ('LZO_cut_dispersion.json', 'y_data', 'meV', 2.),
        ])
    def test_setter_correct_units(self, spectrum1d_file, attr,
                                  unit, scale):
        spec1d = get_spectrum1dcollection(spectrum1d_file)
        check_property_setters(spec1d, attr, unit, scale)

    @pytest.mark.parametrize('spectrum1d_file, attr, unit, err', [
        ('LZO_cut_dispersion.json', 'x_data', 'kg', ValueError),
        ('LZO_cut_dispersion.json', 'y_data', 'mbarn', ValueError)])
    def test_incorrect_unit_conversion(self, spectrum1d_file, attr,
                                       unit, err):
        spec1d = get_spectrum1dcollection(spectrum1d_file)
        new_attr = getattr(spec1d, attr).magnitude*ureg(unit)
        with pytest.raises(err):
            setattr(spec1d, attr, new_attr)


@pytest.mark.unit
class TestSpectrum1DCollectionIndexAccess:
    @pytest.mark.parametrize(
        'spectrum, index, expected_spectrum1d',
        [(get_spectrum1dcollection('gan_bands.json'), 2,
          get_expected_spectrum1d('gan_bands_index_2.json')),
         (get_spectrum1dcollection('gan_bands.json'), -4,
          get_expected_spectrum1d('gan_bands_index_2.json')),
         (get_spectrum1dcollection('methane_pdos.json'), 3,
          get_expected_spectrum1d('methane_pdos_index_3.json')),
         (get_spectrum1dcollection('quartz_dos_collection.json'), 2,
          get_expected_spectrum1d('quartz_dos_collection_index_2.json')),
          ])
    def test_index_individual(self, spectrum, index, expected_spectrum1d):
        extracted_spectrum1d = spectrum[index]
        check_spectrum1d(extracted_spectrum1d, expected_spectrum1d)

    @pytest.mark.parametrize(
        'spectrum, index, expected_spectrum',
        [(get_spectrum1dcollection('gan_bands.json'), slice(2, 5),
          get_expected_spectrum1dcollection('gan_bands_index_2_5.json')),
         (get_spectrum1dcollection('gan_bands.json'), slice(-4, -1),
          get_expected_spectrum1dcollection('gan_bands_index_2_5.json')),
         (get_spectrum1dcollection('methane_pdos.json'), slice(1, 4),
          get_expected_spectrum1dcollection('methane_pdos_index_1_4.json')),
         ])
    def test_index_slice(self, spectrum, index, expected_spectrum):
        extracted_spectrum = spectrum[index]
        check_spectrum1dcollection(extracted_spectrum, expected_spectrum)

    @pytest.mark.parametrize(
        'spectrum, index, expected_error',
        [(get_spectrum1dcollection('gan_bands.json'),
          '1', TypeError),
         (get_spectrum1dcollection('gan_bands.json'),
          6, IndexError),
         (get_spectrum1dcollection('gan_bands.json'),
          slice(2, 6), IndexError)])
    def test_index_errors(self, spectrum, index, expected_error):
        with pytest.raises(expected_error):
            spectrum[index]


@pytest.mark.unit
class TestSpectrum1DCollectionMethods:
    @pytest.mark.parametrize(
        'spectrum, split_kwargs, expected_spectra',
        [(get_spectrum1dcollection('methane_pdos.json'),
          {'indices': [50]},
          [get_spectrum1dcollection(f'methane_pdos_split50_{i}.json')
              for i in range(2)])])
    def test_split(self, spectrum, split_kwargs, expected_spectra):
        spectra = spectrum.split(**split_kwargs)
        for split, expected_split in zip(spectra, expected_spectra):
            check_spectrum1dcollection(split, expected_split)

    @pytest.mark.parametrize(
        'spectrum_file, expected_bin_edges_file, expected_units', [
            ('gan_bands.json',
             'gan_bands_bin_edges.npy',
             ureg('1/angstrom')),
            ('quartz_dos_collection.json',
             'quartz_dos_collection_bin_edges.npy',
             ureg('meV'))])
    def test_get_bin_edges(self, spectrum_file, expected_bin_edges_file,
                           expected_units):
        spec = get_spectrum1dcollection(spectrum_file)
        bin_edges = spec.get_bin_edges()
        expected_bin_edges = np.load(
                os.path.join(get_spectrum_dir(), expected_bin_edges_file))
        assert bin_edges.units == expected_units
        assert_allclose(bin_edges.magnitude, expected_bin_edges)

    @pytest.mark.parametrize(
        'spectrum_file, expected_bin_centres_file, expected_units', [
            ('gan_bands.json',
             'gan_bands_bin_centres.npy',
             ureg('1/angstrom')),
            ('quartz_dos_collection.json',
             'quartz_dos_collection_bin_centres.npy',
             ureg('meV'))])
    def test_get_bin_centres(self, spectrum_file, expected_bin_centres_file,
                             expected_units):
        spec = get_spectrum1dcollection(spectrum_file)
        bin_centres = spec.get_bin_centres()
        expected_bin_centres = np.load(
                os.path.join(get_spectrum_dir(), expected_bin_centres_file))
        assert bin_centres.units == expected_units
        assert_allclose(bin_centres.magnitude,
                        expected_bin_centres)

    def test_get_bin_edges_with_invalid_data_shape_raises_value_error(self):
        spec = get_spectrum1dcollection('gan_bands.json')
        spec._y_data = spec._y_data[:, :51]
        with pytest.raises(ValueError):
            spec.get_bin_edges()

    def test_get_bin_centres_with_invalid_data_shape_raises_value_error(self):
        spec = get_spectrum1dcollection('quartz_dos_collection.json')
        spec._x_data = spec._x_data[:31]
        with pytest.raises(ValueError):
            spec.get_bin_centres()

    # Check the same answer is obtained broadening Spectrum1DCollection,
    # and broadening each spectrum individually
    @pytest.mark.parametrize(
        'spectrum_file, width, shape', [
            ('methane_pdos.json', 10*ureg('1/cm'), 'lorentz'),
            ('quartz_dos_collection.json', 2*ureg('meV'), 'gauss'),
            ('quartz_dos_collection.json', 15*ureg('1/cm'), 'gauss')])
    def test_broaden(self, spectrum_file, width, shape):
        spec_col = get_spectrum1dcollection(spectrum_file)
        broadened_spec_col = spec_col.broaden(width, shape)
        for i, spec in enumerate(spec_col):
            broadened_spec1d = spec.broaden(width, shape)
            check_spectrum1d(broadened_spec_col[i],
                             broadened_spec1d)

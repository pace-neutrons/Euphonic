import os
import json

import numpy as np
from numpy.testing import assert_allclose
from pint import Quantity
import pytest

from euphonic import ureg
from euphonic.spectra import Spectrum1D, Spectrum1DCollection
from tests_and_analysis.test.utils import get_data_path, check_unit_conversion

from .test_spectrum1d import (get_spectrum1d, get_expected_spectrum1d,
                              check_spectrum1d)

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

def get_spectrum_dir():
    return os.path.join(get_data_path(), 'spectrum1dcollection')


def get_json_file(json_filename):
    return os.path.join(get_spectrum_dir(), json_filename)


def get_spectrum(json_filename):
    return Spectrum1DCollection.from_json_file(get_json_file(json_filename))


def get_expected_spectrum(json_filename):
    return ExpectedSpectrum1DCollection(get_json_file(json_filename))

def check_spectrum1dcollection(actual_spectrum, expected_spectrum):

    assert_allclose(actual_spectrum.x_data.magnitude,
                        expected_spectrum.x_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum.x_data.units
            == expected_spectrum.x_data.units)

    assert_allclose(actual_spectrum.y_data.magnitude,
                        expected_spectrum.y_data.magnitude,
                        atol=np.finfo(np.float64).eps)
    assert (actual_spectrum.y_data.units
            == expected_spectrum.y_data.units)

    if expected_spectrum.x_tick_labels is None:
        assert actual_spectrum.x_tick_labels is None
    else:
        assert (actual_spectrum.x_tick_labels
                == expected_spectrum.x_tick_labels)

@pytest.mark.unit
class TestSpectrum1DCollectionCreation:
    @pytest.fixture(params=[
        get_expected_spectrum('gan_bands.json'),
        get_expected_spectrum('methane_pdos.json')])
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
        expected_spectrum = get_expected_spectrum(json_file)
        spectrum = Spectrum1DCollection.from_json_file(
            get_json_file(json_file))
        return spectrum, expected_spectrum

    @pytest.fixture(params=[
        'gan_bands.json',
        'methane_pdos.json'])
    def create_from_dict(self, request):
        json_file = request.param
        expected_spectrum = get_expected_spectrum(json_file)
        spectrum = Spectrum1DCollection.from_dict(
            expected_spectrum.to_dict())
        return spectrum, expected_spectrum

    @pytest.mark.parametrize(('spectrum_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_dict')])
    def test_correct_object_creation(self, spectrum_creator):
        spectrum, expected_spectrum = spectrum_creator
        check_spectrum1dcollection(spectrum, expected_spectrum)

    @pytest.fixture(params=[
        ('x_data',
         get_expected_spectrum('gan_bands.json').x_data.magnitude,
         TypeError),
        ('x_data',
         get_expected_spectrum('gan_bands.json').x_data[:-1],
         ValueError),
        ('y_data',
         get_expected_spectrum('gan_bands.json').y_data.magnitude,
         TypeError),
        ('y_data',
         get_expected_spectrum('gan_bands.json').y_data[:,:-2],
         ValueError),
        ('y_data',
         get_expected_spectrum('gan_bands.json').y_data[0,:].flatten(),
         ValueError),
        ('x_tick_labels',
         get_expected_spectrum('gan_bands.json').x_tick_labels[0],
         TypeError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_spectrum = get_expected_spectrum('gan_bands.json')
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
          get_spectrum('gan_bands_index_2_5.json'))
          ])
    def test_create_from_sequence(self, input_spectra, expected_spectrum):
        spectrum = Spectrum1DCollection.from_spectra(input_spectra)
        check_spectrum1dcollection(spectrum, expected_spectrum)

    @pytest.mark.parametrize(
        'input_spectra, expected_error',
        [([], IndexError),
         ([get_spectrum('gan_bands.json')], TypeError),
         ([f'gan_bands_index_{i}.json' for i in range(2, 5)], TypeError)])
    def test_faulty_create_from_sequence(self, input_spectra, expected_error):
        with pytest.raises(expected_error):
            Spectrum1DCollection.from_spectra(input_spectra)


@pytest.mark.unit
class TestSpectrum1DCollectionSerialisation:

    @pytest.fixture(params=[
        get_spectrum('gan_bands.json'),
        get_spectrum('methane_pdos.json')])
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
        spectrum = get_spectrum(json_file)
        expected_spectrum = get_expected_spectrum(json_file)
        # Convert to dict, then back to object to test
        spectrum_dict = spectrum.to_dict()
        spectrum_from_dict = Spectrum1DCollection.from_dict(spectrum_dict)
        return spectrum_from_dict, expected_spectrum

    def test_serialise_to_dict(self, serialise_to_dict):
        spectrum, expected_spectrum = serialise_to_dict
        check_spectrum1dcollection(spectrum, expected_spectrum)


@pytest.mark.unit
class TestSpectrum1DCollectionIndexAccess:
    @pytest.mark.parametrize(
        'spectrum, index, expected_spectrum1d',
        [(get_spectrum('gan_bands.json'), 2,
          get_expected_spectrum1d('gan_bands_index_2.json')),
         (get_spectrum('gan_bands.json'), -4,
          get_expected_spectrum1d('gan_bands_index_2.json'))])
    def test_index_individual(self, spectrum, index, expected_spectrum1d):
        extracted_spectrum1d = spectrum[index]
        check_spectrum1d(extracted_spectrum1d, expected_spectrum1d)

    @pytest.mark.parametrize(
        'spectrum, index, expected_spectrum',
        [(get_spectrum('gan_bands.json'), slice(2, 5),
          get_expected_spectrum('gan_bands_index_2_5.json')),
         (get_spectrum('gan_bands.json'), slice(-4, -1),
          get_expected_spectrum('gan_bands_index_2_5.json'))])
    def test_index_slice(self, spectrum, index, expected_spectrum):
        extracted_spectrum = spectrum[index]
        check_spectrum1dcollection(extracted_spectrum, expected_spectrum)

    @pytest.mark.parametrize(
        'spectrum, index, expected_error',
        [(get_spectrum('gan_bands.json'), '1', TypeError),
         (get_spectrum('gan_bands.json'), 6, IndexError),
         (get_spectrum('gan_bands.json'), slice(2, 6), IndexError)])
    def test_index_errors(self, spectrum, index, expected_error):
        with pytest.raises(expected_error):
            spectrum[index]

    @pytest.mark.parametrize(
        'spectrum, split_kwargs, expected_spectra',
        [(get_spectrum('methane_pdos.json'), {'indices': [50]},
          [get_spectrum(f'methane_pdos_split50_{i}.json') for i in range(2)])])
    def test_split(self, spectrum, split_kwargs, expected_spectra):
        spectra = spectrum.split(**split_kwargs)
        for split, expected_split in zip(spectra, expected_spectra):
            check_spectrum1dcollection(split, expected_split)

def assert_quantity_allclose(q1, q2):
    assert str(q1.units) == str(q2.units)
    # assert q1.units == q2.units   # why doesn't this work?
    assert_allclose(q1.magnitude, q2.magnitude)


def sample_xy_data(n_spectra=1, n_energies=10, n_labels=2):
    x_data = np.linspace(0, 1, n_energies)* ureg('1/angstrom')
    y_data = np.random.random((n_spectra, n_energies)) * ureg('meV')
    x_label_positions = np.random.choice(range(n_energies),
                                         size=n_labels, replace=False)
    x_tick_labels = [(i, str(i)) for i in x_label_positions]
    return (x_data, y_data, x_tick_labels)

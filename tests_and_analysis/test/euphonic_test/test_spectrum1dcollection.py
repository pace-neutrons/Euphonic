import json

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest_lazy_fixtures import lf as lazy_fixture

from euphonic import ureg
from euphonic.spectra import Spectrum1DCollection
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    check_property_setters,
    check_spectrum1d,
    check_unit_conversion,
    get_expected_spectrum1d,
    get_spectrum1d,
)
from tests_and_analysis.test.utils import (
    check_spectrum_text_header,
    get_castep_path,
    get_data_path,
    get_spectrum_from_text,
)


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
            return (list(map(tuple, self.data['x_tick_labels']))
                    if 'x_tick_labels' in self.data else None)

    @property
    def metadata(self):
        return self.data.get('metadata', {})

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


def get_spectrum_path(*subpaths):
    return get_data_path('spectrum1dcollection', *subpaths)


def get_spectrum1dcollection(json_filename):
    return Spectrum1DCollection.from_json_file(
        get_spectrum_path(json_filename))


def get_expected_spectrum1dcollection(json_filename):
    return ExpectedSpectrum1DCollection(get_spectrum_path(json_filename))


def check_spectrum1dcollection(actual_spectrum,
                               expected_spectrum,
                               y_atol=None):

    if y_atol is None:
        y_atol = np.finfo(np.float64).eps

    assert (actual_spectrum.x_data.units
            == expected_spectrum.x_data.units)
    assert_allclose(actual_spectrum.x_data.magnitude,
                    expected_spectrum.x_data.magnitude,
                    atol=np.finfo(np.float64).eps)

    assert (actual_spectrum.y_data.units
            == expected_spectrum.y_data.units)
    assert_allclose(actual_spectrum.y_data.magnitude,
                    expected_spectrum.y_data.magnitude,
                    atol=y_atol)

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
            get_spectrum_path(json_file))
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
        expected_spec = get_expected_spectrum1dcollection(json_file)
        return spec, expected_spec

    @pytest.mark.parametrize(('spectrum_creator'), [
        lazy_fixture('create_from_constructor'),
        lazy_fixture('create_from_json'),
        lazy_fixture('create_from_dict'),
        lazy_fixture('create_from_castep_phonon_dos')])
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
         get_expected_spectrum1dcollection('gan_bands.json')
         .y_data[0, :].flatten(),
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
         ([get_spectrum1d(f'methane_pdos_index_{i}.json')
           for i in range(1, 4)],
          get_spectrum1dcollection('methane_pdos_index_1_4.json')),
         ([get_spectrum1d('xsq_spectrum1d.json')],
          get_spectrum1dcollection('xsq_from_single_spectrum1d.json')),
         ])
    def test_create_from_sequence(self, input_spectra, expected_spectrum):
        spectrum = Spectrum1DCollection.from_spectra(input_spectra)
        check_spectrum1dcollection(spectrum, expected_spectrum)

    bad_sequences = [
        [
            ['NotASpectrum', get_spectrum1d('gan_bands_index_2.json')],
            TypeError,
        ],
        [
            [get_spectrum1d('gan_bands_index_2.json'), 'NotASpectrum'],
            TypeError,
        ],
        [
            [
                get_spectrum1d('gan_bands_index_2.json'),
                get_spectrum1d('methane_pdos_index_1.json'),
            ],
            ValueError,
        ],
        [
            [
                get_spectrum1d('gan_bands_index_2.json'),
                get_spectrum1d('gan_bands_index_3.json'),
            ],
            ValueError,
        ],
        [
            [
                get_spectrum1d('gan_bands_index_2.json'),
                get_spectrum1d('gan_bands_index_3.json'),
            ],
            ValueError,
        ],
    ]
    # Item 3: Make x_data inconsistent
    bad_sequences[3][0][1].x_data = bad_sequences[3][0][0].x_data * 2.
    # Item 4: Make x_tick_labels inconsistent
    bad_sequences[4][0][1].x_tick_labels = [(0, '$\\Gamma$'), (54, 'X')]

    @pytest.mark.parametrize(
        'input_spectra, expected_error', bad_sequences,
    )
    def test_create_from_bad_sequence(self, input_spectra, expected_error):
        with pytest.raises(expected_error):
            Spectrum1DCollection.from_spectra(input_spectra)

    def test_unsafe_from_sequence(self):
        """Ensure that unsafe from_spectra doesn't check units"""

        spec1 = get_spectrum1d('gan_bands_index_2.json')
        spec2 = get_spectrum1d('gan_bands_index_3.json')

        spec1.x_data_unit = '1/angstrom'
        spec2.x_data_unit = '1/mm'

        with pytest.raises(ValueError):
            Spectrum1DCollection.from_spectra([spec1, spec2])

        Spectrum1DCollection.from_spectra([spec1, spec2], unsafe=True)

    @pytest.mark.parametrize(
        'input_metadata, expected_metadata',
        [([{},
           {'label': 'H3'},
           {'Another key': 'Anything'},
          ],
          {'line_data': [{}, {'label': 'H3'}, {'Another key': 'Anything'}]},
         ),
         ([{'desc': 'PDOS H2', 'common_key_value': 3},
           {'desc': 'PDOS H3', 'label': 'H3', 'common_key_value': 3},
           {'desc': 'PDOS', 'common_key_value': 3},
          ],
          {'common_key_value': 3, 'line_data': [
               {'desc': 'PDOS H2'},
               {'desc': 'PDOS H3', 'label': 'H3'},
               {'desc': 'PDOS'}]},
         ),
         ([{'desc': 'methane PDOS'}, {}, {}],
          {'line_data': [{'desc': 'methane PDOS'}, {}, {}]}),
         ([{'desc': 'methane PDOS'},
           {'desc': 'methane PDOS'},
           {'desc': 'methane PDOS'},
          ],
          {'desc': 'methane PDOS'},
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


class TestSpectrum1DCollectionSerialisation:

    # Note that when writing .text there must be the same number of
    # x_data and y_data points so bin centres will be used, this and
    # using fmt may mean the output spectrum is slightly different.
    # x_tick_labels will also be lost
    @pytest.mark.parametrize('in_json, out_json, kwargs', [
        ('quartz_666_coh_pdos.json',
         'quartz_666_coh_pdos_from_text_file_fmt.json',
         {'fmt': '%.2f'}),
        ('gan_bands.json', 'gan_bands_from_text_file.json', {}),
        ('methane_pdos.json', 'methane_pdos_from_text_file_fmt.json',
         {'fmt': ['%.4f'] + ['%.2f']*5})])
    def test_serialise_to_text_file(self, in_json, out_json, kwargs, tmp_path):
        spectrum = get_spectrum1dcollection(in_json)
        # Serialise
        output_file = tmp_path / 'tmp.test'
        spectrum.to_text_file(output_file, **kwargs)
        # Deserialise
        deserialised_spectrum = get_spectrum_from_text(
            output_file, is_collection=True)
        expected_deserialised_spectrum = get_spectrum1dcollection(out_json)
        check_spectrum_text_header(output_file)
        check_spectrum1dcollection(
            expected_deserialised_spectrum, deserialised_spectrum)

    @pytest.mark.parametrize('spectrum', [
        get_spectrum1dcollection('gan_bands.json'),
        get_spectrum1dcollection('methane_pdos.json')])
    def test_serialise_to_json_file(self, spectrum, tmp_path):
        # Serialise
        output_file = tmp_path / 'tmp.test'
        spectrum.to_json_file(output_file)
        # Deserialise
        deserialised_spectrum = (Spectrum1DCollection
                                 .from_json_file(output_file))
        check_spectrum1dcollection(spectrum, deserialised_spectrum)

    @pytest.mark.parametrize('json_file', [
        'gan_bands.json',
        'methane_pdos.json'])
    def test_serialise_to_dict(self, json_file):
        spectrum = get_spectrum1dcollection(json_file)
        expected_spectrum = get_expected_spectrum1dcollection(json_file)
        # Convert to dict, then back to object to test
        spectrum_dict = spectrum.to_dict()
        spectrum_from_dict = Spectrum1DCollection.from_dict(spectrum_dict)
        check_spectrum1dcollection(spectrum_from_dict, expected_spectrum)


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
        'spectrum, index, expected_spectrum',
        [(get_spectrum1dcollection('gan_bands.json'), np.arange(2, 5),
          get_expected_spectrum1dcollection('gan_bands_index_2_5.json')),
         (get_spectrum1dcollection('gan_bands.json'), [2, 3, 4],
          get_expected_spectrum1dcollection('gan_bands_index_2_5.json')),
         ])
    def test_index_sequence(self, spectrum, index, expected_spectrum):
        extracted_spectrum = spectrum[index]
        check_spectrum1dcollection(extracted_spectrum, expected_spectrum)

    @pytest.mark.parametrize(
        'spectrum, index, expected_error',
        [(get_spectrum1dcollection('gan_bands.json'),
          '1', TypeError),
         (get_spectrum1dcollection('gan_bands.json'),
          np.arange(4.0), TypeError),
         (get_spectrum1dcollection('gan_bands.json'),
          6, IndexError),
         (get_spectrum1dcollection('gan_bands.json'),
          slice(2, 6), IndexError)])
    def test_index_errors(self, spectrum, index, expected_error):
        with pytest.raises(expected_error):
            spectrum[index]


class TestSpectrum1DCollectionMethods:
    @pytest.mark.parametrize(
        'spectrum, split_kwargs, expected_spectra',
        [(get_spectrum1dcollection('methane_pdos.json'),
          {'indices': [50]},
          [get_spectrum1dcollection(f'methane_pdos_split50_{i}.json')
              for i in range(2)])])
    def test_split(self, spectrum, split_kwargs, expected_spectra):
        spectra = spectrum.split(**split_kwargs)
        for split, expected_split in zip(
                spectra, expected_spectra, strict=True):
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
                get_spectrum_path(expected_bin_edges_file))
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
                get_spectrum_path(expected_bin_centres_file))
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

    def test_broaden_bad_width(self):
        spec_col = get_spectrum1dcollection('methane_pdos.json')
        with pytest.raises(TypeError,
                           match='x_width must be a Quantity or Callable'):
            spec_col.broaden(x_width=4)

    def test_variable_broadening(self):
        """Check variable broadening is consistent for collections"""
        def width_function(x):
            return x.to('meV') * 3 + 1.*ureg('meV')

        spectra = get_spectrum1dcollection('quartz_666_coh_pdos.json')
        individually_broadened = [spectrum.broaden(width_function)
                                  for spectrum in spectra]
        collection_broadened = spectra.broaden(width_function)

        for spec1, spec2 in zip(
                individually_broadened, collection_broadened, strict=True):
            check_spectrum1d(spec1, spec2)

    @pytest.mark.parametrize(
        'spectrum_file, summed_spectrum_file', [
            ('quartz_666_pdos.json', 'quartz_666_dos.json'),
        ])
    def test_sum(self, spectrum_file, summed_spectrum_file):
        spec_col = get_spectrum1dcollection(spectrum_file)
        summed_spec = spec_col.sum()
        expected_summed_spec = get_expected_spectrum1d(summed_spectrum_file)
        check_spectrum1d(summed_spec, expected_summed_spec)

    @pytest.mark.parametrize(
        'spectrum_file, group_by_args, grouped_spectrum_file', [
            ('La2Zr2O7_666_incoh_pdos.json', ('species',),
             'La2Zr2O7_666_incoh_species_pdos.json'),
            ('La2Zr2O7_666_coh_pdos.json', ('species',),
             'La2Zr2O7_666_coh_species_pdos.json'),
            ])
    def test_group_by(self, spectrum_file, group_by_args,
                      grouped_spectrum_file):
        spec_col = get_spectrum1dcollection(spectrum_file)
        grouped_spec = spec_col.group_by(*group_by_args)
        expected_grouped_spec = get_expected_spectrum1dcollection(
            grouped_spectrum_file)
        check_spectrum1dcollection(grouped_spec, expected_grouped_spec)

    fake_metadata = {'top_level_key': 'something', 'top_level_int': 10,
                     'line_data': [
                         {'sample': 0, 'inst': 'LET', 'index': 10,
                          'other_data': 'misc'},
                         {'sample': 2, 'inst': 'MARI', 'index': 5,
                          'some_other_data': 5},
                         {'sample': 2, 'inst': 'MARI', 'index': 10,
                          'some_other_data': 5},
                         {'sample': 0, 'inst': 'MAPS', 'index': 10,
                          'other_data': 'another_value'},
                         {'sample': 0, 'inst': 'TOSCA', 'index': 7},
                         {'sample': 0, 'inst': 'TOSCA', 'index': 10},
                         {'sample': 2, 'inst': 'LET', 'index': 10},
                         {'sample': 2, 'inst': 'LET', 'index': 5},
                         {'sample': 1, 'inst': 'LET', 'index': 10}]}

    @pytest.mark.parametrize(
        'spectrum_file, metadata, group_by_args, expected_metadata', [
            ('quartz_666_pdos.json', fake_metadata,
             ('sample', 'inst'),
             {'top_level_key': 'something', 'top_level_int': 10,
              'line_data': [
                  {'sample': 0, 'inst': 'LET', 'index': 10,
                   'other_data': 'misc'},
                  {'sample': 2, 'inst': 'MARI', 'some_other_data': 5},
                  {'sample': 0, 'inst': 'MAPS', 'index': 10,
                   'other_data': 'another_value'},
                  {'sample': 0, 'inst': 'TOSCA'},
                  {'sample': 2, 'inst': 'LET'},
                  {'sample': 1, 'inst': 'LET', 'index': 10}]}),
            ('quartz_666_pdos.json', fake_metadata,
             ('index',),
             {'top_level_key': 'something', 'top_level_int': 10,
              'line_data': [
                  {'index': 10},
                  {'sample': 2, 'index': 5},
                  {'sample': 0, 'inst': 'TOSCA', 'index': 7}]}),
            ])
    def test_group_by_fake_metadata(
            self, spectrum_file, metadata, group_by_args, expected_metadata):
        spec_col = get_spectrum1dcollection(spectrum_file)
        spec_col.metadata = metadata
        grouped_spec = spec_col.group_by(*group_by_args)
        assert  grouped_spec.metadata == expected_metadata

    # Self-consistency test, allows us to test more metadata combinations
    # without generating more test files
    @pytest.mark.parametrize(
        'spectrum_file, metadata, group_by_args, group_indices', [
            ('quartz_666_pdos.json', fake_metadata, ('sample', 'inst'),
             [[0], [1, 2], [3], [4,5], [6,7], [8]]),
            ('quartz_666_pdos.json', fake_metadata, ('index',),
             [[0, 2, 3, 5, 6, 8], [1, 7], [4]])])
    def test_group_by_same_as_index_and_sum_with_fake_metadata(
            self, spectrum_file, metadata, group_by_args, group_indices):
        spec_col = get_spectrum1dcollection(spectrum_file)
        spec_col.metadata = metadata
        grouped_spec = spec_col.group_by(*group_by_args)
        for i, spec in enumerate(grouped_spec):
            check_spectrum1d(spec,
                             spec_col[group_indices[i]].sum())

    @pytest.mark.parametrize(
        'spectrum_file, select_kwargs, selected_spectrum_file', [
            ('methane_pdos.json', {'label': ['H2', 'H3', 'H4']},
             'methane_pdos_index_1_4.json'),
            ('La2Zr2O7_666_coh_incoh_species_append_pdos.json',
             {'weighting': 'coherent'},
             'La2Zr2O7_666_coh_species_pdos.json'),
            ])
    def test_select(self, spectrum_file, select_kwargs,
                    selected_spectrum_file):
        spec_col = get_spectrum1dcollection(spectrum_file)
        selected_spec = spec_col.select(**select_kwargs)
        expected_selected_spec = get_expected_spectrum1dcollection(
            selected_spectrum_file)
        check_spectrum1dcollection(selected_spec, expected_selected_spec)


    # Self-consistency test, allows us to test more combinations
    # without generating more test files
    @pytest.mark.parametrize(
        'spectrum_file, select_kwargs, expected_indices', [
            ('La2Zr2O7_666_coh_incoh_species_append_pdos.json',
             {'weighting': 'coherent'},
             [0, 1, 2]),
            ('La2Zr2O7_666_coh_incoh_species_append_pdos.json',
             {'weighting': 'incoherent', 'species': ['O', 'La']},
             [3, 5]),
            ('La2Zr2O7_666_coh_incoh_species_append_pdos.json',
             {'weighting': 'incoherent', 'species': 'O'},
             [3]),
            ('methane_pdos.json',
             {'desc': 'Methane PDOS', 'label': 'H3'}, [2]),
            ])
    def test_select_same_as_indexing(self, spectrum_file, select_kwargs,
                                     expected_indices):
        spec_col = get_spectrum1dcollection(spectrum_file)
        selected_spec = spec_col.select(**select_kwargs)
        expected_selected_spec = spec_col[expected_indices]
        check_spectrum1dcollection(selected_spec, expected_selected_spec)

    @pytest.mark.parametrize('spectrum_file, metadata, select_kwargs',
            [('quartz_666_pdos.json', fake_metadata,
             {'inst': ['LET', 'TOSCA'], 'index': [7, 5]})])
    def test_select_with_not_all_matching_combinations_doesnt_raise_key_error(
            self, spectrum_file, metadata, select_kwargs):
        spec_col = get_spectrum1dcollection(spectrum_file)
        spec_col.metadata = metadata
        spec_col.select(**select_kwargs)

    @pytest.mark.parametrize('spectrum_file, metadata, select_kwargs',
            [('quartz_666_pdos.json', fake_metadata,
             {'inst': ['LET', 'TOSCA'], 'index': [4, 6]})])
    def test_select_with_no_matches_raises_value_error(
            self, spectrum_file, metadata, select_kwargs):
        spec_col = get_spectrum1dcollection(spectrum_file)
        spec_col.metadata = metadata
        with pytest.raises(ValueError):
            spec_col.select(**select_kwargs)

    @pytest.mark.parametrize(
        'spectrum_file, other_spectrum_file, expected_spectrum_file', [
            ('La2Zr2O7_666_coh_species_pdos.json',
             'La2Zr2O7_666_incoh_species_pdos.json',
             'La2Zr2O7_666_coh_incoh_species_append_pdos.json'),
            ])
    def test_add(self, spectrum_file, other_spectrum_file,
                 expected_spectrum_file):
        spec_col = get_spectrum1dcollection(spectrum_file)
        other_spec_col = get_spectrum1dcollection(other_spectrum_file)
        added_spec = spec_col + other_spec_col
        expected_added_spec = get_expected_spectrum1dcollection(
            expected_spectrum_file)
        check_spectrum1dcollection(added_spec, expected_added_spec)

    def test_mul(self):
        spec = get_spectrum1dcollection('gan_bands.json')

        for i, spec1d in enumerate(spec):
            check_spectrum1d(spec1d * 2.,
                         (spec * 2.)[i])

        check_spectrum1dcollection(spec, (spec * 2.) * 0.5)

        with pytest.raises(AssertionError):
            check_spectrum1dcollection(spec, spec * 2.)

    def test_copy(self):
        spec = get_spectrum1dcollection('gan_bands.json')
        spec.metadata = {'Test': 'item', 'int': 1}

        spec_copy = spec.copy()
        # Copy should be same
        check_spectrum1dcollection(spec, spec_copy)

        # Until data is edited
        spec_copy._y_data *= 2
        with pytest.raises(AssertionError):
            check_spectrum1dcollection(spec, spec_copy)

        spec_copy = spec.copy()
        spec_copy._x_data *= 2
        with pytest.raises(AssertionError):
            check_spectrum1dcollection(spec, spec_copy)

        spec_copy = spec.copy()
        spec_copy.x_tick_labels = [(1, 'different')]
        with pytest.raises(AssertionError):
            check_spectrum1dcollection(spec, spec_copy)

        spec_copy = spec.copy()
        spec_copy.metadata['Test'] = spec_copy.metadata['Test'].upper()
        with pytest.raises(AssertionError):
            check_spectrum1dcollection(spec, spec_copy)

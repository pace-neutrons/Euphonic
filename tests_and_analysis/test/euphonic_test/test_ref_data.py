import json

import pytest
from pint import UndefinedUnitError

from euphonic import ureg
from euphonic.util import get_reference_data


class TestReferenceData:

    def test_bad_collection(self):
        with pytest.raises(ValueError):
            get_reference_data(collection='not-a-real-label')

    def test_bad_physical_property(self):
        with pytest.raises(ValueError):
            get_reference_data(physical_property='not-a-real-property')

    def test_ref_scattering_length(self):
        data = get_reference_data(
            collection='Sears1992',
            physical_property='coherent_scattering_length')

        assert data['Ba'].units == ureg('fm')
        assert data['Hg'].magnitude == pytest.approx(12.692)
        assert data['Sm'].magnitude == pytest.approx(complex(0.80, -1.65))

    def test_ref_cross_section(self):
        data = get_reference_data(collection='BlueBook',
                                  physical_property='coherent_cross_section')

        assert data['Ca'].units == ureg('barn')
        assert data['Cl'].magnitude == pytest.approx(11.5257)

    @staticmethod
    def _dump_data(data, tmpdir, filename):
        filename = tmpdir.join(filename)
        with open(filename, 'wt') as fd:
            json.dump(data, fd)
        return str(filename)

    @pytest.fixture
    def animal_data(self):
        return {'reference': 'Some estimates',
                'physical_property':
                {'size': {'cat': 0.5, 'dog': 0.8, '__units__': 'meter'},
                 'weight': {'cat': 4., 'dog': 20., '__units__': 'kilogram'}}}

    @pytest.mark.parametrize('physical_property', ['size', 'weight'])
    def test_custom_file(self, tmpdir, animal_data, physical_property):

        filename = self._dump_data(animal_data, tmpdir, 'good_data.json')

        loaded_data = get_reference_data(collection=filename,
                                         physical_property=physical_property)

        animal_properties = animal_data['physical_property']
        for animal in 'cat', 'dog':
            assert loaded_data[animal].magnitude == pytest.approx(
                animal_properties[physical_property][animal])
            assert (loaded_data[animal].units
                    == animal_properties[physical_property]['__units__'])

    def test_custom_file_no_units(self, tmpdir):
        test_data_no_units = {'physical_property':
                              {'some_property': {'H': 1}}}
        filename = self._dump_data(test_data_no_units, tmpdir,
                                   'data_no_units.json')

        with pytest.raises(ValueError):
            get_reference_data(collection=filename,
                               physical_property='some_property')

    def test_custom_file_unknown_units(self, tmpdir):
        test_data_unknown_units = {
            'physical_property': {
                'some_property': {'H': 1, '__units__': 'nonsense'}}}
        filename = self._dump_data(test_data_unknown_units, tmpdir,
                                   'data_unknown_units.json')

        with pytest.raises(ValueError):
            get_reference_data(collection=filename,
                               physical_property='some_property')

    def test_custom_file_wrong_property(self, tmpdir):
        test_data_wrong_property = {
            'physical_property': {
                'size': {'cat': 1, '__units__': 'meter'},
                'weight': {'cat': 4, '__units__': 'kg'}}}
        filename = self._dump_data(test_data_wrong_property, tmpdir,
                                   'data_with_size_and_weight.json')

        with pytest.raises(ValueError):
            get_reference_data(collection=filename,
                               physical_property='wrong_property')

    def test_custom_file_no_physical_property(self, tmpdir):
        test_data_no_physical_property = {
            'not_physical_property': {
                'size': {'cat': 1, '__units__': 'meter'},
                'weight': {'cat': 4, '__units__': 'kg'}}}
        filename = self._dump_data(test_data_no_physical_property, tmpdir,
                                   'no_physical_property_data.json')

        with pytest.raises(AttributeError):
            get_reference_data(collection=filename,
                               physical_property='size')

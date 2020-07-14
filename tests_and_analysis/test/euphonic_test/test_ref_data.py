import pytest

from euphonic import ureg
from euphonic.util import get_reference_data

@pytest.mark.unit
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

        assert data['Ba'].units == ureg['fm']
        assert data['Hg'].magnitude == pytest.approx(12.692)
        assert data['Sm'].magnitude == pytest.approx(complex(0.80, -1.65))

    def test_ref_cross_section(self):
        data = get_reference_data(collection='BlueBook',
                                  physical_property='coherent_cross_section')

        assert data['Ca'].units == ureg['barn']
        assert data['Cl'].magnitude == pytest.approx(11.5257)

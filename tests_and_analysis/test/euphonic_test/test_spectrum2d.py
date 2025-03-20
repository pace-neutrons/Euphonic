import json

import numpy as np
from numpy.polynomial import Polynomial
import numpy.testing as npt
from pint import Quantity
import pytest

from euphonic import ureg
from euphonic.spectra import Spectrum2D, apply_kinematic_constraints
from tests_and_analysis.test.utils import (
    check_json_metadata,
    check_property_setters,
    check_unit_conversion,
    does_not_raise,
    get_data_path,
)


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
        return None

    @property
    def metadata(self):
        if 'metadata' in self.data.keys():
            return self.data['metadata']
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


def get_spectrum2d_path(*subpaths):
    return get_data_path('spectrum2d', *subpaths)


def get_spectrum2d(json_filename):
    return Spectrum2D.from_json_file(get_spectrum2d_path(json_filename))


def get_expected_spectrum2d(json_filename):
    return ExpectedSpectrum2D(get_spectrum2d_path(json_filename))


def check_spectrum2d(actual_spectrum2d, expected_spectrum2d, equal_nan=False,
                     z_atol=np.finfo(np.float64).eps):

    assert (actual_spectrum2d.x_data.units
            == expected_spectrum2d.x_data.units)
    npt.assert_allclose(actual_spectrum2d.x_data.magnitude,
                        expected_spectrum2d.x_data.magnitude,
                        atol=np.finfo(np.float64).eps)

    assert (actual_spectrum2d.y_data.units
            == expected_spectrum2d.y_data.units)
    npt.assert_allclose(actual_spectrum2d.y_data.magnitude,
                        expected_spectrum2d.y_data.magnitude,
                        atol=np.finfo(np.float64).eps)

    assert (actual_spectrum2d.z_data.units
            == expected_spectrum2d.z_data.units)
    npt.assert_allclose(actual_spectrum2d.z_data.magnitude,
                        expected_spectrum2d.z_data.magnitude,
                        atol=z_atol,
                        equal_nan=equal_nan)

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
            get_spectrum2d_path(json_file))
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
        'args, spectrum2d_file, broadened_spectrum2d_file, context', [
            (({'x_width': 0.1*ureg('1/angstrom'), 'method': 'convolve'}),
             'quartz_bandstructure_sqw.json',
             'quartz_bandstructure_0.1ang_xbroaden_sqw.json',
             pytest.warns(UserWarning,
                           match="x_data bin widths are not equal")),
            (({'y_width': 2*ureg('meV')}),
             'quartz_bandstructure_sqw.json',
             'quartz_bandstructure_2meV_ybroaden_sqw.json',
             does_not_raise()),
            (({'x_width': 0.1*ureg('1/angstrom'), 'y_width': 2*ureg('meV'),
               'method': 'convolve'}),
             'quartz_bandstructure_sqw.json',
             'quartz_bandstructure_2meV_0.1ang_xybroaden_sqw.json',
             pytest.warns(UserWarning,
                           match="x_data bin widths are not equal")),
            (({'x_width': 0.1*ureg('1/angstrom'), 'y_width': 2*ureg('meV'),
                'shape': 'lorentz', 'method': 'convolve'}),
             'quartz_bandstructure_sqw.json',
             'quartz_bandstructure_xybroaden_lorentz_sqw.json',
             pytest.warns(UserWarning,
                           match="x_data bin widths are not equal")),
            (({'x_width': 0.2*ureg('1/angstrom'), 'y_width': 1.5*ureg('meV'),
               'shape': 'gauss'}),
             'lzo_57L_bragg_sqw.json',
             'lzo_57L_1.5meV_0.1ang_gauss_sqw.json',
             pytest.warns(UserWarning,
                          match="Not all x-axis bins are the same width")),
            (({'x_width': 0.2*ureg('1/angstrom'), 'y_width': 1.5*ureg('meV'),
               'shape': 'lorentz'}),
             'lzo_57L_bragg_sqw.json',
             'lzo_57L_1.5meV_0.1ang_lorentz_sqw.json',
             pytest.warns(UserWarning,
                          match="Not all x-axis bins are the same width")),
            (({'x_width': (lambda x: np.polyval([0.2, -0.5],
                                                x.to('1/nm').magnitude
                                                ) * ureg('1/nm')),
               'y_width': (lambda y: np.polyval([0., -0.4, 3.],
                                                y.to('J').magnitude
                                                ) * ureg('J')),
               'width_fit': 'cubic'},
              'synthetic_x.json', 'synthetic_x_poly_broadened.json',
              does_not_raise())),
            (({'x_width': (lambda x: np.polyval([0.2, -0.5],
                                                x.to('1/nm').magnitude
                                                ) * ureg('1/nm')),
               'y_width': (lambda y: np.polyval([0., -0.4, 3.],
                                                y.to('J').magnitude
                                                ) * ureg('J')),
               'width_fit': 'cheby-log'},
              'synthetic_x.json', 'synthetic_x_poly_broadened_cheby.json',
              does_not_raise())),
                             ])
    def test_broaden(self, args, spectrum2d_file, broadened_spectrum2d_file,
                     context):
        spec2d = get_spectrum2d(spectrum2d_file)
        expected_broadened_spec2d = get_spectrum2d(broadened_spectrum2d_file)
        with context:
            broadened_spec2d = spec2d.broaden(**args)
        check_spectrum2d(broadened_spec2d, expected_broadened_spec2d)

    def test_broaden_invalid_shape_raises_value_error(self):
        spec2d = get_spectrum2d('quartz_bandstructure_sqw.json')
        with pytest.raises(ValueError), pytest.warns(UserWarning):
                spec2d.broaden(x_width=1*ureg('meV'), shape='unknown')

    def test_broaden_uneven_bins_deprecation_raises_value_error(self):
        spec2d = get_spectrum2d('La2Zr2O7_cut_sqw_uneven_bins.json')
        with pytest.raises(ValueError):
            spec2d.broaden(y_width=1*ureg('meV'))

    @pytest.mark.parametrize('unequal_bin_json, unequal_axes', [
        ('La2Zr2O7_cut_sqw_uneven_bins.json', ['y_data']),
        ('quartz_bandstructure_dos_map.json', ['x_data']),
        ('quartz_bandstructure_dos_map_uneven_bins.json', ['x_data', 'y_data'])
        ])
    def test_broaden_uneven_bins_and_explicit_convolve_warns(
            self, unequal_bin_json, unequal_axes):
        spec2d = get_spectrum2d(unequal_bin_json)
        with pytest.warns(UserWarning) as record:
            spec2d.broaden(x_width=1*ureg('1/angstrom'),
                           y_width=1*ureg('meV'),
                           method='convolve')
        # Check warning message includes information about which axis is
        # uneven
        for unequal_ax in unequal_axes:
            assert unequal_ax in record[-1].message.args[0]

    def test_variable_broaden_spectrum2d(self):
        """Check variable broadening is consistent with fixed-width method"""
        spectrum = get_spectrum2d('quartz_bandstructure_sqw.json')

        sigma = 2 * ureg('meV')
        fwhm = 2.3548200450309493 * sigma

        def sigma_function(x):
            poly = Polynomial([sigma.magnitude, 0, 0.])
            return poly(x.to(sigma.units).magnitude) * sigma.units

        fixed_broad_y = spectrum.broaden(y_width=fwhm)
        variable_broad_y = spectrum.broaden(
            y_width=sigma_function, width_convention='std')

        check_spectrum2d(fixed_broad_y, variable_broad_y)

        # Force regular x bins so x-broadening is allowed
        spectrum.x_data = np.linspace(
            1, 10,len(spectrum.get_bin_edges())
        ) * spectrum.get_bin_edges().units
        fixed_broad_x = spectrum.broaden(x_width=fwhm)
        variable_broad_x = spectrum.broaden(
            x_width=sigma_function, width_convention='std')
        check_spectrum2d(fixed_broad_x, variable_broad_x, z_atol=1e-10)

    @pytest.mark.parametrize(
        'spectrum2d_file, ax, expected_bin_edges', [
            ('example_spectrum2d.json', 'x',
             np.array([0., 0.5, 1.5, 2.5, 3.5, 4.5, 6., 8., 10., 12.,
                       13.])*ureg('meter')),
            ('example_xbin_edges_spectrum2d.json', 'x',
             np.array([0., 1., 2., 3., 4., 5., 7., 9., 11.,
                       13.])*ureg('Hz')),
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
                       12.])*ureg('Hz')),
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

    def test_broaden_consistent_across_bin_definition(self):
        """Check broadening results are the same for both bin conventions"""
        spec2d_edges = get_spectrum2d('quartz_bandstructure_sqw.json')

        spec2d_centred = Spectrum2D(spec2d_edges.x_data,
                                    spec2d_edges.get_bin_centres(bin_ax='y'),
                                    spec2d_edges.z_data)

        # Check we really are using both conventions here
        assert len(spec2d_edges.y_data) == len(spec2d_centred.y_data) + 1

        fixed_sigma = 1. * ureg("meV")
        npt.assert_allclose(spec2d_edges.broaden(y_width=fixed_sigma).z_data,
                            spec2d_centred.broaden(y_width=fixed_sigma).z_data)

        def sigma_func(energy: Quantity) -> Quantity:
            return np.ones_like(energy) * ureg("meV")
        npt.assert_allclose(spec2d_edges.broaden(y_width=sigma_func).z_data,
                            spec2d_centred.broaden(y_width=sigma_func).z_data)

    def test_mul(self):
        spec = get_spectrum2d('example_spectrum2d.json')

        npt.assert_allclose(spec.z_data.magnitude * 2.,
                            (spec * 2.).z_data.magnitude)

        check_spectrum2d(spec, (spec * 2.) * 0.5)

        with pytest.raises(AssertionError):
            check_spectrum2d(spec, spec * 2.)

    def test_copy(self):
        spec = get_spectrum2d('example_spectrum2d.json')

        spec_copy = spec.copy()
        # Copy should be same
        check_spectrum2d(spec, spec_copy)

        # Until data is edited
        for attr in '_x_data', '_y_data', '_z_data':
            setattr(spec_copy, attr, getattr(spec, attr) * 2)

            with pytest.raises(AssertionError):
                check_spectrum2d(spec, spec_copy)

            spec_copy = spec.copy()

        spec_copy = spec.copy()
        spec_copy.x_tick_labels = [(1, 'different')]
        with pytest.raises(AssertionError):
            check_spectrum2d(spec, spec_copy)

        spec_copy = spec.copy()
        spec_copy.metadata['description'] = \
            spec_copy.metadata['description'].upper()
        with pytest.raises(AssertionError):
            check_spectrum2d(spec, spec_copy)


class TestKinematicAngles:

    @pytest.mark.parametrize(
        'angle_range, expected',
        [((0, np.pi / 2), (1, 0)),
         ((np.pi / 2, 0), (1, 0)),
         ((0, 1.4 * np.pi), (1, -1)),
         ((-2.25 * np.pi, -2.5 * np.pi), (np.sqrt(2) / 2, 0))
         ])
    def test_cos_range(self, angle_range, expected):
        from euphonic.spectra.base import _get_cos_range
        cos_limits = _get_cos_range(angle_range)
        assert cos_limits == pytest.approx(expected)


class TestKinematicConstraints:

    @pytest.mark.parametrize(
        'kwargs, spectrum2d_file, constrained_file',
        [({'e_i': 300 * ureg('1/cm')},
          'NaCl_band_yaml_dos_map.json',
          'NaCl_constrained_ei_300_cm.json'),
         ({'e_i': 100 * ureg('1/cm'), 'angle_range': (20, 275)},
          'NaCl_band_yaml_dos_map.json',
          'NaCl_constrained_ei_100_cm_angle_20_275.json'),
         ({'e_i': 30 * ureg('meV'), 'angle_range': (-20, 30)},
          'NaCl_band_yaml_dos_map.json',
          'NaCl_constrained_ei_30_meV_angle_-20_30.json'),
         ({'e_f': 32 * ureg('1/cm'), 'angle_range': (45., 135.)},
          'NaCl_band_yaml_dos_map.json',
          'NaCl_constrained_ef_32_cm_angle_45_135.json'),
         ({'e_f': 32 * ureg('1/cm'), 'angle_range': (-135., -45.)},
          'NaCl_band_yaml_dos_map.json',
          'NaCl_constrained_ef_32_cm_angle_45_135.json')
         ])
    def test_kinematic_constraints(self, kwargs,
                                   spectrum2d_file, constrained_file):
        spec2d = get_spectrum2d(spectrum2d_file)
        ref_spec2d = get_spectrum2d(constrained_file)

        constrained_spec2d = apply_kinematic_constraints(spec2d, **kwargs)
        check_spectrum2d(constrained_spec2d, ref_spec2d, equal_nan=True)

    @pytest.mark.parametrize('json_file, kwargs, expected',
                             [('NaCl_band_yaml_dos_map.json',
                               {'e_i': 10 * ureg('1/cm'),
                                'e_f': 20 * ureg('1/cm')},
                               ValueError),
                              ('NaCl_band_yaml_dos_map.json', {}, ValueError),
                              ('example_spectrum2d.json',
                               {'e_i': 20 * ureg('1/cm')},
                               ValueError),
                              ('example_xbin_edges_spectrum2d.json',
                               {'e_i': 20 * ureg('1/cm')},
                               ValueError)])
    def test_kinematic_constraints_invalid(self, json_file, kwargs, expected):
        spec2d = get_spectrum2d(json_file)

        with pytest.raises(expected):
            apply_kinematic_constraints(spec2d, **kwargs)

import numpy as np
from pint import Quantity
from euphonic import ureg, Crystal, Spectrum2D
from euphonic.util import (get_qpoint_labels, _calc_abscissa, _bose_factor,
                           _check_constructor_inputs)
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)

class StructureFactor(object):
    """
    Stores the structure factor calculated per q-point and per phonon
    mode

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    n_qpts : int
        Number of q-points in the object
    qpts : (n_qpts, 3) float ndarray
        Q-point coordinates, in fractional coordinates of the reciprocal
        lattice
    frequencies : (n_qpts, 3*crystal.n_atoms) float Quantity
        Phonon frequencies per q-point and mode
    structure_factors : (n_qpts, 3*crystal.n_atoms) float Quantity
        Structure factor per q-point and mode
    temperature : float Quantity or None
        The temperature used to calculate any temperature-dependent
        parts of the structure factor (e.g. Debye-Waller, Bose
        population factor). None if no temperature-dependent effects
        have been applied
    """

    def __init__(self, crystal, qpts, frequencies, structure_factors,
                 temperature=None):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        qpts : (n_qpts, 3) float ndarray
            Q-point coordinates, in fractional coordinates of the
            reciprocal lattice
        frequencies: (n_qpts, 3*crystal.n_atoms) float Quantity
            Phonon frequencies per q-point and mode
        structure_factors: (n_qpts, 3*crystal.n_atoms) float Quantity
            Structure factor per q-point and mode
        temperature : float Quantity or None
            The temperature used to calculate any temperature-dependent
            parts of the structure factor (e.g. Debye-Waller, Bose
            population factor). None if no temperature-dependent effects
            have been applied
        """
        _check_constructor_inputs(
            [crystal, qpts], [Crystal, np.ndarray], [(), (-1, 3)],
            ['crystal', 'qpts'])
        n_at = crystal.n_atoms
        n_qpts = len(qpts)
        _check_constructor_inputs(
            [frequencies, structure_factors, temperature],
            [Quantity, Quantity, [Quantity, type(None)]],
            [(n_qpts, 3*n_at), (n_qpts, 3*n_at), ()],
            ['frequencies', 'structure_factors', 'temperature'])
        self.crystal = crystal
        self.qpts = qpts
        self.n_qpts = len(qpts)
        self._frequencies = frequencies.to(
            ureg.INTERNAL_ENERGY_UNIT).magnitude
        self.frequencies_unit = str(frequencies.units)
        self._structure_factors = structure_factors.to(
            ureg.INTERNAL_LENGTH_UNIT**2).magnitude
        self.structure_factors_unit = str(structure_factors.units)

        if temperature is not None:
            self._temperature = temperature.to(
                ureg.INTERNAL_TEMPERATURE_UNIT).magnitude
            self.temperature_unit = str(temperature.units)
        else:
            self._temperature = None
            self.temperature_unit = str(ureg.INTERNAL_TEMPERATURE_UNIT)

    @property
    def frequencies(self):
        return self._frequencies*ureg(
            'INTERNAL_ENERGY_UNIT').to(self.frequencies_unit)

    @property
    def structure_factors(self):
        return self._structure_factors*ureg('INTERNAL_LENGTH_UNIT**2').to(
            self.structure_factors_unit)

    @property
    def temperature(self):
        if self._temperature is not None:
            # See https://pint.readthedocs.io/en/latest/nonmult.html
            return Quantity(self._temperature,
                            ureg('INTERNAL_TEMPERATURE_UNIT')).to(
                                self.temperature_unit)
        else:
            return None

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name in ['frequencies_unit', 'structure_factors_unit',
                        'temperature_unit']:
                ureg(getattr(self, name)).to(value)
        super(StructureFactor, self).__setattr__(name, value)

    def calculate_sqw_map(self, e_bins, calc_bose=True, temperature=None):
        """
        Bin the structure factor in energy and apply the Bose population
        factor to produce a a S(Q,w) map

        Parameters
        ----------
        e_bins : (n_e_bins + 1,) float Quantity
            The energy bin edges
        calc_bose : boolean, optional
            Whether to calculate and apply the Bose population factor
        temperature : float Quantity, optional
            The temperature to use to calculate the Bose factor. Is only
            required if StructureFactor.temperature = None, otherwise
            the temperature stored in StructureFactor will be used

        Returns
        -------
        sqw_map : Spectrum2D
            A spectrum containing the q-point bins on the x-axis, energy
            bins on the y-axis and scattering intensities on the z-axis

        Raises
        ------
        ValueError
            If a temperature is provided and isn't consistent with the
            temperature in the StructureFactor object

        Notes
        -----
        StructureFactor.structure_factors is defined as
        :math:`|F(Q, \\nu)|^2` per unit cell. To create an
        :math:`S(Q,\\omega)` map, it is binned in energy and the Bose
        factor is applied [1]_:

        .. math::

          S(Q, \\omega) = |F(Q, \\nu)|^2
          (n_\\nu+\\frac{1}{2}\\pm\\frac{1}{2})
          \\delta(\\omega\\mp\\omega_{q\\nu})

        :math:`n_\\nu` is the Bose-Einstein distribution:

        .. math::

          n_\\nu = \\frac{1}{e^{\\frac{\\hbar\\omega_\\nu}{k_{B}T}} - 1}

        .. [1] M.T. Dove, Structure and Dynamics, Oxford University Press, Oxford, 2003, 225-226
        """
        if calc_bose:
            if not self.temperature is None:
                if (not temperature is None
                        and not np.isclose(temperature, self.temperature)):
                    raise ValueError((
                        'Temperature provided to calculate_sqw_map '
                        '({:~P}) is not consistent with the temperature'
                        'stored in StructureFactor ({:~P})'.format(
                            temperature, self.temperature)))
                temperature = self.temperature

        # Convert units
        freqs = self._frequencies
        e_bins_internal = e_bins.to('INTERNAL_ENERGY_UNIT').magnitude

        # Create initial sqw_map with an extra an energy bin either
        # side, for any branches that fall outside the energy bin range
        sqw_map = np.zeros((self.n_qpts, len(e_bins) + 1))
        sf = self._structure_factors
        if calc_bose and temperature is not None:
            p_intensity = sf*_bose_factor(freqs, temperature.to('K').magnitude)
            n_intensity = sf*_bose_factor(
                -freqs, temperature.to('K').magnitude)
        else:
            p_intensity = sf
            n_intensity = sf

        p_bin = np.digitize(freqs, e_bins_internal)
        n_bin = np.digitize(-freqs, e_bins_internal)

        # Sum intensities into bins
        first_index = np.transpose(
            np.tile(range(self.n_qpts), (3*self.crystal.n_atoms, 1)))
        np.add.at(sqw_map, (first_index, p_bin), p_intensity)
        np.add.at(sqw_map, (first_index, n_bin), n_intensity)
        # Exclude values outside ebin range
        sqw_map = sqw_map[:, 1:-1]*ureg('INTERNAL_LENGTH_UNIT**2').to(
            self.structure_factors_unit)

        abscissa = _calc_abscissa(self.crystal, self.qpts)
        # Calculate q-space ticks and labels
        x_tick_labels = get_qpoint_labels(self.crystal, self.qpts)

        return Spectrum2D(abscissa, e_bins, sqw_map,
                          x_tick_labels=x_tick_labels)

    def to_dict(self):
        """
        Convert to a dictionary. See StructureFactor.from_dict for
        details on keys/values

        Returns
        -------
        dict
        """
        dout = _obj_to_dict(self, ['crystal', 'n_qpts', 'qpts', 'frequencies',
                                   'structure_factors', 'temperature'])
        return dout

    def to_json_file(self, filename):
        """
        Write to a JSON file. JSON fields are equivalent to
        StructureFactor.from_dict keys

        Parameters
        ----------
        filename : str
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a StructureFactor object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'crystal': dict, see Crystal.from_dict
            - 'qpts': (n_qpts, 3) float ndarray
            - 'frequencies': (n_qpts, 3*crystal.n_atoms) float ndarray
            - 'frequencies_unit': str
            - 'structure_factors': (n_qpts, 3*crystal.n_atoms) float ndarray
            - 'structure_factors_unit': str

            There are also the following optional keys:

            - 'temperature': float
            - 'temperature_unit': str

        Returns
        -------
        StructureFactor
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(
            d, quantities=['frequencies', 'structure_factors', 'temperature'],
            optional=['temperature'])
        return StructureFactor(crystal, d['qpts'], d['frequencies'],
                               d['structure_factors'], d['temperature'])

    @classmethod
    def from_json_file(cls, filename):
        """
        Read from a JSON file. See StructureFactor.from_dict for required
        fields

        Parameters
        ----------
        filename : str
            The file to read from

        Returns
        -------
        StructureFactor
        """
        return _obj_from_json_file(cls, filename)

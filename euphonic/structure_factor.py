from typing import Optional, TypeVar, Dict, Any

import numpy as np

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import _obj_to_dict, _process_dict
from euphonic.util import get_qpoint_labels, _calc_abscissa
from euphonic import (ureg, Quantity, Crystal, QpointFrequencies, Spectrum1D,
                      Spectrum2D)


T = TypeVar('T', bound='StructureFactor')


class NoTemperatureError(Exception):
    pass


class StructureFactor(QpointFrequencies):
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
    weights : (n_qpts,) float ndarray
        The weight for each q-point
    temperature : float Quantity or None
        The temperature used to calculate any temperature-dependent
        parts of the structure factor (e.g. Debye-Waller, Bose
        population factor). None if no temperature-dependent effects
        have been applied
    """
    def __init__(self, crystal: Crystal, qpts: np.ndarray,
                 frequencies: Quantity, structure_factors: Quantity,
                 weights: Optional[np.ndarray] = None,
                 temperature: Optional[Quantity] = None) -> None:
        """
        Parameters
        ----------
        crystal
            Lattice and atom information
        qpts
            Shape (n_qpts, 3) float ndarray. Q-point coordinates, in
            fractional coordinates of the reciprocal lattice
        frequencies
            Shape (n_qpts, 3*crystal.n_atoms) float Quantity. Phonon
            frequencies per q-point and mode
        structure_factors
            Shape (n_qpts, 3*crystal.n_atoms) float Quantity. Structure
            factor per q-point and mode
        weights
            Shape (n_qpts,) float ndarray. The weight for each q-point.
            If None, equal weights are assumed
        temperature
            Scalar float Quantity. The temperature used to calculate any
            temperature-dependent parts of the structure factor (e.g.
            Debye-Waller, Bose population factor). None if no
            temperature-dependent effects have been applied
        """
        super().__init__(crystal, qpts, frequencies, weights)
        n_at = crystal.n_atoms
        n_qpts = len(qpts)
        # Check freqs axis 1 shape here - QpointFrequencies doesn't
        # enforce that the number of modes = 3*(number of atoms)
        _check_constructor_inputs(
            [frequencies, structure_factors, temperature],
            [Quantity, Quantity, [Quantity, type(None)]],
            [(n_qpts, 3*n_at), (n_qpts, 3*n_at), ()],
            ['frequencies', 'structure_factors', 'temperature'])
        self._structure_factors = structure_factors.to(
            ureg.bohr**2).magnitude
        self.structure_factors_unit = str(structure_factors.units)

        if temperature is not None:
            self._temperature = temperature.to(ureg.K).magnitude
            self.temperature_unit = str(temperature.units)
        else:
            self._temperature = None
            self.temperature_unit = str(ureg.K)

    @property
    def structure_factors(self):
        return self._structure_factors*ureg('bohr**2').to(
            self.structure_factors_unit)

    @property
    def temperature(self):
        if self._temperature is not None:
            # See https://pint.readthedocs.io/en/latest/nonmult.html
            return Quantity(self._temperature,
                            ureg('K')).to(self.temperature_unit)
        else:
            return None

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['frequencies_unit', 'structure_factors_unit',
                                'temperature_unit'])
        super(StructureFactor, self).__setattr__(name, value)

    def calculate_1d_average(self,
                             e_bins: Quantity,
                             calc_bose: bool = True,
                             temperature: Optional[Quantity] = None,
                             weights: Optional[np.ndarray] = None
                             ) -> Spectrum1D:
        """Bin structure factor in energy, flattening q to produce 1D spectrum

        Bose population factor may be applied. The main purpose of this
        function is to produce a powder-averaged spectrum.

        Parameters
        ----------
        e_bins
            Shape (n_e_bins + 1,) float Quantity. The energy bin edges
        calc_bose
            Whether to calculate and apply the Bose population factor
        temperature
            Scalar float Quantity. The temperature to use to calculate the Bose
            factor. Is only required if StructureFactor.temperature = None,
            otherwise the temperature stored in StructureFactor will be used
        weights
            Dimensionless weights to be applied in averaging, the same
            length as qpts. If no weights are provided the weights
            contained in StructureFactor are used. For details of how
            this argument is interpreted see docs for
            :func:`numpy.average`

        Returns
        -------
        s_w
            1-D neutron scattering spectrum, averaged over all sampled q-points
        """
        if weights is None:
            weights = self.weights

        sqw_map = self._bose_corrected_structure_factor(
            e_bins, calc_bose=calc_bose, temperature=temperature)

        spectrum = np.average(
            sqw_map.magnitude, axis=0, weights=weights)*sqw_map.units
        return Spectrum1D(e_bins, spectrum)

    def calculate_sqw_map(self,
                          e_bins: Quantity,
                          calc_bose: bool = True,
                          temperature: Optional[Quantity] = None
                          ) -> Spectrum2D:
        """
        Bin the structure factor in energy and apply the Bose population
        factor to produce a a S(Q,w) map

        Parameters
        ----------
        e_bins
            Shape (n_e_bins + 1,) float Quantity. The energy bin edges
        calc_bose
            Whether to calculate and apply the Bose population factor
        temperature
            The temperature to use to calculate the Bose factor. Is only
            required if StructureFactor.temperature = None, otherwise
            the temperature stored in StructureFactor will be used

        Returns
        -------
        sqw_map
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

        .. [1] M.T. Dove, Structure and Dynamics, Oxford University Press,
               Oxford, 2003, 225-226
        """

        sqw_map = self._bose_corrected_structure_factor(
            e_bins, calc_bose=calc_bose, temperature=temperature)

        abscissa = _calc_abscissa(self.crystal.reciprocal_cell(), self.qpts)
        # Calculate q-space ticks and labels
        x_tick_labels = get_qpoint_labels(self.qpts,
                                          cell=self.crystal.to_spglib_cell())

        return Spectrum2D(abscissa, e_bins, sqw_map,
                          x_tick_labels=x_tick_labels)

    def _bose_corrected_structure_factor(self, e_bins: Quantity,
                                         calc_bose: bool = True,
                                         temperature: Optional[Quantity] = None
                                         ) -> Quantity:
        """Bin structure factor in energy, return (Bose-populated) array

        Parameters
        ----------
        e_bins
            Shape (n_e_bins + 1,) float Quantity. The energy bin edges
        calc_bose
            Whether to calculate and apply the Bose population factor
        temperature
            Temperature used to calculate the Bose factor. This is only
            required if StructureFactor.temperature = None, otherwise
            the temperature stored in StructureFactor will be used.

        Returns
        -------
        intensities
            Scattering intensities as array over (qpt, energy)
        """
        # Convert units
        freqs = self._frequencies
        e_bins_internal = e_bins.to('hartree').magnitude

        # Create initial sqw_map with an extra an energy bin either
        # side, for any branches that fall outside the energy bin range
        sqw_map = np.zeros((self.n_qpts, len(e_bins) + 1))
        sf = self._structure_factors

        p_intensity = sf
        n_intensity = sf
        if calc_bose:
            try:
                bose = self._bose_factor(temperature)
                p_intensity = (1 + bose)*p_intensity
                n_intensity = bose*n_intensity
            except NoTemperatureError:
                pass

        p_bin = np.digitize(freqs, e_bins_internal)
        n_bin = np.digitize(-freqs, e_bins_internal)

        # Sum intensities into bins
        first_index = np.transpose(
            np.tile(range(self.n_qpts), (3*self.crystal.n_atoms, 1)))
        np.add.at(sqw_map, (first_index, p_bin), p_intensity)
        np.add.at(sqw_map, (first_index, n_bin), n_intensity)
        # Exclude values outside ebin range
        sqw_map = sqw_map[:, 1:-1]*ureg('bohr**2').to(
            self.structure_factors_unit)

        return sqw_map

    def _bose_factor(self, temperature: Optional[Quantity] = None):
        """
        Calculate the Bose factor for the frequencies stored in
        StructureFactor

        Parameters
        ----------
        temperature
            Temperature used to calculate the Bose factor. This is only
            required if StructureFactor.temperature = None, otherwise
            the temperature stored in StructureFactor will be used.

        Returns
        -------
        bose
            Shape (n_qpts, 3*n_atoms) float ndarray. The Bose factor

        Raises
        ------
        ValueError
            If a temperature is provided and isn't consistent with the
            temperature in the StructureFactor object
        NoTemperatureError
            If a temperature isn't provided there is no temperature in
            the StructureFactor object
        """
        if self.temperature is not None:
            if (temperature is not None
                    and not np.isclose(temperature.to('K').magnitude,
                                       self.temperature.to('K').magnitude)):
                raise ValueError((
                    'Temperature provided to calculate the Bose factor '
                    '({:~P}) is not consistent with the temperature '
                    'stored in StructureFactor ({:~P})'.format(
                        temperature, self.temperature)))
            temperature = self.temperature
        if temperature is None:
            raise NoTemperatureError(
                'When calculating the Bose factor, no temperature was '
                'provided, and no temperature could be found in '
                'StructureFactor')
        kB = (1*ureg.k).to('E_h/K').magnitude
        temp = temperature.to('K').magnitude
        bose = np.zeros(self._frequencies.shape)
        if temperature > 0:
            bose = 1/(np.exp(np.absolute(self._frequencies)/(kB*temp)) - 1)
        else:
            bose = 0
        return bose

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See StructureFactor.from_dict for
        details on keys/values
        """
        dout = _obj_to_dict(self, ['crystal', 'n_qpts', 'qpts', 'frequencies',
                                   'structure_factors', 'weights',
                                   'temperature'])
        return dout

    def to_qpoint_frequencies(self) -> QpointFrequencies:
        """
        Create a QpointFrequencies object
        """
        return QpointFrequencies(
            self.crystal, self.qpts, self.frequencies, self.weights)

    @classmethod
    def from_dict(cls: T, d: Dict[str, Any]) -> T:
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

            - 'weights': (n_qpts,) float ndarray
            - 'temperature': float
            - 'temperature_unit': str
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(
            d, quantities=['frequencies', 'structure_factors', 'temperature'],
            optional=['weights', 'temperature'])
        return cls(crystal, d['qpts'], d['frequencies'],
                   d['structure_factors'], d['weights'],
                   d['temperature'])

    @classmethod
    def from_castep(cls):
        ''
        raise AttributeError

    @classmethod
    def from_phonopy(cls):
        ''
        raise AttributeError

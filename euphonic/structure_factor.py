import numpy as np
from euphonic import ureg, Spectrum2D
from euphonic.legacy_plot.dispersion import calc_abscissa, recip_space_labels
from euphonic.util import _bose_factor

class StructureFactor(object):
    """
    Stores the structure factor calculated per q-point and per phonon mode

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    n_qpts : int
        Number of q-points in the object
    qpts : (n_qpts, 3) float ndarray
        Q-point coordinates, in fractional coordinates of the reciprocal lattice
    frequencies : (n_qpts, 3*crystal.n_atoms) float Quantity
        Phonon frequencies per q-point and mode
    structure_factors : (n_qpts, 3*crystal.n_atoms) float Quantity
        Structure factor per q-point and mode
    temperature : float Quantity or None
        The temperature used to calculate any temperature-dependent parts of the
        structure factor (e.g. Debye-Waller, Bose population factor). None if
        no temperature-dependent effects have been applied
    """

    def __init__(self, crystal, qpts, frequencies, structure_factors,
                 temperature=None):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        qpts : (n_qpts, 3) float ndarray
            Q-point coordinates, in fractional coordinates of the reciprocal
            lattice
        frequencies: (n_qpts, 3*crystal.n_atoms) float Quantity
            Phonon frequencies per q-point and mode
        structure_factors: (n_qpts, 3*crystal.n_atoms) float Quantity
            Structure factor per q-point and mode
        """
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
            'INTERNAL_ENERGY_UNIT').to(self.frequencies_unit, 'spectroscopy')

    @property
    def structure_factors(self):
        return self._structure_factors*ureg('INTERNAL_LENGTH_UNIT**2').to(
            self.structure_factors_unit)

    @property
    def temperature(self):
        if self._temperature is not None:
            return self._temperature*ureg('INTERNAL_TEMPERATURE_UNIT').to(
                self.temperature_unit)
        else:
            return None

    def calculate_sqw_map(self, e_bins, calc_bose=True, temperature=None):
        """
        Bin the structure factor in energy to produce a a S(Q,w) map

        Parameters
        ----------
        e_bins : (n_e_bins + 1,) float Quantity
            The energy bin edges
        calc_bose : boolean, optional, default True
            Whether to calculate and apply the Bose population factor
        temperature : float Quantity, default None
            The temperature to use to calculate the Bose factor. Is only
            required if StructureFactor.temperature = None, otherwise the
            temperature stored in StructureFactor will be used

        Returns
        -------
        sqw_map : Spectrum2D
            A spectrum containing the q-point bins on the x-axis, energy bins on
            the y-axis and scattering intensities on the z-axis

        Raises
        ------
        ValueError
            If a temperature is provided and isn't consistent with the
            temperature in the StructureFactor object
        """
        if calc_bose:
            if not self.temperature is None:
                if (not temperature is None
                        and not np.isclose(temperature, self.temperature)):
                    raise ValueError((
                        'Temperature provided to calculate_sqw_map '
                        '({:~P}) is not consistent with the temperature stored '
                        'in StructureFactor ({:~P})'.format(
                            temperature, self.temperature)))
                temperature = self.temperature

        # Convert units
        freqs = self._frequencies
        e_bins_internal = e_bins.to('INTERNAL_ENERGY_UNIT').magnitude

        # Create initial sqw_map with an extra an energy bin either side, for
        # any branches that fall outside the energy bin range
        sqw_map = np.zeros((self.n_qpts, len(e_bins) + 1))
        sf = self._structure_factors
        if calc_bose and temperature is not None:
            p_intensity = sf*_bose_factor(freqs, temperature.to('K').magnitude)
            n_intensity = sf*_bose_factor(-freqs, temperature.to('K').magnitude)
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
        sqw_map = sqw_map[:, 1:-1]*ureg('INTERNAL_LENGTH_UNIT**2').to(
            self.structure_factors_unit)  # Exclude values outside ebin range

        # Calculate qbin edges
        recip = self.crystal.reciprocal_cell().to('1/angstrom').magnitude
        abscissa = calc_abscissa(self.qpts, recip)
        qmid = (abscissa[1:] + abscissa[:-1])/2
        qwidths = qmid + qmid[0]
        qbins = np.concatenate(
            ([0], qwidths, [2*qwidths[-1] - qwidths[-2]]))*ureg('1/angstrom')

        # Calculate q-space ticks and labels
        xlabels, qpts_with_labels = recip_space_labels(self.crystal, self.qpts)
        for i, label in enumerate(xlabels):
            if label == 'GAMMA':
                xlabels[i] = r'$\Gamma$'
        if np.all(xlabels == ''):
            xlabels = np.around(self.qpts[qpts_with_labels, :], decimals=2)

        x_tick_labels = list(zip(qpts_with_labels, xlabels))

        return Spectrum2D(qbins, e_bins, sqw_map, x_tick_labels=x_tick_labels)


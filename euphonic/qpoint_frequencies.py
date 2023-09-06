import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Type

import numpy as np

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic.readers import castep, phonopy
from euphonic.util import (_calc_abscissa, get_qpoint_labels)
from euphonic import (ureg, Crystal, Quantity, Spectrum1D,
                      Spectrum1DCollection, Spectrum2D)
from euphonic.broadening import ErrorFit, _width_interpolated_broadening


AdaptiveMethod = Literal['reference', 'fast']


class QpointFrequencies:
    """
    A class to read and store frequency data at q-points

    Attributes
    ----------
    crystal
        Lattice and atom information
    n_qpts
        Number of q-points in the object
    qpts
        Shape (n_qpts, 3) float ndarray. Q-point coordinates, in
        fractional coordinates of the reciprocal lattice
    frequencies
        Shape (n_qpts, n_branches) float Quantity in energy units.
        Frequencies per q-point and mode
    weights
        Shape (n_qpts,) float ndarray. The weight for each q-point
    """
    T = TypeVar('T', bound='QpointFrequencies')

    def __init__(self, crystal: Crystal, qpts: np.ndarray,
                 frequencies: Quantity,
                 weights: Optional[np.ndarray] = None) -> None:
        """
        Parameters
        ----------
        crystal
            Lattice and atom information
        qpts
            Shape (n_qpts, 3) float ndarray. Q-point coordinates
        frequencies
            Shape (n_qpts, n_branches) float Quantity in energy units.
            Frequencies per q-point and mode
        weights
            Shape (n_qpts,) float ndarray. The weight for each q-point.
            If None, equal weights are assumed
        """
        _check_constructor_inputs(
            [crystal, qpts], [Crystal, np.ndarray], [(), (-1, 3)],
            ['crystal', 'qpts'])
        n_qpts = len(qpts)
        # Unlike QpointPhononModes and StructureFactor, don't test the
        # frequencies shape against number of atoms in the crystal, as
        # we may only have the cell vectors
        _check_constructor_inputs(
            [frequencies, weights],
            [Quantity, [np.ndarray, type(None)]],
            [(n_qpts, -1), (n_qpts,)],
            ['frequencies', 'weights'])
        self.crystal = crystal
        self.qpts = qpts
        self.n_qpts = n_qpts
        self._frequencies = frequencies.to(ureg.hartree).magnitude
        self.frequencies_unit = str(frequencies.units)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.full(self.n_qpts, 1/self.n_qpts)

    @property
    def frequencies(self) -> Quantity:
        return self._frequencies*ureg('hartree').to(self.frequencies_unit)

    @frequencies.setter
    def frequencies(self, value: Quantity) -> None:
        self.frequencies_unit = str(value.units)
        self._frequencies = value.to('hartree').magnitude

    def __setattr__(self, name: str, value: Any) -> None:
        _check_unit_conversion(self, name, value,
                               ['frequencies_unit'])
        super(QpointFrequencies, self).__setattr__(name, value)

    def calculate_dos(
        self,
        dos_bins: Quantity,
        mode_widths: Optional[Quantity] = None,
        mode_widths_min: Quantity = Quantity(0.01, 'meV'),
        adaptive_method: AdaptiveMethod = 'reference',
        adaptive_error: float = 0.01,
        adaptive_error_fit: ErrorFit = 'cubic'
        ) -> Spectrum1D:
        """
        Calculates a density of states, in units of modes per atom per
        energy unit, such that the integrated area is equal to 3.

        Parameters
        ----------
        dos_bins
            Shape (n_e_bins + 1,) float Quantity in energy units. The
            energy bin edges to use for calculating the DOS
        mode_widths
            Shape (n_qpts, n_branches) float Quantity in energy units.
            The broadening width for each mode at each q-point, for
            adaptive broadening
        mode_widths_min
            Scalar float Quantity in energy units. Sets a lower limit on
            the mode widths, as mode widths of zero will result in
            infinitely sharp peaks
        adaptive_method
            String. Specifies whether to use slow, reference adaptive method or
            faster, approximate method.
        adaptive_error
            Scalar float. Acceptable error for gaussian approximations
            when using the fast adaptive method, defined as the absolute
            difference between the areas of the true and approximate gaussians
        adaptive_error_fit
            Select parametrisation of kernel width spacing to adaptive_error.
            'cheby-log' is recommended: for backward-compatibilty, 'cubic' is
            the default.

        Returns
        -------
        dos
            A spectrum containing the energy bins on the x-axis and DOS
            on the y-axis. The DOS is in units of modes per energy unit
            per atom, such that the integrated area is equal to 3.

        Notes
        -----

        The fast adaptive broadening method reduces computation time by
        reducing the number of Gaussian functions that have to be evaluated.
        Broadening kernels are only evaulated for a subset of mode width values
        with intermediate values approximated using interpolation.

        The ``adaptive_error`` keyword argument is used to determine how many
        broadening kernels are computed exactly. The more exact kernels are
        used, the more accurate the Gaussian approximations will be, but
        computation time will also be increased.

        """
        dos = self._calculate_dos(dos_bins, mode_widths=mode_widths,
                                  mode_widths_min=mode_widths_min,
                                  adaptive_method=adaptive_method,
                                  adaptive_error=adaptive_error,
                                  adaptive_error_fit=adaptive_error_fit)
        return Spectrum1D(dos_bins, dos)

    def _calculate_dos(
        self,
        dos_bins: Quantity,
        mode_widths: Optional[Quantity] = None,
        mode_widths_min: Quantity = Quantity(0.01, 'meV'),
        mode_weights: Optional[np.ndarray] = None,
        adaptive_method: AdaptiveMethod = 'reference',
        adaptive_error: float = 0.01,
        adaptive_error_fit: ErrorFit = 'cubic',
        q_idx: Optional[int] = None
        ) -> Quantity:
        """
        Calculates a density of states (same arg defs as calculate_dos),
        but returns just a Quantity containing the y_data, rather
        than a Spectrum1D

        The mode_weights is a shape (n_qpts, n_modes) float array,
        describing the weight of each mode at each q-point.
        This is useful for calculating PDOS, for total DOS each
        mode is equally weighted so each mode_weights value will
        be one, for PDOS it will not

        If q_idx is specified, only calculates a DOS for that specific
        q-point, used for calculating a Q-E DOS-weighted map. In this
        case, mode_widths and mode_weights must be shape (1, n_modes)
        arrays
        """

        adaptive_method_options = ['reference', 'fast']
        if adaptive_method not in adaptive_method_options:
            raise ValueError(f'Invalid value for adaptive_method, '
                             f'got {adaptive_method}, should be one '
                             f'of {adaptive_method_options}')

        if q_idx is None:
            freqs = self._frequencies
            weights = self.weights
        else:
            freqs = self._frequencies[q_idx][np.newaxis, :]
            weights = np.array([self.weights[q_idx]])
        n_modes = freqs.shape[-1]
        # dos_bins commonly contains a 0 bin, and converting from 0 1/cm
        # to 0 hartree causes a RuntimeWarning, so suppress it
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            dos_bins_calc = dos_bins.to('hartree').magnitude
        if mode_weights is not None:
            mode_weights_calc = mode_weights
        else:
            # Normalise DOS to per atom (rather than per unit cell)
            mode_weights_calc = np.full(freqs.shape, 3/freqs.shape[1])
        if mode_widths is not None:
            mode_widths = mode_widths.to('hartree').magnitude
            mode_widths = np.maximum(mode_widths,
                                     mode_widths_min.to('hartree').magnitude)
            if adaptive_method == 'reference':
                # adaptive broadening by summing over individual peaks
                from scipy.stats import norm
                dos_bins_calc = Spectrum1D._bin_edges_to_centres(dos_bins_calc)
                dos = np.zeros(len(dos_bins_calc))
                for q in range(len(freqs)):
                    for m in range(n_modes):
                        pdf = norm.pdf(dos_bins_calc, loc=freqs[q, m],
                                       scale=mode_widths[q, m])
                        dos += pdf*weights[q]*mode_weights_calc[q, m]
            elif adaptive_method == 'fast':
                # fast, approximate method for adaptive broadening
                combined_weights = mode_weights_calc * weights[:, np.newaxis]
                dos = _width_interpolated_broadening(dos_bins_calc,
                                                     freqs, mode_widths,
                                                     combined_weights,
                                                     adaptive_error,
                                                     fit=adaptive_error_fit)
        else:
            bin_idx = np.digitize(freqs, dos_bins_calc)
            # Create DOS with extra bin either side, for any points
            # that fall outside the bin range
            dos = np.zeros(len(dos_bins) + 1)
            bin_widths = np.ones(len(dos_bins) + 1)  # Use ones to avoid div/0
            bin_widths[1:-1] = np.diff(dos_bins_calc)
            mode_weights_calc = (weights[:, np.newaxis]
                                 * mode_weights_calc/bin_widths[bin_idx])
            np.add.at(dos, bin_idx, mode_weights_calc)
            dos = dos[1:-1]

        dos = dos/np.sum(weights)
        # Avoid issues in converting DOS when energy is in cm^-1
        # Pint allows hartree -> cm^-1 but not 1/hartree -> cm
        conv = 1*ureg('hartree').to(dos_bins.units)
        return dos/conv

    def calculate_dos_map(self, dos_bins: Quantity,
                          mode_widths: Optional[Quantity] = None,
                          mode_widths_min: Quantity = Quantity(0.01, 'meV')
                          ) -> Spectrum2D:
        """
        Produces a bandstructure-like plot, using the DOS at each q-point

        Parameters
        ----------
        dos_bins
            Shape (n_e_bins + 1,) float Quantity in energy units. The
            energy bin edges to use for calculating the DOS
        mode_widths
            Shape (n_qpts, n_branches) float Quantity in energy units.
            The broadening width for each mode at each q-point, for
            adaptive broadening
        mode_widths_min
            Scalar float Quantity in energy units. Sets a lower limit on
            the mode widths, as mode widths of zero will result in
            infinitely sharp peaks

        Returns
        -------
        dos_map
            A 2D spectrum containing the q-point bins on the x-axis, energy
            bins on the y-axis and DOS on the z-axis
        """
        x_data, x_tick_labels = self._get_qpt_axis_and_labels()
        dos_map = np.zeros((len(self.qpts), len(dos_bins) - 1))
        for i in range(len(self.qpts)):
            dos_kwargs = {}
            if mode_widths is not None:
                dos_kwargs['mode_widths'] = mode_widths[i][np.newaxis, :]
            dos = self._calculate_dos(
                dos_bins, mode_widths_min=mode_widths_min, q_idx=i,
                **dos_kwargs)
            dos_map[i, :] = dos.magnitude
        return Spectrum2D(x_data, dos_bins, dos_map*(dos.units),
                          x_tick_labels=x_tick_labels)

    def get_dispersion(self) -> Spectrum1DCollection:
        """
        Creates a set of 1-D bands from mode data

        Bands follow the same q-point order as in the qpts array, with
        x-axis spacing corresponding to the absolute distances between
        q-points.  Discontinuities will appear as large jumps on the
        x-axis.

        Returns
        -------
        dispersion
            A sequence of mode bands with a common x-axis
        """
        abscissa = _calc_abscissa(self.crystal.reciprocal_cell(), self.qpts)
        x_tick_labels = get_qpoint_labels(self.qpts,
                                          cell=self.crystal.to_spglib_cell())
        return Spectrum1DCollection(abscissa, self.frequencies.T,
                                    x_tick_labels=x_tick_labels)

    def _get_qpt_axis_and_labels(
            self) -> Tuple[Quantity, List[Tuple[int, str]]]:
        """
        Converts the qpts stored in this object an array of
        scalar distances and applies appropriate symmetry labels

        Returns
        -------
        abscissa
            Shape (n_qpts,) float quantity in 1/length units. The
            scalar q-point distances
        x_tick_labels
            The tick labels
        """
        abscissa = _calc_abscissa(self.crystal.reciprocal_cell(), self.qpts)
        # Calculate q-space ticks and labels
        x_tick_labels = get_qpoint_labels(
            self.qpts, cell=self.crystal.to_spglib_cell())
        return abscissa, x_tick_labels

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See QpointFrequencies.from_dict for
        details on keys/values
        """
        dout = _obj_to_dict(self, ['crystal', 'n_qpts', 'qpts', 'frequencies',
                                   'weights'])
        return dout

    def to_json_file(self, filename: str) -> None:
        """
        Write to a JSON file. JSON fields are equivalent to
        from_dict keys

        Parameters
        ----------
        filename
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Convert a dictionary to a QpointFrequencies object

        Parameters
        ----------
        d
            A dictionary with the following keys/values:

            - 'crystal': dict, see Crystal.from_dict
            - 'qpts': (n_qpts, 3) float ndarray
            - 'frequencies': (n_qpts, n_branches) float ndarray
            - 'frequencies_unit': str

            There are also the following optional keys:

            - 'weights': (n_qpts,) float ndarray
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(d, quantities=['frequencies'], optional=['weights'])
        return cls(crystal, d['qpts'], d['frequencies'],
                   d['weights'])

    @classmethod
    def from_json_file(cls: Type[T], filename: str) -> T:
        """
        Read from a JSON file. See from_dict for
        required fields

        Parameters
        ----------
        filename
            The file to read from
        """
        return _obj_from_json_file(cls, filename)

    @classmethod
    def from_castep(cls: Type[T], filename: str,
                    average_repeat_points: bool = True,
                    prefer_non_loto: bool = False) -> T:
        """
        Reads precalculated phonon mode data from a CASTEP .phonon file

        Parameters
        ----------
        filename
            The path and name of the .phonon file to read
        average_repeat_points
            If multiple frequency/eigenvectors blocks are included with the
            same q-point index (i.e. for Gamma-point with LO-TO splitting),
            scale the weights such that these sum to the given weight
        prefer_non_loto
            If multiple frequency/eigenvector blocks are provided with the same
            q-point index and all-but-one include a direction vector, use the
            data from the point without a direction vector. (i.e. use the
            "exact" Gamma data without non-analytic correction.) This option
            takes priority over average_repeat_points.
        """
        data = castep.read_phonon_data(
            filename,
            read_eigenvectors=False,
            average_repeat_points=average_repeat_points,
            prefer_non_loto=prefer_non_loto)

        return cls.from_dict(data)

    @classmethod
    def from_phonopy(cls: Type[T], path: str = '.',
                     phonon_name: str = 'band.yaml',
                     phonon_format: Optional[str] = None,
                     summary_name: str = 'phonopy.yaml') -> T:
        """
        Reads precalculated phonon mode data from a Phonopy
        mesh/band/qpoints.yaml/hdf5 file. May also read from
        phonopy.yaml for structure information.

        Parameters
        ----------
        path
            Path to directory containing the file(s)
        phonon_name
            Name of Phonopy file including the frequencies
        phonon_format
            Format of the phonon_name file if it isn't obvious from the
            phonon_name extension, one of {'yaml', 'hdf5'}
        summary_name
            Name of Phonopy summary file to read the crystal information
            from. Crystal information in the phonon_name file takes
            priority, but if it isn't present, crystal information is
            read from summary_name instead
        """
        data = phonopy.read_phonon_data(
            path=path, phonon_name=phonon_name, phonon_format=phonon_format,
            summary_name=summary_name, read_eigenvectors=False)
        return cls.from_dict(data)

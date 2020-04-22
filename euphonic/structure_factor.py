from euphonic import ureg

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
    frequencies: (n_qpts, 3*crystal.n_atoms) float Quantity
        Phonon frequencies per q-point and mode
    structure_factors: (n_qpts, 3*crystal.n_atoms) float Quantity
        Structure factor per q-point and mode
    """

    def __init__(self, crystal, qpts, frequencies, structure_factors):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        qpts : (n_qpts, 3) float ndarray
            Q-point coordinates, in fractional coordinates of the reciprocal lattice
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

    @property
    def frequencies(self):
        return self._frequencies*ureg(
            'INTERNAL_ENERGY_UNIT').to(self.frequencies_unit, 'spectroscopy')

    @property
    def structure_factors(self):
        return self._structure_factors*ureg('INTERNAL_LENGTH_UNIT**2').to(
            self.structure_factors_unit)


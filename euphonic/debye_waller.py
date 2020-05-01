import inspect
from pint import Quantity
from euphonic import ureg, Crystal
from euphonic.util import _check_constructor_inputs

class DebyeWaller(object):
    """
    Stores the (n_ions, 3, 3) anisotropic Debye-Waller exponent W_ab. To be used
    in structure factor calculations

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    debye_waller : (n_ions, 3, 3) float Quantity
        The anisotropic Debye-Waller exponent W_ab, where the Debye-Waller
        factor is exp(-W_ab*Q_a*Q_b) where a,b run over the 3 Cartesian
        directions
    temperature : float Quantity
        The temperature the Debye-Waller exponent was calculated at
    """

    def __init__(self, crystal, debye_waller, temperature):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        debye_waller : (n_atoms, 3, 3) Quantity
            The anisotropic Debye-Waller exponent W_ab for each atom, where the
            Debye-Waller factor is exp(-W_ab*Q_a*Q_b) where a,b run over the 3
            Cartesian directions
        temperature : float Quantity
            The temperature the Debye-Waller exponent was calculated at
        """
        _check_constructor_inputs([crystal], [Crystal], [()], ['crystal'])
        n_atoms = crystal.n_atoms
        _check_constructor_inputs(
            [debye_waller, temperature],
            [Quantity, Quantity],
            [(n_atoms, 3, 3), ()],
            inspect.getfullargspec(self.__init__)[0][2:])
        self.crystal = crystal
        self._debye_waller = debye_waller.to(
            ureg.INTERNAL_LENGTH_UNIT**2).magnitude
        self._temperature = temperature.to(
            ureg.INTERNAL_TEMPERATURE_UNIT).magnitude

        self.debye_waller_unit = str(debye_waller.units)
        self.temperature_unit = str(temperature.units)

    @property
    def debye_waller(self):
        return self._debye_waller*ureg('INTERNAL_LENGTH_UNIT**2').to(
            self.debye_waller_unit)

    @property
    def temperature(self):
        # See https://pint.readthedocs.io/en/latest/nonmult.html
        return Quantity(self._temperature,
                        ureg('INTERNAL_TEMPERATURE_UNIT')).to(
                            self.temperature_unit)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name in ['debye_waller_unit', 'temperature_unit']:
                ureg(getattr(self, name)).to(value)
        super(DebyeWaller, self).__setattr__(name, value)

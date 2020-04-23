from euphonic import ureg

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
        debye_waller : (n_ions, 3, 3) Quantity
            The anisotropic Debye-Waller exponent W_ab, where the Debye-Waller
            factor is exp(-W_ab*Q_a*Q_b) where a,b run over the 3 Cartesian
            directions
        temperature : float Quantity
            The temperature the Debye-Waller exponent was calculated at
        """
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
        return self._temperature*ureg('INTERNAL_TEMPERATURE_UNIT').to(
            self.temperature_unit)
import inspect
from typing import TypeVar, Dict, Any, Type

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity, Crystal


class DebyeWaller:
    """
    Stores the (n_ions, 3, 3) anisotropic Debye-Waller exponent
    W_ab to be used in structure factor calculations

    Attributes
    ----------
    crystal
        Lattice and atom information
    debye_waller
        Shape (n_ions, 3, 3) float Quantity. The anisotropic
        Debye-Waller exponent W_ab for each atom, where the
        Debye-Waller factor is exp(-W_ab*Q_a*Q_b) where a,b run over
        the 3 Cartesian directions
    temperature
        Scalar float Quantity. The temperature the Debye-Waller
        exponent was calculated at
    """
    T = TypeVar('T', bound='DebyeWaller')

    def __init__(self, crystal: Crystal, debye_waller: Quantity,
                 temperature: Quantity) -> None:
        """
        Parameters
        ----------
        crystal
            Lattice and atom information
        debye_waller
            Shape (n_atoms, 3, 3) Quantity. The anisotropic
            Debye-Waller exponent W_ab for each atom, where the
            Debye-Waller factor is exp(-W_ab*Q_a*Q_b) where a,b run
            over the 3 Cartesian directions
        temperature
            Scalar float Quantity. The temperature the Debye-Waller
            exponent was calculated at
        """
        _check_constructor_inputs([crystal], [Crystal], [()], ['crystal'])
        n_atoms = crystal.n_atoms
        _check_constructor_inputs(
            [debye_waller, temperature],
            [Quantity, Quantity],
            [(n_atoms, 3, 3), ()],
            inspect.getfullargspec(self.__init__)[0][2:])
        self.crystal = crystal
        self._debye_waller = debye_waller.to(ureg.bohr**2).magnitude
        self._temperature = temperature.to(ureg.K).magnitude

        self.debye_waller_unit = str(debye_waller.units)
        self.temperature_unit = str(temperature.units)

    @property
    def debye_waller(self) -> Quantity:
        return self._debye_waller*ureg('bohr**2').to(
            self.debye_waller_unit)

    @debye_waller.setter
    def debye_waller(self, value: Quantity) -> None:
        self.debye_waller_unit = str(value.units)
        self._debye_waller = value.to('bohr**2').magnitude

    @property
    def temperature(self) -> Quantity:
        # See https://pint.readthedocs.io/en/latest/nonmult.html
        return Quantity(self._temperature, ureg('K')).to(self.temperature_unit)

    @temperature.setter
    def temperature(self, value: Quantity) -> None:
        self.temperature_unit = str(value.units)
        self._temperature = value.to('K').magnitude

    def __setattr__(self, name: str, value: Any) -> None:
        _check_unit_conversion(self, name, value,
                               ['debye_waller_unit', 'temperature_unit'])
        super(DebyeWaller, self).__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See DebyeWaller.from_dict for details
        on keys/values

        Returns
        -------
        dict
        """
        dout = _obj_to_dict(self, ['crystal', 'debye_waller', 'temperature'])
        return dout

    def to_json_file(self, filename: str) -> None:
        """
        Write to a JSON file. JSON fields are equivalent to
        DebyeWaller.from_dict keys

        Parameters
        ----------
        filename
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Convert a dictionary to a DebyeWaller object

        Parameters
        ----------
        d
            A dictionary with the following keys/values:

            - 'crystal': dict, see Crystal.from_dict
            - 'debye_waller': (n_atoms, 3, 3) float ndarray
            - 'debye_waller_unit': str
            - 'temperature': float
            - 'temperature_unit': str

        Returns
        -------
        DebyeWaller
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(d, quantities=['debye_waller', 'temperature'],
                          optional=['temperature'])
        return cls(crystal, d['debye_waller'], d['temperature'])

    @classmethod
    def from_json_file(cls: Type[T], filename: str) -> T:
        """
        Read from a JSON file. See DebyeWaller.from_dict for required
        fields

        Parameters
        ----------
        filename
            The file to read from

        Returns
        -------
        DebyeWaller
        """
        return _obj_from_json_file(cls, filename)

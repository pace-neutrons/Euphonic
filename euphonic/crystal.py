import inspect
from math import ceil
from typing import List, Tuple, TypeVar, Dict, Any

import numpy as np

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity


CR = TypeVar('CR', bound='Crystal')


class Crystal:
    """
    Stores lattice and atom information

    Attributes
    ----------
    cell_vectors
        Shape (3, 3) float Quantity in length units. Cartesian unit
        cell vectors. cell_vectors[0] = a, cell_vectors[:, 0] = x etc.
    n_atoms
        Number of atoms in the unit cell
    atom_r
        Shape (n_atoms, 3) float ndarray. The fractional position of
        each atom within the unit cell
    atom_type
        Shape (n_atoms,) string ndarray. The chemical symbols of each
        atom in the unit cell
    atom_mass
        Shape (n_atoms,) float Quantity in mass units. The mass of each
        atom in the unit cell
    """

    def __init__(self, cell_vectors: Quantity, atom_r: np.ndarray,
                 atom_type: np.ndarray, atom_mass: Quantity) -> None:
        """
        Parameters
        ----------
        cell_vectors
            Shape (3, 3) float Quantity in length units. Cartesian unit
            cell vectors. cell_vectors[0] = a,
            cell_vectors[:, 0] = x etc.
        atom_r
            Shape (n_atoms, 3) float ndarray. The fractional position
            of each atom within the unit cell
        atom_type
            Shape (n_atoms,) string ndarray. The chemical symbols of
            each atom in the unit cell
        atom_mass
            Shape (n_atoms,) float Quantity in mass units. The mass
            of each atom in the unit cell
        """
        n_atoms = len(atom_r)
        # Allow empty structure information, but convert to correct
        # shape/type. This is required here to allow reading from .json
        # files where attrs will have been converted to a list so
        # shape/type information will have been lost
        if n_atoms == 0 and atom_r.shape == (0,):
            atom_r = np.zeros((0, 3), dtype=np.float64)
        if n_atoms == 0 and atom_type.shape == (0,):
            atom_type = np.array([], dtype='<U32')

        _check_constructor_inputs(
            [cell_vectors, atom_r, atom_type, atom_mass],
            [Quantity, np.ndarray, np.ndarray, Quantity],
            [(3, 3), (n_atoms, 3), (n_atoms,), (n_atoms,)],
            inspect.getfullargspec(self.__init__)[0][1:])
        self._cell_vectors = cell_vectors.to(ureg.bohr).magnitude
        self.n_atoms = n_atoms
        self.atom_r = atom_r
        self.atom_type = atom_type
        self._atom_mass = atom_mass.to(ureg.m_e).magnitude

        self.cell_vectors_unit = str(cell_vectors.units)
        self.atom_mass_unit = str(atom_mass.units)
                
    @property
    def cell_vectors(self):
        return self._cell_vectors*ureg('bohr').to(
            self.cell_vectors_unit)

    @property
    def atom_mass(self):
        return self._atom_mass*ureg('m_e').to(
            self.atom_mass_unit)

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['cell_vectors_unit', 'atom_mass_unit'])
        super(Crystal, self).__setattr__(name, value)

    def reciprocal_cell(self) -> Quantity:
        """
        Calculates the reciprocal lattice vectors

        Returns
        -------
        recip
            Shape (3, 3) float Quantity in 1/length units, the
            reciprocal lattice vectors
        """
        cv = self._cell_vectors

        bxc = np.cross(cv[1], cv[2])
        cxa = np.cross(cv[2], cv[0])
        axb = np.cross(cv[0], cv[1])
        vol = self._cell_volume()
        norm = 2*np.pi/vol

        recip = np.array([norm*bxc,
                          norm*cxa,
                          norm*axb])*(1/ureg.bohr)

        return recip.to(1/ureg(self.cell_vectors_unit))

    def cell_volume(self) -> Quantity:
        """
        Calculates the cell volume

        Returns
        -------
        volume
            Scalar float quantity in length**3 units. The cell volume
        """
        vol = self._cell_volume()*ureg.bohr**3
        return vol.to(ureg(self.cell_vectors_unit)**3)

    def _cell_volume(self):
        cv = self._cell_vectors
        return np.dot(cv[0], np.cross(cv[1], cv[2]))

    def get_mp_grid_spec(self,
                         spacing: Quantity = 0.1 * ureg('1/angstrom')
                         ) -> Tuple[int, int, int]:
        """
        Get suggested divisions for Monkhorst-Pack grid

        Determine a mesh for even Monkhorst-Pack sampling of the reciprocal
        cell

        Parameters
        ----------
        spacing
            Scalar float quantity in 1/length units. Maximum
            reciprocal-space distance between q-point samples

        Returns
        -------
        grid_spec
            The number of divisions for each reciprocal lattice vector
        """

        recip_length_unit = spacing.units
        lattice = self.reciprocal_cell().to(recip_length_unit)
        grid_spec = np.linalg.norm(lattice.magnitude, axis=1
                                   ) / spacing.magnitude
        # math.ceil is better than np.ceil because it returns ints
        return tuple([ceil(x) for x in grid_spec])

    def to_spglib_cell(self) -> Tuple[List[List[float]],
                                      List[List[float]],
                                      List[int]]:
        """
        Convert to a 'cell' as defined by spglib

        Returns
        -------
        cell
            cell = (lattice, positions, numbers), where lattice is the
            lattice vectors, positions are the fractional atomic
            positions, and numbers are integers distinguishing the
            atomic species
        """
        _, unique_atoms = np.unique(self.atom_type, return_inverse=True)
        cell = (self.cell_vectors.magnitude.tolist(),
                self.atom_r.tolist(),
                unique_atoms.tolist())
        return cell

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See Crystal.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        dout = _obj_to_dict(self, ['cell_vectors', 'n_atoms', 'atom_r',
                                   'atom_type', 'atom_mass'])
        return dout

    def to_json_file(self, filename: str):
        """
        Write to a JSON file. JSON fields are equivalent to
        Crystal.from_dict keys

        Parameters
        ----------
        filename
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls: CR, d: Dict[str, Any]) -> CR:
        """
        Convert a dictionary to a Crystal object

        Parameters
        ----------
        d
            A dictionary with the following keys/values:

            - 'cell_vectors': (3, 3) float ndarray
            - 'cell_vectors_unit': str
            - 'atom_r': (n_atoms, 3) float ndarray
            - 'atom_type': (n_atoms,) str ndarray
            - 'atom_mass': (n_atoms,) float np.ndaaray
            - 'atom_mass_unit': str

        Returns
        -------
        crystal
        """
        d = _process_dict(d, quantities=['cell_vectors', 'atom_mass'])
        return cls(d['cell_vectors'], d['atom_r'], d['atom_type'],
                   d['atom_mass'])

    @classmethod
    def from_json_file(cls: CR, filename: str) -> CR:
        """
        Read from a JSON file. See Crystal.from_dict for required fields

        Parameters
        ----------
        filename
            The file to read from

        Returns
        -------
        crystal
        """
        return _obj_from_json_file(cls, filename)

    @classmethod
    def from_cell_vectors(cls: CR, cell_vectors: Quantity) -> CR:
        """
        Create a Crystal object from just cell vectors, containing no
        detailed structure information (atomic positions, species,
        masses)

        Parameters
        ----------
        cell_vectors
            Shape (3, 3) float Quantity in length units. Cartesian
            unit cell vectors. cell_vectors[0] = a,
            cell_vectors[:, 0] = x etc.

        Returns
        -------
        crystal
        """
        return cls(cell_vectors,
                   np.array([]), np.array([]),
                   Quantity(np.array([], dtype=np.float64), 'amu'))

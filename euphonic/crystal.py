import inspect

import numpy as np

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity


class Crystal(object):
    """
    Stores lattice and atom information

    Attributes
    ----------
    cell_vectors : (3, 3) float Quantity
        Unit cell vectors
    n_atoms : int
        Number of atoms in the unit cell
    atom_r : (n_atoms, 3) float ndarray
        The fractional position of each atom within the unit cell
    atom_type : (n_atoms,) string ndarray
        The chemical symbols of each atom in the unit cell. Atoms are in
        the same order as in atom_r
    atom_mass : (n_atoms,) float Quantity
        The mass of each atom in the unit cell
    """

    def __init__(self, cell_vectors, atom_r, atom_type, atom_mass):
        """
        Parameters
        ----------
        cell_vectors : (3, 3) float Quantity
            Cartesian unit cell vectors. cell_vectors[0] = a,
            cell_vectors[:, 0] = x etc.
        atom_r : (n_atoms, 3) float ndarray
            The fractional position of each atom within the unit cell
        atom_type : (n_atoms,) string ndarray
            The chemical symbols of each atom in the unit cell. Atoms
            are in the same order as in atom_r
        atom_mass : (n_atoms,) float Quantity
            The mass of each atom in the unit cell, in the same order as
            atom_r
        """
        n_atoms = len(atom_r)
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

    def reciprocal_cell(self):
        """
        Calculates the reciprocal lattice vectors

        Returns
        -------
        recip : (3, 3) float Quantity
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

    def cell_volume(self):
        vol = self._cell_volume()*ureg.bohr**3
        return vol.to(ureg(self.cell_vectors_unit)**3)

    def _cell_volume(self):
        cv = self._cell_vectors
        return np.dot(cv[0], np.cross(cv[1], cv[2]))

    def to_spglib_cell(self):
        """
        Convert to a 'cell' as defined by spglib

        Returns
        -------
        cell : tuple of lists
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

    def to_dict(self):
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

    def to_json_file(self, filename):
        """
        Write to a JSON file. JSON fields are equivalent to
        Crystal.from_dict keys

        Parameters
        ----------
        filename : str
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a Crystal object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'cell_vectors': (3, 3) float ndarray
            - 'cell_vectors_unit': str
            - 'atom_r': (n_atoms, 3) float ndarray
            - 'atom_type': (n_atoms,) str ndarray
            - 'atom_mass': (n_atoms,) float np.ndaaray
            - 'atom_mass_unit': str

        Returns
        -------
        Crystal
        """
        d = _process_dict(d, quantities=['cell_vectors', 'atom_mass'])
        return Crystal(d['cell_vectors'], d['atom_r'], d['atom_type'],
                       d['atom_mass'])

    @classmethod
    def from_json_file(cls, filename):
        """
        Read from a JSON file. See Crystal.from_dict for required fields

        Parameters
        ----------
        filename : str
            The file to read from

        Returns
        -------
        Crystal
        """
        return _obj_from_json_file(cls, filename)

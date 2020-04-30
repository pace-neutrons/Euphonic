import inspect
import numpy as np
from pint import Quantity
from euphonic import ureg
from euphonic.io import _obj_to_json_file, _obj_from_json_file
from euphonic.util import _check_unit, _check_constructor_inputs


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
        The chemical symbols of each atom in the unit cell. Atoms are in the
        same order as in atom_r
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
            The chemical symbols of each atom in the unit cell. Atoms are in the
            same order as in atom_r
        atom_mass : (n_atoms,) float Quantity
            The mass of each atom in the unit cell, in the same order as atom_r
        """
        n_atoms = len(atom_r)
        _check_constructor_inputs(
            [cell_vectors, atom_r, atom_type, atom_mass],
            [Quantity, np.ndarray, np.ndarray, Quantity],
            [(3, 3), (n_atoms, 3), (n_atoms,), (n_atoms,)],
            inspect.getfullargspec(self.__init__)[0][1:])
        self._cell_vectors = cell_vectors.to(
            ureg.INTERNAL_LENGTH_UNIT).magnitude
        self.n_atoms = n_atoms
        self.atom_r = atom_r
        self.atom_type = atom_type
        self._atom_mass = atom_mass.to(ureg.INTERNAL_MASS_UNIT).magnitude

        self.cell_vectors_unit = str(cell_vectors.units)
        self.atom_mass_unit = str(atom_mass.units)
                
    @property
    def cell_vectors(self):
        return self._cell_vectors*ureg('INTERNAL_LENGTH_UNIT').to(
            self.cell_vectors_unit)

    @property
    def atom_mass(self):
        return self._atom_mass*ureg('INTERNAL_MASS_UNIT').to(
            self.atom_mass_unit)

    def __setattr__(self, name, value):
        if name == 'cell_vectors_unit':
            _check_unit(value, '[length]')
        elif name == 'ion_mass_unit':
            _check_unit(value, '[mass]')
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
                          norm*axb])*(1/ureg.INTERNAL_LENGTH_UNIT)

        return recip.to(1/ureg(self.cell_vectors_unit))

    def cell_volume(self):
        vol = self._cell_volume()*ureg.INTERNAL_LENGTH_UNIT**3
        return vol.to(ureg(self.cell_vectors_unit)**3)

    def _cell_volume(self):
        cv = self._cell_vectors
        return np.dot(cv[0], np.cross(cv[1], cv[2]))

    def to_dict(self):
        d = vars(self).copy()
        d['cell_vectors'] = d.pop('_cell_vectors')*ureg(
            'INTERNAL_LENGTH_UNIT').to(self.cell_vectors_unit).magnitude
        d['atom_mass'] = d.pop('_atom_mass')*ureg(
            'INTERNAL_MASS_UNIT').to(self.atom_mass_unit).magnitude
        return d

    def to_json_file(self, filename):
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls, d):
        mu = d['atom_mass_unit']
        lu = d['cell_vectors_unit']
        return cls(d['cell_vectors']*ureg(lu), d['atom_r'],
                   d['atom_type'], d['atom_mass']*ureg(mu))

    @classmethod
    def from_json_file(cls, filename):
        return _obj_from_json_file(cls, filename)

import inspect
from math import ceil
from typing import List, Tuple, Type, TypeVar, Dict, Any
from collections import OrderedDict

import numpy as np
from spglib import get_symmetry

from euphonic.util import _cell_vectors_to_volume, _get_unique_elems_and_idx
from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity

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
    T = TypeVar('T', bound='Crystal')

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
    def cell_vectors(self) -> Quantity:
        return self._cell_vectors*ureg('bohr').to(
            self.cell_vectors_unit)

    @cell_vectors.setter
    def cell_vectors(self, value: Quantity) -> None:
        self.cell_vectors_unit = str(value.units)
        self._cell_vectors = value.to('bohr').magnitude

    @property
    def atom_mass(self) -> Quantity:
        return self._atom_mass*ureg('m_e').to(
            self.atom_mass_unit)

    @atom_mass.setter
    def atom_mass(self, value: Quantity) -> None:
        self.atom_mass_unit = str(value.units)
        self._atom_mass = value.to('m_e').magnitude

    def __setattr__(self, name: str, value: Any) -> None:
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
            Scalar float Quantity in length**3 units. The cell volume
        """
        vol = self._cell_volume()*ureg.bohr**3
        return vol.to(ureg(self.cell_vectors_unit)**3)

    def _cell_volume(self) -> float:
        return _cell_vectors_to_volume(self._cell_vectors)

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

    def get_species_idx(self) -> 'OrderedDict[str, np.ndarray]':
        """
        Returns a dictionary of each species and their indices

        Returns
        -------
        species_idx
            An ordered dictionary containing each unique species
            symbol as the keys, and their indices as the values,
            in the same order as they appear in atom_type
        """
        species_dict = _get_unique_elems_and_idx(
            [tuple([at]) for at in self.atom_type])
        # Convert tuples back to string
        return OrderedDict([(str(key[0]), value)
                            for key, value in species_dict.items()])

    def get_symmetry_equivalent_atoms(
            self, tol: Quantity = Quantity(1e-5, 'angstrom')
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the rotational and translational symmetry operations
        as obtained by spglib.get_symmetry, and also the equivalent
        atoms that each atom gets mapped onto for each symmetry
        operation

        Parameters
        ----------
        tol
           Scalar float Quantity in length units. The distance
           tolerance, if the distance between atoms is less than
           this, they are considered to be equivalent. This is
           also passed to spglib.get_symmetry as symprec

        Returns
        -------
        rotations
           Shape (n_symmetry_ops, 3, 3) integer np.ndarray. The
           rotational symmetry matrices as returned by
           spglib.get_symmetry
        translations
           Shape (n_symmetry_ops, 3) float np.ndarray. The
           rotational symmetry matrices as returned by
           spglib.get_symmetry
        equivalent_atoms
           Shape (n_symmetry_ops, n_atoms) integer np.ndarray.
           The equivalent atoms for each symmetry operation. e.g.
           equivalent_atoms[s, i] = j means symmetry operation s
           maps atom i to atom j

        """
        tol_calc = tol.to('bohr').magnitude
        symprec = tol.to(self.cell_vectors_unit).magnitude
        # Sometimes if symprec is very low, even the identity
        # symmetry op won't be found, and None will be returned
        # For some reason this can't always be reproduced
        symm = get_symmetry(self.to_spglib_cell(), symprec=symprec)
        if symm is None:
            raise RuntimeError(f'spglib.get_symmetry returned None with '
                               f'symprec={symprec}. Try increasing tol')
        n_ops = len(symm['rotations'])
        equiv_atoms = np.full((n_ops, self.n_atoms), -1, dtype=np.int32)
        atom_r_symm = (np.einsum('ijk,lk->ilj', symm['rotations'], self.atom_r)
                       + symm['translations'][:, np.newaxis, :])
        atom_r_symm -= np.floor(atom_r_symm + 0.5)

        species_idx = self.get_species_idx()
        for spec, idx in species_idx.items():
            for i in idx:
                atom_r_symm_i = atom_r_symm[:, i, :]
                # Difference between symmetry-transformed atom i and all
                # other atoms of that species for each symmetry operation
                diff_frac = (atom_r_symm_i[:, np.newaxis, :]
                             - self.atom_r[np.newaxis, idx, :])
                diff_frac -= np.floor(diff_frac + 0.5)
                diff_cart = np.einsum('ijk,kl->ijl', diff_frac, self._cell_vectors)
                diff_r = np.linalg.norm(diff_cart, axis=2)
                equiv_idx = np.where(diff_r < tol_calc)
                # There should be one matching atom per symm op
                if not np.array_equal(equiv_idx[0], np.arange(n_ops)):
                    for op_idx, diff_r_op in enumerate(diff_r):
                        equiv_idx_op = np.where(diff_r_op < tol_calc)[0]
                        err_info = (f'for {spec} atom at {self.atom_r[i]} for '
                                    f'symmetry op {op_idx}. Rotation '
                                    f'{symm["rotations"][op_idx]} translation '
                                    f'{symm["translations"][op_idx]}')
                        if len(equiv_idx_op) == 0:
                            raise RuntimeError(f'No equivalent atom found {err_info}')
                        elif len(equiv_idx_op) > 1:
                            raise RuntimeError(f'Multiple equivalent atoms found {err_info}')
                equiv_atoms[:, i] = idx[equiv_idx[1]]

        return symm['rotations'], symm['translations'], equiv_atoms

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

    def to_json_file(self, filename: str) -> None:
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
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
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
    def from_json_file(cls: Type[T], filename: str) -> T:
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
    def from_cell_vectors(cls: Type[T], cell_vectors: Quantity) -> T:
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

"""Export to JSON for phonon visualisation website:

- https://henriquemiranda.github.io/phononwebsite/index.html
"""

from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from euphonic.crystal import Crystal
from euphonic.qpoint_phonon_modes import QpointPhononModes
from euphonic.spectra import XTickLabels
from euphonic.util import _calc_abscissa, get_qpoint_labels


ComplexPair = tuple[float, float]


class PhononWebsiteData(TypedDict):
    """Data container for export to phonon visualisation website

    Specification: https://henriquemiranda.github.io/phononwebsite/index.html

    line_breaks are currently not implemented

    """
    name: str
    natoms: int
    lattice: list[list[float]]
    atom_types: list[str]
    atom_numbers: list[int]
    formula: str
    repetitions: list[int]
    atom_pos_car: list[list[float]]
    atom_pos_red: list[list[float]]
    highsym_qpts: list[tuple[int, str]]
    qpoints: list[list[float]]
    distances: list[float]  # Cumulative distance from first q-point
    eigenvalues: list[float]  # in cm-1
    vectors: list[list[list[tuple[ComplexPair, ComplexPair, ComplexPair]]]]
    line_breaks: list[tuple[int, int]]


def write_phonon_website_json(
    modes: QpointPhononModes,
    output_file: str | Path = "phonons.json",
    name: str = "Euphonic export",
    x_tick_labels: XTickLabels | None = None,
) -> None:

    """Dump to .json for use with phonon website visualiser

    Use with javascript application at
    https://henriquemiranda.github.io/phononwebsite

    Parameters
    ----------
    modes
        Phonon frequencies and eigenvectors along q-point path
    output_file
        Path to output file
    name
        Set "name" metadata, to be used as figure title
    x_tick_labels
        index and label for high symmetry labels (if known)

    """

    with open(output_file, 'w', encoding='utf-8') as fd:
        json.dump(_modes_to_phonon_website_dict(modes=modes,
                                                name=name,
                                                x_tick_labels=x_tick_labels),
                  fd)


def _crystal_website_data(crystal: Crystal) -> dict[str, Any]:
    elements = [
        '_', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
        'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
        'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
        'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
        'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
        'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
        'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
        'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
        'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    def get_z(symbol: str) -> int:
        try:
            return elements.index(symbol)
        except ValueError:  # Symbol not found
            return 0

    def symbols_to_formula(symbols: list[str]) -> str:
        symbol_counts = Counter(symbols)

        return "".join(f"{symbol}{symbol_counts[symbol]}"
                       for symbol in sorted(symbol_counts))

    return dict(
        natoms=len(crystal.atom_type),
        lattice=crystal.cell_vectors.to("angstrom").magnitude.tolist(),
        atom_types=crystal.atom_type.tolist(),
        atom_numbers=list(map(get_z, crystal.atom_type)),
        formula=symbols_to_formula(crystal.atom_type),
        atom_pos_red=crystal.atom_r.tolist(),
        atom_pos_car=(crystal.atom_r @ crystal.cell_vectors
                      ).to("angstrom").magnitude.tolist()
    )


def _remove_breaks(distances: np.ndarray, btol: float = 10.) -> list[int]:
    """Collapse large breaks in cumulative-distance array

    These correspond to discontinuous regions of the x-axis: in euphonic
    plotting this is usually handled by splitting the spectrum and plotting
    to new axes, but Phonon Website does not handle this.

    Data is modified in-place

    A list of identified breakpoints is returned

    """
    diff = np.diff(distances)
    median = np.median(diff)
    breakpoints = np.where((diff / median) > btol)[0] + 1

    for line_break in reversed(breakpoints):
        distances[line_break:] -= diff[line_break - 1]

    return breakpoints.tolist()


def _find_duplicates(distances: np.ndarray) -> list[int]:
    """Identify breakpoints where a q-point is repeated in list

    Return the higher index of two points: this works more nicely with phonon
    website interpretation of line break points.
    """

    diff = np.diff(distances)
    duplicates = np.where(np.isclose(diff, 0.))[0] + 1

    return duplicates.tolist()


def _combine_neighbouring_labels(x_tick_labels: XTickLabels) -> XTickLabels:
    """Merge neighbouring labels in x_tick_label list

    If labels are the same, only keep one.

    If labels are different, join with |

    e.g.::

      [(1, "X"), (2, "X"), (4, "A"), (7, "Y"), (8, "Z")]

      -->

      [(2, "X"), (4, "A"), (8, "Y|Z")]

    """
    labels = dict(x_tick_labels)

    for (index, next_index) in pairwise(sorted(labels)):

        if index + 1 == next_index:  # Labels are neighbours

            if labels[index] != labels[next_index]:  # Combine differing labels
                labels[next_index] = f"{labels[index]}|{labels[next_index]}"

            del labels[index]  # Drop redundant label

    return sorted(labels.items())


def _modes_to_phonon_website_dict(
    modes: QpointPhononModes,
    name: str | None = None,
    repetitions: tuple[int, int, int] = (2, 2, 2),
    x_tick_labels: XTickLabels | None = None,
) -> PhononWebsiteData:
    """Convert QpointPhononModes to dict of phonon-website-friendly data"""

    qpts = modes.qpts
    eigenvectors = modes.eigenvectors

    abscissa = _calc_abscissa(modes.crystal.reciprocal_cell(), qpts)

    duplicates = _find_duplicates(abscissa)
    breakpoints = _remove_breaks(abscissa)

    breakpoints = sorted(set([0] + duplicates + breakpoints + [len(abscissa)]))
    line_breaks = list(pairwise(breakpoints))

    if x_tick_labels is None:
        x_tick_labels = get_qpoint_labels(qpts,
                                          cell=modes.crystal.to_spglib_cell())

    x_tick_labels = [(int(key) + 1, str(value))
                     for key, value in x_tick_labels]
    x_tick_labels = _combine_neighbouring_labels(x_tick_labels)

    mass_weights = 1 / np.sqrt(modes.crystal.atom_mass)
    vectors = eigenvectors * mass_weights[None, None, :, None]
    # Convert complex numbers to a final axis over (real, imag)
    vectors = vectors.view(float).reshape(*eigenvectors.shape[:-1], 3, 2)

    dat = PhononWebsiteData(
        name=name,
        **_crystal_website_data(modes.crystal),
        highsym_qpts=x_tick_labels,
        distances=abscissa.magnitude.tolist(),
        qpoints=modes.qpts.tolist(),
        eigenvalues=modes.frequencies.to("1/cm").magnitude.tolist(),
        vectors=vectors.tolist(),
        repetitions=repetitions,
        line_breaks=line_breaks
    )

    return dat

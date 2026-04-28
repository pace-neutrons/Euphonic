"""Band structure utilities"""

from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, TypedDict

import numpy as np
import seekpath

from euphonic import (
    ForceConstants,
    QpointFrequencies,
    QpointPhononModes,
    Quantity,
)
from euphonic.util import (
    spglib_new_errors,
)


def _get_tick_labels(bandpath: dict) -> list[tuple[int, str]]:
    """Convert x-axis labels from seekpath format to euphonic format

    i.e.::

        ['L', '', '', 'X', '', 'GAMMA']   -->

        [(0, 'L'), (3, 'X'), (5, '$\\Gamma$')]
    """
    label_indices = np.where(bandpath['explicit_kpoints_labels'])[0]
    labels = (r'$\Gamma$' if label == 'GAMMA' else label
              for label in
              np.take(bandpath['explicit_kpoints_labels'], label_indices))
    return list(zip(label_indices, labels, strict=True))


def _get_break_points(bandpath: dict) -> list[int]:
    """Get information about band path labels and break points

    Parameters
    ----------
    bandpath
        Bandpath dictionary from Seekpath

    Returns
    -------
    break_points
        Indices at which the spectrum should be split into subplots
    """
    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    labels = np.array(bandpath['explicit_kpoints_labels'])

    special_point_bools = np.fromiter(
        map(bool, labels), dtype=bool)

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    adjacent_non_empty_labels = (
        special_point_bools[:-1] & special_point_bools[1:]
    )

    adjacent_different_labels = (labels[:-1] != labels[1:])

    break_points = np.where(
        adjacent_non_empty_labels & adjacent_different_labels,
    )[0]
    return (break_points + 1).tolist()


def _insert_gamma(bandpath: dict) -> None:
    """Modify seekpath.get_explicit_k_path() results; duplicate Gamma

    This enables LO-TO splitting to be included
    """
    import numpy as np
    gamma_indices = np.where(
        np.array(bandpath['explicit_kpoints_labels'][1:-1]) == 'GAMMA',
    )[0] + 1

    rel_kpts = bandpath['explicit_kpoints_rel'].tolist()
    labels = bandpath['explicit_kpoints_labels']
    for i in reversed(gamma_indices.tolist()):
        rel_kpts.insert(i, [0., 0., 0.])
        labels.insert(i, 'GAMMA')

    bandpath['explicit_kpoints_rel'] = np.array(rel_kpts)
    bandpath['explicit_kpoints_labels'] = labels

    # These unused properties have been invalidated: safer
    # to leave None than incorrect values
    bandpath['explicit_kpoints_abs'] = None
    bandpath['explicit_kpoints_linearcoord'] = None
    bandpath['explicit_segments'] = None


XTickLabels = list[tuple[int, str]]
SplitArgs = dict[str, Any]



class BandpathDict(TypedDict, total=False):
    """Dictionary returned by seekpath.get_explicit_k_path_orig_cell.

    Not a complete specification, but these are the parts we care about.
    """

    explicit_kpoints_labels: Sequence[str]
    explicit_kpoints_rel: Iterable[float]
    is_supercell: bool


def _convert_labels_to_fractions(
        bandpath: BandpathDict, *, limit: int = 32) -> None:
    """Replace high-symmetry labels in seekpath data with simple fractions

    bandpath:
        dict from seekpath.get_explicit_k_path_orig_cell

    limit:
        maximum numerator value for float rounded to fraction
    """
    for i, (label, qpt) in enumerate(zip(bandpath['explicit_kpoints_labels'],
                                         bandpath['explicit_kpoints_rel'],
                                         strict=True)):
        if label:
            qpt_label = ' '.join(str(Fraction(x).limit_denominator(limit))
                            for x in qpt)
            bandpath['explicit_kpoints_labels'][i] = qpt_label


def _bands_from_force_constants(data: ForceConstants,
                                q_distance: Quantity,
                                insert_gamma: bool = True,
                                frequencies_only: bool = False,
                                **calc_modes_kwargs,
) -> tuple[QpointPhononModes | QpointFrequencies, XTickLabels, SplitArgs]:
    structure = data.crystal.to_spglib_cell()
    with spglib_new_errors():
        bandpath = seekpath.get_explicit_k_path_orig_cell(
            structure,
            reference_distance=q_distance.to('1 / angstrom').magnitude)

    if insert_gamma:
        _insert_gamma(bandpath)

    # If input structure was not primitive, the high-symmetry points are not
    # really meaningful. Indicate this by converting to numerical form.
    if bandpath.get('is_supercell'):
        _convert_labels_to_fractions(bandpath, limit=32)

    x_tick_labels = _get_tick_labels(bandpath)
    split_args = {'indices': _get_break_points(bandpath)}

    print(
        'Computing phonon modes: {n_modes} modes across {n_qpts} q-points'
        .format(n_modes=(data.crystal.n_atoms * 3),
                n_qpts=len(bandpath['explicit_kpoints_rel'])))
    qpts = bandpath['explicit_kpoints_rel']

    if frequencies_only:
        modes = data.calculate_qpoint_frequencies(qpts,
                                                  reduce_qpts=False,
                                                  **calc_modes_kwargs)
    else:
        modes = data.calculate_qpoint_phonon_modes(qpts,
                                                   reduce_qpts=False,
                                                   **calc_modes_kwargs)
    return modes, x_tick_labels, split_args

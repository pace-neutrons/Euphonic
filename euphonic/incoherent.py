"""Incoherent neutron scattering methods"""
from typing import List

import numpy as np

from euphonic import Crystal, ureg, Quantity
from euphonic.qpoint_phonon_modes import QpointPhononModes

def anisotropic_dw(modes: QpointPhononModes,
                 temperature: Quantity = 10. * ureg('K'),
                 q: Quantity = np.array([1., 1., 1.]) * ureg('1 / angstrom'),
                 ref='Mitchell2005') -> np.ndarray:
    """Calculate Debye-Waller exponent in anisotropic approximation

    i.e. for a Debye-Waller factor exp(-2W) this is -2W

    This code is initially being used to check equivalence
    of various descriptions in literature

    Args:
        modes: Phonon modes on input q-point mesh
        temperature: System temperature
        q: Vector q

    """

    # Get normalised q-point weights
    weights = modes.weights / np.sum(modes.weights)

    # hbar term is dropped as we work in energy rather than angular freq
    coth = 1 / np.tanh(modes.frequencies.to('eV')
                       / (2. * ureg('k').to('eV / K') * temperature.to('K')))

    if ref =='Mitchell2005':
        # Interpret Mitchell as literally as possible:
        # - obtain u as vector including coth
        # - then multiply by coth again in the DW sum

        u = modes.eigenvectors * np.sqrt(        
            ureg('hbar')
            / 2
            / modes.crystal.atom_mass[None, None, :, None]
            / (modes.frequencies[:, :, None, None].to('Hz') * 2 * np.pi)
            * coth[:, :, None, None]
            ).to('angstrom')

        u_coth_sum = (u
                      # * coth[:, :, None, None]
                      * weights[:, None, None, None])
        u_coth_sum = u_coth_sum[modes.frequencies > 0].sum(axis=0)

        # In the book the square is written as (Q. Σ(u coth(beta)))^2
        # but this seems like simplistic notation that would remove
        # the off-diagonal contributions: Q and u_j are 3-vectors so
        # Q.u is a scalar and diagonal terms are lost.
        #
        # Cope thesis clarifies that time averaged
        # <(Q.u)^2> = Q^T <u u^T> Q, where <u u^T> = B
        # suggesting that something like q @ u x u @ q
        # is the correct way to apply this squaring operation.

        # So instead of
        # return -((u_coth_sum @ q)**2)

        return -(q @ np.einsum('ij,ik->ijk', u_coth_sum, u_coth_sum.conj()) @ q)

    elif ref =='Mitchell2005_B':
        # A more "normal" calculation based on the quadratic tensor, but
        # still with the multi-coth oddness from literal reading of Mitchell

        e_cross_e = np.einsum('ijkl,ijkm->ijklm',
                              modes.eigenvectors, modes.eigenvectors.conj())
        u2 = e_cross_e * (
              ureg('hbar')
              / (2 * 2 * np.pi)
              / modes.crystal.atom_mass[None, None, :,
                                        None, None]
              / modes.frequencies[:, :, None,
                                  None, None].to('Hz')
              * coth[:, :, None, None, None]
              ).to('angstrom^2')

        u2_coth_sum = (u2
                       * coth[:, :, None, None, None]**2
                       * weights[:, None, None, None, None])
        u2_coth_sum = u2_coth_sum[modes.frequencies > 0].sum(axis=0)

        return -(q @ u2_coth_sum @ q)

    elif ref =='conventional':

        e_cross_e = np.einsum('ijkl,ijkm->ijklm',
                              modes.eigenvectors, modes.eigenvectors.conj())
        u2 = _calc_B_no_bose(modes) * coth[:, :, None, None, None]
        u2_weighted = (u2 * weights[:, None, None, None, None])
        a = u2_weighted[modes.frequencies > 0].sum(axis=0)

        return -(q @ a @ q)

def isotropic_dw(modes: QpointPhononModes,
                 temperature: Quantity = 10. * ureg('K'),
                 q: Quantity = 1. * ureg('1 / angstrom')):
    """Calculate Debye-Waller exponent in fully-isotropic powder approximation"""
    # Get normalised q-point weights
    weights = modes.weights / np.sum(modes.weights)

    # hbar term is dropped as we work in energy rather than angular freq
    coth = 1 / np.tanh(modes.frequencies.to('eV')
                       / (2. * ureg('k').to('eV / K') * temperature.to('K')))

    u2 = _calc_B_no_bose(modes) * coth[:, :, None, None, None]

    u2_weighted = (u2 * weights[:, None, None, None, None])
    tr_a = np.einsum('ijkk->j', u2_weighted[modes.frequencies > 0])

    return -(q**2 * tr_a / 3)

def _calc_B_no_bose(modes: QpointPhononModes):
    """Calculate the mass-and-frequency-weighted eigenvector cross-products"""
    e_cross_e = np.einsum('ijkl,ijkm->ijklm',
                          modes.eigenvectors, modes.eigenvectors.conj())
    b = e_cross_e * (
        ureg('hbar')
        / (2 * 2 * np.pi)
        / modes.crystal.atom_mass[None, None, :,
                                  None, None]
        / modes.frequencies[:, :, None,
                            None, None].to('Hz')
    ).to('angstrom^2')

    return b


def athermal_isotropic_s1(modes: QpointPhononModes,
                          q: Quantity = 1. * ureg('1 / angstrom'),
                          bins: Quantity = None):
    """The simplest isotropic incoherent method: q^2 B/3 exp(q^2 A/3)"""
    if bins is None:
        bins = np.linspace(0,
                           modes.frequencies.magnitude.max() * 1.05,
                           1000
                           ) * modes.frequencies.units

    freq_mask = np.ones(modes.frequencies.shape, dtype=bool)
    freq_mask[modes.frequencies.magnitude < 0] = False

    b = _calc_B_no_bose(modes)

    tr_b = np.einsum('ijkll->ijk', b)
    tr_a = (tr_b * modes.weights[:, None, None])[freq_mask].sum(axis=0)

    s_atoms = (q**2 * (tr_b / 3)
               * np.exp(-q**2 * tr_a[None, None, :] / 3)
               * _get_cross_sections(modes.crystal)[None, None, :]
               )
    s = s_atoms.sum(axis=2).real

    intensity, _ = np.histogram(modes.frequencies[freq_mask].to(bins.units).magnitude,
                                bins=bins.magnitude,
                                weights=s[freq_mask].magnitude)
    return bins, (intensity * s.units / bins.units)


def athermal_almost_isotropic_s1(modes: QpointPhononModes,
                                 q: Quantity = 1. * ureg('1 / angstrom'),
                                 bins: Quantity = None):
    """Almost-isotropic approximation with mode-dependent exponent"""
    if bins is None:
        bins = np.linspace(0,
                           modes.frequencies.magnitude.max() * 1.05,
                           1000
                           ) * modes.frequencies.units

    freq_mask = np.ones(modes.frequencies.shape, dtype=bool)
    freq_mask[modes.frequencies.magnitude < 0] = False

    b = _calc_B_no_bose(modes)
    a = (b * modes.weights[:, None, None, None, None])[freq_mask].sum(axis=0)

    tr_b = np.einsum('ijkll->ijk', b)
    tr_a = (tr_b * modes.weights[:, None, None])[freq_mask].sum(axis=0)

    alpha = 0.2 * (tr_a + 2 * np.einsum('ijklm,klm->ijk', b, a) / tr_b)

    s_atoms = (q**2 * (tr_b / 3)
               * np.exp(-q**2 * alpha)
               * _get_cross_sections(modes.crystal)[None, None, :]
               )
    s = s_atoms.sum(axis=2).real

    intensity, _ = np.histogram(modes.frequencies[freq_mask].to(bins.units).magnitude,                                bins=bins.magnitude,
                                weights=s[freq_mask].magnitude)
    return bins, (intensity * s.units / bins.units)


def thermal_isotropic_s1(modes: QpointPhononModes,
                         q: Quantity = 1. * ureg('1 / angstrom'),
                         temperature: Quantity = 10 * ureg('K'),
                         bins: Quantity = None):
    """The simplest isotropic incoherent method: q^2 B/3 exp(q^2 A/3)"""
    if bins is None:
        bins = np.linspace(0,
                           modes.frequencies.magnitude.max() * 1.05,
                           1000
                           ) * modes.frequencies.units

    freq_mask = np.ones(modes.frequencies.shape, dtype=bool)
    freq_mask[modes.frequencies.magnitude < 0] = False

    coth = 1 / np.tanh(modes.frequencies.to('eV')
                       / (2. * ureg('k').to('eV / K') * temperature.to('K')))

    b = _calc_B_no_bose(modes) * coth[:, :, None, None, None]

    tr_b = np.einsum('ijkll->ijk', b)
    tr_a = (tr_b * modes.weights[:, None, None])[freq_mask].sum(axis=0)

    s_atoms = (q**2 * (tr_b / 3)
               * np.exp(-q**2 * tr_a[None, None, :] / 3)
               * _get_cross_sections(modes.crystal)[None, None, :]
               )
    s = s_atoms.sum(axis=2).real

    intensity, _ = np.histogram(modes.frequencies[freq_mask].to(bins.units).magnitude,
                                bins=bins.magnitude,
                                weights=s[freq_mask].magnitude)
    return bins, (intensity * s.units / bins.units)


def athermal_almost_isotropic_s1(modes: QpointPhononModes,
                                 q: Quantity = 1. * ureg('1 / angstrom'),
                                 bins: Quantity = None):
    """Almost-isotropic approximation with mode-dependent exponent"""
    if bins is None:
        bins = np.linspace(0,
                           modes.frequencies.magnitude.max() * 1.05,
                           1000
                           ) * modes.frequencies.units

    freq_mask = np.ones(modes.frequencies.shape, dtype=bool)
    freq_mask[modes.frequencies.magnitude < 0] = False

    b = _calc_B_no_bose(modes)
    a = (b * modes.weights[:, None, None, None, None])[freq_mask].sum(axis=0)

    tr_b = np.einsum('ijkll->ijk', b)
    tr_a = (tr_b * modes.weights[:, None, None])[freq_mask].sum(axis=0)

    alpha = 0.2 * (tr_a + 2 * np.einsum('ijklm,klm->ijk', b, a) / tr_b)

    s_atoms = (q**2 * (tr_b / 3)
               * np.exp(-q**2 * alpha)
               * _get_cross_sections(modes.crystal)[None, None, :]
               )
    s = s_atoms.sum(axis=2).real

    intensity, _ = np.histogram(modes.frequencies[freq_mask].to(bins.units).magnitude,                                bins=bins.magnitude,
                                weights=s[freq_mask].magnitude)
    return bins, (intensity * s.units / bins.units)


def _get_cross_sections(crystal: Crystal) -> List[Quantity]:
    from euphonic.util import get_reference_data
    element_cross_sections = get_reference_data(collection='BlueBook',
                                                physical_property='incoherent_cross_section')

    cross_sections = np.array([element_cross_sections[symbol].to('barn').magnitude
                               for symbol in crystal.atom_type]
                              ) * ureg('barn')
    return cross_sections
    

def numerical_isotropic_dw(modes: QpointPhononModes,
                             temperature: Quantity = 10. * ureg('K'),
                             q: Quantity = 1. * ureg('1 / angstrom'),
                             npts: int = 100):

    from euphonic.sampling import golden_sphere

    # Get normalised q-point weights
    weights = modes.weights / np.sum(modes.weights)

    # hbar term is dropped as we work in energy rather than angular freq
    coth = 1 / np.tanh(modes.frequencies.to('eV')
                       / (2. * ureg('k').to('eV / K') * temperature.to('K')))

    e_cross_e = np.einsum('ijkl,ijkm->ijklm',
                          modes.eigenvectors, modes.eigenvectors.conj())
    u2 = e_cross_e * (
        ureg('hbar')
        / (2 * 2 * np.pi)
        / modes.crystal.atom_mass[None, None, :,
                                  None, None]
        / modes.frequencies[:, :, None,
                            None, None].to('Hz')
        * coth[:, :, None, None, None]
    ).to('angstrom^2')

    u2_weighted = (u2 * weights[:, None, None, None, None])
    a = u2_weighted[modes.frequencies > 0].sum(axis=0)

    q_a_sum = np.zeros(len(a))
    for direction_vector in golden_sphere(npts, cartesian=True, jitter=False):
        q_vector = q * direction_vector
        q_a_sum += (q_vector @ a @ q_vector).real

    return -q_a_sum / npts
    
# def isotropic_dw(modes: QpointPhononModes,
#                  temperature: Quantity = 298. * ureg('K'),
#                  q: Quantity = 1. * ureg('1 / angstrom'),
#                  ref='Mitchell2005') -> np.ndarray:
#     """Calculate Debye-Waller exponent in isotropic approximation

#     i.e. for a Debye-Waller factor exp(-2W) this is -2W

#     This code is initially being used to check equivalence
#     of various descriptions in literature

#     Args:
#         modes: Phonon modes on input q-point mesh
#         temperature: System temperature
#         q: Absolute (scalar) value of q

#     """

#     # Get normalised q-point weights
#     weights = modes.weights / np.sum(modes.weights)

#     coth = 1 / np.tanh(ureg('hbar').to('eV s') * modes.frequencies.to('rad / s')
#                        / (2. * ureg('k').to('eV / K') * temperature.to('K')))

#     u2 = (modes.eigenvectors * ureg('hbar')
#           / 2
#           / modes.crystal.atom_mass[np.newaxis, np.newaxis, :, np.newaxis]
#           / modes.frequencies[:, :, np.newaxis, np.newaxis].to('Hz')
#           * coth[:, :, np.newaxis, np.newaxis]
#           ).to('angstrom^2')

#     u_coth_sum = np.sqrt(u2) * coth[:, :, np.newaxis, np.newaxis] * weights[:, np.newaxis, np.newaxis, np.newaxis]


#     u_coth_sum = np.einsum('ijkl->k', u_coth_sum)
    # u2_sum = -q**2 * u2 * (coth**2)[:, :, np.newaxis, np.newaxis] * weights[:, np.newaxis, np.newaxis, np.newaxis]
    # u2_sum = np.einsum('ijkl->il', u2)

    # return -q**2 * u2_sum
#     return - (np.dot(q, u_coth_sum))**2


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    from euphonic.cli.utils import load_data_from_file

    parser = ArgumentParser()
    parser.add_argument('filename', type=Path, help='Data file with QpointPhononModes')

    args = parser.parse_args()

    modes = load_data_from_file(args.filename)
    if not isinstance(modes, QpointPhononModes):
        raise TypeError("Data could not be interpreted as QpointPhononModes")

    print(anisotropic_dw(modes))

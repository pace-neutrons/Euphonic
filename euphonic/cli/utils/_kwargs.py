"""Functions extracting useful groups of kwargs from the argparse Namespace"""

from argparse import Namespace
from typing import Any


def _plot_label_kwargs(args: Namespace, default_xlabel: str = '',
                       default_ylabel: str = '') -> dict[str, str]:
    """Collect title/label arguments that can be passed to plot_nd
    """
    plot_kwargs = {'title': args.title,
                   'xlabel': getattr(args, 'xlabel', None) or default_xlabel,
                   'ylabel': getattr(args, 'ylabel', None) or default_ylabel}
    return plot_kwargs


def _calc_modes_kwargs(args: Namespace) -> dict[str, Any]:
    """
    Collect arguments that can be passed to
    ForceConstants.calculate_qpoint_phonon_modes()
    """
    return {'asr': args.asr, 'dipole_parameter': args.dipole_parameter,
            'use_c': args.use_c, 'n_threads': args.n_threads}

def _brille_calc_modes_kwargs(args: Namespace) -> dict[str, Any]:
    """
    Collect arguments that can be passed to
    BrilleInterpolator.calculate_qpoint_phonon_modes()
    """
    if args.n_threads is None:
        # Nothing specified, allow defaults
        return {}

    return {'useparallel': args.n_threads > 1, 'threads': args.n_threads}


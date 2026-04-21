from collections.abc import Sequence

from euphonic import (
    Spectrum1D,
    Spectrum1DCollection,
)
from euphonic.util import (
    format_error,
)


def _arrange_pdos_groups(pdos: Spectrum1DCollection,
                         cl_arg_pdos: Sequence[str],
                         ) -> Spectrum1D | Spectrum1DCollection:
    """
    Convert PDOS returned by calculate_pdos to PDOS/DOS
    wanted as CL output according to --pdos
    """
    dos = pdos.sum()
    if cl_arg_pdos is not None:
        # Only label total DOS if there are other lines on the plot
        dos.metadata['label'] = 'Total'
        pdos = pdos.group_by('species')
        for line_metadata in pdos.metadata['line_data']:
            line_metadata['label'] = line_metadata['species']
        if len(cl_arg_pdos) > 0:
            pdos = pdos.select(species=cl_arg_pdos)
            dos = pdos
        else:
            dos = Spectrum1DCollection.from_spectra([dos, *pdos])
    return dos


def _get_pdos_weighting(cl_arg_weighting: str) -> str | None:
    """
    Convert CL --weighting to weighting for calculate_pdos
    e.g. --weighting coherent-dos to weighting=coherent
    """
    if cl_arg_weighting == 'dos':
        return None

    idx = cl_arg_weighting.rfind('-')
    if idx == -1:
        msg = format_error(
            f'Unexpected weighting "{cl_arg_weighting}"',
            fix='Check weighting argument. Should be e.g. "coherent-dos".',
       )
        raise ValueError(msg)

    return cl_arg_weighting[:idx]

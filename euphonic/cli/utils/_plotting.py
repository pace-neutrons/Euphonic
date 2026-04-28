from argparse import (
    Namespace,
)
from pathlib import Path

from euphonic import (
    ureg,
)

MplStyle = str | dict[str, str]


def matplotlib_save_or_show(save_filename: Path | str | None = None) -> None:
    """
    Save or show the current matplotlib plot.
    Show if save_filename is not None which by default it is.

    Parameters
    ----------
    save_filename
        The file to save the plot in
    """
    import matplotlib.pyplot as plt
    if save_filename is not None:
        plt.savefig(save_filename)
        print(f'Saved plot to {Path(save_filename).resolve()}')
    else:
        plt.show()


def _compose_style(
        *, user_args: Namespace, base: list[MplStyle] | None,
        ) -> list[MplStyle]:
    """Combine user-specified style options with default stylesheets

    Args:
        user_args: from _get_cli_parser().parse_args()
        base: Euphonic default styles for this plot

    N.B. matplotlib applies styles from left to right, so the right-most
    elements of the list take the highest priority. This function builds a
    list in the order:

    [base style(s), user style(s), CLI arguments]
    """

    style = base if not user_args.no_base_style and base is not None else []

    if user_args.style:
        style += user_args.style

    # Explicit args take priority over any other
    explicit_args = {}
    for user_arg, mpl_property in {'cmap': 'image.cmap',
                                   'fontsize': 'font.size',
                                   'font': 'font.sans-serif',
                                   'linewidth': 'lines.linewidth',
                                   'figsize': 'figure.figsize'}.items():
        if getattr(user_args, user_arg, None):
            explicit_args.update({mpl_property: getattr(user_args, user_arg)})

    if 'font.sans-serif' in explicit_args:
        explicit_args.update({'font.family': 'sans-serif'})

    if 'figure.figsize' in explicit_args:
        dimensioned_figsize = [dim * ureg(user_args.figsize_unit)
                               for dim in explicit_args['figure.figsize']]
        explicit_args['figure.figsize'] = [dim.to('inches').magnitude
                                           for dim in dimensioned_figsize]

    style.append(explicit_args)
    return style


def _get_title(filename: str, title: str | None = None) -> str:
    """Get a plot title: either user-provided string, or from filename"""
    return title if title is not None else Path(filename).stem

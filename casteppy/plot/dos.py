import numpy as np
import matplotlib.pyplot as plt


def plot_dos(dos, dos_down, bins, units, title='', fermi=[], mirror=False):
    """
    Creates a Matplotlib figure of the density of states

    Parameters
    ----------
    dos : list of floats
        L - 1 length list of the spin up density of states for each bin, where
        L is the lengh of the bins parameter. Can be empty if only spin down
        frequencies are present
    dos_down : list of floats
        L - 1 length list of the spin down density of states for each bin,
        where L is the lengh of the bins parameter. Can be empty if only spin
        up frequencies are present
    bins : list of floats
        One dimensional list of the energy bin edges
    units : string
        String specifying the energy bin units. Used for axis labels
    title : string
        The figure title. Default: ''
    fermi : list of floats
        1 or 2 length list specifying the fermi energy/energies. Default: []
    mirror : boolean
        Whether to reflect the dos_down frequencies in the x-axis.
        Default: False

    Returns
    -------
    fig : Matplotlib Figure
        Figure containing the subplot containing the plotted density of states
    """

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # X-axis label formatting
    # Replace 1/cm with cm^-1
    inverse_unit_index = units.find('/')
    if inverse_unit_index > -1:
        units = units[inverse_unit_index+1:]
        ax.set_xlabel('Energy (' + units + r'$^{-1}$)')
    else:
        ax.set_xlabel('Energy (' + units + ')')
    ax.set_ylabel('g(E)')
    ax.minorticks_on()

    # Calculate bin centres
    bwidth = bins[1] - bins[0]
    bin_centres = [b + bwidth/2 for b in bins[:-1]]

    # Plot dos and Fermi energy
    if len(dos) > 0:
        ax.plot(bin_centres, dos, label='alpha', lw=1.0)
    if len(dos_down) > 0:
        if mirror:
            ax.plot(bin_centres, np.negative(dos_down), label='beta', lw=1.0)
            ax.axhline(y=0, c='k', lw=1.0)
        else:
            ax.plot(bin_centres, dos_down, label='beta', lw=1.0)
    if len(fermi) > 0:
        for i, ef in enumerate(fermi):
            if i == 0:
                ax.axvline(x=ef, ls='dashed', c='k', label=r'$\epsilon_F$')
            else:
                ax.axvline(x=ef, ls='dashed', c='k')
    ax.legend()
    if not mirror:
        ax.set_ylim(bottom=0) # Need to set limits after plotting the data
    plt.tight_layout()

    return fig

import numpy as np
import matplotlib.pyplot as plt


def plot_dos(data, title='', mirror=False, up=True, down=True):
    """
    Creates a Matplotlib figure of the density of states

    Parameters
    ----------
    data : PhononData or BandsData object
        Data object for which calculate_dos has been called, containing dos
        and dos_bins attributes for plotting
    title : string
        The figure title. Default: ''
    mirror : boolean
        Whether to reflect the dos_down frequencies in the x-axis.
        Default: False
    up : boolean, optional
        Whether to plot spin up frequencies (if applicable). Default: True
    down : boolean, optional
        Whether to plot spin down frequencies (if applicable). Default: True

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
    units_str = '{:~P}'.format(data.dos_bins.units)
    inverse_unit_index = units_str.find('/')
    if inverse_unit_index > -1:
        units_str = units_str[inverse_unit_index+1:]
        ax.set_xlabel('Energy (' + units_str + r'$^{-1}$)')
    else:
        ax.set_xlabel('Energy (' + units_str + ')')
    ax.set_ylabel('g(E)')
    ax.minorticks_on()

    # Calculate bin centres
    bwidth = data.dos_bins[1] - data.dos_bins[0]
    bin_centres = data.dos_bins[:-1] + bwidth/2

    # Plot dos and Fermi energy
    if up:
        ax.plot(bin_centres, data.dos, label='alpha', lw=1.0)
    if down and hasattr(data, 'dos_down') and len(data.dos_down) > 0:
        if mirror:
            ax.plot(bin_centres, np.negative(data.dos_down), label='beta',
                    lw=1.0)
            ax.axhline(y=0, c='k', lw=1.0)
        else:
            ax.plot(bin_centres, data.dos_down, label='beta', lw=1.0)
    if hasattr(data, 'fermi'):
        for i, ef in enumerate(data.fermi.magnitude):
            if i == 0:
                ax.axvline(x=ef, ls='dashed', c='k', label=r'$\epsilon_F$')
            else:
                ax.axvline(x=ef, ls='dashed', c='k')
    ax.legend()
    if not mirror:
        ax.set_ylim(bottom=0)  # Need to set limits after plotting the data
    plt.tight_layout()

    return fig

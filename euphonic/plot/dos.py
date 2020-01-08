import sys
import numpy as np


def output_grace(data, seedname='out', mirror=False, up=True, down=True):
    """
    Creates a .agr Grace file of the dos

    Parameters
    ----------
    data : Data object
        Data object for which calculate_dos has been called, containing dos
        and dos_bins attributes for plotting
    seedname : string, optional
        Determines the figure title and output file name, seedname.agr.
        Default: 'out'
    mirror : boolean
        Whether to reflect the dos_down frequencies in the x-axis (if
        applicable)
        Default: False
    up : boolean, optional
        Whether to plot spin up dos (if applicable). Default: True
    down : boolean, optional
        Whether to plot spin down dos (if applicable). Default: True

    """
    try:
        from PyGrace.grace import Grace
    except ImportError:
        warnings.warn(('PyGrace is not installed, attempting to write .agr '
                       ' Grace file anyway. If using Python 2, you can install'
                       'PyGrace from https://github.com/pygrace/pygrace'),
                      stacklevel=2)

    # X-axis label formatting
    # Replace 1/cm with cm^-1
    units_str = data._e_units
    inverse_unit_index = units_str.find('/')
    if inverse_unit_index > -1:
        units_str = units_str[inverse_unit_index+1:]
        xaxis_label = 'Energy ({0}\\S-1\\N)'.format(units_str)
    else:
        xaxis_label = 'Energy ({0})'.format(units_str)
    yaxis_label = 'g(E)'
    legend_labels = ['\\f{Symbol}a\\f{}', '\\f{Symbol}b\\f{}',
                     '\\f{Symbol}e\\f{}\\sf\\N']

    # Calculate bin centres
    dos_bins = data.dos_bins.magnitude
    bwidth = dos_bins[1] - dos_bins[0]
    bin_centres = dos_bins[:-1] + bwidth/2

    if 'PyGrace' in sys.modules:
        grace = Grace()
        grace.background_fill = 'on'
        graph = grace.add_graph()
        graph.yaxis.label.text = yaxis_label
        graph.xaxis.label.text = xaxis_label

        if up:
            ds = graph.add_dataset(
                zip(bin_centres.magnitude, data.dos), legend=legend_labels[0])
            ds.line.configure(linewidth=2.0, color=2)
            ds.symbol.shape = 0
        if down and hasattr(data, 'dos_down') and len(data.dos_down) > 0:
            if mirror:
                ds = graph.add_dataset(
                    zip(bin_centres.magnitude, np.negative(data.dos_down)),
                    legend=legend_labels[1])
            else:
                ds = graph.add_dataset(
                    zip(bin_centres.magnitude, data.dos_down),
                    legend=legend_labels[1])
            ds.line.configure(linewidth=2.0, color=4)
            ds.symbol.shape = 0

        if hasattr(data, 'fermi'):
            for i, ef in enumerate(data.fermi.magnitude):
                if mirror:
                    ds = graph.add_dataset(
                        zip([ef, ef],
                            [-np.max(data.dos_down), np.max(data.dos)]))
                else:
                    ds = graph.add_dataset(
                        zip([ef, ef], [0, np.max([np.max(data.dos),
                                                  np.max(data.dos_down)])]))
                if i == 0:
                    ds.configure(legend=legend_labels[2])
                ds.line.configure(linewidth=2.0, color=1, linestyle=3)
                ds.symbol.shape = 0

        graph.set_world_to_limits()
        grace.write_file(seedname + '_dos.agr')

    else:
        with open(seedname + '_dos.agr', 'w') as f:
            f.write('@with g0\n')
            f.write('@title "{0}"\n'.format(seedname))
            f.write('@world xmin {0:.3f}\n'.format(dos_bins[0].magnitude))
            f.write('@world xmax {0:.3f}\n'.format(dos_bins[-1].magnitude))
            f.write('@view ymin 0.35\n')
            f.write('@view xmax 0.75\n')
            f.write('@legend 0.625, 0.825\n')
            f.write('@autoscale onread xyaxes\n')

            f.write('@yaxis label "{0}"\n'.format(yaxis_label))
            f.write('@xaxis label "{0}"\n'.format(xaxis_label))

            if mirror:
                f.write('@altxaxis  on\n')
                f.write('@altxaxis  type zero true\n')
                f.write('@altxaxis  tick off\n')
                f.write('@altxaxis  ticklabel off\n')

            # Write dos
            n_sets = 0
            if up:
                f.write('@G0.S{0:d} legend "{1}"\n'
                        .format(n_sets, legend_labels[0]))
                f.write('@G0.S{0:d} line color 2\n'.format(n_sets))
                f.write('@target G0.S{0:d}\n'.format(n_sets))
                f.write('@type xy\n')
                for i, x in enumerate(bin_centres.magnitude):
                    f.write('{0: .15e} {1: .15e}\n'.format(x, data.dos[i]))
                f.write('&\n')
                n_sets += 1
            if down and hasattr(data, 'dos_down') and len(data.dos_down) > 0:
                f.write('@G0.S{0:d} legend "{1}"\n'
                        .format(n_sets, legend_labels[1]))
                f.write('@G0.S{0:d} line color 4\n'.format(n_sets))
                f.write('@target G0.S{0:d}\n'.format(n_sets))
                f.write('@type xy\n')
                if mirror:
                    for i, x in enumerate(bin_centres.magnitude):
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(x, -data.dos_down[i]))
                else:
                    for i, x in enumerate(bin_centres.magnitude):
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(x, data.dos_down[i]))
                f.write('&\n')
                n_sets += 1

            # Write Fermi level
            if hasattr(data, 'fermi'):
                for i, ef in enumerate(data.fermi.magnitude):
                    if i == 0:
                        f.write('@G0.S{0:d} legend "{1}"\n'
                                .format(n_sets, legend_labels[2]))
                    f.write('@G0.S{0:d} line linestyle 3\n'.format(n_sets))
                    f.write('@G0.S{0:d} line color 1\n'.format(n_sets))
                    f.write('@target G0.S{0:d}\n'.format(n_sets))
                    f.write('@type xy\n')
                    if mirror:
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(ef, -np.max(data.dos_down)))
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(ef, np.max(data.dos)))
                    else:
                        f.write('{0: .15e} {1: .15e}\n'.format(ef, 0))
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(ef, np.max([np.max(data.dos),
                                                    np.max(data.dos_down)])))
                    f.write('&\n')
                    n_sets += 1


def plot_dos(data, title='', mirror=False, up=True, down=True, **line_kwargs):
    """
    Creates a Matplotlib figure of the density of states

    Parameters
    ----------
    data : Data object
        Data object for which calculate_dos has been called, containing dos
        and dos_bins attributes for plotting
    title : string
        The figure title. Default: ''
    mirror : boolean
        Whether to reflect the dos_down frequencies in the x-axis (if
        applicable). Default: False
    up : boolean, optional
        Whether to plot spin up dos (if applicable). Default: True
    down : boolean, optional
        Whether to plot spin down dos (if applicable). Default: True
    **line_kwargs : matplotlib.line.Line2D properties, optional
        Used in the axes.plot command to specify properties like linewidth,
        linestyle
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If matplotlib.pyplot can be imported, returns a Figure containing the
        subplot containing the plotted density of states, otherwise returns
        None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(('Cannot import Matplotlib to plot dos (maybe '
                       'Matplotlib is not installed?). To install Euphonic\'s'
                       ' optional Matplotlib dependency, try:\n\npip install'
                       ' euphonic[matplotlib]\n'))
        raise

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # X-axis label formatting
    # Replace 1/cm with cm^-1
    units_str = data._e_units
    inverse_unit_index = units_str.find('/')
    if inverse_unit_index > -1:
        units_str = units_str[inverse_unit_index+1:]
        ax.set_xlabel('Energy (' + units_str + r'$^{-1}$)')
    else:
        ax.set_xlabel('Energy (' + units_str + ')')
    ax.set_ylabel('g(E)')
    ax.minorticks_on()

    # Calculate bin centres
    dos_bins = data.dos_bins.magnitude
    bwidth = dos_bins[1] - dos_bins[0]
    bin_centres = dos_bins[:-1] + bwidth/2

    # Plot dos and Fermi energy
    if up:
        if hasattr(data, 'dos_down'):
            label = 'alpha'
        else:
            label = None
        ax.plot(bin_centres, data.dos, label=label, lw=1.0, **line_kwargs)
    if down and hasattr(data, 'dos_down') and len(data.dos_down) > 0:
        if mirror:
            ax.plot(bin_centres, np.negative(data.dos_down), label='beta',
                    lw=1.0, **line_kwargs)
            ax.axhline(y=0, c='k', lw=1.0)
        else:
            ax.plot(bin_centres, data.dos_down, label='beta', lw=1.0,
                    **line_kwargs)
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

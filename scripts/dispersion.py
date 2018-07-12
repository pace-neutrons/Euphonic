"""
Parse a *.castep, *.phonon or *.bands output file from new CASTEP for
vibrational frequency data and output a matplotlib plot of the electronic
or vibrational band structure or dispersion.
"""

import argparse
import matplotlib.pyplot as plt
import casteppy.general as cpy

def main():
    args = parse_arguments()
    ureg = cpy.set_up_unit_registry()

    # Read data
    with open(args.filename, 'r') as f:
        read_eigenvecs = args.reorder
        (cell_vec, ion_pos, ion_type, qpts, weights, freqs, freq_down,
            i_intens, r_intens, eigenvecs, fermi) = cpy.read_input_file(
                f, ureg, args.units, args.up, args.down, False, False,
                read_eigenvecs)

    # Calculate and plot dispersion
    # Reorder frequencies if eigenvectors have been read and the flag
    # has been set
    if eigenvecs.size > 0 and args.reorder:
        if freqs.size > 0:
            freqs = cpy.reorder_freqs(freqs, qpts, eigenvecs)
        if freq_down.size > 0:
            freq_down = cpy.reorder_freqs(freq_down, qpts, eigenvecs)

    # Get positions of q-points along x-axis
    recip_latt = cpy.reciprocal_lattice(cell_vec)
    abscissa = cpy.calc_abscissa(qpts, recip_latt)

    # Get labels for high symmetry / fractional q-point coordinates
    labels, qpts_with_labels = cpy.recip_space_labels(
        qpts, cell_vec, ion_pos, ion_type)

    fig = cpy.plot_dispersion(abscissa, freqs, freq_down, args.units,
                              args.filename, xticks=abscissa[qpts_with_labels],
                              xlabels=labels,
                              fermi=[f.magnitude for f in fermi],
                              btol=args.btol)

    # Save or show figure
    if args.s:
        plt.savefig(args.s)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Extract phonon or bandstructure data from .castep,
                       .phonon or .bands files and plot the band structure
                       with matplotlib""")
    parser.add_argument(
        'filename',
        help="""The .castep, .phonon or .bands file to extract the
                bandstructure data from""")
    parser.add_argument(
        '-v',
        action='store_true',
        help='Be verbose about progress')
    parser.add_argument(
        '-units',
        default='eV',
        help="""Convert frequencies to specified units for plotting (e.g
                1/cm, Ry). Default: eV""")
    parser.add_argument(
        '-s',
        default=None,
        help='Save resulting plot to a file')

    spin_group = parser.add_mutually_exclusive_group()
    spin_group.add_argument(
        '-up',
        action='store_true',
        help='Extract and plot only spin up from *.castep or *.bands')
    spin_group.add_argument(
        '-down',
        action='store_true',
        help='Extract and plot only spin down from *.castep or *.bands')

    disp_group = parser.add_argument_group(
        'Dispersion arguments',
        'Arguments specific to plotting the band structure')
    disp_group.add_argument(
        '-reorder',
        action='store_true',
        help="""Try to determine branch crossings from eigenvectors and
                rearrange frequencies accordingly (only applicable if
                using a .phonon file)""")
    disp_group.add_argument(
        '-btol',
        default=10.0,
        type=float,
        help="""The tolerance for plotting sections of reciprocal space on
                different subplots, as a fraction of the median distance
                between q-points. Default: 10.0""")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

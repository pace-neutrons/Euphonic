"""
Parse a *.castep, *.phonon or *.band output file from new CASTEP for
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
        (cell_vec, ion_pos, ion_type, qpts, weights, freqs, freq_down,
            i_intens, r_intens, eigenvecs, fermi) = cpy.read_input_file(
                f, ureg, args.units, args.up, args.down, args.ir, args.raman)

    # Calculate and plot DOS
    # Set default DOS bin and broadening width based on whether it's
    # electronic or vibrational
    if args.b == None:
        if f.name.endswith('.bands'):
            bwidth = 0.05*ureg.eV
        else:
            bwidth = 1.0*(1/ureg.cm)
        bwidth.ito(args.units, 'spectroscopy')
    else:
        bwidth = args.b*ureg[args.units]
    if args.w == None:
        if f.name.endswith('.bands'):
            gwidth = 0.1*ureg.eV
        else:
            gwidth = 10.0*(1/ureg.cm)
        gwidth.ito(args.units, 'spectroscopy')
    else:
        gwidth = args.w*ureg[args.units]

    if args.ir:
        dos, dos_down, bins = cpy.calculate_dos(
            freqs, freq_down, weights, bwidth.magnitude, gwidth.magnitude,
            args.lorentz, intensities=i_intens)
    else:
        dos, dos_down, bins = cpy.calculate_dos(
            freqs, freq_down, weights, bwidth.magnitude, gwidth.magnitude,
            args.lorentz)

        fig = cpy.plot_dos(dos, dos_down, bins, args.units, args.filename,
                       fermi=[f.magnitude for f in fermi], mirror=args.mirror)

    # Save or show figure
    if args.s:
        plt.savefig(args.s)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Extract phonon or bandstructure data from .castep,
                       .phonon or .bands files and plot the band structure
                       (default) or density of states with matplotlib""")
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

    dos_group = parser.add_argument_group(
        'DOS arguments',
        'Arguments specific to plotting the density of states')
    dos_group.add_argument(
        '-dos',
        action='store_true',
        help='Plot density of states instead of a dispersion plot')
    dos_group.add_argument(
        '-ir',
        action='store_true',
        help='Extract IR intensities from .phonon and use to weight DOS')
    dos_group.add_argument(
        '-raman',
        action='store_true',
        help="""Extract Raman intensities from .phonon and calculate a Raman
                spectrum""")
    dos_group.add_argument(
        '-w',
        default=None,
        type=float,
        help="""Set Gaussian/Lorentzian FWHM for broadening (in units specified
                by -units argument or default eV). Default: 0.1 eV for
                electronic DOS, 10.0/cm for vibrational DOS""")
    dos_group.add_argument(
        '-b',
        default=None,
        type=float,
        help="""Set histogram resolution for binning (in units specified by
                -units argument or default eV). Default: 0.05 eV for electronic
                DOS, 1.0/cm for vibrational DOS""")
    dos_group.add_argument(
        '-lorentz',
        action='store_true',
        help='Use Lorentzian broadening instead of Gaussian')
    dos_group.add_argument(
        '-mirror',
        action='store_true',
        help='Plot spin down electronic DOS mirrored in the x axis')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

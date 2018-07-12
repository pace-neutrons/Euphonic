import numpy as np
import casteppy.parsers.bands as bands
import casteppy.parsers.phonon as phonon

def read_input_file(f, ureg, units, up=False, down=False, ir=False,
                    raman=False, read_eigenvecs=False):
    """
    Reads data from a .bands, .phonon, or .castep_bin file. If the specified
    file doesn't contain all the required data (e.g. ionic positions) looks in
    other .bands, .phonon or .castep_bin files of the same structure name

    Parameters
    ----------
    f : file object
        File object in read mode for the file containing the data
    ureg : UnitRegistry
        Unit registry (from Pint module) for converting frequencies and Fermi
        energies to units specified by units argument
    units : string
        String specifying the units of the output frequencies. For valid
        values see the Pint docs: http://pint.readthedocs.io/en/0.8.1/
    up : boolean
        Read only spin up frequencies from the .bands file. Default: False
    down : boolean
        Read only spin down frequencies from the .bands file. Default: False
    ir : boolean
        Whether to read and store IR intensities (only applicable if
        reading a .phonon file). Default: False
    raman : boolean
        Whether to read and store Raman intensities (only applicable if
        reading a .phonon file). Default: False
    read_eigenvecs : boolean
        Whether to read and store the eigenvectors (only applicable if
        reading a .phonon file). Default: False

    Returns
    -------
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    ion_pos : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell. If the ion positions can't be found, ion_pos is empty
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_pos. If the ion types can't
        be found, ion_type is empty
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    weights : list of floats
        List of length M containing the weight for each q-point, where
        M = number of q-points
    freqs : list of floats
        M x N list of band frequencies in units specified by the
        'units' parameter, where M = number of q-points and N = number of
        bands. If there are 2 spin components, these are the spin up
        frequencies. If -down is specified, freqs is empty
    freq_down : list of floats
        M x N list of spin down band frequencies in units specified by the
        'units' parameter, where M = number of q-points and N = number of
        bands. If there are no spin down frequencies, or if -up is specified
        freq_down is empty
    i_intens: list of floats
        M x N list of IR intensities, where M = number of q-points and
        N = number of bands. Empty if no IR intensities in .phonon file
    r_intens: list of floats
        M x N list of Raman intensities, where M = number of q-points and
        N = number of bands. Empty if no Raman intensities in .phonon file
    eigenvecs: list of complex floats
        M x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points and
        L = number of ions*number of bands. If read_eigenvecs is False,
        eigenvecs is empty
    fermi : list of floats
        List of length 1 or 2 containing the Fermi energy/energies in atomic
        units. If the file doesn't contain the fermi energy, fermi is empty
    """
    ion_pos = []
    ion_type = []
    freqs = np.array([])
    freq_down = np.array([])
    i_intens = np.array([])
    r_intens = np.array([])
    eigenvecs = np.array([])
    fermi = np.array([])

    # If file is in this directory
    if f.name.rfind('/') == -1:
        dir_name = '.'
        structure_name = f.name[:f.name.rfind('.')]
    # If file is in another directory
    else:
        dir_name = f.name[:f.name.rfind('/')]
        structure_name = f.name[f.name.rfind('/') + 1:f.name.rfind('.')]

    if f.name.endswith('.bands'):
        (fermi, cell_vec, qpts, weights, freqs,
            freq_down) = bands.read_dot_bands(f, ureg, up, down, units)
        # Try and get point group info by reading ionic positions from .phonon
        phonon_file = dir_name + '/' + structure_name + '.phonon'
        try:
            with open(phonon_file, 'r') as pf:
                ion_pos, ion_type = phonon.read_dot_phonon_header(pf)[4:6]
        except IOError:
            pass

    elif f.name.endswith('.phonon'):
        (cell_vec, ion_pos, ion_type, qpts, weights, freqs, i_intens, r_intens,
            eigenvecs) = phonon.read_dot_phonon(f, ureg, units, ir, raman,
                                         read_eigenvecs)

    else:
        sys.exit('Error: Please supply a .bands or .phonon file')

    return (cell_vec, ion_pos, ion_type, qpts, weights, freqs, freq_down,
            i_intens, r_intens, eigenvecs, fermi)


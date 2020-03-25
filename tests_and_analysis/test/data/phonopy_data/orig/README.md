Phonopy runs in three modes; band, mesh, qpoints, which have files named after them.

The commands to generate required test data from VASP files are given in these two links

    [phonopy_vasp](https://phonopy.github.io/phonopy/vasp.html)

VASP also offers DFTP, but the above procedure is common to most interfaces.

    [phonopy_vasp_dftp](https://phonopy.github.io/phonopy/vasp-dfpt.html)

To summarise the commands:

Generate force sets:

    phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml
    
Run phonopy post-process to get output data (band.yaml/band.hdf5), phonopy.yaml
and force constants. It will try to use FORCE_SETS, then output force constants
if writefc is set. If FORCE_CONSTANTS exists `FORCE_CONSANTS = READ` can be set
in the config:

    phonopy -p band.conf [--hdf5] [--writefc] [--nac]

All options have corresponding command line flags and config file entries, see
the command options/settings tags pages in the phonopy documentation. The config
files are set to output all data possible using `INCLUDE_ALL = .TRUE.` to the
phonopy.yaml and FORCE_CONSTANTS. If born effective charge and dielectric constant
are required, `--nac` can be set.

class Data(object):
    """
    A general superclass to store data read from CASTEP files
    """

    def convert_e_units(self, units):
        """
        Convert energy units of relevant attributes in place e.g. dos_bins

        Parameters
        ----------
        units : str
            The units to convert to e.g. '1/cm', 'hartree', 'eV'
        """
        if hasattr(self, 'dos_bins'):
            self.dos_bins.ito(units, 'spectroscopy')

        if hasattr(self, 'sqw_ebins'):
            self.sqw_ebins.ito(units, 'spectroscopy')

    def convert_l_units(self, units):
        """
        Convert length units of relevant attributes in place e.g. cell_vec

        Parameters
        ----------
        units : str
            The units to convert to e.g. 'angstrom', 'bohr'
        """
        self.cell_vec.ito(units)

class Data(object):
    """
    A general superclass to store data read from CASTEP files
    """
    def convert_e_units(self, units):
        if hasattr(self, 'dos_bins'):
            self.dos_bins.ito(units, 'spectroscopy')


    def convert_l_units(self, units):
        self.cell_vec.ito(units)
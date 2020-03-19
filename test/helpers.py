import numpy as np
from euphonic import ureg
from euphonic.crystal import Crystal

def mock_crystal(cell_vectors):
            crystal = Crystal(cell_vectors, 1, np.array([0., 0., 0.]),
                              np.array(['mock']), np.array(1.)*ureg('amu'))
            return crystal
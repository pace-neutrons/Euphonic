.. _data_objs:

=====================
Euphonic Data Objects
=====================

InterpolationData
=================

This object can be used to calculate phonon frequencies and eigenvectors at any
arbitrary q via Fourier interpolation of a force constants matrix. This object
contains the force constants, and any additional information (e.g. cell vectors,
ionic positions, supercell matrix) required for interpolation. For how to create
an ``InterpolationData`` object, see the **Reading Data** tutorial section for
the code you want to use. More details on the attributes/methods
:ref:`here <interpolation_data>`

PhononData
==========

This object contains precalculated phonon frequencies and eigenvectors at
certain q-points, but no force constants matrix information so cannot be used
for interpolation. For more information on how to create a ``PhononData``
object, see the **Reading Data** tutorial section for the code you want to use.
More details on the attributes/methods :ref:`here <phonon_data>`

BandsData
=========

This object contains **electronic** frequencies at certain k-points, and can be
used for dispersion plots or simple DOS. Its use is limited and only supports
reading CASTEP ``.bands`` files. More details on the attributes/methods
:ref:`here <bands_data>`



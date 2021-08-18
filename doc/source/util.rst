.. _utils:

Utilities
=========

.. _ref_data:

Reference data
--------------

Reference data is stored as JSON files and accessed using :py:func:`euphonic.util.get_reference_data`.
As well as inbuilt data sets (e.g. ``"Sears1992"``), user files may be specified.
For examples of the data format, see the documentation of that function or
`take a look <https://github.com/pace-neutrons/Euphonic/blob/master/euphonic/data/sears-1992.json>`_
in the Euphonic source repository. The key point is that each "collection" file should contain a dictionary
with a ``"physical_properties"`` key corresponding to another dictionary of reference data.
The units should be specified with the special key ``__units__``; this will automatically
be used to wrap the data to ``pint.Quantity`` when accessed.

euphonic.util module
--------------------

.. automodule:: euphonic.util
   :members:
   :undoc-members:
   :show-inheritance:

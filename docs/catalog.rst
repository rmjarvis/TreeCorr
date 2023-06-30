Input Data
==========

The Catalog class
-----------------

.. autoclass:: treecorr.Catalog
    :members:

Other utilities related to catalogs
-----------------------------------

.. autofunction::
    treecorr.read_catalogs
.. autofunction::
    treecorr.calculateVarK
.. autofunction::
    treecorr.calculateVarG
.. autofunction::
    treecorr.calculateVarV
.. automodule:: treecorr.catalog
    :members:
    :exclude-members: Catalog, read_catalogs, calculateVarG, calculateVarK, calculateVarV

File Readers
------------

.. autoclass:: treecorr.reader.FitsReader
    :members:
.. autoclass:: treecorr.reader.HdfReader
    :members:
.. autoclass:: treecorr.reader.AsciiReader
    :members:
.. autoclass:: treecorr.reader.PandasReader
    :members:
.. autoclass:: treecorr.reader.ParquetReader
    :members:

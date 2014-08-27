The corr2 driver script
=======================

The corr2 function from python
------------------------------

.. autofunction:: treecorr.corr2

The corr2 executable
--------------------

Also installed with TreeCorr is an executable script, called corr2.
The script takes one required command-line argument, which is the 
name of a configuration file:

    corr2 config_file

A sample configuration file is provided, called sample.params.  See the
TreeCorr wiki page

https://github.com/rmjarvis/TreeCorr/wiki/Configuration-Parameters

for the complete documentation about the allowed parameters.

You can also specify parameters on the command line after the name of 
the configuration file. e.g.:

    corr2 config_file file_name=file1.dat g2_file_name=file1.out
    corr2 config_file file_name=file2.dat g2_file_name=file2.out
    ...

This can be useful when running the program from a script for lots of input 
files.

Other utilities related to corr2
--------------------------------

.. automodule:: treecorr.corr2
    :members:
    :undoc-members:
    :exclude-members: corr2


Utilities related to the configuration dict
-------------------------------------------

.. automodule:: treecorr.config
    :undoc-members:
    :members:


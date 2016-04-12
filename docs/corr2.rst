The corr2 and corr3 driver scripts
==================================

The corr2 function from python
------------------------------

.. autofunction:: treecorr.corr2

The corr3 function from python
------------------------------

.. autofunction:: treecorr.corr3

The corr2 and corr3 executables
-------------------------------

Also installed with TreeCorr are two executable scripts, called corr2
and corr3.  The scripts takes one required command-line argument, which
is the name of a configuration file::

    corr2 config.yaml
    corr3 config.yaml

A sample configuration file is provided, called sample_config.yaml.
See the TreeCorr wiki page

https://github.com/rmjarvis/TreeCorr/wiki/Configuration-Parameters

for the complete documentation about the allowed parameters.

YAML is the recommended format for the configuration file, but we
also allow JSON files if you prefer, or a legacy format, which is
like an .ini file, but without the section headings, consisting of
key = value lines.  The three formats are normally distinguished
by their extensions (.yaml, .json, or .params respectively), but
you can also give the file type explicitly with the -f option. E.g.::

    corr2 my_config_file.txt -f params

would specify that the configuration file ``my_config_file.txt`` uses
the legacy "params" format.

You can also specify parameters on the command line after the name of 
the configuration file. e.g.::

    corr2 config.yaml file_name=file1.dat gg_file_name=file1.out
    corr2 config.yaml file_name=file2.dat gg_file_name=file2.out
    ...

This can be useful when running the program from a script for lots of input 
files.

Other utilities related to corr2 and corr3
------------------------------------------

.. automodule:: treecorr.corr2
    :members:
    :exclude-members: corr2

.. automodule:: treecorr.corr3
    :members:
    :exclude-members: corr3


Utilities related to the configuration dict
-------------------------------------------

.. automodule:: treecorr.config
    :members:


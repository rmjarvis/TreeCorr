Using configuration files
=========================

Most of the TreeCorr classes can take a ``config`` parameter in lieu
of a set of keyword arguments.  This is not necessarily incredibly
useful when driving the code from Python; however, it enables running
the code from some executable scripts, described below.

Specifically, the parameters defined in the configuration file are
loaded into a Python dict, which is passed to each of the classes
as needed.  The advantage of this is that TreeCorr will only use the
parameters it actually needs when initializing each object.
Any additional parameters (e.g. those
that are relevant to a different class) are ignored.

The corr2 and corr3 executables
-------------------------------

Along with the installed Python library, TreeCorr also includes
two executable scripts, called ``corr2`` and ``corr3``.
The scripts takes one required command-line argument, which
is the name of a configuration file::

    corr2 config.yaml
    corr3 config.yaml

A sample configuration file is provided, called sample_config.yaml.

For the complete documentation about the allowed parameters, see:

.. toctree::

    params

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

Python API and corr2/corr3 parity
---------------------------------

The ``corr2`` and ``corr3`` executables and the Python ``treecorr.corr2`` /
``treecorr.corr3`` functions use the same configuration logic and core processing
pipeline, so they produce matching results when given equivalent configs and inputs.

What is equivalent:

* Same parameter names and meanings from :doc:`params`.
* Same correlation calculations and estimators.
* Same output products when the same output file options are set.

What differs:

* The executables are file-driven, one-shot runs from the command line.
* Direct class usage in Python (`Catalog`, `Corr2`, `Corr3`) provides lower-level
  control, such as:

  - splitting processing into ``process_auto`` / ``process_cross`` / ``finalize``
  - custom post-processing with ``estimate_cov(..., func=...)`` and
    ``estimate_multi_cov(..., func=...)``
  - explicit in-memory workflows without writing intermediate files

When to choose which interface:

* Use ``corr2``/``corr3`` for reproducible, config-driven batch jobs.
* Use direct Python classes for iterative analysis, custom data vectors, and
  non-standard control flow.

The corr2 function from python
------------------------------

The same functionality that you have from the ``corr2`` executable is available in python via the
`corr2` function::

    import treecorr
    config = treecorr.read_config(config_file)
    config['file_name'] = 'catalog.dat'
    config['gg_file_name'] = 'gg.out'
    treecorr.corr2(config)

.. autofunction:: treecorr.corr2

The corr3 function from python
------------------------------

.. autofunction:: treecorr.corr3


Other utilities related to corr2 and corr3
------------------------------------------

.. autofunction:: treecorr.exec_corr2.print_corr2_params

.. autofunction:: treecorr.exec_corr3.print_corr3_params


Utilities related to the configuration dict
-------------------------------------------

.. automodule:: treecorr.config
    :members:


File Writers
------------

.. autoclass:: treecorr.writer.FitsWriter
    :members:
.. autoclass:: treecorr.writer.HdfWriter
    :members:
.. autoclass:: treecorr.writer.AsciiWriter
    :members:

Changes from version 3.2 to 3.3
===============================

The numbers at the ends of some items below indicate which issue is connected
with the change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+3.3%22+is%3Aclosed

Dependency change:
------------------

- Switched from using ctypes (which is native to python) to cffi for the C++
  wrapping.  This should be a seamless change, since setup.py should download
  cffi for you if you need it, but it does require libffi to be installed
  on your system.  This is usually already true, but if not, see the cffi
  docs here: https://cffi.readthedocs.org/en/latest/installation.html
  for platform-specific instructions for installing libffi.
- Made fitsio a required dependency, rather than allowing astropy.io.fits or
  pyfits as an optional way to read/write FITS files.  The fitsio package is
  almost always faster, and it's an easy dependency, since it is installable
  via either pip or easy_install.


API changes:
------------

- Changed the default configuration format to be based on the file extension.
  If your config file is named _something_.params, then it will still work
  the same way.  Otherwise, you may need to add '-f params' when you run
  corr2 or corr3. (#18)
- Changed the attributes ``min_sep``, ``max_sep``, and ``logr`` to have units
  of ``sep_units`` if they are given, rather than radians.  The previous
  behavior was somewhat confusing.  (#39)
- Changed the output separations to convert from the chord distance to the
  corresponding great circle distance when using spherical coordinates.  (#39)
- Removed the ``z_units`` option for ``Catalog``, since it doesn't actually
  make sense.  Any ``z`` field should be in physical units, not arcsec,
  degrees, etc.


New features:
-------------

- Added support for YAML and JSON configuration files.  YAML is now the
  recommended format.  If your config file ends with '.yaml', then this
  will automatically be used.  If it ends with '.params', the old parameter
  list style will be used.  And if it ends with '.json', JSON will be used.
  You can also specify the format directly with the -f (or --file_type)
  command-line argument for corr2 and corr3. (#18)
- Added ``min_rpar`` and ``max_rpar`` options for the Rperp metric (as well
  as the new Rlens metric -- see below) to set the minimum and maximum
  separations in the r_parallel direction. (#34)
- Added new metric Arc, which calculates the true great circle separation
  throughout the calculation, rather than calculating chord distances and
  then converting to the corresponding angles at the end.  Most people will
  prefer the speed of the chord distances, but this provides a way to compare
  the accuracy of the two approaches. (#39)
- Added a new split_method='random', which chooses a random point between the
  25th percentile and 75th percentile at which to make the split each time.
  This is potentially useful when the large-scale shape of your field is very
  regular (e.g. a perfect rectangle) to avoid numerical issues related to the
  larger cells all having identical shapes.  (#40)
- Added new metric Rlens, which is the projected separation at the distance of
  the lens (here taken to be the object in the first catalog of the cross-
  correlation). (#41)
- Changed the write commands to create the output directory if necessary. (#42)
- Made the C and C++ header files be installed along with the python code and
  the compiled library.  So in case anyone wants to use the C++ code directly
  in another application, rather than through the python interface, that's now
  easier to do.  The directory to add to the include path is available as
  ``treecorr.include_dir`` and the library to link is ``treecorr.lib_file``.


Bug fixes:
----------

- Fixed a bug where num_threads=1 didn't actually do anything, so it wasn't
  possible to get TreeCorr not to use multi-threading.  Thanks to Eric Baxter
  for this bug report.
- Fixed a big where 3-pt correlations could sometimes seg fault due to a
  calculation coming out as nan. (#42)


Version 3.3.1 bug fix:
----------------------

- Fixed a bug where numerical rounding imprecision could let dsq come out
  negative for the Rperp metric if the lens and source were identical (i.e.
  had the same position, so dsq should be 0).


Version 3.3.2 bug fix:
----------------------

- Fixed a problem with pip installation on linux boxes that the include files
  were not being installed into the right place.


Version 3.3.3 bug fix:
----------------------

- Fixed an error when writing output files in the current directory.


Version 3.3.4 bug fix:
----------------------

- Fixed an error in how the corr3 script calculates the compensated nnn
  statistic.


Version 3.3.5 bug fixes:
------------------------

- Added an exception if sep_units is used with either Rperp or Rlens.  This
  would really be a user error, not a TreeCorr bug, but it was not obvious
  that this was invalid, so raising the exception should help users notice
  the error.
- Added a timeout for the ffi question in setup.py to avoid pip installation
  hanging. (#45)

Version 3.3.6 bug fixes:
------------------------

- Fixed an import error in some installations with Python 3 where the library
  file gets a different name like _treecorr.cpython-34m.so or similar. (#49)
- Added a bit more output when running with the default verbose=1.

Version 3.3.7 dependency change:
--------------------------------

- Changed the pandas dependency specification from pandas==0.18 to just pandas.
  The pandas 0.19.0rc1 version had been on pip was broken, so we used 0.18.
  But they fixed that, so we removed the restriction. (#59)

Version 3.3.8 installation fix
------------------------------

- Include the LICENSE in the manifest.

Version 3.3.9 installation fix
------------------------------

- Changed how we determine the C compiler that setuptools will use.  The old
  method could sometimes get the wrong one, which could lead to errors in the
  correct OpenMP flag to use.

Version 3.3.10 installation fix
-------------------------------

- Fixed how setup.py decides whether clang compiler can use OpenMP.

Version 3.3.11 installation fix
-------------------------------

- Added more things to MANIFEST since pip no longer includes everything it
  needs automatically, so pip install treecorr was failing.

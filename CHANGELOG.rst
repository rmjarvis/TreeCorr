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


API changes:
------------

- Change Field classes to take ``min_size`` and ``max_size`` directly, rather
  than having these be calculated from ``min_sep``, ``max_sep``, ``b`` and
  ``metric``.


New features:
-------------

- Added a new split_method='random', which chooses a random point between the
  25th percentile and 75th percentile at which to make the split each time.
  This is potentially useful when the large-scale shape of your field is very
  regular (e.g. a perfect rectangle) to avoid numerical issues related to the
  larger cells all having identical shapes.  (#40)
- Made the C and C++ header files installed with the python code and the
  compiled library.  So in case anyone wants to use the C++ code directly in
  another application, rather than through the python interface, that's now
  easier to do.  The directory to add to the include path is available as
  `treecorr.include_dir` and the library to link is `treecorr.lib_file`.


Bug fixes:
----------


Changes from version 3.2.0 to 3.2.1
===================================

Bug fixes:
----------

- Fixed a bug for GG cross-correlation that somehow there wasn't a unit test
  for (and now there is).



Changes from version 3.1 to 3.2
===============================

The numbers at the ends of some items below indicate which issue is connected
with the change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+3.2%22+is%3Aclosed


API changes:
------------

- Changed the default value of ``bin_slop`` so that ``bin_slop * bin_size`` is
  at most 0.1.  When this product (which I refer to as b) is larger than 0.1,
  there can be significant inaccuracies in the resulting correlation function.
  Now, if ``bin_slop`` is unspecified, it is set to ``0.1/bin_size`` if 
  ``bin_size > 0.1``, or just ``1.0`` if ``bin_size <= 0.1``. (#15)
- Changed output columns that were labeled with ``<...>`` to instead use
  ``mean`` in front.  e.g. ``<R>`` -> ``meanR``.  Some programs don't handle 
  column names with punctuation well, so this avoids the issue.  Similarly,
  changed ``xi+`` and ``xi-`` to ``xip``, ``xim`` and ``<Map^2>`` to ``Mapsq``.
- Changed the ``<R>`` (now ``meanR``) column in the output to actually be the
  mean value of R for that bin. It used to really be exp(<logR>).  Also added 
  a ``meanlogR`` column.  This means if you were accessing the output columsn
  by column number, you will need to change your code to add 1, since most of
  the columns have been bumped 1 column to the right.  (While you're at it,
  you should switch to accessing them by column name instead, maybe via the
  FITS I/O option...) (#28)


New features:
-------------

- Added three point functions.  So far just NNN, KKK, and GGG are implemented.
  See the docstrings for the classes ``NNNCorrelation``, ``GGGCorrelation``,
  and ``KKKCorrelation`` for details about how to use them. (#3, #4, #33)
- Added ability to use the perpendicular component of the distance as the
  separation metric with ``process(cat, metric='Rperp')`` (#17)
- Added the ability for the Correlation classes to write to FITS binary tables.
  If an output file name has an extension that starts with '.fit', then it will
  write a FITS binary table, rather than an ASCII file.  Or you can specify the
  file type with the ``file_type`` parameter. (#22)
- Added a read() function to the Correlation classes to read data back in from
  and output file.  The combination of this with the FITS write lets you save a
  Correlation object to disk and recover it without loss of information.  This
  can be used to split a problem up to be run on multiple machines. (#23)
- Added ability to add one Correlation object to another to enable doing the
  calculation in parts over several processes. (#23)
- Added the option of calling ``NNCorrelation.write()`` with no ``rr``
  parameter.  In this case, it will just write a column with the number of
  pairs (along with the regular separation columns), rather than calculate the
  correlation function.
- Added ability to write Catalogs to either FITS or ASCII files.
- Added ability to use different weights for the position centroiding than for
  the regular weighted values being correlated.  This enables jackknife 
  calculations to have the exact same tree structure each time by setting
  the value weights ``w=0`` for the points to exclude each time, but keep
  ``wpos`` the same so the cells are identical.  This removes some numerical
  noise in the calculation that otherwise occurs. (#31)
- Added ability to specify 3D coordinates with (x,y,z) in addition to using
  (ra,dec,r).


Bug fixes:
----------

- Added a ``num_threads`` parameter to the various ``process`` functions.  The
  'num_threads' parameter had been checked only when loading the config dict
  from a file, which meant that there was no obvious way to change the number
  of threads in the python layer. This is now possible. (#21)
- Fixed NN correlations to properly use weight values.  This is particularly
  useful for setting some points to have w=0 to exclude them from the sums,
  which didn't used to work. (#29)

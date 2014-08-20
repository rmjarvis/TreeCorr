Changes from 2.6 to 3.0
=======================

Version 3 represents a complete overhaul of the corr2 code.  
Much of the I/O code in particular is now done in python, rather than C++.
In addition, the C++ code is all available from within python, which means
that it should be easy for people to write their own scripts using the 
python module if they want to do something a bit different with how the 
data is read in or to do something directly with the computed correlation
function in python.

In addition there are a few fairly minor changes in the API.  I think these
are primarily edge cases, so you should not have any trouble adapting to
these changes if they affect you.

API changes:
------------

- Removed smooth_scale option.  It is easy enough for people to do this
  themselves if they want, so it seems unnecessary.  Plus, the way it
  did the smoothing is actually a pretty bad way to do it.  Better to just
  get rid of that option entirely.
- Changed the handling of file_name and file_name2 a bit.  Before, cross-
  correlations could be done by giving both file names as `file_name` in
  a list.  For anything but the simplest applications, this could get
  confusing in terms of how other parameters were handled.  So now, you
  should always use both file_name and file_name2 for cross-correlations.
- Removed the do_auto_corr and do_cross_corr options.  These were
  similarly confusing, so the current behavior is equivalent to setting
  do_auto_corr=True and do_cross_corr=True.  i.e. A list of files is
  conceptually equivalent to having everything in a single file.
  If you want to do something more complicated, you should now use the
  python interface to get precisely the pairings that you want.
- Removed the project option from the corr2 parameters.  That option
  was really only for me to debug the spherical trig code, but since that
  is working, there is really no reason for anyone to use it.  I do
  not want users to see this option and think it is a good idea or necessary
  for spherical coordinates.  If you really want to though, the functionality
  still exists at the python layer, so you can do the projections yourself
  there.
- Added another verbosity level between what was 0 and 1.  Now we have:
    0 - Errors only (equivalent to old 0)
    1 - Warnings (new default verbosity)
    2 - Progress (equivalent to old 1)
    3 - Debugging (equivalent to old 2)
- Added a configuration parameter, output_dots, which says whether to output
  the progress dots during the calculation in the C++ layer.  By default
  this is turned on if verbose is 2 or 3, and off otherwise.


New features:
-------------

Of course, making this a python module is the main new feature with this
version.  From the python layer, you can now have much more control over the
calculation.  For more information, please see:

  https://github.com/rmjarvis/TreeCorr/wiki/Guide-to-using-TreeCorr-in-Python

You can also use the python help command to get more information about particular
classes or functions to read their doc strings.

Some highlights of the new functionality:

- Can construct a Catalog directly from numpy arrays, rather than necessarily
  reading the catalog from a file.
- Can access the calculated correlation functions directly as a numpy arrays,
  rather than necessarily writing them to a file.
- The catalogs and constructed trees remain available to be used multiple times,
  which can be a huge efficiency gain for some applications, since you can
  cut out the I/O steps.
- The pairs of galaxies to be processed can be customized as desired.  For
  example, if you have several observations of the same field, you can cross-
  correlate shear measurements from different exposures and skip the auto-
  correlations to avoid some of the systematic errors.

Other new features:

- Added the option of giving weights to N fields, just like we allow for shear
  and kappa fields.  The weights are ignored for NN correlations, but they
  can be useful for NG or NK.
- Added the ability to do correlations in 3D, rather than just in angular 
  coordinates.  This is only available in conjunction with ra,dec coordinates.
  If you include an `r_col`, this will be the distance to each object.  Then
  the separations between objects will be done in 3D using whatever units you
  use for r (MPc for instance).


Changes in the output files:
----------------------------

- Removed the extra columns that had been written out in the case of
  compensated NG and NK calculations.
- Removed the extraneous <R> column from the norm output file.
- Changed the default precision from 3 to 4.


Bug fixes:
----------

- Fixed a bug in text file input where sometimes the data from two rows
  was treated as a single row with twice as many values.
- Fixed a (related) bug in how randoms are read in that they no longer need
  to have all the columns as the data files if it has more than just positions.
  Just the positions in the random files need to have the same column numbers
  or names as in the data files.
- Fixed a compiler error that started with recent versions of g++.
- Fixed what I believe was a bug in how the sigma columns for NN correlation 
  functions was calculated.  I believe it was too small by a factor of sqrt(2)
  for auto-correlations.

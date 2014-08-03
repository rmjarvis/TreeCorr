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

- Removed smooth_scale option.  It is easy enough for people to do this
  themselves if they want, so it seems unnecessary.  Plus, the way it
  did the smoothing is actually a pretty bad way to do it.  Better to just
  get rid of that option entirely.
- Removed the extra columns that had been written out in the case of
  compensated NG and NK calculations.
- Changed the handling of file_name and file_name2 slightly.  Before, the
  two files for a cross-correlation could be specified with just file_name
  using 2 entries.  Now these need to be listed as separate key words
  with file_name and file_name2.
- Added another verbosity level between what was 0 and 1.  Now we have:
    0 - Errors only (equivalent to old 0)
    1 - Warnings (new default verbosity)
    2 - Progress (equivalent to old 1)
    3 - Debugging (equivalent to old 2)
- Added a configuration parameter, output_dots, which says whether to output
  the progress dots during the calculation in the C++ layer.  By default
  this is turned on if verbose is 2 or 3, and off otherwise.
- Removed the extraneous <R> column from the norm output file.
- Fixed a bug in text file input where sometimes the data from two rows
  was treated as a single row with twice as many values.
- Fixed a (related) bug in how randoms are read in that they no longer need
  to have all the columns as the data files if it has more than just positions.
  Just the positions in the random files need to have the same column numbers
  or names as in the data files.
- Fixed a compiler error that started with recent versions of g++.
- Added the option of giving weights to N fields, just like we allow for shear
  and kappa fields.  The weights are ignored for NN correlations, but they
  can be useful for NG or NK.
- Fixed what I believe was a bug in how the sigma columns for NN correlation 
  functions was calculated.  I believe it was too small by a factor of sqrt(2)
  for auto-correlations.

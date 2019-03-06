Changes from version 3.3 to 3.4
===============================

The numbers at the ends of some items below indicate which issue is connected
with the change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+3.4%22+is%3Aclosed

Dependency changes:
-------------------

- Added dependency on LSSTDESC.Coord, and removed the TreeCorr implementation
  of the same functionality.


API changes:
------------

- Added some attributes of the various Correlation objects to the output files
  to improve serialization -- the ability to write to a file and then read back
  in that file and get back something equivalent.  For FITS output files, this
  change is non-intrusive.  But for ASCII files, it adds a comment line at the
  top with the relevant information.  Therefore, if you are reading from ASCII
  output files, you might need to slightly change your code to skip the first
  line.  (e.g. with np.genfromtxt(..., skip_header=1).)


New features:
-------------

- Added a new concept, called bin_type for all the Correlation objects.  There
  are currently three possible options for bin_type:
  - 'Log' is equivalent to the previous behavior of binning in log space.
  - 'Linear' bins linearly in r.  (#5)
  - 'TwoD' bins linearly in x and y.  (#70)
- Improved efficiency of runs that use bin_slop < 1. (Especially << 1). (#16)
- Added a distinction between bin_slop=0 and bin_slop>0, but very close
  (say 1.e-16).  The former will traverse the tree all the way to the
  leaves, never grouping objects into cells.  The latter will group objects
  when all pairs fall into the same bin.
- Improved both the speed and accuracy of the Rlens metric calculation.
- Added the ability to use min_rpar and max_rpar with the Arc metric. (#61)
- Added a different definition of Rperp, called FisherRperp, which follows
  the definition in Fisher et al, 1994.  This definition is both more standard
  and slightly faster to calculate, so we will switch Rperp to match this
  metric in the next major version release (4.0).  For now, it is only
  available as FisherRperp.  The name OldRperp is added as an alias of the
  current Rperp metric, and it will remain available as such after 4.0. (#73)
- Added better messaging when OpenMP is not found to work with the available
  clang compiler. (#75)


Bug fixes:
----------

- Added tot attribute to the NN and NNN output files, which fixes an error
  where NNCorrelation and NNNCorrelation did not round trip correctly through
  a FITS output file.  Now the tot attribute is set properly when reading.
- Fixed the Catalog.copy() method, which wasn't working properly.
- Fixed an error in the Schneider NMap calculation.

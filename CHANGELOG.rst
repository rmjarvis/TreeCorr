Changes from version 3.3 to 3.4
===============================

The numbers at the ends of some items below indicate which issue is connected
with the change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+3.4%22+is%3Aclosed

Dependency changes
------------------

- Added dependency on LSSTDESC.Coord, and removed the TreeCorr implementation
  of the same functionality.
- Made fitsio and pandas dependencies optional, which means that setup.py won't
  automatically install them for you.  If you plan to use TreeCorr with FITS
  files, you will need to install fitsio yourself.  Pandas is never required,
  but if you plan to use TreeCorr with ASCII input catalogs, installing pandas
  will provide a performance improvement. (#57)


API changes
-----------

- Added some attributes of the various Correlation objects to the output files
  to improve serialization -- the ability to write to a file and then read back
  in that file and get back something equivalent.  For FITS output files, this
  change is non-intrusive.  But for ASCII files, it adds a comment line at the
  top with the relevant information.  Therefore, if you are reading from ASCII
  output files, you might need to slightly change your code to skip the first
  line.  (e.g. with np.genfromtxt(..., skip_header=1).)
- Updated the caching of the fields built for a given catalog to only cache
  one (the most recently built) field, rather than all fields built.  If you
  had been relying on multiple fields being cached, this could lead to a
  performance regression for you.  However, you can update the number of
  fields cached with catalog.resize_cache(n). (#53)
- Using bin_slop=0 no longer does the brute force calculation.  Rather, it
  stops traversing the tree when all pairs for a given pair of cells would
  fall into the same bin.  This distinction is normally not important, and
  the new behavior is merely a performance increase.  However, there is a
  numerical difference for shear correlations, as the shear projection is not
  exactly equivalent.  To obtain the brute force calculation, use the new
  `brute=True` option.
- Changed how the 3pt correlation v binning is specified.  Now, you should
  only specify the range and number of bins for |v|. The negative v's will
  also be accumulated over the same range of absolute values as the positive
  v's. This should normally not be a hardship, since if you want to accumulate
  negative v's, you probably want the same range as the positive v's. But it
  enables you, for instance, to accumulate all flattened triangles with
  0.8 < |v| < 1.0 in one pass rather than two.
- Changed the behavior of specifying min_sep, max_sep, and bin_size (omitting
  nbins) to respect max_sep and reduce bin_size slightly, rather than
  respect bin_size and increase max_sep.  This seems like the behavior that
  most people would expect for this combination.


Performance Improvements
------------------------

- Improved efficiency of runs that use bin_slop < 1. (Especially << 1). (#16)
- Reduced the memory required for the constructed trees slightly. (By 8 bytes
  per galaxy.)
- Updated the caching of the fields to allow for more flexibility about how
  many fields are cached for a given catalog.  The default is to cache 1 field,
  which is normally appropriate, but you can use catalog.resize_cache(n) to
  either increase this number or to tell it not to cache at all (n=0). (#53)
- Added a catalog.clear_cache() function, which lets you manually clear the
  cache to release the memory of the cached field(s). (#53)
- Improved both the speed and accuracy of the Rlens metric calculation.
- Improved both the speed and accuracy of 3pt correlation functions.


New features
------------

- Added a new concept, called bin_type for all the Correlation objects.  There
  are currently three possible options for bin_type:
  - 'Log' is equivalent to the previous behavior of binning in log space.
  - 'Linear' bins linearly in r. (#5)
  - 'TwoD' bins linearly in x and y. (#70)
- Added a distinction between bin_slop=0 and bin_slop>0, but very close
  (say 1.e-16).  The former will traverse the tree all the way to the
  leaves, never grouping objects into cells.  The latter will group objects
  when all pairs fall into the same bin.
- Added the ability to use min_rpar and max_rpar with the Arc metric. (#61)
- Added a different definition of Rperp, called FisherRperp, which follows
  the definition in Fisher et al, 1994.  This definition is both more standard
  and slightly faster to calculate, so we will switch Rperp to match this
  metric in the next major version release (4.0).  For now, it is only
  available as FisherRperp.  The name OldRperp is added as an alias of the
  current Rperp metric, and it will remain available as such after 4.0. (#73)
- Added better messaging when OpenMP is not found to work with the available
  clang compiler. (#75)
- Added new methods Field.count_near and Field.get_near, which return the
  number of or the indices of points in the field that are near a given
  other coordinate. (#44)
- Added new method BinnedCorr2.sample_pairs, which returns a random sampling
  of pairs within a given range of separations.  E.g. a sample of pairs that
  fell into a given bin of the correlation function. (#67)
- Added `brute` option for Correlation instances.  This is equivalent to the
  old behavior of `bin_slop=0`.
- Added 'Periodic' metric. (#56)
- Added `min_top` option for Fields.
- Added calculation of <Map^3> and related quantities.
- Added option to provide R values for MapSq and related statistics.


Bug fixes
---------

- Added tot attribute to the NN and NNN output files, which fixes an error
  where NNCorrelation and NNNCorrelation did not round trip correctly through
  a FITS output file.  Now the tot attribute is set properly when reading.
- Fixed the Catalog.copy() method, which wasn't working properly.
- Fixed an error in the Schneider NMap calculation.
- Fixed a factor of 2 missing in the estimate of varxi. (#72)

Changes from version 3.3 to 4.0
===============================

This version includes a number of backwards-incompatible API changes, so
we are making it a major version update.  In most cases, these should not be
very onerous to deal with, but many of them would have been annoying to make
work in a backwards-compatible way.  Therefore, you are advised to read through
the list of changes carefully to see what aspects of your code might need to
be adjusted slightly to work with version 4.x.

The main feature updates here include:

- Linear binning
- Two-dimensional 2-point correlations
- Large speed improvements when bin_slop << 1
- Update to the Rperp metric to better match other usage in literature
- New "Periodic" metric
- New Field.get_near funtion
- New BinnedCorr2.sample_pairs function

Lots of other more minor updates as detailed in the list below.  The numbers at
the ends of some items below indicate which issue or PR is connected with the
change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+4.0%22+is%3Aclosed

Dependency changes
------------------

- Added dependency on LSSTDESC.Coord, and removed the TreeCorr implementation
  of the same functionality. (#77)
- Made fitsio dependency optional, which means that setup.py won't automatically
  install it for you.  If you plan to use TreeCorr with FITS files, you will
  need to install fitsio yourself. (#57)
- Made pandas dependency optional, which means that setup.py won't automatically
  install it for you.  Pandas is never required, but if you plan to use TreeCorr
  with ASCII input catalogs, installing pandas will provide a performance
  improvement. (#57)
- Removed dependency on future. (#85)


API changes
-----------

- Removed the treecorr Angle, AngleUnit, and CelestialCoord classes, along
  with other related functions.  These have never been highly used in TreeCorr,
  and all such usage are now relegated to the LSSTDESC.Coord package.
  If you have been using any of these features, you should switch you code
  to using Coord instead. (#77)
- Added some attributes of the various Correlation objects to the output files
  to improve serialization -- the ability to write to a file and then read back
  in that file and get back something equivalent.  For FITS output files, this
  change is non-intrusive.  But for ASCII files, it adds a comment line at the
  top with the relevant information.  Therefore, if you are reading from ASCII
  output files, you might need to slightly change your code to skip the first
  line.  (e.g. with np.genfromtxt(..., skip_header=1).)
- Changed the meaning of the Rperp metric to match the definition in Fisher
  et al, 1994.  This definition is probably closer to what people have expected
  the Rperp metric to mean, so this is likely an improvement for most use
  cases.  However, the difference between the two is not completely negligible
  for typical use cases, so we have preserved the previous functionality as
  metric=OldRperp.  If you care about preserving this, you should either
  change your code to use this metric name instead of Rperp, or set
  ``treecorr.Rperp_alias = 'OldRperp'`` before using it. (#73)
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
  ``brute=True`` option. (#82)
- Changed how the 3pt correlation v binning is specified.  Now, you should
  only specify the range and number of bins for positive v values. The negative
  v's will also be accumulated over the same range of absolute values as the
  positive v's. This should normally not be a hardship, since if you want to
  accumulate negative v's, you probably want the same range as the positive
  v's. But it enables you, for instance, to accumulate all flattened triangles
  with 0.8 < abs(v) < 1.0 in one pass rather than two. (#85)
- Changed the behavior of specifying min_sep, max_sep, and bin_size (omitting
  nbins) to respect max_sep and reduce bin_size slightly, rather than
  respect bin_size and increase max_sep.  This seems like the behavior that
  most people would expect for this combination. (#85)
- Changed the name of the variance attributes for shear-shear correlations
  to varxip and varxim, rather than just a single varxi.  For the current
  naive shot-noise-only estimate of the variance, they are equal, but we plan
  to have other variance estimates where they could differ.  So this change
  is preparing for future work that will need the difference.
- Changed the name of the variance attributes for shear-shear-shear correlations
  to vargam0, vargam1, vargam2, and vargam3, rather than just a single vargam.
- Changed the column names in GG and GGG output files for the sigma estimates
  to sigma_xip and sigma_xim for GG and sigma_gam0, sigma_gam1, sigma_gam2,
  and sigma_gam3 for GGG.
- Changed the column names in output files that used an upper case R (R_nom,
  meanR, and meanlogR) to use a lower case r (r_nom, meanr, meanlogr) to match
  the rest of the documentation.
- Removed support for python 2.6.  (Probably no one cares...)
- Removed deprecated aliases in the config processing.


Performance Improvements
------------------------

- Improved efficiency of runs that use bin_slop < 1. (Especially << 1). (#16)
- Reduced the memory required for the constructed trees slightly. (By 8 bytes
  per galaxy.) (#82)
- Updated the caching of the fields to allow for more flexibility about how
  many fields are cached for a given catalog.  The default is to cache 1 field,
  which is normally appropriate, but you can use catalog.resize_cache(n) to
  either increase this number or to tell it not to cache at all (n=0). (#53)
- Added a catalog.clear_cache() function, which lets you manually clear the
  cache to release the memory of the cached field(s). (#53)
- Improved both the speed and accuracy of the Rlens metric calculation. (#77)
- Improved both the speed and accuracy of 3pt correlation functions. (#85)


New features
------------

- Added a new concept, called bin_type for all the Correlation objects.  There
  are currently three possible options for bin_type:

  - 'Log' is equivalent to the previous behavior of binning in log space.
  - 'Linear' bins linearly in r. (#5)
  - 'TwoD' bins linearly in x and y. (#70)

- Added the ability to use min_rpar and max_rpar with the Arc metric. (#61)
- Added a different definition of Rperp, called FisherRperp, which follows
  the definition in Fisher et al, 1994.  This definition is both more standard
  and slightly faster to calculate.  We think this is the definition that most
  people expect when using Rperp, so we have changed Rperp to be an alias of
  FisherRperp.  The name OldRperp has been added to perserve the old meaning
  of Rperp in case that is important for any science use cases. (#73)
- Added better messaging when OpenMP is not found to work with the available
  clang compiler. (#75)
- Added new methods Field.count_near and Field.get_near, which return the
  number of or the indices of points in the field that are near a given
  other coordinate. (#44)
- Added new method BinnedCorr2.sample_pairs, which returns a random sampling
  of pairs within a given range of separations.  E.g. a sample of pairs that
  fell into a given bin of the correlation function. (#67)
- Added ``brute`` option for Correlation instances.  This is equivalent to the
  old behavior of ``bin_slop=0``. (#82)
- Added 'Periodic' metric. (#56)
- Added ``min_top`` option for Fields. (#84)
- Added calculation of <Map^3> and related quantities. (#85)
- Added option to provide R values for MapSq and related statistics. (#85)


Bug fixes
---------

- Added tot attribute to the NN and NNN output files, which fixes an error
  where NNCorrelation and NNNCorrelation did not round trip correctly through
  a FITS output file.  Now the tot attribute is set properly when reading.
- Fixed the Catalog.copy() method, which wasn't working properly.
- Fixed an error in the Schneider NMap calculation. (#77)
- Fixed a factor of 2 missing in the estimate of varxi. (#72)


Changes from version 4.0.0 to 4.0.1
-----------------------------------

- Fixed an error in the installation when the compiler is not recognized that
  would cause it to go into an infinite loop.


Changes from version 4.0.1 to 4.0.2
-----------------------------------

- Fixed an error in the bin_slop treatment for Linear binning when sep_units
  is not 1 (i.e. radians for angular separations).


Changes from version 4.0.2 to 4.0.3
-----------------------------------

- Fixed an error that could sometimes cause a seg fault for certain input data
  sets.

Changes from version 4.0.3 to 4.0.4
-----------------------------------

- Fixed an error in the treatment of rpar for Rperp and OldRperp metrics, where
  it was not letting rpar be negative when the object in cat1 is farther away
  than the object in cat2.  (I.e. when r1 > r2, rpar should be < 0.)
  [UPDATE: This fix didn't actually take for some reason.  Now fixed in 4.0.6.]

Changes from version 4.0.4 to 4.0.5
-----------------------------------

- Fixed an error in the variance calculation, which had been NaN if any of the
  input g1,g2 or k (as appropriate) were NaN. (#90)

Changes from version 4.0.5 to 4.0.6
-----------------------------------

- Fixed an error in the treatment of rpar for Rperp and OldRperp metrics, where
  it was not letting rpar be negative when the object in cat1 is farther away
  than the object in cat2.  (I.e. when r1 > r2, rpar should be < 0.)
  [This was the fix that was supposed to have been active in 4.0.4.]
- Fixed a subtle bug in the wpos handling during Cell construction.

Changes from version 4.0.6 to 4.0.7
-----------------------------------

- Added -stdlib=libc++ flag for clang.
- Added support for some other varieties of OpenMP.
- Added support for using ccache when compiling.

Changes from version 4.0.7 to 4.0.8
-----------------------------------

- Fixed error in the check for whether ccache is available.

Changes from version 4.0.8 to 4.0.9
-----------------------------------

- Fixed error in code to build tree when some weights are < 0. (#95)


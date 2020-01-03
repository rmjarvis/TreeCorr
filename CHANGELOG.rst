Changes from version 4.0 to 4.1
===============================

This release mostly adds the ability to compute jackknife and other estimates of
the variance (or covariance matrix), which do a better job of capturing sample
variance than the default (and previously only) method of computing just the
shot noise component.

This requires dividing the input Catalog into patches, which can be done either by
giving patch numbers explicitly for each galaxy or by having TreeCorr split it
into some number of patches for you automatically (using KMeans).

There is one "bug fix", which is worth calling out, since it's possible that some
people have relied on the "feature" which this fixes.  The npairs attribute had
only included pairs where both objects have w != 0.  This led to some problems
with the new patches stuff when the input catalogs had some w=0 objects.  So
now the number that is accumulated in npairs includes objects with w=0. If this
change causes problems for you, it is likely that you only need to switch
``npairs`` to ``weight`` in your code.  But if that is not sufficient, please
open an issue and we can investigate how to accommodate your use case.


Performance Improvements
------------------------



New features
------------

- Added ability to run k-means algorithm on a catalog, which is much faster than other
  existing python k-means codes.  Also produces reliably more uniform patches.  See the
  discussion on PR #88 for details.
- Added the ability to compute jackknife and sample variance estimates by dividing the
  full calculation into correlations across patches.  Set ``var_method`` in the
  Correlation class or use the ``estimate_cov`` method.
- Added ``treecorr.estimate_multi_cov`` function, which will the compute covariance
  matrix across multiple statistics that have been run using the same set of patches.


Bug fixes
---------

- Changed npairs to include pairs with w=0 objects.  This had been an undocumented
  and not particularly well-motivated "feature", but it led to some problems with the
  new kmeans stuff when there are some w=0 points in the field.

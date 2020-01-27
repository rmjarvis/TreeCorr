Changes from version 4.0 to 4.1
===============================

This release mostly adds the ability to compute jackknife and other estimates of
the variance (or covariance matrix), which do a better job of capturing sample
variance than the default (and previously only) method of computing just the
shot noise component.

This requires dividing the input Catalog into patches, which can be done by:

1. giving patch numbers explicitly for each galaxy.
2. having TreeCorr split it into some number of patches for you automatically (using KMeans).
3. giving the patch centers to use, which will assign each galaxy to the patch corresponding
   to the nearest center position.

Output Format Changes
---------------------

- When writing an `NNCorrelation` to a file, if `NNCorrelation.calculatXi` has
  already been called, then the calculated ``xi`` and ``varxi`` will be written
  to the output file, even if you don't provide the random catalog to the
  `NNCorrealtion.write` function.
- Similarly, if `NGCorrelation.calculateXi` or `NKCorrelation.calculateXi` has
  been called using a random catalog, then the ``xi`` and ``varxi`` columns in
  the output file will be the compensated statistics, rather than the raw ones.


Performance Improvements
------------------------

- Delayed the loading of Catalogs from files until the data is actually needed.


New features
------------

- Added ability to run k-means algorithm on a catalog, which is much faster than other
  existing python k-means codes.  Also produces reliably more uniform patches.  See the
  discussion on PR #88 for details.
- Added the ability to compute jackknife, sample, and bootstrap variance and covariance
  estimates by dividing the full calculation into correlations across patches.
  Set ``var_method`` in the Correlation class or use the ``estimate_cov`` method.
- Added ``treecorr.estimate_multi_cov`` function, which will the compute covariance
  matrix across multiple statistics that have been run using the same set of patches.
- Added every_nth option for Catalogs to read in a fraction of the rows.
- After calling `NNCorrelation.calculateXi`, the calculated ``xi``, ``varxi`` and
  ``cov`` are available as attributes.
- Added ``keep_zero_weight`` option to include wpos=0 objects in the total npairs
  if you want.  The default is to throw out wpos=0 objects at the start, but there
  may be reasons to keep them, so that's now an option when building the Catalog.


Bug fixes
---------


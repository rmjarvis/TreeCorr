Changes from version 4.0 to 4.1
===============================

This release mostly adds the ability to compute jackknife and other estimates of
the variance (or covariance matrix), which do a better job of capturing sample
variance than the default (and previously only) method of computing just the
shot noise component.

This requires dividing the input Catalog into patches, which can be done by:

1. Giving patch numbers explicitly for each galaxy.
2. Having TreeCorr split it into some number of patches for you automatically (using K-Means).
3. Giving the patch centers to use, which will assign each galaxy to the patch corresponding
   to the nearest center position.

For more details about making patches, see `Patches
<https://rmjarvis.github.io/TreeCorr/_build/html/patches.html>`_

For details about using patches to compute better covariance matrices,
see `Covariance Estimates
<https://rmjarvis.github.io/TreeCorr/_build/html/cov.html>`_

`Relevant PRs and Issues,
<https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+4.1%22+is%3Aclosed>`_
whose issue numbers are listed below for the relevant items.

Output Format Changes
---------------------

- When writing an `NNCorrelation` to a file, if `NNCorrelation.calculateXi` has
  already been called, then the calculated ``xi`` and ``varxi`` will be written
  to the output file, even if you don't provide the random catalog to the
  `NNCorrelation.write` function. (#99)
- Similarly, if `NGCorrelation.calculateXi` or `NKCorrelation.calculateXi` has
  been called using a random catalog, then the ``xi`` and ``varxi`` columns in
  the output file will be the compensated statistics, rather than the raw ones.
  (#99)


Performance Improvements
------------------------

- Improved the speed of reading in FITS files, mostly by telling fitsio to read
  all of the columns at once rather than doing them one at a time. (#104)
- Delayed the loading of Catalogs from files until the data is actually needed.
  Specifically, if the full catalog is never needed, and only patches are used,
  then this allows for a much reduced memory footprint with a new ``low_mem``
  option for ``process`` calls. (#103)
- Added OpenMP parallelization to the (ra,dec) -> (x,y,z) calculation to speed
  up that step during Catalog loading. (#104)


New features
------------

- Added `Field.run_kmeans` to run K-Means algorithm on a Field to determine a
  good set of patches to use.  This will be run automatically simply by setting
  ``npatch`` when building a `Catalog`. (#88)
- Added ability to input patch numbers from input file or input arrays using
  ``patch_col`` or ``patch`` respectively. (#96)
- Added ability to write patch centers to a file with `Catalog.write_patch_centers`.
  (#99)
- Added ability to define patch numbers from patch centers (either a file or
  the ``patch_centers`` attribute of another Catalog). (#99)
- Added the ability to compute jackknife, sample, bootstrap, and marked bootstrap
  variance and covariance estimates by dividing the full calculation into
  correlations across patches.  Set ``var_method`` in the Correlation class or
  use the `BinnedCorr2.estimate_cov` method. (#24, #50, #96)
- Added `estimate_multi_cov` function, which will the compute covariance
  matrix across multiple statistics that have been run using the same set of patches.
  (#96)
- Added ``every_nth`` option for Catalogs to read in a fraction of the rows.  (#99)
- After calling `NNCorrelation.calculateXi`, the calculated ``xi``, ``varxi`` and
  ``cov`` are available as attributes. (#99)
- Added ``keep_zero_weight`` option to include wpos=0 objects in the total npairs
  if you want.  The default is to throw out wpos=0 objects at the start, but there
  may be reasons to keep them, so that's now an option when building the Catalog.
  (#99)
- Added ``allow_xyz`` option to allow x,y,z columns be provided along with ra,dec
  columns.  Normally this is not allowed, but if you know they are consistent,
  then this option allows them to be input directly to avoid recalculating them.
  (#103)
- Added ``low_mem`` option to ``process`` calls to unload patches that aren't being
  used, thus saving memory at the expense of some extra I/O time. (#103)
- Added ``save_patch_dir`` as an optional location to write patch catalogs for increased
  efficiency when using ``low_mem`` option. (#103)
- Added ``comm`` option to ``process`` calls to use MPI to split a job up over
  multiple machines. (#98, #104)


Deprecations
------------

- The `process_pairwise <NNCorrelation.process_pairwise>` functions have all been
  deprecated.  I can't remember anymore why I added this feature, and I don't think
  anyone uses it (or the associated `SimpleFields <NSimpleField>`).  If you need this
  functionality, please open an issue to let me know, and I can keep them.  Otherwise,
  I'll remove them as usesless cruft.


Bug Fixes
---------

- Fixed a bug in 3-point calculations that could cause "Failed Assert: kr < _nbins".


Changes from version 4.1.0 to 4.1.1
-----------------------------------

- Made sure ra,dec,r catalogs make patches using just ra,dec, not full 3D position.
- Fixed a bug when using ra,dec,r catalog with ``save_patch_dir`` option.

Changes from version 4.1.1 to 4.1.2
-----------------------------------

- Fixed a bug when reading ascii files with every_nth != 1.

Changes from version 4.1.2 to 4.1.3
-----------------------------------

- Fixed the same every_nth bug when pandas is not installed.

Changes from version 4.1.3 to 4.1.4
-----------------------------------

- Fixed a bug when using every_nth in conjunction with explicit numpy arrays.

Changes from version 4.1.4 to 4.1.5
-----------------------------------

- Fixed read_catalogs to work properly with patches.

Changes from version 4.1.5 to 4.1.6
-----------------------------------

- Fixed a bug in `GGGCorrelation.calculateMap3`, which could sometimes take
  the sqrt of negative numbers, resulting in nans.

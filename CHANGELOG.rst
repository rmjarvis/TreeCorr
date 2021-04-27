Changes from version 4.1 to 4.2
===============================

This will be the last (non-bug-fix) TreeCorr release to support Python 2.7.
It is becoming harder to continue to support this platform now that it is
officially sunsetted, so we encourage all users to switch their code to
Python 3.x ASAP.  TreeCorr is currently compatible with Python versions
2.7, 3.5, 3.6, 3.7, 3.8, 3.9.  We will probably also drop 3.5 in the next
release as well, since that is also retired at this point.


API Changes
-----------

- Changed the 3pt process function, when given 3 arguments, to accumulate all
  triangles where points from the three input catalogs can fall into any
  of the three triangle corners.  If you need the old behavior of keeping
  track of which catalog goes into which triangle corner, the new
  ???CrossCorrelation classes do so both more efficiently and more
  accurately than the previous code. (#115)
- Changed the `NNNCorrelation.calculateZeta` function to only take
  RDD and DRR parameters for the cross terms (if desired) rather than all
  six (DDR, DRD, RDD, DRR, RDR, RRD).  The new cross-correlation behavior of
  the `NNNCorelation.process` function now efficiently calculates in RDD what
  used to be calculated in three calls for DDR, DRD, and RDD.  Likewise the
  new DRR calculates what used to require DRR, RDR, and RRD. (#122)

Performance improvements
------------------------

- Only show at most 10 rows with NaNs in cases where there are lots of such
  rows. (#111)
- No longer remakes patches and writes them to disk if they are already present
  on disk.  If you want to force a rewrite for any reason, you can explicitly
  call `Catalog.write_patches`. (#119)
- Computing the data/random cross correlations for 3pt are now much faster,
  since you only need one call for RDD and one for DRR, not all the 6 different
  permuations. (Specifically, DDR, DRD, RDR, RRD are no longer needed.) (#122)

New features
------------

- Allow min_rpar and max_rpar for Euclidean metric.  Also Periodic, although
  I don't know if that's useful. (#101)
- Added ability to read from hdf5 catalogs.  (#106)
- Added ability to use named columns in ASCII files if the file has column
  names. (#108)
- Added optional initialize=False and finalize=False options to process
  functions. (#109)
- Added ability to do 3 point cross-correlations properly, rather than the not
  exactly correct version that had been implemented.  Not all combinations are
  implemented.  So far just GGG, KKK, and NNN. (#115)
- Added ability to read from parquet catalogs.  (#117)
- Added optional func parameter to estimate_cov and estimate_multi_cov. (#118)
- Added ability to write output files as hdf5.  (#122)

Bug fixes
---------

Changes from version 4.1 to 4.2
===============================


Performance improvements
------------------------

- Only show at most 10 rows with NaNs in cases where there are lots of such
  rows. (#111)
- No longer remakes patches and writes them to disk if they are already present
  on disk.  If you want to force a rewrite for any reason, you can explicitly
  call `Catalog.write_patches`. (#119)

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

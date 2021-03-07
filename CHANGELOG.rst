Changes from version 4.1 to 4.2
===============================


Performance improvements
------------------------


New features
------------

- Added ability to read from hdf5 catalogs.  (#106)
- Added ability to use named columns in ASCII files if the file has column names. (#108)
- Added ability to do 3 point cross-correlations properly, rather than the not exactly
  correct version that had been implemented.  Not all combinations are implemented.
  So far just GGG, KKK, and NNN. (#115)
- Added ability to read from parquet catalogs.  (#117)
- Added optional func parameter to estimate_multi_cov. (#118)

Bug fixes
---------

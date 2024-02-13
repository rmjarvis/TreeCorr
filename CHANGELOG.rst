Changes from version 4.3 to 5.0
===============================

See the listing below for the complete list of new features and changes.
`Relevant PRs and Issues,
<https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+4.4%22+is%3Aclosed>`_
whose issue numbers are listed below for the relevant items.


Dependency Change
-----------------

- Switched from cffi to pybind11 for the C++ bindings. (#155)


API Changes
-----------

- When making a `Catalog`, if ``patch`` is an integer, ``npatch`` is now required.  This used to
  be usually required implicitly in how TreeCorr expected things to be set up downstream, but
  there were some use cases where a user could get away with not providing ``npatch`` and things
  would nonetheless still work.  But now we explicitly check for it, so those use cases do
  require passing ``npatch`` now.  (#150)
- Renamed the base classes BinnedCorr2 -> `Corr2` and BinnedCorr3 -> `Corr3`.  These are not
  normally used directly by users, so it shouldn't be noticeable in user code. (#155)
- Removed all deprecations from the 4.x series. (#156)
- Removed support for reading back in output files from the 3.x series. (#165)
- Removed the 3pt CrossCorrelation classes, which used to be the way to get ordered three-point
  correlations.  But they were rather unwieldy and not very intuitive.  The new ``ordered``
  option to the three-point ``process`` methods is much simpler and more efficient for the common
  case of only wanting a single order for the catalogs. (#165)
- Switched the default behavior of 3pt cross-correlations to respect the order of the catalogs
  in the triangle definitions.  That is, points from cat1 will be at P1 in the triangle,
  points from cat2 at P2, and points from cat3 at P3.  To recover the old behavior, you may
  use the new ``ordered=False`` option. (#166)
- Switched the default binning for three-point correlations to LogSAS, rather than LogRUV. (#166)
- Changed estimate_cov with method='shot' to only return the diagonal, rather than gratuitously
  making a full, mostly empty diagonal matrix. (#166)
- Changed name of Catalog.write kwarg from cat_precision to just precision. (#169)


Performance improvements
------------------------

- Reduced the compiled library size, and refactored things so the new correlation types would not
  add nearly as much to the compiled size as they would have previously. (#157)
- Made a small (~5-10%) improvment in speed of most 2pt correlation runs. (#157)
- Made variance calculations more efficient when using var_method='shot'.  Now it doesn't
  gratuitiously make a full covariance matrix, only to then extract the diagonal. (#166)
- Added the multipole algorithm for three-point correlations, descibed in Porth et al (2023)
  for GGG, and previously in Chen & Szapudi (2005), Slepian & Eisenstein (2015) and Philcox et al
  (2022) for NNN and KKK.  This algorithm is much, much faster than the 3 point calculation that
  TreeCorr had done, so it is now the default.  However, this algorithm only works with SAS
  binning, so LogSAS is now the default binning for three-point correlations. (#167, #171)


New features
------------

- Give a better error message when patch is given as an integer, but npatch is not provided. (#150)
- Allow vark, varg, varv for a Catalog be specifiable on input, rather than calculated directly
  from the corresponding values. (#154)
- Allow numpy.random.Generator for rng arguments (in addition to legacy RandomState). (#157)
- Added spin-1 correlations using the letter V (for Vector), including `NVCorrelation`,
  `KVCorrelation` and `VVCorrelation`. (#158)
- Added spin-3 and spin-4 correlations using the letters T (for Trefoil) and Q (for Quatrefoil)
  respectively, including `NTCorrelation`, `KTCorrelation`, `TTCorrelation`, `NQCorrelation`,
  `KQCorrelation` and `QQCorrelation`. (#160)
- Automatically recognize .h5 as an HDF5 suffix in file names. (#161)
- Added ``ordered=True`` option to the 3pt ``process`` methods for keeping the order of the
  catalogs fixed in the triangle orientation. (#165)
- Added ``bin_type='LogSAS'`` for 3pt correlations. (#165)
- Added ``bin_type='LogMultipole'`` for 3pt correlations and method `GGGCorrelation.toSAS` to
  convert from this format to the LogSAS binning if desired. (#167)
- Added ``patch_method`` option to ``process``, and specifically a "local" option.  This is
  not particularly recommended for most use cases, but it is required for the multipole
  three-point method, for which it is the default. (#169)
- Added ``angle_slop`` option to separately tune the allowed angular slop from using cells,
  irrespective of the binning. (#170)


Bug fixes
---------

- Fixed a rare potential bug in TwoD binning. (#157)
- Allowed both lens and random catalogs to have only 1 patch when source catalog has patches
  for NG, NK correlations. (#158)
- Fixed slight error in the variance calculation when using initialize/finalize options of
  process functions. (#158)
- Fixed bug that could cause `Catalog.write_patches` to not work correctly if patch files were
  already written in the ``save_patch_dir``. (#158)
- Fixed slight error in the shot-noise variance for G correlations.  It used to assume that the
  mean shear is 0, which is often very close to true.  Now it uses the actual mean. (#159)
- Fixed a very slight error in the parallel transport code, which is probably only noticeable
  for fields extremely close to a pole. (#160)

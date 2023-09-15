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
- Removed support for reading back in output files from the 3.x series.


Performance improvements
------------------------

- Reduced the compiled library size, and refactored things so the new correlation types would not
  add nearly as much to the compiled size as they would have previously. (#157)
- Made a small (~5-10%) improvment in speed of most 2pt correlation runs. (#157)


New features
------------

- Give a better error message when patch is given as an integer, but npatch is not provided. (#150)
- Allow vark, varg, varv for a Catalog be specifiable on input, rather than calculated directly
  from the corresponding values. (#154)
- Allow numpy.random.Generator for rng arguments (in addition to legacy RandomState). (#157)
- Added spin-1 correlations using the letter V (for Velocity), including `NVCorrelation`,
  `KVCorrelation` and `VVCorrelation`. (#158)
- Added spin-3 and spin-4 correlations using the letters T (for Trefoil) and Q (for Quatrefoil)
  respectively, including `NTCorrelation`, `KTCorrelation`, `TTCorrelation`, `NQCorrelation`,
  `KQCorrelation` and `QQCorrelation`. (#160)


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

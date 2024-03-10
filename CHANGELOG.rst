Changes from version 4.3 to 5.0
===============================

This is a major version update to TreeCorr, since it contains a number of non-backwards-
compatible changes to the API.

The most important change involves the three-point correlations.  TreeCorr now implements
the multipole algorithm of Porth et al, 2023 (arXiv:2309.08601), which is much faster than the
previous 3-cell recursion over triangles.  Enough so that I don't anticipate people ever wanting
to use the old algorithm.  However, this algorithm requires a different binning than we
used to use -- it requires binning according to two sides of the triangle and the angle
between them, rather than the three side lengths (using a somewhat awkward formulation
in terms of ratios of side lengths).

The new three-point binning scheme is called ``bin_type="LogSAS"``.  This is now the default
binning for all three-point correlation classes.  Furthermore, the default algorithm is
``algo="multipole"``, which first computes the multipole version of the correlation function
using the Porth et al algorithm.  Then it converts back to regular configuration space
the the LogSAS binning.

The old versions are still available in case there are use cases for which they are superior
in some way.  I do use them in the test suite still for comparison purposes.  To use the
old binning, you now need to explicitly specify ``bin_type="LogRUV"`` in the Correlation class,
and to use the old algorithm of accumulating triangle directly, use ``algo="triangle"``
when calling `Corr3.process`.

I also changed how three-point cross correlations are handled, since I wasn't very happy with
my old implementation.  Now, you can indicate whether or not you want the three points
to keep their ordering in the triangle with the parameter ``ordered`` in the `Corr3.process`
function.  If ``ordered=False``, then points from the (2 or 3) catalogs are allowed to take
any position in the triangle.  If ``ordered=True`` (the default), then points from the first
catalog will only for point P1 in the triangle, points from the second catalog will only be at P2,
and points from the third will only be at P3.  This seems to be a more intuitive way to control
this than the old ``CrossCorrelation`` classes.

Another big change in this release is the addition of more fields for the two-point correlations.
TreeCorr now implements correlations of spin-1 vector fields, as well as complex-valued
fields with spin=0, 3, or 4.  (TreeCorr had already implemented spin-2 of course.)
The letters for each of these are V, Z, T, and Q respectively.  I only did the pairings of each of
these with itself, counts (N), and real scalar fields (K).  However, it would not be too hard
to add more if someone has a use case for a pairing of two complex fields with different spins.

A complete list of all new features and changes is given below.
`Relevant PRs and Issues,
<https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+5.0%22+is%3Aclosed>`_
whose issue numbers are listed below for the relevant items.


Dependency Change
-----------------

- Switched from cffi to pybind11 for the C++ bindings. (#155)
- If using fitsio, it now must be version > 1.0.6. (#173)


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
- Added additionaly information in the header of output files to enable `Corr2.from_file`. (#172)


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

- Added spin-1 correlations using the letter V (for Vector), including `NVCorrelation`,
  `KVCorrelation` and `VVCorrelation`. (#81, #158)
- Give a better error message when patch is given as an integer, but npatch is not provided. (#150)
- Added ``x_eval``, ``y_eval``, etc. which let you calculate a derived quantity from an input
  catalog using Python eval on columns in the file. (#151, #173)
- Allow vark, varg, varv for a Catalog be specifiable on input, rather than calculated directly
  from the corresponding values. (#154, #159)
- Allow numpy.random.Generator for rng arguments (in addition to legacy RandomState). (#157)
- Added spin-3 and spin-4 correlations using the letters T (for Trefoil) and Q (for Quatrefoil)
  respectively, including `NTCorrelation`, `KTCorrelation`, `TTCorrelation`, `NQCorrelation`,
  `KQCorrelation` and `QQCorrelation`. (#160)
- Automatically recognize .h5 as an HDF5 suffix in file names. (#161)
- Added ``ordered=True`` option to the 3pt ``process`` methods for keeping the order of the
  catalogs fixed in the triangle orientation. (#165)
- Added ``bin_type='LogSAS'`` for 3pt correlations. (#165)
- Added ``bin_type='LogMultipole'`` for 3pt correlations and method `Corr3.toSAS` to
  convert from this format to the LogSAS binning if desired. (#167)
- Added serialization of rr, dr, etc. when writing with write_patch_results=True option,
  so you no longer have to separately write files for them to recover the covariance. (#168, #172)
- Added ``patch_method`` option to ``process``, and specifically a "local" option.  This is
  not particularly recommended for most use cases, but it is required for the multipole
  three-point method, for which it is the default. (#169)
- Added ``angle_slop`` option to separately tune the allowed angular slop from using cells,
  irrespective of the binning. (#170)
- Added ``algo`` option to 3-point ``process`` functions to conrol whether to use new
  multipole algorithm or the old triangle algorithm. (#171)
- Added `Corr2.from_file` class methods to construct a Correlation object from a file without
  needing to know the correct configuration parameters. (#172)
- Added ``write_cov`` option to write functions to include the covariance in the output file.
  (#172)
- Added complex, spin-0 correlations using the letter Z, including `NZCorrelation`,
  `KZCorrelation`, and `ZZCorrelation`. (#174)


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

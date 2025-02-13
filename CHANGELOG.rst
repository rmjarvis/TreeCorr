Changes from version 5.0 to 5.1
===============================

A complete list of all new features and changes is given below.
`Relevant PRs and Issues,
<https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+5.1%22+is%3Aclosed>`_
whose issue numbers are listed below for the relevant items.


API Changes
-----------

- This isn't quite an API change, but it's worth highlighting.  We left the default behavior
  of the cross_patch_weight to match the behavior of previous versions of TreeCorr.
  However, it now emits a warning that you should probably switch to using
  ``cross_patch_weight='match'`` for jackknife covariances or ``cross_patch_weight='geom'``
  for bootstrap covariances.  We may in the future switch these to be the default values,
  so if you want any existing scripts you have to keep the current behavior, you should
  explicitly set ``cross_patch_weight='simple'`` to avoid the warning.  And if you want the
  improved weighting, you should update your script to the appropriate value. (#180)


Performance improvements
------------------------

- Added an option to the process commands, ``corr_only=True``, which will skip the computations
  of ancillary quantities like ``meanlogr``, ``meanphi``, and ``npairs``, which are not
  necessary for the calculation of the correlation function.  This doesn't make much difference
  for most classes, but for `NNCorrelation`, it can be a significant speedup. (#182)


New features
------------

- Added many new classes for three-point functions with mixed field types in the different
  vertices, such as NNG, NKK, KGK, etc.  See `Three-point Correlation Functions` for
  details about all the new classes. (#32, #178, #179, #181)
- Added the ability to use the metrics Rlens and Rperp with three-point correlations. (#177, #184)
- Added the ability to use ``min_rpar`` and ``max_rpar`` with three-point correlations.
  (#177, #184)
- Added a new option for how to handle pairs that cross between two patches when doing
  patch-based covariance estimates.  This work is based on the paper by
  `Mohammad and Percival (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.1289M/>`_,
  who recommend using "match" for jackknife covariances and "geom" for bootstrap= covariances.
  The default is called "simple" and is the same behavior as what TreeCorr has been doing in
  previous versions, but we recommend users explicitly set ``cross_patch_weight`` to the
  appropriate value to take advantage of the more optimal weighting. (#180, #183)

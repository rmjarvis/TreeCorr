
Previous History
================

Changes from version 5.0 to 5.1
--------------------------------

User-impact highlights:

* Expanded three-point support to include mixed field-type classes
  (e.g. NNG, NKK, KGG and permutations).
* Added three-point support for ``Rlens``/``Rperp`` and ``min_rpar``/``max_rpar``.
* Added improved covariance weighting options via ``cross_patch_weight``; ``'match'``
  (jackknife) and ``'geom'`` (bootstrap) are now recommended, with ``'simple'`` retained
  for backward compatibility.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/5.1/CHANGELOG.rst>`__

Changes from version 4.3 to 5.0
--------------------------------

User-impact highlights:

* Major three-point workflow update: default ``bin_type='LogSAS'`` and
  ``algo='multipole'`` for large speed gains.
* Redesigned three-point cross-correlation handling around ``ordered`` in ``Corr3.process``,
  replacing older CrossCorrelation classes.
* Added new two-point spin-field families (Z, V, T, Q).
* Switched C++ bindings from cffi to pybind11 and removed accumulated 4.x deprecations.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/5.0/CHANGELOG.rst>`__

Changes from version 4.2 to 4.3
--------------------------------

User-impact highlights:

* Added covariance design-matrix helpers and improved patch-result I/O.
* Began transition of many function parameters to keyword-only usage
  (positional forms still worked but were deprecated).

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/4.3/CHANGELOG.rst>`__

Changes from version 4.1 to 4.2
--------------------------------

User-impact highlights:

* Extended patch-based workflows to three-point correlations.
* Improved three-point cross-correlation behavior for NNN/KKK/GGG.
* Added covariance estimation of arbitrary derived data vectors via the ``func`` argument.
* Added additional catalog I/O options, including HDF5 and Parquet.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/4.2/CHANGELOG.rst>`__

Changes from version 4.0 to 4.1
--------------------------------

User-impact highlights:

* Introduced modern patch-based covariance estimation
  (jackknife/sample/bootstrap/marked bootstrap).
* Added practical patch tooling: k-means patching, patch-center files, and ``patch_col``.
* Added large-scale workflow features including ``low_mem``, ``save_patch_dir``, and MPI.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/4.1/CHANGELOG.rst>`__

Changes from version 3.3 to 4.0
--------------------------------

User-impact highlights:

* Major API shift with new linear and TwoD binning options.
* Added/updated metrics (including Periodic and Fisher-style ``Rperp`` behavior).
* Improved performance in difficult ``bin_slop`` regimes.
* Updated output naming conventions and variance attributes, and moved coordinate internals
  to LSSTDESC.Coord.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/4.0/CHANGELOG.rst>`__

Changes from version 3.2 to 3.3
--------------------------------

User-impact highlights:

* Added YAML/JSON config support and new Arc/Rlens metric options.
* Updated separation-unit behavior to be less confusing in outputs.
* Changed C++ wrapping from ctypes to cffi and updated FITS I/O expectations around fitsio.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/3.3/CHANGELOG.rst>`__

Changes from version 3.1 to 3.2
--------------------------------

User-impact highlights:

* Introduced three-point correlations (NNN/KKK/GGG).
* Added ``Rperp`` support.
* Added FITS I/O support for correlation objects.
* Improved split-and-accumulate workflows by allowing objects to be read, written, and combined.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/3.2/CHANGELOG.rst>`__

Changes from version 3.0 to 3.1
--------------------------------

User-impact highlights:

* Renamed G2/NN/K2-style class and config names to GG/NN/KK naming.
* Transitioned documentation to Sphinx-hosted docs.
* Added stability fixes around file lists, logging verbosity, and compiler/OpenMP detection.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/3.1/CHANGELOG.rst>`__

Changes from version 2.6 to 3.0
--------------------------------

User-impact highlights:

* Major architectural overhaul that introduced a first-class Python-module interface.
* Enabled direct Catalog construction from arrays and in-memory access to computed vectors.
* Enabled more flexible custom workflows than executable-only usage.
* Simplified legacy corr2 options and clarified ``file_name``/``file_name2`` cross-correlation semantics.

`Full changelog <https://github.com/rmjarvis/TreeCorr/blob/releases/3.0/CHANGELOG.md>`__

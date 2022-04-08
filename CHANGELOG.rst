Changes from version 4.2 to 4.3
===============================

See the listing below for the complete list of new features and changes.
`Relevant PRs and Issues,
<https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+4.3%22+is%3Aclosed>`_
whose issue numbers are listed below for the relevant items.

Starting with this version, TreeCorr no longer supports Python 2.7.
We currently support python versions 3.6, 3.7, 3,8, 3.9.


API Changes
-----------

- Many function parameters are now keyword-only.  The old syntax allowing these parameters
  to be positional still works, but is deprecated. (#129)


Performance improvements
------------------------

- Added ability to compute patch-based covariance matrices using MPI. (#138, #139)


New features
------------


Bug fixes
---------


Changes from version 4.3 to 4.4
===============================

See the listing below for the complete list of new features and changes.
`Relevant PRs and Issues,
<https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+4.4%22+is%3Aclosed>`_
whose issue numbers are listed below for the relevant items.


API Changes
-----------

- When making a `Catalog`, if ``patch`` is an integer, ``npatch`` is now required.  This used to
  be usually required implicitly in how TreeCorr expected things to be set up downstream, but
  there were some use cases where a user could get away with not providing ``npatch`` and things
  would nonetheless still work.  But now we explicitly check for it, so those use cases do
  require passing ``npatch`` now.  (#150)


Performance improvements
------------------------



New features
------------

- Give a better error message when patch is given as an integer, but npatch is not provided. (#150)


Bug fixes
---------


Changes from 3.0 to 3.1
=======================

API changes:
------------


New features:
-------------

- Added a warning if a file_list does not have any entries.
- Added a warning if Catalog is passed an input array that is not 1-d, and automatically
  reshape it to 1-d.

Bug fixes:
----------

- Fixed an error in the handling of file_name as a list of file names.
- Allowed verbose parameter to correctly reset the logging verbosity on calls to Catalog, etc.
  It used to only get set correctly on the first call, so if later values of verbose were
  different, they did not work right.

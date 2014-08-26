Changes from 3.0 to 3.1
=======================


API changes:
------------


Changes to output files:
------------------------

- Removed the extra dots I put in the headers of the output files.  They were
  meant to help visually separate the columns, but it confuses some automatic
  parsers (like astropy as pointed out by Joe Zuntz).  (#14)


New features:
-------------

- Added a warning if a file_list does not have any entries. (#10)
- Added a warning if Catalog is passed an input array that is not 1-d, and 
  automatically reshape it to 1-d. (#11)


Bug fixes:
----------

- Fixed an error in the handling of file_name as a list of file names. (#10)
- Allowed verbose parameter to correctly reset the logging verbosity on calls
  to Catalog, etc.  It used to only get set correctly on the first call, so if
  later values of verbose were different, they did not work right. (#11)
- Fixed some errors in the sample.params file. (#12)

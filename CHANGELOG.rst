Changes from version 3.3 to 3.4
===============================

The numbers at the ends of some items below indicate which issue is connected
with the change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+3.4%22+is%3Aclosed

Dependency changes:
-------------------


API changes:
------------

- Added tot attribute to the NN and NNN output files.  This may require slight
  changes to code that reads in ASCII files.  There should not be any problem
  reading the file if you use FITS format.



New features:
-------------



Bug fixes:
----------

- Added tot attribute to the NN and NNN output files, which fixes an error
  where NNCorrelation and NNNCorrelation did not round trip correctly through
  a FITS output file.  Now the tot attribute is set properly when reading.

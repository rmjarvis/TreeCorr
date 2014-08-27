Changes from version 3.0 to 3.1
===============================


API changes:
------------

- Changed class names G2Correlation -> GGCorrelation, N2Correlation ->
  NNCorrelation, and K2Correlation->KKCorrelation.  I realized that I had
  been kind of inconsistent in my use of G2 vs GG to mean the two-point
  shear correaltion.  Considering that g1,g2 are the shear components, I 
  think the former may be confusing, so I changed everything to use 
  the GG notation throughout the code and documentation, most critically
  in these class names. (#9)

- Changed the config parameters g2_file_name -> gg_file_name, n2_file_name ->
  nn_file_name, and k2_file_name -> kk_file_name for the same reason as
  above.  In this case though, I kept the old parameter names as valid
  aliases for the new names.  With verbose >= 1, you will get a warning
  about the name change, but it will still work. (#9)


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

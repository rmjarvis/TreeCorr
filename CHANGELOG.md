Changes from 2.6 to 3.0
=======================

Version 3 represents a complete overhaul of the corr2 code.  
Much of the I/O code in particular is now done in python, rather than C++.
In addition, the C++ code is all available from within python, which means
that it should be easy for people to write their own scripts using the 
python module if they want to do something a bit different with how the 
data is read in or to do something directly with the computed correlation
function in python.

In addition there are a few fairly minor changes in the API.  I think these
are primarily edge cases, and the code should raise an exception if your
configuration file does any of these to indicate that you need to change it.

- Removed smooth_scale option.  It is easy enough for people to do this
  themselves if they want, so it seems unnecessary.
- Removed the extra columns that had been written out in the case of
  compensated NG and NK calculations.
- Changed the handling of file_name and file_name2 slightly.  Before, the
  two files for a cross-correlation could be specified with just file_name
  using 2 entries.  Now these need to be listed as separate key words
  with file_name and file_name2.
  


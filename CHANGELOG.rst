Changes from version 3.1 to 3.2
===============================

The numbers at the ends of some items below indicate which issue is connected
with the change:

https://github.com/rmjarvis/TreeCorr/issues?q=milestone%3A%22Version+3.2%22+is%3Aclosed


Dependency change:
------------------

- Removed the option of using pyfits for the FITS I/O.  Now fitsio is a 
  required dependency.  The installation procedure should handle this for you,
  so this change shouldn't matter at all to you.  Because that is so easy, it
  didn't seem worth maintaining two options for the FITS dependency.  In most
  respects, fitsio is the better package, so I went with that one.


API changes:
------------

- Changed the default value of bin_slop so that bin_slop * bin_size is at most
  0.1.  When this product (which I refer to as b) is larger than 0.1, there
  can be significant inaccuracies in the resulting correlation function.  So
  now, if bin_slop is unspecified, it is set to 0.1/bin_size if bin_size > 0.1,
  or just 1.0 if bin_size <= 0.1. (#15)
- Changed the <R> column in the output to actually be <R> (it used to really
  be exp(<logR>) and added <logR> column.  This means if you were accessing the
  output columsn by column number, you will need to change your code to add
  1, since most of the columns have been bumped 1 column to the right.
  (While you're at it, you should switch to accessing them by column name
  instead, maybe via the FITS I/O option...) (#28)
- Changed xi+ and xi- column names in the GG output files to xip and xim.  The
  former names didn't work well with numpy.genfromtxt(... names=True), since
  it removes punctuation, so the resulting names became ambiguous.
  Note: Even though there was no problem with the old names in FITS files,
  I changed the names there as well to be consistent.
- Removed option of giving m2_uform parameter in the config dict or kwargs
  when constructing a `GGCorrelation` object.  Now this needs to be specified
  when doing anything that uses it (e.g. `calculateMapSq`).


New features:
-------------

- Added three point functions.  So far just NNN, KKK, and GGG. (#3, #4, #33)
- Added ability to use the perpendicular component of the distance as the
  separation metric with `process(cat, metric='Rperp')` (#17)
- Added the ability for the Correlation classes to write FITS binary tables.
  If an output file name has an extension that starts with '.fit', then it will
  write a FITS binary table, rather than an ASCII file.  Or you can specify the
  file type with the file_type parameter. (#22)
- Added a read() function to the Correlation classes to read data back in from
  and output file.  The combination of this with the FITS write lets you save a
  Correlation object to disk and recover it without loss of information.  This
  can be used to split a problem up to be run on multiple machines. (#23)
- Added ability to add one Correlation object to another to enable doing the
  calculation in parts over several processes. (#23)
- Added the option of calling `NNCorrelation.write()` with no `rr` parameter.
  In this case, it will just write a column with the number of pairs (along
  with the regular separation columns), rather than calculate the correlation 
  function.
- Added ability to write Catalogs to either FITS or ASCII files.


Bug fixes:
----------

- Changed the `process` functions to check for a 'num_threads' parameter in
  the config dict.  The 'num_threads' had been checked only when loading the 
  config dict from a file, which meant that there was no obvious way to change
  the number of threads in the python layer.  Now having that parameter in 
  the config dict does the right thing in all the places where it is
  relevant. (#21)

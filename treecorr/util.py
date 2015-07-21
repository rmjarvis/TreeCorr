# Copyright (c) 2003-2014 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: util
"""

def gen_write(self, file_name, col_names, columns, file_type=None):
    """Write some columns to an output file with the given column names.

    We do this basic functionality a lot, so put the code to do it in one place.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    :param file_type:   Which kind of file to write to. (default: determine from the file_name
                        extension)
    """
    import numpy
    if len(col_names) != len(columns):
        raise ValueError("col_names and columns are not the same length.")

    # Figure out which file type the catalog is
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        else:
            file_type = 'ASCII'
        self.logger.info("file_type assumed to be %s from the file name.",file_type)

    if file_type == 'FITS':
        gen_write_fits(self, file_name, col_names, columns)
    elif file_type == 'ASCII':
        gen_write_ascii(self, file_name, col_names, columns)
    else:
        raise ValueError("Invalid file_type %s"%file_type)


def gen_write_ascii(self, file_name, col_names, columns):
    """Write some columns to an output ASCII file with the given column names.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.  These will be written
                        in a header comment line at the top of the output file.
    :param columns:     A list of numpy arrays with the data to write.
    """
    import numpy
    import treecorr
    
    ncol = len(col_names)
    data = numpy.empty( (self.nbins, ncol) )
    for i,col in enumerate(columns):
        data[:,i] = col

    prec = treecorr.config.get(self.config,'precision',int,4)
    width = prec+8
    # Note: python 2.6 needs the numbers, so can't just do "{:^%d}"*ncol
    # Also, I have the first one be 1 shorter to allow space for the initial #.
    header_form = "{0:^%d}"%(width-1)
    for i in range(1,ncol):
        header_form += " {%d:^%d}"%(i,width)
    header = header_form.format(*col_names)
    fmt = '%%%d.%de'%(width,prec)
    try:
        numpy.savetxt(file_name, data, fmt=fmt, header=header)
    except (AttributeError, TypeError):
        # header was added with version 1.7, so do it by hand if not available.
        with open(file_name, 'w') as fid:
            fid.write('#' + header + '\n')
            numpy.savetxt(fid, data, fmt=fmt) 


def gen_write_fits(self, file_name, col_names, columns):
    """Write some columns to an output FITS file with the given column names.
    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    """
    import fitsio
    import numpy

    data = numpy.empty(self.nbins, dtype=[ (name,'f8') for name in col_names ])
    for (name, col) in zip(col_names, columns):
        data[name] = col

    fitsio.write(file_name, data, clobber=True)


def gen_read(self, file_name, file_type=None):
    """Read some columns from an input file.

    We do this basic functionality a lot, so put the code to do it in one place.
    Note that the input file is expected to have been written by TreeCorr using the 
    gen_write function, so we don't have a lot of flexibility in the input structure.

    :param file_name:   The name of the file to read.
    :param file_type:   Which kind of file to write to. (default: determine from the file_name
                        extension)

    :returns: a numpy ndarray with named columns
    """
    import numpy
    # Figure out which file type the catalog is
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        else:
            file_type = 'ASCII'
        self.logger.info("file_type assumed to be %s from the file name.",file_type)

    if file_type == 'FITS':
        import fitsio
        data = fitsio.read(file_name)
    elif file_type == 'ASCII':
        import numpy
        data = numpy.genfromtxt(file_name, names=True)
    else:
        raise ValueError("Invalid file_type %s"%file_type)

    return data




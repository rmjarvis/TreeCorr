# Copyright (c) 2003-2020 by Mike Jarvis
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

import os
import numpy as np

def ensure_dir(target):
    d = os.path.dirname(target)
    if d != '':
        if not os.path.exists(d):
            os.makedirs(d)

class AsciiWriter(object):
    """Write data to an ASCII (text) file.
    """
    def __init__(self, file_name, *, precision=4, logger=None):
        """
        Parameters:
            file_name:      The file name
            precision:      The number of digits of precision to output.
            logger:         If desired, a logger object for logging. (default: None)
        """
        self.file_name = file_name
        self.logger = logger
        self.set_precision(precision)
        self._file = None
        ensure_dir(file_name)

    def set_precision(self, precision):
        self.precision = precision
        self.width = precision+8
        self.fmt = '%%%d.%de'%(self.width, self.precision)

    @property
    def file(self):
        if self._file is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        return self._file

    def write(self, col_names, columns, *, params=None, ext=None):
        """Write some columns to an output ASCII file with the given column names.

        Parameters:
            col_names:      A list of columns names for the given columns.  These will be written
                            in a header comment line at the top of the output file.
            columns:        A list of numpy arrays with the data to write.
            params:         A dict of extra parameters to write at the top of the output file.
            ext:            Optional ext name for these data. (default: None)
        """
        ncol = len(col_names)
        data = np.empty( (len(columns[0]), ncol) )
        for i,col in enumerate(columns):
            data[:,i] = col

        # Note: The first one is 1 shorter to allow space for the initial #.
        header = ("#" + "{:^%d}"%(self.width-1) +
                    " {:^%d}"%(self.width) * (ncol-1) + "\n").format(*col_names)

        if ext is not None:
            s = '## %s\n'%ext
            self.file.write(s.encode())
        if params is not None:
            s = '## %r\n'%(params)
            self.file.write(s.encode())
        self.file.write(header.encode())
        np.savetxt(self.file, data, fmt=self.fmt)

    def __enter__(self):
        self._file = open(self.file_name, 'wb')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._file.close()
        self._file = None


class FitsWriter(object):
    """Writer interface for FITS files.
    """
    def __init__(self, file_name, *, logger=None):
        """
        Parameters:
            file_name:      The file name
            logger:         If desired, a logger object for logging. (default: None)
        """
        try:
            import fitsio  # noqa: F401
        except ImportError:
            if logger:
                logger.error("Unable to import fitsio.  Cannot write to %s"%file_name)
            raise
        self.file_name = file_name
        self.logger = logger
        self._file = None
        ensure_dir(file_name)

    def set_precision(self, precision):
        pass

    @property
    def file(self):
        if self._file is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        return self._file

    def write(self, col_names, columns, *, params=None, ext=None):
        """Write some columns to an output ASCII file with the given column names.

        If name is not None, then it is used as the name of the extension for these data.

        Parameters:
            col_names:      A list of columns names for the given columns.  These will be written
                            in a header comment line at the top of the output file.
            columns:        A list of numpy arrays with the data to write.
            params:         A dict of extra parameters to write at the top of the output file.
            ext:            Optional ext name for these data. (default: None)
        """
        data = np.empty(len(columns[0]), dtype=[ (c,'f8') for c in col_names ])
        for (c, col) in zip(col_names, columns):
            data[c] = col
        self.file.write(data, header=params, extname=ext)

    def __enter__(self):
        import fitsio
        self._file = fitsio.FITS(self.file_name, 'rw', clobber=True)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()
        self._file = None


class HdfWriter(object):
    """Writer interface for HDF5 files.
    Uses h5py to read columns, etc.
    """
    def __init__(self, file_name, *, logger=None):
        """
        Parameters:
            file_name:     The file name
            logger:         If desired, a logger object for logging. (default: None)
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            if logger:
                logger.error("Unable to import h5py.  Cannot write to %s"%file_name)
            raise
        self.file_name = file_name
        self.logger = logger
        self._file = None
        ensure_dir(file_name)

    def set_precision(self, precision):
        pass

    @property
    def file(self):
        if self._file is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        return self._file

    def write(self, col_names, columns, *, params=None, ext=None):
        """Write some columns to an output ASCII file with the given column names.

        If name is not None, then it is used as the name of the extension for these data.

        Parameters:
            col_names:      A list of columns names for the given columns.  These will be written
                            in a header comment line at the top of the output file.
            columns:        A list of numpy arrays with the data to write.
            params:         A dict of extra parameters to write at the top of the output file.
            ext:            Optional group name for these data. (default: None)
        """
        if ext is not None:
            hdf = self.file.create_group(ext)
        else:
            hdf = self.file
        if params is not None:
            hdf.attrs.update(params)
        for (name, col) in zip(col_names, columns):
            hdf.create_dataset(name, data=col)

    def __enter__(self):
        import h5py
        self._file = h5py.File(self.file_name, 'w')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # closes file at end of "with" statement
        self._file.close()
        self._file = None

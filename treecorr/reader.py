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

"""
This is a set of helper classes that read data for treecorr.Catalog objects.
They have a bit of a mixed bag of functions, not intended to be complete,
just to cover the needs of that class.

HDF and FITS files differ in that the former can have different length columns
in the same data group, unlike fits HDUs, which have columns of all the same length.
Where possible we check for that here, and a user would have to be fairly determined to
trigger this.

The other difference is that fitsio reads a structured array, whereas h5py will
read a dictionary of arrays. This does not cause problems here, since both are
indexed by string, but may prevent usage elsewhere. If so we could convert them to
both provide dicts.
"""
import numpy as np
from .config import get_from_list

class AsciiReader:
    """Reader interface for ASCII files.
    Uses pandas if possible, else numpy.
    """
    can_slice = True

    def __init__(self, file_name, delimiter=None, comment_marker='#'):
        """
        Parameters:
            file_name (str)         The file name
            delimiter (str):        What delimiter to use between values.  (default: None,
                                    which means any whitespace)
            comment_marker (str):   What token indicates a comment line. (default: '#')
        """
        self.file_name = file_name
        self.delimiter = delimiter
        self.comment_marker = comment_marker
        self._data = None

    def __contains__(self, ext):
        """Check if ext is None.

        ASCII files don't have extensions, so the only ext allowed is None.

        Parameters:
            ext (str):      The extension to check

        Returns:
            Whether ext is None
        """
        # None is the only valid "extension" for ASCII files
        return ext is None

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        None is the only valid extension for ASCII files.

        Parameters:
            ext (str)   The extension to check
        """
        if ext is not None:
            raise ValueError("Invalid ext={} for file {}".format(ext,self.file_name))

    @property
    def data(self):
        if self._data is not None:
            return self._data
        if self.ncols is None:
            raise RuntimeError('Cannot read when not in a "with" context')

        # I want read_csv to ignore header lines that start with the comment marker, but
        # there is currently a bug in read_csv that messing things up when we do this.
        # cf. https://github.com/pydata/pandas/issues/4623
        # For now, my workaround in to count how many lines start with the comment marker
        # and skip them by hand.
        skiprows = 0
        with open(self.file_name, 'r') as fid:
            for line in fid:  # pragma: no branch
                if line.startswith(self.comment_marker): skiprows += 1
                else: break
        #skiprows += self.start
        #if self.end is None:
            #nrows = None
        #else:
            #nrows = self.end - self.start
        #if self.every_nth != 1:
            #start = skiprows
            #skiprows = lambda x: x < start or (x-start) % self.every_nth != 0
            #nrows = (nrows-1) // self.every_nth + 1
        try:
            import pandas
            if self.delimiter is None:
                data = pandas.read_csv(self.file_name, comment=self.comment_marker,
                                       delim_whitespace=True,
                                       header=None, skiprows=skiprows)
            else:
                data = pandas.read_csv(self.file_name, comment=self.comment_marker,
                                       delimiter=self.delimiter,
                                       header=None, skiprows=skiprows)
            data = data.dropna(axis=0).values
        except ImportError:
            #if self.every_nth == 1:
            if True:
                data = np.genfromtxt(self.file_name, comments=self.comment_marker,
                                     delimiter=self.delimiter,
                                     skip_header=skiprows)
            else:
                # Numpy can't handle skiprows being a function.  Have to do this manually.
                data = np.genfromtxt(self.file_name, comments=self.comment_marker,
                                     delimiter=self.delimiter,
                                     skip_header=start, max_rows=self.end - self.start)
                data = data[::self.every_nth]

        # If only one row, and not using pands, then the shape comes in as one-d.  Reshape it:
        if len(data.shape) == 1:
            data = data.reshape(1,-1)

        self._data = data
        return self._data

    def read(self, ext, cols, s):
        """Read a slice of a column or list of columns from a specified extension.

        Parameters:
            ext (str):          The extension (ignored)
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read

        Returns:
            The data as a dict
        """
        if np.isscalar(cols):
            return self.data[:,int(cols)-1][s]
        else:
            return {col : self.data[:,int(col)-1][s] for col in cols}

    def row_count(self, ext=None, col=None):
        """Count the number of rows in the file.

        Parameters:
            ext (str):  The extension (ignored)
            col (str):  The column to use (ignored)

        Returns:
            The number of rows
        """
        return self.data.shape[0]

    def names(self, ext):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str)   The extension (ignored)

        Returns:
            A list of string column names
        """
        return [str(i) for i in range(1,self.ncols+1)]

    def __enter__(self):
        # Context manager, enables "with AsciiReader(filename) as f:"

        # Do a trivial read of 1 row, just to get basic info about columns
        data = np.genfromtxt(self.file_name, comments=self.comment_marker, delimiter=self.delimiter,
                             max_rows=1)
        if len(data.shape) != 1:  # pragma: no cover
            raise IOError('Unable to parse the input catalog as a numpy array')
        self.ncols = data.shape[0]

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._data = None

    def choose_extension(self, config, name, num, default=None):
        """Select an extension name or index from a configuration.

        Only None is valid, so this always returns None.

        Parameters:
            config (dict):      The config to choose from
            name (str):         The parameter name to get
            num (int):          If the value is a list, which item to get
            default (int/str):  The default if not found (optional)

        Returns:
            None
        """
        return None


class FitsReader:
    """Reader interface for FITS files.
    Uses fitsio to read columns, etc.
    """

    def __init__(self, file_name):
        """
        Parameters:
            file_name (str)     The file name
        """
        import fitsio

        self.file = None  # Only works inside a with block.

        # record file name to know what to open when entering
        self.file_name = file_name

        # There is a bug in earlier fitsio versions that prevents slicing
        self.can_slice = fitsio.__version__ > '1.0.6'

    def read(self, ext, cols, s):
        """Read a slice of a column or list of columns from a specified extension

        Parameters:
            ext (str):          The FITS extension to use
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read

        Returns:
            The data as a recarray
        """
        return self.file[ext][cols][s]

    def row_count(self, ext, col=None):
        """Count the number of rows in the named extension

        For compatibility with the HDF interface, which can have columns
        of different lengths, we allow a second argument, col, but it is
        ignored here.

        Parameters:
            ext (str):  The FITS extension to use
            col (str):  The column to use (ignored)

        Returns:
            The number of rows
        """
        return self.file[ext].get_nrows()

    def names(self, ext):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str)   The extension to search for columns

        Returns:
            A list of string column names
        """
        return self.file[ext].get_colnames()

    def __contains__(self, ext):
        """Check if there is an extension with the given name or index in the file.

        This may be either a name or an integer.

        Parameters:
            ext (str/int):  The extension to check for

        Returns:
            Whether the extension exists
        """
        return ext in self.file

    def __enter__(self):
        # Context manager, enables "with FitsReader(filename) as f:"
        import fitsio
        self.file = fitsio.FITS(self.file_name, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Context manager closer - we just close the file at the end,
        # regardless of the error
        self.file.close()
        self.file = None

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        The ext must both exist and be a table (not an image)

        Parameters:
            ext (str/int)   The extension to check
        """
        import fitsio

        if ext not in self:
            raise ValueError("Invalid ext={} for file {} (does not exist)".format(
                             ext, self.file_name))

        if not isinstance(self.file[ext], fitsio.hdu.TableHDU):
            raise ValueError("Invalid ext={} for file {} (Not a TableHDU)".format(
                             ext, self.file_name))

    def choose_extension(self, config, name, num, default=None):
        """Select an extension name or index from a configuration.

        If no key is found or default supplied, fall back to the first FITS extension.

        Parameters:
            config (dict):      The config to choose from
            name (str):         The parameter name to get
            num (int):          If the value is a list, which item to get
            default (int/str):  The default if not found (optional)

        Returns:
            A valid extension to use
        """
        # get the value as a string - if it's actually an int
        # we will convert below
        ext = get_from_list(config, name, num, str)

        # If not found, use the default if present, otherwise the global
        # default of 1
        if ext is None:
            if default is None:
                ext = 1
            else:
                ext = default

        # FITS extensions can be indexed by number or
        # string.  Try converting to an integer if the current
        # value is not found.  If not let the error be caught later.
        if ext not in self:
            try:
                ext = int(ext)
            except ValueError:
                pass

        return ext


class HdfReader:
    """Reader interface for HDF5 files.
    Uses h5py to read columns, etc.
    """
    # h5py can always accept slices as indices
    can_slice = True

    def __init__(self, file_name):
        """
        Parameters:
            file_name (str)     The file name
        """
        import h5py  # Just to check right away that it will work.
        self.file = None  # Only works inside a with block.
        self.file_name = file_name

    def __contains__(self, ext):
        """Check if there is an extension with the given name in the file.

        Parameters:
            ext (str):      The extension to check for

        Returns:
            Whether the extension exists
        """
        return ext in self.file.keys()

    def _group(self, ext):
        # get a group from a name, using
        # the root if the group is empty
        if ext == '':
            ext = '/'
        return self.file[ext]

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        The ext must exist - there is no other requirement for HDF files.

        Parameters:
            ext (str)   The extension to check
        """
        # Allow '' as an alias of '/'
        if ext != '' and ext not in self:
            raise ValueError("Invalid ext={} for file {} (does not exist)".format(
                             ext,self.file_name))

    def read(self, ext, cols, s):
        """Read a slice of a column or list of columns from a specified extension.

        Slices should always be used when reading HDF files - using a sequence of
        integers is painfully slow.

        Parameters:
            ext (str):          The HDF (sub-)group to use
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read

        Returns:
            The data as a dict
        """
        g = self._group(ext)
        if np.isscalar(cols):
            return g[cols][s]
        else:
            return {col : g[col][s] for col in cols}

    def row_count(self, ext, col):
        """Count the number of rows in the named extension and column

        Unlike in FitsReader, col is required.

        Parameters:
            ext (str):  The HDF group name to use
            col (str):  The column to use

        Returns:
            The number of rows
        """
        return self._group(ext)[col].size

    def names(self, ext):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str)   The extension to search for columns

        Returns:
            A list of string column names
        """
        return list(self._group(ext).keys())

    def __enter__(self):
        # Context manager, enables "with HdfReader(filename) as f:"
        import h5py
        self.file = h5py.File(self.file_name, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # closes file at end of "with" statement
        self.file.close()
        self.file = None

    def choose_extension(self, config, name, num, default=None):
        """Select an extension name or index from a configuration.

        If no key is found or default supplied, fall back to the
        HDF root object ('/')

        Parameters:
            config (dict):      The config to choose from
            name (str):         The parameter name to get
            num (int):          If the value is a list, which item to get
            default (int/str):  The default if not found (optional)

        Returns:
            The HDF group name
        """
        ext = get_from_list(config, name, num, str)

        # If not found, use the default if present, otherwise the global
        # default of using the root of the file
        if ext is None:
            if default is None:
                ext = '/'
            else:
                ext = default

        return ext
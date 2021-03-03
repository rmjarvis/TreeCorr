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

class AsciiReader(object):
    """Reader interface for ASCII files using numpy.
    """
    can_slice = True
    default_ext = None

    def __init__(self, file_name, delimiter=None, comment_marker='#'):
        """
        Parameters:
            file_name (str):        The file name
            delimiter (str):        What delimiter to use between values.  (default: None,
                                    which means any whitespace)
            comment_marker (str):   What token indicates a comment line. (default: '#')
        """
        self.file_name = file_name
        self.delimiter = delimiter
        self.comment_marker = comment_marker
        self.ncols = None
        self.nrows = None

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
            ext (str):  The extension to check
        """
        if ext is not None:
            raise ValueError("Invalid ext={} for file {}".format(ext,self.file_name))


    def read(self, cols, s=slice(None), ext=None):
        """Read a slice of a column or list of columns from a specified extension.

        Parameters:
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read (default: all)
            ext (str):          The extension (ignored)

        Returns:
            The data as a dict or single numpy array as appropriate
        """
        if self.ncols is None:
            raise RuntimeError('Illegal operation when not in a "with" context')

        # Figure out how many rows to skip at the start
        if isinstance(s, slice) and s.start is not None:
            skiprows = self.comment_rows + s.start
        else:
            skiprows = self.comment_rows

        # And how many to read (if we know)
        # Note: genfromtxt can't handle a skip, so defer that to later.
        # Also if s is an array, that won't work here either.
        if isinstance(s, slice) and s.start is not None and s.stop is not None:
            nrows = s.stop - s.start
        else:
            nrows = None

        # And which columns to read
        geti = lambda col: self.col_names.index(col) if col in self.col_names else int(col)-1
        if np.isscalar(cols):
            icols = [geti(cols)]
        else:
            icols = [geti(col) for col in cols]

        # Actually read the data
        data = np.genfromtxt(self.file_name, comments=self.comment_marker,
                             delimiter=self.delimiter, usecols=icols,
                             skip_header=skiprows, max_rows=nrows)

        # If only one column, then the shape comes in as one-d.  Reshape it:
        if len(icols) == 1:
            data = data.reshape(len(data),1)

        # If only one row, then the shape comes in as one-d.  Reshape it:
        if len(data.shape) == 1:
            data = data.reshape(1,len(data))

        # Select the rows we want if start/end wasn't sufficient.
        if isinstance(s, slice):
            data = data[::s.step,:]
        else:
            data = data[s,:]

        # Return is slightly different if we have multiple columns or not.
        if np.isscalar(cols):
            return data[:,0]
        else:
            return {col : data[:,i] for i,col in enumerate(cols)}

    def row_count(self, col=None, ext=None):
        """Count the number of rows in the file.

        Parameters:
            col (str):  The column to use (ignored)
            ext (str):  The extension (ignored)

        Returns:
            The number of rows
        """
        if self.ncols is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        if self.nrows is not None:
            return self.nrows
        # cf. https://stackoverflow.com/a/850962/1332281
        # On my system with python 3.7, bufcount was the fastest among these solutions.
        # I also found 256K was the optimal buf size for my system.  Probably YMMV, but
        # micro-optimizing this is not so important.  It's never used by treecorr.
        # Only the FitsReader ever calls row_count other than from the unit tests.
        with open(self.file_name) as f:
            lines = 0
            buf_size = 256 * 1024
            buf = f.read(buf_size)
            while buf:
                lines += buf.count('\n')
                buf = f.read(buf_size)
        self.nrows = lines - self.comment_rows
        return self.nrows

    def names(self, ext=None):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str):  The extension (ignored)

        Returns:
            A list of string column names
        """
        if self.ncols is None:
            raise RuntimeError('Cannot get names when not in a "with" context')

        # Include both int values as strings and any real names we know about.
        return [str(i+1) for i in range(self.ncols)] + list(self.col_names)

    def __enter__(self):
        # See how many comment rows there are at the start
        self.comment_rows = 0
        with open(self.file_name, 'r') as fid:
            for line in fid:  # pragma: no branch
                if line.startswith(self.comment_marker): self.comment_rows += 1
                else: break

        # Do a trivial read of 1 row, just to get basic info about columns
        self.ncols = None
        if self.comment_rows >= 1:
            try:
                data = np.genfromtxt(self.file_name, comments=self.comment_marker,
                                     delimiter=self.delimiter, names=True,
                                     skip_header=self.comment_rows-1, max_rows=1)
                self.col_names = data.dtype.names
                self.ncols = len(self.col_names)
            except Exception:
                pass
        if self.ncols is None:
            data = np.genfromtxt(self.file_name, comments=self.comment_marker,
                                 delimiter=self.delimiter, max_rows=1)
            self.col_names = []
            if len(data.shape) != 1:  # pragma: no cover
                raise IOError('Unable to parse the input catalog as a numpy array')
            self.ncols = data.shape[0]

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.ncols = None  # Marker that we are not in context
        self.nrows = None


class PandasReader(AsciiReader):
    """Reader interface for ASCII files using pandas.
    """
    def __init__(self, file_name, delimiter=None, comment_marker='#'):
        """
        Parameters:
            file_name (str):        The file name
            delimiter (str):        What delimiter to use between values.  (default: None,
                                    which means any whitespace)
            comment_marker (str):   What token indicates a comment line. (default: '#')
        """
        # Do this immediately, so we get an ImportError if it isn't available.
        import pandas  # noqa: F401

        AsciiReader.__init__(self, file_name, delimiter, comment_marker)
        # This is how pandas handles whitespace
        self.sep = r'\s+' if self.delimiter is None else self.delimiter


    def read(self, cols, s=slice(None), ext=None):
        """Read a slice of a column or list of columns from a specified extension.

        Parameters:
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read (default: all)
            ext (str):          The extension (ignored)

        Returns:
            The data as a dict or single numpy array as appropriate
        """
        import pandas
        if self.ncols is None:
            raise RuntimeError('Cannot read when not in a "with" context')

        # Figure out how many rows to skip at the start
        if isinstance(s, slice) and s.start is not None:
            skiprows = self.comment_rows + s.start
        else:
            skiprows = self.comment_rows

        # And how many to read (if we know)
        # Note: genfromtxt can't handle a skip, so defer that to later.
        # Also if s is an array, that won't work here either.
        if isinstance(s, slice) and s.start is not None and s.stop is not None:
            nrows = s.stop - s.start
        else:
            nrows = None

        # Pandas has the ability to skip according to a function, so we can accommodate
        # arbitrary s (either slice or array of indices):
        if isinstance(s, slice) and s.step is not None:
            start = skiprows
            skiprows = lambda x: x < start or (x-start) % s.step != 0
            if nrows is not None:
                nrows = (nrows-1) // s.step + 1

        if not isinstance(s, slice):
            # Then s is a numpy array of indices
            start = skiprows
            ss = set(s)  # for efficiency
            skiprows = lambda x: x-start not in ss

        # And which columns to read
        geti = lambda col: self.col_names.index(col) if col in self.col_names else int(col)-1
        if np.isscalar(cols):
            icols = [geti(cols)]
        else:
            icols = [geti(col) for col in cols]

        # Actually read the data
        df = pandas.read_csv(self.file_name, comment=self.comment_marker,
                             sep=self.sep, usecols=icols, header=None,
                             skiprows=skiprows, nrows=nrows)

        # Return is slightly different if we have multiple columns or not.
        if np.isscalar(cols):
            return df.iloc[:,0].to_numpy()
        else:
            return {col : df.loc[:,icols[i]].to_numpy() for i,col in enumerate(cols)}

class ParquetReader():
    """Reader interface for Parquet files using pandas.
    """
    can_slice = True
    default_ext = None

    def __init__(self, file_name, delimiter=None, comment_marker='#'):
        """
        Parameters:
            file_name (str):        The file name
            delimiter (str):        What delimiter to use between values.  (default: None,
                                    which means any whitespace)
            comment_marker (str):   What token indicates a comment line. (default: '#')
        """
        # Do this immediately, so we get an ImportError if it isn't available.
        import pandas  # noqa: F401

        self.file_name = file_name
        self._df = None

    @property
    def df(self):
        if self._df is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        return self._df

    def __contains__(self, ext):
        """Check if ext is None.

        Parquet files don't have extensions, so the only ext allowed is None.

        Parameters:
            ext (str):      The extension to check

        Returns:
            Whether ext is None
        """
        # None is the only valid "extension" for Parquet files
        return ext is None

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        None is the only valid extension for ASCII files.

        Parameters:
            ext (str):  The extension to check
        """
        if ext is not None:
            raise ValueError("Invalid ext={} for file {}".format(ext,self.file_name))

    def read(self, cols, s=slice(None), ext=None):
        """Read a slice of a column or list of columns from a specified extension.

        Parameters:
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read (default: all)
            ext (str):          The extension (ignored)

        Returns:
            The data as a recarray or simple numpy array as appropriate
        """
        if np.isscalar(cols):
            return self.df[cols][s].to_numpy()
        else:
            return self.df[cols][s].to_records()

    def row_count(self, col=None, ext=None):
        """Count the number of rows in the named extension and column

        Unlike in FitsReader, col is required.

        Parameters:
            col (str):  The column to use (ignored)
            ext (str):  The extension (ignored)

        Returns:
            The number of rows
        """
        return len(self.df)

    def names(self, ext=None):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str):  The extension to search for columns (ignored)

        Returns:
            A list of string column names
        """
        return self.df.columns

    def __enter__(self):
        import pandas
        self._df = pandas.read_parquet(self.file_name)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # free the memory in the dataframe at end of "with" statement
        self._df = None


class FitsReader(object):
    """Reader interface for FITS files.
    Uses fitsio to read columns, etc.
    """
    default_ext = 1

    def __init__(self, file_name):
        """
        Parameters:
            file_name (str):    The file name
        """
        import fitsio

        self._file = None  # Only works inside a with block.

        # record file name to know what to open when entering
        self.file_name = file_name

        # There is a bug in earlier fitsio versions that prevents slicing
        self.can_slice = fitsio.__version__ > '1.0.6'

    @property
    def file(self):
        if self._file is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        return self._file

    def __contains__(self, ext):
        """Check if there is an extension with the given name or index in the file.

        This may be either a name or an integer.

        Parameters:
            ext (str/int):  The extension to check for (default: 1)

        Returns:
            Whether the extension exists
        """
        ext = self._update_ext(ext)
        return ext in self.file

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        The ext must both exist and be a table (not an image)

        Parameters:
            ext (str/int):  The extension to check
        """
        import fitsio

        ext = self._update_ext(ext)
        if ext not in self:
            raise ValueError("Invalid ext={} for file {} (does not exist)".format(
                             ext, self.file_name))

        if not isinstance(self.file[ext], fitsio.hdu.TableHDU):
            raise ValueError("Invalid ext={} for file {} (Not a TableHDU)".format(
                             ext, self.file_name))

    def _update_ext(self, ext):
        # FITS extensions can be indexed by number or
        # string.  Try converting to an integer if the current
        # value is not found.  If not let the error be caught later.
        if ext not in self.file:
            try:
                ext = int(ext)
            except ValueError:
                pass
        return ext

    def read(self, cols, s=slice(None), ext=1):
        """Read a slice of a column or list of columns from a specified extension

        Parameters:
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read (default: all)
            ext (str/int)):     The FITS extension to use (default: 1)

        Returns:
            The data as a recarray or simple numpy array as appropriate
        """
        ext = self._update_ext(ext)
        return self.file[ext][cols][s]

    def row_count(self, col=None, ext=1):
        """Count the number of rows in the named extension

        For compatibility with the HDF interface, which can have columns
        of different lengths, we allow a second argument, col, but it is
        ignored here.

        Parameters:
            col (str):      The column to use (ignored)
            ext (str/int):  The FITS extension to use (default: 1)

        Returns:
            The number of rows
        """
        ext = self._update_ext(ext)
        return self.file[ext].get_nrows()

    def names(self, ext=1):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str/int):  The extension to search for columns (default: 1)

        Returns:
            A list of string column names
        """
        ext = self._update_ext(ext)
        return self.file[ext].get_colnames()

    def __enter__(self):
        import fitsio
        self._file = fitsio.FITS(self.file_name, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Context manager closer - we just close the file at the end,
        # regardless of the error
        self.file.close()
        self._file = None


class HdfReader(object):
    """Reader interface for HDF5 files.
    Uses h5py to read columns, etc.
    """
    # h5py can always accept slices as indices
    can_slice = True
    default_ext = '/'

    def __init__(self, file_name):
        """
        Parameters:
            file_name (str):    The file name
        """
        import h5py  # noqa: F401  Just to check right away that it will work.

        self._file = None  # Only works inside a with block.
        self.file_name = file_name

    @property
    def file(self):
        if self._file is None:
            raise RuntimeError('Illegal operation when not in a "with" context')
        return self._file

    def __contains__(self, ext):
        """Check if there is an extension with the given name in the file.

        Parameters:
            ext (str):      The extension to check for

        Returns:
            Whether the extension exists
        """
        return ext in self.file

    def _group(self, ext):
        # get a group from a name, using
        # the root if the group is empty
        return self.file[ext]

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        The ext must exist - there is no other requirement for HDF files.

        Parameters:
            ext (str):  The extension to check
        """
        if ext not in self:
            raise ValueError("Invalid ext={} for file {} (does not exist)".format(
                             ext,self.file_name))

    def read(self, cols, s=slice(None), ext='/'):
        """Read a slice of a column or list of columns from a specified extension.

        Slices should always be used when reading HDF files - using a sequence of
        integers is painfully slow.

        Parameters:
            cols (str/list):    The name(s) of column(s) to read
            s (slice/array):    A slice object or selection of integers to read (default: all)
            ext (str):          The HDF (sub-)group to use (default: '/')

        Returns:
            The data as a dict or single numpy array as appropriate
        """
        g = self._group(ext)
        if np.isscalar(cols):
            return g[cols][s]
        else:
            return {col : g[col][s] for col in cols}

    def row_count(self, col, ext='/'):
        """Count the number of rows in the named extension and column

        Unlike in FitsReader, col is required.

        Parameters:
            col (str):  The column to use
            ext (str):  The HDF group name to use (default: '/')

        Returns:
            The number of rows
        """
        return self._group(ext)[col].size

    def names(self, ext='/'):
        """Return a list of the names of all the columns in an extension

        Parameters:
            ext (str):  The extension to search for columns (default: '/')

        Returns:
            A list of string column names
        """
        return list(self._group(ext).keys())

    def __enter__(self):
        import h5py
        self._file = h5py.File(self.file_name, 'r')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # closes file at end of "with" statement
        self._file.close()
        self._file = None

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

class FitsReader:
    """Reader interface for FITS files.
    Uses fitsio to read columns, etc.
    """
    def __init__(self, file_name):
        """Open a file

        Parameters
        ----------
        file_name: str
        """
        import fitsio
        self.file = fitsio.FITS(file_name, 'r')

        # record file name to make error messages more useful
        self.file_name = file_name

        # There is a bug in earlier fitsio versions that prevents
        # slicing
        self.can_slice = fitsio.__version__ > '1.0.6'

    def read(self, ext, cols, s):
        """Read a slice of a column or list of columns from a specified extension

        Parameters
        ----------
        ext: int/str
            The FITS extension to use
        cols: str/list
            The name(s) of column(s) to read
        s: slice/array
            A slice object or selection of integers to read

        Returns
        -------
        data: recarray
            An array of the read data
        """
        return self.file[ext][cols][s]

    def row_count(self, ext, col=None):
        """Count the number of rows in the named extension

        For compatibility with the HDF interface, which can have columns
        of different lengths, we allow a second argument, col, but it is
        ignored here.

        Parameters
        ----------
        ext: int/str
            The fits extension to use
        col: any
            Ignored here, but nominally a string column name

        Returns
        -------
        count: int
        """
        return self.file[ext].get_nrows()

    def names(self, ext):
        """Return a list of the names of all the columns in an extension

        Parameters
        ----------
        ext
        """
        return self.file[ext].get_colnames()

    def __contains__(self, ext):
        """Check if there is an extension with the given name or
        index in the file.

        Parameters
        ----------
        ext: int/str
            name or index to search for

        Returns
        -------
        bool
            Whether the extension exists
        """
        return ext in self.file

    def __enter__(self):
        # Context manager, enables "with FitsReader(filename) as f:"
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Context manager closer - we just close the file at the end,
        # regardless of the error
        self.file.close()

    def check_valid_ext(self, ext):
        """Check if an extension is valid for reading, and raise ValueError if not.

        The ext must both exist and be a table (not an image)

        Parameters
        ----------
        ext: int/str
            The extension to check
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

        Parameters
        ----------
        config: dict
            config to choose from
        name: str
            parameter name to get
        num: int
            if the value is a list, which item to get
        default: int/str
            optional, the fall-back if not found

        Returns
        -------
        ext: int/str
            The value that can be used to look up the extension
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
        import h5py
        self.file = h5py.File(file_name, 'r')
        self.file_name = file_name

    def __contains__(self, ext):
        """Check if there is an extension with the given name in the file.

        Parameters
        ----------
        ext: str
            name or index to search for

        Returns
        -------
        bool
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

        Parameters
        ----------
        ext: str
            The extension to check
        """
        # Allow '' as an alias of '/'
        if ext != '' and ext not in self:
            raise ValueError("Invalid ext={} for file {} (does not exist)".format(
                             ext,self.file_name))

    def read(self, ext, cols, s):
        """Read a slice of a column or list of columns from a specified extension.

        Slices should always be used when reading HDF files - using a sequence of
        integers is painfully slow.

        Parameters
        ----------
        ext: str
            The HDF (sub-)group to use
        cols: str/list
            The name(s) of column(s) to read
        s: slice/array
            A slice object or selection of integers to read

        Returns
        -------
        data: dict
            The data that is read.
        """
        g = self._group(ext)
        if np.isscalar(cols):
            data = g[cols][s]
        else:
            data = {col: g[col][s] for col in cols}
        return data

    def row_count(self, ext, col):
        """Count the number of rows in the named extension and column

        Unlike in FitsReader, col is required.

        Parameters
        ----------
        ext: str
            The HDF group name to use
        col: str
            The column to use

        Returns
        -------
        count: int
        """
        return self._group(ext)[col].size

    def names(self, ext):
        """Return a list of the names of all the columns in an extension

        Parameters
        ----------
        ext: str
            The extension to search for columns

        Returns
        -------
        names: list
            A list of string column names
        """
        return list(self._group(ext).keys())

    def __enter__(self):
        # Context manager, enables "with HdfReader(filename) as f:"
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # closes file at end of "with" statement
        self.file.close()

    def choose_extension(self, config, name, num, default=None):
        """Select an extension name or index from a configuration.

        If no key is found or default supplied, fall back to the
        HDF root object ('/')

        Parameters
        ----------
        config: dict
            config to choose from
        name: str
            parameter name to get
        num: int
            if the value is a list, which item to get
        default: str
            optional, the fall-back if not found

        Returns
        -------
        ext: str
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

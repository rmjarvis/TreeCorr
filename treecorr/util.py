# Copyright (c) 2003-2019 by Mike Jarvis
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

import numpy as np
import os
import coord

from . import _lib, _ffi, Rperp_alias

def ensure_dir(target):
    d = os.path.dirname(target)
    if d != '':
        if not os.path.exists(d):
            os.makedirs(d)

def set_omp_threads(num_threads, logger=None):
    """Set the number of OpenMP threads to use in the C++ layer.

    :param num_threads: The target number of threads to use
    :param logger:      If desired, a logger object for logging any warnings here. (default: None)

    :returns:           The  number of threads OpenMP reports that it will use.  Typically this
                        matches the input, but OpenMP reserves the right not to comply with
                        the requested number of threads.
    """
    input_num_threads = num_threads  # Save the input value.

    # If num_threads is auto, get it from cpu_count
    if num_threads is None or num_threads <= 0:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
        if logger:
            logger.debug('multiprocessing.cpu_count() = %d',num_threads)

    # Tell OpenMP to use this many threads
    if logger:
        logger.debug('Telling OpenMP to use %d threads',num_threads)
    num_threads = _lib.SetOMPThreads(num_threads)

    # Report back appropriately.
    if logger:
        logger.debug('OpenMP reports that it will use %d threads',num_threads)
        if num_threads > 1:
            logger.info('Using %d threads.',num_threads)
        elif input_num_threads is not None and input_num_threads != 1:
            # Only warn if the user specifically asked for num_threads != 1.
            logger.warning("Unable to use multiple threads, since OpenMP is not enabled.")

    return num_threads

def get_omp_threads():
    """Get the number of OpenMP threads currently set to be used in the C++ layer.

    :returns:           The  number of threads OpenMP reports that it will use.
    """
    return _lib.GetOMPThreads()

def parse_file_type(file_type, file_name, output=False, logger=None):
    """Parse the file_type from the file_name if necessary

    :param file_type:   The input file_type.  If None, then parse from file_name's extension.
    :param file_name:   The filename to use for parsing if necessary.
    :param output:      Limit to output file types (FITS/ASCII)? (default: False)
    :param logger:      A logger if desired. (default: None)

    :returns: The parsed file_type.
    """
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        elif ext.lower().startswith('.hdf'):
            file_type = 'HDF'
        elif not output and ext.lower().startswith('.par'):
            file_type = 'Parquet'
        else:
            file_type = 'ASCII'
        if logger:
            logger.info("   file_type assumed to be %s from the file name.",file_type)
    return file_type.upper()

def gen_write(file_name, col_names, columns, params=None, precision=4, file_type=None, logger=None):
    """Write some columns to an output file with the given column names.

    We do this basic functionality a lot, so put the code to do it in one place.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    :param params:      A dict of extra parameters to write at the top of the output file (for
                        ASCII output) or in the header (for FITS output).  (default: None)
    :param precision:   Output precision for ASCII. (default: 4)
    :param file_type:   Which kind of file to write to. (default: determine from the file_name
                        extension)
    :param logger:      If desired, a logger object for logging. (default: None)
    """
    if len(col_names) != len(columns):
        raise ValueError("col_names and columns are not the same length.")
    if len(columns) == 0:
        raise ValueError("len(columns) == 0")
    for col in columns[1:]:
        if col.shape != columns[0].shape:
            raise ValueError("columns are not all the same shape")
    columns = [ col.flatten() for col in columns ]

    ensure_dir(file_name)

    # Figure out which file type to use.
    file_type = parse_file_type(file_type, file_name, output=True, logger=logger)

    if file_type == 'FITS':
        try:
            import fitsio  # noqa: F401
        except ImportError:
            if logger:
                logger.error("Unable to import fitsio.  Cannot write to %s"%file_name)
            raise
        with fitsio.FITS(file_name, 'rw', clobber=True) as fits:
            gen_write_fits(fits, col_names, columns, params)
    elif file_type == 'ASCII':
        with open(file_name, 'wb') as fid:
            gen_write_ascii(fid, col_names, columns, params, precision=precision)
    elif file_type == 'HDF':
        try:
            import h5py  # noqa: F401
        except ImportError:
            if logger:
                logger.error("Unable to import h5py.  Cannot write to %s"%file_name)
            raise
        with h5py.File(file_name, 'w') as hdf:
            gen_write_hdf(hdf, col_names, columns, params)

    else:
        raise ValueError("Invalid file_type %s"%file_type)

def gen_write_ascii(fid, col_names, columns, params, precision=4):
    """Write some columns to an output ASCII file with the given column names.

    :param fid:         An open file handler in binary mode. E.g. fid = open(file_name, 'wb')
    :param col_names:   A list of columns names for the given columns.  These will be written
                        in a header comment line at the top of the output file.
    :param columns:     A list of numpy arrays with the data to write.
    :param params:      A dict of extra parameters to write at the top of the output file.
    :param precision:   Output precision for ASCII. (default: 4)
    """
    ncol = len(col_names)
    data = np.empty( (len(columns[0]), ncol) )
    for i,col in enumerate(columns):
        data[:,i] = col

    width = precision+8
    # Note: The first one is 1 shorter to allow space for the initial #.
    header = ("#" + "{:^%d}"%(width-1) + " {:^%d}"%(width) * (ncol-1) + "\n").format(*col_names)
    fmt = '%%%d.%de'%(width,precision)

    if params is not None:
        s = '## %r\n'%(params)
        fid.write(s.encode())
    fid.write(header.encode())
    np.savetxt(fid, data, fmt=fmt)

def gen_write_fits(fits, col_names, columns, params, ext=None):
    """Write some columns to a new FITS extension with the given column names.

    :param fits:        An open fitsio.FITS handler. E.g. fits = fitsio.FITS(file_name, 'rw')
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    :param params:      A dict of extra parameters to write in the FITS header.
    :param ext:         An optional name for the extension to write. (default: None)
    """
    data = np.empty(len(columns[0]), dtype=[ (name,'f8') for name in col_names ])
    for (name, col) in zip(col_names, columns):
        data[name] = col
    fits.write(data, header=params, extname=ext)

def gen_write_hdf(hdf, col_names, columns, params, group=None):
    """Write some columns to a new FITS extension with the given column names.

    :param hdf:         An open h5py.File handle. E.g. hdf = h5py.File(file_name, 'w')
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    :param params:      A dict of extra parameters to write in the HDF attributes.
    :param group:       An optional name for the group to write in. (default: None,
                        and the data is written to the root group).
    """
    if group is not None:
        hdf = hdf.create_group(group)
    if params is not None:
        hdf.attrs.update(params)
    for (name, col) in zip(col_names, columns):
        hdf.create_dataset(name, data=col)

def gen_multi_write(file_name, col_names, group_names, columns,
                    params=None, precision=4, file_type=None, logger=None):
    """Write multiple groups of columns to an output file with the given column names.

    We do this basic functionality a lot, so put the code to do it in one place.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns (same for each group).
    :param group_names: A list of group names.  These become the hdu names in FITS format,
                        names for blocks of rows in ASCII format, and group names in HDF.
    :param columns:     A list of groups, each of which is a list of numpy arrays with the data to
                        write.
    :param params:      A dict of extra parameters to write at the top of the output file (for
                        ASCII output) or in the header (for FITS output).  (default: None)
    :param precision:   Output precision for ASCII. (default: 4)
    :param file_type:   Which kind of file to write to. (default: determine from the file_name
                        extension)
    :param logger:      If desired, a logger object for logging. (default: None)
    """
    if len(group_names) != len(columns):
        raise ValueError("group_names and columns are not the same length.")
    for i, col_group in enumerate(columns):
        if len(col_names) != len(col_group):
            raise ValueError("col_names and columns[%d] are not the same length."%i)
    if len(group_names) == 0:
        raise ValueError("len(group_names) == 0")
    if len(col_names) == 0:
        raise ValueError("len(col_names) == 0")
    for group in columns:
        for col in group:
            if col.shape != columns[0][0].shape:
                raise ValueError("columns are not all the same shape")

    columns = [ [ col.flatten() for col in group ] for group in columns ]

    ensure_dir(file_name)

    # Figure out which file type to use.
    file_type = parse_file_type(file_type, file_name, output=True, logger=logger)

    if file_type == 'FITS':
        try:
            import fitsio  # noqa: F401
        except ImportError:
            if logger:
                logger.error("Unable to import fitsio.  Cannot write to %s"%file_name)
            raise
        with fitsio.FITS(file_name, 'rw', clobber=True) as fits:
            for name, cols in zip(group_names, columns):
                gen_write_fits(fits, col_names, cols, params, ext=name)
    elif file_type == 'ASCII':
        with open(file_name, 'wb') as fid:
            for name, cols in zip(group_names, columns):
                s = '## %s: %d\n'%(name, len(cols[0]))
                fid.write(s.encode())
                gen_write_ascii(fid, col_names, cols, params, precision=precision)
    elif file_type == "HDF":
        try:
            import h5py
        except ImportError:
            if logger:
                logger.error("Unable to import h5py.  Cannot write to %s"%file_name)
            raise
        with h5py.File(file_name, 'w') as hdf:
            for name, cols in zip(group_names, columns):
                gen_write_hdf(hdf, col_names, cols, params, group=name)
    else:
        raise ValueError("Invalid file_type %s"%file_type)

def gen_read(file_name, file_type=None, logger=None):
    """Read some columns from an input file.

    We do this basic functionality a lot, so put the code to do it in one place.

    .. note::

        The input file is expected to have been written by TreeCorr using the
        `gen_write` function, so we don't have a lot of flexibility in the input structure.

    :param file_name:   The name of the file to read.
    :param file_type:   Which kind of file to read. (default: determine from the file_name
                        extension)
    :param logger:      If desired, a logger object for logging. (default: None)

    :returns: (data, params), a numpy ndarray with named columns, and a dict of extra parameters.
    """
    # Figure out which file type to use.
    file_type = parse_file_type(file_type, file_name, output=True, logger=logger)

    if file_type == 'FITS':
        try:
            import fitsio
        except ImportError:
            if logger:
                logger.error("Unable to import fitsio.  Cannot read %s"%file_name)
            raise
        with fitsio.FITS(file_name) as fits:
            return gen_read_fits(fits, ext=1)
    elif file_type == 'HDF':
        try:
            import h5py
        except ImportError:
            if logger:
                logger.error("Unable to import h5py.  Cannot read %s"%file_name)
            raise
        with h5py.File(file_name, 'r') as hdf:
            return gen_read_hdf(hdf)

    elif file_type == 'ASCII':
        with open(file_name) as fid:
            return gen_read_ascii(fid)
    else:
        raise ValueError("Invalid file_type %s"%file_type)

def gen_read_ascii(fid, max_rows=None):
    """Read some columns from an input ASCII file.

    :param fid:         An open file handler in binary mode. E.g. fid = open(file_name, 'rb')
    :param max_rows:    How many rows to read. (default: None, which means go to the end)

    :returns: (data, params), a numpy ndarray with named columns, and a dict of extra parameters.
    """
    header = next(fid)
    params = {}
    if header[1] == '#':
        assert header[0] == '#'
        params = eval(header[2:].strip())
        header = next(fid)
    names = header[1:].split()
    data = np.genfromtxt(fid, names=names, max_rows=max_rows)
    return data, params

def gen_read_fits(fits, ext=1):
    """Read some columns from an input FITS file.

    :param fits:        An open fitsio.FITS handler. E.g. fits = fitsio.FITS(file_name)
    :param ext:         An optional name or number for the extension to read. (default: 1)

    :returns: (data, params), a numpy ndarray with named columns, and a dict of extra parameters.
    """
    data = fits[ext].read()
    params = fits[ext].read_header()
    return data, params

def gen_read_hdf(hdf, group="/"):
    """Read the columns from an input HDF5 file.

    :param hdf:         An open h5py.File handler. E.g. hdf = h5py.File(file_name, "r")
    :param group:       An optional name group to read. (default: "/", meaning the file root)

    :returns: (data, params), a numpy ndarray with named columns, and a dict of extra parameters.
    """
    try:
        hdf = hdf[group]
    except KeyError:
        raise OSError("Group name %s not found in HDF5 file."%(group))
    params = dict(hdf.attrs)

    # This does not actually load the column
    col_vals =  list(hdf.values())
    col_names = list(hdf.keys())

    ncol = len(col_names)
    sz = col_vals[0].size
    dtype=[(name, col.dtype) for (name, col) in zip(col_names, col_vals)]
    data = np.empty(sz, dtype=dtype)

    # Now we actually read everything
    for (name, col) in zip(col_names, col_vals):
        data[name] = col[:]

    return data, params


def gen_multi_read(file_name, group_names, file_type=None, logger=None):
    """Read some columns from an input file.

    We do this basic functionality a lot, so put the code to do it in one place.

    .. note::

        The input file is expected to have been written by TreeCorr using the
        `gen_write` function, so we don't have a lot of flexibility in the input structure.

    :param file_name:   The name of the file to read.
    :param group_names: A list of group names.  These are the hdu names in FITS format or
                        names for blocks of rows in ASCII format.
    :param file_type:   Which kind of file to read. (default: determine from the file_name
                        extension)
    :param logger:      If desired, a logger object for logging. (default: None)

    :returns: list of (data, params) pairs, one for each group
    """
    # Figure out which file type to use.
    file_type = parse_file_type(file_type, file_name, output=True, logger=logger)

    if file_type == 'FITS':
        try:
            import fitsio
        except ImportError:
            if logger:
                logger.error("Unable to import fitsio.  Cannot read %s"%file_name)
            raise
        out = []
        with fitsio.FITS(file_name) as fits:
            for name in group_names:
                data, params = gen_read_fits(fits, ext=name)
                out.append( (data,params) )
        return out
    elif file_type == 'ASCII':
        out = []
        with open(file_name) as fid:
            for name in group_names:
                group_line = next(fid)
                name1, max_rows = group_line[2:].split()
                name1 = name1[:-1]  # strip off final :
                if name1 != name:
                    raise OSError("Mismatch in group names. Expected %s, found %s"%(name,name1))
                max_rows = int(max_rows)
                data, params = gen_read_ascii(fid, max_rows=max_rows)
                out.append( (data,params) )
        return out
    elif file_type == "HDF":
        try:
            import h5py
        except ImportError:
            if logger:
                logger.error("Unable to import h5py.  Cannot read %s"%file_name)
            raise
        out = []
        with h5py.File(file_name, 'r') as hdf:
            for name in group_names:
                data, params = gen_read_hdf(hdf, group=name)
                out.append( (data,params) )
        return out
    else:
        raise ValueError("Invalid file_type %s"%file_type)

class LRU_Cache(object):
    """ Simplified Least Recently Used Cache.
    Mostly stolen from http://code.activestate.com/recipes/577970-simplified-lru-cache/,
    but added a method for dynamic resizing.  The least recently used cached item is
    overwritten on a cache miss.

    :param user_function:  A python function to cache.
    :param maxsize:        Maximum number of inputs to cache.  [Default: 1024]

    Usage
    -----
    >>> def slow_function(*args) # A slow-to-evaluate python function
    >>>    ...
    >>>
    >>> v1 = slow_function(*k1)  # Calling function is slow
    >>> v1 = slow_function(*k1)  # Calling again with same args is still slow
    >>> cache = galsim.utilities.LRU_Cache(slow_function)
    >>> v1 = cache(*k1)  # Returns slow_function(*k1), slowly the first time
    >>> v1 = cache(*k1)  # Returns slow_function(*k1) again, but fast this time.

    Methods
    -------
    >>> cache.resize(maxsize) # Resize the cache, either upwards or downwards.  Upwards resizing
                              # is non-destructive.  Downwards resizing will remove the least
                              # recently used items first.
    """
    def __init__(self, user_function, maxsize=1024):
        # Link layout:     [PREV, NEXT, KEY, RESULT]
        self.root = [None, None, None, None]
        self.user_function = user_function
        self.cache = {}

        last = self.root
        for i in range(maxsize):
            key = object()
            self.cache[key] = last[1] = last = [last, self.root, key, None]
        self.root[0] = last
        self.count = 0

    def __call__(self, *key, **kwargs):
        link = self.cache.get(key)
        if link is not None:
            # Cache hit: move link to last position
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = self.root[0]
            last[1] = self.root[0] = link
            link[0] = last
            link[1] = self.root
            return result
        # Cache miss: evaluate and insert new key/value at root, then increment root
        #             so that just-evaluated value is in last position.
        result = self.user_function(*key, **kwargs)
        self.root[2] = key
        self.root[3] = result
        oldroot = self.root
        self.root = self.root[1]
        oldkey = self.root[2]
        self.root[2] = None
        self.root[3] = None
        self.cache[key] = oldroot
        del self.cache[oldkey]
        if self.count < self.size: self.count += 1
        return result

    def values(self):
        """Lists all items stored in the cache"""
        return list([v[3] for v in self.cache.values() if v[3] is not None])

    @property
    def last_value(self):
        """Return the most recently used value"""
        return self.root[0][3]

    def resize(self, maxsize):
        """ Resize the cache.  Increasing the size of the cache is non-destructive, i.e.,
        previously cached inputs remain in the cache.  Decreasing the size of the cache will
        necessarily remove items from the cache if the cache is already filled.  Items are removed
        in least recently used order.

        :param maxsize: The new maximum number of inputs to cache.
        """
        oldsize = len(self.cache)
        if maxsize == oldsize:
            return
        else:
            if maxsize < 0:
                raise ValueError("Invalid maxsize")
            elif maxsize < oldsize:
                for i in range(oldsize - maxsize):
                    # Delete root.next
                    current_next_link = self.root[1]
                    new_next_link = self.root[1] = self.root[1][1]
                    new_next_link[0] = self.root
                    del self.cache[current_next_link[2]]
                self.count = min(self.count, maxsize)
            else: # maxsize > oldsize:
                for i in range(maxsize - oldsize):
                    # Insert between root and root.next
                    key = object()
                    self.cache[key] = link = [self.root, self.root[1], key, None]
                    self.root[1][0] = link
                    self.root[1] = link

    def clear(self):
        """ Clear all items from the cache.
        """
        maxsize = len(self.cache)
        self.cache.clear()
        last = self.root
        for i in range(maxsize):
            last[3] = None  # Sever pointer to any existing result.
            key = object()
            self.cache[key] = last[1] = last = [last, self.root, key, None]
        self.root[0] = last
        self.count = 0

    @property
    def size(self):
        return len(self.cache)

def double_ptr(x):
    """
    Cast x as a double* to pass to library C functions

    :param x:   A numpy array assumed to have dtype = float.

    :returns:   A version of the array that can be passed to cffi C functions.
    """
    if x is None:
        return _ffi.cast('double*', 0)
    else:
        # This fails if x is read_only
        #return _ffi.cast('double*', _ffi.from_buffer(x))
        # This works, presumably by ignoring the numpy read_only flag.  Although, I think it's ok.
        return _ffi.cast('double*', x.ctypes.data)

def long_ptr(x):
    """
    Cast x as a long* to pass to library C functions

    :param x:   A numpy array assumed to have dtype = int.

    :returns:   A version of the array that can be passed to cffi C functions.
    """
    if x is None:  # pragma: no cover   (I don't ever have x=None for this one.)
        return _ffi.cast('long*', 0)
    else:
        return _ffi.cast('long*', x.ctypes.data)

def parse_metric(metric, coords, coords2=None, coords3=None):
    """
    Convert a string metric into the corresponding enum to pass to the C code.
    """
    if coords2 is None:
        auto = True
    else:
        auto = False
        # Special Rlens doesn't care about the distance to the sources, so spherical is fine
        # for cat2, cat3 in that case.
        if metric == 'Rlens':
            if coords2 == 'spherical': coords2 = '3d'
            if coords3 == 'spherical': coords3 = '3d'

        if metric == 'Arc':
            # If all coords are 3d, then leave it 3d, but if any are spherical,
            # then convert to spherical.
            if all([c in [None, '3d'] for c in [coords, coords2, coords3]]):
                # Leave coords as '3d'
                pass
            elif any([c not in [None, 'spherical', '3d'] for c in [coords, coords2, coords3]]):
                raise ValueError("Arc metric is only valid for catalogs with spherical positions.")
            elif any([c == 'spherical' for c in [coords, coords2, coords3]]):  # pragma: no branch
                # Switch to spherical
                coords = 'spherical'
            else:  # pragma: no cover
                # This is impossible now, but here in case we add additional coordinates.
                raise ValueError("Cannot correlate catalogs with different coordinate systems.")
        else:
            if ( (coords2 != coords) or (coords3 is not None and coords3 != coords) ):
                raise ValueError("Cannot correlate catalogs with different coordinate systems.")

    if coords not in ['flat', 'spherical', '3d']:
        raise ValueError("Invalid coords %s"%coords)

    if metric not in ['Euclidean', 'Rperp', 'OldRperp', 'FisherRperp', 'Rlens', 'Arc', 'Periodic']:
        raise ValueError("Invalid metric %s"%metric)

    if metric in ['Rperp', 'OldRperp', 'FisherRperp'] and coords != '3d':
        raise ValueError("%s metric is only valid for catalogs with 3d positions."%metric)
    if metric == 'Rlens' and auto:
        raise ValueError("Rlens metric is only valid for cross correlations.")
    if metric == 'Rlens' and coords != '3d':
        raise ValueError("Rlens metric is only valid for catalogs with 3d positions.")
    if metric == 'Arc' and coords not in ['spherical', '3d']:
        raise ValueError("Arc metric is only valid for catalogs with spherical positions.")

    return coords, metric

def coord_enum(coords):
    """Return the C++-layer enum for the given string value of coords.
    """
    if coords == 'flat':
        return _lib.Flat
    elif coords == 'spherical':
        return _lib.Sphere
    elif coords == '3d':
        return _lib.ThreeD
    else:
        raise ValueError("Invalid coords %s"%coords)

def metric_enum(metric):
    """Return the C++-layer enum for the given string value of metric.
    """
    if metric == 'Euclidean':
        return _lib.Euclidean
    elif metric == 'Rperp':
        return metric_enum(Rperp_alias)
    elif metric == 'FisherRperp':
        return _lib.Rperp
    elif metric in ['OldRperp']:
        return _lib.OldRperp
    elif metric == 'Rlens':
        return _lib.Rlens
    elif metric == 'Arc':
        return _lib.Arc
    elif metric == 'Periodic':
        return _lib.Periodic
    else:
        raise ValueError("Invalid metric %s"%metric)

def parse_xyzsep(args, kwargs, _coords):
    """Parse the different options for passing a coordinate and separation.

    The allowed parameters are:

    1. If _coords == Flat:

        :param x:       The x coordinate of the location for which to count nearby points.
        :param y:       The y coordinate of the location for which to count nearby points.
        :param sep:     The separation distance

    2. If _coords == ThreeD:

    Either
        :param x:       The x coordinate of the location for which to count nearby points.
        :param y:       The y coordinate of the location for which to count nearby points.
        :param z:       The z coordinate of the location for which to count nearby points.
        :param sep:     The separation distance

    Or
        :param ra:      The right ascension of the location for which to count nearby points.
        :param dec:     The declination of the location for which to count nearby points.
        :param r:       The distance to the location for which to count nearby points.
        :param sep:     The separation distance

    3. If _coords == Sphere:

        :param ra:      The right ascension of the location for which to count nearby points.
        :param dec:     The declination of the location for which to count nearby points.
        :param sep:     The separation distance as an angle

    For all angle parameters (ra, dec, sep), this quantity may be a coord.Angle instance, or
    units maybe be provided as ra_units, dec_units or sep_units respectively.

    Finally, in cases where ra, dec are allowed, a coord.CelestialCoord instance may be
    provided as the first argument.

    :returns: The effective (x, y, z, sep) as a tuple.
    """
    radec = False
    if _coords == _lib.Flat:
        if len(args) == 0:
            if 'x' not in kwargs:
                raise TypeError("Missing required argument x")
            if 'y' not in kwargs:
                raise TypeError("Missing required argument y")
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            x = kwargs.pop('x')
            y = kwargs.pop('y')
            sep = kwargs.pop('sep')
        elif len(args) == 1:
            raise TypeError("x,y should be given as either args or kwargs, not mixed.")
        elif len(args) == 2:
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            x,y = args
            sep = kwargs.pop('sep')
        elif len(args) == 3:
            x,y,sep = args
        else:
            raise TypeError("Too many positional args")
        z = 0

    elif _coords == _lib.ThreeD:
        if len(args) == 0:
            if 'x' in kwargs:
                if 'y' not in kwargs:
                    raise TypeError("Missing required argument y")
                if 'z' not in kwargs:
                    raise TypeError("Missing required argument z")
                x = kwargs.pop('x')
                y = kwargs.pop('y')
                z = kwargs.pop('z')
            else:
                if 'ra' not in kwargs:
                    raise TypeError("Missing required argument ra")
                if 'dec' not in kwargs:
                    raise TypeError("Missing required argument dec")
                ra = kwargs.pop('ra')
                dec = kwargs.pop('dec')
                radec = True
                if 'r' not in kwargs:
                    raise TypeError("Missing required argument r")
                r = kwargs.pop('r')
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            sep = kwargs.pop('sep')
        elif len(args) == 1:
            if not isinstance(args[0], coord.CelestialCoord):
                raise TypeError("Invalid unnamed argument %r"%args[0])
            ra = args[0].ra
            dec = args[0].dec
            radec = True
            if 'r' not in kwargs:
                raise TypeError("Missing required argument r")
            r = kwargs.pop('r')
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            sep = kwargs.pop('sep')
        elif len(args) == 2:
            if isinstance(args[0], coord.CelestialCoord):
                ra = args[0].ra
                dec = args[0].dec
                radec = True
                r = args[1]
            else:
                ra, dec = args
                radec = True
                if 'r' not in kwargs:
                    raise TypeError("Missing required argument r")
                r = kwargs.pop('r')
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            sep = kwargs.pop('sep')
        elif len(args) == 3:
            if isinstance(args[0], coord.CelestialCoord):
                ra = args[0].ra
                dec = args[0].dec
                radec = True
                r = args[1]
                sep = args[2]
            elif isinstance(args[0], coord.Angle):
                ra, dec, r = args
                radec = True
                if 'sep' not in kwargs:
                    raise TypeError("Missing required argument sep")
                sep = kwargs.pop('sep')
            elif 'ra_units' in kwargs or 'dec_units' in kwargs:
                ra, dec, r = args
                radec = True
                if 'sep' not in kwargs:
                    raise TypeError("Missing required argument sep")
                sep = kwargs.pop('sep')
            else:
                x, y, z = args
                if 'sep' not in kwargs:
                    raise TypeError("Missing required argument sep")
                sep = kwargs.pop('sep')
        elif len(args) == 4:
            if isinstance(args[0], coord.Angle):
                ra, dec, r, sep = args
                radec = True
            elif 'ra_units' in kwargs or 'dec_units' in kwargs:
                ra, dec, r, sep = args
                radec = True
            else:
                x, y, z, sep = args
        else:
            raise TypeError("Too many positional args")

    else:  # Sphere
        if len(args) == 0:
            if 'ra' not in kwargs:
                raise TypeError("Missing required argument ra")
            if 'dec' not in kwargs:
                raise TypeError("Missing required argument dec")
            ra = kwargs.pop('ra')
            dec = kwargs.pop('dec')
            radec = True
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            sep = kwargs.pop('sep')
        elif len(args) == 1:
            if not isinstance(args[0], coord.CelestialCoord):
                raise TypeError("Invalid unnamed argument %r"%args[0])
            ra = args[0].ra
            dec = args[0].dec
            radec = True
            if 'sep' not in kwargs:
                raise TypeError("Missing required argument sep")
            sep = kwargs.pop('sep')
        elif len(args) == 2:
            if isinstance(args[0], coord.CelestialCoord):
                ra = args[0].ra
                dec = args[0].dec
                radec = True
                sep = args[1]
            else:
                ra, dec = args
                radec = True
                if 'sep' not in kwargs:
                    raise TypeError("Missing required argument sep")
                sep = kwargs.pop('sep')
        elif len(args) == 3:
            ra, dec, sep = args
            radec = True
        else:
            raise TypeError("Too many positional args")
        if not isinstance(sep, coord.Angle):
            if 'sep_units' not in kwargs:
                raise TypeError("Missing required argument sep_units")
            sep = sep * coord.AngleUnit.from_name(kwargs.pop('sep_units'))
        # We actually want the chord distance for this angle.
        sep = 2. * np.sin(sep/2.)

    if radec:
        if not isinstance(ra, coord.Angle):
            if 'ra_units' not in kwargs:
                raise TypeError("Missing required argument ra_units")
            ra = ra * coord.AngleUnit.from_name(kwargs.pop('ra_units'))
        if not isinstance(dec, coord.Angle):
            if 'dec_units' not in kwargs:
                raise TypeError("Missing required argument dec_units")
            dec = dec * coord.AngleUnit.from_name(kwargs.pop('dec_units'))
        x,y,z = coord.CelestialCoord(ra, dec).get_xyz()
        if _coords == _lib.ThreeD:
            x *= r
            y *= r
            z *= r
    if len(kwargs) > 0:
        raise TypeError("Invalid kwargs: %s"%(kwargs))

    return float(x), float(y), float(z), float(sep)

class lazy_property(object):
    """
    This decorator will act similarly to @property, but will be efficient for multiple access
    to values that require some significant calculation.

    It works by replacing the attribute with the computed value, so after the first access,
    the property (an attribute of the class) is superseded by the new attribute of the instance.

    Usage::

        @lazy_property
        def slow_function_to_be_used_as_a_property(self):
            x =  ...  # Some slow calculation.
            return x

    Base on an answer from http://stackoverflow.com/a/6849299
    This implementation taken from GalSim utilities.py
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value

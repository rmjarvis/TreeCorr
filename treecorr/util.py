# Copyright (c) 2003-2015 by Mike Jarvis
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

import treecorr
import numpy
import os

def ensure_dir(target):
    d = os.path.dirname(target)
    if d != '':
        if not os.path.exists(d):
            os.makedirs(d)


def gen_write(file_name, col_names, columns, prec=4, file_type=None, logger=None):
    """Write some columns to an output file with the given column names.

    We do this basic functionality a lot, so put the code to do it in one place.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    :param prec:        Output precision for ASCII. (default: 4)
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

    # Figure out which file type the catalog is
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        else:
            file_type = 'ASCII'
        if logger:
            logger.info("file_type assumed to be %s from the file name.",file_type)

    if file_type == 'FITS':
        gen_write_fits(file_name, col_names, columns)
    elif file_type == 'ASCII':
        gen_write_ascii(file_name, col_names, columns, prec=prec)
    else:
        raise ValueError("Invalid file_type %s"%file_type)


def gen_write_ascii(file_name, col_names, columns, prec=4):
    """Write some columns to an output ASCII file with the given column names.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.  These will be written
                        in a header comment line at the top of the output file.
    :param columns:     A list of numpy arrays with the data to write.
    :param prec:        Output precision for ASCII. (default: 4)
    """
    ncol = len(col_names)
    data = numpy.empty( (len(columns[0]), ncol) )
    for i,col in enumerate(columns):
        data[:,i] = col

    width = prec+8
    # Note: python 2.6 needs the numbers, so can't just do "{:^%d}"*ncol
    # Also, I have the first one be 1 shorter to allow space for the initial #.
    header_form = "{0:^%d}"%(width-1)
    for i in range(1,ncol):
        header_form += " {%d:^%d}"%(i,width)
    header = header_form.format(*col_names)
    fmt = '%%%d.%de'%(width,prec)
    ensure_dir(file_name)
    with open(file_name, 'wb') as fid:
        h = '#' + header + '\n'
        fid.write(h.encode())
        numpy.savetxt(fid, data, fmt=fmt)


def gen_write_fits(file_name, col_names, columns):
    """Write some columns to an output FITS file with the given column names.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    """
    import fitsio
    ensure_dir(file_name)
    data = numpy.empty(len(columns[0]), dtype=[ (name,'f8') for name in col_names ])
    for (name, col) in zip(col_names, columns):
        data[name] = col
    fitsio.write(file_name, data, clobber=True)


def gen_read(file_name, file_type=None, logger=None):
    """Read some columns from an input file.

    We do this basic functionality a lot, so put the code to do it in one place.
    Note that the input file is expected to have been written by TreeCorr using the 
    gen_write function, so we don't have a lot of flexibility in the input structure.

    :param file_name:   The name of the file to read.
    :param file_type:   Which kind of file to write to. (default: determine from the file_name
                        extension)
    :param logger:      If desired, a logger object for logging. (default: None)

    :returns: a numpy ndarray with named columns
    """
    # Figure out which file type the catalog is
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        else:
            file_type = 'ASCII'
        if logger:
            logger.info("file_type assumed to be %s from the file name.",file_type)

    if file_type == 'FITS':
        import fitsio
        data = fitsio.read(file_name)
    elif file_type == 'ASCII':
        data = numpy.genfromtxt(file_name, names=True)
    else:
        raise ValueError("Invalid file_type %s"%file_type)

    return data


class LRU_Cache:
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
        self.root = root = [None, None, None, None]
        self.user_function = user_function
        self.cache = cache = {}

        last = root
        for i in range(maxsize):
            key = object()
            cache[key] = last[1] = last = [last, root, key, None]
        root[0] = last

    def __call__(self, *key):
        cache = self.cache
        root = self.root
        link = cache.get(key)
        if link is not None:
            # Cache hit: move link to last position
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = root[0]
            last[1] = root[0] = link
            link[0] = last
            link[1] = root
            return result
        # Cache miss: evaluate and insert new key/value at root, then increment root
        #             so that just-evaluated value is in last position.
        result = self.user_function(*key)
        root[2] = key
        root[3] = result
        oldroot = root
        root = self.root = root[1]
        root[2], oldkey = None, root[2]
        root[3], oldvalue = None, root[3]
        del cache[oldkey]
        cache[key] = oldroot
        return result

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
            root = self.root
            cache = self.cache
            if maxsize < oldsize:
                for i in range(oldsize - maxsize):
                    # Delete root.next
                    current_next_link = root[1]
                    new_next_link = root[1] = root[1][1]
                    new_next_link[0] = root
                    del cache[current_next_link[2]]
            elif maxsize > oldsize:
                for i in range(maxsize - oldsize):
                    # Insert between root and root.next
                    key = object()
                    cache[key] = link = [root, root[1], key, None]
                    root[1][0] = link
                    root[1] = link
            else:
                raise ValueError("Invalid maxsize: {0:}".format(maxsize))

def double_ptr(x):
    """
    Cast x as a double* to pass to library C functions

    :param x:   A numpy array assumed to have dtype = float.

    :returns:   A version of the array that can be passed to cffi C functions.
    """
    # This fails if x is read_only
    #return treecorr._ffi.cast('double*', treecorr._ffi.from_buffer(x))
    # This works, presumably by ignoring the numpy read_only flag.  Although, I think it's ok.
    return treecorr._ffi.cast('double*', x.ctypes.data)

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

        if ( (coords2 != coords) or (coords3 is not None and coords3 != coords) ):
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

    if coords == 'flat':
        c = treecorr._lib.Flat
    elif coords == 'spherical':
        c = treecorr._lib.Sphere
    else:
        c = treecorr._lib.ThreeD

    if metric == 'Euclidean':
        m = treecorr._lib.Euclidean
    elif metric == 'Rperp':
        if coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")
        m = treecorr._lib.Perp
    elif metric == 'Rlens':
        if auto:
            raise ValueError("Rlens metric is only valid for cross correlations.")
        if coords != '3d':
            raise ValueError("Rlens metric is only valid for catalogs with 3d positions.")
        m = treecorr._lib.Lens
    elif metric == 'Arc':
        if coords != 'spherical':
            raise ValueError("Arc metric is only valid for catalogs with spherical positions.")
        m = treecorr._lib.Arc
    else:
        raise ValueError("Invalid metric %s"%metric)

    return c, m


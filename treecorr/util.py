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

import treecorr
import numpy as np
import os
import warnings
import coord

def ensure_dir(target):
    d = os.path.dirname(target)
    if d != '':
        if not os.path.exists(d):
            os.makedirs(d)

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

    # Figure out which file type the catalog is
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        else:
            file_type = 'ASCII'
        if logger:  # pragma: no branch  (We always provide a logger.)
            logger.info("file_type assumed to be %s from the file name.",file_type)

    if file_type.upper() == 'FITS':
        try:
            import fitsio
        except ImportError:
            logger.error("Unable to import fitsio.  Cannot write to %s"%file_name)
            raise
        gen_write_fits(file_name, col_names, columns, params)
    elif file_type.upper() == 'ASCII':
        gen_write_ascii(file_name, col_names, columns, params, precision=precision)
    else:
        raise ValueError("Invalid file_type %s"%file_type)


def gen_write_ascii(file_name, col_names, columns, params, precision=4):
    """Write some columns to an output ASCII file with the given column names.

    :param file_name:   The name of the file to write to.
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
    # Note: python 2.6 needs the numbers, so can't just do "{:^%d}"*ncol
    # Also, I have the first one be 1 shorter to allow space for the initial #.
    header_form = "{0:^%d}"%(width-1)
    for i in range(1,ncol):
        header_form += " {%d:^%d}"%(i,width)
    header = header_form.format(*col_names)
    fmt = '%%%d.%de'%(width,precision)
    ensure_dir(file_name)
    with open(file_name, 'wb') as fid:
        if params is not None:
            s = '## %r\n'%(params)
            fid.write(s.encode())
        h = '#' + header + '\n'
        fid.write(h.encode())
        np.savetxt(fid, data, fmt=fmt)


def gen_write_fits(file_name, col_names, columns, params):
    """Write some columns to an output FITS file with the given column names.

    :param file_name:   The name of the file to write to.
    :param col_names:   A list of columns names for the given columns.
    :param columns:     A list of numpy arrays with the data to write.
    :param params:      A dict of extra parameters to write in the FITS header.
    """
    import fitsio
    ensure_dir(file_name)
    data = np.empty(len(columns[0]), dtype=[ (name,'f8') for name in col_names ])
    for (name, col) in zip(col_names, columns):
        data[name] = col
    fitsio.write(file_name, data, header=params, clobber=True)


def gen_read(file_name, file_type=None, logger=None):
    """Read some columns from an input file.

    We do this basic functionality a lot, so put the code to do it in one place.
    Note that the input file is expected to have been written by TreeCorr using the
    gen_write function, so we don't have a lot of flexibility in the input structure.

    :param file_name:   The name of the file to read.
    :param file_type:   Which kind of file to read. (default: determine from the file_name
                        extension)
    :param logger:      If desired, a logger object for logging. (default: None)

    :returns: (data, params), a numpy ndarray with named columns, and a dict of extra parameters.
    """
    # Figure out which file type the catalog is
    if file_type is None:
        import os
        name, ext = os.path.splitext(file_name)
        if ext.lower().startswith('.fit'):
            file_type = 'FITS'
        else:
            file_type = 'ASCII'
        if logger:  # pragma: no branch  (We always provide a logger.)
            logger.info("file_type assumed to be %s from the file name.",file_type)

    if file_type.upper() == 'FITS':
        try:
            import fitsio
        except ImportError:
            logger.error("Unable to import fitsio.  Cannot read %s"%file_name)
            raise
        data = fitsio.read(file_name)
        params = fitsio.read_header(file_name, 1)
    elif file_type.upper() == 'ASCII':
        with open(file_name) as fid:
            header = fid.readline()
            params = {}
            skip = 0
            if header[1] == '#':  # pragma: no branch  (All our files have this.)
                assert header[0] == '#'
                params = eval(header[2:].strip())
                header = fid.readline()
                skip = 1
        data = np.genfromtxt(file_name, names=True, skip_header=skip)
    else:
        raise ValueError("Invalid file_type %s"%file_type)

    return data, params


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
        self.root[2], oldkey = None, self.root[2]
        self.root[3], oldvalue = None, self.root[3]
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
        return treecorr._ffi.cast('double*', 0)
    else:
        # This fails if x is read_only
        #return treecorr._ffi.cast('double*', treecorr._ffi.from_buffer(x))
        # This works, presumably by ignoring the numpy read_only flag.  Although, I think it's ok.
        return treecorr._ffi.cast('double*', x.ctypes.data)

def long_ptr(x):
    """
    Cast x as a long* to pass to library C functions

    :param x:   A numpy array assumed to have dtype = int.

    :returns:   A version of the array that can be passed to cffi C functions.
    """
    if x is None:  # pragma: no cover   (I don't ever have x=None for this one.)
        return treecorr._ffi.cast('long*', 0)
    else:
        return treecorr._ffi.cast('long*', x.ctypes.data)

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
            if coords not in ['spherical', '3d']:
                raise ValueError("Arc metric is only valid for catalogs with spherical positions.")
            # If all coords are 3d, then leave it 3d, but if any are spherical,
            # then convert to spherical.
            if all([c in [None, '3d'] for c in [coords, coords2, coords3]]):
                # Leave coords as '3d'
                pass
            elif any([c not in [None, 'spherical', '3d'] for c in [coords, coords2, coords3]]):
                raise ValueError("Arc metric is only valid for catalogs with spherical positions.")
            elif any([c == 'spherical' for c in [coords, coords2, coords3]]):
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

    return coords, metric

def coord_enum(coords):
    """Return the C++-layer enum for the given string value of coords.
    """
    if coords == 'flat':
        return treecorr._lib.Flat
    elif coords == 'spherical':
        return treecorr._lib.Sphere
    elif coords == '3d':
        return treecorr._lib.ThreeD
    else:
        raise ValueError("Invalid coords %s"%coords)

def metric_enum(metric):
    """Return the C++-layer enum for the given string value of metric.
    """
    if metric == 'Euclidean':
        return treecorr._lib.Euclidean
    elif metric == 'Rperp':
        return metric_enum(treecorr.Rperp_alias)
    elif metric == 'FisherRperp':
        return treecorr._lib.Rperp
    elif metric in ['OldRperp']:
        return treecorr._lib.OldRperp
    elif metric == 'Rlens':
        return treecorr._lib.Rlens
    elif metric == 'Arc':
        return treecorr._lib.Arc
    elif metric == 'Periodic':
        return treecorr._lib.Periodic
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
    if _coords == treecorr._lib.Flat:
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

    elif _coords == treecorr._lib.ThreeD:
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
        if _coords == treecorr._lib.ThreeD:
            x *= r
            y *= r
            z *= r
    if len(kwargs) > 0:
        raise TypeError("Invalid kwargs: %s"%(kwargs))

    return float(x), float(y), float(z), float(sep)

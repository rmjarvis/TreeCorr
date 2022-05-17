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
import functools
import inspect
import warnings

from . import _lib, _ffi, Rperp_alias
from .writer import AsciiWriter, FitsWriter, HdfWriter
from .reader import AsciiReader, FitsReader, HdfReader

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

def make_writer(file_name, precision=4, file_type=None, logger=None):
    """Factory function to make a writer instance of the correct type.
    """
    # Figure out which file type to use.
    file_type = parse_file_type(file_type, file_name, output=True, logger=logger)
    if file_type == 'FITS':
        writer = FitsWriter(file_name, logger=logger)
    elif file_type == 'HDF':
        writer = HdfWriter(file_name, logger=logger)
    elif file_type == 'ASCII':
        writer = AsciiWriter(file_name, precision=precision, logger=logger)
    else:
        raise ValueError("Invalid file_type %s"%file_type)
    return writer

def make_reader(file_name, file_type=None, logger=None):
    """Factory function to make a writer instance of the correct type.
    """
    # Figure out which file type to use.
    file_type = parse_file_type(file_type, file_name, output=False, logger=logger)

    if file_type == 'FITS':
        reader = FitsReader(file_name, logger=logger)
    elif file_type == 'HDF':
        reader = HdfReader(file_name, logger=logger)
    elif file_type == 'ASCII':
        reader = AsciiReader(file_name, logger=logger)
    else:
        raise ValueError("Invalid file_type %s"%file_type)
    return reader

class LRU_Cache(object):
    """ Simplified Least Recently Used Cache.
    Mostly stolen from http://code.activestate.com/recipes/577970-simplified-lru-cache/,
    but added a method for dynamic resizing.  The least recently used cached item is
    overwritten on a cache miss.

    Note: This has additional functionality beyond what functools.lru_cache provides.
          1. The ability to resize the maxsize non-destructively.
          2. The key is only on the args, not kwargs, so a logger can be provided as a kwarg
             without triggering a cache miss.

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



def depr_pos_kwargs(fn):
    """
    This decorator will allow the old API where keywords are allowed as positional variables,
    but it will give a deprecation warning about it.

    @depr_pos_kwargs
    def func_with_kwargs(a, *, b=3, c=4):
        ...

    # Expected usage:
    func_with_kwargs(1, b=5, c=9)

    # This works, but gives a deprecation warning
    func_with_kwargs(1, 5, 9)
    """
    # Note: this is inspired by the legacy_api_wrap decorator by flying-sheep, which does something
    # similar.
    # https://github.com/flying-sheep/legacy-api-wrap/blob/master/legacy_api_wrap.py
    # However, it was reimplemented from scratch my MJ.

    params = inspect.signature(fn).parameters
    nparams = len(params)
    npos = np.sum([p.kind in [p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD] for p in params.values()])
    assert nparams > npos  # Otherwise developer probably forgot to add the * to the signature!

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) > npos:
            # Make sure providing too many params is still a TypeError.
            if len(args) > nparams:
                raise TypeError("{} takes at most {} arguments but {} were given.".format(
                        fn.__name__, nparams, len(args)))

            # Which names need to turn into kwargs?
            kw_names = list(params.keys())[npos:len(args)]

            # Warn about deprecated syntax
            warnings.warn(
                "Use of keyword-only arguments as positional arguments is deprecated in "+
                "the function " + fn.__name__ + ". " +
                "The following parameters now require an explicit keyword name: "+
                str(kw_names), FutureWarning)

            # But make it work.
            for a, n in zip(args[npos:], kw_names):
                kwargs[n] = a
            args = args[:npos]

        return fn(*args, **kwargs)

    return wrapper

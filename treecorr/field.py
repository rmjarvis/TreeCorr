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
.. module:: field
"""

import numpy as np
import weakref
import treecorr

def _parse_split_method(split_method):
    if split_method == 'middle': return 0
    elif split_method == 'median': return 1
    elif split_method == 'mean': return 2
    else: return 3  # random


class Field(object):
    """A Field in TreeCorr is the object that stores the tree structure we use for efficient
    calculation of the correlation functions.

    The root "cell" in the tree has information about the whole field, including the total
    number of points, the total weight, the mean position, the size (by which we mean the
    maximum distance of any point from the mean position), and possibly more information depending
    on which kind of field we have.

    It also points to two sub-cells which each describe about half the points.  These are commonly
    referred to as "daughter cells".  They in turn point to two more cells each, and so on until
    we get to cells that are considered "small enough" according to the **min_size** parameter given
    in the constructor.  These lowest level cells are referred to as "leaves".

    Technically, a Field doesn't have just one of these trees.  To make parallel computation
    more efficient, we actually skip the first few layers of the tree as described above and
    store a list of root cells.  The three parameters that determine how many of these there
    will be are **max_size**, **min_top**, and **max_top**:

        - **max_size** sets the maximum size cell that we want to make sure we have in the trees,
          so the root cells will be at least this large.  The default is None, which means
          we care about all sizes, so there may be only one root cell (but typically more
          because of **min_top**).
        - **min_top** sets the minimum number of initial levels to skip.  The default is 3,
          which means there will be at least 8 root cells (assuming ntot >= 8).
        - **max_top** sets the maximum number of initial levels to skip.  The default is 10,
          which means there could be up to 1024 root cells.

    Finally, the **split_method** parameter sets how the points in a cell should be divided
    when forming the two daughter cells.  The split is always done according to whichever
    dimension has the largest extent.  E.g. if max(\|x - meanx\|) is larger than max(\|y - meany\|)
    and (for 3d) max(\|z - meanz\|), then it will split according to the x values.  But then
    it may split in different ways according to **split_method**.  The allowed values are:

        - 'mean' means to divide the points at the average (mean) value of x, y or z.
        - 'median' means to divide the points at the median value of x, y, or z.
        - 'middle' means to divide the points at midpoint between the minimum and maximum values.
        - 'random' means to divide the points randomly somewhere between the 40th and 60th
          percentile locations in the sorted list.

    Field itself is an abstract base class for the specific types of field classes.
    As such, it cannot be constructed directly.  You should make one of the concrete subclasses:

        - NField describes a field of objects to be counted only.
        - KField describes a field of points sampling a scalar field (e.g. kappa in the
          weak lensing context).  In addition to the above values, cells keep track of
          the mean kappa value in the given region.
        - GField describes a field of points sampling a spinor field (e.g. gamma in the
          weak lensing context).  In addition to the above values, cells keep track of
          the mean (complex) gamma value in the given region.
    """
    def __init__(self):
        raise NotImplementedError("Field is an abstract base class.  It cannot be instantiated.")

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes.
        """
        return treecorr._lib.FieldGetNTopLevel(self.data, self._d, self._coords)

    @property
    def cat(self):
        """The catalog from which this field was constructed.

        It is stored as a weakref, so if the Catalog has already been garbage collected, this
        might be None.
        """
        # _cat is a weakref.  This gets back to a Catalog object.
        return self._cat()

    def count_near(self, *args, **kwargs):
        """Count how many points are near a given coordinate.

        Use the existing tree structure to count how many points are within some given separation
        of a target coordinate.

        There are several options for how to specify the reference coordinate, which depends
        on the type of coordinate system this field implements.

        1. For flat 2-dimensional coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            sep (float):     The separation distance

        2. For 3-dimensional Cartesian coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            z (float):       The z coordinate of the target location
            sep (float):     The separation distance

        3. For spherical coordinates:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A coord.CelestialCoord object in lieu of (ra, dec)
            sep (float or Angle):   The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float
            sep_units (str):        The units of sep if given as a float

        4. For spherical coordinates with distances:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A coord.CelestialCoord object in lieu of (ra, dec)
            r (float):              The distance to the target location
            sep (float):            The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float

        In all cases, for parameters that angles (ra, dec, sep for 'spherical'), you may either
        provide this quantity as a coord.Angle instance, or you may provide ra_units, dec_units
        or sep_units respectively to specify which angular units are providing.

        Finally, in cases where ra, dec are allowed, you may instead provide a
        coord.CelestialCoord instance as the first argument to specify both RA and Dec.
        """
        if self.min_size == 0:
            # If min_size = 0, then regular method is already exact.
            x,y,z,sep = treecorr.util.parse_xyzsep(args, kwargs, self._coords)
            return self._count_near(x, y, z, sep)
        else:
            # Otherwise, we need to expand the radius a bit and then check the actual radii
            # using the catalog values.  This is already done in get_near, so just do that
            # and take the len of the result.
            return len(self.get_near(*args, **kwargs))

    def _count_near(self, x, y, z, sep):
        # If self.min_size > 0, these results may be approximate, since the tree will have
        # grouped points within this separation together.
        return treecorr._lib.FieldCountNear(self.data, x, y, z, sep, self._d, self._coords)

    def get_near(self, *args, **kwargs):
        """Get the indices of points near a given coordinate.

        Use the existing tree structure to find the points that are within some given separation
        of a target coordinate.

        There are several options for how to specify the reference coordinate, which depends
        on the type of coordinate system this field implements.

        1. For flat 2-dimensional coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            sep (float):     The separation distance

        2. For 3-dimensional Cartesian coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            z (float):       The z coordinate of the target location
            sep (float):     The separation distance

        3. For spherical coordinates:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A coord.CelestialCoord object in lieu of (ra, dec)
            sep (float or Angle):   The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float
            sep_units (str):        The units of sep if given as a float

        4. For spherical coordinates with distances:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A coord.CelestialCoord object in lieu of (ra, dec)
            r (float):              The distance to the target location
            sep (float):            The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float

        In all cases, for parameters that angles (ra, dec, sep for 'spherical'), you may either
        provide this quantity as a coord.Angle instance, or you may provide ra_units, dec_units
        or sep_units respectively to specify which angular units are providing.

        Finally, in cases where ra, dec are allowed, you may instead provide a
        coord.CelestialCoord instance as the first argument to specify both RA and Dec.
        """
        x,y,z,sep = treecorr.util.parse_xyzsep(args, kwargs, self._coords)
        if self.min_size == 0:
            # If min_size == 0, then regular method is already exact.
            ind = self._get_near(x, y, z, sep)
        else:
            # Expand the radius by the minimum size of the cells.
            sep1 = sep + self.min_size
            # Get those indices
            ind = self._get_near(x, y, z, sep1)
            # Now check the actual radii of these points using the catalog x,y,z values.
            rsq = (self.cat.x[ind]-x)**2 + (self.cat.y[ind]-y)**2
            if self._coords != treecorr._lib.Flat:
                rsq += (self.cat.z[ind]-z)**2
            # Select the ones with r < sep
            near = rsq < sep**2
            ind = ind[near]
        # It comes back unsorted, so sort it.  (Not really required, but nicer output.)
        return np.sort(ind)

    def _get_near(self, x, y, z, sep):
        # If self.min_size > 0, these results may be approximate, since the tree will have
        # grouped points within this separation together.
        from treecorr.util import long_ptr as lp
        # First count how many there are, so we can allocate the array for the indices.
        n = self._count_near(x, y, z, sep)
        ind = np.empty(n, dtype=int)
        # Now fill the array with the indices of the nearby points.
        treecorr._lib.FieldGetNear(self.data, x, y, z, sep, self._d, self._coords, lp(ind), n)
        return ind


class NField(Field):
    """This class stores the positions and number of objects in a tree structure from which it is
    efficient to compute correlation functions.

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep). (default: 0)
    :param max_size:    The maximum radius cell required (usually max_sep). (default: None)
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
                        (default: 'mean')
    :param brute        Whether to force traversal to the leaves for this field. (default: False)
    :param min_top:     The minimum number of top layers to use when setting up the field.
                        (default: 3)
    :param max_top:     The maximum number of top layers to use when setting up the field.
                        (default: 10)
    :param coords       The kind of coordinate system to use. (default: cat.coords)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size=0, max_size=None, split_method='mean', brute=False,
                 min_top=3, max_top=10, coords=None, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building NField from cat %s',cat.name)
            else:
                logger.info('Building NField')

        self._cat = weakref.ref(cat)
        self.min_size = float(min_size) if not brute else 0.
        self.max_size = float(max_size) if max_size is not None else np.inf
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self._d = 1  # NData
        self.brute = bool(brute)
        self.min_top = int(min_top)
        self.max_top = int(max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildNField(dp(cat.x), dp(cat.y), dp(cat.z),
                                              dp(cat.w), dp(cat.wpos), cat.ntot,
                                              self.min_size, self.max_size, self._sm,
                                              self.brute, self.min_top, self.max_top, self._coords)
        if logger:
            logger.debug('Finished building NField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            # I don't get this, but sometimes it gets here when the fft.lock is already locked.
            # When that happens, this will freeze in a `with ffi._lock` line in the ffi api.py.
            # So, don't do that, and just accept the memory leak instead.
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyNField(self.data, self._coords)


class KField(Field):
    """This class stores the values of a scalar field (kappa in the weak lensing context) in a
    tree structure from which it is efficient to compute correlation functions.

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep). (default: 0)
    :param max_size:    The maximum radius cell required (usually max_sep). (default: None)
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
                        (default: 'mean')
    :param brute        Whether to force traversal to the leaves for this field. (default: False)
    :param min_top:     The minimum number of top layers to use when setting up the field.
                        (default: 3)
    :param max_top:     The maximum number of top layers to use when setting up the field.
                        (default: 10)
    :param coords       The kind of coordinate system to use. (default: cat.coords)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size=0, max_size=None, split_method='mean', brute=False,
                 min_top=3, max_top=10, coords=None, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building KField from cat %s',cat.name)
            else:
                logger.info('Building KField')

        self._cat = weakref.ref(cat)
        self.min_size = float(min_size) if not brute else 0.
        self.max_size = float(max_size) if max_size is not None else np.inf
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self._d = 2  # KData
        self.brute = bool(brute)
        self.min_top = int(min_top)
        self.max_top = int(max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildKField(dp(cat.x), dp(cat.y), dp(cat.z),
                                              dp(cat.k),
                                              dp(cat.w), dp(cat.wpos), cat.ntot,
                                              self.min_size, self.max_size, self._sm,
                                              self.brute, self.min_top, self.max_top, self._coords)
        if logger:
            logger.debug('Finished building KField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyKField(self.data, self._coords)


class GField(Field):
    """This class stores the values of a spinor field (gamma in the weak lensing context) in a
    tree structure from which it is efficient to compute correlation functions.

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep). (default: 0)
    :param max_size:    The maximum radius cell required (usually max_sep). (default: None)
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
                        (default: 'mean')
    :param brute        Whether to force traversal to the leaves for this field. (default: False)
    :param min_top:     The minimum number of top layers to use when setting up the field.
                        (default: 3)
    :param max_top:     The maximum number of top layers to use when setting up the field.
                        (default: 10)
    :param coords       The kind of coordinate system to use. (default: cat.coords)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size=0, max_size=None, split_method='mean', brute=False,
                 min_top=3, max_top=10, coords=None, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building GField from cat %s',cat.name)
            else:
                logger.info('Building GField')

        self._cat = weakref.ref(cat)
        self.min_size = float(min_size) if not brute else 0.
        self.max_size = float(max_size) if max_size is not None else np.inf
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self._d = 3  # GData
        self.brute = bool(brute)
        self.min_top = int(min_top)
        self.max_top = int(max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildGField(dp(cat.x), dp(cat.y), dp(cat.z),
                                              dp(cat.g1), dp(cat.g2),
                                              dp(cat.w), dp(cat.wpos), cat.ntot,
                                              self.min_size, self.max_size, self._sm,
                                              self.brute, self.min_top, self.max_top, self._coords)
        if logger:
            logger.debug('Finished building GField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyGField(self.data, self._coords)


class SimpleField(object):
    """A SimpleField is like a Field, but only stores the leaves as a list, skipping all the
    tree stuff.

    Again, this is an abstract base class, which cannot be instantiated.  You should
    make one of the concrete subclasses:

        - NSimpleField describes a field of objects to be counted only.
        - KSimpleField describes a field of points sampling a scalar field.
        - GSimpleField describes a field of points sampling a spinor field.
    """
    def __init__(self):
        raise NotImplementedError(
            "SimpleField is an abstract base class.  It cannot be instantiated.")


class NSimpleField(SimpleField):
    """This class stores the positions as a list, skipping all the tree stuff.

    An NSimpleField is typically created from a Catalog object using

        >>> nfield = cat.getNSimpleField()

    :param cat:         The catalog from which to make the field.
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building NSimpleField from cat %s',cat.name)
            else:
                logger.info('Building NSimpleField')
        self._d = 1  # NData
        self.coords = cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildNSimpleField(dp(cat.x), dp(cat.y), dp(cat.z),
                                                    dp(cat.w), dp(cat.wpos), cat.ntot,
                                                    self._coords)
        if logger:
            logger.debug('Finished building NSimpleField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyNSimpleField(self.data, self._coords)


class KSimpleField(SimpleField):
    """This class stores the kappa field as a list, skipping all the tree stuff.

    A KSimpleField is typically created from a Catalog object using

        >>> kfield = cat.getKSimpleField()

    :param cat:         The catalog from which to make the field.
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building KSimpleField from cat %s',cat.name)
            else:
                logger.info('Building KSimpleField')
        self._d = 2  # KData
        self.coords = cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildKSimpleField(dp(cat.x), dp(cat.y), dp(cat.z),
                                                    dp(cat.k),
                                                    dp(cat.w), dp(cat.wpos), cat.ntot,
                                                    self._coords)
        if logger:
            logger.debug('Finished building KSimpleField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyKSimpleField(self.data, self._coords)


class GSimpleField(SimpleField):
    """This class stores the shear field as a list, skipping all the tree stuff.

    A GSimpleField is typically created from a Catalog object using

        >>> gfield = cat.getGSimpleField()

    :param cat:         The catalog from which to make the field.
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building GSimpleField from cat %s',cat.name)
            else:
                logger.info('Building GSimpleField')
        self._d = 3  # GData
        self.coords = cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildGSimpleField(dp(cat.x), dp(cat.y), dp(cat.z),
                                                    dp(cat.g1), dp(cat.g2),
                                                    dp(cat.w), dp(cat.wpos), cat.ntot,
                                                    self._coords)
        if logger:
            logger.debug('Finished building KSimpleField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyGSimpleField(self.data, self._coords)


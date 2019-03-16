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
.. module:: field
"""

import treecorr
import numpy as np

def _parse_split_method(split_method):
    if split_method == 'middle': return 0
    elif split_method == 'median': return 1
    elif split_method == 'mean': return 2
    else: return 3  # random


class NField(object):
    """This class stores the positions in a tree structure from which it is efficient
    to compute the two-point correlation functions.

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep). (default: 0)
    :param max_size:    The maximum radius cell required (usually max_sep). (default: None)
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
                        (default: 'mean')
    :param max_top:     The maximum number of top layers to use when setting up the field.
                        (default: 10)
    :param coords       The kind of coordinate system to use. (default: cat.coords)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size=0, max_size=None, split_method='mean', max_top=10, coords=None,
                 logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building NField from cat %s',cat.name)
            else:
                logger.info('Building NField')

        self.min_size = float(min_size)
        self.max_size = float(max_size) if max_size is not None else 1.e300
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self.max_top = int(max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildNField(dp(cat.x), dp(cat.y), dp(cat.z),
                                              dp(cat.w), dp(cat.wpos), cat.ntot,
                                              self.min_size, self.max_size, self._sm,
                                              self.max_top, self._coords)
        if logger:
            logger.debug('Finished building NField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            treecorr._lib.DestroyNField(self.data, self._coords)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        return treecorr._lib.NFieldGetNTopLevel(self.data, self._coords)

    def count_near(self, x, y, z, sep):
        """Count how many points are near a given coordinate.

        Note: This function requires the coordinate to be given as (x,y,z) and the separation
              in compatible units.  There is also a version of this function in the Catalog
              class, which allows for more flexible specification of the coordinate
              (e.g. using ra, dec).

        :param x:       The x coordinate of the location for which to count nearby points.
        :param y:       The y coordinate of the location for which to count nearby points.
        :param z:       The z coordinate of the location for which to count nearby points.
        :param sep:     The separation distance
        """
        x = float(x)
        y = float(y)
        z = float(z)
        sep = float(sep)
        return treecorr._lib.NFieldCountNear(self.data, x, y, z, sep, self._coords)


class KField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep). (default: 0)
    :param max_size:    The maximum radius cell required (usually max_sep). (default: None)
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
                        (default: 'mean')
    :param max_top:     The maximum number of top layers to use when setting up the field.
                        (default: 10)
    :param coords       The kind of coordinate system to use. (default: cat.coords)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size=0, max_size=None, split_method='mean', max_top=10, coords=None,
                 logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building KField from cat %s',cat.name)
            else:
                logger.info('Building KField')

        self.min_size = float(min_size)
        self.max_size = float(max_size) if max_size is not None else 1.e300
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self.max_top = int(max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildKField(dp(cat.x), dp(cat.y), dp(cat.z),
                                              dp(cat.k),
                                              dp(cat.w), dp(cat.wpos), cat.ntot,
                                              self.min_size, self.max_size, self._sm,
                                              self.max_top, self._coords)
        if logger:
            logger.debug('Finished building KField (%s)',self.coords)


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            treecorr._lib.DestroyKField(self.data, self._coords)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        return treecorr._lib.NFieldGetNTopLevel(self.data, self._coords)

    def count_near(self, x, y, z, sep):
        """Count how many points are near a given coordinate.

        Note: This function requires the coordinate to be given as (x,y,z) and the separation
              in compatible units.  There is also a version of this function in the Catalog
              class, which allows for more flexible specification of the coordinate
              (e.g. using ra, dec).

        :param x:       The x coordinate of the location for which to count nearby points.
        :param y:       The y coordinate of the location for which to count nearby points.
        :param z:       The z coordinate of the location for which to count nearby points.
        :param sep:     The separation distance
        """
        x = float(x)
        y = float(y)
        z = float(z)
        sep = float(sep)
        return treecorr._lib.KFieldCountNear(self.data, x, y, z, sep, self._coords)



class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep). (default: 0)
    :param max_size:    The maximum radius cell required (usually max_sep). (default: None)
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
                        (default: 'mean')
    :param max_top:     The maximum number of top layers to use when setting up the field.
                        (default: 10)
    :param coords       The kind of coordinate system to use. (default: cat.coords)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size=0, max_size=None, split_method='mean', max_top=10, coords=None,
                 logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building GField from cat %s',cat.name)
            else:
                logger.info('Building GField')

        self.min_size = float(min_size)
        self.max_size = float(max_size) if max_size is not None else 1.e300
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self.max_top = int(max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = treecorr.util.coord_enum(self.coords)  # These are the C++-layer enums

        self.data = treecorr._lib.BuildGField(dp(cat.x), dp(cat.y), dp(cat.z),
                                              dp(cat.g1), dp(cat.g2),
                                              dp(cat.w), dp(cat.wpos), cat.ntot,
                                              self.min_size, self.max_size, self._sm,
                                              self.max_top, self._coords)
        if logger:
            logger.debug('Finished building GField (%s)',self.coords)


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            treecorr._lib.DestroyGField(self.data, self._coords)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        return treecorr._lib.GFieldGetNTopLevel(self.data, self._coords)

    def count_near(self, x, y, z, sep):
        """Count how many points are near a given coordinate.

        Note: This function requires the coordinate to be given as (x,y,z) and the separation
              in compatible units.  There is also a version of this function in the Catalog
              class, which allows for more flexible specification of the coordinate
              (e.g. using ra, dec).

        :param x:       The x coordinate of the location for which to count nearby points.
        :param y:       The y coordinate of the location for which to count nearby points.
        :param z:       The z coordinate of the location for which to count nearby points.
        :param sep:     The separation distance
        """
        x = float(x)
        y = float(y)
        z = float(z)
        sep = float(sep)
        return treecorr._lib.GFieldCountNear(self.data, x, y, z, sep, self._coords)


class NSimpleField(object):
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
            treecorr._lib.DestroyNSimpleField(self.data, self._coords)


class KSimpleField(object):
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
            treecorr._lib.DestroyKSimpleField(self.data, self._coords)


class GSimpleField(object):
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
            treecorr._lib.DestroyGSimpleField(self.data, self._coords)


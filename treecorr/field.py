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
import numpy

def _parse_split_method(split_method):
    if split_method == 'middle': return 0
    elif split_method == 'median': return 1
    else: return 2


class NField(object):
    """This class stores the positions in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep).
    :param max_size:    The maximum radius cell required (usually max_sep).
    :param split_method: Which split method to use ('mean', 'median', or 'middle')
                        (default: 'mean')
    :param max_top:     The maximum number of top layers to use when setting up the field. 
                        (default: 10)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size, max_size, split_method='mean', max_top=10, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building NField from cat %s',cat.name)
            else:
                logger.info('Building NField')

        self.min_size = min_size
        self.max_size = max_size
        self.split_method = split_method
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            self.data = treecorr.lib.BuildNFieldFlat(dp(cat.x),dp(cat.y),
                                                     dp(cat.w),dp(cat.wpos),cat.ntot,
                                                     min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building NField 2D')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr.lib.BuildNFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                           dp(cat.w),dp(cat.wpos),cat.ntot,
                                                           min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building NField Sphere')
            else:
                self.data = treecorr.lib.BuildNField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                       dp(cat.w),dp(cat.wpos),cat.ntot,
                                                       min_size,max_size,sm,max_top)
                self.spher = False
                if logger:
                    logger.debug('Finished building NField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr.lib.DestroyNFieldFlat(self.data)
            elif self.spher:
                treecorr.lib.DestroyNFieldSphere(self.data)
            else:
                treecorr.lib.DestroyNField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return treecorr.lib.NFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return treecorr.lib.NFieldSphereGetNTopLevel(self.data)
        else:
            return treecorr.lib.NField3DGetNTopLevel(self.data)


class KField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep).
    :param max_size:    The maximum radius cell required (usually max_sep).
    :param split_method: Which split method to use ('mean', 'median', or 'middle')
                        (default: 'mean')
    :param max_top:     The maximum number of top layers to use when setting up the
                        field. (default: 10)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size, max_size, split_method='mean', max_top=10, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building KField from cat %s',cat.name)
            else:
                logger.info('Building KField')

        self.min_size = min_size
        self.max_size = max_size
        self.split_method = split_method
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            self.data = treecorr.lib.BuildKFieldFlat(dp(cat.x),dp(cat.y),
                                                     dp(cat.k),
                                                     dp(cat.w),dp(cat.wpos),cat.ntot,
                                                     min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building KField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr.lib.BuildKFieldThreeD(dp(cat.x),dp(cat.y),dp(cat.z),
                                                           dp(cat.k),
                                                           dp(cat.w),dp(cat.wpos),cat.ntot,
                                                           min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building KField Sphere')
            else:
                self.data = treecorr.lib.BuildKField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                       dp(cat.k),
                                                       dp(cat.w),dp(cat.wpos),cat.ntot,
                                                       min_size,max_size,sm,max_top)
                self.spher = False
                if logger:
                    logger.debug('Finished building KField 3D')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr.lib.DestroyKFieldFlat(self.data)
            elif self.spher:
                treecorr.lib.DestroyKFieldSphere(self.data)
            else:
                treecorr.lib.DestroyKField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return treecorr.lib.NFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return treecorr.lib.NFieldSphereGetNTopLevel(self.data)
        else:
            return treecorr.lib.NField3DGetNTopLevel(self.data)



class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep).
    :param max_size:    The maximum radius cell required (usually max_sep).
    :param split_method: Which split method to use ('mean', 'median', or 'middle')
                        (default: 'mean')
    :param max_top:     The maximum number of top layers to use when setting up the
                        field. (default: 10)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_size, max_size, split_method='mean', max_top=10, logger=None):
        from treecorr.util import double_ptr as dp
        if logger:
            if cat.name != '':
                logger.info('Building GField from cat %s',cat.name)
            else:
                logger.info('Building GField')

        self.min_size = min_size
        self.max_size = max_size
        self.split_method = split_method
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            self.data = treecorr.lib.BuildGFieldFlat(dp(cat.x),dp(cat.y),
                                                     dp(cat.g1),dp(cat.g2),
                                                     dp(cat.w),dp(cat.wpos),cat.ntot,
                                                     min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building GField Flat')
        else: 
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr.lib.BuildGFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                           dp(cat.g1),dp(cat.g2),
                                                           dp(cat.w),dp(cat.wpos),cat.ntot,
                                                           min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building GField Sphere')
            else:
                self.data = treecorr.lib.BuildGField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                       dp(cat.g1),dp(cat.g2),
                                                       dp(cat.w),dp(cat.wpos),cat.ntot,
                                                       min_size,max_size,sm,max_top)
                self.spher = False
                if logger:
                    logger.debug('Finished building GField 3D')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr.lib.DestroyGFieldFlat(self.data)
            elif self.spher:
                treecorr.lib.DestroyGFieldSphere(self.data)
            else:
                treecorr.lib.DestroyGField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return treecorr.lib.GFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return treecorr.lib.GFieldSphereGetNTopLevel(self.data)
        else:
            return treecorr.lib.GField3DGetNTopLevel(self.data)


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

        if cat.coords == 'flat':
            self.flat = True
            self.data = treecorr.lib.BuildNSimpleFieldFlat(dp(cat.x),dp(cat.y),
                                                           dp(cat.w),dp(cat.wpos),cat.ntot)
            if logger:
                logger.debug('Finished building NSimpleField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr.lib.BuildNSimpleFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                                 dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building NSimpleField Sphere')
            else:
                self.data = treecorr.lib.BuildNSimpleField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                             dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = False
                if logger:
                    logger.debug('Finished building NSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr.lib.DestroyNSimpleFieldFlat(self.data)
            elif self.spher:
                treecorr.lib.DestroyNSimpleFieldSphere(self.data)
            else:
                treecorr.lib.DestroyNSimpleField3D(self.data)


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

        if cat.coords == 'flat':
            self.flat = True
            self.data = treecorr.lib.BuildKSimpleFieldFlat(dp(cat.x),dp(cat.y),
                                                           dp(cat.k),
                                                           dp(cat.w),dp(cat.wpos),cat.ntot)
            if logger:
                logger.debug('Finished building KSimpleField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr.lib.BuildKSimpleFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                                 dp(cat.k),
                                                                 dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building KSimpleField Sphere')
            else:
                self.data = treecorr.lib.BuildKSimpleField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                             dp(cat.k),
                                                             dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = False
                if logger:
                    logger.debug('Finished building KSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr.lib.DestroyKSimpleFieldFlat(self.data)
            elif self.spher:
                treecorr.lib.DestroyKSimpleFieldSphere(self.data)
            else:
                treecorr.lib.DestroyKSimpleField3D(self.data)


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

        if cat.coords == 'flat':
            self.flat = True
            self.data = treecorr.lib.BuildGSimpleFieldFlat(dp(cat.x),dp(cat.y),
                                                           dp(cat.g1),dp(cat.g2),
                                                           dp(cat.w),dp(cat.wpos),cat.ntot)
            if logger:
                logger.debug('Finished building GSimpleField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr.lib.BuildGSimpleFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                                 dp(cat.g1),dp(cat.g2),
                                                                 dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building GSimpleField Sphere')
            else:
                self.data = treecorr.lib.BuildGSimpleField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                             dp(cat.g1),dp(cat.g2),
                                                             dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = False
                if logger:
                    logger.debug('Finished building GSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr.lib.DestroyGSimpleFieldFlat(self.data)
            elif self.spher:
                treecorr.lib.DestroyGSimpleFieldSphere(self.data)
            else:
                treecorr.lib.DestroyGSimpleField3D(self.data)


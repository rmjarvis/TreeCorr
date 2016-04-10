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
    elif split_method == 'mean': return 2
    else: return 3


class NField(object):
    """This class stores the positions in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep).
    :param max_size:    The maximum radius cell required (usually max_sep).
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
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
            self.data = treecorr._lib.BuildNFieldFlat(dp(cat.x),dp(cat.y),
                                                      dp(cat.w),dp(cat.wpos),cat.ntot,
                                                      min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building NField 2D')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr._lib.BuildNFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                            dp(cat.w),dp(cat.wpos),cat.ntot,
                                                            min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building NField Sphere')
            else:
                self.data = treecorr._lib.BuildNField3D(dp(cat.x),dp(cat.y),dp(cat.z),
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
                treecorr._lib.DestroyNFieldFlat(self.data)
            elif self.spher:
                treecorr._lib.DestroyNFieldSphere(self.data)
            else:
                treecorr._lib.DestroyNField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return treecorr._lib.NFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return treecorr._lib.NFieldSphereGetNTopLevel(self.data)
        else:
            return treecorr._lib.NField3DGetNTopLevel(self.data)


class KField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep).
    :param max_size:    The maximum radius cell required (usually max_sep).
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
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
            self.data = treecorr._lib.BuildKFieldFlat(dp(cat.x),dp(cat.y),
                                                      dp(cat.k),
                                                      dp(cat.w),dp(cat.wpos),cat.ntot,
                                                      min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building KField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr._lib.BuildKFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                            dp(cat.k),
                                                            dp(cat.w),dp(cat.wpos),cat.ntot,
                                                            min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building KField Sphere')
            else:
                self.data = treecorr._lib.BuildKField3D(dp(cat.x),dp(cat.y),dp(cat.z),
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
                treecorr._lib.DestroyKFieldFlat(self.data)
            elif self.spher:
                treecorr._lib.DestroyKFieldSphere(self.data)
            else:
                treecorr._lib.DestroyKField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return treecorr._lib.NFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return treecorr._lib.NFieldSphereGetNTopLevel(self.data)
        else:
            return treecorr._lib.NField3DGetNTopLevel(self.data)



class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_size, max_size, b)

    :param cat:         The catalog from which to make the field.
    :param min_size:    The minimum radius cell required (usually min_sep).
    :param max_size:    The maximum radius cell required (usually max_sep).
    :param split_method: Which split method to use ('mean', 'median', 'middle', or 'random')
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
            self.data = treecorr._lib.BuildGFieldFlat(dp(cat.x),dp(cat.y),
                                                      dp(cat.g1),dp(cat.g2),
                                                      dp(cat.w),dp(cat.wpos),cat.ntot,
                                                      min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building GField Flat')
        else: 
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr._lib.BuildGFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                            dp(cat.g1),dp(cat.g2),
                                                            dp(cat.w),dp(cat.wpos),cat.ntot,
                                                            min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building GField Sphere')
            else:
                self.data = treecorr._lib.BuildGField3D(dp(cat.x),dp(cat.y),dp(cat.z),
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
                treecorr._lib.DestroyGFieldFlat(self.data)
            elif self.spher:
                treecorr._lib.DestroyGFieldSphere(self.data)
            else:
                treecorr._lib.DestroyGField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return treecorr._lib.GFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return treecorr._lib.GFieldSphereGetNTopLevel(self.data)
        else:
            return treecorr._lib.GField3DGetNTopLevel(self.data)


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
            self.data = treecorr._lib.BuildNSimpleFieldFlat(dp(cat.x),dp(cat.y),
                                                            dp(cat.w),dp(cat.wpos),cat.ntot)
            if logger:
                logger.debug('Finished building NSimpleField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr._lib.BuildNSimpleFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                                  dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building NSimpleField Sphere')
            else:
                self.data = treecorr._lib.BuildNSimpleField3D(dp(cat.x),dp(cat.y),dp(cat.z),
                                                              dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = False
                if logger:
                    logger.debug('Finished building NSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                treecorr._lib.DestroyNSimpleFieldFlat(self.data)
            elif self.spher:
                treecorr._lib.DestroyNSimpleFieldSphere(self.data)
            else:
                treecorr._lib.DestroyNSimpleField3D(self.data)


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
            self.data = treecorr._lib.BuildKSimpleFieldFlat(dp(cat.x),dp(cat.y),
                                                            dp(cat.k),
                                                            dp(cat.w),dp(cat.wpos),cat.ntot)
            if logger:
                logger.debug('Finished building KSimpleField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr._lib.BuildKSimpleFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                                  dp(cat.k),
                                                                  dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building KSimpleField Sphere')
            else:
                self.data = treecorr._lib.BuildKSimpleField3D(dp(cat.x),dp(cat.y),dp(cat.z),
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
                treecorr._lib.DestroyKSimpleFieldFlat(self.data)
            elif self.spher:
                treecorr._lib.DestroyKSimpleFieldSphere(self.data)
            else:
                treecorr._lib.DestroyKSimpleField3D(self.data)


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
            self.data = treecorr._lib.BuildGSimpleFieldFlat(dp(cat.x),dp(cat.y),
                                                            dp(cat.g1),dp(cat.g2),
                                                            dp(cat.w),dp(cat.wpos),cat.ntot)
            if logger:
                logger.debug('Finished building GSimpleField Flat')
        else:
            self.flat = False
            if cat.coords == 'spherical':
                self.data = treecorr._lib.BuildGSimpleFieldSphere(dp(cat.x),dp(cat.y),dp(cat.z),
                                                                  dp(cat.g1),dp(cat.g2),
                                                                  dp(cat.w),dp(cat.wpos),cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building GSimpleField Sphere')
            else:
                self.data = treecorr._lib.BuildGSimpleField3D(dp(cat.x),dp(cat.y),dp(cat.z),
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
                treecorr._lib.DestroyGSimpleFieldFlat(self.data)
            elif self.spher:
                treecorr._lib.DestroyGSimpleFieldSphere(self.data)
            else:
                treecorr._lib.DestroyGSimpleField3D(self.data)


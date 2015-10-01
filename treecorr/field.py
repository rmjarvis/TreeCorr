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

# Start by loading up the relevant C functions using ctypes
import numpy
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))

# Some convenient aliases:
cvoid_ptr = ctypes.c_void_p
cdouble = ctypes.c_double
clong = ctypes.c_long
cint = ctypes.c_int
cdouble_ptr = ctypes.POINTER(cdouble)

# Define the restypes and argtypes for the C functions:
_treecorr.BuildNFieldFlat.restype = cvoid_ptr
_treecorr.BuildNFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, 
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cdouble, cint, cint ]
_treecorr.BuildNField3D.restype = cvoid_ptr
_treecorr.BuildNField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong,
    cdouble, cdouble, cdouble, cint, cint, cint ]
_treecorr.BuildNFieldPerp.restype = cvoid_ptr
_treecorr.BuildNFieldPerp.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong,
    cdouble, cdouble, cdouble, cint, cint ]

_treecorr.DestroyNFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNFieldPerp.argtypes = [ cvoid_ptr ]

_treecorr.BuildKFieldFlat.restype = cvoid_ptr
_treecorr.BuildKFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, 
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cdouble, cint, cint ]
_treecorr.BuildKField3D.restype = cvoid_ptr
_treecorr.BuildKField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong,
    cdouble, cdouble, cdouble, cint, cint, cint ]
_treecorr.BuildKFieldPerp.restype = cvoid_ptr
_treecorr.BuildKFieldPerp.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cdouble, cint, cint ]

_treecorr.DestroyKFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKFieldPerp.argtypes = [ cvoid_ptr ]

_treecorr.BuildGFieldFlat.restype = cvoid_ptr
_treecorr.BuildGFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cdouble, cint, cint ]
_treecorr.BuildGField3D.restype = cvoid_ptr
_treecorr.BuildGField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cdouble, cint, cint, cint ]
_treecorr.BuildGFieldPerp.restype = cvoid_ptr
_treecorr.BuildGFieldPerp.argtypes = [
    cdouble_ptr, cdouble_ptr,cdouble_ptr,  cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cdouble, cint, cint ]

_treecorr.DestroyGFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGFieldPerp.argtypes = [ cvoid_ptr ]

_treecorr.BuildNSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildNSimpleFieldFlat.argtypes = [ 
    cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildNSimpleField3D.restype = cvoid_ptr
_treecorr.BuildNSimpleField3D.argtypes = [ 
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildNSimpleFieldPerp.restype = cvoid_ptr
_treecorr.BuildNSimpleFieldPerp.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]

_treecorr.DestroyNSimpleFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNSimpleField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNSimpleFieldPerp.argtypes = [ cvoid_ptr ]

_treecorr.BuildKSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildKSimpleFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildKSimpleField3D.restype = cvoid_ptr
_treecorr.BuildKSimpleField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildKSimpleFieldPerp.restype = cvoid_ptr
_treecorr.BuildKSimpleFieldPerp.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]

_treecorr.DestroyKSimpleFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKSimpleField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKSimpleFieldPerp.argtypes = [ cvoid_ptr ]

_treecorr.BuildGSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildGSimpleFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildGSimpleField3D.restype = cvoid_ptr
_treecorr.BuildGSimpleField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildGSimpleFieldPerp.restype = cvoid_ptr
_treecorr.BuildGSimpleFieldPerp.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, 
    cdouble_ptr, cdouble_ptr, clong ]

_treecorr.DestroyGSimpleFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGSimpleField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGSimpleFieldPerp.argtypes = [ cvoid_ptr ]

_treecorr.NFieldFlatGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.NFieldFlatGetNTopLevel.restype = clong
_treecorr.NField3DGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.NField3DGetNTopLevel.restype = clong
_treecorr.NFieldPerpGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.NFieldPerpGetNTopLevel.restype = clong
_treecorr.KFieldFlatGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.KFieldFlatGetNTopLevel.restype = clong
_treecorr.KField3DGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.KField3DGetNTopLevel.restype = clong
_treecorr.KFieldPerpGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.KFieldPerpGetNTopLevel.restype = clong
_treecorr.GFieldFlatGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.GFieldFlatGetNTopLevel.restype = clong
_treecorr.GField3DGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.GField3DGetNTopLevel.restype = clong
_treecorr.GFieldPerpGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.GFieldPerpGetNTopLevel.restype = clong

def _parse_split_method(split_method):
    if split_method == 'middle': return cint(0)
    elif split_method == 'median': return cint(1)
    else: return cint(2)


class NField(object):
    """This class stores the positions in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_sep, max_sep, b)

    :param cat:         The catalog from which to make the field.
    :param min_sep:     The minimum separation between points that will be needed.
    :param max_sep:     The maximum separation between points that will be needed.
    :param b:           The b parameter that will be used for the correlation function.
                        This should be bin_size * bin_slop.
    :param split_method: Which split method to use ('mean', 'median', or 'middle')
                        (default: 'mean')
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param max_top:     The maximum number of top layers to use when setting up the
                            field. (default: 10)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_sep, max_sep, b, split_method='mean', metric='Euclidean',
                 max_top=10, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building NField from cat %s',cat.name)
            else:
                logger.info('Building NField')

        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if metric == 'Rperp' and cat.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method
        self.metric = metric

        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNFieldFlat(x,y,w,wpos,cat.ntot,min_sep,max_sep,b,sm,max_top)
            if logger:
                logger.debug('Finished building NField 2D')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                spher = 1
            else:
                spher = 0
            if self.metric == 'Rperp':
                # Go a bit smller than min_sep for Rperp metric, since the simple calculation of
                # what minimum size to use isn't exactly accurate in this case.
                min_sep /= 2.
                self.data = _treecorr.BuildNFieldPerp(x,y,z,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                      sm,max_top)
                self.perp = True
                if logger:
                    logger.debug('Finished building NField Perp')
            else:
                self.data = _treecorr.BuildNField3D(x,y,z,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                    sm,max_top,spher)
                self.perp = False
                if logger:
                    logger.debug('Finished building NField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyNFieldFlat(self.data)
            elif self.perp:
                _treecorr.DestroyNFieldPerp(self.data)
            else:
                _treecorr.DestroyNField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return _treecorr.NFieldFlatGetNTopLevel(self.data)
        elif self.perp:
            return _treecorr.NFieldPerpGetNTopLevel(self.data)
        else:
            return _treecorr.NField3DGetNTopLevel(self.data)


class KField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_sep, max_sep, b)

    :param cat:         The catalog from which to make the field.
    :param min_sep:     The minimum separation between points that will be needed.
    :param max_sep:     The maximum separation between points that will be needed.
    :param b:           The b parameter that will be used for the correlation function.
                        This should be bin_size * bin_slop.
    :param split_method: Which split method to use ('mean', 'median', or 'middle')
                        (default: 'mean')
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param max_top:     The maximum number of top layers to use when setting up the
                        field. (default: 10)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_sep, max_sep, b, split_method='mean', metric='Euclidean',
                 max_top=10, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building KField from cat %s',cat.name)
            else:
                logger.info('Building KField')

        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if metric == 'Rperp' and cat.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method
        self.metric = metric

        k = cat.k.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKFieldFlat(x,y,k,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                  sm,max_top)
            if logger:
                logger.debug('Finished building KField Flat')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                spher = 1
            else:
                spher = 0
            if self.metric == 'Rperp':
                # Go a bit smller than min_sep for Rperp metric, since the simple calculation of
                # what minimum size to use isn't exactly accurate in this case.
                min_sep /= 2.
                self.data = _treecorr.BuildKFieldPerp(x,y,z,k,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                      sm,max_top)
                self.perp = True
                if logger:
                    logger.debug('Finished building KField Perp')
            else:
                self.data = _treecorr.BuildKField3D(x,y,z,k,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                    sm,max_top,spher)
                self.perp = False
                if logger:
                    logger.debug('Finished building KField 3D')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyKFieldFlat(self.data)
            elif self.perp:
                _treecorr.DestroyKFieldPerp(self.data)
            else:
                _treecorr.DestroyKField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return _treecorr.NFieldFlatGetNTopLevel(self.data)
        elif self.perp:
            return _treecorr.NFieldPerpGetNTopLevel(self.data)
        else:
            return _treecorr.NField3DGetNTopLevel(self.data)



class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_sep, max_sep, b)

    :param cat:         The catalog from which to make the field.
    :param min_sep:     The minimum separation between points that will be needed.
    :param max_sep:     The maximum separation between points that will be needed.
    :param b:           The b parameter that will be used for the correlation function.
                        This should be bin_size * bin_slop.
    :param split_method: Which split method to use ('mean', 'median', or 'middle')
                        (default: 'mean')
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param max_top:     The maximum number of top layers to use when setting up the
                        field. (default: 10)
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, min_sep, max_sep, b, split_method='mean', metric='Euclidean',
                 max_top=10, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building GField from cat %s',cat.name)
            else:
                logger.info('Building GField')

        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if metric == 'Rperp' and cat.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method
        self.metric = metric

        g1 = cat.g1.ctypes.data_as(cdouble_ptr)
        g2 = cat.g2.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGFieldFlat(x,y,g1,g2,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                  sm,max_top)
            if logger:
                logger.debug('Finished building GField Flat')
        else: 
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                spher = 1
            else:
                spher = 0
            if self.metric == 'Rperp':
                # Go a bit smller than min_sep for Rperp metric, since the simple calculation of
                # what minimum size to use isn't exactly accurate in this case.
                min_sep /= 2.
                self.data = _treecorr.BuildGFieldPerp(x,y,z,g1,g2,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                      sm,max_top)
                self.perp = True
                if logger:
                    logger.debug('Finished building GField Perp')
            else:
                self.data = _treecorr.BuildGField3D(x,y,z,g1,g2,w,wpos,cat.ntot,min_sep,max_sep,b,
                                                    sm,max_top,spher)
                self.perp = False
                if logger:
                    logger.debug('Finished building GField 3D')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyGFieldFlat(self.data)
            elif self.perp:
                _treecorr.DestroyGFieldPerp(self.data)
            else:
                _treecorr.DestroyGField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return _treecorr.GFieldFlatGetNTopLevel(self.data)
        elif self.perp:
            return _treecorr.GFieldPerpGetNTopLevel(self.data)
        else:
            return _treecorr.GField3DGetNTopLevel(self.data)


class NSimpleField(object):
    """This class stores the positions as a list, skipping all the tree stuff.

    An NSimpleField is typically created from a Catalog object using

        >>> nfield = cat.getNSimpleField()

    :param cat:         The catalog from which to make the field.
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, metric='Euclidean', logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building NSimpleField from cat %s',cat.name)
            else:
                logger.info('Building NSimpleField')

        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if metric == 'Rperp' and cat.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)

        self.metric = metric

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNSimpleFieldFlat(x,y,w,wpos,cat.ntot)
            if logger:
                logger.debug('Finished building NSimpleField Flat')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                spher = 1
            else:
                spher = 0
            if self.metric == 'Rperp':
                self.data = _treecorr.BuildNSimpleFieldPerp(x,y,z,w,wpos,cat.ntot)
                self.perp = True
                if logger:
                    logger.debug('Finished building NSimpleField Perp')
            else:
                self.data = _treecorr.BuildNSimpleField3D(x,y,z,w,wpos,cat.ntot,spher)
                self.perp = False
                if logger:
                    logger.debug('Finished building NSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyNSimpleFieldFlat(self.data)
            elif self.perp:
                _treecorr.DestroyNSimpleFieldPerp(self.data)
            else:
                _treecorr.DestroyNSimpleField3D(self.data)


class KSimpleField(object):
    """This class stores the kappa field as a list, skipping all the tree stuff.

    A KSimpleField is typically created from a Catalog object using

        >>> kfield = cat.getKSimpleField()

    :param cat:         The catalog from which to make the field.
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, metric='Euclidean', logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building KSimpleField from cat %s',cat.name)
            else:
                logger.info('Building KSimpleField')

        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if metric == 'Rperp' and cat.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        k = cat.k.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)

        self.metric = metric

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKSimpleFieldFlat(x,y,k,w,wpos,cat.ntot)
            if logger:
                logger.debug('Finished building KSimpleField Flat')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                spher = 1
            else:
                spher = 0
            if self.metric == 'Rperp':
                self.data = _treecorr.BuildKSimpleFieldPerp(x,y,z,k,w,wpos,cat.ntot)
                self.perp = True
                if logger:
                    logger.debug('Finished building KSimpleField Perp')
            else:
                self.data = _treecorr.BuildKSimpleField3D(x,y,z,k,w,wpos,cat.ntot,spher)
                self.perp = False
                if logger:
                    logger.debug('Finished building KSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyKSimpleFieldFlat(self.data)
            elif self.perp:
                _treecorr.DestroyKSimpleFieldPerp(self.data)
            else:
                _treecorr.DestroyKSimpleField3D(self.data)


class GSimpleField(object):
    """This class stores the shear field as a list, skipping all the tree stuff.

    A GSimpleField is typically created from a Catalog object using

        >>> gfield = cat.getGSimpleField()

    :param cat:         The catalog from which to make the field.
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, metric='Euclidean', logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building GSimpleField from cat %s',cat.name)
            else:
                logger.info('Building GSimpleField')

        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if metric == 'Rperp' and cat.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        g1 = cat.g1.ctypes.data_as(cdouble_ptr)
        g2 = cat.g2.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)

        self.metric = metric

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGSimpleFieldFlat(x,y,g1,g2,w,wpos,cat.ntot)
            if logger:
                logger.debug('Finished building GSimpleField Flat')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                spher = 1
            else:
                spher = 0
            if self.metric == 'Rperp':
                self.data = _treecorr.BuildGSimpleFieldPerp(x,y,z,g1,g2,w,wpos,cat.ntot)
                self.perp = True
                if logger:
                    logger.debug('Finished building GSimpleField Perp')
            else:
                self.data = _treecorr.BuildGSimpleField3D(x,y,z,g1,g2,w,wpos,cat.ntot,spher)
                self.perp = False
                if logger:
                    logger.debug('Finished building GSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyGSimpleFieldFlat(self.data)
            elif self.perp:
                _treecorr.DestroyGSimpleFieldPerp(self.data)
            else:
                _treecorr.DestroyGSimpleField3D(self.data)


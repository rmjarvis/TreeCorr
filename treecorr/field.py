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
    cdouble, cdouble, cint, cint ]
_treecorr.BuildNField3D.restype = cvoid_ptr
_treecorr.BuildNField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong,
    cdouble, cdouble, cint, cint ]
_treecorr.BuildNFieldSphere.restype = cvoid_ptr
_treecorr.BuildNFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong,
    cdouble, cdouble, cint, cint ]

_treecorr.DestroyNFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNFieldSphere.argtypes = [ cvoid_ptr ]

_treecorr.BuildKFieldFlat.restype = cvoid_ptr
_treecorr.BuildKFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, 
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cint, cint ]
_treecorr.BuildKField3D.restype = cvoid_ptr
_treecorr.BuildKField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong,
    cdouble, cdouble, cint, cint ]
_treecorr.BuildKFieldSphere.restype = cvoid_ptr
_treecorr.BuildKFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cint, cint ]

_treecorr.DestroyKFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKFieldSphere.argtypes = [ cvoid_ptr ]

_treecorr.BuildGFieldFlat.restype = cvoid_ptr
_treecorr.BuildGFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cint, cint ]
_treecorr.BuildGField3D.restype = cvoid_ptr
_treecorr.BuildGField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cint, cint ]
_treecorr.BuildGFieldSphere.restype = cvoid_ptr
_treecorr.BuildGFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr,cdouble_ptr,  cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong, 
    cdouble, cdouble, cint, cint ]

_treecorr.DestroyGFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGFieldSphere.argtypes = [ cvoid_ptr ]

_treecorr.BuildNSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildNSimpleFieldFlat.argtypes = [ 
    cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildNSimpleField3D.restype = cvoid_ptr
_treecorr.BuildNSimpleField3D.argtypes = [ 
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildNSimpleFieldSphere.restype = cvoid_ptr
_treecorr.BuildNSimpleFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]

_treecorr.DestroyNSimpleFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNSimpleField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNSimpleFieldSphere.argtypes = [ cvoid_ptr ]

_treecorr.BuildKSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildKSimpleFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildKSimpleField3D.restype = cvoid_ptr
_treecorr.BuildKSimpleField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildKSimpleFieldSphere.restype = cvoid_ptr
_treecorr.BuildKSimpleFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]

_treecorr.DestroyKSimpleFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKSimpleField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKSimpleFieldSphere.argtypes = [ cvoid_ptr ]

_treecorr.BuildGSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildGSimpleFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildGSimpleField3D.restype = cvoid_ptr
_treecorr.BuildGSimpleField3D.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildGSimpleFieldSphere.restype = cvoid_ptr
_treecorr.BuildGSimpleFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, 
    cdouble_ptr, cdouble_ptr, clong ]

_treecorr.DestroyGSimpleFieldFlat.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGSimpleField3D.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGSimpleFieldSphere.argtypes = [ cvoid_ptr ]

_treecorr.NFieldFlatGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.NFieldFlatGetNTopLevel.restype = clong
_treecorr.NField3DGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.NField3DGetNTopLevel.restype = clong
_treecorr.NFieldSphereGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.NFieldSphereGetNTopLevel.restype = clong
_treecorr.KFieldFlatGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.KFieldFlatGetNTopLevel.restype = clong
_treecorr.KField3DGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.KField3DGetNTopLevel.restype = clong
_treecorr.KFieldSphereGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.KFieldSphereGetNTopLevel.restype = clong
_treecorr.GFieldFlatGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.GFieldFlatGetNTopLevel.restype = clong
_treecorr.GField3DGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.GField3DGetNTopLevel.restype = clong
_treecorr.GFieldSphereGetNTopLevel.argtypes = [ cvoid_ptr ]
_treecorr.GFieldSphereGetNTopLevel.restype = clong

def _parse_split_method(split_method):
    if split_method == 'middle': return cint(0)
    elif split_method == 'median': return cint(1)
    else: return cint(2)


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
        if logger:
            if cat.name != '':
                logger.info('Building NField from cat %s',cat.name)
            else:
                logger.info('Building NField')

        self.min_size = min_size
        self.max_size = max_size
        self.split_method = split_method

        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNFieldFlat(x,y,w,wpos,cat.ntot,
                                                  min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building NField 2D')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                self.data = _treecorr.BuildNFieldSphere(x,y,z,w,wpos,cat.ntot,
                                                        min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building NField Sphere')
            else:
                self.data = _treecorr.BuildNField3D(x,y,z,w,wpos,cat.ntot,
                                                    min_size,max_size,sm,max_top)
                self.spher = False
                if logger:
                    logger.debug('Finished building NField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyNFieldFlat(self.data)
            elif self.spher:
                _treecorr.DestroyNFieldSphere(self.data)
            else:
                _treecorr.DestroyNField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return _treecorr.NFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return _treecorr.NFieldSphereGetNTopLevel(self.data)
        else:
            return _treecorr.NField3DGetNTopLevel(self.data)


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
        if logger:
            if cat.name != '':
                logger.info('Building KField from cat %s',cat.name)
            else:
                logger.info('Building KField')

        self.min_size = min_size
        self.max_size = max_size
        self.split_method = split_method

        k = cat.k.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKFieldFlat(x,y,k,w,wpos,cat.ntot,
                                                  min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building KField Flat')
        else:
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                self.data = _treecorr.BuildKFieldThreeD(x,y,z,k,w,wpos,cat.ntot,
                                                        min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building KField Sphere')
            else:
                self.data = _treecorr.BuildKField3D(x,y,z,k,w,wpos,cat.ntot,
                                                    min_size,max_size,sm,max_top)
                self.spher = False
                if logger:
                    logger.debug('Finished building KField 3D')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyKFieldFlat(self.data)
            elif self.spher:
                _treecorr.DestroyKFieldSphere(self.data)
            else:
                _treecorr.DestroyKField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return _treecorr.NFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return _treecorr.NFieldSphereGetNTopLevel(self.data)
        else:
            return _treecorr.NField3DGetNTopLevel(self.data)



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
        if logger:
            if cat.name != '':
                logger.info('Building GField from cat %s',cat.name)
            else:
                logger.info('Building GField')

        self.min_size = min_size
        self.max_size = max_size
        self.split_method = split_method

        g1 = cat.g1.ctypes.data_as(cdouble_ptr)
        g2 = cat.g2.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)
        sm = _parse_split_method(split_method)

        if cat.coords == 'flat':
            self.flat = True
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGFieldFlat(x,y,g1,g2,w,wpos,cat.ntot,
                                                  min_size,max_size,sm,max_top)
            if logger:
                logger.debug('Finished building GField Flat')
        else: 
            self.flat = False
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            z = cat.z.ctypes.data_as(cdouble_ptr)
            if cat.coords == 'spherical':
                self.data = _treecorr.BuildGFieldSphere(x,y,z,g1,g2,w,wpos,cat.ntot,
                                                        min_size,max_size,sm,max_top)
                self.spher = True
                if logger:
                    logger.debug('Finished building GField Sphere')
            else:
                self.data = _treecorr.BuildGField3D(x,y,z,g1,g2,w,wpos,cat.ntot,
                                                    min_size,max_size,sm,max_top)
                self.spher = False
                if logger:
                    logger.debug('Finished building GField 3D')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyGFieldFlat(self.data)
            elif self.spher:
                _treecorr.DestroyGFieldSphere(self.data)
            else:
                _treecorr.DestroyGField3D(self.data)

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes."""
        if self.flat:
            return _treecorr.GFieldFlatGetNTopLevel(self.data)
        elif self.spher:
            return _treecorr.GFieldSphereGetNTopLevel(self.data)
        else:
            return _treecorr.GField3DGetNTopLevel(self.data)


class NSimpleField(object):
    """This class stores the positions as a list, skipping all the tree stuff.

    An NSimpleField is typically created from a Catalog object using

        >>> nfield = cat.getNSimpleField()

    :param cat:         The catalog from which to make the field.
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building NSimpleField from cat %s',cat.name)
            else:
                logger.info('Building NSimpleField')

        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)

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
                self.data = _treecorr.BuildNSimpleFieldSphere(x,y,z,w,wpos,cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building NSimpleField Sphere')
            else:
                self.data = _treecorr.BuildNSimpleField3D(x,y,z,w,wpos,cat.ntot)
                self.spher = False
                if logger:
                    logger.debug('Finished building NSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyNSimpleFieldFlat(self.data)
            elif self.spher:
                _treecorr.DestroyNSimpleFieldSphere(self.data)
            else:
                _treecorr.DestroyNSimpleField3D(self.data)


class KSimpleField(object):
    """This class stores the kappa field as a list, skipping all the tree stuff.

    A KSimpleField is typically created from a Catalog object using

        >>> kfield = cat.getKSimpleField()

    :param cat:         The catalog from which to make the field.
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building KSimpleField from cat %s',cat.name)
            else:
                logger.info('Building KSimpleField')

        k = cat.k.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)

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
                self.data = _treecorr.BuildKSimpleFieldSphere(x,y,z,k,w,wpos,cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building KSimpleField Sphere')
            else:
                self.data = _treecorr.BuildKSimpleField3D(x,y,z,k,w,wpos,cat.ntot)
                self.spher = False
                if logger:
                    logger.debug('Finished building KSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyKSimpleFieldFlat(self.data)
            elif self.spher:
                _treecorr.DestroyKSimpleFieldSphere(self.data)
            else:
                _treecorr.DestroyKSimpleField3D(self.data)


class GSimpleField(object):
    """This class stores the shear field as a list, skipping all the tree stuff.

    A GSimpleField is typically created from a Catalog object using

        >>> gfield = cat.getGSimpleField()

    :param cat:         The catalog from which to make the field.
    :param logger:      A logger file if desired (default: None)
    """
    def __init__(self, cat, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building GSimpleField from cat %s',cat.name)
            else:
                logger.info('Building GSimpleField')

        g1 = cat.g1.ctypes.data_as(cdouble_ptr)
        g2 = cat.g2.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        wpos = cat.wpos.ctypes.data_as(cdouble_ptr)

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
                self.data = _treecorr.BuildGSimpleFieldSphere(x,y,z,g1,g2,w,wpos,cat.ntot)
                self.spher = True
                if logger:
                    logger.debug('Finished building GSimpleField Sphere')
            else:
                self.data = _treecorr.BuildGSimpleField3D(x,y,z,g1,g2,w,wpos,cat.ntot,spher)
                self.spher = False
                if logger:
                    logger.debug('Finished building GSimpleField 3D')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.flat:
                _treecorr.DestroyGSimpleFieldFlat(self.data)
            elif self.spher:
                _treecorr.DestroyGSimpleFieldSphere(self.data)
            else:
                _treecorr.DestroyGSimpleField3D(self.data)


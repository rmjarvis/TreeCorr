# Copyright (c) 2003-2014 by Mike Jarvis
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

import treecorr

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
_treecorr.BuildNFieldSphere.restype = cvoid_ptr
_treecorr.BuildNFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong, cdouble, cdouble, cdouble, cint ]
_treecorr.BuildNFieldFlat.restype = cvoid_ptr
_treecorr.BuildNFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, clong, cdouble, cdouble, cdouble, cint ]
_treecorr.DestroyNFieldSphere.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNFieldFlat.argtypes = [ cvoid_ptr ]

_treecorr.BuildKFieldSphere.restype = cvoid_ptr
_treecorr.BuildKFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr,cdouble_ptr,  cdouble_ptr, cdouble_ptr, clong, cdouble, cdouble, cdouble, cint ]
_treecorr.BuildKFieldFlat.restype = cvoid_ptr
_treecorr.BuildKFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong, cdouble, cdouble, cdouble, cint ]
_treecorr.DestroyKFieldSphere.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKFieldFlat.argtypes = [ cvoid_ptr ]

_treecorr.BuildGFieldSphere.restype = cvoid_ptr
_treecorr.BuildGFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr,cdouble_ptr,  cdouble_ptr, cdouble_ptr, cdouble_ptr,
    clong, cdouble, cdouble, cdouble, cint ]
_treecorr.BuildGFieldFlat.restype = cvoid_ptr
_treecorr.BuildGFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr,
    clong, cdouble, cdouble, cdouble, cint ]
_treecorr.DestroyGFieldSphere.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGFieldFlat.argtypes = [ cvoid_ptr ]

_treecorr.BuildNSimpleFieldSphere.restype = cvoid_ptr
_treecorr.BuildNSimpleFieldSphere.argtypes = [ cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildNSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildNSimpleFieldFlat.argtypes = [ cdouble_ptr, cdouble_ptr, cdouble_ptr, clong ]
_treecorr.DestroyNSimpleFieldSphere.argtypes = [ cvoid_ptr ]
_treecorr.DestroyNSimpleFieldFlat.argtypes = [ cvoid_ptr ]

_treecorr.BuildKSimpleFieldSphere.restype = cvoid_ptr
_treecorr.BuildKSimpleFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildKSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildKSimpleFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong ]
_treecorr.DestroyKSimpleFieldSphere.argtypes = [ cvoid_ptr ]
_treecorr.DestroyKSimpleFieldFlat.argtypes = [ cvoid_ptr ]

_treecorr.BuildGSimpleFieldSphere.restype = cvoid_ptr
_treecorr.BuildGSimpleFieldSphere.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong ]
_treecorr.BuildGSimpleFieldFlat.restype = cvoid_ptr
_treecorr.BuildGSimpleFieldFlat.argtypes = [
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr, clong ]
_treecorr.DestroyGSimpleFieldSphere.argtypes = [ cvoid_ptr ]
_treecorr.DestroyGSimpleFieldFlat.argtypes = [ cvoid_ptr ]


def parse_split_method(split_method):
    if split_method == 'middle': return cint(0)
    elif split_method == 'median': return cint(1)
    else: return cint(2)


class NField(object):
    """This class stores the positions in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep=None, max_sep=None, b=None, logger=None, config=None,
                 **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        self.logger.info('Building NField from cat %s',cat.name)

        split_method = self.config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%split_method)

        if min_sep is None:
            if 'min_sep' not in self.config:
                raise AttributeError("min_sep is required")
            min_sep = self.config['min_sep']
        if max_sep is None:
            if 'max_sep' not in self.config:
                raise AttributeError("min_sep is required")
            max_sep = self.config['max_sep']
        if b is None:
            if 'bin_size' not in self.config:
                raise AttributeError("b or bin_size is required")
            bin_size = self.config['bin_size']
            b = bin_size * self.config.get('bin_slop',1.)
            
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        w = cat.w.ctypes.data_as(cdouble_ptr)
        sm = parse_split_method(split_method)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            ra = cat.ra.ctypes.data_as(cdouble_ptr)
            dec = cat.dec.ctypes.data_as(cdouble_ptr)
            if cat.r is None:
                r = None
            else:
                r = cat.r.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNFieldSphere(ra,dec,r,w,cat.nobj,min_sep,max_sep,b,sm)
            self.logger.debug('Finished building NField Sphere')
        else:
            # Then build field with flat sky approximation
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNFieldFlat(x,y,w,cat.nobj,min_sep,max_sep,b,sm)
            self.logger.debug('Finished building NField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyNFieldSphere(self.data)
            else:
                _treecorr.DestroyNFieldFlat(self.data)


class KField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep=None, max_sep=None, b=None, logger=None, config=None,
                 **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        self.logger.info('Building KField from cat %s',cat.name)

        split_method = self.config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%split_method)

        if min_sep is None:
            if 'min_sep' not in self.config:
                raise AttributeError("min_sep is required")
            min_sep = self.config['min_sep']
        if max_sep is None:
            if 'max_sep' not in self.config:
                raise AttributeError("min_sep is required")
            max_sep = self.config['max_sep']
        if b is None:
            if 'bin_size' not in self.config:
                raise AttributeError("b or bin_size is required")
            bin_size = self.config['bin_size']
            b = bin_size * self.config.get('bin_slop',1.)
         
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        k = cat.k.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        sm = parse_split_method(split_method)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            ra = cat.ra.ctypes.data_as(cdouble_ptr)
            dec = cat.dec.ctypes.data_as(cdouble_ptr)
            if cat.r is None:
                r = None
            else:
                r = cat.r.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKFieldSphere(ra,dec,r,k,w,cat.nobj,min_sep,max_sep,b,sm)
            self.logger.debug('Finished building KField Sphere')
        else:
            # Then build field with flat sky approximation
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKFieldFlat(x,y,k,w,cat.nobj,min_sep,max_sep,b,sm)
            self.logger.debug('Finished building KField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyKFieldSphere(self.data)
            else:
                _treecorr.DestroyKFieldFlat(self.data)


class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep=None, max_sep=None, b=None, logger=None, config=None,
                 **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        self.logger.info('Building GField from cat %s',cat.name)

        split_method = self.config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%split_method)

        if min_sep is None:
            if 'min_sep' not in self.config:
                raise AttributeError("min_sep is required")
            min_sep = self.config['min_sep']
        if max_sep is None:
            if 'max_sep' not in self.config:
                raise AttributeError("min_sep is required")
            max_sep = self.config['max_sep']
        if b is None:
            if 'bin_size' not in self.config:
                raise AttributeError("b or bin_size is required")
            bin_size = self.config['bin_size']
            b = bin_size * self.config.get('bin_slop',1.)
         
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        g1 = cat.g1.ctypes.data_as(cdouble_ptr)
        g2 = cat.g2.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)
        sm = parse_split_method(split_method)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            ra = cat.ra.ctypes.data_as(cdouble_ptr)
            dec = cat.dec.ctypes.data_as(cdouble_ptr)
            if cat.r is None:
                r = None
            else:
                r = cat.r.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGFieldSphere(ra,dec,r,g1,g2,w,cat.nobj,min_sep,max_sep,b,sm)
            self.logger.debug('Finished building GField Sphere')
        else:
            # Then build field with flat sky approximation
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGFieldFlat(x,y,g1,g2,w,cat.nobj,min_sep,max_sep,b,sm)
            self.logger.debug('Finished building GField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyGFieldSphere(self.data)
            else:
                _treecorr.DestroyGFieldFlat(self.data)


class NSimpleField(object):
    """This class stores the positions as a list, skipping all the tree stuff.

    An NSimpleField is typically created from a Catalog object using

        >>> nfield = cat.getNSimpleField()
    """
    def __init__(self, cat, logger=None, config=None, **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        self.logger.info('Building NSimpleField from cat %s',cat.name)

        w = cat.w.ctypes.data_as(cdouble_ptr)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            ra = cat.ra.ctypes.data_as(cdouble_ptr)
            dec = cat.dec.ctypes.data_as(cdouble_ptr)
            if cat.r is None:
                r = None
            else:
                r = cat.r.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNSimpleFieldSphere(ra,dec,r,w,cat.nobj)
            self.logger.debug('Finished building NSimpleField Sphere')
        else:
            # Then build field with flat sky approximation
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildNSimpleFieldFlat(x,y,w,cat.nobj)
            self.logger.debug('Finished building NSimpleField Flat')



    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyNSimpleFieldSphere(self.data)
            else:
                _treecorr.DestroyNSimpleFieldFlat(self.data)


class KSimpleField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A KSimpleField is typically created from a Catalog object using

        >>> kfield = cat.getKSimpleField()
    """
    def __init__(self, cat, logger=None, config=None, **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        self.logger.info('Building KSimpleField from cat %s',cat.name)

        k = cat.k.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            ra = cat.ra.ctypes.data_as(cdouble_ptr)
            dec = cat.dec.ctypes.data_as(cdouble_ptr)
            if cat.r is None:
                r = None
            else:
                r = cat.r.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKSimpleFieldSphere(ra,dec,r,k,w,cat.nobj)
            self.logger.debug('Finished building KSimpleField Sphere')
        else:
            # Then build field with flat sky approximation
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildKSimpleFieldFlat(x,y,k,w,cat.nobj)
            self.logger.debug('Finished building KSimpleField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyKSimpleFieldSphere(self.data)
            else:
                _treecorr.DestroyKSimpleFieldFlat(self.data)


class GSimpleField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GSimpleField is typically created from a Catalog object using

        >>> gfield = cat.getGSimpleField()
    """
    def __init__(self, cat, logger=None, config=None, **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        self.logger.info('Building GSimpleField from cat %s',cat.name)

        g1 = cat.g1.ctypes.data_as(cdouble_ptr)
        g2 = cat.g2.ctypes.data_as(cdouble_ptr)
        w = cat.w.ctypes.data_as(cdouble_ptr)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            ra = cat.ra.ctypes.data_as(cdouble_ptr)
            dec = cat.dec.ctypes.data_as(cdouble_ptr)
            if cat.r is None:
                r = None
            else:
                r = cat.r.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGSimpleFieldSphere(ra,dec,r,g1,g2,w,cat.nobj)
            self.logger.debug('Finished building GSimpleField Sphere')
        else:
            # Then build field with flat sky approximation
            x = cat.x.ctypes.data_as(cdouble_ptr)
            y = cat.y.ctypes.data_as(cdouble_ptr)
            self.data = _treecorr.BuildGSimpleFieldFlat(x,y,g1,g2,w,cat.nobj)
            self.logger.debug('Finished building GSimpleField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyGSimpleFieldSphere(self.data)
            else:
                _treecorr.DestroyGSimpleFieldFlat(self.data)


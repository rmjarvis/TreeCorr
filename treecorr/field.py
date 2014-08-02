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
# 3. Neither the name of the {organization} nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

import treecorr

# Start by loading up the relevant C functions using ctypes
import numpy
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))


class NField(object):
    """This class stores the positions in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep, max_sep, b, logger=None, config=None):
        logger.info('Building NField from cat %s',cat.file_name)

        if config is None: config = {}
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(config.get('verbose',0),
                                                       config.get('log_file',None))

        split_method = config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%split_method)

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        # an alias
        double_ptr = ctypes.POINTER(ctypes.c_double)

        w = cat.w.ctypes.data_as(double_ptr)
        if split_method == 'middle':
            sm = ctypes.c_int(0)
        elif split_method == 'median':
            sm = ctypes.c_int(1)
        else:
            sm = ctypes.c_int(2)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            _treecorr.BuildNFieldSphere.restype = ctypes.c_void_p
            _treecorr.BuildNFieldSphere.argtypes = [
                double_ptr, double_ptr, double_ptr,
                ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int ]
            ra = cat.ra.ctypes.data_as(double_ptr)
            dec = cat.dec.ctypes.data_as(double_ptr)
            self.data = _treecorr.BuildNFieldSphere(ra,dec,w,cat.nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building NField Sphere')
        else:
            # Then build field with flat sky approximation
            _treecorr.BuildNFieldFlat.restype = ctypes.c_void_p
            _treecorr.BuildNFieldFlat.argtypes = [
                double_ptr, double_ptr, double_ptr,
                ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int ]
            x = cat.x.ctypes.data_as(double_ptr)
            y = cat.y.ctypes.data_as(double_ptr)
            self.data = _treecorr.BuildNFieldFlat(x,y,w,cat.nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building NField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyNFieldSphere.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyNFieldSphere(self.data)
            else:
                _treecorr.DestroyNFieldFlat.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyNFieldFlat(self.data)


class KField(object):
    """This class stores the kappa field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A KField is typically created from a Catalog object using

        >>> gfield = cat.getKField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep, max_sep, b, logger=None, config=None):
        logger.info('Building KField from cat %s',cat.file_name)
 
        if config is None: config = {}
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(config.get('verbose',0),
                                                       config.get('log_file',None))

        split_method = config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%split_method)

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        # an alias
        double_ptr = ctypes.POINTER(ctypes.c_double)

        k = cat.k.ctypes.data_as(double_ptr)
        w = cat.w.ctypes.data_as(double_ptr)
        if split_method == 'middle':
            sm = ctypes.c_int(0)
        elif split_method == 'median':
            sm = ctypes.c_int(1)
        else:
            sm = ctypes.c_int(2)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            _treecorr.BuildKFieldSphere.restype = ctypes.c_void_p
            _treecorr.BuildKFieldSphere.argtypes = [
                double_ptr, double_ptr, double_ptr, double_ptr,
                ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int ]
            ra = cat.ra.ctypes.data_as(double_ptr)
            dec = cat.dec.ctypes.data_as(double_ptr)
            self.data = _treecorr.BuildKFieldSphere(ra,dec,k,w,cat.nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building KField Sphere')
        else:
            # Then build field with flat sky approximation
            _treecorr.BuildKFieldFlat.restype = ctypes.c_void_p
            _treecorr.BuildKFieldFlat.argtypes = [
                double_ptr, double_ptr, double_ptr, double_ptr,
                ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int ]
            x = cat.x.ctypes.data_as(double_ptr)
            y = cat.y.ctypes.data_as(double_ptr)
            self.data = _treecorr.BuildKFieldFlat(x,y,k,w,cat.nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building KField Flat')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyKFieldSphere.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyKFieldSphere(self.data)
            else:
                _treecorr.DestroyKFieldFlat.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyKFieldFlat(self.data)


class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep, max_sep, b, logger=None, config=None):
        logger.info('Building GField from cat %s',cat.file_name)

        if config is None: config = {}
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(config.get('verbose',0),
                                                       config.get('log_file',None))

        split_method = config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%split_method)

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        # an alias
        double_ptr = ctypes.POINTER(ctypes.c_double)

        g1 = cat.g1.ctypes.data_as(double_ptr)
        g2 = cat.g2.ctypes.data_as(double_ptr)
        w = cat.w.ctypes.data_as(double_ptr)
        if split_method == 'middle':
            sm = ctypes.c_int(0)
        elif split_method == 'median':
            sm = ctypes.c_int(1)
        else:
            sm = ctypes.c_int(2)

        self.sphere = (cat.x is None)

        if self.sphere:
            # Then build field for spherical coordinates
            _treecorr.BuildGFieldSphere.restype = ctypes.c_void_p
            _treecorr.BuildGFieldSphere.argtypes = [
                double_ptr, double_ptr, double_ptr, double_ptr, double_ptr,
                ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int ]
            ra = cat.ra.ctypes.data_as(double_ptr)
            dec = cat.dec.ctypes.data_as(double_ptr)
            self.sphere = True
            self.data = _treecorr.BuildGFieldSphere(ra,dec,g1,g2,w,cat.nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building GField Sphere')
        else:
            # Then build field with flat sky approximation
            _treecorr.BuildGFieldFlat.restype = ctypes.c_void_p
            _treecorr.BuildGFieldFlat.argtypes = [
                double_ptr, double_ptr, double_ptr, double_ptr, double_ptr,
                ctypes.c_long, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int ]
            x = cat.x.ctypes.data_as(double_ptr)
            y = cat.y.ctypes.data_as(double_ptr)
            self.sphere = False
            self.data = _treecorr.BuildGFieldFlat(x,y,g1,g2,w,cat.nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building GField Flat')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.sphere:
                _treecorr.DestroyGFieldSphere.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyGFieldSphere(self.data)
            else:
                _treecorr.DestroyGFieldFlat.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyGFieldFlat(self.data)


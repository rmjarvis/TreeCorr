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

# Some aliases for my own benefit...
double_ptr = ctypes.POINTER(ctypes.c_double)

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))


class GField(object):
    """This class stores the shear field in a tree structure from which it is efficient
    to compute the two-point correlation functions.  

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_sep, max_sep, b)
    """
    def __init__(self, cat, min_sep, max_sep, b, logger=None, config=None):
        logger.info('Build GField from cat %s',cat.file_name)

        if config is None: config = {}
        if logger is not None:
            self.logger = logger
        else:
            self.logger = treecorr.config.setup_logger(config.get('verbose',0),
                                                       config.get('log_file',None))

        split_method = config.get('split_method','mean')
        if split_method not in ['middle', 'median', 'mean']:
            raise AttributeError("Invalid split_method %s"%split_method)

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.b = b
        self.split_method = split_method

        g1 = cat.g1.ctypes.data_as(double_ptr)
        g2 = cat.g2.ctypes.data_as(double_ptr)
        w = cat.w.ctypes.data_as(double_ptr)
        min_sep = ctypes.c_double(min_sep)
        max_sep = ctypes.c_double(max_sep)
        b = ctypes.c_double(b)
        if split_method == 'middle':
            sm = ctypes.c_int(0)
        elif split_method == 'median':
            sm = ctypes.c_int(1)
        else:
            sm = ctypes.c_int(2)
        nobj = ctypes.c_long(cat.nobj)

        if cat.x is not None:
            # Then build field with flat sky approximation
            _treecorr.BuildGFieldFlat.restype = ctypes.c_void_p
            x = cat.x.ctypes.data_as(double_ptr)
            y = cat.y.ctypes.data_as(double_ptr)
            self.coord = 'flat'
            self.data = _treecorr.BuildGFieldFlat(x,y,g1,g2,w,nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building GField Flat')
        else:
            # Then build field for spherical coordinates
            _treecorr.BuildGFieldSphere.restype = ctypes.c_void_p
            ra = cat.ra.ctypes.data_as(double_ptr)
            dec = cat.dec.ctypes.data_as(double_ptr)
            self.coord = 'sphere'
            self.data = _treecorr.BuildGFieldSphere(ra,dec,g1,g2,w,nobj,min_sep,max_sep,b,sm)
            logger.debug('Finished building GField Sphere')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        if hasattr(self,'data'):    # In case __init__ failed to get that far
            if self.coord == 'flat':
                _treecorr.DestroyGFieldFlat.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyGFieldFlat(self.data)
            else:
                _treecorr.DestroyGFieldSphere.argtypes = [ ctypes.c_void_p ]
                _treecorr.DestroyGFieldSphere(self.data)

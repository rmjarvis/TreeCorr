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

import os
import numpy
import ctypes
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))
_treecorr.SetOMPThreads.restype = ctypes.c_int
_treecorr.SetOMPThreads.argtypes = [ ctypes.c_int ]

def SetOMPThreads(num_threads):
    return _treecorr.SetOMPThreads(num_threads)

class BinnedCorr2(object):
    """This class stores the results of a 2-point correlation calculation, along with some
    ancillary data.

    This is a base class that is not intended to be constructed directly.  But it has a few
    helper functions that derived classes can use to help perform their calculations.  See
    the derived classes for more details:

        G2Correlation - handles shear-shear correlation functions
        N2Correlation - handles count-count correlation functions
        K2Correlation - handles kappa-kappa correlation functions
        NGCorrelation - handles count-shear correlation functions
        NKCorrelation - handles count-kappa correlation functions
        KGCorrelation - handles kappa-shear correlation functions

    Note that when we refer to kappa in the correlation function, that is because I
    come from a weak lensing perspective.  But really any scalar quantity may be used
    here.  CMB temperature fluctuations for example.

    The constructor for all derived classes take a config dict as the first argument,
    since this is often how we keep track of parameters, but if you don't want to 
    use one or if you want to change some parameters from what are in a config dict,
    then you can use normal kwargs, which take precedence over anything in the config dict.

    Exactly three of the following 4 parameters are required either in the config dict or
    in kwargs:

        nbins - How many bins to use
        bin_size - The width of the bins in log(separation)
        min_sep - The minimum separation; the left edge of the first bin
        max_sep - The maximum separation; the right edge of the last bin

    Any three of these may be provided.  The fourth number will be calculated from them.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        import math
        import numpy
        if kwargs:
            import copy
            config = copy.copy(config)
            config.update(kwargs)
        self.config = config

        if logger is None:
            self.logger = treecorr.config.setup_logger(self.config.get('verbose',0),
                                                       self.config.get('log_file',None))
        else:
            self.logger = logger

        self.output_dots = self.config.get('output_dots',False)

        if 'x_col' not in self.config and 'sep_units' not in self.config:
            raise AttributeError("sep_units is required if not using x_col,y_col")
        self.sep_units = self.config.get('sep_units','arcsec')
        self.sep_units = treecorr.angle_units[self.sep_units]
        self.log_sep_units = math.log(self.sep_units)
        if 'nbins' not in self.config:
            if 'max_sep' not in self.config:
                raise AttributeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config:
                raise AttributeError("Missing required parameter min_sep")
            if 'bin_size' not in self.config:
                raise AttributeError("Missing required parameter bin_size")
            self.min_sep = self.config['min_sep'] * self.sep_units
            self.max_sep = self.config['max_sep'] * self.sep_units
            self.bin_size = self.config['bin_size']
            self.nbins = int(math.ceil(math.log(self.max_sep/self.min_sep)/self.bin_size))
        elif 'bin_size' not in self.config:
            if 'max_sep' not in self.config:
                raise AttributeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config:
                raise AttributeError("Missing required parameter min_sep")
            self.min_sep = self.config['min_sep'] * self.sep_units
            self.max_sep = self.config['max_sep'] * self.sep_units
            self.nbins = self.config['nbins']
            self.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
        elif 'max_sep' not in self.config:
            if 'min_sep' not in self.config:
                raise AttributeError("Missing required parameter min_sep")
            self.min_sep = self.config['min_sep'] * self.sep_units
            self.nbins = self.config['nbins']
            self.bin_size = self.config['bin_size']
            self.max_sep = exp(self.nbins*self.bin_size)*self.min_sep
        else:
            if 'min_sep' in self.config:
                raise AttributeError("Only 3 of min_sep, max_sep, bin_size, nbins are allowed.")
            self.max_sep = self.config['max_sep'] * self.sep_units
            self.nbins = self.config['nbins']
            self.bin_size = self.config['bin_size']
            self.min_sep = self.max_sep*exp(-self.nbins*self.bin_size)
        self.logger.info("nbins = %d, min,max sep = %e..%e radians, bin_size = %f",
                         self.nbins,self.min_sep,self.max_sep,self.bin_size)

        self.bin_slop = self.config.get('bin_slop', 1.0)
        self.b = self.bin_size * self.bin_slop
        # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
        self.logr = numpy.linspace(start=0, stop=self.nbins*self.bin_size, 
                                   num=self.nbins, endpoint=False)
        # Offset by the position of the center of the first bin.
        self.logr += math.log(self.min_sep) + 0.5*self.bin_size

        # And correct the units:
        self.logr -= self.log_sep_units

        # All correlation functions use these, so go ahead and set them up here.
        self.meanlogr = numpy.zeros( (self.nbins, ) )
        self.varxi = numpy.zeros( (self.nbins, ) )
        self.weight = numpy.zeros( (self.nbins, ) )
        self.npairs = numpy.zeros( (self.nbins, ) )
        self.tot = 0

    def gen_write(self, file_name, headers, columns):
        """Write some columns to an output file with the given headers.

        We do this basic functionality a lot, so put the code to do it in one place.

        headers should be a list of strings.
        columns should be a list of numpy arrays.
        """
        import numpy
        if len(headers) != len(columns):
            raise ValueError("headers and columns are not the same length.")
    
        ncol = len(headers)
        output = numpy.empty( (self.nbins, ncol) )
        for i,col in enumerate(columns):
            output[:,i] = col

        prec = self.config.get('precision',3)
        width = prec+8
        header_form = "{:^%d}"%(width-1) + (ncol-1)*(".{:^%d}"%width)
        header = header_form.format(*headers)
        fmt = '%%%d.%de'%(width,prec)
        numpy.savetxt(file_name, output, fmt=fmt, header=header)

    def _process_all_auto(self, cat1):
        if self.config.get('do_auto_corr',False) or len(cat1) == 1:
            for c1 in cat1:
                self.process_auto(c1)
                self.tot += 0.5*c1.nobj**2

        if self.config.get('do_cross_corr',True):
            for i,c1 in enumerate(cat1):
                for c2 in cat1[i+1:]:
                    self.process_cross(c1,c2)
                    self.tot += c1.nobj*c2.nobj

    def _process_all_cross(self, cat1, cat2):
        if self.config.get('pairwise',False):
            if len(cat1) != len(cat2):
                raise RuntimeError("Number of files for 1 and 2 must be equal for pairwise.")
            for c1,c2 in zip(cat1,cat2):
                if c1.nobj != c2.nobj:
                    raise RuntimeError("Number of objects must be equal for pairwise.")
                self.process_pairwise(c1,c2)
                self.tot += c1.nobj
        else:
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2)
                    self.tot += c1.nobj*c2.nobj
 

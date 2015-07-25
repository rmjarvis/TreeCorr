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

"""
.. module:: binnedcorr3
"""

import treecorr

class BinnedCorr3(object):
    """This class stores the results of a 3-point correlation calculation, along with some
    ancillary data.

    This is a base class that is not intended to be constructed directly.  But it has a few
    helper functions that derived classes can use to help perform their calculations.  See
    the derived classes for more details:

    - :class:`~treecorr.NNNCorrelation` handles count-count-count correlation functions

    Three-point correlations are a bit more complicated than two-point, since the data need
    to be binned in triangles, not just the separation between two points.  We characterize the
    triangles according to the following three parameters based on the three side lenghts
    of the triangle with d1 >= d2 >= d3.

        r = d3
        u = d3 / d2
        v = +-(d1 - d2) / d3

    The orientation of the triangle is specified by the sign of v.  
    Positive v triangles have the three sides d1,d2,d3 in counter-clockwise orientation.
    Negative v triangles have the three sides d1,d2,d3 in clockwise orientation.

    The constructor for all derived classes take a config dict as the first argument,
    since this is often how we keep track of parameters, but if you don't want to 
    use one or if you want to change some parameters from what are in a config dict,
    then you can use normal kwargs, which take precedence over anything in the config dict.

    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    The following parameters may be given either in the config dict or as a named kwarg:

    :param nbins:       How many bins to use for the r binning. (Exactly three of nbins, bin_size,
                        min_sep, max_sep are required. If nbins is not given, it will be 
                        calculated from the values of the other three, rounding up to the next
                        highest integer. In this case, max_sep will be readjusted to account for
                        this rounding up.)
    :param bin_size:    The width of the bins in log(separation). (Exactly three of nbins, 
                        bin_size, min_sep, max_sep are required.  If bin_size is not given, it will
                        be calculated from the values of the other three.)
    :param min_sep:     The minimum separation in units of sep_units, if relevant. (Exactly three
                        of nbins, bin_size, min_sep, max_sep are required.  If min_sep is not
                        given, it will be calculated from the values of the other three.)
    :param max_sep:     The maximum separation in units of sep_units, if relevant. (Exactly three
                        of nbins, bin_size, min_sep, max_sep are required.  If max_sep is not
                        given, it will be calculated from the values of the other three.  If nbins
                        is not given, then max_sep will be adjusted as needed to allow nbins to be
                        an integer value.)
    :param sep_units:   The units to use for the separation values, given as a string.  This 
                        includes both min_sep and max_sep above, as well as the units of the 
                        output R column.  Valid options are arcsec, arcmin, degrees, hours,
                        radians.  (default: radians)
    :param bin_slop:    How much slop to allow in the placement of pairs in the bins.
                        If bin_slop = 1, then the bin into which a particular pair is placed may
                        be incorrect by at most 1.0 bin widths.  (default: None, which means to
                        use bin_slop=1 if bin_size <= 0.1, or 0.1/bin_size if bin_size > 0.1.
                        This mean the error will be at most 0.1 in log(sep), which has been found
                        to yield good results for most application.
    :param nubins:      Analogous to nbins for the u direction.  (The default is to calculate from
                        ubin_size = binsize, min_u = 0, max_u = 1, but this can be overridden by 
                        specifying up to 3 of these four parametes.)
    :param ubin_size:   Analogous to bin_size for the u direction. (default: bin_size)
    :param min_u:       Analogous to min_sep for the u direction. (default: 0)
    :param max_u:       Analogous to max_sep for the u direction. (default: 1)
    :param nvbins:      Analogous to nbins for the v direction.  (The default is to calculate from
                        vbin_size = binsize, min_v = -1, max_v = 1, but this can be overridden by 
                        specifying up to 3 of these four parametes.)
    :param vbin_size:   Analogous to bin_size for the v direction. (default: bin_size)
    :param min_v:       Analogous to min_sep for the v direction. (default: -1)
    :param max_v:       Analogous to max_sep for the v direction. (default: 1)
    :param verbose:     If no logger is provided, this will optionally specify a logging level to
                        use.
                        - 0 means no logging output (default)
                        - 1 means to output warnings only
                        - 2 means to output various progress information
                        - 3 means to output extensive debugging information
    :param log_file:    If no logger is provided, this will specify a file to write the logging
                        output.  (default: None; i.e. output to standard output)
    :param output_dots: Whether to output progress dots during the calcualtion of the correlation
                        function. (default: False unless verbose is given and >= 2, in which case
                        True)
    :param split_method:  How to split the cells in the tree when building the tree structure.
                        Options are:
                        - mean: Use the arithmetic mean of the coordinate being split. (default)
                        - median: Use the median of the coordinate being split.
                        - middle: Use the middle of the range; i.e. the average of the minimum and
                          maximum value.
    :param precision:   The precision to use for the output values. This should be an integer,
                        which specifies how many digits to write. (default: 4)
    :param num_threads: How many OpenMP threads to use during the calculations.  (default: 0,
                        which means to query the number of cpu cores and use that many threads.)
    """
    def __init__(self, config=None, logger=None, **kwargs):
        import math
        import numpy
        self.config = treecorr.config.merge_config(config,kwargs)
        if logger is None:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,0),
                    self.config.get('log_file',None))
        else:
            self.logger = logger

        if 'output_dots' in self.config:
            self.output_dots = treecorr.config.get(self.config,'output_dots',bool)
        elif 'verbose' in self.config:
            self.output_dots = treecorr.config.get(self.config,'verbose',int,0) >= 2
        else:
            self.output_dots = False

        self.sep_units = treecorr.config.get(self.config,'sep_units',str,'radians')
        self.log_sep_units = math.log(self.sep_units)
        if 'nbins' not in self.config:
            if 'max_sep' not in self.config:
                raise AttributeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config:
                raise AttributeError("Missing required parameter min_sep")
            if 'bin_size' not in self.config:
                raise AttributeError("Missing required parameter bin_size")
            self.min_sep = float(self.config['min_sep']) * self.sep_units
            self.max_sep = float(self.config['max_sep']) * self.sep_units
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self.bin_size = float(self.config['bin_size'])
            self.nbins = int(math.ceil(math.log(self.max_sep/self.min_sep)/self.bin_size))
            # Update max_sep given this value of nbins
            self.max_sep = math.exp(self.nbins*self.bin_size)*self.min_sep
        elif 'bin_size' not in self.config:
            if 'max_sep' not in self.config:
                raise AttributeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config:
                raise AttributeError("Missing required parameter min_sep")
            self.min_sep = float(self.config['min_sep']) * self.sep_units
            self.max_sep = float(self.config['max_sep']) * self.sep_units
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self.nbins = int(self.config['nbins'])
            self.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
        elif 'max_sep' not in self.config:
            if 'min_sep' not in self.config:
                raise AttributeError("Missing required parameter min_sep")
            self.min_sep = float(self.config['min_sep']) * self.sep_units
            self.nbins = int(self.config['nbins'])
            self.bin_size = float(self.config['bin_size'])
            self.max_sep = math.exp(self.nbins*self.bin_size)*self.min_sep
        else:
            if 'min_sep' in self.config:
                raise AttributeError("Only 3 of min_sep, max_sep, bin_size, nbins are allowed.")
            self.max_sep = float(self.config['max_sep']) * self.sep_units
            self.nbins = int(self.config['nbins'])
            self.bin_size = float(self.config['bin_size'])
            self.min_sep = self.max_sep*math.exp(-self.nbins*self.bin_size)
        self.logger.info("nbins = %d, min,max sep = %e..%e radians, bin_size = %f",
                         self.nbins,self.min_sep,self.max_sep,self.bin_size)

        if 'nubins' not in self.config:
            self.min_u = float(self.config.get('min_u', 0.))
            self.max_u = float(self.config.get('max_u', 1.))
            self.ubin_size = float(self.config.get('ubin_size', self.bin_size))
            if self.min_u >= self.max_u:
                raise ValueError("max_u must be larger than min_u")
            self.nubins = int(math.ceil((self.max_u-self.min_u)/self.ubin_size))
            self.min_u = self.max_u - self.nubins*self.ubin_size
            if self.min_u < 0.:
                self.min_u = 0.
                self.ubin_size = (self.max_u-self.min_u)/self.nubins
        elif 'ubin_size' not in self.config:
            self.min_u = float(self.config.get('min_u', 0.))
            self.max_u = float(self.config.get('max_u', 1.))
            if self.min_u >= self.max_u:
                raise ValueError("max_u must be larger than min_u")
            self.nubins = int(self.config['nubins'])
            self.ubin_size = (self.max_u-self.min_u)/self.nubins
        elif 'min_u' not in self.config:
            self.max_u = float(self.config.get('max_u', 1.))
            self.nubins = int(self.config['nubins'])
            self.ubin_size = float(self.config['ubin_size'])
            if self.ubin_size * (self.nubins-1) >= 1.:
                raise ValueError("Cannot specify ubin_size * nubins > 1.")
            self.min_u = self.max_u - self.nubins*self.ubin_size
            if self.min_u < 0.:
                self.min_u = 0.
                self.ubin_size = (self.max_u-self.min_u)/self.nubins
        else:
            if 'max_u' in self.config:
                raise AttributeError("Only 3 of min_u, max_u, ubin_size, nubins are allowed.")
            self.min_u = float(self.config['min_u'])
            self.nubins = int(self.config['nubins'])
            self.ubin_size = float(self.config['ubin_size'])
            if self.ubin_size * (self.nubins-1) >= 1.:
                raise ValueError("Cannot specify ubin_size * nubins > 1.")
            self.max_u = self.min_u + self.nubins*self.ubin_size
            if self.max_u > 1.: 
                self.max_u = 1.
                self.ubin_size = (self.max_u-self.min_u)/self.nubins

        if 'nvbins' not in self.config:
            self.min_v = float(self.config.get('min_v', -1.))
            self.max_v = float(self.config.get('max_v', 1.))
            self.vbin_size = float(self.config.get('vbin_size', self.bin_size))
            if self.min_v >= self.max_v:
                raise ValueError("max_v must be larger than min_v")
            self.nvbins = int(math.ceil((self.max_v-self.min_v)/self.vbin_size))
            # If one of min_v or max_v is specified, keep it exact.
            # Otherwise expand both values out as needed.  Also, make sure nvbins is even.
            if ('min_v' in self.config) == ('max_v' in self.config):
                if self.nvbins % 2 == 1: self.nvbins += 1
                cen = (self.min_v + self.max_v)/2.
                self.min_v = cen - self.nvbins*self.vbin_size/2.
                self.max_v = cen + self.nvbins*self.vbin_size/2.
            elif 'min_v' in config:
                self.max_v = self.min_v + self.nvbins*self.vbin_size
            else:
                self.min_v = self.max_v - self.nvbins*self.vbin_size
            if self.min_v < -1.: 
                self.min_v = -1.
            if self.max_v > 1.: 
                self.max_v = 1.
            self.vbin_size = (self.max_v-self.min_v)/self.nvbins
        elif 'vbin_size' not in self.config:
            self.min_v = float(self.config.get('min_v', -1.))
            self.max_v = float(self.config.get('max_v', 1.))
            if self.min_v >= self.max_v:
                raise ValueError("max_v must be larger than min_v")
            self.nvbins = int(self.config['nvbins'])
            self.vbin_size = (self.max_v-self.min_v)/self.nvbins
        elif 'min_v' not in self.config and 'max_v' not in self.config:
            self.nvbins = int(self.config['nvbins'])
            self.vbin_size = float(self.config['vbin_size'])
            if self.vbin_size * (self.nvbins-1) >= 1.:
                raise ValueError("Cannot specify vbin_size * nvbins > 1.")
            self.max_v = self.nvbins*self.vbin_size
            if self.max_v > 1.: self.max_v = 1.
            self.min_v = -self.max_v
            self.vbin_size = (self.max_v-self.min_v)/self.nvbins
        elif 'min_v' in self.config:
            if 'max_v' in self.config:
                raise AttributeError("Only 3 of min_v, max_v, vbin_size, nvbins are allowed.")
            self.min_v = float(self.config['min_v'])
            self.nvbins = int(self.config['nvbins'])
            self.vbin_size = float(self.config['vbin_size'])
            self.max_v = self.min_v + self.nvbins*self.vbin_size
            if self.max_v > 1.:
                raise ValueError("Cannot specify min_v + vbin_size * nvbins > 1.")
        else:
            self.max_v = float(self.config['max_v'])
            self.nvbins = int(self.config['nvbins'])
            self.vbin_size = float(self.config['vbin_size'])
            self.min_v = self.max_v - self.nvbins*self.vbin_size
            if self.min_v < -1.:
                raise ValueError("Cannot specify max_v - vbin_size * nvbins < -1.")

        self.split_method = self.config.get('split_method','mean')
        if self.split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%self.split_method)
        self.logger.debug("Using split_method = %s",self.split_method)

        self.bin_slop = treecorr.config.get(self.config,'bin_slop',float,-1.0)
        if self.bin_slop < 0.0:
            if self.bin_size <= 0.1:
                self.bin_slop = 1.0
                self.b = self.bin_size
            else:
                self.bin_slop = 0.1/self.bin_size  # The stored bin_slop corresponds to lnr bins.
                self.b = 0.1
            if self.ubin_size <= 0.1:
                self.bu = self.ubin_size
            else:
                self.bu = 0.1
            if self.vbin_size <= 0.1:
                self.bv = self.vbin_size
            else:
                self.bv = 0.1
        else:
            self.b = self.bin_size * self.bin_slop
            self.bu = self.ubin_size * self.bin_slop
            self.bv = self.vbin_size * self.bin_slop

        if self.b > 0.100001:  # Add some numerical slop
            self.logger.warn("Using bin_slop = %f, bin_size = %f",self.bin_slop,self.bin_size)
            self.logger.warn("The b parameter is bin_slop * bin_size = %f",self.b)
            self.logger.warn("It is generally recommended to use b <= 0.1 for most applications.")
            self.logger.warn("Larger values of this b parameter may result in significant"+
                             "inaccuracies.")
        else:
            self.logger.debug("Using bin_slop = %f, b = %f",self.bin_slop,self.b)
            self.logger.debug("bu = %f, bv = %f",self.bu,self.bv)

        # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
        self.logr = numpy.linspace(start=0, stop=self.nbins*self.bin_size, 
                                   num=self.nbins, endpoint=False)
        # Offset by the position of the center of the first bin.
        self.logr += math.log(self.min_sep) + 0.5*self.bin_size

        # And correct the units:
        self.logr -= self.log_sep_units

        self.u = numpy.linspace(start=0, stop=self.nubins*self.ubin_size, 
                                num=self.nubins, endpoint=False)
        self.u += self.min_u + 0.5*self.ubin_size

        self.v = numpy.linspace(start=0, stop=self.nvbins*self.vbin_size, 
                                num=self.nvbins, endpoint=False)
        self.v += self.min_v + 0.5*self.vbin_size


    gen_write = treecorr.util.gen_write
    gen_read = treecorr.util.gen_read

    def _process_all_auto(self, cat1):
        for c1 in cat1:
            self.process_auto(c1)
            for c2 in cat1:
                if c2 is not c1:
                    self.process_cross21(c1,c2)
                    for c3 in cat1:
                        if c3 is not c1 and c3 is not c2:
                            self.process_cross(c1,c2,c3)


    def _process_all_cross21(self, cat1, cat2):
        for c1 in cat1:
            for c2 in cat2:
                self.process_cross21(c1,c2)
            for c3 in cat1:
                if c3 is not c1:
                    self.process_cross(c1,c2,c3)

    def _process_all_cross(self, cat1, cat2, cat3):
        for c1 in cat1:
            for c2 in cat2:
                for c3 in cat3:
                    self.process_cross(c1,c2,c3)
 

    def _set_num_threads(self):
        num_threads = self.config.get('num_threads',None)
        self.logger.debug('Set num_threads = %d',num_threads)
        treecorr.set_omp_threads(num_threads, self.logger)

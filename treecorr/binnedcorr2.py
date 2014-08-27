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

class BinnedCorr2(object):
    """This class stores the results of a 2-point correlation calculation, along with some
    ancillary data.

    This is a base class that is not intended to be constructed directly.  But it has a few
    helper functions that derived classes can use to help perform their calculations.  See
    the derived classes for more details:

    - :class:`~treecorr.GGCorrelation` handles shear-shear correlation functions
    - :class:`~treecorr.NNCorrelation` handles count-count correlation functions
    - :class:`~treecorr.KKCorrelation` handles kappa-kappa correlation functions
    - :class:`~treecorr.NGCorrelation` handles count-shear correlation functions
    - :class:`~treecorr.NKCorrelation` handles count-kappa correlation functions
    - :class:`~treecorr.KGCorrelation` handles kappa-shear correlation functions

    Note that when we refer to kappa in the correlation function, that is because I
    come from a weak lensing perspective.  But really any scalar quantity may be used
    here.  CMB temperature fluctuations for example.

    The constructor for all derived classes take a config dict as the first argument,
    since this is often how we keep track of parameters, but if you don't want to 
    use one or if you want to change some parameters from what are in a config dict,
    then you can use normal kwargs, which take precedence over anything in the config dict.

    Exactly three of the following 4 parameters are required either in the config dict or
    in kwargs:

        :nbins:     How many bins to use
        :bin_size:  The width of the bins in log(separation)
        :min_sep:   The minimum separation; the left edge of the first bin
        :max_sep:   The maximum separation; the right edge of the last bin

    Any three of these may be provided.  The fourth number will be calculated from them.

    Note that if bin_size, min_sep, and max_sep are specified, then the nominal number of
    bins is not necessarily and integer.  In this case, nbins will be rounded up to the 
    next higher integer, and max_sep will be updated to account for this.

    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
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
            self.output_dots = treecorr.config.get(self.config,'verbose',int) >= 2
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

        self.split_method = self.config.get('split_method','mean')
        if self.split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%self.split_method)

        self.bin_slop = treecorr.config.get(self.config,'bin_slop',float,1.0)
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


    def gen_write(self, file_name, headers, columns):
        """Write some columns to an output file with the given headers.

        We do this basic functionality a lot, so put the code to do it in one place.

        :param file_name:   The name of the file to write to.
        :param headers:     A list of strings to use for the header strings of each column.
        :param columns:     A list of numpy arrays with the data to write.
        """
        import numpy
        if len(headers) != len(columns):
            raise ValueError("headers and columns are not the same length.")
    
        ncol = len(headers)
        data = numpy.empty( (self.nbins, ncol) )
        for i,col in enumerate(columns):
            data[:,i] = col

        prec = treecorr.config.get(self.config,'precision',int,4)
        width = prec+8
        # Note: python 2.6 needs the numbers, so can't just do "{:^%d}"*ncol
        # Also, I have the first one be 1 shorter to allow space for the initial #.
        header_form = "{0:^%d}"%(width-1)
        for i in range(1,ncol):
            header_form += " {%d:^%d}"%(i,width)
        header = header_form.format(*headers)
        fmt = '%%%d.%de'%(width,prec)
        try:
            numpy.savetxt(file_name, data, fmt=fmt, header=header)
        except (AttributeError, TypeError):
            # header was added with version 1.7, so do it by hand if not available.
            with open(file_name, 'w') as fid:
                fid.write('#' + header + '\n')
                numpy.savetxt(fid, data, fmt=fmt) 


    def _process_all_auto(self, cat1):
        for c1 in cat1:
            self.process_auto(c1)

        for i,c1 in enumerate(cat1):
            for c2 in cat1[i+1:]:
                self.process_cross(c1,c2)


    def _process_all_cross(self, cat1, cat2):
        if treecorr.config.get(self.config,'pairwise',bool,False):
            if len(cat1) != len(cat2):
                raise RuntimeError("Number of files for 1 and 2 must be equal for pairwise.")
            for c1,c2 in zip(cat1,cat2):
                if c1.nobj != c2.nobj:
                    raise RuntimeError("Number of objects must be equal for pairwise.")
                self.process_pairwise(c1,c2)
        else:
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2)
 

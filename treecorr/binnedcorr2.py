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
.. module:: binnedcorr2
"""

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

    :param config:      A configuration dict that can be used to pass in the below kwargs if
                        desired.  This dict is allowed to have addition entries in addition
                        to those listed below, which are ignored here. (default: NoneP
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    The following parameters may be given either in the config dict or as a named kwarg:

    :param nbins:       How many bins to use. (Exactly three of nbins, bin_size, min_sep, max_sep
                        are required. If nbins is not given, it will be calculated from the values
                        of the other three, rounding up to the next highest integer. In this case,
                        max_sep will be readjusted to account for this rounding up.)
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

    :param verbose:     If no logger is provided, this will optionally specify a logging level to
                        use:

                        - 0 means no logging output (default)
                        - 1 means to output warnings only
                        - 2 means to output various progress information
                        - 3 means to output extensive debugging information

    :param log_file:    If no logger is provided, this will specify a file to write the logging
                        output.  (default: None; i.e. output to standard output)
    :param output_dots: Whether to output progress dots during the calcualtion of the correlation
                        function. (default: False unless verbose is given and >= 2, in which case
                        True)

    :param split_method: How to split the cells in the tree when building the tree structure.
                        Options are:

                        - mean = Use the arithmetic mean of the coordinate being split. (default)
                        - median = Use the median of the coordinate being split.
                        - middle = Use the middle of the range; i.e. the average of the minimum and
                          maximum value.

    :param max_top:     The maximum number of top layers to use when setting up the field. 
                        The top-level cells are the cells where each calculation job starts.
                        There will typically be of order 2^max_top top-level cells. (default: 10)
    :param precision:   The precision to use for the output values. This should be an integer,
                        which specifies how many digits to write. (default: 4)
    :param pairwise:    Whether to use a different kind of calculation for cross correlations
                        whereby corresponding items in the two catalogs are correlated pairwise
                        rather than the usual case of every item in one catalog being correlated
                        with every item in the other catalog. (default: False)
    :param m2_uform:    The default functional form to use for aperture mass calculations.  See
                        :GGCorrelation.calculateMapSq: for more details. (default: 'Crittenden')
    :param metric:      Which metric to use for distance measurements.  Options are:

                        - 'Euclidean' = straight line Euclidean distance between two points.
                          For spherical coordinates (ra,dec without r), this is the chord
                          distance between points on the unit sphere.
                        - 'Rperp' = the perpendicular component of the distance. For two points
                          with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                          distance and :math:`Rparallel = |r1-r2|`, then we define
                          :math:`Rperp^2 = d^2 - Rparallel^2`.

                        (default: 'Euclidean')

    :param num_threads: How many OpenMP threads to use during the calculation.  
                        (default: use the number of cpu cores; this value can also be given in
                        the constructor in the config dict.) Note that this won't work if the 
                        system's C compiler is clang prior to version 3.7.
    """
    _valid_params = {
        'nbins' : (int, False, None, None,
                'The number of output bins to use.'),
        'bin_size' : (float, False, None, None,
                'The size of the output bins in log(sep).'),
        'min_sep' : (float, False, None, None,
                'The minimum separation to include in the output.'),
        'max_sep' : (float, False, None, None,
                'The maximum separation to include in the output.'),
        'sep_units' : (str, False, None, treecorr.angle_units.keys(),
                'The units to use for min_sep and max_sep.  Also the units of the output r columns'),
        'bin_slop' : (float, False, None, None,
                'The fraction of a bin width by which it is ok to let the pairs miss the correct bin.',
                'The default is to use 1 if bin_size <= 0.1, or 0.1/bin_size if bin_size > 0.1.'),
        'verbose' : (int, False, 1, [0, 1, 2, 3],
                'How verbose the code should be during processing. ',
                '0 = Errors Only, 1 = Warnings, 2 = Progress, 3 = Debugging'),
        'log_file' : (str, False, None, None,
                'If desired, an output file for the logging output.',
                'The default is to write the output to stdout.'),
        'output_dots' : (bool, False, None, None,
                'Whether to output dots to the stdout during the C++-level computation.',
                'The default is True if verbose >= 2 and there is no log_file.  Else False.'),
        'split_method' : (str, False, 'mean', ['mean', 'median', 'middle'],
                'Which method to use for splitting cells.'),
        'max_top' : (int, False, 10, None,
                'The maximum number of top layers to use when setting up the field.'),
        'precision' : (int, False, 4, None,
                'The number of digits after the decimal in the output.'),
        'pairwise' : (bool, True, False, None,
                'Whether to do a pair-wise cross-correlation '),
        'num_threads' : (int, False, None, None,
                'How many threads should be used. num_threads <= 0 means auto based on num cores.'),
        'm2_uform' : (str, False, 'Crittenden', ['Crittenden', 'Schneider'],
                'The function form of the mass aperture.'),
        'metric': (str, False, 'Euclidean', ['Euclidean', 'Rperp'],
                'Which metric to use for the distance measurements'),
    }
    def __init__(self, config=None, logger=None, **kwargs):
        import math
        import numpy
        self.config = treecorr.config.merge_config(config,kwargs,BinnedCorr2._valid_params)
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
        self.sep_unit_name = self.config.get('sep_units','')
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
        if self.sep_unit_name == '':
            self.logger.info("nbins = %d, min,max sep = %g..%g, bin_size = %g",
                             self.nbins,self.min_sep,self.max_sep,self.bin_size)
        else:
            self.logger.info("nbins = %d, min,max sep = %g..%g %s, bin_size = %g",
                             self.nbins,self.min_sep/self.sep_units,self.max_sep/self.sep_units,
                             self.sep_unit_name,self.bin_size)

        self.split_method = self.config.get('split_method','mean')
        if self.split_method not in ['middle', 'median', 'mean']:
            raise ValueError("Invalid split_method %s"%self.split_method)
        self.logger.debug("Using split_method = %s",self.split_method)

        self.max_top = treecorr.config.get(self.config,'max_top',int,10)

        self.bin_slop = treecorr.config.get(self.config,'bin_slop',float,-1.0)
        if self.bin_slop < 0.0:
            if self.bin_size <= 0.1:
                self.bin_slop = 1.0
            else:
                self.bin_slop = 0.1/self.bin_size
        self.b = self.bin_size * self.bin_slop
        if self.b > 0.100001:  # Add some numerical slop
            self.logger.warn("Using bin_slop = %g, bin_size = %g",self.bin_slop,self.bin_size)
            self.logger.warn("The b parameter is bin_slop * bin_size = %g",self.b)
            self.logger.warn("It is generally recommended to use b <= 0.1 for most applications.")
            self.logger.warn("Larger values of this b parameter may result in significant"+
                             "inaccuracies.")
        else:
            self.logger.debug("Using bin_slop = %g, b = %g",self.bin_slop,self.b)

        # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
        self.logr = numpy.linspace(start=0, stop=self.nbins*self.bin_size, 
                                   num=self.nbins, endpoint=False)
        # Offset by the position of the center of the first bin.
        self.logr += math.log(self.min_sep) + 0.5*self.bin_size

        # And correct the units:
        self.logr -= self.log_sep_units


    def _process_all_auto(self, cat1, metric, num_threads):
        for i,c1 in enumerate(cat1):
            self.process_auto(c1,metric,num_threads)
            for c2 in cat1[i+1:]:
                self.process_cross(c1,c2,metric,num_threads)


    def _process_all_cross(self, cat1, cat2, metric, num_threads):
        if treecorr.config.get(self.config,'pairwise',bool,False):
            if len(cat1) != len(cat2):
                raise RuntimeError("Number of files for 1 and 2 must be equal for pairwise.")
            for c1,c2 in zip(cat1,cat2):
                if c1.ntot != c2.ntot:
                    raise RuntimeError("Number of objects must be equal for pairwise.")
                self.process_pairwise(c1,c2,metric,num_threads)
        else:
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2,metric,num_threads)
 

    def _set_num_threads(self, num_threads):
        if num_threads is None:
            num_threads = self.config.get('num_threads',None)
        # Recheck.
        if num_threads is None:
            self.logger.debug('Set num_threads automatically')
        else:
            self.logger.debug('Set num_threads = %d',num_threads)
        treecorr.set_omp_threads(num_threads, self.logger)

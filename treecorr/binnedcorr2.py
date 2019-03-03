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
import math
import numpy
import sys

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

    There are a number of possible definitions for the distance between two points, which
    are appropriate for different use cases.  These are specified by the :metric: parameter.
    The possible options are:

        - 'Euclidean' = straight line Euclidean distance between two points.  For spherical
          coordinates (ra,dec without r), this is the chord distance between points on the
          unit sphere.
        - 'FisherRperp' = the perpendicular component of the distance, following the
          definitions in Fisher et al, 1994 (MNRAS, 267, 927). For two points with vector
          positions from Earth `r1, r2`, if :math:`r` is the vector :math:`r2-r1` and
          :math:`L = (r1+r2)/2`, then we take :math:`Rpar = L \cdot r / |L|` and
          :math:`Rperp^2 = d^2 - Rpar^2`.
        - 'OldRperp' = the perpendicular component of the distance. For two points with
          distance from Earth `r1, r2`, if `d` is the normal Euclidean distance, then we
          take :math:`Rpar = r2-r1` and :math:`Rperp^2 = d^2 - Rpar^2`.
        - 'Rperp' is currently an alias for OldRperp.  In version 4.0, it will switch to
          being equivalent to FisherRperp.
        - 'Rlens' = the distance from the first object (taken to be a lens) to the line
          connecting Earth and the second object (taken to be a lensed source).
        - 'Arc' = the true great circle distance for spherical coordinates.

    There are also a few different possibile binning prescriptions to define the range of
    distances, which should be placed into each bin.

        - 'Log' - logarithmic binning in the distance.  The bin steps will be uniform in
          log(r) from log(min_sep) .. log(max_sep).
        - 'Linear' - linear binning in the distance.  The bin steps will be uniform in r
          from min_sep .. max_sep.
        - 'TwoD' = 2-dimensional binning from x = (-max_sep .. max_sep) and
          y = (-max_sep .. max_sep).  The bin steps will be uniform in both x and y.
          (i.e. linear in x,y)


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
                        output distance values.  Valid options are arcsec, arcmin, degrees, hours,
                        radians.  (default: radians if angular units make sense, but for 3-d
                        or flat 2-d positions, the default will just match the units of x,y[,z]
                        coordinates)
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
                        - random: Use a random point somewhere in the middle two quartiles of the
                          range.

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

    :param metric:      Which metric to use for distance measurements.  Options are listed above.
                        (default: 'Euclidean')
    :param bin_type:    What type of binning should be used.  Options are listed above.
                        (default: 'Log')
    :param min_rpar:    For the 'Rperp' metric, the minimum difference in Rparallel to allow
                        for pairs being included in the correlation function. (default: None)
    :param max_rpar:    For the 'Rperp' metric, the maximum difference in Rparallel to allow
                        for pairs being included in the correlation function. (default: None)

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
                'The units to use for min_sep and max_sep.  Also the units of the output distances'),
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
        'split_method' : (str, False, 'mean', ['mean', 'median', 'middle', 'random'],
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
        'metric': (str, False, 'Euclidean', ['Euclidean', 'Rperp', 'FisherRperp', 'OldRperp',
                                             'Rlens', 'Arc'],
                'Which metric to use for the distance measurements'),
        'bin_type': (str, False, 'Log', ['Log', 'Linear', 'TwoD'],
                'Which type of binning should be used'),
        'min_rpar': (float, False, None, None,
                'For Rperp metric, the minimum difference in Rparallel for pairs to include'),
        'max_rpar': (float, False, None, None,
                'For Rperp metric, the maximum difference in Rparallel for pairs to include'),
    }

    def __init__(self, config=None, logger=None, **kwargs):
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

        bin_type = self.config.get('bin_type', None)

        self.sep_units = treecorr.config.get(self.config,'sep_units',str,'radians')
        self.sep_unit_name = self.config.get('sep_units','')
        self.log_sep_units = math.log(self.sep_units)
        if 'nbins' not in self.config:
            if 'max_sep' not in self.config:
                raise AttributeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config and bin_type != 'TwoD':
                raise AttributeError("Missing required parameter min_sep")
            if 'bin_size' not in self.config:
                raise AttributeError("Missing required parameter bin_size")
            self.min_sep = float(self.config.get('min_sep',0))
            self.max_sep = float(self.config['max_sep'])
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self.bin_size = float(self.config['bin_size'])
            self.nbins = None
        elif 'bin_size' not in self.config:
            if 'max_sep' not in self.config:
                raise AttributeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config and bin_type != 'TwoD':
                raise AttributeError("Missing required parameter min_sep")
            self.min_sep = float(self.config.get('min_sep',0))
            self.max_sep = float(self.config['max_sep'])
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self.nbins = int(self.config['nbins'])
            self.bin_size = None
        elif 'max_sep' not in self.config:
            if 'min_sep' not in self.config and bin_type != 'TwoD':
                raise AttributeError("Missing required parameter min_sep")
            self.min_sep = float(self.config.get('min_sep',0))
            self.nbins = int(self.config['nbins'])
            self.bin_size = float(self.config['bin_size'])
            self.max_sep = None
        else:
            if bin_type == 'TwoD':
                raise AttributeError("Only 2 of max_sep, bin_size, nbins are allowed "
                                     "for bin_type='TwoD'.")
            if 'min_sep' in self.config:
                raise AttributeError("Only 3 of min_sep, max_sep, bin_size, nbins are allowed.")
            self.max_sep = float(self.config['max_sep'])
            self.nbins = int(self.config['nbins'])
            self.bin_size = float(self.config['bin_size'])
            self.min_sep = None

        if bin_type == 'Log':
            if self.nbins is None:
                self.nbins = int(math.ceil(math.log(self.max_sep/self.min_sep)/self.bin_size))
                # Update max_sep given this value of nbins
                self.max_sep = math.exp(self.nbins*self.bin_size)*self.min_sep
            elif self.bin_size is None:
                self.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
            elif self.max_sep is None:
                self.max_sep = math.exp(self.nbins*self.bin_size)*self.min_sep
            else:
                self.min_sep = self.max_sep*math.exp(-self.nbins*self.bin_size)

            # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
            self.logr = numpy.linspace(0, self.nbins*self.bin_size, self.nbins, endpoint=False,
                                       dtype=float)
            # Offset by the position of the center of the first bin.
            self.logr += math.log(self.min_sep) + 0.5*self.bin_size
            self.rnom = numpy.exp(self.logr)
            self._nbins = self.nbins
            self._bintype = treecorr._lib.Log
            target_max_b = 0.1
            bwarning_text = "b <= 0.1"
        elif bin_type == 'Linear':
            if self.nbins is None:
                self.nbins = int(math.ceil((self.max_sep-self.min_sep)/self.bin_size))
                # Update max_sep given this value of nbins
                self.max_sep = self.min_sep + self.nbins*self.bin_size
            elif self.bin_size is None:
                self.bin_size = (self.max_sep-self.min_sep)/self.nbins
            elif self.max_sep is None:
                self.max_sep = self.min_sep + self.nbins*self.bin_size
            else:
                self.min_sep = self.max_sep - self.nbins*self.bin_size

            self.rnom = numpy.linspace(self.min_sep, self.max_sep, self.nbins, endpoint=False,
                                       dtype=float)
            # Offset by the position of the center of the first bin.
            self.rnom += 0.5*self.bin_size
            self.logr = numpy.log(self.rnom)
            self._nbins = self.nbins
            self._bintype = treecorr._lib.Linear
            target_max_b = 0.1 * self.bin_size
            bwarning_text = "bin_slop <= 0.1"
        elif bin_type == 'TwoD':
            if self.nbins is None:
                self.nbins = int(math.ceil(2.*self.max_sep / self.bin_size))
                self.max_sep = self.nbins * self.bin_size / 2.
            elif self.bin_size is None:
                self.bin_size = 2.*self.max_sep/self.nbins
            else:
                self.max_sep = self.nbins * self.bin_size / 2.

            sep = numpy.linspace(-self.max_sep, self.max_sep, self.nbins, endpoint=False,
                                 dtype=float)
            sep += 0.5 * self.bin_size
            self.dx, self.dy = numpy.meshgrid(sep, sep)
            self.rnom = numpy.sqrt(self.dx**2 + self.dy**2)
            self.logr = numpy.zeros_like(self.rnom)
            numpy.log(self.rnom, out=self.logr, where=self.rnom > 0)
            self.logr[self.rnom==0.] = -numpy.inf
            self._nbins = self.nbins**2
            self._bintype = treecorr._lib.TwoD
            target_max_b = 0.1 * self.bin_size
            bwarning_text = "bin_slop <= 0.1"
        else:
            raise ValueError("Invalid bin_type %s")

        if self.sep_unit_name == '':
            self.logger.info("nbins = %d, min,max sep = %g..%g, bin_size = %g",
                             self.nbins, self.min_sep, self.max_sep, self.bin_size)
        else:
            self.logger.info("nbins = %d, min,max sep = %g..%g %s, bin_size = %g",
                             self.nbins, self.min_sep, self.max_sep, self.sep_unit_name,
                             self.bin_size)
        # The underscore-prefixed names are in natural units (radians for angles)
        self._min_sep = self.min_sep * self.sep_units
        self._max_sep = self.max_sep * self.sep_units

        self.split_method = self.config.get('split_method','mean')
        if self.split_method not in ['middle', 'median', 'mean', 'random']:
            raise ValueError("Invalid split_method %s"%self.split_method)
        self.logger.debug("Using split_method = %s",self.split_method)

        self.max_top = treecorr.config.get(self.config,'max_top',int,10)

        self.bin_slop = treecorr.config.get(self.config,'bin_slop',float,-1.0)
        if self.bin_slop < 0.0:
            if self.bin_size <= target_max_b:
                self.bin_slop = 1.0
            else:
                self.bin_slop = target_max_b/self.bin_size
        self.b = self.bin_size * self.bin_slop
        if self.b > target_max_b * 1.0001:  # Add some numerical slop
            self.logger.warning(
                    "Using bin_slop = %g, bin_size = %g\n"%(self.bin_slop,self.bin_size)+
                    "The b parameter is bin_slop * bin_size = %g\n"%(self.b)+
                    "It is generally recommended to use %s for most applications.\n"%bwarning_text+
                    "Larger values of bin_slop may result in significant inaccuracies.")
        else:
            self.logger.debug("Using bin_slop = %g, b = %g",self.bin_slop,self.b)

        self._coords = None
        self._metric = None
        self.min_rpar = treecorr.config.get(self.config,'min_rpar',float,-sys.float_info.max)
        self.max_rpar = treecorr.config.get(self.config,'max_rpar',float,sys.float_info.max)

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

    def _set_metric(self, metric, coords1, coords2=None):
        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        coords, metric = treecorr.util.parse_metric(metric, coords1, coords2)
        if self._coords != None or self._metric != None:
            if coords != self._coords:
                self.logger.warning("Detected a change in catalog coordinate systems.\n"+
                                    "This probably doesn't make sense!")
            if metric != self._metric:
                self.logger.warning("Detected a change in metric.\n"+
                                    "This probably doesn't make sense!")
        if metric not in [treecorr._lib.Rperp, treecorr._lib.OldRperp, treecorr._lib.Rlens]:
            if self.min_rpar != -sys.float_info.max:
                raise ValueError("min_rpar is only valid with either Rlens or Rperp metric.")
            if self.max_rpar != sys.float_info.max:
                raise ValueError("max_rpar is only valid with either Rlens or Rperp metric.")
        else:
            if metric == 'Rperp':
                self.logger.warning(
                    "WARNING: The definition of Rperp will change in version 4.0\n"
                    "to match the definition in Fisher et al, 1994.\n"
                    "The new definition can be used now with metric='FisherRperp'.\n"
                    "After 4.0, the current Rperp will be available as metric='OldRperp'.\n")
            if self.sep_units != 1.:
                raise ValueError("sep_units is invalid with either Rlens or Rperp metric. "+
                                 "min_sep and max_sep should be in the same units as r (or x,y,z)")
        self._coords = coords
        self._metric = metric

    def _apply_units(self, mask):
        if self._coords == treecorr._lib.Sphere and self._metric == treecorr._lib.Euclidean:
            # Then our distances are all angles.  Convert from the chord distance to a real angle.
            # L = 2 sin(theta/2)
            self.meanr[mask] = 2. * numpy.arcsin(self.meanr[mask]/2.)
            self.meanlogr[mask] = numpy.log( 2. * numpy.arcsin(numpy.exp(self.meanlogr[mask])/2.) )
        self.meanr[mask] /= self.sep_units
        self.meanlogr[mask] -= self.log_sep_units

    def _get_minmax_size(self):
        if self._metric == treecorr._lib.Euclidean:
            # The minimum size cell that will be useful is one where two cells that just barely
            # don't split have (d + s1 + s2) = minsep
            # The largest s2 we need to worry about is s2 = 2s1.
            # i.e. d = minsep - 3s1  and s1 = 0.5 * bd
            #      d = minsep - 1.5 bd
            #      d = minsep / (1+1.5 b)
            #      s = 0.5 * b * minsep / (1+1.5 b)
            #        = b * minsep / (2+3b)
            min_size = self._min_sep * self.b / (2.+3.*self.b)

            # The maximum size cell that will be useful is one where a cell of size s will
            # be split at the maximum separation even if the other size = 0.
            # i.e. max_size = max_sep * b
            max_size = self._max_sep * self.b
            return min_size, max_size
        else:
            # For other metrics, the above calculation doesn't really apply, so just skip
            # this relatively modest optimization and go all the way to the leaves.
            # (And for the max_size, always split 10 levels for the top-level cells.)
            return 0., 0.


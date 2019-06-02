# Copyright (c) 2003-2019 by Mike Jarvis
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

import math
import numpy as np
import sys
import coord
import treecorr

class BinnedCorr2(object):
    """This class stores the results of a 2-point correlation calculation, along with some
    ancillary data.

    This is a base class that is not intended to be constructed directly.  But it has a few
    helper functions that derived classes can use to help perform their calculations.  See
    the derived classes for more details:

    - `GGCorrelation` handles shear-shear correlation functions
    - `NNCorrelation` handles count-count correlation functions
    - `KKCorrelation` handles kappa-kappa correlation functions
    - `NGCorrelation` handles count-shear correlation functions
    - `NKCorrelation` handles count-kappa correlation functions
    - `KGCorrelation` handles kappa-shear correlation functions

    Note that when we refer to kappa in the correlation function, that is because I
    come from a weak lensing perspective.  But really any scalar quantity may be used
    here.  CMB temperature fluctuations for example.

    The constructor for all derived classes take a config dict as the first argument,
    since this is often how we keep track of parameters, but if you don't want to
    use one or if you want to change some parameters from what are in a config dict,
    then you can use normal kwargs, which take precedence over anything in the config dict.

    There are a number of possible definitions for the distance between two points, which
    are appropriate for different use cases.  These are specified by the **metric** parameter.
    The possible options are:

        - 'Euclidean' = straight line Euclidean distance between two points.
        - 'FisherRperp' = the perpendicular component of the distance, following the
          definitions in Fisher et al, 1994 (MNRAS, 267, 927).
        - 'OldRperp' = the perpendicular component of the distance using the definition
          of Rperp from TreeCorr v3.x.
        - 'Rperp' = an alias for FisherRperp.  You can change it to be an alias for
          OldRperp if you want by setting ``treecorr.Rperp_alias = 'OldRperp'`` before
          using it.
        - 'Rlens' = the distance from the first object (taken to be a lens) to the line
          connecting Earth and the second object (taken to be a lensed source).
        - 'Arc' = the true great circle distance for spherical coordinates.
        - 'Periodic' = Like Euclidean, but with periodic boundaries.

    See `Metrics` for more information about these various metric options.

    There are also a few different possibile binning prescriptions to define the range of
    distances, which should be placed into each bin.

        - 'Log' - logarithmic binning in the distance.  The bin steps will be uniform in
          log(r) from log(min_sep) .. log(max_sep).
        - 'Linear' - linear binning in the distance.  The bin steps will be uniform in r
          from min_sep .. max_sep.
        - 'TwoD' = 2-dimensional binning from x = (-max_sep .. max_sep) and
          y = (-max_sep .. max_sep).  The bin steps will be uniform in both x and y.
          (i.e. linear in x,y)

    See `Binning` for more information about the different binning options.

    Parameters:
        config (dict):      A configuration dict that can be used to pass in the below kwargs if
                            desired.  This dict is allowed to have addition entries in addition
                            to those listed below, which are ignored here. (default: None)
        logger:             If desired, a logger object for logging. (default: None, in which case
                            one will be built according to the config dict's verbose level.)

    Keyword Arguments:

        nbins (int):        How many bins to use. (Exactly three of nbins, bin_size, min_sep,
                            max_sep are required. If nbins is not given, it will be calculated from
                            the values of the other three, rounding up to the next highest integer.
                            In this case, bin_size will be readjusted to account for this rounding
                            up.)
        bin_size (float):   The width of the bins in log(separation). (Exactly three of nbins,
                            bin_size, min_sep, max_sep are required.  If bin_size is not given, it
                            will be calculated from the values of the other three.)
        min_sep (float):    The minimum separation in units of sep_units, if relevant. (Exactly
                            three of nbins, bin_size, min_sep, max_sep are required.  If min_sep is
                            not given, it will be calculated from the values of the other three.)
        max_sep (float):    The maximum separation in units of sep_units, if relevant. (Exactly
                            three of nbins, bin_size, min_sep, max_sep are required.  If max_sep is
                            not given, it will be calculated from the values of the other three.

        sep_units (str):    The units to use for the separation values, given as a string.  This
                            includes both min_sep and max_sep above, as well as the units of the
                            output distance values.  Valid options are arcsec, arcmin, degrees,
                            hours, radians.  (default: radians if angular units make sense, but for
                            3-d or flat 2-d positions, the default will just match the units of
                            x,y[,z] coordinates)
        bin_slop (float):   How much slop to allow in the placement of pairs in the bins.
                            If bin_slop = 1, then the bin into which a particular pair is placed
                            may be incorrect by at most 1.0 bin widths.  (default: None, which
                            means to use a bin_slop that gives a maximum error of 10% on any bin,
                            which has been found to yield good results for most application.
        brute (bool):       Whether to use the "brute force" algorithm.  (default: False) Options
                            are:

                             - False (the default): Stop at non-leaf cells whenever the error in
                               the separation is compatible with the given bin_slop.
                             - True: Go to the leaves for both catalogs.
                             - 1: Always go to the leaves for cat1, but stop at non-leaf cells of
                               cat2 when the error is compatible with the given bin_slop.
                             - 2: Always go to the leaves for cat2, but stop at non-leaf cells of
                               cat1 when the error is compatible with the given bin_slop.

        verbose (int):      If no logger is provided, this will optionally specify a logging level
                            to use:

                             - 0 means no logging output
                             - 1 means to output warnings only (default)
                             - 2 means to output various progress information
                             - 3 means to output extensive debugging information

        log_file (str):     If no logger is provided, this will specify a file to write the logging
                            output.  (default: None; i.e. output to standard output)
        output_dots (boo):  Whether to output progress dots during the calcualtion of the
                            correlation function. (default: False unless verbose is given and >= 2,
                            in which case True)

        split_method (str): How to split the cells in the tree when building the tree structure.
                            Options are:

                            - mean = Use the arithmetic mean of the coordinate being split.
                              (default)
                            - median = Use the median of the coordinate being split.
                            - middle = Use the middle of the range; i.e. the average of the minimum
                              and maximum value.
                            - random: Use a random point somewhere in the middle two quartiles of
                              the range.

        min_top (int):      The minimum number of top layers to use when setting up the field.
                            (default: 3)
        max_top (int):      The maximum number of top layers to use when setting up the field.
                            The top-level cells are where each calculation job starts. There will
                            typically be of order 2^max_top top-level cells. (default: 10)
        precision (int):    The precision to use for the output values. This specifies how many
                            digits to write. (default: 4)
        pairwise (bool):    Whether to use a different kind of calculation for cross correlations
                            whereby corresponding items in the two catalogs are correlated pairwise
                            rather than the usual case of every item in one catalog being correlated
                            with every item in the other catalog. (default: False)
        m2_uform (str):     The default functional form to use for aperture mass calculations.
                            see `calculateMapSq` for more details.  (default: 'Crittenden')

        metric (str):       Which metric to use for distance measurements.  Options are listed
                            above.  (default: 'Euclidean')
        bin_type (str):     What type of binning should be used.  Options are listed above.
                            (default: 'Log')
        min_rpar (float):   For any metric that supports it, the minimum difference in Rparallel
                            to allow for pairs being included in the correlation function.
                            (default: None)
        max_rpar (float):   For any metric that supports it,, the maximum difference in Rparallel
                            to allow for pairs being included in the correlation function.
                            (default: None)
        period (float):     For the 'Periodic' metric, the period to use in all directions.
                            (default: None)
        xperiod (float):    For the 'Periodic' metric, the period to use in the x direction.
                            (default: period)
        yperiod (float):    For the 'Periodic' metric, the period to use in the y direction.
                            (default: period)
        zperiod (float):    For the 'Periodic' metric, the period to use in the z direction.
                            (default: period)

        num_threads (int):  How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler cannot use OptnMP (e.g. clang prior to version 3.7.)
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
        'sep_units' : (str, False, None, coord.AngleUnit.valid_names,
                'The units to use for min_sep and max_sep.  Also the units of the output distances'),
        'bin_slop' : (float, False, None, None,
                'The fraction of a bin width by which it is ok to let the pairs miss the correct bin.',
                'The default is to use 1 if bin_size <= 0.1, or 0.1/bin_size if bin_size > 0.1.'),
        'brute' : (bool, False, False, [False, True, 1, 2],
                'Whether to use brute-force algorithm'),
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
        'min_top' : (int, False, 3, None,
                'The minimum number of top layers to use when setting up the field.'),
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
                                             'Rlens', 'Arc', 'Periodic'],
                'Which metric to use for the distance measurements'),
        'bin_type': (str, False, 'Log', ['Log', 'Linear', 'TwoD'],
                'Which type of binning should be used'),
        'min_rpar': (float, False, None, None,
                'The minimum difference in Rparallel for pairs to include'),
        'max_rpar': (float, False, None, None,
                'The maximum difference in Rparallel for pairs to include'),
        'period': (float, False, None, None,
                'The period to use for all directions for the Periodic metric'),
        'xperiod': (float, False, None, None,
                'The period to use for the x direction for the Periodic metric'),
        'yperiod': (float, False, None, None,
                'The period to use for the y direction for the Periodic metric'),
        'zperiod': (float, False, None, None,
                'The period to use for the z direction for the Periodic metric'),
    }

    def __init__(self, config=None, logger=None, **kwargs):
        self.config = treecorr.config.merge_config(config,kwargs,BinnedCorr2._valid_params)
        if logger is None:
            self.logger = treecorr.config.setup_logger(
                    treecorr.config.get(self.config,'verbose',int,1),
                    self.config.get('log_file',None))
        else:
            self.logger = logger

        if 'output_dots' in self.config:
            self.output_dots = treecorr.config.get(self.config,'output_dots',bool)
        else:
            self.output_dots = treecorr.config.get(self.config,'verbose',int,1) >= 2

        self.bin_type = self.config.get('bin_type', None)

        self.sep_units = self.config.get('sep_units','')
        self._sep_units = treecorr.config.get(self.config,'sep_units',str,'radians')
        self._log_sep_units = math.log(self._sep_units)
        if 'nbins' not in self.config:
            if 'max_sep' not in self.config:
                raise TypeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config and self.bin_type != 'TwoD':
                raise TypeError("Missing required parameter min_sep")
            if 'bin_size' not in self.config:
                raise TypeError("Missing required parameter bin_size")
            self.min_sep = float(self.config.get('min_sep',0))
            self.max_sep = float(self.config['max_sep'])
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self.bin_size = float(self.config['bin_size'])
            self.nbins = None
        elif 'bin_size' not in self.config:
            if 'max_sep' not in self.config:
                raise TypeError("Missing required parameter max_sep")
            if 'min_sep' not in self.config and self.bin_type != 'TwoD':
                raise TypeError("Missing required parameter min_sep")
            self.min_sep = float(self.config.get('min_sep',0))
            self.max_sep = float(self.config['max_sep'])
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self.nbins = int(self.config['nbins'])
            self.bin_size = None
        elif 'max_sep' not in self.config:
            if 'min_sep' not in self.config and self.bin_type != 'TwoD':
                raise TypeError("Missing required parameter min_sep")
            self.min_sep = float(self.config.get('min_sep',0))
            self.nbins = int(self.config['nbins'])
            self.bin_size = float(self.config['bin_size'])
            self.max_sep = None
        else:
            if self.bin_type == 'TwoD':
                raise TypeError("Only 2 of max_sep, bin_size, nbins are allowed "
                                     "for bin_type='TwoD'.")
            if 'min_sep' in self.config:
                raise TypeError("Only 3 of min_sep, max_sep, bin_size, nbins are allowed.")
            self.max_sep = float(self.config['max_sep'])
            self.nbins = int(self.config['nbins'])
            self.bin_size = float(self.config['bin_size'])
            self.min_sep = None

        if self.bin_type == 'Log':
            if self.nbins is None:
                self.nbins = int(math.ceil(math.log(self.max_sep/self.min_sep)/self.bin_size))
                # Update bin_size given this value of nbins
                self.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
            elif self.bin_size is None:
                self.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
            elif self.max_sep is None:
                self.max_sep = math.exp(self.nbins*self.bin_size)*self.min_sep
            else:
                self.min_sep = self.max_sep*math.exp(-self.nbins*self.bin_size)

            # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
            self.logr = np.linspace(0, self.nbins*self.bin_size, self.nbins, endpoint=False,
                                       dtype=float)
            # Offset by the position of the center of the first bin.
            self.logr += math.log(self.min_sep) + 0.5*self.bin_size
            self.rnom = np.exp(self.logr)
            half_bin = np.exp(0.5*self.bin_size)
            self.left_edges = self.rnom / half_bin
            self.right_edges = self.rnom * half_bin
            self._nbins = self.nbins
            self._bintype = treecorr._lib.Log
            min_log_bin_size = self.bin_size
            max_log_bin_size = self.bin_size
            max_good_slop = 0.1 / self.bin_size
        elif self.bin_type == 'Linear':
            if self.nbins is None:
                self.nbins = int(math.ceil((self.max_sep-self.min_sep)/self.bin_size))
                # Update bin_size given this value of nbins
                self.bin_size = (self.max_sep-self.min_sep)/self.nbins
            elif self.bin_size is None:
                self.bin_size = (self.max_sep-self.min_sep)/self.nbins
            elif self.max_sep is None:
                self.max_sep = self.min_sep + self.nbins*self.bin_size
            else:
                self.min_sep = self.max_sep - self.nbins*self.bin_size

            self.rnom = np.linspace(self.min_sep, self.max_sep, self.nbins, endpoint=False,
                                       dtype=float)
            # Offset by the position of the center of the first bin.
            self.rnom += 0.5*self.bin_size
            self.left_edges = self.rnom - 0.5*self.bin_size
            self.right_edges = self.rnom + 0.5*self.bin_size
            self.logr = np.log(self.rnom)
            self._nbins = self.nbins
            self._bintype = treecorr._lib.Linear
            min_log_bin_size = self.bin_size / self.max_sep
            max_log_bin_size = self.bin_size / (self.min_sep + self.bin_size/2)
            max_good_slop = 0.1 / max_log_bin_size
        elif self.bin_type == 'TwoD':
            if self.nbins is None:
                self.nbins = int(math.ceil(2.*self.max_sep / self.bin_size))
                self.bin_size = 2.*self.max_sep/self.nbins
            elif self.bin_size is None:
                self.bin_size = 2.*self.max_sep/self.nbins
            else:
                self.max_sep = self.nbins * self.bin_size / 2.

            sep = np.linspace(-self.max_sep, self.max_sep, self.nbins, endpoint=False,
                                 dtype=float)
            sep += 0.5 * self.bin_size
            self.dx, self.dy = np.meshgrid(sep, sep)
            self.left_edges = self.dx - 0.5*self.bin_size
            self.right_edges = self.dx + 0.5*self.bin_size
            self.bottom_edges = self.dy - 0.5*self.bin_size
            self.top_edges = self.dy + 0.5*self.bin_size
            self.rnom = np.sqrt(self.dx**2 + self.dy**2)
            self.logr = np.zeros_like(self.rnom)
            np.log(self.rnom, out=self.logr, where=self.rnom > 0)
            self.logr[self.rnom==0.] = -np.inf
            self._nbins = self.nbins**2
            self._bintype = treecorr._lib.TwoD
            min_log_bin_size = self.bin_size / self.max_sep
            max_log_bin_size = self.bin_size / (self.min_sep + self.bin_size/2)
            max_good_slop = 0.1 / max_log_bin_size
        else:  # pragma: no cover  (Already checked by config layer)
            raise ValueError("Invalid bin_type %s"%self.bin_type)

        if self.sep_units == '':
            self.logger.info("nbins = %d, min,max sep = %g..%g, bin_size = %g",
                             self.nbins, self.min_sep, self.max_sep, self.bin_size)
        else:
            self.logger.info("nbins = %d, min,max sep = %g..%g %s, bin_size = %g",
                             self.nbins, self.min_sep, self.max_sep, self.sep_units,
                             self.bin_size)
        # The underscore-prefixed names are in natural units (radians for angles)
        self._min_sep = self.min_sep * self._sep_units
        self._max_sep = self.max_sep * self._sep_units
        if self.bin_type in ['Linear', 'TwoD']:
            self._bin_size = self.bin_size * self._sep_units
            min_log_bin_size *= self._sep_units
        else:
            self._bin_size = self.bin_size

        self.split_method = self.config.get('split_method','mean')
        self.logger.debug("Using split_method = %s",self.split_method)

        self.min_top = treecorr.config.get(self.config,'min_top',int,3)
        self.max_top = treecorr.config.get(self.config,'max_top',int,10)

        self.bin_slop = treecorr.config.get(self.config,'bin_slop',float,-1.0)
        if self.bin_slop < 0.0:
            self.bin_slop = min(max_good_slop, 1.0)
        self.b = min_log_bin_size * self.bin_slop
        if self.bin_slop > max_good_slop + 0.0001:  # Add some numerical slop
            self.logger.warning(
                "Using bin_slop = %g, bin_size = %g, b = %g\n"%(self.bin_slop,self.bin_size,self.b)+
                "It is recommended to use bin_slop <= %s in this case.\n"%max_good_slop+
                "Larger values of bin_slop (and hence b) may result in significant inaccuracies.")
        else:
            self.logger.debug("Using bin_slop = %g, b = %g",self.bin_slop,self.b)

        self.brute = treecorr.config.get(self.config,'brute',bool,False)
        if self.brute:
            self.logger.info("Doing brute force calculation%s.",
                             self.brute is 1 and " for first field" or
                             (self.brute is 2 and " for second field" or ""))
        self.coords = None
        self.metric = None
        self.min_rpar = treecorr.config.get(self.config,'min_rpar',float,-sys.float_info.max)
        self.max_rpar = treecorr.config.get(self.config,'max_rpar',float,sys.float_info.max)
        if self.min_rpar > self.max_rpar:
            raise ValueError("min_rpar must be <= max_rpar")
        period = treecorr.config.get(self.config,'period',float,0)
        self.xperiod = treecorr.config.get(self.config,'xperiod',float,period)
        self.yperiod = treecorr.config.get(self.config,'yperiod',float,period)
        self.zperiod = treecorr.config.get(self.config,'zperiod',float,period)

    def _process_all_auto(self, cat1, metric, num_threads):
        for i,c1 in enumerate(cat1):
            self.process_auto(c1,metric,num_threads)
            for c2 in cat1[i+1:]:
                self.process_cross(c1,c2,metric,num_threads)

    def _process_all_cross(self, cat1, cat2, metric, num_threads):
        if treecorr.config.get(self.config,'pairwise',bool,False):
            if len(cat1) != len(cat2):
                raise ValueError("Number of files for 1 and 2 must be equal for pairwise.")
            for c1,c2 in zip(cat1,cat2):
                if c1.ntot != c2.ntot:
                    raise ValueError("Number of objects must be equal for pairwise.")
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
        if metric not in ['Rperp', 'OldRperp', 'FisherRperp', 'Rlens', 'Arc']:
            if self.min_rpar != -sys.float_info.max:
                raise ValueError("min_rpar is not valid for %s metric."%metric)
            if self.max_rpar != sys.float_info.max:
                raise ValueError("max_rpar is not valid for %s metric."%metric)
        coords, metric = treecorr.util.parse_metric(metric, coords1, coords2)
        if self.sep_units != '' and coords == '3d' and metric != 'Arc':
            raise ValueError("sep_units is invalid with 3d coordinates. "
                             "min_sep and max_sep should be in the same units as r (or x,y,z).")
        if self.coords != None or self.metric != None:
            if coords != self.coords:
                self.logger.warning("Detected a change in catalog coordinate systems.\n"+
                                    "This probably doesn't make sense!")
            if metric != self.metric:
                self.logger.warning("Detected a change in metric.\n"+
                                    "This probably doesn't make sense!")
        if metric == 'Periodic':
            if self.xperiod == 0 or self.yperiod == 0 or (coords=='3d' and self.zperiod == 0):
                raise ValueError("Periodic metric requires setting the period to use.")
        else:
            if self.xperiod != 0 or self.yperiod != 0 or self.zperiod != 0:
                raise ValueError("period options are not valid for %s metric."%metric)
        self.coords = coords  # These are the regular string values
        self.metric = metric
        self._coords = treecorr.util.coord_enum(coords)  # These are the C++-layer enums
        self._metric = treecorr.util.metric_enum(metric)

    def _apply_units(self, mask):
        if self.coords == 'spherical' and self.metric == 'Euclidean':
            # Then our distances are all angles.  Convert from the chord distance to a real angle.
            # L = 2 sin(theta/2)
            self.meanr[mask] = 2. * np.arcsin(self.meanr[mask]/2.)
            self.meanlogr[mask] = np.log( 2. * np.arcsin(np.exp(self.meanlogr[mask])/2.) )
        self.meanr[mask] /= self._sep_units
        self.meanlogr[mask] -= self._log_sep_units

    def _get_minmax_size(self):
        if self.metric == 'Euclidean':
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

    def sample_pairs(self, n, cat1, cat2, min_sep, max_sep, metric=None):
        """Return a random sample of n pairs whose separations fall between min_sep and max_sep.

        This would typically be used to get some random subset of the indices of pairs that
        fell into a particular bin of the correlation.  E.g. to get 100 pairs from the third
        bin of a BinnedCorr2 instance, corr, you could write::

            >>> min_sep = corr.left_edges[2]   # third bin has i=2
            >>> max_sep = corr.right_edges[2]
            >>> i1, i2, sep = corr.sample_pairs(100, cat1, cat2, min_sep, max_sep)

        The min_sep and max_sep should use the same units as were defined which constructing
        the corr instance.

        The selection process will also use the same bin_slop as specified (either explicitly or
        implicitly) when constructing the corr instance.  This means that some of the pairs may
        have actual separations slightly outside of the specified range.  If you want a selection
        using an exact range without any slop, you should construct a new Correlation instance
        with brute=True or bin_slop=0, and call sample_pairs with that.

        The returned separations will likewise correspond to the separation of the cells in the
        tree where the correlation function would have decided that the pair falls into the
        given bin.  Therefore, if these cells were not leaf cells, then they will not typically
        be equal to the real separation for the given metric.

        Also, note that min_sep and max_sep may be arbitrary.  There is no requirement that they
        be edges of one of the standard bins for this correlation function.  There is also no
        requirement that this correlation instance has already accumulated pairs via a call
        to process with these catalogs.

        Parameters:
            n (int):            How many samples to return.
            cat1 (Catalog):     The catalog from which to sample the first object of each pair.
            cat2 (Catalog):     The catalog from which to sample the second object of each pair.
                                (This may be the same as cat1.)
            min_sep (float):    The minimum separation for the returned pairs (modulo some slop
                                allowed by the bin_slop parameter).
            max_sep (float):    The maximum separation for the returned pairs (modulo some slop
                                allowed by the bin_slop parameter).
            metric (str):       Which metric to use.  See `Metrics` for details.  (default:
                                self.metric, or 'Euclidean' if not set yet)

        Returns:
            Tuple containing

                - i1 (array): indices of objects from cat1
                - i2 (array): indices of objects from cat2
                - sep (array): separations of the pairs of objects (i1,i2)
        """
        from .util import long_ptr as lp
        from .util import double_ptr as dp

        if metric is None:
            metric = self.config.get('metric', 'Euclidean')

        self._set_metric(metric, cat1.coords, cat2.coords)

        f1 = cat1.field
        f2 = cat2.field

        if f1 is None or f1._coords != self._coords:
            # I don't really know if it's possible to get the coords out of sync,
            # so the 2nd check might be superfluous.
            # The first one though is definitely possible, so we need to check that.
            self.logger.debug("In sample_pairs, making default field for cat1")
            min_size, max_size = self._get_minmax_size()
            f1 = cat1.getNField(min_size, max_size, self.split_method,
                                self.brute is True or self.brute is 1,
                                self.min_top, self.max_top, self.coords)
        if f2 is None or f2._coords != self._coords:
            self.logger.debug("In sample_pairs, making default field for cat2")
            min_size, max_size = self._get_minmax_size()
            f2 = cat2.getNField(min_size, max_size, self.split_method,
                                self.brute is True or self.brute is 2,
                                self.min_top, self.max_top, self.coords)

        # Apply units to min_sep, max_sep:
        min_sep *= self._sep_units
        max_sep *= self._sep_units

        i1 = np.zeros(n, dtype=int)
        i2 = np.zeros(n, dtype=int)
        sep = np.zeros(n, dtype=float)
        ntot = treecorr._lib.SamplePairs(self.corr, f1.data, f2.data, min_sep, max_sep,
                                         f1._d, f2._d, self._coords, self._bintype, self._metric,
                                         lp(i1), lp(i2), dp(sep), n)

        if ntot < n:
            n = ntot
            i1 = i1[:n]
            i2 = i2[:n]
            sep = sep[:n]
        # Convert back to nominal units
        sep /= self._sep_units
        self.logger.info("Sampled %d pairs out of a total of %d.", n, ntot)

        return i1, i2, sep

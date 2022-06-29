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
.. module:: binnedcorr3
"""

import math
import numpy as np
import sys
import coord

from . import _lib
from .config import merge_config, setup_logger, get
from .util import parse_metric, metric_enum, coord_enum, set_omp_threads, lazy_property
from .util import make_reader
from .util import depr_pos_kwargs
from .binnedcorr2 import estimate_multi_cov, build_multi_cov_design_matrix

class Namespace(object):
    pass

class BinnedCorr3(object):
    """This class stores the results of a 3-point correlation calculation, along with some
    ancillary data.

    This is a base class that is not intended to be constructed directly.  But it has a few
    helper functions that derived classes can use to help perform their calculations.  See
    the derived classes for more details:

    - `NNNCorrelation` handles count-count-count correlation functions
    - `KKKCorrelation` handles kappa-kappa-kappa correlation functions
    - `GGGCorrelation` handles gamma-gamma-gamma correlation functions

    Three-point correlations are a bit more complicated than two-point, since the data need
    to be binned in triangles, not just the separation between two points.  We characterize the
    triangles according to the following three parameters based on the three side lenghts
    of the triangle with d1 >= d2 >= d3.

    .. math::
        r &= d2 \\\\
        u &= \\frac{d3}{d2} \\\\
        v &= \\pm \\frac{(d1 - d2)}{d3} \\\\

    The orientation of the triangle is specified by the sign of v.
    Positive v triangles have the three sides d1,d2,d3 in counter-clockwise orientation.
    Negative v triangles have the three sides d1,d2,d3 in clockwise orientation.

    .. note::
        We always bin the same way for positive and negative v values, and the binning
        specification for v should just be for the positive values.  E.g. if you specify
        min_v=0.2, max_v=0.6, then TreeCorr will also accumulate triangles with
        -0.6 < v < -0.2 in addition to those with 0.2 < v < 0.6.

    The constructor for all derived classes take a config dict as the first argument,
    since this is often how we keep track of parameters, but if you don't want to
    use one or if you want to change some parameters from what are in a config dict,
    then you can use normal kwargs, which take precedence over anything in the config dict.

    There are three implemented definitions for the ``metric``, which defines how to calculate
    the distance between two points, for three-point corretions:

        - 'Euclidean' = straight line Euclidean distance between two points.  For spherical
          coordinates (ra,dec without r), this is the chord distance between points on the
          unit sphere.
        - 'Arc' = the true great circle distance for spherical coordinates.
        - 'Periodic' = Like Euclidean, but with periodic boundaries.

          .. note::

            The triangles for three-point correlations can become ambiguous if d1 > period/2,
            which means the maximum d2 (max_sep) should be less than period/4.
            This is not enforced.

    So far, there is only one allowed value for the ``bin_type`` for three-point correlations.

        - 'LogRUV' - The bin steps will be uniform in log(r) from log(min_sep) .. log(max_sep).
          The u and v values are binned linearly from min_u .. max_u and min_v .. max_v.


    Parameters:
        config (dict):      A configuration dict that can be used to pass in the below kwargs if
                            desired.  This dict is allowed to have addition entries in addition
                            to those listed below, which are ignored here. (default: None)
        logger:             If desired, a logger object for logging. (default: None, in which case
                            one will be built according to the config dict's verbose level.)

    Keyword Arguments:

        nbins (int):        How many bins to use. (Exactly three of nbins, bin_size, min_sep,
                            max_sep are required. If nbins is not given or set to None, it will be
                            calculated from the values of the other three, rounding up to the next
                            highest integer. In this case, bin_size will be readjusted to account
                            for this rounding up.)
        bin_size (float):   The width of the bins in log(separation). (Exactly three of nbins,
                            bin_size, min_sep, max_sep are required.  If bin_size is not given or
                            set to None, it will be calculated from the values of the other three.)
        min_sep (float):    The minimum separation in units of sep_units, if relevant. (Exactly
                            three of nbins, bin_size, min_sep, max_sep are required.  If min_sep is
                            not given or set to None, it will be calculated from the values of the
                            other three.)
        max_sep (float):    The maximum separation in units of sep_units, if relevant. (Exactly
                            three of nbins, bin_size, min_sep, max_sep are required.  If max_sep is
                            not given or set to None, it will be calculated from the values of the
                            other three.)

        sep_units (str):    The units to use for the separation values, given as a string.  This
                            includes both min_sep and max_sep above, as well as the units of the
                            output distance values.  Valid options are arcsec, arcmin, degrees,
                            hours, radians.  (default: radians if angular units make sense, but for
                            3-d or flat 2-d positions, the default will just match the units of
                            x,y[,z] coordinates)
        bin_slop (float):   How much slop to allow in the placement of triangles in the bins.
                            If bin_slop = 1, then the bin into which a particular pair is placed
                            may be incorrect by at most 1.0 bin widths.  (default: None, which
                            means to use a bin_slop that gives a maximum error of 10% on any bin,
                            which has been found to yield good results for most application.

        nubins (int):       Analogous to nbins for the u values.  (The default is to calculate from
                            ubin_size = binsize, min_u = 0, max_u = 1, but this can be overridden
                            by specifying up to 3 of these four parametes.)
        ubin_size (float):  Analogous to bin_size for the u values. (default: bin_size)
        min_u (float):      Analogous to min_sep for the u values. (default: 0)
        max_u (float):      Analogous to max_sep for the u values. (default: 1)

        nvbins (int):       Analogous to nbins for the positive v values.  (The default is to
                            calculate from vbin_size = binsize, min_v = 0, max_v = 1, but this can
                            be overridden by specifying up to 3 of these four parametes.)
        vbin_size (float):  Analogous to bin_size for the v values. (default: bin_size)
        min_v (float):      Analogous to min_sep for the positive v values. (default: 0)
        max_v (float):      Analogous to max_sep for the positive v values. (default: 1)

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
        output_dots (bool): Whether to output progress dots during the calcualtion of the
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
                            (default: :math:`\\max(3, \\log_2(N_{\\rm cpu}))`)
        max_top (int):      The maximum number of top layers to use when setting up the field.
                            The top-level cells are where each calculation job starts. There will
                            typically be of order :math:`2^{\\rm max\\_top}` top-level cells.
                            (default: 10)
        precision (int):    The precision to use for the output values. This specifies how many
                            digits to write. (default: 4)

        metric (str):       Which metric to use for distance measurements.  Options are listed
                            above.  (default: 'Euclidean')
        bin_type (str):     What type of binning should be used.  Only one option currently.
                            (default: 'LogRUV')
        period (float):     For the 'Periodic' metric, the period to use in all directions.
                            (default: None)
        xperiod (float):    For the 'Periodic' metric, the period to use in the x direction.
                            (default: period)
        yperiod (float):    For the 'Periodic' metric, the period to use in the y direction.
                            (default: period)
        zperiod (float):    For the 'Periodic' metric, the period to use in the z direction.
                            (default: period)

        var_method (str):   Which method to use for estimating the variance. Options are:
                            'shot', 'jackknife', 'sample', 'bootstrap', 'marked_bootstrap'.
                            (default: 'shot')
        num_bootstrap (int): How many bootstrap samples to use for the 'bootstrap' and
                            'marked_bootstrap' var_methods.  (default: 500)
        rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                            random number generation. (default: None)

        num_threads (int):  How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.)

                            .. note::

                                This won't work if the system's C compiler cannot use OpenMP
                                (e.g. clang prior to version 3.7.)
    """
    _valid_params = {
        'nbins' : (int, False, None, None,
                'The number of output bins to use for sep dimension.'),
        'bin_size' : (float, False, None, None,
                'The size of the output bins in log(sep).'),
        'min_sep' : (float, False, None, None,
                'The minimum separation to include in the output.'),
        'max_sep' : (float, False, None, None,
                'The maximum separation to include in the output.'),
        'sep_units' : (str, False, None, coord.AngleUnit.valid_names,
                'The units to use for min_sep and max_sep.  Also the units of the output '
                'distances'),
        'bin_slop' : (float, False, None, None,
                'The fraction of a bin width by which it is ok to let the triangles miss the '
                'correct bin.',
                'The default is to use 1 if bin_size <= 0.1, or 0.1/bin_size if bin_size > 0.1.'),
        'nubins' : (int, False, None, None,
                'The number of output bins to use for u dimension.'),
        'ubin_size' : (float, False, None, None,
                'The size of the output bins in u.'),
        'min_u' : (float, False, None, None,
                'The minimum u to include in the output.'),
        'max_u' : (float, False, None, None,
                'The maximum u to include in the output.'),
        'nvbins' : (int, False, None, None,
                'The number of output bins to use for positive v values.'),
        'vbin_size' : (float, False, None, None,
                'The size of the output bins in v.'),
        'min_v' : (float, False, None, None,
                'The minimum |v| to include in the output.'),
        'max_v' : (float, False, None, None,
                'The maximum |v| to include in the output.'),
        'brute' : (bool, False, False, [False, True],
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
        'min_top' : (int, False, None, None,
                'The minimum number of top layers to use when setting up the field.'),
        'max_top' : (int, False, 10, None,
                'The maximum number of top layers to use when setting up the field.'),
        'precision' : (int, False, 4, None,
                'The number of digits after the decimal in the output.'),
        'metric': (str, False, 'Euclidean', ['Euclidean', 'Arc', 'Periodic'],
                'Which metric to use for the distance measurements'),
        'bin_type': (str, False, 'LogRUV', ['LogRUV'],
                'Which type of binning should be used'),
        'period': (float, False, None, None,
                'The period to use for all directions for the Periodic metric'),
        'xperiod': (float, False, None, None,
                'The period to use for the x direction for the Periodic metric'),
        'yperiod': (float, False, None, None,
                'The period to use for the y direction for the Periodic metric'),
        'zperiod': (float, False, None, None,
                'The period to use for the z direction for the Periodic metric'),

        'var_method': (str, False, 'shot',
                ['shot', 'jackknife', 'sample', 'bootstrap', 'marked_bootstrap'],
                'The method to use for estimating the variance'),
        'num_bootstrap': (int, False, 500, None,
                'How many bootstrap samples to use for the var_method=bootstrap and '
                'marked_bootstrap'),
        'num_threads' : (int, False, None, None,
                'How many threads should be used. num_threads <= 0 means auto based on num cores.'),
    }

    @depr_pos_kwargs
    def __init__(self, config=None, *, logger=None, rng=None, **kwargs):
        self._corr = None  # Do this first to make sure we always have it for __del__
        self.config = merge_config(config,kwargs,BinnedCorr3._valid_params)
        if logger is None:
            self.logger = setup_logger(get(self.config,'verbose',int,1),
                                       self.config.get('log_file',None))
        else:
            self.logger = logger

        # We'll make a bunch of attributes here, which we put into a namespace called _ro.
        # These are the core attributes that won't ever be changed after construction.
        # This is an efficiency optimization (both memory and flops), since it will allow
        # copy() to just copy a pointer to the _ro namespace without having to copy each
        # individual attribute separately.
        # The access of these attributes are all via read-only properties.
        self._ro = Namespace()

        if 'output_dots' in self.config:
            self._ro.output_dots = get(self.config,'output_dots',bool)
        else:
            self._ro.output_dots = get(self.config,'verbose',int,1) >= 2

        self._ro.bin_type = self.config.get('bin_type', None)
        self._ro._bintype = _lib.Log

        self._ro.sep_units = self.config.get('sep_units','')
        self._ro._sep_units = get(self.config,'sep_units',str,'radians')
        self._ro._log_sep_units = math.log(self._sep_units)
        if self.config.get('nbins', None) is None:
            if self.config.get('max_sep', None) is None:
                raise TypeError("Missing required parameter max_sep")
            if self.config.get('min_sep', None) is None:
                raise TypeError("Missing required parameter min_sep")
            if self.config.get('bin_size', None) is None:
                raise TypeError("Missing required parameter bin_size")
            self._ro.min_sep = float(self.config['min_sep'])
            self._ro.max_sep = float(self.config['max_sep'])
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            bin_size = float(self.config['bin_size'])
            self._ro.nbins = int(math.ceil(math.log(self.max_sep/self.min_sep)/bin_size))
            # Update self.bin_size given this value of nbins
            self._ro.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
            # Note in this case, bin_size is saved as the nominal bin_size from the config
            # file, and self.bin_size is the one for the radial bins.  We'll use the nominal
            # bin_size as the default bin_size for u and v below.
        elif self.config.get('bin_size', None) is None:
            if self.config.get('max_sep', None) is None:
                raise TypeError("Missing required parameter max_sep")
            if self.config.get('min_sep', None) is None:
                raise TypeError("Missing required parameter min_sep")
            self._ro.min_sep = float(self.config['min_sep'])
            self._ro.max_sep = float(self.config['max_sep'])
            if self.min_sep >= self.max_sep:
                raise ValueError("max_sep must be larger than min_sep")
            self._ro.nbins = int(self.config['nbins'])
            bin_size = self._ro.bin_size = math.log(self.max_sep/self.min_sep)/self.nbins
        elif self.config.get('max_sep', None) is None:
            if self.config.get('min_sep', None) is None:
                raise TypeError("Missing required parameter min_sep")
            self._ro.min_sep = float(self.config['min_sep'])
            self._ro.nbins = int(self.config['nbins'])
            bin_size = self._ro.bin_size = float(self.config['bin_size'])
            self._ro.max_sep = math.exp(self.nbins*bin_size)*self.min_sep
        else:
            if self.config.get('min_sep', None) is not None:
                raise TypeError("Only 3 of min_sep, max_sep, bin_size, nbins are allowed.")
            self._ro.max_sep = float(self.config['max_sep'])
            self._ro.nbins = int(self.config['nbins'])
            bin_size = self._ro.bin_size = float(self.config['bin_size'])
            self._ro.min_sep = self.max_sep*math.exp(-self.nbins*bin_size)
        if self.sep_units == '':
            self.logger.info("r: nbins = %d, min,max sep = %g..%g, bin_size = %g",
                             self.nbins, self.min_sep, self.max_sep, self.bin_size)
        else:
            self.logger.info("r: nbins = %d, min,max sep = %g..%g %s, bin_size = %g",
                             self.nbins, self.min_sep, self.max_sep, self.sep_units,
                             self.bin_size)
        # The underscore-prefixed names are in natural units (radians for angles)
        self._ro._min_sep = self.min_sep * self._sep_units
        self._ro._max_sep = self.max_sep * self._sep_units
        self._ro._bin_size = self.bin_size  # There is not Linear, but if I add it, need to apply
                                            # units to _bin_size in that case as well.

        self._ro.min_u = float(self.config.get('min_u', 0.))
        self._ro.max_u = float(self.config.get('max_u', 1.))
        if self.min_u >= self.max_u:
            raise ValueError("max_u must be larger than min_u")
        if self.min_u < 0. or self.max_u > 1.:
            raise ValueError("Invalid range for u: %f - %f"%(self.min_u, self.max_u))
        self._ro.ubin_size = float(self.config.get('ubin_size', bin_size))
        if 'nubins' not in self.config:
            self._ro.nubins = int(math.ceil((self.max_u-self.min_u-1.e-10)/self.ubin_size))
        elif 'max_u' in self.config and 'min_u' in self.config and 'ubin_size' in self.config:
            raise TypeError("Only 3 of min_u, max_u, ubin_size, nubins are allowed.")
        else:
            self._ro.nubins = self.config['nubins']
            # Allow min or max u to be implicit from nubins and ubin_size
            if 'ubin_size' in self.config:
                if 'min_u' not in self.config:
                    self._ro.min_u = max(self.max_u - self.nubins * self.ubin_size, 0.)
                if 'max_u' not in self.config:
                    self._ro.max_u = min(self.min_u + self.nubins * self.ubin_size, 1.)
        # Adjust ubin_size given the other values
        self._ro.ubin_size = (self.max_u-self.min_u)/self.nubins
        self.logger.info("u: nbins = %d, min,max = %g..%g, bin_size = %g",
                         self.nubins,self.min_u,self.max_u,self.ubin_size)

        self._ro.min_v = float(self.config.get('min_v', 0.))
        self._ro.max_v = float(self.config.get('max_v', 1.))
        if self.min_v >= self.max_v:
            raise ValueError("max_v must be larger than min_v")
        if self.min_v < 0 or self.max_v > 1.:
            raise ValueError("Invalid range for |v|: %f - %f"%(self.min_v, self.max_v))
        self._ro.vbin_size = float(self.config.get('vbin_size', bin_size))
        if 'nvbins' not in self.config:
            self._ro.nvbins = int(math.ceil((self.max_v-self.min_v-1.e-10)/self.vbin_size))
        elif 'max_v' in self.config and 'min_v' in self.config and 'vbin_size' in self.config:
            raise TypeError("Only 3 of min_v, max_v, vbin_size, nvbins are allowed.")
        else:
            self._ro.nvbins = self.config['nvbins']
            # Allow min or max v to be implicit from nvbins and vbin_size
            if 'vbin_size' in self.config:
                if 'max_v' not in self.config:
                    self._ro.max_v = min(self.min_v + self.nvbins * self.vbin_size, 1.)
                else:  # min_v not in config
                    self._ro.min_v = max(self.max_v - self.nvbins * self.vbin_size, -1.)
        # Adjust vbin_size given the other values
        self._ro.vbin_size = (self.max_v-self.min_v)/self.nvbins
        self.logger.info("v: nbins = %d, min,max = %g..%g, bin_size = %g",
                         self.nvbins,self.min_v,self.max_v,self.vbin_size)

        self._ro.split_method = self.config.get('split_method','mean')
        self.logger.debug("Using split_method = %s",self.split_method)

        self._ro.min_top = get(self.config,'min_top',int,None)
        self._ro.max_top = get(self.config,'max_top',int,10)

        self._ro.bin_slop = get(self.config,'bin_slop',float,-1.0)
        if self.bin_slop < 0.0:
            if self.bin_size <= 0.1:
                self._ro.bin_slop = 1.0
                self._ro.b = self.bin_size
            else:
                self._ro.bin_slop = 0.1/self.bin_size  # The stored bin_slop corresponds to lnr bins.
                self._ro.b = 0.1
            if self.ubin_size <= 0.1:
                self._ro.bu = self.ubin_size
            else:
                self._ro.bu = 0.1
            if self.vbin_size <= 0.1:
                self._ro.bv = self.vbin_size
            else:
                self._ro.bv = 0.1
        else:
            self._ro.b = self.bin_size * self.bin_slop
            self._ro.bu = self.ubin_size * self.bin_slop
            self._ro.bv = self.vbin_size * self.bin_slop

        if self.b > 0.100001:  # Add some numerical slop
            self.logger.warning(
                    "Using bin_slop = %g, bin_size = %g\n"%(self.bin_slop,self.bin_size)+
                    "The b parameter is bin_slop * bin_size = %g"%(self.b)+
                    "  bu = %g, bv = %g\n"%(self.bu,self.bv)+
                    "It is generally recommended to use b <= 0.1 for most applications.\n"+
                    "Larger values of this b parameter may result in significant inaccuracies.")
        else:
            self.logger.debug("Using bin_slop = %g, b = %g, bu = %g, bv = %g",
                              self.bin_slop,self.b,self.bu,self.bv)

        # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
        self._ro.logr1d = np.linspace(start=0, stop=self.nbins*self.bin_size,
                                      num=self.nbins, endpoint=False)
        # Offset by the position of the center of the first bin.
        self._ro.logr1d += math.log(self.min_sep) + 0.5*self.bin_size

        self._ro.u1d = np.linspace(start=0, stop=self.nubins*self.ubin_size,
                                   num=self.nubins, endpoint=False)
        self._ro.u1d += self.min_u + 0.5*self.ubin_size

        self._ro.v1d = np.linspace(start=0, stop=self.nvbins*self.vbin_size,
                                   num=self.nvbins, endpoint=False)
        self._ro.v1d += self.min_v + 0.5*self.vbin_size
        self._ro.v1d = np.concatenate([-self.v1d[::-1],self.v1d])

        self._ro.logr = np.tile(self.logr1d[:, np.newaxis, np.newaxis],
                                (1, self.nubins, 2*self.nvbins))
        self._ro.u = np.tile(self.u1d[np.newaxis, :, np.newaxis],
                             (self.nbins, 1, 2*self.nvbins))
        self._ro.v = np.tile(self.v1d[np.newaxis, np.newaxis, :],
                             (self.nbins, self.nubins, 1))
        self._ro.rnom = np.exp(self.logr)
        self._ro.rnom1d = np.exp(self.logr1d)
        self._ro.brute = get(self.config,'brute',bool,False)
        if self.brute:
            self.logger.info("Doing brute force calculation.",)
        self.coords = None
        self.metric = None
        period = get(self.config,'period',float,0)
        self._ro.xperiod = get(self.config,'xperiod',float,period)
        self._ro.yperiod = get(self.config,'yperiod',float,period)
        self._ro.zperiod = get(self.config,'zperiod',float,period)
        self._ro._nbins = len(self._ro.logr.ravel())

        self._ro.var_method = get(self.config,'var_method',str,'shot')
        self._ro.num_bootstrap = get(self.config,'num_bootstrap',int,500)
        self.results = {}  # for jackknife, etc. store the results of each pair of patches.
        self.npatch1 = self.npatch2 = self.npatch3 = 1
        self._rng = rng

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.RandomState()
        return self._rng

    # Properties for all the read-only attributes ("ro" stands for "read-only")
    @property
    def output_dots(self): return self._ro.output_dots
    @property
    def bin_type(self): return self._ro.bin_type
    @property
    def sep_units(self): return self._ro.sep_units
    @property
    def _sep_units(self): return self._ro._sep_units
    @property
    def _log_sep_units(self): return self._ro._log_sep_units
    @property
    def min_sep(self): return self._ro.min_sep
    @property
    def max_sep(self): return self._ro.max_sep
    @property
    def min_u(self): return self._ro.min_u
    @property
    def max_u(self): return self._ro.max_u
    @property
    def min_v(self): return self._ro.min_v
    @property
    def max_v(self): return self._ro.max_v
    @property
    def bin_size(self): return self._ro.bin_size
    @property
    def ubin_size(self): return self._ro.ubin_size
    @property
    def vbin_size(self): return self._ro.vbin_size
    @property
    def nbins(self): return self._ro.nbins
    @property
    def nubins(self): return self._ro.nubins
    @property
    def nvbins(self): return self._ro.nvbins
    @property
    def logr1d(self): return self._ro.logr1d
    @property
    def u1d(self): return self._ro.u1d
    @property
    def v1d(self): return self._ro.v1d
    @property
    def logr(self): return self._ro.logr
    @property
    def u(self): return self._ro.u
    @property
    def v(self): return self._ro.v
    @property
    def rnom(self): return self._ro.rnom
    @property
    def rnom1d(self): return self._ro.rnom1d
    @property
    def _bintype(self): return self._ro._bintype
    @property
    def _nbins(self): return self._ro._nbins
    @property
    def _min_sep(self): return self._ro._min_sep
    @property
    def _max_sep(self): return self._ro._max_sep
    @property
    def _bin_size(self): return self._ro._bin_size
    @property
    def split_method(self): return self._ro.split_method
    @property
    def min_top(self): return self._ro.min_top
    @property
    def max_top(self): return self._ro.max_top
    @property
    def bin_slop(self): return self._ro.bin_slop
    @property
    def b(self): return self._ro.b
    @property
    def bu(self): return self._ro.bu
    @property
    def bv(self): return self._ro.bv
    @property
    def brute(self): return self._ro.brute
    @property
    def xperiod(self): return self._ro.xperiod
    @property
    def yperiod(self): return self._ro.yperiod
    @property
    def zperiod(self): return self._ro.zperiod
    @property
    def var_method(self): return self._ro.var_method
    @property
    def num_bootstrap(self): return self._ro.num_bootstrap
    @property
    def _d1(self): return self._ro._d1
    @property
    def _d2(self): return self._ro._d2
    @property
    def _d3(self): return self._ro._d3

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_corr',None)
        d.pop('_ok',None)     # Remake this as needed.
        d.pop('logger',None)  # Oh well.  This is just lost in the copy.  Can't be pickled.
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._corr = None
        self.logger = setup_logger(get(self.config,'verbose',int,1),
                                   self.config.get('log_file',None))

    def clear(self):
        """Clear all data vectors, the results dict, and any related values.
        """
        self._clear()
        self.results = {}
        self.npatch1 = self.npatch2 = self.npatch3 = 1
        self.__dict__.pop('_ok',None)

    @property
    def nonzero(self):
        """Return if there are any values accumulated yet.  (i.e. ntri > 0)
        """
        return np.any(self.ntri)

    def _add_tot(self, i, j, k, c1, c2, c3):
        # No op for all but NNCorrelation, which needs to add the tot value
        pass

    def _trivially_zero(self, c1, c2, c3, metric):
        # For now, ignore the metric.  Just be conservative about how much space we need.
        x1,y1,z1,s1 = c1._get_center_size()
        x2,y2,z2,s2 = c2._get_center_size()
        x3,y3,z3,s3 = c3._get_center_size()
        d3 = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
        d1 = ((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2)**0.5
        d2 = ((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2)**0.5
        d3, d2, d1 = sorted([d1,d2,d3])
        return (d2 > s1 + s2 + s3 + 2*self._max_sep)  # The 2* is where we are being conservative.

    def _process_all_auto(self, cat1, metric, num_threads, comm=None, low_mem=False):

        def is_my_job(my_indices, i, j, k, n):
            # Helper function to figure out if a given (i,j,k) job should be done on the
            # current process.

            # Always my job if not using MPI.
            if my_indices is None:
                return True

            # Now the tricky part.  If using MPI, we need to divide up the jobs smartly.
            # The first point is to divvy up the auto jobs evenly.  This is where most of the
            # work is done, so we want those to be spreads as evenly as possibly across procs.
            # Therefore, if all indices are mine, then do the job.
            # This reduces the number of catalogs this machine needs to load up.
            n1 = np.sum([i in my_indices, j in my_indices, k in my_indices])
            if n1 == 3:
                self.logger.info("Rank %d: Job (%d,%d,%d) is mine.",rank,i,j,k)
                return True

            # If none of the indices are mine, then it's not my job.
            if n1 == 0:
                return False

            # When only one or two of the indices are mine, then we follow the same kind of
            # procedure as we did in 2pt.  There, we decided based on the parity of i.
            # Here that turns into i mod 3.
            if ( (i % 3 == 0 and i in my_indices) or
                 (i % 3 == 1 and j in my_indices) or
                 (i % 3 == 2 and k in my_indices) ):
                self.logger.info("Rank %d: Job (%d,%d,%d) is mine.",rank,i,j,k)
                return True
            else:
                return False

        if len(cat1) == 1 and cat1[0].npatch == 1:
            self.process_auto(cat1[0], metric=metric, num_threads=num_threads)

        else:
            # When patch processing, keep track of the pair-wise results.
            if self.npatch1 == 1:
                self.npatch1 = cat1[0].npatch if cat1[0].npatch != 1 else len(cat1)
                self.npatch2 = self.npatch3 = self.npatch1
            n = self.npatch1

            # Setup for deciding when this is my job.
            if comm:
                size = comm.Get_size()
                rank = comm.Get_rank()
                my_indices = np.arange(n * rank // size, n * (rank+1) // size)
                self.logger.info("Rank %d: My indices are %s",rank,my_indices)
            else:
                my_indices = None

            temp = self.copy()
            for ii,c1 in enumerate(cat1):
                i = c1.patch if c1.patch is not None else ii
                if is_my_job(my_indices, i, i, i, n):
                    temp.clear()
                    self.logger.info('Process patch %d auto',i)
                    temp.process_auto(c1, metric=metric, num_threads=num_threads)
                    if (i,i,i) in self.results and self.results[(i,i,i)].nonzero:
                        self.results[(i,i,i)] += temp
                    else:
                        self.results[(i,i,i)] = temp.copy()
                    self += temp

                for jj,c2 in list(enumerate(cat1))[::-1]:
                    j = c2.patch if c2.patch is not None else jj
                    if i < j:
                        if is_my_job(my_indices, i, j, j, n):
                            temp.clear()
                            # One point in c1, 2 in c2.
                            if not self._trivially_zero(c1,c2,c2,metric):
                                self.logger.info('Process patches %d,%d cross12',i,j)
                                temp.process_cross12(c1, c2, metric=metric, num_threads=num_threads)
                            else:
                                self.logger.info('Skipping %d,%d pair, which are too far apart ' +
                                                 'for this set of separations',i,j)
                            if temp.nonzero:
                                if (i,j,j) in self.results and self.results[(i,j,j)].nonzero:
                                    self.results[(i,j,j)] += temp
                                else:
                                    self.results[(i,j,j)] = temp.copy()
                                self += temp
                            else:
                                # NNNCorrelation needs to add the tot value
                                self._add_tot(i, j, j, c1, c2, c2)

                            temp.clear()
                            # One point in c2, 2 in c1.
                            if not self._trivially_zero(c1,c1,c2,metric):
                                self.logger.info('Process patches %d,%d cross12',j,i)
                                temp.process_cross12(c2, c1, metric=metric, num_threads=num_threads)
                            if temp.nonzero:
                                if (i,i,j) in self.results and self.results[(i,i,j)].nonzero:
                                    self.results[(i,i,j)] += temp
                                else:
                                    self.results[(i,i,j)] = temp.copy()
                                self += temp
                            else:
                                # NNNCorrelation needs to add the tot value
                                self._add_tot(i, i, j, c2, c1, c1)

                        # One point in each of c1, c2, c3
                        for kk,c3 in enumerate(cat1):
                            k = c3.patch if c3.patch is not None else kk
                            if j < k and is_my_job(my_indices, i, j, k, n):
                                temp.clear()

                                if not self._trivially_zero(c1,c2,c3,metric):
                                    self.logger.info('Process patches %d,%d,%d cross',i,j,k)
                                    temp.process_cross(c1, c2, c3, metric=metric,
                                                       num_threads=num_threads)
                                else:
                                    self.logger.info('Skipping %d,%d,%d, which are too far apart ' +
                                                     'for this set of separations',i,j,k)
                                if temp.nonzero:
                                    if (i,j,k) in self.results and self.results[(i,j,k)].nonzero:
                                        self.results[(i,j,k)] += temp
                                    else:
                                        self.results[(i,j,k)] = temp.copy()
                                    self += temp
                                else:
                                    # NNNCorrelation needs to add the tot value
                                    self._add_tot(i, j, k, c1, c2, c3)
                                if low_mem:
                                    c3.unload()

                        if low_mem and jj != ii+1:
                            # Don't unload i+1, since that's the next one we'll need.
                            c2.unload()
                if low_mem:
                    c1.unload()
            if comm is not None:
                rank = comm.Get_rank()
                size = comm.Get_size()
                self.logger.info("Rank %d: Completed jobs %s",rank,list(self.results.keys()))
                # Send all the results back to rank 0 process.
                if rank > 0:
                    comm.send(self, dest=0)
                else:
                    for p in range(1,size):
                        temp = comm.recv(source=p)
                        self += temp
                        self.results.update(temp.results)

    def _process_all_cross12(self, cat1, cat2, metric, num_threads, comm=None, low_mem=False):

        def is_my_job(my_indices, i, j, k, n1, n2):
            # Helper function to figure out if a given (i,j,k) job should be done on the
            # current process.

            # Always my job if not using MPI.
            if my_indices is None:
                return True

            # If n1 is n, then this can be simple.  Just split according to i.
            n = max(n1,n2)
            if n1 == n:
                if i in my_indices:
                    self.logger.info("Rank %d: Job (%d,%d,%d) is mine.",rank,i,j,k)
                    return True
                else:
                    return False

            # If not, then this looks like the decision for 2pt auto using j,k.
            if j in my_indices and k in my_indices:
                self.logger.info("Rank %d: Job (%d,%d,%d) is mine.",rank,i,j,k)
                return True

            if j not in my_indices and k not in my_indices:
                return False

            if k-j < n//2:
                ret = j % 2 == (0 if j in my_indices else 1)
            else:
                ret = k % 2 == (0 if k in my_indices else 1)
            if ret:
                self.logger.info("Rank %d: Job (%d,%d,%d) is mine.",rank,i,j,k)
            return ret

        if len(cat1) == 1 and len(cat2) == 1 and cat1[0].npatch == 1 and cat2[0].npatch == 1:
            self.process_cross12(cat1[0], cat2[0], metric=metric, num_threads=num_threads)
        else:
            # When patch processing, keep track of the pair-wise results.
            if self.npatch1 == 1:
                self.npatch1 = cat1[0].npatch if cat1[0].npatch != 1 else len(cat1)
            if self.npatch2 == 1:
                self.npatch2 = cat2[0].npatch if cat2[0].npatch != 1 else len(cat2)
                self.npatch3 = self.npatch2
            if self.npatch1 != self.npatch2 and self.npatch1 != 1 and self.npatch2 != 1:
                raise RuntimeError("Cross correlation requires both catalogs use the same patches.")

            # Setup for deciding when this is my job.
            n1 = self.npatch1
            n2 = self.npatch2
            if comm:
                size = comm.Get_size()
                rank = comm.Get_rank()
                n = max(n1,n2)
                my_indices = np.arange(n * rank // size, n * (rank+1) // size)
                self.logger.info("Rank %d: My indices are %s",rank,my_indices)
            else:
                my_indices = None

            temp = self.copy()
            for ii,c1 in enumerate(cat1):
                i = c1.patch if c1.patch is not None else ii
                for jj,c2 in enumerate(cat2):
                    j = c2.patch if c2.patch is not None else jj
                    if is_my_job(my_indices, i, i, j, n1, n2):
                        temp.clear()
                        # One point in c1, 2 in c2.
                        if not self._trivially_zero(c1,c2,c2,metric):
                            self.logger.info('Process patches %d,%d cross12',i,j)
                            temp.process_cross12(c1, c2, metric=metric, num_threads=num_threads)
                        else:
                            self.logger.info('Skipping %d,%d pair, which are too far apart ' +
                                             'for this set of separations',i,j)
                        if temp.nonzero or i==j or n1==1 or n2==1:
                            if (i,j,j) in self.results and self.results[(i,j,j)].nonzero:
                                self.results[(i,j,j)] += temp
                            else:
                                self.results[(i,j,j)] = temp.copy()
                            self += temp
                        else:
                            # NNNCorrelation needs to add the tot value
                            self._add_tot(i, j, j, c1, c2, c2)

                    # One point in each of c1, c2, c3
                    for kk,c3 in list(enumerate(cat2))[::-1]:
                        k = c3.patch if c3.patch is not None else kk
                        if j < k and is_my_job(my_indices, i, j, k, n1, n2):
                            temp.clear()

                            if not self._trivially_zero(c1,c2,c3,metric):
                                self.logger.info('Process patches %d,%d,%d cross',i,j,k)
                                temp.process_cross(c1, c2, c3, metric=metric,
                                                   num_threads=num_threads)
                            else:
                                self.logger.info('Skipping %d,%d,%d, which are too far apart ' +
                                                 'for this set of separations',i,j,k)
                            if temp.nonzero:
                                if (i,j,k) in self.results and self.results[(i,j,k)].nonzero:
                                    self.results[(i,j,k)] += temp
                                else:
                                    self.results[(i,j,k)] = temp.copy()
                                self += temp
                            else:
                                # NNNCorrelation needs to add the tot value
                                self._add_tot(i, j, k, c1, c2, c3)
                            if low_mem:
                                c3.unload()

                    if low_mem and jj != ii+1:
                        # Don't unload i+1, since that's the next one we'll need.
                        c2.unload()
                if low_mem:
                    c1.unload()
            if comm is not None:
                rank = comm.Get_rank()
                size = comm.Get_size()
                self.logger.info("Rank %d: Completed jobs %s",rank,list(self.results.keys()))
                # Send all the results back to rank 0 process.
                if rank > 0:
                    comm.send(self, dest=0)
                else:
                    for p in range(1,size):
                        temp = comm.recv(source=p)
                        self += temp
                        self.results.update(temp.results)

    def _process_all_cross(self, cat1, cat2, cat3, metric, num_threads, comm=None, low_mem=False):

        def is_my_job(my_indices, i, j, k, n1, n2, n3):
            # Helper function to figure out if a given (i,j,k) job should be done on the
            # current process.

            # Always my job if not using MPI.
            if my_indices is None:
                return True

            # Just split up according to one of the catalogs.
            n = max(n1,n2,n3)
            if n1 == n:
                m = i
            elif n2 == n:
                m = j
            else:
                m = k
            if m in my_indices:
                self.logger.info("Rank %d: Job (%d,%d,%d) is mine.",rank,i,j,k)
                return True
            else:
                return False

        if (len(cat1) == 1 and len(cat2) == 1 and len(cat3) == 1 and
                cat1[0].npatch == 1 and cat2[0].npatch == 1 and cat3[0].npatch == 1):
            self.process_cross(cat1[0], cat2[0], cat3[0], metric=metric, num_threads=num_threads)
        else:
            # When patch processing, keep track of the pair-wise results.
            if self.npatch1 == 1:
                self.npatch1 = cat1[0].npatch if cat1[0].npatch != 1 else len(cat1)
            if self.npatch2 == 1:
                self.npatch2 = cat2[0].npatch if cat2[0].npatch != 1 else len(cat2)
            if self.npatch3 == 1:
                self.npatch3 = cat3[0].npatch if cat3[0].npatch != 1 else len(cat3)
            if self.npatch1 != self.npatch2 and self.npatch1 != 1 and self.npatch2 != 1:
                raise RuntimeError("Cross correlation requires all catalogs use the same patches.")
            if self.npatch1 != self.npatch3 and self.npatch1 != 1 and self.npatch3 != 1:
                raise RuntimeError("Cross correlation requires all catalogs use the same patches.")

            # Setup for deciding when this is my job.
            n1 = self.npatch1
            n2 = self.npatch2
            n3 = self.npatch3
            if comm:
                size = comm.Get_size()
                rank = comm.Get_rank()
                n = max(n1,n2,n3)
                my_indices = np.arange(n * rank // size, n * (rank+1) // size)
                self.logger.info("Rank %d: My indices are %s",rank,my_indices)
            else:
                my_indices = None

            temp = self.copy()
            for ii,c1 in enumerate(cat1):
                i = c1.patch if c1.patch is not None else ii
                for jj,c2 in enumerate(cat2):
                    j = c2.patch if c2.patch is not None else jj
                    for kk,c3 in enumerate(cat3):
                        k = c3.patch if c3.patch is not None else kk
                        if is_my_job(my_indices, i, j, k, n1, n2, n3):
                            temp.clear()
                            if not self._trivially_zero(c1,c2,c3,metric):
                                self.logger.info('Process patches %d,%d,%d cross',i,j,k)
                                temp.process_cross(c1, c2, c3, metric=metric,
                                                   num_threads=num_threads)
                            else:
                                self.logger.info('Skipping %d,%d,%d, which are too far apart ' +
                                                 'for this set of separations',i,j,k)
                            if (temp.nonzero or (i==j==k)
                                    or (i==j and n3==1) or (i==k and n2==1) or (j==k and n1==1)
                                    or (n1==n2==1) or (n1==n3==1) or (n2==n3==1)):
                                if (i,j,k) in self.results and self.results[(i,j,k)].nonzero:
                                    self.results[(i,j,k)] += temp
                                else:
                                    self.results[(i,j,k)] = temp.copy()
                                self += temp
                            else:
                                # NNNCorrelation needs to add the tot value
                                self._add_tot(i, j, k, c1, c2, c3)
                            if low_mem:
                                c3.unload()
                    if low_mem and jj != ii+1:
                        # Don't unload i+1, since that's the next one we'll need.
                        c2.unload()
                if low_mem:
                    c1.unload()
            if comm is not None:
                rank = comm.Get_rank()
                size = comm.Get_size()
                self.logger.info("Rank %d: Completed jobs %s",rank,list(self.results.keys()))
                # Send all the results back to rank 0 process.
                if rank > 0:
                    comm.send(self, dest=0)
                else:
                    for p in range(1,size):
                        temp = comm.recv(source=p)
                        self += temp
                        self.results.update(temp.results)

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        Usually, this is just self.zeta.  But if the metric is TwoD, this becomes
        self.zeta.ravel().

        And for `GGGCorrelation`, it is the concatenation of the four different correlations
        [gam0.ravel(), gam1.ravel(), gam2.ravel(), gam3.ravel()].
        """
        return self.zeta.ravel()

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        This is the weight array corresponding to `getStat`. Usually just self.weight, but
        raveled for TwoD and duplicated for GGGCorrelation to match what `getStat` does in
        those cases.
        """
        return self.weight.ravel()

    @depr_pos_kwargs
    def estimate_cov(self, method, *, func=None, comm=None):
        """Estimate the covariance matrix based on the data

        This function will calculate an estimate of the covariance matrix according to the
        given method.

        Options for ``method`` include:

            - 'shot' = The variance based on "shot noise" only.  This includes the Poisson
              counts of points for N statistics, shape noise for G statistics, and the observed
              scatter in the values for K statistics.  In this case, the returned covariance
              matrix will be diagonal, since there is no way to estimate the off-diagonal terms.
            - 'jackknife' = A jackknife estimate of the covariance matrix based on the scatter
              in the measurement when excluding one patch at a time.
            - 'sample' = An estimate based on the sample covariance of a set of samples,
              taken as the patches of the input catalog.
            - 'bootstrap' = A bootstrap covariance estimate. It selects patches at random with
              replacement and then generates the statistic using all the auto-correlations at
              their selected repetition plus all the cross terms that aren't actually auto terms.
            - 'marked_bootstrap' = An estimate based on a marked-point bootstrap resampling of the
              patches.  Similar to bootstrap, but only samples the patches of the first catalog and
              uses all patches from the second catalog that correspond to each patch selection of
              the first catalog.  cf. https://ui.adsabs.harvard.edu/abs/2008ApJ...681..726L/

        Both 'bootstrap' and 'marked_bootstrap' use the num_bootstrap parameter, which can be set on
        construction.

        .. note::

            For most classes, there is only a single statistic, ``zeta``, so this calculates a
            covariance matrix for that vector.  `GGGCorrelation` has four: ``gam0``, ``gam1``,
            ``gam2``, and ``gam3``, so in this case the full data vector is ``gam0`` followed by
            ``gam1``, then ``gam2``, then ``gam3``, and this calculates the covariance matrix for
            that full vector including both statistics.  The helper function `getStat` returns the
            relevant statistic in all cases.

        In all cases, the relevant processing needs to already have been completed and finalized.
        And for all methods other than 'shot', the processing should have involved an appropriate
        number of patches -- preferably more patches than the length of the vector for your
        statistic, although this is not checked.

        The default data vector to use for the covariance matrix is given by the method
        `getStat`.  As noted above, this is usually just self.zeta.  However, there is an option
        to compute the covariance of some other function of the correlation object by providing
        an arbitrary function, ``func``, which should act on the current correlation object
        and return the data vector of interest.

        For instance, for an `GGGCorrelation`, you might want to compute the covariance of just
        gam0 and ignore the others.  In this case you could use

            >>> func = lambda ggg: ggg.gam0

        The return value from this func should be a single numpy array. (This is not directly
        checked, but you'll probably get some kind of exception if it doesn't behave as expected.)

        .. note::

            The optional ``func`` parameter is not valid in conjunction with ``method='shot'``.
            It only works for the methods that are based on patch combinations.

        This function can be parallelized by passing the comm argument as an mpi4py communicator
        to parallelize using that.  For MPI, all processes should have the same inputs.
        If method == "shot" then parallelization has no effect.

        Parameters:
            method (str):       Which method to use to estimate the covariance matrix.
            func (function):    A unary function that acts on the current correlation object and
                                returns the desired data vector. [default: None, which is
                                equivalent to ``lambda corr: corr.getStat()``.
            comm (mpi comm)     If not None, run under MPI

        Returns:
            A numpy array with the estimated covariance matrix.
        """
        if func is not None:
            # Need to convert it to a function of the first item in the list.
            all_func = lambda corrs: func(corrs[0])
        else:
            all_func = None
        return estimate_multi_cov([self], method, func=all_func, comm=comm)

    def build_cov_design_matrix(self, method, *, func=None, comm=None):
        """Build the design matrix that is used for estimating the covariance matrix.

        The design matrix for patch-based covariance estimates is a matrix where each row
        corresponds to a different estimate of the data vector, :math:`\\zeta_i` (or
        :math:`f(\\zeta_i)` if using the optional ``func`` parameter).

        The different of rows in the matrix for each valid ``method`` are:

            - 'shot': This method is not valid here.
            - 'jackknife': The data vector when excluding a single patch.
            - 'sample': The data vector using only a single patch for the first catalog.
            - 'bootstrap': The data vector for a random resampling of the patches keeping the
              sample total number, but allowing some to repeat.  Cross terms from repeated patches
              are excluded (since they are really auto terms).
            - 'marked_bootstrap': The data vector for a random resampling of patches in the first
              catalog, using all patches for the second catalog.  Based on the algorithm in
              Loh(2008).

        See `estimate_cov` for more details.

        The return value includes both the design matrix and a vector of weights (the total weight
        array in the computed correlation functions).  The weights are used for the sample method
        when estimating the covariance matrix.  The other methods ignore them, but they are provided
        here in case they are useful.

        Parameters:
            method (str):       Which method to use to estimate the covariance matrix.
            func (function):    A unary function that takes the list ``corrs`` and returns the
                                desired full data vector. [default: None, which is equivalent to
                                ``lambda corrs: np.concatenate([c.getStat() for c in corrs])``]
            comm (mpi comm)     If not None, run under MPI

        Returns:
            A, w: numpy arrays with the design matrix and weights respectively.
        """
        if func is not None:
            # Need to convert it to a function of the first item in the list.
            all_func = lambda corrs: func(corrs[0])
        else:
            all_func = None
        return build_multi_cov_design_matrix([self], method=method, func=all_func, comm=comm)

    def _set_num_threads(self, num_threads):
        if num_threads is None:
            num_threads = self.config.get('num_threads',None)
        if num_threads is None:
            self.logger.debug('Set num_threads automatically from ncpu')
        else:
            self.logger.debug('Set num_threads = %d',num_threads)
        set_omp_threads(num_threads, self.logger)

    def _set_metric(self, metric, coords1, coords2=None, coords3=None):
        if metric is None:
            metric = get(self.config,'metric',str,'Euclidean')
        coords, metric = parse_metric(metric, coords1, coords2, coords3)
        if self.coords is not None or self.metric is not None:
            if coords != self.coords:
                self.logger.warning("Detected a change in catalog coordinate systems. "+
                                    "This probably doesn't make sense!")
            if metric != self.metric:
                self.logger.warning("Detected a change in metric. "+
                                    "This probably doesn't make sense!")
        if metric == 'Periodic':
            if self.xperiod == 0 or self.yperiod == 0 or (coords=='3d' and self.zperiod == 0):
                raise ValueError("Periodic metric requires setting the period to use.")
        else:
            if self.xperiod != 0 or self.yperiod != 0 or self.zperiod != 0:
                raise ValueError("period options are not valid for %s metric."%metric)
        self.coords = coords
        self.metric = metric
        self._coords = coord_enum(coords)
        self._metric = metric_enum(metric)

    def _apply_units(self, mask):
        if self.coords == 'spherical' and self.metric == 'Euclidean':
            # Then our distances are all angles.  Convert from the chord distance to a real angle.
            # L = 2 sin(theta/2)
            self.meand1[mask] = 2. * np.arcsin(self.meand1[mask]/2.)
            self.meanlogd1[mask] = np.log(2.*np.arcsin(np.exp(self.meanlogd1[mask])/2.))
            self.meand2[mask] = 2. * np.arcsin(self.meand2[mask]/2.)
            self.meanlogd2[mask] = np.log(2.*np.arcsin(np.exp(self.meanlogd2[mask])/2.))
            self.meand3[mask] = 2. * np.arcsin(self.meand3[mask]/2.)
            self.meanlogd3[mask] = np.log(2.*np.arcsin(np.exp(self.meanlogd3[mask])/2.))

        self.meand1[mask] /= self._sep_units
        self.meanlogd1[mask] -= self._log_sep_units
        self.meand2[mask] /= self._sep_units
        self.meanlogd2[mask] -= self._log_sep_units
        self.meand3[mask] /= self._sep_units
        self.meanlogd3[mask] -= self._log_sep_units

    def _get_minmax_size(self):
        if self.metric == 'Euclidean':
            # The minimum separation we care about is that of the smallest size, which is
            # min_sep * min_u.  Do the same calculation as for 2pt to get to min_size.
            b1 = min(self.b, self.bu, self.bv)
            min_size = self._min_sep * self.min_u * b1 / (2.+3.*b1)

            # This time, the maximum size is d1 * b.  d1 can be as high as 2*max_sep.
            b2 = max(self.b, self.bu, self.bv)
            max_size = 2. * self._max_sep * b2
            return min_size, max_size
        else:
            return 0., 0.

    # The three-point versions of the covariance helpers.
    # Note: the word "pairs" in many of these was appropriate for 2pt, but in the 3pt case
    # these actually refer to triples (i,j,k).

    def _get_npatch(self):
        return max(self.npatch1, self.npatch2, self.npatch3)

    def _calculate_xi_from_pairs(self, pairs):
        # Compute the xi data vector for the given list of pairs.
        # pairs is input as a list of (i,j) values.

        # This is the normal calculation.  It needs to be overridden when there are randoms.
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()

    def _jackknife_pairs(self):
        if self.npatch3 == 1:
            if self.npatch2 == 1:
                # k=m=0
                return [ [(j,k,m) for j,k,m in self.results.keys() if j!=i]
                         for i in range(self.npatch1) ]
            elif self.npatch1 == 1:
                # j=m=0
                return [ [(j,k,m) for j,k,m in self.results.keys() if k!=i]
                         for i in range(self.npatch2) ]
            else:
                # m=0
                assert self.npatch1 == self.npatch2
                return [ [(j,k,m) for j,k,m in self.results.keys() if j!=i and k!=i]
                         for i in range(self.npatch1) ]
        elif self.npatch2 == 1:
            if self.npatch1 == 1:
                # j=k=0
                return [ [(j,k,m) for j,k,m in self.results.keys() if m!=i]
                         for i in range(self.npatch3) ]
            else:
                # k=0
                assert self.npatch1 == self.npatch3
                return [ [(j,k,m) for j,k,m in self.results.keys() if j!=i and m!=i]
                         for i in range(self.npatch1) ]
        elif self.npatch1 == 1:
            # j=0
            assert self.npatch2 == self.npatch3
            return [ [(j,k,m) for j,k,m in self.results.keys() if k!=i and m!=i]
                     for i in range(self.npatch2) ]
        else:
            assert self.npatch1 == self.npatch2 == self.npatch3
            return [ [(j,k,m) for j,k,m in self.results.keys() if j!=i and k!=i and m!=i]
                     for i in range(self.npatch1) ]

    def _sample_pairs(self):
        if self.npatch3 == 1:
            if self.npatch2 == 1:
                # k=m=0
                return [ [(j,k,m) for j,k,m in self.results.keys() if j==i]
                         for i in range(self.npatch1) ]
            elif self.npatch1 == 1:
                # j=m=0
                return [ [(j,k,m) for j,k,m in self.results.keys() if k==i]
                         for i in range(self.npatch2) ]
            else:
                # m=0
                assert self.npatch1 == self.npatch2
                return [ [(j,k,m) for j,k,m in self.results.keys() if j==i]
                         for i in range(self.npatch1) ]
        elif self.npatch2 == 1:
            if self.npatch1 == 1:
                # j=k=0
                return [ [(j,k,m) for j,k,m in self.results.keys() if m==i]
                         for i in range(self.npatch3) ]
            else:
                # k=0
                assert self.npatch1 == self.npatch3
                return [ [(j,k,m) for j,k,m in self.results.keys() if j==i]
                         for i in range(self.npatch1) ]
        elif self.npatch1 == 1:
            # j=0
            assert self.npatch2 == self.npatch3
            return [ [(j,k,m) for j,k,m in self.results.keys() if k==i]
                     for i in range(self.npatch2) ]
        else:
            assert self.npatch1 == self.npatch2 == self.npatch3
            return [ [(j,k,m) for j,k,m in self.results.keys() if j==i]
                     for i in range(self.npatch1) ]

    @lazy_property
    def _ok(self):
        ok = np.zeros((self.npatch1, self.npatch2, self.npatch3), dtype=bool)
        for (i,j,k) in self.results:
            ok[i,j,k] = True
        return ok

    def _marked_pairs(self, indx):
        if self.npatch3 == 1:
            if self.npatch2 == 1:
                return [ (i,0,0) for i in indx if self._ok[i,0,0] ]
            elif self.npatch1 == 1:
                return [ (0,i,0) for i in indx if self._ok[0,i,0] ]
            else:
                assert self.npatch1 == self.npatch2
                # Select all pairs where first point is in indx (repeating i as appropriate)
                return [ (i,j,0) for i in indx for j in range(self.npatch2) if self._ok[i,j,0] ]
        elif self.npatch2 == 1:
            if self.npatch1 == 1:
                return [ (0,0,i) for i in indx if self._ok[0,0,i] ]
            else:
                assert self.npatch1 == self.npatch3
                # Select all pairs where first point is in indx (repeating i as appropriate)
                return [ (i,0,j) for i in indx for j in range(self.npatch3) if self._ok[i,0,j] ]
        elif self.npatch1 == 1:
            assert self.npatch2 == self.npatch3
            # Select all pairs where first point is in indx (repeating i as appropriate)
            return [ (0,i,j) for i in indx for j in range(self.npatch3) if self._ok[0,i,j] ]
        else:
            assert self.npatch1 == self.npatch2 == self.npatch3
            # Select all pairs where first point is in indx (repeating i as appropriate)
            return [ (i,j,k) for i in indx for j in range(self.npatch2)
                                           for k in range(self.npatch3) if self._ok[i,j,k] ]

    def _bootstrap_pairs(self, indx):
        if self.npatch3 == 1:
            if self.npatch2 == 1:
                return [ (i,0,0) for i in indx if self._ok[i,0,0] ]
            elif self.npatch1 == 1:
                return [ (0,i,0) for i in indx if self._ok[0,i,0] ]
            else:
                assert self.npatch1 == self.npatch2
                return ([ (i,i,0) for i in indx if self._ok[i,i,0] ] +
                        [ (i,j,0) for i in indx for j in indx if self._ok[i,j,0] and i!=j ])
        elif self.npatch2 == 1:
            if self.npatch1 == 1:
                return [ (0,0,i) for i in indx if self._ok[0,0,i] ]
            else:
                assert self.npatch1 == self.npatch3
                return ([ (i,0,i) for i in indx if self._ok[i,0,i] ] +
                        [ (i,0,j) for i in indx for j in indx if self._ok[i,0,j] and i!=j ])
        elif self.npatch1 == 1:
            assert self.npatch2 == self.npatch3
            return ([ (0,i,i) for i in indx if self._ok[0,i,i] ] +
                    [ (0,i,j) for i in indx for j in indx if self._ok[0,i,j] and i!=j ])
        else:
            # Like for 2pt we want to avoid getting extra copies of what are actually
            # auto-correlations coming from two indices equalling each other in (i,j,k).
            # This time, get each (i,i,i) once.
            # Then get (i,i,j), (i,j,i), and (j,i,i) once per each (i,j) pair with i!=j
            # repeated as often as they show up in the double for loop.
            # Finally get all triples (i,j,k) where they are all different repeated as often
            # as they show up in the triple for loop.
            assert self.npatch1 == self.npatch2 == self.npatch3
            return ([ (i,i,i) for i in indx if self._ok[i,i,i] ] +
                    [ (i,i,j) for i in indx for j in indx if self._ok[i,i,j] and i!=j ] +
                    [ (i,j,i) for i in indx for j in indx if self._ok[i,j,i] and i!=j ] +
                    [ (j,i,i) for i in indx for j in indx if self._ok[j,i,i] and i!=j ] +
                    [ (i,j,k) for i in indx for j in indx if i!=j
                              for k in indx if self._ok[i,j,k] and (i!=k and j!=k) ])

    def _write(self, writer, name, write_patch_results, zero_tot=False):
        # These helper properties define what to write for each class.
        col_names = self._write_col_names
        data = self._write_data
        params = self._write_params
        params['num_rows'] = len(self.rnom.ravel())

        if write_patch_results:
            # Note: Only include npatch1, npatch2 in serialization if we are also serializing
            # results.  Otherwise, the corr that is read in will behave oddly.
            params['npatch1'] = self.npatch1
            params['npatch2'] = self.npatch2
            params['npatch3'] = self.npatch3
            num_patch_tri = len(self.results)
            if zero_tot:
                i = 0
                for key, corr in self.results.items():
                    if not corr._nonzero:
                        zp_name = name + '_zp_%d'%i
                        params[zp_name] = repr((key, corr.tot))
                        num_patch_tri -= 1
                        i += 1
                params['num_zero_patch'] = i
            params['num_patch_tri'] = num_patch_tri

        writer.write(col_names, data, params=params, ext=name)
        if write_patch_results:
            writer.set_precision(16)
            i = 0
            for key, corr in self.results.items():
                if zero_tot and not corr._nonzero: continue
                col_names = corr._write_col_names
                data = corr._write_data
                params = corr._write_params
                params['key'] = repr(key)
                pp_name = name + '_pp_%d'%i
                writer.write(col_names, data, params=params, ext=pp_name)
                i += 1
            assert i == num_patch_tri

    def _read(self, reader, name=None):
        name = 'main' if 'main' in reader and name is None else name
        params = reader.read_params(ext=name)
        num_rows = params.get('num_rows', None)
        num_patch_tri = params.get('num_patch_tri', 0)
        num_zero_patch = params.get('num_zero_patch', 0)
        name = 'main' if num_patch_tri and name is None else name
        data = reader.read_data(max_rows=num_rows, ext=name)

        # This helper function defines how to set the attributes for each class
        # based on what was read in.
        self._read_from_data(data, params)

        self.results = {}
        for i in range(num_zero_patch):
            zp_name = name + '_zp_%d'%i
            key, tot = eval(params[zp_name])
            self.results[key] = self._zero_copy(tot)
        for i in range(num_patch_tri):
            pp_name = name + '_pp_%d'%i
            corr = self.copy()
            params = reader.read_params(ext=pp_name)
            data = reader.read_data(max_rows=num_rows, ext=pp_name)
            corr._read_from_data(data, params)
            key = eval(params['key'])
            self.results[key] = corr

# Copyright (c) 2003-2024 by Mike Jarvis
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
.. module:: corr3base
"""

import math
import numpy as np
import sys
import itertools
import collections
import coord

from . import _treecorr
from .config import merge_config, setup_logger, get, make_minimal_config
from .util import parse_metric, metric_enum, coord_enum, set_omp_threads, lazy_property
from .util import make_writer, make_reader, spin_by_letter
from .corr2base import estimate_multi_cov, build_multi_cov_design_matrix, Corr2
from .catalog import Catalog, calculateMeanW

class Namespace(object):
    pass

class Corr3(object):
    r"""This class stores the results of a 3-point correlation calculation, along with some
    ancillary data.

    This is a base class that is not intended to be constructed directly.  But it has a few
    helper functions that derived classes can use to help perform their calculations.  See
    the derived classes for more details:

    - `NNNCorrelation` handles count-count-count correlation functions
    - `KKKCorrelation` handles scalar-scalar-scalar correlation functions
    - `GGGCorrelation` handles shear-shear-shear correlation functions

    Three-point correlations are a bit more complicated than two-point, since the data need
    to be binned in triangles, not just the separation between two points.

    There are currenlty three different ways to quantify the triangle shapes.

    1. The triangle can be defined by its three side lengths (i.e. SSS congruence).
       In this case, we characterize the triangles according to the following three parameters
       based on the three side lengths with d1 >= d2 >= d3.

        .. math::

            r &= d2 \\
            u &= \frac{d3}{d2} \\
            v &= \pm \frac{(d1 - d2)}{d3} \\

        The orientation of the triangle is specified by the sign of v.
        Positive v triangles have the three sides d1,d2,d3 in counter-clockwise orientation.
        Negative v triangles have the three sides d1,d2,d3 in clockwise orientation.

        .. note::

            We always bin the same way for positive and negative v values, and the binning
            specification for v should just be for the positive values.  E.g. if you specify
            min_v=0.2, max_v=0.6, then TreeCorr will also accumulate triangles with
            -0.6 < v < -0.2 in addition to those with 0.2 < v < 0.6.

    2. The triangle can be defined by two of the sides and the angle between them (i.e. SAS
       congruence).  The vertex point between the two sides is considered point "1" (P1), so
       the two sides (opposite points 2 and 3) are called d2 and d3.  The angle between them is
       called phi, and it is measured in radians.

       The orientation is defined such that 0 <= phi <= pi is the angle sweeping from d2 to d3
       counter-clockwise.

       Unlike the SSS definition where every triangle is uniquely placed in a single bin, this
       definition forms a triangle with each object at the central vertex, P1, so for
       auto-correlations, each triangle is placed in bins three times.  For cross-correlations,
       the order of the points is such that objects in the first catalog are at the central
       vertex, P1, objects in the second catalog are at P2, which is opposite d2 (i.e.
       at the end of line segment d3 from P1), and objects in the third catalog are at P3,
       opposite d3 (i.e. at the end of d2 from P1).

    3. The third option is a multipole expansion of the SAS description.  This idea was initially
       developed by Chen & Szapudi (2005, ApJ, 635, 743) and then further refined by
       Slepian & Eisenstein (2015, MNRAS, 454, 4142), Philcox et al (2022, MNRAS, 509, 2457),
       and Porth et al (2024, A&A, 689, 227).  The latter in particular showed how to use this
       method for non-spin-0 correlations (GGG in particular).

       The basic idea is to do a Fourier transform of the phi binning to convert the phi
       bins into n bins.

       .. math::

           \zeta(d_2, d_3, \phi) = \frac{1}{2\pi} \sum_n \mathcal{Z}_n(d_2,d_3) e^{i n \phi}

       Formally, this is exact if the sum goes from :math:`-\infty .. \infty`.  Truncating this
       sum at :math:`\pm n_\mathrm{max}` is similar to binning in theta with this many bins
       for :math:`\phi` within the range :math:`0 <= \phi <= \pi`.

       The above papers show that this multipole expansion allows for a much more efficient
       calculation, since it can be done with a kind of 2-point calculation.
       We provide methods to convert the multipole output into the SAS binning if desired, since
       that is often more convenient in practice.

    The constructor for all derived classes take a config dict as the first argument,
    since this is often how we keep track of parameters, but if you don't want to
    use one or if you want to change some parameters from what are in a config dict,
    then you can use normal kwargs, which take precedence over anything in the config dict.

    There are a number of possible definitions for the distance between two points, which
    are appropriate for different use cases.  These are specified by the ``metric`` parameter.
    The possible options are:

        - 'Euclidean' = straight line Euclidean distance between two points.  For spherical
          coordinates (ra,dec without r), this is the chord distance between points on the
          unit sphere.
        - 'FisherRperp' = the perpendicular component of the distance, following the
          definitions in Fisher et al, 1994 (MNRAS, 267, 927).
        - 'OldRperp' = the perpendicular component of the distance using the definition
          of Rperp from TreeCorr v3.x.
        - 'Rperp' = an alias for FisherRperp.  You can change it to be an alias for
          OldRperp if you want by setting ``treecorr.Rperp_alias = 'OldRperp'`` before
          using it.
        - 'Rlens' = the distance from the first object (taken to be a lens) to the line
          connecting Earth and each of the other two objects (taken to be lensed sources).
        - 'Arc' = the true great circle distance for spherical coordinates.
        - 'Periodic' = Like Euclidean, but with periodic boundaries.

          .. note::

            The triangles for three-point correlations can become ambiguous if a triangle
            side length d > period/2, which means for the SSS triangle definition, max_sep
            (the maximum d2) should be less than period/4, and for SAS, max_sep should be less
            than period/2.  This is not enforced.

    See `Metrics` for more information about these various metric options.

    There are three allowed value for the ``bin_type`` for three-point correlations.

        - 'LogRUV' uses the SSS description given above converted to r,u,v. The bin steps will be
          uniform in log(r) from log(min_sep) .. log(max_sep).  The u and v values are binned
          linearly from min_u .. max_u and min_v .. max_v.
        - 'LogSAS' uses the SAS description given above. The bin steps will be uniform in log(d)
          for both d2 and d3 from log(min_sep) .. log(max_sep).  The phi values are binned
          linearly from min_phi .. max_phi.  This is the default.
        - 'LogMultipole' uses the multipole description given above. The bin steps will be uniform
          in log(d) for both d2 and d3 from log(min_sep) .. log(max_sep), and the n value range
          from -max_n .. max_n, inclusive.

    Objects of any `Corr3` subclass hold the following attributes:

    Attributes:
        nbins:      The number of bins in logr where r = d2.
        bin_size:   The size of the bins in logr.
        min_sep:    The minimum separation being considered.
        max_sep:    The maximum separation being considered.
        logr1d:     The nominal centers of the nbins bins in log(r).

    If the bin_type is LogRUV, then it will have these attributes:

    Attributes:
        nubins:     The number of bins in u where u = d3/d2.
        ubin_size:  The size of the bins in u.
        min_u:      The minimum u being considered.
        max_u:      The maximum u being considered.
        nvbins:     The number of bins in v where v = +-(d1-d2)/d3.
        vbin_size:  The size of the bins in v.
        min_v:      The minimum v being considered.
        max_v:      The maximum v being considered.
        u1d:        The nominal centers of the nubins bins in u.
        v1d:        The nominal centers of the nvbins bins in v.

    If the bin_type is LogSAS, then it will have these attributes:

    Attributes:
        nphi_bins:  The number of bins in phi.
        phi_bin_size: The size of the bins in phi.
        min_phi:    The minimum phi being considered.
        max_phi:    The maximum phi being considered.
        phi1d:      The nominal centers of the nphi_bins bins in phi.

    If the bin_type is LogMultipole, then it will have these attributes:

    Attributes:
        max_n:      The maximum multipole index n being stored.
        n1d:        The multipole index n in the 2*max_n+1 bins of the third bin direction.

    In addition, the following attributes are numpy arrays whose shape is:

        * (nbins, nubins, nvbins) if bin_type is LogRUV
        * (nbins, nbins, nphi_bins) if bin_type is LogSAS
        * (nbins, nbins, 2*max_n+1) if bin_type is LogMultipole

    If bin_type is LogRUV:

    Attributes:
        logr:       The nominal center of each bin in log(r).
        rnom:       The nominal center of each bin converted to regular distance.
                    i.e. r = exp(logr).
        u:          The nominal center of each bin in u.
        v:          The nominal center of each bin in v.
        meanu:      The (weighted) mean value of u for the triangles in each bin.
        meanv:      The (weighted) mean value of v for the triangles in each bin.

    If bin_type is LogSAS:

    Attributes:
        logd2:      The nominal center of each bin in log(d2).
        d2nom:      The nominal center of each bin converted to regular d2 distance.
                    i.e. d2 = exp(logd2).
        logd3:      The nominal center of each bin in log(d3).
        d3nom:      The nominal center of each bin converted to regular d3 distance.
                    i.e. d3 = exp(logd3).
        phi:        The nominal center of each angular bin.
        meanphi:    The (weighted) mean value of phi for the triangles in each bin.

    If bin_type is LogMultipole:

    Attributes:
        logd2:      The nominal center of each bin in log(d2).
        d2nom:      The nominal center of each bin converted to regular d2 distance.
                    i.e. d2 = exp(logd2).
        logd3:      The nominal center of each bin in log(d3).
        d3nom:      The nominal center of each bin converted to regular d3 distance.
                    i.e. d3 = exp(logd3).
        n:          The multipole index n for each bin.

    For any bin_type:

    Attributes:
        meand1:     The (weighted) mean value of d1 for the triangles in each bin.
        meanlogd1:  The (weighted) mean value of log(d1) for the triangles in each bin.
        meand2:     The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        meanlogd2:  The (weighted) mean value of log(d2) for the triangles in each bin.
        meand3:     The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd3:  The (weighted) mean value of log(d3) for the triangles in each bin.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_auto`
        and/or `process_cross`, then the units will not be applied to ``meanr`` or
        ``meanlogr`` until the ``finalize`` function is called.

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
                            which has been found to yield good results for most application.)
        angle_slop (float): How much slop to allow in the angular direction. This works very
                            similarly to bin_slop, but applies to the projection angle of a pair
                            of cells. The projection angle for any two objects in a pair of cells
                            will differ by no more than angle_slop radians from the projection
                            angle defined by the centers of the cells. (default: 0.1)
        brute (bool):       Whether to use the "brute force" algorithm.  (default: False) Options
                            are:

                             - False (the default): Stop at non-leaf cells whenever the error in
                               the separation is compatible with the given bin_slop and angle_slop.
                             - True: Go to the leaves for both catalogs.
                             - 1: Always go to the leaves for cat1, but stop at non-leaf cells of
                               cat2 when the error is compatible with the given slop values.
                             - 2: Always go to the leaves for cat2, but stop at non-leaf cells of
                               cat1 when the error is compatible with the given slop values.


        nphi_bins (int):    Analogous to nbins for the phi values when bin_type=LogSAS.  (The
                            default is to calculate from phi_bin_size = bin_size, min_phi = 0,
                            max_u = np.pi, but this can be overridden by specifying up to 3 of
                            these four parametes.)
        phi_bin_size (float): Analogous to bin_size for the phi values. (default: bin_size)
        min_phi (float):    Analogous to min_sep for the phi values. (default: 0)
        max_phi (float):    Analogous to max_sep for the phi values. (default: np.pi)
        phi_units (str):    The units to use for the phi values, given as a string.  This
                            includes both min_phi and max_phi above, as well as the units of the
                            output meanphi values.  Valid options are arcsec, arcmin, degrees,
                            hours, radians.  (default: radians)

        max_n (int):        The maximum value of n to store for the multipole binning.
                            (required if bin_type=LogMultipole)

        nubins (int):       Analogous to nbins for the u values when bin_type=LogRUV.  (The default
                            is to calculate from ubin_size = bin_size, min_u = 0, max_u = 1, but
                            this can be overridden by specifying up to 3 of these four parametes.)
        ubin_size (float):  Analogous to bin_size for the u values. (default: bin_size)
        min_u (float):      Analogous to min_sep for the u values. (default: 0)
        max_u (float):      Analogous to max_sep for the u values. (default: 1)

        nvbins (int):       Analogous to nbins for the positive v values when bin__type=LogRUV.
                            (The default is to calculate from vbin_size = bin_size, min_v = 0,
                            max_v = 1, but this can be overridden by specifying up to 3 of these
                            four parametes.)
        vbin_size (float):  Analogous to bin_size for the v values. (default: bin_size)
        min_v (float):      Analogous to min_sep for the positive v values. (default: 0)
        max_v (float):      Analogous to max_sep for the positive v values. (default: 1)

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
                            (default: :math:`\max(3, \log_2(N_{\rm cpu}))`)
        max_top (int):      The maximum number of top layers to use when setting up the field.
                            The top-level cells are where each calculation job starts. There will
                            typically be of order :math:`2^{\rm max\_top}` top-level cells.
                            (default: 10)
        precision (int):    The precision to use for the output values. This specifies how many
                            digits to write. (default: 4)

        metric (str):       Which metric to use for distance measurements.  Options are listed
                            above.  (default: 'Euclidean')
        bin_type (str):     What type of binning should be used.  Options are listed above.
                            (default: 'LogSAS')
        min_rpar (float):   The minimum difference in Rparallel to allow for pairs being included
                            in the correlation function.  (default: None)
        max_rpar (float):   The maximum difference in Rparallel to allow for pairs being included
                            in the correlation function.  (default: None)
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
        cross_patch_weight (str): How to weight pairs that cross between two patches when one patch
                            is deselected (e.g. in a jackknife sense) and the other is selected.
                            (default None)
        rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                            random number generation. (default: None)

        num_threads (int):  How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.)

                            .. note::

                                This won't work if the system's C compiler cannot use OpenMP
                                (e.g. clang prior to version 3.7.)
    """
    _default_angle_slop = 0.1

    # A dict pointing from _letters to cls.  E.g. _lookup_dict['GGG'] = GGGCorrelation
    _lookup_dict = {}

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
        'angle_slop' : (float, False, None, None,
                'The maximum difference in the projection angle for any pair of objects relative '
                'to that of the pair of cells being used for an accumulated triangle'),
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
        'nphi_bins' : (int, False, None, None,
                'The number of output bins to use for phi dimension.'),
        'phi_bin_size' : (float, False, None, None,
                'The size of the output bins in phi.'),
        'min_phi' : (float, False, None, None,
                'The minimum phi to include in the output.'),
        'max_phi' : (float, False, None, None,
                'The maximum phi to include in the output.'),
        'phi_units' : (str, False, None, coord.AngleUnit.valid_names,
                'The units to use for min_phi and max_phi.'),
        'max_n' : (int, False, None, None,
                'The maximum n to store for multipole binning.'),
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
        'metric': (str, False, 'Euclidean', ['Euclidean', 'Rperp', 'FisherRperp', 'OldRperp',
                                             'Rlens', 'Arc', 'Periodic'],
                'Which metric to use for the distance measurements'),
        'bin_type': (str, False, 'LogSAS', ['LogRUV', 'LogSAS', 'LogMultipole'],
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

        'var_method': (str, False, 'shot',
                ['shot', 'jackknife', 'sample', 'bootstrap', 'marked_bootstrap'],
                'The method to use for estimating the variance'),
        'num_bootstrap': (int, False, 500, None,
                'How many bootstrap samples to use for the var_method=bootstrap and '
                'marked_bootstrap'),
        'cross_patch_weight': (str, False, None, ['simple', 'mean', 'match', 'geom'],
                'How to weight pairs that cross between a selected and unselected patch'),
        'num_threads' : (int, False, None, None,
                'How many threads should be used. num_threads <= 0 means auto based on num cores.'),
    }

    def __init__(self, config=None, *, logger=None, rng=None, **kwargs):
        self._corr = None  # Do this first to make sure we always have it for __del__
        self.config = merge_config(config,kwargs,Corr3._valid_params)
        if logger is None:
            self._logger_name = 'treecorr.Corr3'
            self.logger = setup_logger(get(self.config,'verbose',int,1),
                                       self.config.get('log_file',None), self._logger_name)
        else:
            self.logger = logger
            self._logger_name = logger.name

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

        self._ro.bin_type = self.config.get('bin_type', 'LogSAS')

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
        self._ro._bin_size = self.bin_size  # There is no Linear, but if I add it, need to apply
                                            # units to _bin_size in that case as well.

        if self.bin_type == 'LogRUV':
            for key in ['min_phi', 'max_phi', 'nphi_bins', 'phi_bin_size',
                        'max_n']:
                if key in self.config:
                    raise TypeError("%s is invalid for bin_type=LogRUV"%key)
            self._ro._bintype = _treecorr.LogRUV
            self._ro.min_u = float(self.config.get('min_u', 0.))
            self._ro.max_u = float(self.config.get('max_u', 1.))
            if self.min_u < 0. or self.max_u > 1.:
                raise ValueError("Invalid range for u: %f - %f"%(self.min_u, self.max_u))
            if self.min_u >= self.max_u:
                raise ValueError("max_u must be larger than min_u")
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
            if self.min_v < 0 or self.max_v > 1.:
                raise ValueError("Invalid range for |v|: %f - %f"%(self.min_v, self.max_v))
            if self.min_v >= self.max_v:
                raise ValueError("max_v must be larger than min_v")
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
        elif self.bin_type == 'LogSAS':
            for key in ['min_u', 'max_u', 'nubins', 'ubin_size',
                        'min_v', 'max_v', 'nvbins', 'vbin_size']:
                if key in self.config:
                    raise TypeError("%s is invalid for bin_type=LogSAS"%key)
            self._ro._bintype = _treecorr.LogSAS
            # Note: we refer to phi as u in the _ro namespace to make function calls easier.
            self._ro.phi_units = self.config.get('phi_units','')
            self._ro._phi_units = get(self.config,'phi_units',str,'radians')
            self._ro.min_u = float(self.config.get('min_phi', 0.))
            self._ro.max_u = float(self.config.get('max_phi', np.pi / self._phi_units))
            self._ro.min_u = self.min_phi * self._phi_units
            self._ro.max_u = self.max_phi * self._phi_units
            if self.min_phi < 0 or self.max_phi > np.pi:
                raise ValueError("Invalid range for phi: %f - %f"%(
                                 self.min_phi/self._phi_units, self.max_phi/self._phi_units))
            if self.min_phi >= self.max_phi:
                raise ValueError("max_phi must be larger than min_phi")
            self._ro.ubin_size = float(self.config.get('phi_bin_size', bin_size))
            if 'nphi_bins' not in self.config:
                self._ro.nubins = (self.max_phi-self.min_phi-1.e-10)/self.phi_bin_size
                self._ro.nubins = int(math.ceil(self._ro.nubins))
            elif ('max_phi' in self.config and 'min_phi' in self.config
                    and 'phi_bin_size' in self.config):
                raise TypeError("Only 3 of min_phi, max_phi, phi_bin_size, and nphi_bins are "
                                "allowed.")
            else:
                self._ro.nubins = self.config['nphi_bins']
                # Allow min or max phi to be implicit from nphi_bins and phi_bin_size
                if 'phi_bin_size' in self.config:
                    if 'max_phi' not in self.config:
                        self._ro.max_u = self.min_phi + self.nphi_bins * self.phi_bin_size
                        self._ro.max_u = min(self._ro.max_u, np.pi)
                    else:  # min_phi not in config
                        self._ro.min_u = self.max_phi - self.nphi_bins * self.phi_bin_size
                        self._ro.min_u = max(self._ro.min_u, 0.)
            # Adjust phi_bin_size given the other values
            self._ro.ubin_size = (self.max_phi-self.min_phi)/self.nphi_bins
            self._ro.min_v = self._ro.max_v = self._ro.nvbins = self._ro.vbin_size = 0
        elif self.bin_type == 'LogMultipole':
            for key in ['min_u', 'max_u', 'nubins', 'ubin_size',
                        'min_v', 'max_v', 'nvbins', 'vbin_size',
                        'min_phi', 'max_phi', 'nphi_bins', 'phi_bin_size']:
                if key in self.config:
                    raise TypeError("%s is invalid for bin_type=LogMultipole"%key)
            self._ro._bintype = _treecorr.LogMultipole
            if self.config.get('max_n', None) is None:
                raise TypeError("Missing required parameter max_n")
            self._ro.nubins = int(self.config['max_n'])
            if self.max_n < 0:
                raise ValueError("max_n must be non-negative")
            self._ro.min_u = self._ro.max_u = self._ro.ubin_size = 0
            self._ro.min_v = self._ro.max_v = self._ro.nvbins = self._ro.vbin_size = 0
        else:  # pragma: no cover  (Already checked by config layer)
            raise ValueError("Invalid bin_type %s"%self.bin_type)

        self._ro.split_method = self.config.get('split_method','mean')
        self.logger.debug("Using split_method = %s",self.split_method)

        self._ro.min_top = get(self.config,'min_top',int,None)
        self._ro.max_top = get(self.config,'max_top',int,10)

        self._ro.bin_slop = get(self.config,'bin_slop',float,-1.0)
        if self.bin_type == 'LogMultipole':
            self._ro.angle_slop = get(self.config,'angle_slop',float,np.pi/(2*self.max_n+1))
        else:
            self._ro.angle_slop = get(self.config,'angle_slop',float,self._default_angle_slop)
        if self.bin_slop < 0.0:
            if self.bin_size <= 0.1:
                self._ro.bin_slop = 1.0
                self._ro.b = self.bin_size
            else:
                self._ro.bin_slop = 0.1/self.bin_size  # The stored bin_slop corresponds to lnr bins.
                self._ro.b = 0.1
            if self.bin_type == 'LogRUV':
                if self.ubin_size <= 0.1:
                    self._ro.bu = self.ubin_size
                else:
                    self._ro.bu = 0.1
                if self.vbin_size <= 0.1:
                    self._ro.bv = self.vbin_size
                else:
                    self._ro.bv = 0.1
            elif self.bin_type == 'LogSAS':
                if self._ro.ubin_size <= 0.1:
                    self._ro.bu = self._ro.ubin_size
                else:
                    self._ro.bu = 0.1
                self._ro.bv = 0
            else:
                # LogMultipole
                self._ro.bu = self._ro.bv = 0
        else:
            self._ro.b = self._bin_size * self.bin_slop
            if self.bin_type == 'LogRUV':
                self._ro.bu = self.ubin_size * self.bin_slop
                self._ro.bv = self.vbin_size * self.bin_slop
            elif self.bin_type == 'LogSAS':
                self._ro.bu = self._ro.ubin_size * self.bin_slop
                self._ro.bv = 0
            else:
                # LogMultipole
                self._ro.bu = self._ro.bv = 0

        if self.b > 0.100001 and self.angle_slop > 0.1:
            self.logger.warning(
                "Using bin_slop = %g, angle_slop = %g, bin_size = %g (b = %g)\n"%(
                    self.bin_slop, self.angle_slop, self.bin_size, self.b)+
                "It is recommended to use either bin_slop <= %g or angle_slop <= 0.1.\n"%(
                    0.1/self.bin_size)+
                "Larger values of bin_slop/angle_slop may result in significant inaccuracies.")
        else:
            self.logger.debug("Using bin_slop = %g, angle_slop = %g (b = %g, bu = %g, bv = %g)",
                              self.bin_slop, self.angle_slop, self.b, self.bu, self.bv)

        # This makes nbins evenly spaced entries in log(r) starting with 0 with step bin_size
        self._ro.logr1d = np.linspace(start=0, stop=self.nbins*self.bin_size,
                                      num=self.nbins, endpoint=False)
        # Offset by the position of the center of the first bin.
        self._ro.logr1d += math.log(self.min_sep) + 0.5*self.bin_size
        self._ro.rnom1d = np.exp(self.logr1d)

        if self.bin_type == 'LogRUV':
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
            self._ro._nbins = len(self._ro.logr.ravel())

            self.data_shape = self._ro.logr.shape
            self.meanu = np.zeros(self.data_shape, dtype=float)
            self.meanv = np.zeros(self.data_shape, dtype=float)

        elif self.bin_type == 'LogSAS':
            # LogSAS
            self._ro.phi1d = np.linspace(start=0, stop=self.nphi_bins*self.phi_bin_size,
                                         num=self.nphi_bins, endpoint=False)
            self._ro.phi1d += self.min_phi + 0.5*self.phi_bin_size
            self._ro.logd2 = np.tile(self.logr1d[:, np.newaxis, np.newaxis],
                                     (1, self.nbins, self.nphi_bins))
            self._ro.logd3 = np.tile(self.logr1d[np.newaxis, :, np.newaxis],
                                     (self.nbins, 1, self.nphi_bins))
            self._ro.phi = np.tile(self.phi1d[np.newaxis, np.newaxis, :],
                                   (self.nbins, self.nbins, 1))
            self._ro.d2nom = np.exp(self.logd2)
            self._ro.d3nom = np.exp(self.logd3)
            self._ro._nbins = len(self._ro.logd2.ravel())

            self.data_shape = self._ro.logd2.shape
            # Also name these with the same names as above to make them easier to use.
            # We have properties to alias meanu as meanphi. meanv will remain all 0.
            self.meanu = np.zeros(self.data_shape, dtype=float)
            self.meanv = np.zeros(self.data_shape, dtype=float)

        else:
            # LogMultipole
            self._ro.logd2 = np.tile(self.logr1d[:, np.newaxis, np.newaxis],
                                     (1, self.nbins, 2*self.max_n+1))
            self._ro.logd3 = np.tile(self.logr1d[np.newaxis, :, np.newaxis],
                                     (self.nbins, 1, 2*self.max_n+1))
            self._ro.d2nom = np.exp(self.logd2)
            self._ro.d3nom = np.exp(self.logd3)
            self._ro._nbins = len(self._ro.logd2.ravel())
            self._ro.n1d = np.arange(-self.max_n, self.max_n+1, dtype=int)
            self._ro.n = np.tile(self.n1d[np.newaxis, np.newaxis, :],
                                 (self.nbins, self.nbins, 1))

            self.data_shape = self._ro.logd2.shape
            # Also name these with the same names as above to make them easier to use.
            # We won't use them for this bin type though.
            self.meanu = np.zeros(self.data_shape, dtype=float)
            self.meanv = np.zeros(self.data_shape, dtype=float)

        # We always keep track of these.
        self.meand1 = np.zeros(self.data_shape, dtype=float)
        self.meanlogd1 = np.zeros(self.data_shape, dtype=float)
        self.meand2 = np.zeros(self.data_shape, dtype=float)
        self.meanlogd2 = np.zeros(self.data_shape, dtype=float)
        self.meand3 = np.zeros(self.data_shape, dtype=float)
        self.meanlogd3 = np.zeros(self.data_shape, dtype=float)

        self.weightr = np.zeros(self.data_shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self.weighti = np.zeros(self.data_shape, dtype=float)
        else:
            self.weighti = np.array([])
        self.ntri = np.zeros(self.data_shape, dtype=float)
        self._cov = None
        self._varzeta = None
        self._var_num = 0
        self._corr_only = False

        # Sub-classes will allocate space for the ones we actually need.
        self._z = [np.array([]) for _ in range(8)]

        self._ro.brute = get(self.config,'brute',bool,False)
        if self.brute:
            self.logger.info("Doing brute force calculation.",)
        self.coords = None
        self.metric = None
        self._ro.min_rpar = get(self.config,'min_rpar',float,-sys.float_info.max)
        self._ro.max_rpar = get(self.config,'max_rpar',float,sys.float_info.max)
        if self.min_rpar > self.max_rpar:
            raise ValueError("min_rpar must be <= max_rpar")
        period = get(self.config,'period',float,0)
        self._ro.xperiod = get(self.config,'xperiod',float,period)
        self._ro.yperiod = get(self.config,'yperiod',float,period)
        self._ro.zperiod = get(self.config,'zperiod',float,period)

        self._ro.var_method = get(self.config,'var_method',str,'shot')
        self._ro.num_bootstrap = get(self.config,'num_bootstrap',int,500)
        self._ro.cross_patch_weight = get(self.config,'cross_patch_weight',str,None)
        self.results = {}  # for jackknife, etc. store the results of each pair of patches.
        self.npatch1 = self.npatch2 = self.npatch3 = 1
        self._rng = rng

    def __init_subclass__(cls):
        super().__init_subclass__()
        Corr3._lookup_dict[cls._letters] = cls

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng()
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
    def bin_size(self): return self._ro.bin_size
    @property
    def nbins(self): return self._ro.nbins
    @property
    def logr1d(self): return self._ro.logr1d
    @property
    def rnom1d(self): return self._ro.rnom1d
    @property
    def bu(self): return self._ro.bu
    @property
    def bv(self): return self._ro.bv

    # LogRUV:
    @property
    def rnom(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.rnom
    @property
    def logr(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.logr
    @property
    def min_u(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.min_u
    @property
    def max_u(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.max_u
    @property
    def min_v(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.min_v
    @property
    def max_v(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.max_v
    @property
    def ubin_size(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.ubin_size
    @property
    def vbin_size(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.vbin_size
    @property
    def nubins(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.nubins
    @property
    def nvbins(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.nvbins
    @property
    def u1d(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.u1d
    @property
    def v1d(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.v1d
    @property
    def u(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.u
    @property
    def v(self):
        assert self.bin_type == 'LogRUV'
        return self._ro.v

    # LogSAS
    @property
    def d2nom(self):
        assert self.bin_type in ['LogSAS', 'LogMultipole']
        return self._ro.d2nom
    @property
    def d3nom(self):
        assert self.bin_type in ['LogSAS', 'LogMultipole']
        return self._ro.d3nom
    @property
    def logd2(self):
        assert self.bin_type in ['LogSAS', 'LogMultipole']
        return self._ro.logd2
    @property
    def logd3(self):
        assert self.bin_type in ['LogSAS', 'LogMultipole']
        return self._ro.logd3
    @property
    def min_phi(self):
        assert self.bin_type == 'LogSAS'
        return self._ro.min_u
    @property
    def max_phi(self):
        assert self.bin_type == 'LogSAS'
        return self._ro.max_u
    @property
    def phi_bin_size(self):
        assert self.bin_type == 'LogSAS'
        return self._ro.ubin_size
    @property
    def nphi_bins(self):
        assert self.bin_type == 'LogSAS'
        return self._ro.nubins
    @property
    def phi1d(self):
        assert self.bin_type == 'LogSAS'
        return self._ro.phi1d
    @property
    def phi(self):
        assert self.bin_type == 'LogSAS'
        return self._ro.phi
    @property
    def phi_units(self): return self._ro.phi_units
    @property
    def _phi_units(self): return self._ro._phi_units
    @property
    def meanphi(self):
        assert self.bin_type == 'LogSAS'
        return self.meanu

    # LogMultipole:
    @property
    def max_n(self):
        assert self.bin_type == 'LogMultipole'
        return self._ro.nubins
    @property
    def n1d(self):
        assert self.bin_type == 'LogMultipole'
        return self._ro.n1d
    @property
    def n(self):
        assert self.bin_type == 'LogMultipole'
        return self._ro.n

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
    def angle_slop(self): return self._ro.angle_slop
    @property
    def b(self): return self._ro.b
    @property
    def brute(self): return self._ro.brute
    @property
    def min_rpar(self): return self._ro.min_rpar
    @property
    def max_rpar(self): return self._ro.max_rpar
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
    def cross_patch_weight(self): return self._ro.cross_patch_weight

    @property
    def weight(self):
        if self.weighti.size:
            return self.weightr + 1j * self.weighti
        else:
            return self.weightr

    @property
    def cov(self):
        """The estimated covariance matrix
        """
        if self._cov is None:
            self._cov = self.estimate_cov(self.var_method)
        if self._cov.ndim == 1:
            return np.diag(self._cov)
        else:
            return self._cov

    @property
    def cov_diag(self):
        """A possibly more efficient way to access just the diagonal of the covariance matrix.

        If var_method == 'shot', then this won't make the full covariance matrix, just to
        then pull out the diagonal.
        """
        if self._cov is None:
            self._cov = self.estimate_cov(self.var_method)
        if self._cov.ndim == 1:
            return self._cov
        else:
            return self._cov.diagonal()

    @property
    def corr(self):
        if self._corr is None:
            self._corr = self._builder(
                    self._bintype,
                    self._min_sep, self._max_sep, self.nbins, self._bin_size, self.b,
                    self.angle_slop,
                    self._ro.min_u,self._ro.max_u,self._ro.nubins,self._ro.ubin_size,self.bu,
                    self._ro.min_v,self._ro.max_v,self._ro.nvbins,self._ro.vbin_size,self.bv,
                    self.min_rpar, self.max_rpar,
                    self.xperiod, self.yperiod, self.zperiod,
                    self._z[0], self._z[1], self._z[2], self._z[3],
                    self._z[4], self._z[5], self._z[6], self._z[7],
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv,
                    self.weightr, self.weighti, self.ntri)
        return self._corr

    @property
    def _zetas(self):
        zetas = []
        if self._z[0].size:
            if self._z[1].size:
                zetas += [self._z[0] + 1j*self._z[1]]
            else:
                zetas += [self._z[0]]
        if self._z[2].size:
            zetas += [self._z[2] + 1j*self._z[3]]
        if self._z[4].size:
            zetas += [self._z[4] + 1j*self._z[5]]
            zetas += [self._z[6] + 1j*self._z[7]]
        return zetas

    def _equal_binning(self, other, brief=False):
        # A helper function to test if two Corr3 objects have the same binning parameters
        if self.bin_type == 'LogRUV':
            eq = (other.bin_type == 'LogRUV' and
                  self.min_sep == other.min_sep and
                  self.max_sep == other.max_sep and
                  self.nbins == other.nbins and
                  self.min_u == other.min_u and
                  self.max_u == other.max_u and
                  self.nubins == other.nubins and
                  self.min_v == other.min_v and
                  self.max_v == other.max_v and
                  self.nvbins == other.nvbins)
        elif self.bin_type == 'LogSAS':
            # LogSAS
            eq = (other.bin_type == 'LogSAS' and
                  self.min_sep == other.min_sep and
                  self.max_sep == other.max_sep and
                  self.nbins == other.nbins and
                  self.min_phi == other.min_phi and
                  self.max_phi == other.max_phi and
                  self.nphi_bins == other.nphi_bins)
        else:
            # LogMultipole
            eq = (other.bin_type == 'LogMultipole' and
                  self.min_sep == other.min_sep and
                  self.max_sep == other.max_sep and
                  self.nbins == other.nbins and
                  self.max_n == other.max_n)
        if brief or not eq:
            return eq
        else:
            return (self.sep_units == other.sep_units and
                    (self.bin_type != 'LogSAS' or self.phi_units == other.phi_units) and
                    self.coords == other.coords and
                    self.bin_slop == other.bin_slop and
                    self.angle_slop == other.angle_slop and
                    self.min_rpar == other.min_rpar and
                    self.max_rpar == other.max_rpar and
                    self.xperiod == other.xperiod and
                    self.yperiod == other.yperiod and
                    self.zperiod == other.zperiod)

    def _equal_bin_data(self, other):
        # A helper function to test if two Corr3 objects have the same measured bin values
        equal_d = (np.array_equal(self.meand1, other.meand1) and
                   np.array_equal(self.meanlogd1, other.meanlogd1) and
                   np.array_equal(self.meand2, other.meand2) and
                   np.array_equal(self.meanlogd2, other.meanlogd2) and
                   np.array_equal(self.meand3, other.meand3) and
                   np.array_equal(self.meanlogd3, other.meanlogd3))
        if self.bin_type == 'LogRUV':
            return (other.bin_type == 'LogRUV' and equal_d and
                    np.array_equal(self.meanu, other.meanu) and
                    np.array_equal(self.meanv, other.meanv))
        elif self.bin_type == 'LogSAS':
            return (other.bin_type == 'LogSAS' and equal_d and
                    np.array_equal(self.meanphi, other.meanphi))
        else:
            # LogMultipole
            return (other.bin_type == 'LogMultipole' and equal_d)

    def __eq__(self, other):
        """Return whether two Correlation instances are equal"""
        return (isinstance(other, self.__class__) and
                self._equal_binning(other) and
                self._equal_bin_data(other) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.ntri, other.ntri) and
                all(np.array_equal(self._z[i], other._z[i]) for i in range(8)))

    def copy(self):
        """Make a copy"""
        ret = self.__class__.__new__(self.__class__)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to by deep copied.
                ret.__dict__[key] = item.copy()
            else:
                # For everything else, shallow copy is fine.
                # In particular don't deep copy config or logger
                # Most of the rest are scalars, which copy fine this way.
                ret.__dict__[key] = item
        ret._z = [zi.copy() for zi in self._z]
        ret._corr = None # We'll want to make a new one of these if we need it.
        return ret

    def __repr__(self):
        kwargs = make_minimal_config(self.config, Corr3._valid_params)
        kwargs_str = ', '.join(f'{k}={v!r}' for k,v in kwargs.items())
        return f'{self._cls}({kwargs_str})'

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_corr',None)
        d.pop('_ok',None)     # Remake this as needed.
        d.pop('logger',None)  # Oh well.  This is just lost in the copy.  Can't be pickled.
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._corr = None
        if self._logger_name is not None:
            self.logger = setup_logger(get(self.config,'verbose',int,1),
                                       self.config.get('log_file',None), self._logger_name)

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
        return np.any(self.weight)

    @lazy_property
    def _nonzero(self):
        # The lazy version when we can be sure the object isn't going to accumulate any more.
        return self.nonzero

    def _add_tot(self, ijk, c1, c2, c3):
        # No op for all but NNCorrelation, which needs to add the tot value
        pass

    def _trivially_zero(self, c1, c2, c3, ordered):
        x1,y1,z1,s1 = c1._get_center_size()
        x2,y2,z2,s2 = c2._get_center_size()
        x3,y3,z3,s3 = c3._get_center_size()
        return self.corr.triviallyZero(self._metric, self._coords,
                                       x1, y1, z1, s1, x2, y2, z2, s2, x3, y3, z3, s3,
                                       ordered, (c2 is c3 or c1 is c2))

    _make_expanded_patch = Corr2._make_expanded_patch

    def _single_process12(self, c1, c2, ijj, metric, ordered, num_threads, corr_only,
                          temp, force_write):
        # Helper function for _process_all_auto, etc. for doing 12 cross pairs
        temp._clear()

        if c2 is not None and not self._trivially_zero(c1,c2,c2,ordered):
            self.logger.info('Process patches %s cross12',ijj)
            temp.process_cross12(c1, c2, metric=metric, ordered=ordered, num_threads=num_threads,
                                 corr_only=corr_only)
        else:
            self.logger.info('Skipping %s pair, which are too far apart ' +
                             'for this set of separations',ijj)
        if temp.nonzero or force_write:
            if ijj in self.results and self.results[ijj].nonzero:
                self.results[ijj] += temp
            else:
                self.results[ijj] = temp.copy()
            self += temp
        else:
            # NNNCorrelation needs to add the tot value
            self._add_tot(ijj, c1, c2, c2)

    def _single_process21(self, c1, c2, iij, metric, ordered, num_threads, corr_only,
                          temp, force_write):
        # Helper function for _process_all_cross21, etc. for doing 21 cross pairs
        temp._clear()

        if c1 is not None and not self._trivially_zero(c1,c1,c2,ordered):
            self.logger.info('Process patches %s cross21',iij)
            temp.process_cross21(c1, c2, metric=metric, ordered=ordered, num_threads=num_threads,
                                 corr_only=corr_only)
        else:
            self.logger.info('Skipping %s pair, which are too far apart ' +
                             'for this set of separations',iij)
        if temp.nonzero or force_write:
            if iij in self.results and self.results[iij].nonzero:
                self.results[iij] += temp
            else:
                self.results[iij] = temp.copy()
            self += temp
        else:
            # NNNCorrelation needs to add the tot value
            self._add_tot(iij, c1, c1, c2)

    def _single_process123(self, c1, c2, c3, ijk, metric, ordered, num_threads, corr_only,
                           temp, force_write):
        # Helper function for _process_all_auto, etc. for doing 123 cross triples
        temp._clear()

        if c2 is not None and c3 is not None and not self._trivially_zero(c1,c2,c3,ordered):
            self.logger.info('Process patches %s cross',ijk)
            temp.process_cross(c1, c2, c3, metric=metric, ordered=ordered, num_threads=num_threads,
                               corr_only=corr_only)
        else:
            self.logger.info('Skipping %s, which are too far apart ' +
                             'for this set of separations',ijk)
        if temp.nonzero or force_write:
            if ijk in self.results and self.results[ijk].nonzero:
                self.results[ijk] += temp
            else:
                self.results[ijk] = temp.copy()
            self += temp
        else:
            # NNNCorrelation needs to add the tot value
            self._add_tot(ijk, c1, c2, c3)

    def _process_all_auto(self, cat1, metric, num_threads, corr_only, comm, low_mem, local):

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
            self.process_auto(cat1[0], metric=metric, num_threads=num_threads, corr_only=corr_only)

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

            self._set_metric(metric, cat1[0].coords)
            temp = self.copy()
            temp.results = {}

            if local:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    if is_my_job(my_indices, i, i, i, n):
                        c1e = self._make_expanded_patch(c1, cat1, metric, low_mem)
                        self.logger.info('Process patch %d with surrounding local patches',i)
                        self._single_process12(c1, c1e, (i,i,i), metric, 1,
                                               num_threads, corr_only, temp, True)
                        if low_mem:
                            c1.unload()
            else:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    if is_my_job(my_indices, i, i, i, n):
                        temp._clear()
                        self.logger.info('Process patch %d auto',i)
                        temp.process_auto(c1, metric=metric, num_threads=num_threads,
                                          corr_only=corr_only)
                        if (i,i,i) in self.results and self.results[(i,i,i)].nonzero:
                            self.results[(i,i,i)] += temp
                        else:
                            self.results[(i,i,i)] = temp.copy()
                        self += temp

                    for jj,c2 in list(enumerate(cat1))[::-1]:
                        j = c2._single_patch if c2._single_patch is not None else jj
                        if i < j:
                            if is_my_job(my_indices, i, j, j, n):
                                # One point in c1, 2 in c2.
                                self._single_process12(c1, c2, (i,j,j), metric, 0,
                                                    num_threads, corr_only, temp, False)
                                # One point in c2, 2 in c1.
                                self._single_process12(c2, c1, (i,i,j), metric, 0,
                                                    num_threads, corr_only, temp, False)

                            # One point in each of c1, c2, c3
                            for kk,c3 in enumerate(cat1):
                                k = c3._single_patch if c3._single_patch is not None else kk
                                if j < k and is_my_job(my_indices, i, j, k, n):
                                    self._single_process123(c1, c2, c3, (i,j,k), metric, 0,
                                                            num_threads, corr_only, temp, False)
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

    def _process_all_cross12(self, cat1, cat2, metric, ordered, num_threads, corr_only,
                             comm, low_mem, local):

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
            self.process_cross12(cat1[0], cat2[0], metric=metric, ordered=ordered,
                                 num_threads=num_threads, corr_only=corr_only)
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

            self._set_metric(metric, cat1[0].coords)
            temp = self.copy()
            temp.results = {}
            ordered1 = 1 if ordered else 0

            if local:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    if is_my_job(my_indices, i, i, i, n1, n2):
                        c2e = self._make_expanded_patch(c1, cat2, metric, low_mem)
                        self.logger.info('Process patch %d with surrounding local patches',i)
                        self._single_process12(c1, c2e, (i,i,i), metric, 1,
                                               num_threads, corr_only, temp, True)
                        if low_mem:
                            c1.unload()
                if not ordered:
                    # local method doesn't do unordered properly as is.
                    # It can only handle ordered=1 (or 3).
                    # So in this case, we need to repeat with c2 in the first spot.
                    for ii,c2 in enumerate(cat2):
                        i = c2._single_patch if c2._single_patch is not None else ii
                        if (is_my_job(my_indices, i, i, i, n1, n2)
                                and self._letter1 == self._letter2):
                            c1e = self._make_expanded_patch(c2, cat1, metric, low_mem)
                            c2e = self._make_expanded_patch(c2, cat2, metric, low_mem)
                            self.logger.info('Process patch %d from cat2 with surrounding local patches',i)
                            self._single_process123(c2, c1e, c2e, (i,i,i), metric, 1,
                                                    num_threads, corr_only, temp, True)
                            if low_mem:
                                c2.unload()
            else:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    for jj,c2 in enumerate(cat2):
                        j = c2._single_patch if c2._single_patch is not None else jj
                        if is_my_job(my_indices, i, j, j, n1, n2):
                            self._single_process12(c1, c2, (i,j,j), metric, ordered,
                                                   num_threads, corr_only, temp,
                                                   (i==j or n1==1 or n2==1))
                        # One point in each of c1, c2, c3
                        for kk,c3 in list(enumerate(cat2))[::-1]:
                            k = c3._single_patch if c3._single_patch is not None else kk
                            if j < k and is_my_job(my_indices, i, j, k, n1, n2):
                                self._single_process123(c1, c2, c3, (i,j,k), metric, ordered1,
                                                        num_threads, corr_only, temp, False)
                                if low_mem and kk != jj+1:
                                    # Don't unload j+1, since that's the next one we'll need.
                                    c3.unload()
                        if low_mem:
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

    def _process_all_cross21(self, cat1, cat2, metric, ordered, num_threads, corr_only,
                             comm, low_mem, local):

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
            self.process_cross21(cat1[0], cat2[0], metric=metric, ordered=ordered,
                                 num_threads=num_threads, corr_only=corr_only)
        else:
            # When patch processing, keep track of the pair-wise results.
            if self.npatch1 == 1:
                self.npatch1 = cat1[0].npatch if cat1[0].npatch != 1 else len(cat1)
                self.npatch2 = self.npatch1
            if self.npatch3 == 1:
                self.npatch3 = cat2[0].npatch if cat2[0].npatch != 1 else len(cat2)
            if self.npatch1 != self.npatch3 and self.npatch1 != 1 and self.npatch3 != 1:
                raise RuntimeError("Cross correlation requires both catalogs use the same patches.")

            # Setup for deciding when this is my job.
            n1 = self.npatch1
            n2 = self.npatch3
            if comm:
                size = comm.Get_size()
                rank = comm.Get_rank()
                n = max(n1,n2)
                my_indices = np.arange(n * rank // size, n * (rank+1) // size)
                self.logger.info("Rank %d: My indices are %s",rank,my_indices)
            else:
                my_indices = None

            self._set_metric(metric, cat1[0].coords)
            temp = self.copy()
            temp.results = {}
            ordered3 = 3 if (ordered or self._letter1 != self._letter3) else 0

            if local:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    if is_my_job(my_indices, i, i, i, n1, n2):
                        c2e = self._make_expanded_patch(c1, cat1, metric, low_mem)
                        c3e = self._make_expanded_patch(c1, cat2, metric, low_mem)
                        self.logger.info('Process patch %d with surrounding local patches',i)
                        self._single_process123(c1, c2e, c3e, (i,i,i), metric,
                                                1 if not ordered else 4,
                                                num_threads, corr_only, temp, True)
                        if low_mem:
                            c1.unload()
            else:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    for kk,c3 in enumerate(cat2):
                        k = c3._single_patch if c3._single_patch is not None else kk
                        if is_my_job(my_indices, i, i, k, n1, n2):
                            self._single_process21(c1, c3, (i,i,k), metric, ordered,
                                                   num_threads, corr_only, temp,
                                                   (i==k or n1==1 or n2==1))
                        # One point in each of c1, c2, c3
                        for jj,c2 in list(enumerate(cat1))[::-1]:
                            j = c2._single_patch if c2._single_patch is not None else jj
                            if i < j and is_my_job(my_indices, i, j, k, n1, n2):
                                self._single_process123(c1, c2, c3, (i,j,k), metric, ordered3,
                                                        num_threads, corr_only, temp, False)
                                if low_mem and jj != ii+1:
                                    # Don't unload i+1, since that's the next one we'll need.
                                    c2.unload()
                        if low_mem:
                            c3.unload()
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

    def _process_all_cross(self, cat1, cat2, cat3, metric, ordered, num_threads, corr_only,
                           comm, low_mem, local):

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
            self.process_cross(cat1[0], cat2[0], cat3[0], metric=metric, ordered=ordered,
                               num_threads=num_threads, corr_only=corr_only)
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

            self._set_metric(metric, cat1[0].coords)
            temp = self.copy()
            temp.results = {}

            # Convert bool into corresponding int
            ordered = 0 if ordered is False else 4 if ordered is True else ordered

            if local:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    if is_my_job(my_indices, i, i, i, n1, n2, n3):
                        c2e = self._make_expanded_patch(c1, cat2, metric, low_mem)
                        c3e = self._make_expanded_patch(c1, cat3, metric, low_mem)
                        self.logger.info('Process patch %d with surrounding local patches',i)
                        self._single_process123(c1, c2e, c3e, (i,i,i), metric,
                                                1 if ordered in (0,1) else 4,
                                                num_threads, corr_only, temp, True)
                        if low_mem:
                            c1.unload()
                if ordered in (0,3):
                    # local method doesn't do unordered properly as is.
                    # It can only handle ordered=1 or 4.
                    # So in this case, we need to repeat with c2 and c3 in the first spot.
                    for ii,c2 in enumerate(cat2):
                        i = c2._single_patch if c2._single_patch is not None else ii
                        if (is_my_job(my_indices, i, i, i, n1, n2, n3)
                                and self._letter1 == self._letter2):
                            c1e = self._make_expanded_patch(c2, cat1, metric, low_mem)
                            c3e = self._make_expanded_patch(c2, cat3, metric, low_mem)
                            self.logger.info('Process patch %d from cat2 with surrounding local patches',i)
                            self._single_process123(c2, c1e, c3e, (i,i,i), metric,
                                                    1 if ordered == 0 else 4,
                                                    num_threads, corr_only, temp, True)
                            if low_mem:
                                c2.unload()
                if ordered in (0,2):
                    for ii,c3 in enumerate(cat3):
                        i = c3._single_patch if c3._single_patch is not None else ii
                        if (is_my_job(my_indices, i, i, i, n1, n2, n3)
                                and self._letter1 == self._letter3):
                            c1e = self._make_expanded_patch(c3, cat1, metric, low_mem)
                            c2e = self._make_expanded_patch(c3, cat2, metric, low_mem)
                            self.logger.info('Process patch %d from cat3 with surrounding local patches',i)
                            self._single_process123(c3, c2e, c1e, (i,i,i), metric,
                                                    1 if ordered == 0 else 4,
                                                    num_threads, corr_only, temp, True)
                            if low_mem:
                                c3.unload()
            else:
                for ii,c1 in enumerate(cat1):
                    i = c1._single_patch if c1._single_patch is not None else ii
                    for jj,c2 in enumerate(cat2):
                        j = c2._single_patch if c2._single_patch is not None else jj
                        for kk,c3 in enumerate(cat3):
                            k = c3._single_patch if c3._single_patch is not None else kk
                            if is_my_job(my_indices, i, j, k, n1, n2, n3):
                                self._single_process123(c1, c2, c3, (i,j,k), metric, ordered,
                                                        num_threads, corr_only, temp,
                                                        (i==j==k or n1==1 or n2==1 or n3==1))
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

        Usually, this is just self.zeta.  But in case we have a multi-dimensional array
        at some point (like TwoD for 2pt), use self.zeta.ravel().

        And for `GGGCorrelation`, it is the concatenation of the four different correlations
        [gam0.ravel(), gam1.ravel(), gam2.ravel(), gam3.ravel()].
        """
        return self.zeta.ravel()

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        This is the weight array corresponding to `getStat`. Usually just self.weight.ravel(),
        but duplicated for GGGCorrelation to match what `getStat` does in that case.
        """
        # For most bin types, this is just the normal weight.
        # but for LogMultipole, the absolute value is what we want.
        return np.abs(self.weight.ravel())

    def process_auto(self, cat, *, metric=None, num_threads=None, corr_only=False):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the ``finalize`` command will
        finish the calculation of meand1, meanlogd1, etc.

        This method is only valid for classes that have the same type of value in all
        three triangle vertices.  (E.g. NNN, GGG, KKK)

        Parameters:
            cat (Catalog):      The catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            corr_only (bool):   Whether to skip summing quantities that are not essential for
                                computing the correlation function. (default: False)
        """
        # The implementation is the same for all classes that can call this.
        assert self._letter1 == self._letter2 == self._letter3
        if cat.name == '':
            self.logger.info(f'Starting process {self._letters} auto-correlations')
        else:
            self.logger.info(f'Starting process {self._letters} auto-correlations for cat %s.',
                             cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        self._corr_only = corr_only
        min_size, max_size = self._get_minmax_size()

        getField = getattr(cat, f"get{self._letter1}Field")
        field = getField(min_size=min_size, max_size=max_size,
                         split_method=self.split_method, brute=bool(self.brute),
                         min_top=self.min_top, max_top=self.max_top,
                         coords=self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        self.corr.processAuto(field.data, self.output_dots, bool(corr_only), self._metric)

    def process_cross12(self, cat1, cat2, *, metric=None, ordered=True, num_threads=None,
                        corr_only=False):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto- or cross-correlation calculation.  E.g. when splitting up a large catalog into
        patches, this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        This method is only valid for classes that have the same type of value in vertices
        2 and 3.  (E.g. KKK, KGG, NKK)

        Parameters:
            cat1 (Catalog):     The first catalog to process. (1 point in each triangle will come
                                from this catalog.)
            cat2 (Catalog):     The second catalog to process. (2 points in each triangle will come
                                from this catalog.)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            corr_only (bool):   Whether to skip summing quantities that are not essential for
                                computing the correlation function. (default: False)
        """
        assert self._letter2 == self._letter3
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process %s (1-2) cross-correlations', self._letters)
        else:
            self.logger.info('Starting process %s (1-2) cross-correlations for cats %s, %s.',
                             self._letters, cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        self._corr_only = corr_only
        min_size, max_size = self._get_minmax_size()

        getField1 = getattr(cat1, f"get{self._letter1}Field")
        getField2 = getattr(cat2, f"get{self._letter2}Field")
        f1 = getField1(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 1,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)
        f2 = getField2(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 2,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        ocode = 1 if ordered or self._letter1 != self._letter2 else 0
        self.corr.processCross12(f1.data, f2.data, ocode, self.output_dots,
                                 bool(corr_only), self._metric)

    def process_cross21(self, cat1, cat2, *, metric=None, ordered=True, num_threads=None,
                        corr_only=False):
        """Process two catalogs, accumulating the 3pt cross-correlation, where two of the
        points in each triangle come from the first catalog, and one comes from the second.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto- or cross-correlation calculation.  E.g. when splitting up a large catalog into
        patches, this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        This method is only valid for classes that have the same type of value in vertices
        1 and 2.  (E.g. KKK, KKG, NNK)

        Parameters:
            cat1 (Catalog):     The first catalog to process. (2 points in each triangle will come
                                from this catalog.)
            cat2 (Catalog):     The second catalog to process. (1 point in each triangle will come
                                from this catalog.)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            corr_only (bool):   Whether to skip summing quantities that are not essential for
                                computing the correlation function. (default: False)
        """
        assert self._letter1 == self._letter2
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process %s (2-1) cross-correlations', self._letters)
        else:
            self.logger.info('Starting process %s (2-1) cross-correlations for cats %s, %s.',
                             self._letters, cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        self._corr_only = corr_only
        min_size, max_size = self._get_minmax_size()

        getField1 = getattr(cat1, f"get{self._letter1}Field")
        getField2 = getattr(cat2, f"get{self._letter3}Field")
        f1 = getField1(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 1,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)
        f2 = getField2(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 2,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        ocode = 3 if ordered or self._letter2 != self._letter3 else 0
        self.corr.processCross21(f1.data, f2.data, ocode, self.output_dots,
                                 bool(corr_only), self._metric)

    def process_cross(self, cat1, cat2, cat3, *, metric=None, ordered=True, num_threads=None,
                      corr_only=False):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto- or cross-correlation calculation.  E.g. when splitting up a large catalog into
        patches, this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            cat3 (Catalog):     The third catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            corr_only (bool):   Whether to skip summing quantities that are not essential for
                                computing the correlation function. (default: False)
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process %s cross-correlations', self._letters)
        else:
            self.logger.info('Starting process %s cross-correlations for cats %s, %s, %s.',
                             self._letters, cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self._set_num_threads(num_threads)
        self._corr_only = corr_only
        min_size, max_size = self._get_minmax_size()

        getField1 = getattr(cat1, f"get{self._letter1}Field")
        getField2 = getattr(cat2, f"get{self._letter2}Field")
        getField3 = getattr(cat3, f"get{self._letter3}Field")
        f1 = getField1(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 1,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)
        f2 = getField2(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 2,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)
        f3 = getField3(min_size=min_size, max_size=max_size,
                       split_method=self.split_method,
                       brute=self.brute is True or self.brute == 3,
                       min_top=self.min_top, max_top=self.max_top,
                       coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        ocode = 4 if ordered is True else 0 if ordered is False else ordered
        self.corr.processCross(f1.data, f2.data, f3.data, ocode, self.output_dots,
                               bool(corr_only), self._metric)

    def process(self, cat1, cat2=None, cat3=None, *, metric=None, ordered=True, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True,
                patch_method=None, algo=None, max_n=None, corr_only=False):
        """Compute the 3pt correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        .. note::

            For cross correlations where the third field type is different from the other two
            (e.g. KKG, NNG, etc.) then the 2 argument version will use the first catalog
            for first two vertices and the second for the third vertex, since that's the
            only valid combination for those correlation types.

            E.g. ``kkg.process(cat1, cat2)`` is equivalent to ``kkg.process(cat1, cat1, cat2)``,
            except it will be slightly more efficient, since it knows the first two vertices
            are from a single field.

        For cross correlations, the default behavior is to use cat1 for the first vertex (P1),
        cat2 for the second vertex (P2), and cat3 for the third vertex (P3).  If only two
        catalogs are given, vertices P2 and P3 both come from cat2.  The sides d1, d2, d3,
        used to define the binning, are taken to be opposte P1, P2, P3 respectively.

        However, if you want to accumulate triangles where objects from each catalog can take
        any position in the triangles, you can set ``ordered=False``.  In this case, triangles
        will be formed where P1, P2 and P3 can come from any input catalog, so long as there is
        one from cat1, one from cat2, and one from cat3 (or two from cat2 if cat3 is None).

        All catalog arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        .. note::

            In addition to ordered = True or False, you may also set ordered to 1, 2 or 3
            which means that the catalog in that position is fixed, but the other two
            vertices are unordered.  E.g. if ordered=3, then P3 will always come from cat3,
            but P1 and P2 will each come from one of cat1 or cat2 in either order.
            This option is only valid when all three catalogs (cat1, cat2, cat3) are given.

        In addition to computing the correlation function, this function also computes a
        number of ancillary quantities that are useful for interpreting the resulting correlation
        function, including the attributes ``meand1``, ``meanlogd1``, etc.  These almost never
        impart significant extra computation time for three-point correlations, but we provide
        the option ``corr_only=True`` in analogy to the `two-point version <Corr2.process>`,
        which skips these computations.  In this case the resulting ``meand?`` and ``meanlogd?``
        attributes are the values for the nominal centers of the bins, not the actual mean values.
        And ``ntri`` is estimated from the total computed ``weight`` and the mean weight in the
        catalogs.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first field.
            cat2 (Catalog):     A catalog or list of catalogs for the second field.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third field.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            ordered (bool):     Whether to fix the order of the triangle vertices to match the
                                catalogs. (see above; default: True)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            comm (mpi4py.Comm): If running MPI, an mpi4py Comm object to communicate between
                                processes.  If used, the rank=0 process will have the final
                                computation. This only works if using patches. (default: None)
            low_mem (bool):     Whether to sacrifice a little speed to try to reduce memory usage.
                                This only works if using patches. (default: False)
            initialize (bool):  Whether to begin the calculation with a call to
                                `Corr3.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to finalize.
                                (default: True)
            patch_method (str): Which patch method to use. (default is to use 'local' if
                                bin_type=LogMultipole, and 'global' otherwise)
            algo (str):         Which accumulation algorithm to use. (options are 'triangle' or
                                'multipole'; default is 'multipole' unless bin_type is 'LogRUV',
                                which can only use 'triangle')  cf. `Three-point Algorithm`.
            max_n (int):        If using the multpole algorithm, and this is not directly using
                                bin_type='LogMultipole', then this is the value of max_n to use
                                for the multipole part of the calculation. (default is to use
                                2pi/phi_bin_size; this value can also be given in the constructor
                                in the config dict.)
            corr_only (bool):   Whether to skip summing quantities that are not essential for
                                computing the correlation function. (default: False)
        """
        import math

        if algo is None:
            multipole = (self.bin_type != 'LogRUV')
        else:
            if algo not in ['triangle', 'multipole']:
                raise ValueError("Invalid algo %s"%algo)
            multipole = (algo == 'multipole')
            if multipole and self.bin_type == 'LogRUV':
                raise ValueError("LogRUV binning cannot use algo='multipole'")

        if multipole and self.bin_type != 'LogMultipole':
            config = self.config.copy()
            config['bin_type'] = 'LogMultipole'
            if max_n is None:
                max_n = 2.*np.pi / self.phi_bin_size
                # If max_n was given in constructor, use that instead.
                max_n = config.get('max_n', max_n)
            for key in ['min_phi', 'max_phi', 'nphi_bins', 'phi_bin_size']:
                config.pop(key, None)
            corr = self.__class__(config, max_n=max_n)
            corr.process(cat1, cat2, cat3,
                         metric=metric, ordered=ordered, num_threads=num_threads,
                         corr_only=corr_only, comm=comm, low_mem=low_mem,
                         initialize=initialize, finalize=finalize,
                         patch_method=patch_method, algo='multipole')
            corr.toSAS(target=self)
            if corr_only:
                self.ntri[:] = np.abs(self.weight) / self._meanwww
            return

        if patch_method is None:
            local = (self.bin_type == 'LogMultipole')
        else:
            if patch_method not in ['local', 'global']:
                raise ValueError("Invalid patch_method %s"%patch_method)
            local = (patch_method == 'local')
            if not local and self.bin_type == 'LogMultipole':
                raise ValueError("LogMultipole binning cannot use patch_method='global'")

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if cat2 is not None and not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)
        if cat3 is not None and not isinstance(cat3,list):
            cat3 = cat3.get_patches(low_mem=low_mem)

        if initialize:
            self.clear()
            self._corr_only = corr_only

        if cat2 is None:
            if not self._letter1 == self._letter2 == self._letter3:
                raise ValueError("{} cannot use one catalog version of process".format(self._letters))
            if cat3 is not None:
                raise ValueError("For two catalog case, use cat1,cat2, not cat1,cat3")
            if not (ordered is True or ordered is False):
                raise ValueError("The integer options for ordered are only valid with 3 catalogs")
            self._process_all_auto(cat1, metric, num_threads, corr_only, comm, low_mem, local)
        elif cat3 is None:
            if not (ordered is True or ordered is False):
                raise ValueError("The integer options for ordered are only valid with 3 catalogs")
            if self._letter2 == self._letter3:
                self._process_all_cross12(cat1, cat2, metric, ordered, num_threads, corr_only,
                                          comm, low_mem, local)
            elif self._letter1 == self._letter2:
                self._process_all_cross21(cat1, cat2, metric, ordered, num_threads, corr_only,
                                          comm, low_mem, local)
            else:
                raise ValueError("{} cannot use two catalog version of process".format(self._letters))
        else:
            self._process_all_cross(cat1, cat2, cat3, metric, ordered, num_threads, corr_only,
                                    comm, low_mem, local)

        if finalize:
            if cat2 is None:
                var1 = var2 = var3 = self._calculateVar1(cat1, low_mem=low_mem)
                if var1 is not None:
                    self.logger.info(f"var%s = %f: {self._sig1} = %f",
                                     self._letter1.lower(), var1, math.sqrt(var1))
            elif cat3 is None:
                if self._letter2 == self._letter3:
                    var1 = self._calculateVar1(cat1, low_mem=low_mem)
                    var2 = var3 = self._calculateVar2(cat2, low_mem=low_mem)
                    if var1 is not None:
                        self.logger.info(f"var%s1 = %f: {self._sig1} = %f",
                                         self._letter1, var1, math.sqrt(var1))
                    if var2 is not None:
                        self.logger.info(f"var%s2 = %f: {self._sig2} = %f",
                                         self._letter2, var2, math.sqrt(var2))
                else:
                    assert self._letter1 == self._letter2
                    var1 = var2 = self._calculateVar1(cat1, low_mem=low_mem)
                    var3 = self._calculateVar3(cat2, low_mem=low_mem)
                    if var1 is not None:
                        self.logger.info(f"var%s1 = %f: {self._sig1} = %f",
                                         self._letter1, var1, math.sqrt(var1))
                    if var3 is not None:
                        self.logger.info(f"var%s2 = %f: {self._sig2} = %f",
                                         self._letter2, var3, math.sqrt(var3))
            else:
                var1 = self._calculateVar1(cat1, low_mem=low_mem)
                var2 = self._calculateVar2(cat2, low_mem=low_mem)
                var3 = self._calculateVar3(cat3, low_mem=low_mem)
                if var1 is not None:
                    self.logger.info(f"var%s1 = %f: {self._sig1} = %f",
                                     self._letter1, var1, math.sqrt(var1))
                if var2 is not None:
                    self.logger.info(f"var%s2 = %f: {self._sig2} = %f",
                                     self._letter2, var2, math.sqrt(var2))
                if var3 is not None:
                    self.logger.info(f"var%s3 = %f: {self._sig3} = %f",
                                     self._letter3, var3, math.sqrt(var3))
            if corr_only:
                w1 = calculateMeanW(cat1, low_mem=low_mem)
                if cat2 is None:
                    w2 = w3 = w1
                elif cat3 is None:
                    w2 = calculateMeanW(cat2, low_mem=low_mem)
                    if self._letter2 == self._letter3:
                        w3 = w2
                    else:
                        w3 = w1
                else:
                    w2 = calculateMeanW(cat2, low_mem=low_mem)
                    w3 = calculateMeanW(cat3, low_mem=low_mem)
                self._meanwww = w1*w2*w3

            finalize_args = [v for v in [var1, var2, var3] if v is not None]
            self.finalize(*finalize_args)

    def _finalize(self):
        mask1 = self.weightr != 0

        if self.bin_type != 'LogMultipole':
            for i in range(8):
                if self._z[i].size:
                    self._z[i][mask1] /= self.weightr[mask1]

        if self._corr_only:
            if self.bin_type == 'LogRUV':
                self.meand2[:] = self.rnom
                self.meanlogd2[:] = self.logr
                self.meanu[:] = self.u
                self.meanv[:] = self.v
                self.meand3[:] = self.u * self.meand2
                self.meanlogd3[:] = np.log(self.meand3)
                self.meand1[:] = np.abs(self.v) * self.meand3 + self.meand2
                self.meanlogd1[:] = np.log(self.meand1)
            else:
                self.meand2[:] = self.d2nom
                self.meanlogd2[:] = self.logd2
                self.meand3[:] = self.d3nom
                self.meanlogd3[:] = self.logd3
                if self.bin_type == 'LogSAS':
                    self.meanphi[:] = self.phi
                    self.meand1[:] = np.sqrt(self.d2nom**2 + self.d3nom**2
                                          - 2*self.d2nom*self.d3nom*np.cos(self.phi))
                    self.meanlogd1[:] = np.log(self.meand1)
                else:
                    self.meanu[:] = 0
                    self.meand1[:] = 0
                    self.meanlogd1[:] = 0
            self.ntri[:] = self.weightr / self._meanwww
        else:
            self.meand2[mask1] /= self.weightr[mask1]
            self.meanlogd2[mask1] /= self.weightr[mask1]
            self.meand3[mask1] /= self.weightr[mask1]
            self.meanlogd3[mask1] /= self.weightr[mask1]
            if self.bin_type != 'LogMultipole':
                self.meand1[mask1] /= self.weightr[mask1]
                self.meanlogd1[mask1] /= self.weightr[mask1]
                self.meanu[mask1] /= self.weightr[mask1]
            if self.bin_type == 'LogRUV':
                self.meanv[mask1] /= self.weightr[mask1]

            # Update the units
            self._apply_units(mask1)

            # Set to nominal when no triangles in bin.
            mask2 = self.weightr == 0
            if self.bin_type == 'LogRUV':
                self.meand2[mask2] = self.rnom[mask2]
                self.meanlogd2[mask2] = self.logr[mask2]
                self.meanu[mask2] = self.u[mask2]
                self.meanv[mask2] = self.v[mask2]
                self.meand3[mask2] = self.u[mask2] * self.meand2[mask2]
                self.meanlogd3[mask2] = np.log(self.meand3[mask2])
                self.meand1[mask2] = np.abs(self.v[mask2]) * self.meand3[mask2] + self.meand2[mask2]
                self.meanlogd1[mask2] = np.log(self.meand1[mask2])
            else:
                self.meand2[mask2] = self.d2nom[mask2]
                self.meanlogd2[mask2] = self.logd2[mask2]
                self.meand3[mask2] = self.d3nom[mask2]
                self.meanlogd3[mask2] = self.logd3[mask2]
                if self.bin_type == 'LogSAS':
                    self.meanu[mask2] = self.phi[mask2]
                    self.meand1[mask2] = np.sqrt(self.d2nom[mask2]**2 + self.d3nom[mask2]**2
                                                 - 2*self.d2nom[mask2]*self.d3nom[mask2]*
                                                 np.cos(self.phi[mask2]))
                    self.meanlogd1[mask2] = np.log(self.meand1[mask2])
                else:
                    self.meanu[mask2] = 0
                    self.meand1[mask2] = 0
                    self.meanlogd1[mask2] = 0

            if self.bin_type == 'LogMultipole':
                # Multipole only sets the meand values at [i,j,max_n].
                # (This is also where the complex weight is just a scalar = sum(www),
                # so the above normalizations are correct.)
                # Broadcast those to the rest of the values in the third dimension.
                self.ntri[:,:,:] = self.ntri[:,:,self.max_n][:,:,np.newaxis]
                self.meand2[:,:,:] = self.meand2[:,:,self.max_n][:,:,np.newaxis]
                self.meanlogd2[:,:,:] = self.meanlogd2[:,:,self.max_n][:,:,np.newaxis]
                self.meand3[:,:,:] = self.meand3[:,:,self.max_n][:,:,np.newaxis]
                self.meanlogd3[:,:,:] = self.meanlogd3[:,:,self.max_n][:,:,np.newaxis]

    def _clear(self):
        """Clear the data vectors
        """
        for i in range(8):
            self._z[i][:] = 0.
        self.meand1[:] = 0.
        self.meanlogd1[:] = 0.
        self.meand2[:] = 0.
        self.meanlogd2[:] = 0.
        self.meand3[:] = 0.
        self.meanlogd3[:] = 0.
        self.meanu[:] = 0.
        if self.bin_type == 'LogRUV':
            self.meanv[:] = 0.
        self.weightr[:] = 0.
        if self.bin_type == 'LogMultipole':
            self.weighti[:] = 0.
        self.ntri[:] = 0.
        self._cov = None
        self._varzeta = None

    def __iadd__(self, other):
        """Add a second Correlation object's data to this one.

        .. note::

            For this to make sense, both objects should not have had ``finalize`` called yet.
            Then, after adding them together, you should call ``finalize`` on the sum.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can only add another {self._cls} object")
        if not self._equal_binning(other, brief=True):
            raise ValueError(f"{self._cls} to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords, other.coords, other.coords)
        if not other.nonzero: return self
        for i in range(8):
            self._z[i][:] += other._z[i][:]
        self.meand1[:] += other.meand1[:]
        self.meanlogd1[:] += other.meanlogd1[:]
        self.meand2[:] += other.meand2[:]
        self.meanlogd2[:] += other.meanlogd2[:]
        self.meand3[:] += other.meand3[:]
        self.meanlogd3[:] += other.meanlogd3[:]
        self.meanu[:] += other.meanu[:]
        if self.bin_type == 'LogRUV':
            self.meanv[:] += other.meanv[:]
        self.weightr[:] += other.weightr[:]
        if self.bin_type == 'LogMultipole':
            self.weighti[:] += other.weighti[:]
        self.ntri[:] += other.ntri[:]
        return self

    def _sum(self, others, corr_only):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other,w in others:
        #         self += w*other
        # if w*other was valid syntax, which it isn't.

        def x(a, w):
            return a if w == 1 else a*w

        for i in range(8):
            if self._z[i].size:
                np.sum([x(c._z[i],w) for c,w in others], axis=0, out=self._z[i])
        np.sum([x(c.weightr,w) for c,w in others], axis=0, out=self.weightr)
        if self.bin_type == 'LogMultipole':
            np.sum([x(c.weighti,w) for c,w in others], axis=0, out=self.weighti)

        if not corr_only:
            np.sum([x(c.meand1,w) for c,w in others], axis=0, out=self.meand1)
            np.sum([x(c.meanlogd1,w) for c,w in others], axis=0, out=self.meanlogd1)
            np.sum([x(c.meand2,w) for c,w in others], axis=0, out=self.meand2)
            np.sum([x(c.meanlogd2,w) for c,w in others], axis=0, out=self.meanlogd2)
            np.sum([x(c.meand3,w) for c,w in others], axis=0, out=self.meand3)
            np.sum([x(c.meanlogd3,w) for c,w in others], axis=0, out=self.meanlogd3)
            np.sum([x(c.meanu,w) for c,w in others], axis=0, out=self.meanu)
            if self.bin_type == 'LogRUV':
                np.sum([x(c.meanv,w) for c,w in others], axis=0, out=self.meanv)
            np.sum([x(c.ntri,w) for c,w in others], axis=0, out=self.ntri)

        self._cov = None
        self._varzeta = None

    def _calculate_varzeta(self, n):
        self._varzeta = [np.zeros(self.data_shape) for _ in range(n)]
        if self._var_num != 0:
            for i in range(n):
                self._varzeta[i].ravel()[:] = self.cov_diag[i*self._nbins:(i+1)*self._nbins].real

    def _x_to_natural_phases(self):
        phases = [None] * len(self._zetas)
        if len(set('VGTQ') & set(self._letters)) > 0:
            # Now fix the projection.
            # The multipole algorithm uses the Porth et al x projection.
            # We need to switch that to the canoical centroid projection.

            # (Comments here are for GGG, but we can use this for any class
            # that has at least one non-spin 0 value needing a projection.)

            # Define some complex "vectors" where p1 is at the origin and
            # p3 is on the x axis:
            # s = p3 - p1
            # t = p2 - p1
            # u = angle bisector of s, t
            # q1 = (s+t)/3.  (this is the centroid)
            # q2 = q1-t
            # q3 = q1-s
            s = self.meand2
            t = self.meand3 * np.exp(1j * self.meanphi * self._phi_units)
            u = (1 + t/np.abs(t))/2
            q1 = (s+t)/3.
            q2 = q1-t
            q3 = q1-s

            # Currently the projection is as follows:
            # g1 is projected along u
            # g2 is projected along t
            # g3 is projected along s
            #
            # We want to have
            # g1 projected along q1
            # g2 projected along q2
            # g3 projected along q3
            #
            # The phases to multiply by are exp(2iphi_current) * exp(-2iphi_target). I.e.
            # g1phase = (u conj(q1))**2 / |u conj(q1)|**2
            # g2phase = (t conj(q2))**2 / |t conj(q2)|**2
            # g3phase = (s conj(q3))**2 / |s conj(q3)|**2
            g1phase = (u * np.conj(q1))**spin_by_letter(self._letter1)
            g2phase = (t * np.conj(q2))**spin_by_letter(self._letter2)
            g3phase = (s * np.conj(q3))**spin_by_letter(self._letter3)
            g1phase /= np.abs(g1phase)
            g2phase /= np.abs(g2phase)
            g3phase /= np.abs(g3phase)

            # Now just multiply each gam by the appropriate combination of phases.
            phases[0] = g1phase * g2phase * g3phase

            if len(self._zetas) == 4:
                phases[1] = np.conj(g1phase) * g2phase * g3phase
                phases[2] = g1phase * np.conj(g2phase) * g3phase
                phases[3] = g1phase * g2phase * np.conj(g3phase)
            elif len(self._zetas) == 2:
                # This one is a little tricky.  It means there are two spinny vertices.
                # If letter1 is N or K, then the conjugate is vertex 2
                # otherwise it is vertex 1.
                if self._letter1 in 'NK':
                    assert np.all(g1phase == 1)
                    phases[1] = np.conj(g2phase) * g3phase
                else:
                    phases[1] = np.conj(g1phase) * g2phase * g3phase
        return phases

    def toSAS(self, *, target=None, **kwargs):
        """Convert a multipole-binned correlation to the corresponding SAS binning.

        This is only valid for bin_type == LogMultipole.

        Keyword Arguments:
            target:     A target Correlation object with LogSAS binning to write to.
                        If this is not given, a new object will be created based on the
                        configuration paramters of the current object. (default: None)
            **kwargs:   Any kwargs that you want to use to configure the returned object.
                        Typically, might include min_phi, max_phi, nphi_bins, phi_bin_size.
                        The default phi binning is [0,pi] with nphi_bins = self.max_n.

        Returns:
            An object with bin_type=LogSAS containing the same information as this object,
            but with the SAS binning.
        """
        # Each class will add a bit to this.  The implemenation here is the common code
        # that applies to all the different classes.

        if self.bin_type != 'LogMultipole':
            raise TypeError("toSAS is invalid for bin_type = %s"%self.bin_type)

        if target is None:
            config = self.config.copy()
            config['bin_type'] = 'LogSAS'
            max_n = config.pop('max_n')
            if 'nphi_bins' not in kwargs and 'phi_bin_size' not in kwargs:
                config['nphi_bins'] = max_n
            sas = self.__class__(config, **kwargs)
        else:
            if not isinstance(target, self.__class__):
                raise ValueError(f"target must be an instance of {self.__class__}")
            sas = target
            sas.clear()
        if not np.array_equal(sas.rnom1d, self.rnom1d):
            raise ValueError("toSAS cannot change sep parameters")

        # Copy these over
        sas.meand2[:,:,:] = self.meand2[:,:,0][:,:,None]
        sas.meanlogd2[:,:,:] = self.meanlogd2[:,:,0][:,:,None]
        sas.meand3[:,:,:] = self.meand3[:,:,0][:,:,None]
        sas.meanlogd3[:,:,:] = self.meanlogd3[:,:,0][:,:,None]
        sas.npatch1 = self.npatch1
        sas.npatch2 = self.npatch2
        sas.npatch3 = self.npatch3
        sas.coords = self.coords
        sas.metric = self.metric
        sas._var_num = self._var_num

        # Use nominal for meanphi
        sas.meanu[:] = sas.phi / sas._phi_units
        # Compute d1 from actual d2,d3 and nominal phi
        sas.meand1[:] = np.sqrt(sas.meand2**2 + sas.meand3**2
                                - 2*sas.meand2 * sas.meand3 * np.cos(sas.phi))
        sas.meanlogd1[:] = np.log(sas.meand1)

        # Eqn 26 of Porth et al, 2023
        # N(d2,d3,phi) = 1/2pi sum_n N_n(d2,d3) exp(i n phi)
        expiphi = np.exp(1j * self.n1d[:,None] * sas.phi1d)
        sas.weightr[:] = np.real(self.weight.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size

        # For ntri, we recorded the total ntri for each pair of d2,d3.
        # Allocate those proportionally to the weights.
        # Note: Multipole counts the weight for all 0 < phi < 2pi.
        # We reduce this by the fraction of this covered by [min_phi, max_phi].
        # (Typically 1/2, since usually [0,pi].)
        phi_frac = (sas.max_phi - sas.min_phi) / (2*np.pi)
        denom = np.sum(sas.weight, axis=2)
        denom[denom==0] = 1  # Don't divide by 0
        ratio = self.ntri[:,:,0] / denom * phi_frac
        sas.ntri[:] = sas.weight * ratio[:,:,None]

        for k,v in self.results.items():
            temp = sas.copy()
            temp.weightr[:] = np.real(v.weight.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size
            temp.ntri[:] = temp.weight * ratio[:,:,None]

            # Undo the normalization of the d arrays.
            temp.meand1 *= temp.weightr
            temp.meand2 *= temp.weightr
            temp.meand3 *= temp.weightr
            temp.meanlogd1 *= temp.weightr
            temp.meanlogd2 *= temp.weightr
            temp.meanlogd3 *= temp.weightr
            temp.meanu *= temp.weightr

            sas.results[k] = temp

        if self._z[0].size == 0:
            # NNN doesn't have a zeta to compute
            return sas

        # Z(d2,d3,phi) = 1/2pi sum_n Z_n(d2,d3) exp(i n phi)
        expiphi = np.exp(1j * self.n1d[:,None] * sas.phi1d)

        zetas = [z.dot(expiphi) / (2*np.pi) * sas.phi_bin_size for z in self._zetas]

        # We leave the gam_mu unnormalized in the Multipole class, so after the FT,
        # we still need to divide by weight.
        mask = sas.weightr != 0
        for z in zetas:
            z[mask] /= sas.weightr[mask]

        phases = sas._x_to_natural_phases()
        zetas = [z*p if p is not None else z for z,p in zip(zetas, phases)]

        for i in range(len(zetas)):
            sas._z[2*i][:] = np.real(zetas[i])
            if sas._z[2*i+1].size:
                # KKK for instance has real zeta, so no z[1]
                sas._z[2*i+1][:] = np.imag(zetas[i])

        for k,v in self.results.items():
            temp = sas.results[k]
            zetas = [z.dot(expiphi) / (2*np.pi) * sas.phi_bin_size for z in v._zetas]
            for i in range(len(zetas)):
                if phases[i] is not None:
                    zetas[i] *= phases[i]
                temp._z[2*i][:] = np.real(zetas[i])
                if sas._z[2*i+1].size:
                    temp._z[2*i+1][:] = np.imag(zetas[i])

        return sas

    def estimate_cov(self, method, *, func=None, comm=None, num_bootstrap=None,
                     cross_patch_weight=None):
        """Estimate the covariance matrix based on the data

        This function will calculate an estimate of the covariance matrix according to the
        given method.

        Options for ``method`` include:

            - 'shot' = The variance based on "shot noise" only.  This includes the Poisson
              counts of points for N statistics, shape noise for G statistics, and the observed
              scatter in the values for K statistics.  In this case, the returned value will
              only be the diagonal.  Use np.diagonal(cov) if you actually want a full
              matrix from this.
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

        Another relevant parameter is 'cross_patch_weight'. This parameter controls how
        triangles that cross between two or three patches are weighted when some, but not all
        of the patches are selected.  See Mohammad and Percival (2021)
        (https://arxiv.org/abs/2109.07071) for an in-depth discussion of these options for
        two-point statistics.  We use a similar definitions for three-point statistics.
        Briefly the options are: (TODO!  This is aspirational so far.)

            - 'simple' = Don't use any triangles where any object is in a deselected patch.
              This is currently the default for all methods.
            - 'mean' = Use a weight of 1/3 for any triangle with one object in a selected patch
              and the other two in deselected patches, and 2/3 for any triangle with two objects
              in selected patches.
            - 'geom' = Use the geometric mean of the three patch weights for each triangle.
            - 'match' = Use the "optimal" weight that matches the effect of auto- and cross-pairs
              for two-point jackknife covariances derived by Mohammad and Percival
              (w = n_patch / (2 + sqrt(2) (n_patch-1))).
              There is a similar formula for triangles that span three different patches
              (w = sqrt(2) n_patch / 3 (n_patch - 1 + sqrt(2))), which we use for those triples.

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
                                returns the desired data vector. (default: None, which is
                                equivalent to ``lambda corr: corr.getStat()``)
            comm (mpi4py.Comm): If using MPI, an mpi4py Comm object to communicate between
                                processes.  (default: None)
            num_bootstrap (int): How many bootstrap samples to use for the 'bootstrap' and
                                'marked_bootstrap' var_methods.  (default: 500; this value
                                can also be given in the constructor.)
            cross_patch_weight (str): How to weight pairs that cross between two patches when one
                                patch is deselected (e.g. in a jackknife sense) and the other is
                                selected.  (default 'simple'; this value can also be given in
                                the constructor.)

        Returns:
            A numpy array with the estimated covariance matrix.
        """
        if num_bootstrap is None:
            num_bootstrap = self.num_bootstrap
        if func is not None:
            # Need to convert it to a function of the first item in the list.
            all_func = lambda corrs: func(corrs[0])
        else:
            all_func = None
        return estimate_multi_cov([self], method, func=all_func, comm=comm,
                                  num_bootstrap=num_bootstrap,
                                  cross_patch_weight=cross_patch_weight)

    def build_cov_design_matrix(self, method, *, func=None, comm=None, num_bootstrap=None,
                                cross_patch_weight=None):
        r"""Build the design matrix that is used for estimating the covariance matrix.

        The design matrix for patch-based covariance estimates is a matrix where each row
        corresponds to a different estimate of the data vector, :math:`\zeta_i` (or
        :math:`f(\zeta_i)` if using the optional ``func`` parameter).

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
                                desired full data vector. (default: None, which is equivalent to
                                ``lambda corrs: np.concatenate([c.getStat() for c in corrs])``)
            comm (mpi4py.Comm): If using MPI, an mpi4py Comm object to communicate between
                                processes.  (default: None)
            num_bootstrap (int): How many bootstrap samples to use for the 'bootstrap' and
                                'marked_bootstrap' var_methods.  (default: 500; this value
                                can also be given in the constructor.)
            cross_patch_weight (str): How to weight pairs that cross between two patches when one
                                patch is deselected (e.g. in a jackknife sense) and the other is
                                selected.  (default 'simple'; this value can also be given in the
                                constructor.)

        Returns:
            A, w: numpy arrays with the design matrix and weights respectively.
        """
        if num_bootstrap is None:
            num_bootstrap = self.num_bootstrap
        if func is not None:
            # Need to convert it to a function of the first item in the list.
            all_func = lambda corrs: func(corrs[0])
        else:
            all_func = None
        return build_multi_cov_design_matrix([self], method=method, func=all_func, comm=comm,
                                             num_bootstrap=num_bootstrap,
                                             cross_patch_weight=cross_patch_weight)

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
        if any(c not in ['3d',None] for c in [coords1, coords2, coords3]):
            if self.min_rpar != -sys.float_info.max:
                raise ValueError("min_rpar is only valid for 3d coordinates")
            if self.max_rpar != sys.float_info.max:
                raise ValueError("max_rpar is only valid for 3d coordinates")
        if self.sep_units != '' and coords == '3d' and metric != 'Arc':
            raise ValueError("sep_units is invalid with 3d coordinates. "
                             "min_sep and max_sep should be in the same units as r (or x,y,z).")
        if metric == 'Periodic':
            if self.xperiod == 0 or self.yperiod == 0 or (coords=='3d' and self.zperiod == 0):
                raise ValueError("Periodic metric requires setting the period to use.")
        else:
            if self.xperiod != 0 or self.yperiod != 0 or self.zperiod != 0:
                raise ValueError("period options are not valid for %s metric."%metric)

        if self.coords is not None or self.metric is not None:
            if coords != self.coords:
                self.logger.warning("Detected a change in catalog coordinate systems.\n"+
                                    "This probably doesn't make sense!")
            if metric != self.metric:
                self.logger.warning("Detected a change in metric.\n"+
                                    "This probably doesn't make sense!")

        self.coords = coords  # These are the regular string values
        self.metric = metric
        self._coords = coord_enum(coords)  # These are the C++-layer enums
        self._metric = metric_enum(metric)

    @lazy_property
    def _zero_array(self):
        # An array of all zeros with the same shape as the data arrays
        z = np.zeros(self.data_shape)
        z.flags.writeable=False  # Just to make sure we get an error if we try to change it.
        return z

    def _apply_units(self, mask):
        if self.coords == 'spherical' and self.metric == 'Euclidean':
            # Then we need to convert from the chord triangles to great circle triangles.

            # If SAS, then first fix phi.
            # The real spherical trig formula is:
            # cos(c) = cos(a) cos(b) + sin(a) sin(b) cos(phi)
            # Using d1 = 2 sin(c/2), d2 = 2 sin(a/2), d3 = 2 sin(b/2), this becomes:
            # d1^2 = d2^2 + d3^2 - 2 d2 d3 [ 1/4 d2 d3 + cos(a/2) cos(b/2) cos(phi) ]
            # The thing in [] is what we currently have for cos(phi).
            if self.bin_type == 'LogSAS':
                cosphi = np.cos(self.meanphi[mask])
                cosphi -= 0.25 * self.meand2[mask] * self.meand3[mask]
                cosphi /= np.sqrt( (1 - self.meand2[mask]**2/4) * (1 - self.meand3[mask]**2/4) )
                cosphi[cosphi < -1] = -1  # Just in case...
                cosphi[cosphi > 1] = 1
                self.meanphi[mask] = np.arccos(cosphi)

            # Also convert the chord distance to a real angle.
            # L = 2 sin(theta/2)
            if self.bin_type != 'LogMultipole':
                self.meand1[mask] = 2. * np.arcsin(self.meand1[mask]/2.)
                self.meanlogd1[mask] = np.log(2.*np.arcsin(np.exp(self.meanlogd1[mask])/2.))
            self.meand2[mask] = 2. * np.arcsin(self.meand2[mask]/2.)
            self.meanlogd2[mask] = np.log(2.*np.arcsin(np.exp(self.meanlogd2[mask])/2.))
            self.meand3[mask] = 2. * np.arcsin(self.meand3[mask]/2.)
            self.meanlogd3[mask] = np.log(2.*np.arcsin(np.exp(self.meanlogd3[mask])/2.))

        if self.bin_type == 'LogSAS':
            self.meanphi[mask] /= self._phi_units
        if self.bin_type != 'LogMultipole':
            self.meand1[mask] /= self._sep_units
            self.meanlogd1[mask] -= self._log_sep_units
        self.meand2[mask] /= self._sep_units
        self.meanlogd2[mask] -= self._log_sep_units
        self.meand3[mask] /= self._sep_units
        self.meanlogd3[mask] -= self._log_sep_units

    def _get_minmax_size(self):
        if self.metric == 'Euclidean':
            if self.bin_type == 'LogRUV':
                # The minimum separation we care about is that of the smallest size, which is
                # min_sep * min_u.  Do the same calculation as for 2pt to get to min_size.
                b1 = min(self.angle_slop, self.b, self.bu, self.bv)
                min_size = self._min_sep * self.min_u * b1 / (2.+3.*b1)

                # This time, the maximum size is d1 * b.  d1 can be as high as 2*max_sep.
                b2 = min(self.angle_slop, self.b)
                max_size = 2. * self._max_sep * b2
            elif self.bin_type == 'LogSAS':
                # LogSAS
                b1 = min(self.angle_slop, self.b)
                min_size1 = self._min_sep * b1 / (2.+3.*b1)
                b2 = min(self.angle_slop, self.bu)
                min_size2 = 2 * self._min_sep * np.tan(self.min_phi/2) * b2 / (2+3*b2)
                min_size = min(min_size1, min_size2)
                max_size = self._max_sep * b1
            else:
                # LogMultipole
                b1 = min(self.angle_slop, self.b)
                min_size = 2 * self._min_sep * b1 / (2.+3*b1)
                max_size = self._max_sep * b1
            return min_size, max_size
        else:
            return 0., 0.

    # The three-point versions of the covariance helpers.
    # Note: the word "pairs" in many of these was appropriate for 2pt, but in the 3pt case
    # these actually refer to triples (i,j,k).

    def _get_npatch(self):
        return max(self.npatch1, self.npatch2, self.npatch3)

    def _calculate_xi_from_pairs(self, pairs, corr_only):
        # Compute the xi data vector for the given list of pairs.
        # "pairs" here are really triples, input as a list of (i,j,k) values.

        # This is the normal calculation.  It needs to be overridden when there are randoms.
        self._sum([(self.results[(i,j,k)],w) for i,j,k,w in pairs], corr_only)
        self._finalize()

    def _keep_ok(self, pairs):
        return [(i,j,k,w) for i,j,k,w in pairs if self._ok[i,j,k] and w != 0]

    class TripleIterator(collections.abc.Iterator):
        def __init__(self, results, npatch1, npatch2, npatch3, index, cpw, ok=None):
            self.results = results
            self.npatch1 = npatch1
            self.npatch2 = npatch2
            self.npatch3 = npatch3
            self.index = index
            self.cpw = cpw
            self.ok = ok

        def __iter__(self):
            self.gen = iter(self.make_gen())
            return self

        def __next__(self):
            return next(self.gen)

    class JackknifeTripleIterator(TripleIterator):
        def make_gen(self):
            if self.cpw not in ['simple', 'mean', 'match']:
                raise ValueError(f"cross_patch_weight = {self.cpw} is invalid for jackknife")
            if self.npatch3 == 1:
                if self.npatch2 == 1:
                    # j=k=0
                    return ((i,j,k,1) for i,j,k in self.results.keys() if i!=self.index)
                elif self.npatch1 == 1:
                    # i=k=0
                    return ((i,j,k,1) for i,j,k in self.results.keys() if j!=self.index)
                elif self.cpw == 'simple':
                    # k=0
                    assert self.npatch1 == self.npatch2
                    return ((i,j,k,1) for i,j,k in self.results.keys()
                            if i!=self.index and j!=self.index)
                elif self.cpw == 'match':
                    w = 1 - self.npatch1 / (2 + 2**0.5 * (self.npatch1-1))
                    assert self.npatch1 == self.npatch2
                    return ((i,j,k, w if i == self.index or j == self.index else 1)
                            for i,j,k in self.results.keys() if i!=self.index or j!=self.index)
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch2
                    return ((i,j,k, 0.5 if i == self.index or j == self.index else 1)
                            for i,j,k in self.results.keys() if i!=self.index or j!=self.index)
            elif self.npatch2 == 1:
                if self.npatch1 == 1:
                    # i=j=0
                    return ((i,j,k,1) for i,j,k in self.results.keys() if k!=self.index)
                elif self.cpw == 'simple':
                    # j=0
                    assert self.npatch1 == self.npatch3
                    return ((i,j,k,1) for i,j,k in self.results.keys()
                            if i!=self.index and k!=self.index)
                elif self.cpw == 'match':
                    w = 1 - self.npatch1 / (2 + 2**0.5 * (self.npatch1-1))
                    assert self.npatch1 == self.npatch3
                    return ((i,j,k, w if i == self.index or k == self.index else 1)
                            for i,j,k in self.results.keys() if i!=self.index or k!=self.index)
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch3
                    return ((i,j,k, 0.5 if i == self.index or k == self.index else 1)
                            for i,j,k in self.results.keys() if i!=self.index or k!=self.index)
            elif self.npatch1 == 1:
                # i=0
                assert self.npatch2 == self.npatch3
                if self.cpw == 'simple':
                    return ((i,j,k,1) for i,j,k in self.results.keys()
                            if j!=self.index and k!=self.index)
                elif self.cpw == 'match':
                    w = 1 - self.npatch2 / (2 + 2**0.5 * (self.npatch2-1))
                    assert self.npatch2 == self.npatch3
                    return ((i,j,k, w if j == self.index or k == self.index else 1)
                            for i,j,k in self.results.keys() if j!=self.index or k!=self.index)
                else:
                    assert self.cpw == 'mean'
                    return ((i,j,k, 0.5 if j == self.index or k == self.index else 1)
                            for i,j,k in self.results.keys() if j!=self.index or k!=self.index)
            else:
                assert self.npatch1 == self.npatch2 == self.npatch3
                if self.cpw == 'simple':
                    return ((i,j,k,1) for i,j,k in self.results.keys()
                            if i!=self.index and j!=self.index and k!=self.index)
                elif self.cpw == 'match':
                    # For 3pt, there are two different optimal match weights depending
                    # on the nature of the triangle.  If two points are in one patch, and one
                    # point is in another, then the same weight as MP22 is appropriate.
                    # For triangles where all three points are in different patches, the
                    # derivation is as follows:
                    #
                    #           n(n-1)(n-2)/6 DDDm - beta (n-1)(n-2)/2 DDDk
                    # theta_c = -------------------------------------------
                    #                n(n-1)(n-2)/6 - beta (n-1)(n-2)/2
                    #
                    # Then, (theta_c - <theta_c>)^2 comes out to
                    #   9 S beta^2 n (n-1) / 2 (3beta-n)^2
                    # As in MP22, set this equal to n S / (n-1) and solve for beta, we get
                    # beta = sqrt(2) n / 3 (n - 1 + sqrt(2))
                    w1 = 1 - self.npatch2 / (2 + 2**0.5 * (self.npatch2-1))
                    w2 = 1 - 2**0.5 * self.npatch1 / (3*(self.npatch1 - 1 + 2**0.5))
                    assert self.npatch1 == self.npatch2 == self.npatch3
                    return ((i,j,k,
                                1 if (i != self.index and j != self.index and k != self.index)
                                else w2 if (i != j and i != k and j != k)
                                else w1)
                            for i,j,k in self.results.keys()
                            if i!= self.index or j!=self.index or k!=self.index)
                else:
                    assert self.cpw == 'mean'
                    # 1/3 if 2 points are index, 2/3 if 1 point is index, 1 if none are index
                    return ((i,j,k, sum(p!=self.index for p in (i,j,k))/3)
                            for i,j,k in self.results.keys()
                            if i!=self.index or j!=self.index or k!=self.index)

    def _jackknife_pairs(self, cross_patch_weight):
        np = max(self.npatch1, self.npatch2, self.npatch3)
        return [self.JackknifeTripleIterator(self.results, self.npatch1, self.npatch2,
                                             self.npatch3, i, cross_patch_weight)
                for i in range(np)]

    class SampleTripleIterator(TripleIterator):
        def make_gen(self):
            if self.cpw not in ['simple', 'mean']:
                raise ValueError(f"cross_patch_weight = {self.cpw} is invalid for sample")
            if self.npatch3 == 1:
                if self.npatch2 == 1:
                    # j=k=0
                    return ((i,j,k,1) for i,j,k in self.results.keys() if i==self.index)
                elif self.npatch1 == 1:
                    # i=k=0
                    return ((i,j,k,1) for i,j,k in self.results.keys() if j==self.index)
                elif self.cpw == 'simple':
                    # k=0
                    assert self.npatch1 == self.npatch2
                    return ((i,j,k,1) for i,j,k in self.results.keys() if i==self.index)
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch2
                    return ((i,j,k, 0.5 if i!=self.index or j!=self.index else 1)
                            for i,j,k in self.results.keys() if i==self.index or j==self.index)
            elif self.npatch2 == 1:
                if self.npatch1 == 1:
                    # i=j=0
                    return ((i,j,k,1) for i,j,k in self.results.keys() if k==self.index)
                elif self.cpw == 'simple':
                    # j=0
                    assert self.npatch1 == self.npatch3
                    return ((i,j,k,1) for i,j,k in self.results.keys() if i==self.index)
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch3
                    return ((i,j,k, 0.5 if i!=self.index or k!=self.index else 1)
                            for i,j,k in self.results.keys() if i==self.index or k==self.index)
            elif self.npatch1 == 1:
                # i=0
                assert self.npatch2 == self.npatch3
                if self.cpw == 'simple':
                    return ((i,j,k,1) for i,j,k in self.results.keys() if j==self.index)
                else:
                    assert self.cpw == 'mean'
                    return ((i,j,k, 0.5 if j!=self.index or k!=self.index else 1)
                            for i,j,k in self.results.keys() if j==self.index or k==self.index)
            else:
                assert self.npatch1 == self.npatch2 == self.npatch3
                if self.cpw == 'simple':
                    return ((i,j,k,1) for i,j,k in self.results.keys() if i==self.index)
                else:
                    assert self.cpw == 'mean'
                    return ((i,j,k, sum(p==self.index for p in (i,j,k))/3)
                            for i,j,k in self.results.keys()
                            if i==self.index or j==self.index or k==self.index)

    def _sample_pairs(self, cross_patch_weight):
        np = max(self.npatch1, self.npatch2, self.npatch3)
        return [self.SampleTripleIterator(self.results, self.npatch1, self.npatch2,
                                          self.npatch3, i, cross_patch_weight)
                for i in range(np)]

    @lazy_property
    def _ok(self):
        ok = np.zeros((self.npatch1, self.npatch2, self.npatch3), dtype=bool)
        for (i,j,k) in self.results:
            ok[i,j,k] = True
        return ok

    class MarkedTripleIterator(TripleIterator):
        def make_gen(self):
            if self.cpw not in ['simple', 'mean']:
                raise ValueError(f"cross_patch_weight = {self.cpw} is invalid for marked_bootstrap")
            if self.npatch3 == 1:
                if self.npatch2 == 1:
                    return ( (i,0,0,1) for i in self.index if self.ok[i,0,0] )
                elif self.npatch1 == 1:
                    return ( (0,j,0,1) for j in self.index if self.ok[0,j,0] )
                elif self.cpw == 'simple':
                    assert self.npatch1 == self.npatch2
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (i,j,0,w) for (i,w) in zip(index, weights)
                             for j in range(self.npatch2) if self.ok[i,j,0] )
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch2
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (i,j,0,w) if i == j else (i,j,0,w/2) if self.ok[i,j,0] else (j,i,0,w/2)
                             for i,w in zip(index, weights)
                             for j in range(self.npatch2) if self.ok[i,j,0] or self.ok[j,i,0] )
            elif self.npatch2 == 1:
                if self.npatch1 == 1:
                    return ( (0,0,k,1) for k in self.index if self.ok[0,0,k] )
                elif self.cpw == 'simple':
                    assert self.npatch1 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (i,0,k,w) for (i,w) in zip(index, weights)
                             for k in range(self.npatch3) if self.ok[i,0,k] )
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (i,0,k,w) if i == k else (i,0,k,w/2) if self.ok[i,0,k] else (k,0,i,w/2)
                             for i,w in zip(index, weights)
                             for k in range(self.npatch3) if self.ok[i,0,k] or self.ok[k,0,i] )
            elif self.npatch1 == 1:
                assert self.npatch2 == self.npatch3
                if self.cpw == 'simple':
                    assert self.npatch2 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (0,j,k,w) for (j,w) in zip(index, weights)
                             for k in range(self.npatch3) if self.ok[0,j,k] )
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch2 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (0,j,k,w) if j == k else (0,j,k,w/2) if self.ok[0,j,k] else (0,k,j,w/2)
                             for j,w in zip(index, weights)
                             for k in range(self.npatch3) if self.ok[0,j,k] or self.ok[0,k,j] )
            else:
                assert self.npatch1 == self.npatch2 == self.npatch3
                if self.cpw == 'simple':
                    assert self.npatch1 == self.npatch2 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (i,j,k,w) for (i,w) in zip(index, weights)
                             for j in range(self.npatch2)
                             for k in range(self.npatch3) if self.ok[i,j,k] )
                else:
                    assert self.cpw == 'mean'
                    assert self.npatch1 == self.npatch2 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    return ( (i,j,k, sum(p==i for p in (i,j,k))/3 * w)
                             for i,w in zip(index, weights)
                             for j in range(self.npatch2)
                             for k in range(self.npatch3) if self.ok[i,j,k] )

    def _marked_pairs(self, index, cross_patch_weight):
        return self.MarkedTripleIterator(self.results, self.npatch1, self.npatch2,
                                         self.npatch3, index, cross_patch_weight, self._ok)

    class BootstrapTripleIterator(TripleIterator):
        def make_gen(self):
            if self.cpw not in ['simple', 'mean', 'geom']:
                raise ValueError(f"cross_patch_weight = {self.cpw} is invalid for bootstrap")
            if self.npatch3 == 1:
                if self.npatch2 == 1:
                    return ( (i,0,0,1) for i in self.index if self.ok[i,0,0] )
                elif self.npatch1 == 1:
                    return ( (0,j,0,1) for j in self.index if self.ok[0,j,0] )
                else:
                    assert self.npatch1 == self.npatch2
                    index, weights = np.unique(self.index, return_counts=True)
                    ret1 = ( (i,i,0,w) for (i,w) in zip(index, weights) if self.ok[i,i,0] )
                    if self.cpw == 'simple':
                        ret2 = ( (i,j,0,w1*w2) for (i,w1) in zip(index, weights)
                                 for (j,w2) in zip(index,weights) if self.ok[i,j,0] and i != j )
                    elif self.cpw == 'mean':
                        wdict = dict(zip(index, weights))
                        ret2 = ( (i,j,0,(wdict.get(i,0)+wdict.get(j,0))/2)
                                for i in range(self.npatch1) for j in range(self.npatch2)
                                if self.ok[i,j,0] and i!=j )
                    else:
                        assert self.cpw == 'geom'
                        ret2 = ( (i,j,0,(w1*w2)**0.5) for (i,w1) in zip(index, weights)
                                 for (j,w2) in zip(index,weights) if self.ok[i,j,0] and i != j )
                    return itertools.chain(ret1, ret2)
            elif self.npatch2 == 1:
                if self.npatch1 == 1:
                    return [ (0,0,i,1) for i in self.index if self.ok[0,0,i] ]
                else:
                    assert self.npatch1 == self.npatch3
                    index, weights = np.unique(self.index, return_counts=True)
                    ret1 = ( (i,0,i,w) for (i,w) in zip(index, weights) if self.ok[i,0,i] )
                    if self.cpw == 'simple':
                        ret2 = ( (i,0,k,w1*w2) for (i,w1) in zip(index, weights)
                                 for (k,w2) in zip(index,weights) if self.ok[i,0,k] and i != k )
                    elif self.cpw == 'mean':
                        wdict = dict(zip(index, weights))
                        ret2 = ( (i,0,k,(wdict.get(i,0)+wdict.get(k,0))/2)
                                for i in range(self.npatch1) for k in range(self.npatch3)
                                if self.ok[i,0,k] and i!=k )
                    else:
                        assert self.cpw == 'geom'
                        ret2 = ( (i,0,k,(w1*w2)**0.5) for (i,w1) in zip(index, weights)
                                 for (k,w2) in zip(index,weights) if self.ok[i,0,k] and i != k )
                    return itertools.chain(ret1, ret2)
            elif self.npatch1 == 1:
                assert self.npatch2 == self.npatch3
                index, weights = np.unique(self.index, return_counts=True)
                ret1 = ( (0,j,j,w) for (j,w) in zip(index, weights) if self.ok[0,j,j] )
                if self.cpw == 'simple':
                    ret2 = ( (0,j,k,w1*w2) for (j,w1) in zip(index, weights)
                                for (k,w2) in zip(index,weights) if self.ok[0,j,k] and j != k )
                elif self.cpw == 'mean':
                    wdict = dict(zip(index, weights))
                    ret2 = ( (0,j,k,(wdict.get(j,0)+wdict.get(k,0))/2)
                            for j in range(self.npatch2) for k in range(self.npatch3)
                            if self.ok[0,j,k] and j!=k )
                else:
                    assert self.cpw == 'geom'
                    ret2 = ( (0,j,k,(w1*w2)**0.5) for (j,w1) in zip(index, weights)
                                for (k,w2) in zip(index,weights) if self.ok[0,j,k] and j != k )
                return itertools.chain(ret1, ret2)
            else:
                # Like for 2pt we want to avoid getting extra copies of what are actually
                # auto-correlations coming from two indices equalling each other in (i,j,k).
                # This time, get each (i,i,i) once.
                # Then get (i,i,j), (i,j,i), and (j,i,i) once per each (i,j) pair with i!=j
                # repeated as often as they show up in the double for loop.
                # Finally get all triples (i,j,k) where they are all different repeated as often
                # as they show up in the triple for loop.
                assert self.npatch1 == self.npatch2 == self.npatch3
                index, weights = np.unique(self.index, return_counts=True)
                ret1 = ( (i,i,i,w) for (i,w) in zip(index, weights) if self.ok[i,i,i] )
                if self.cpw == 'simple':
                    ret2 = ( (i,j,k, (w1*w2 if (i==k or j==k) else w1*w3 if i==j else w1*w2*w3))
                                for (i,w1) in zip(index, weights)
                                for (j,w2) in zip(index,weights)
                                for (k,w3) in zip(index,weights)
                                if self.ok[i,j,k] and not i==j==k )
                elif self.cpw == 'mean':
                    wdict = dict(zip(index, weights))
                    ret2 = ( (i,j,k,(wdict.get(i,0)+wdict.get(j,0)+wdict.get(k,0))/3)
                            for i in range(self.npatch1) for j in range(self.npatch2)
                            for k in range(self.npatch3) if self.ok[i,j,k] and not i==j==k )
                else:
                    assert self.cpw == 'geom'
                    ret2 = ( (i,j,k,(w1*w2*w3)**(1./3.)) for (i,w1) in zip(index, weights)
                                for (j,w2) in zip(index,weights)
                                for (k,w3) in zip(index,weights)
                                if self.ok[i,j,k] and not i==j==k )
                return itertools.chain(ret1, ret2)

    def _bootstrap_pairs(self, index, cross_patch_weight):
        return self.BootstrapTripleIterator(self.results, self.npatch1, self.npatch2,
                                            self.npatch3, index, cross_patch_weight, self._ok)

    def _check_cpw(self, method, cross_patch_weight):
        if cross_patch_weight is None:
            cross_patch_weight = self.cross_patch_weight
        if cross_patch_weight is None:
            cross_patch_weight = 'simple'
            if method == 'jackknife':
                self.logger.warning(
                    "Using the default cross_patch_weight='simple' may be less accurate than "
                    "using cross_patch_weight='match'.  See the docs for details about this "
                    "option.  It may become the new default value in a future version.\n"
                    "Set cross_patch_weight='simple' explicitly to suppress this message.")
            elif method == 'bootstrap':
                self.logger.warning(
                    "Using the default cross_patch_weight='simple' may be less accurate than "
                    "using cross_patch_weight='geom'.  See the docs for details about this "
                    "option.  It may become the new default value in a future version.\n"
                    "Set cross_patch_weight='simple' explicitly to suppress this message.")
        return cross_patch_weight

    @property
    def _write_params(self):
        params = make_minimal_config(self.config, Corr3._valid_params)
        # Add in a couple other things we want to preserve that aren't construction kwargs.
        params['coords'] = self.coords
        params['metric'] = self.metric
        params['corr'] = self._letters
        return params

    @property
    def _write_col_names(self):
        if self.bin_type == 'LogRUV':
            col_names = ['r_nom', 'u_nom', 'v_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanu', 'meanv']
        elif self.bin_type == 'LogSAS':
            col_names = ['d2_nom', 'd3_nom', 'phi_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanphi']
        else:
            col_names = ['d2_nom', 'd3_nom', 'n',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3']
        col_names += self._write_class_col_names
        if self.bin_type == 'LogMultipole':
            col_names += ['weightr', 'weighti', 'ntri']
        else:
            col_names += ['weight', 'ntri']
        return col_names

    @property
    def _write_data(self):
        if self.bin_type == 'LogRUV':
            data = [ self.rnom, self.u, self.v,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3, self.meanu, self.meanv ]
        elif self.bin_type == 'LogSAS':
            data = [ self.d2nom, self.d3nom, self.phi,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3, self.meanphi ]
        else:
            data = [ self.d2nom, self.d3nom, self.n,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3 ]
        data += self._write_class_data
        if self.bin_type == 'LogMultipole':
            data += [ self.weightr, self.weighti, self.ntri ]
        else:
            data += [ self.weight, self.ntri ]
        data = [ col.flatten() for col in data ]
        return data

    def _write(self, writer, name, write_patch_results, write_cov=False, zero_tot=False):
        if name is None and (write_patch_results or write_cov):
            name = 'main'
        # These helper properties define what to write for each class.
        col_names = self._write_col_names
        data = self._write_data
        params = self._write_params

        if write_patch_results:
            # Note: Only include npatch1, npatch2 in serialization if we are also serializing
            # results.  Otherwise, the corr that is read in will behave oddly.
            params['npatch1'] = self.npatch1
            params['npatch2'] = self.npatch2
            params['npatch3'] = self.npatch3
            params['num_rows'] = self._nbins
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
        if write_cov:
            params['cov_shape'] = self.cov.shape

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
        if write_cov:
            writer.write_array(self.cov, ext='cov')

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        r"""Write the correlation function to the file, file_name.

        For bin_type = LogRUV, the output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        r_nom           The nominal center of the bin in r = d2 where d1 > d2 > d3
        u_nom           The nominal center of the bin in u = d3/d2
        v_nom           The nominal center of the bin in v = +-(d1-d2)/d3
        meanu           The mean value :math:`\langle u\rangle` of triangles that fell
                        into each bin
        meanv           The mean value :math:`\langle v\rangle` of triangles that fell
                        into each bin
        ==========      ================================================================

        For bin_type = LogSAS, the output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        d2_nom          The nominal center of the bin in d2
        d3_nom          The nominal center of the bin in d3
        phi_nom         The nominal center of the bin in phi, the opening angle between
                        d2 and d3 in the counter-clockwise direction
        meanphi         The mean value :math:`\langle phi\rangle` of triangles that fell
                        into each bin
        ==========      ================================================================

        For bin_type = LogMultipole, the output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        d2_nom          The nominal center of the bin in d2
        d3_nom          The nominal center of the bin in d3
        n               The multipole index n
        ==========      ================================================================

        In addition, all bin types include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        meand1          The mean value :math:`\langle d1\rangle` of triangles that fell
                        into each bin
        meanlogd1       The mean value :math:`\langle \log(d1)\rangle` of triangles that
                        fell into each bin
        meand2          The mean value :math:`\langle d2\rangle` of triangles that fell
                        into each bin
        meanlogd2       The mean value :math:`\langle \log(d2)\rangle` of triangles that
                        fell into each bin
        meand3          The mean value :math:`\langle d3\rangle` of triangles that fell
                        into each bin
        meanlogd3       The mean value :math:`\langle \log(d3)\rangle` of triangles that
                        fell into each bin {}
        weight          The total weight of triangles contributing to each bin.
                        (For LogMultipole, this is split into real and imaginary parts,
                        weightr and weighti.)
        ntri            The number of triangles contributing to each bin
        ==========      ================================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        self.logger.info(f'Writing {self._letters} correlations to {file_name}')
        precision = self.config.get('precision', 4) if precision is None else precision
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov)

    def _read(self, reader, name=None, params=None):
        name = 'main' if 'main' in reader and name is None else name
        if params is None:
            params = reader.read_params(ext=name)
        num_rows = params.get('num_rows', None)
        num_patch_tri = params.get('num_patch_tri', 0)
        num_zero_patch = params.get('num_zero_patch', 0)
        cov_shape = params.get('cov_shape', None)
        name = 'main' if num_patch_tri and name is None else name
        data = reader.read_data(max_rows=num_rows, ext=name)

        # Version 5.0 used weight_re, weight_im.  These are now weightr, weighti.
        # Fix in place to keep backwards compatibility.
        if 'weight_re' in data.dtype.names:
            dt = np.dtype([(n.replace('_re','r').replace('_im','i'),t) for (n,t) in data.dtype.descr])
            data = data.astype(dt)

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
        if cov_shape is not None:
            if isinstance(cov_shape, str):
                cov_shape = eval(cov_shape)
            self._cov = reader.read_array(cov_shape, ext='cov')

    def _read_from_data(self, data, params):
        s = self.data_shape
        self.meand1 = data['meand1'].reshape(s)
        self.meanlogd1 = data['meanlogd1'].reshape(s)
        self.meand2 = data['meand2'].reshape(s)
        self.meanlogd2 = data['meanlogd2'].reshape(s)
        self.meand3 = data['meand3'].reshape(s)
        self.meanlogd3 = data['meanlogd3'].reshape(s)
        if self.bin_type == 'LogRUV':
            self.meanu = data['meanu'].reshape(s)
            self.meanv = data['meanv'].reshape(s)
        elif self.bin_type == 'LogSAS':
            self.meanu = data['meanphi'].reshape(s)
        if self.bin_type == 'LogMultipole':
            self.weightr = data['weightr'].reshape(s)
            self.weighti = data['weighti'].reshape(s)
        else:
            if 'weight' in data.dtype.names:
                # NNN calls this DDD, rather than weight.  Let that class handle it.
                # But here, don't error if weight is missing.
                self.weightr = data['weight'].reshape(s)
        self.ntri = data['ntri'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)
        self.npatch3 = params.get('npatch3', 1)

    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS or HDF5 file, so
        there is no loss of information.

        .. warning::

            The current object should be constructed with the same configuration parameters as
            the one being read.  e.g. the same min_sep, max_sep, etc.  This is not checked by
            the read function.  For most use cases, you should prefer `from_file`, which will
            automatically construct the object with the correct configuration parameters given
            the information in the file.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info(f'Reading {self._letters} correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

    @classmethod
    def from_file(cls, file_name, *, file_type=None, logger=None, rng=None):
        """Create a new instance from an output file.

        This should be a file that was written by TreeCorr.

        .. note::

            This classmethod may be called either using the base class or the class type that
            wrote the file.  E.g. if the file was written by `GGGCorrelation`, then either
            of the following would work and be equivalent:

                >>> ggg = treecorr.GGGCorrelation.from_file(file_name)
                >>> ggg = treecorr.Corr3.from_file(file_name)

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII', 'FITS', or 'HDF').  (default: determine
                                the type automatically from the extension of file_name.)
            logger (Logger):    If desired, a logger object to use for logging. (default: None)
            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                                random number generation. (default: None)

        Returns:
            A Correlation object, constructed from the information in the file.
        """
        if cls is Corr3:
            # Then need to figure out what class to make first.
            with make_reader(file_name, file_type, logger) as reader:
                name = 'main' if 'main' in reader else None
                params = reader.read_params(ext=name)
                letters = params.get('corr', None)
                if letters not in Corr3._lookup_dict:
                    raise OSError("%s does not seem to be a valid treecorr output file."%file_name)
                cls = Corr3._lookup_dict[letters]
                return cls.from_file(file_name, file_type=file_type, logger=logger, rng=rng)
        if logger:
            logger.info(f'Building {cls._cls} from %s',file_name)
        with make_reader(file_name, file_type, logger) as reader:
            name = 'main' if 'main' in reader else None
            params = reader.read_params(ext=name)
            letters = params.get('corr', None)
            if letters not in Corr3._lookup_dict:
                raise OSError("%s does not seem to be a valid treecorr output file."%file_name)
            if params['corr'] != cls._letters:
                raise OSError("Trying to read a %sCorrelation output file with %s"%(
                              params['corr'], cls.__name__))
            kwargs = make_minimal_config(params, Corr3._valid_params)
            corr = cls(**kwargs, logger=logger, rng=rng)
            corr.logger.info(f'Reading {cls._letters} correlations from %s', file_name)
            corr._read(reader, name=name, params=params)
        return corr

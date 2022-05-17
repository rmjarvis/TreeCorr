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
.. module:: nnncorrelation
"""

import numpy as np

from . import _lib, _ffi
from .binnedcorr3 import BinnedCorr3
from .util import double_ptr as dp
from .util import make_writer, make_reader, lazy_property
from .util import depr_pos_kwargs


class NNNCorrelation(BinnedCorr3):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    See the doc string of `BinnedCorr3` for a description of how the triangles are binned.

    Ojects of this class holds the following attributes:

    Attributes:
        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        nbins:      The number of bins in logr where r = d2
        bin_size:   The size of the bins in logr
        min_sep:    The minimum separation being considered
        max_sep:    The maximum separation being considered
        nubins:     The number of bins in u where u = d3/d2
        ubin_size:  The size of the bins in u
        min_u:      The minimum u being considered
        max_u:      The maximum u being considered
        nvbins:     The number of bins in v where v = +-(d1-d2)/d3
        vbin_size:  The size of the bins in v
        min_v:      The minimum v being considered
        max_v:      The maximum v being considered
        logr1d:     The nominal centers of the nbins bins in log(r).
        u1d:        The nominal centers of the nubins bins in u.
        v1d:        The nominal centers of the nvbins bins in v.

    In addition, the following attributes are numpy arrays whose shape is (nbins, nubins, nvbins):

    Attributes:
        logr:       The nominal center of the bin in log(r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        u:          The nominal center of the bin in u.
        v:          The nominal center of the bin in v.
        meand1:     The (weighted) mean value of d1 for the triangles in each bin.
        meanlogd1:  The mean value of log(d1) for the triangles in each bin.
        meand2:     The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        meanlogd2:  The mean value of log(d2) for the triangles in each bin.
        meand2:     The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd2:  The mean value of log(d3) for the triangles in each bin.
        meanu:      The mean value of u for the triangles in each bin.
        meanv:      The mean value of v for the triangles in each bin.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).
        tot:        The total number of triangles processed, which is used to normalize
                    the randoms if they have a different number of triangles.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_auto` and/or
        `process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
        the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> nnn = treecorr.NNNCorrelation(config)
        >>> nnn.process(cat)         # For auto-correlation.
        >>> rrr.process(rand)        # Likewise for random-random correlations
        >>> drr.process(cat,rand)    # If desired, also do data-random correlations
        >>> rdd.process(rand,cat)    # Also with two data and one random
        >>> nnn.write(file_name,rrr=rrr,drr=drr,...)  # Write out to a file.
        >>> zeta,varzeta = nnn.calculateZeta(rrr=rrr,drr=drr,rdd=rdd)  # Or get zeta directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `BinnedCorr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `BinnedCorr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    @depr_pos_kwargs
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NNNCorrelation`.  See class doc for details.
        """
        BinnedCorr3.__init__(self, config, logger=logger, **kwargs)

        self._ro._d1 = 1  # NData
        self._ro._d2 = 1  # NData
        self._ro._d3 = 1  # NData
        shape = self.logr.shape
        self.meand1 = np.zeros(shape, dtype=float)
        self.meanlogd1 = np.zeros(shape, dtype=float)
        self.meand2 = np.zeros(shape, dtype=float)
        self.meanlogd2 = np.zeros(shape, dtype=float)
        self.meand3 = np.zeros(shape, dtype=float)
        self.meanlogd3 = np.zeros(shape, dtype=float)
        self.meanu = np.zeros(shape, dtype=float)
        self.meanv = np.zeros(shape, dtype=float)
        self.weight = np.zeros(shape, dtype=float)
        self.ntri = np.zeros(shape, dtype=float)
        self.tot = 0.
        self._rrr_weight = None
        self._rrr = None
        self._drr = None
        self._rdd = None
        self._write_rrr = None
        self._write_drr = None
        self._write_rdd = None
        self.logger.debug('Finished building NNNCorr')

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _lib.BuildCorr3(
                    self._d1, self._d2, self._d3, self._bintype,
                    self._min_sep,self._max_sep,self.nbins,self._bin_size,self.b,
                    self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                    self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                    self.xperiod, self.yperiod, self.zperiod,
                    dp(None), dp(None), dp(None), dp(None),
                    dp(None), dp(None), dp(None), dp(None),
                    dp(self.meand1), dp(self.meanlogd1), dp(self.meand2), dp(self.meanlogd2),
                    dp(self.meand3), dp(self.meanlogd3), dp(self.meanu), dp(self.meanv),
                    dp(self.weight), dp(self.ntri))
        return self._corr

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if self._corr is not None:
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyCorr3(self.corr, self._d1, self._d2, self._d3, self._bintype)

    def __eq__(self, other):
        """Return whether two `NNNCorrelation` instances are equal"""
        return (isinstance(other, NNNCorrelation) and
                self.nbins == other.nbins and
                self.bin_size == other.bin_size and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.sep_units == other.sep_units and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nubins == other.nubins and
                self.ubin_size == other.ubin_size and
                self.min_v == other.min_v and
                self.max_v == other.max_v and
                self.nvbins == other.nvbins and
                self.vbin_size == other.vbin_size and
                self.coords == other.coords and
                self.bin_type == other.bin_type and
                self.bin_slop == other.bin_slop and
                self.xperiod == other.xperiod and
                self.yperiod == other.yperiod and
                self.zperiod == other.zperiod and
                self.tot == other.tot and
                np.array_equal(self.meand1, other.meand1) and
                np.array_equal(self.meanlogd1, other.meanlogd1) and
                np.array_equal(self.meand2, other.meand2) and
                np.array_equal(self.meanlogd2, other.meanlogd2) and
                np.array_equal(self.meand3, other.meand3) and
                np.array_equal(self.meanlogd3, other.meanlogd3) and
                np.array_equal(self.meanu, other.meanu) and
                np.array_equal(self.meanv, other.meanv) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.ntri, other.ntri))

    def copy(self):
        """Make a copy"""
        ret = NNNCorrelation.__new__(NNNCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to by deep copied.
                ret.__dict__[key] = item.copy()
            else:
                # For everything else, shallow copy is fine.
                # In particular don't deep copy config or logger
                # Most of the rest are scalars, which copy fine this way.
                # And the read-only things are all in _ro.
                # The results dict is trickier.  We rely on it being copied in places, but we
                # never add more to it after the copy, so shallow copy is fine.
                ret.__dict__[key] = item
        ret._corr = None # We'll want to make a new one of these if we need it.
        if self._drr is not None:
            ret._drr = self._drr.copy()
        if self._rdd is not None:
            ret._rdd = self._rdd.copy()
        if self._rrr is not None:
            ret._rrr = self._rrr.copy()
        return ret

    @lazy_property
    def _zero_array(self):
        # An array of all zeros with the same shape as self.weight (and other data arrays)
        z = np.zeros_like(self.weight)
        z.flags.writeable=False  # Just to make sure we get an error if we try to change it.
        return z

    def _zero_copy(self, tot):
        # A minimal "copy" with zero for the weight array, and the given value for tot.
        ret = NNNCorrelation.__new__(NNNCorrelation)
        ret._ro = self._ro
        ret.coords = self.coords
        ret.metric = self.metric
        ret.config = self.config
        ret.meand1 = self._zero_array
        ret.meanlogd1 = self._zero_array
        ret.meand2 = self._zero_array
        ret.meanlogd2 = self._zero_array
        ret.meand3 = self._zero_array
        ret.meanlogd3 = self._zero_array
        ret.meanu = self._zero_array
        ret.meanv = self._zero_array
        ret.weight = self._zero_array
        ret.ntri = self._zero_array
        ret.tot = tot
        ret._corr = None
        ret._rrr = ret._drr = ret._rdd = None
        ret._write_rrr = ret._write_drr = ret._write_rdd = None
        # This override is really the main advantage of using this:
        setattr(ret, '_nonzero', False)
        return ret

    def __repr__(self):
        return 'NNNCorrelation(config=%r)'%self.config

    @depr_pos_kwargs
    def process_auto(self, cat, *, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        Parameters:
            cat (Catalog):      The catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat.name == '':
            self.logger.info('Starting process NNN auto-correlations')
        else:
            self.logger.info('Starting process NNN auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getNField(min_size=min_size, max_size=max_size,
                              split_method=self.split_method, brute=bool(self.brute),
                              min_top=self.min_top, max_top=self.max_top,
                              coords=self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        _lib.ProcessAuto3(self.corr, field.data, self.output_dots,
                          field._d, self._coords, self._bintype, self._metric)
        self.tot += (1./6.) * cat.sumw**3

    @depr_pos_kwargs
    def process_cross12(self, cat1, cat2, *, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto-correlation calculation.  E.g. when splitting up a large catalog into patches,
        this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        Parameters:
            cat1 (Catalog):     The first catalog to process. (1 point in each triangle will come
                                from this catalog.)
            cat2 (Catalog):     The second catalog to process. (2 points in each triangle will come
                                from this catalog.)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NNN (1-2) cross-correlations')
        else:
            self.logger.info('Starting process NNN (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        _lib.ProcessCross12(self.corr, self.corr, self.corr,
                            f1.data, f2.data, self.output_dots,
                            f1._d, f2._d, self._coords,
                            self._bintype, self._metric)
        self.tot += cat1.sumw * cat2.sumw**2 / 2.

    @depr_pos_kwargs
    def process_cross(self, cat1, cat2, cat3, *, metric=None, num_threads=None):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto-correlation calculation.  E.g. when splitting up a large catalog into patches,
        this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            cat3 (Catalog):     The third catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process NNN cross-correlations')
        else:
            self.logger.info('Starting process NNN cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f3 = cat3.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 3,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 6 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        _lib.ProcessCross3(self.corr, self.corr, self.corr,
                           self.corr, self.corr, self.corr,
                           f1.data, f2.data, f3.data, self.output_dots,
                           f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)
        self.tot += cat1.sumw * cat2.sumw * cat3.sumw

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.meand1[mask1] /= self.weight[mask1]
        self.meanlogd1[mask1] /= self.weight[mask1]
        self.meand2[mask1] /= self.weight[mask1]
        self.meanlogd2[mask1] /= self.weight[mask1]
        self.meand3[mask1] /= self.weight[mask1]
        self.meanlogd3[mask1] /= self.weight[mask1]
        self.meanu[mask1] /= self.weight[mask1]
        self.meanv[mask1] /= self.weight[mask1]

        # Update the units
        self._apply_units(mask1)

        # Use meanlogr when available, but set to nominal when no triangles in bin.
        self.meand2[mask2] = self.rnom[mask2]
        self.meanlogd2[mask2] = self.logr[mask2]
        self.meanu[mask2] = self.u[mask2]
        self.meanv[mask2] = self.v[mask2]
        self.meand3[mask2] = self.u[mask2] * self.meand2[mask2]
        self.meanlogd3[mask2] = np.log(self.meand3[mask2])
        self.meand1[mask2] = self.v[mask2] * self.meand3[mask2] + self.meand2[mask2]
        self.meanlogd1[mask2] = np.log(self.meand1[mask2])

    def finalize(self):
        """Finalize the calculation of meand1, meanlogd1, etc.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanlogr, meanu, meanv by dividing by the total weight.
        """
        self._finalize()

    @lazy_property
    def _nonzero(self):
        # The lazy version when we can be sure the object isn't going to accumulate any more.
        return self.nonzero

    def _clear(self):
        """Clear the data vectors
        """
        self.meand1[:,:,:] = 0.
        self.meanlogd1[:,:,:] = 0.
        self.meand2[:,:,:] = 0.
        self.meanlogd2[:,:,:] = 0.
        self.meand3[:,:,:] = 0.
        self.meanlogd3[:,:,:] = 0.
        self.meanu[:,:,:] = 0.
        self.meanv[:,:,:] = 0.
        self.weight[:,:,:] = 0.
        self.ntri[:,:,:] = 0.
        self.tot = 0.

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        tot = np.sum([c.tot for c in others])
        # Empty ones were only needed for tot.  Remove them now.
        others = [c for c in others if c._nonzero]
        if len(others) == 0:
            self._clear()
        else:
            np.sum([c.meand1 for c in others], axis=0, out=self.meand1)
            np.sum([c.meanlogd1 for c in others], axis=0, out=self.meanlogd1)
            np.sum([c.meand2 for c in others], axis=0, out=self.meand2)
            np.sum([c.meanlogd2 for c in others], axis=0, out=self.meanlogd2)
            np.sum([c.meand3 for c in others], axis=0, out=self.meand3)
            np.sum([c.meanlogd3 for c in others], axis=0, out=self.meanlogd3)
            np.sum([c.meanu for c in others], axis=0, out=self.meanu)
            np.sum([c.meanv for c in others], axis=0, out=self.meanv)
            np.sum([c.weight for c in others], axis=0, out=self.weight)
            np.sum([c.ntri for c in others], axis=0, out=self.ntri)
        self.tot = tot

    def _add_tot(self, i, j, k, c1, c2, c3):
        # When storing results from a patch-based run, tot needs to be accumulated even if
        # the total weight being accumulated comes out to be zero.
        # This only applies to NNNCorrelation.  For the other ones, this is a no op.
        tot = c1.sumw * c2.sumw * c3.sumw
        if c2 is c3:
            # Account for 1/2 factor in cross12 cases.
            tot /= 2.
        self.tot += tot
        # We also have to keep all pairs in the results dict, otherwise the tot calculation
        # gets messed up.  We need to accumulate the tot value of all pairs, even if
        # the resulting weight is zero.
        self.results[(i,j,k)] = self._zero_copy(tot)

    def __iadd__(self, other):
        """Add a second `NNNCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `NNNCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, NNNCorrelation):
            raise TypeError("Can only add another NNNCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.nubins == other.nubins and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nvbins == other.nvbins and
                self.min_v == other.min_v and
                self.max_v == other.max_v):
            raise ValueError("NNNCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords, other.coords, other.coords)
        self.tot += other.tot

        # If other is empty, then we're done now.
        if not other.nonzero:
            return self

        self.meand1[:] += other.meand1[:]
        self.meanlogd1[:] += other.meanlogd1[:]
        self.meand2[:] += other.meand2[:]
        self.meanlogd2[:] += other.meanlogd2[:]
        self.meand3[:] += other.meand3[:]
        self.meanlogd3[:] += other.meanlogd3[:]
        self.meanu[:] += other.meanu[:]
        self.meanv[:] += other.meanv[:]
        self.weight[:] += other.weight[:]
        self.ntri[:] += other.ntri[:]
        return self

    @depr_pos_kwargs
    def process(self, cat1, cat2=None, cat3=None, *, metric=None, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True):
        """Accumulate the 3pt correlation of the points in the given Catalog(s).

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation.

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        .. note::

            For a correlation of multiple catalogs, it typically matters which corner of the
            triangle comes from which catalog, which is not kept track of by this function.
            The final accumulation will have d1 > d2 > d3 regardless of which input catalog
            appears at each corner.  The class which keeps track of which catalog appears
            in each position in the triangle is `NNNCrossCorrelation`.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first N field.
            cat2 (Catalog):     A catalog or list of catalogs for the second N field.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third N field.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            comm (mpi4py.Comm): If running MPI, an mpi4py Comm object to communicate between
                                processes.  If used, the rank=0 process will have the final
                                computation. This only works if using patches. (default: None)
            low_mem (bool):     Whether to sacrifice a little speed to try to reduce memory usage.
                                This only works if using patches. (default: False)
            initialize (bool):  Whether to begin the calculation with a call to
                                `BinnedCorr3.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
        """
        if initialize:
            self.clear()

        if not isinstance(cat1,list): cat1 = cat1.get_patches()
        if cat2 is not None and not isinstance(cat2,list): cat2 = cat2.get_patches()
        if cat3 is not None and not isinstance(cat3,list): cat3 = cat3.get_patches()

        if cat2 is None:
            if cat3 is not None:
                raise ValueError("For two catalog case, use cat1,cat2, not cat1,cat3")
            self._process_all_auto(cat1, metric, num_threads)
        elif cat3 is None:
            self._process_all_cross12(cat1, cat2, metric, num_threads, comm, low_mem)
        else:
            self._process_all_cross(cat1, cat2, cat3, metric, num_threads, comm, low_mem)

        if finalize:
            self.finalize()

    def _mean_weight(self):
        mean_np = np.mean(self.ntri)
        return 1 if mean_np == 0 else np.mean(self.weight)/mean_np

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        This raises a RuntimeError if calculateZeta has not been run yet.
        """
        if self._rrr_weight is None:
            raise RuntimeError("You need to call calculateZeta before calling estimate_cov.")
        return self.zeta.ravel()

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        This is the weight array corresponding to `getStat`.  In this case, it is the denominator
        RRR from the calculation done by calculateZeta().
        """
        if self._rrr_weight is not None:
            return self._rrr_weight.ravel()
        else:
            return self.tot

    @depr_pos_kwargs
    def calculateZeta(self, *, rrr, drr=None, rdd=None):
        r"""Calculate the 3pt function given another 3pt function of random
        points using the same mask, and possibly cross correlations of the data and random.

        There are two possible formulae that are currently supported.

        1. The simplest formula to use is :math:`\zeta^\prime = (DDD-RRR)/RRR`.
           In this case, only rrr needs to be given, the `NNNCorrelation` of a random field.
           However, note that in this case, the return value is not normally called :math:`\zeta`.
           Rather, this is an estimator of

           .. math::

               \zeta^\prime(d_1,d_2,d_3) = \zeta(d_1,d_2,d_3) + \xi(d_1) + \xi(d_2) + \xi(d_3)

           where :math:`\xi` is the two-point correlation function for each leg of the triangle.
           You would typically want to calculate that separately and subtract off the
           two-point contributions.

        2. For auto-correlations, a better formula is :math:`\zeta = (DDD-RDD+DRR-RRR)/RRR`.
           In this case, RDD is the number of triangles where 1 point comes from the randoms
           and 2 points are from the data. Similarly, DRR has 1 point from the data and 2 from
           the randoms.  These are what are calculated from calling::

                >>> drr.process(data_cat, rand_cat)
                >>> rdd.process(rand_cat, data_cat)

           .. note::

                One might thing the formula should be :math:`\zeta = (DDD-3RDD+3DRR-RRR)/RRR`
                by analogy with the 2pt Landy-Szalay formula. However, the way these are
                calculated, the object we are calling RDD already includes triangles where R
                is in each of the 3 locations.  So it is really more like RDD + DRD + DDR.
                These are not computed separately.  Rather the single computation of ``rdd``
                described above accumulates all three permutations together.  So that one
                object includes everything for the second term.  Likewise ``drr`` has all the
                permutations that are relevant for the third term.

        - If only rrr is provided, the first formula will be used.
        - If all of rrr, drr, rdd are provided then the second will be used.

        Parameters:
            rrr (NNNCorrelation):   The auto-correlation of the random field (RRR)
            drr (NNNCorrelation):   DRR if desired. (default: None)
            rdd (NNNCorrelation):   RDD if desired. (default: None)

        Returns:
            Tuple containing

                - zeta = array of :math:`\zeta(d_1,d_2,d_3)`
                - varzeta = array of variance estimates of :math:`\zeta(d_1,d_2,d_3)`
        """
        # Each random ntri value needs to be rescaled by the ratio of total possible tri.
        if rrr.tot == 0:
            raise ValueError("rrr has tot=0.")

        if (rdd is not None) != (drr is not None):
            raise TypeError("Must provide both rdd and drr (or neither).")

        # rrrf is the factor to scale rrr weights to get something commensurate to the ddd density.
        rrrf = self.tot / rrr.tot

        # Likewise for the other two potential randoms:
        if drr is not None:
            if drr.tot == 0:
                raise ValueError("drr has tot=0.")
            drrf = self.tot / drr.tot
        if rdd is not None:
            if rdd.tot == 0:
                raise ValueError("rdd has tot=0.")
            rddf = self.tot / rdd.tot

        # Calculate zeta based on which randoms are provided.
        denom = rrr.weight * rrrf
        if rdd is None:
            self.zeta = self.weight - denom
        else:
            self.zeta = self.weight - rdd.weight * rddf + drr.weight * drrf - denom

        # Divide by DRR in all cases.
        if np.any(rrr.weight == 0):
            self.logger.warning("Warning: Some bins for the randoms had no triangles.")
            denom[rrr.weight==0] = 1.  # guard against division by 0.
        self.zeta /= denom

        # Set up necessary info for estimate_cov

        # First the bits needed for shot noise covariance:
        dddw = self._mean_weight()
        rrrw = rrr._mean_weight()
        if drr is not None:
            drrw = drr._mean_weight()
        if rdd is not None:
            rddw = rdd._mean_weight()

        # Note: The use of varzeta_factor for the shot noise varzeta is even less justified
        #       than in the NN varxi case.  This is merely motivated by analogy with the
        #       2pt version.
        if rdd is None:
            varzeta_factor = 1 + rrrf*rrrw/dddw
        else:
            varzeta_factor = 1 + drrf*drrw/dddw + rddf*rddw/dddw + rrrf*rrrw/dddw
        self._var_num = dddw * varzeta_factor**2  # Should this be **3? Hmm...
        self._rrr_weight = rrr.weight * rrrf

        # Now set up the bits needed for patch-based covariance
        self._rrr = rrr
        self._drr = drr
        self._rdd = rdd

        if len(self.results) > 0:
            # Check that all use the same patches as ddd
            if rrr.npatch1 != 1:
                if rrr.npatch1 != self.npatch1:
                    raise RuntimeError("If using patches, RRR must be run with the same patches "
                                       "as DDD")
            if drr is not None and (len(drr.results) == 0 or drr.npatch1 != self.npatch1
                                    or drr.npatch2 not in (self.npatch2, 1)):
                raise RuntimeError("DRR must be run with the same patches as DDD")
            if rdd is not None and (len(rdd.results) == 0 or rdd.npatch2 != self.npatch2
                                    or rdd.npatch1 not in (self.npatch1, 1)):
                raise RuntimeError("RDD must be run with the same patches as DDD")

            # If there are any rrr,drr,rdd patch sets that aren't in results, then we need to add
            # some dummy results to make sure all the right ijk "pair"s are computed when we make
            # the vectors for the covariance matrix.
            add_ijk = set()
            if rrr.npatch1 != 1:
                for ijk in rrr.results:
                    if ijk not in self.results:
                        add_ijk.add(ijk)

            if drr is not None and drr.npatch2 != 1:
                for ijk in drr.results:
                    if ijk not in self.results:
                        add_ijk.add(ijk)

            if rdd is not None and rdd.npatch1 != 1:
                for ijk in rdd.results:
                    if ijk not in self.results:
                        add_ijk.add(ijk)

            if len(add_ijk) > 0:
                for ijk in add_ijk:
                    self.results[ijk] = self._zero_copy(0)
                self.__dict__.pop('_ok',None)  # If it was already made, it will need to be redone.

        # Now that it's all set up, calculate the covariance and set varzeta to the diagonal.
        self.cov = self.estimate_cov(self.var_method)
        self.varzeta = self.cov.diagonal().reshape(self.zeta.shape)
        return self.zeta, self.varzeta

    def _calculate_xi_from_pairs(self, pairs):
        # Note: we keep the notation ij and pairs here, even though they are really ijk and
        # triples.
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()
        if self._rrr is None:
            return
        ddd = self.weight
        if len(self._rrr.results) > 0:
            # This is the usual case.  R has patches just like D.
            # Calculate rrr and rrrf in the normal way based on the same pairs as used for DDD.
            pairs1 = [ij for ij in pairs if self._rrr._ok[ij[0],ij[1],ij[2]]]
            self._rrr._sum([self._rrr.results[ij] for ij in pairs1])
            ddd_tot = self.tot
        else:
            # In this case, R was not run with patches.
            # We need to scale RRR down by the relative area.
            # The approximation we'll use is that tot in the auto-correlations is
            # proportional to area**3.
            # The sum of tot**(1/3) when i=j=k gives an estimate of the fraction of the total area.
            area_frac = np.sum([self.results[ij].tot**(1./3.) for ij in pairs
                                if ij[0] == ij[1] == ij[2]])
            area_frac /= np.sum([cij.tot**(1./3.) for ij,cij in self.results.items()
                                 if ij[0] == ij[1] == ij[2]])
            # First figure out the original total for all DDD that had the same footprint as RRR.
            ddd_tot = np.sum([self.results[ij].tot for ij in self.results])
            # The rrrf we want will be a factor of area_frac smaller than the original
            # ddd_tot/rrr_tot.  We can effect this by multiplying the full ddd_tot by area_frac
            # and use that value normally below.  (Also for drrf and rddf.)
            ddd_tot *= area_frac

        rrr = self._rrr.weight
        rrrf = ddd_tot / self._rrr.tot

        if self._drr is not None:
            if self._drr.npatch2 == 1:
                # If r doesn't have patches, then convert all (i,i,i) pairs to (i,0,0).
                pairs2 = [(ij[0],0,0) for ij in pairs if ij[0] == ij[1] == ij[2]]
            else:
                pairs2 = [ij for ij in pairs if self._drr._ok[ij[0],ij[1],ij[2]]]
            self._drr._sum([self._drr.results[ij] for ij in pairs2])
            drr = self._drr.weight
            drrf = ddd_tot / self._drr.tot
        if self._rdd is not None:
            if self._rdd.npatch1 == 1:
                # If r doesn't have patches, then convert all (i,i,j) pairs to (0,i,j)
                # and all (i,j,i to (0,j,i).
                pairs3 = [(0,ij[1],ij[2]) for ij in pairs if ij[0] == ij[1] or ij[0] == ij[2]]
            else:
                pairs3 = [ij for ij in pairs if self._rdd._ok[ij[0],ij[1],ij[2]]]
            self._rdd._sum([self._rdd.results[ij] for ij in pairs3])
            rdd = self._rdd.weight
            rddf = ddd_tot / self._rdd.tot
        denom = rrr * rrrf
        if self._drr is None:
            zeta = ddd - denom
        else:
            zeta = ddd - rdd * rddf + drr * drrf - denom
        denom[denom == 0] = 1  # Guard against division by zero.
        self.zeta = zeta / denom
        self._rrr_weight = denom

    @depr_pos_kwargs
    def write(self, file_name, *, rrr=None, drr=None, rdd=None, file_type=None, precision=None,
              write_patch_results=False):
        r"""Write the correlation function to the file, file_name.

        Normally, at least rrr should be provided, but if this is None, then only the
        basic accumulated number of triangles are output (along with the columns parametrizing
        the size and shape of the triangles).

        If at least rrr is given, then it will output an estimate of the final 3pt correlation
        function, :math:`\zeta`. There are two possible formulae that are currently supported.

        1. The simplest formula to use is :math:`\zeta^\prime = (DDD-RRR)/RRR`.
           In this case, only rrr needs to be given, the `NNNCorrelation` of a random field.
           However, note that in this case, the return value is not what is normally called
           :math:`\zeta`.  Rather, this is an estimator of

           .. math::
               \zeta^\prime(d_1,d_2,d_3) = \zeta(d_1,d_2,d_3) + \xi(d_1) + \xi(d_2) + \xi(d_3)

           where :math:`\xi` is the two-point correlation function for each leg of the triangle.
           You would typically want to calculate that separately and subtract off the
           two-point contributions.

        2. For auto-correlations, a better formula is :math:`\zeta = (DDD-RDD+DRR-RRR)/RRR`.
           In this case, RDD is the number of triangles where 1 point comes from the randoms
           and 2 points are from the data. Similarly, DRR has 1 point from the data and 2 from
           the randoms.
           For this case, all combinations rrr, drr, and rdd must be provided.

        The output file will include the following columns:

        ==========      ================================================================
        Column          Description
        ==========      ================================================================
        r_nom           The nominal center of the bin in r = d2 where d1 > d2 > d3
        u_nom           The nominal center of the bin in u = d3/d2
        v_nom           The nominal center of the bin in v = +-(d1-d2)/d3
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
                        fell into each bin
        meanu           The mean value :math:`\langle u\rangle` of triangles that fell
                        into each bin
        meanv           The mean value :math:`\langle v\rangle` of triangles that fell
                        into each bin
        zeta            The estimator :math:`\zeta(r,u,v)` (if rrr is given)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
                        (if rrr is given)
        DDD             The total weight of DDD triangles in each bin
        RRR             The total weight of RRR triangles in each bin (if rrr is given)
        DRR             The total weight of DRR triangles in each bin (if drr is given)
        RDD             The total weight of RDD triangles in each bin (if rdd is given)
        ntri            The number of triangles contributing to each bin
        ==========      ================================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):        The name of the file to write to.
            rrr (NNNCorrelation):   The auto-correlation of the random field (RRR)
            drr (NNNCorrelation):   DRR if desired. (default: None)
            rdd (NNNCorrelation):   RDD if desired. (default: None)
            file_type (str):        The type of file to write ('ASCII' or 'FITS').
                                    (default: determine the type automatically from the extension
                                    of file_name.)
            precision (int):        For ASCII output catalogs, the desired precision. (default: 4;
                                    this value can also be given in the constructor in the config
                                    dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
        """
        self.logger.info('Writing NNN correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        self._write_rrr = rrr
        self._write_drr = drr
        self._write_rdd = rdd
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, name, write_patch_results, zero_tot=True)
        self._write_rrr = None
        self._write_drr = None
        self._write_rdd = None

    @property
    def _write_col_names(self):
        rrr = self._write_rrr
        drr = self._write_drr
        rdd = self._write_rdd
        col_names = [ 'r_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv' ]
        if rrr is None:
            col_names += [ 'DDD', 'ntri' ]
        else:
            col_names += [ 'zeta','sigma_zeta','DDD','RRR' ]
            if drr is not None:
                col_names += ['DRR','RDD']
            col_names += [ 'ntri' ]
        return col_names

    @property
    def _write_data(self):
        data = [ self.rnom, self.u, self.v,
                 self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                 self.meand3, self.meanlogd3, self.meanu, self.meanv ]
        rrr = self._write_rrr
        drr = self._write_drr
        rdd = self._write_rdd
        if rrr is None:
            if drr is not None or rdd is not None:
                raise TypeError("rrr must be provided if other combinations are not None")
            data += [ self.weight, self.ntri ]
        else:
            # This will check for other invalid combinations of rrr, drr, etc.
            zeta, varzeta = self.calculateZeta(rrr=rrr, drr=drr, rdd=rdd)

            data += [ zeta, np.sqrt(varzeta),
                      self.weight, rrr.weight * (self.tot/rrr.tot) ]

            if drr is not None:
                data += [ drr.weight * (self.tot/drr.tot), rdd.weight * (self.tot/rdd.tot) ]
            data += [ self.ntri ]

        data = [ col.flatten() for col in data ]
        return data

    @property
    def _write_params(self):
        return { 'tot' : self.tot, 'coords' : self.coords, 'metric' : self.metric,
                 'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

    @depr_pos_kwargs
    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `NNNCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NNN correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

    def _read_from_data(self, data, params):
        s = self.logr.shape
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self._ro.rnom = data['R_nom'].reshape(s)
        else:
            self._ro.rnom = data['r_nom'].reshape(s)
        self.meand1 = data['meand1'].reshape(s)
        self.meanlogd1 = data['meanlogd1'].reshape(s)
        self.meand2 = data['meand2'].reshape(s)
        self.meanlogd2 = data['meanlogd2'].reshape(s)
        self.meand3 = data['meand3'].reshape(s)
        self.meanlogd3 = data['meanlogd3'].reshape(s)
        self.meanu = data['meanu'].reshape(s)
        self.meanv = data['meanv'].reshape(s)
        self.weight = data['DDD'].reshape(s)
        self.ntri = data['ntri'].reshape(s)
        if 'zeta' in data.dtype.names:
            self.zeta = data['zeta'].reshape(s)
            self.varzeta = data['sigma_zeta'].reshape(s)**2
        self.tot = params['tot']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)
        self.npatch3 = params.get('npatch3', 1)


class NNNCrossCorrelation(BinnedCorr3):
    r"""This class handles the calculation a 3-point count-count-count cross-correlation
    function.

    For 3-point cross correlations, it matters which of the two or three fields falls on
    each corner of the triangle.  E.g. is field 1 on the corner opposite d1 (the longest
    size of the triangle) or is it field 2 (or 3) there?  This is in contrast to the 2-point
    correlation where the symmetry of the situation means that it doesn't matter which point
    is identified with each field.  This makes it significantly more complicated to keep track
    of all the relevant information for a 3-point cross correlation function.

    The `NNNCorrelation` class holds a single :math:`\zeta` functions describing all
    possible triangles, parameterized according to their relative side lengths ordered as
    d1 > d2 > d3.

    For a cross-correlation of two fields: N1 - N1 - N2 (i.e. the N1 field is at two of the
    corners and N2 is at one corner), then we need three these :math:`\zeta` functions
    to capture all of the triangles, since the N2 points may be opposite d1 or d2 or d3.
    For a cross-correlation of three fields: N1 - N2 - N3, we need six sets, to account for
    all of the possible permutations relative to the triangle sides.

    Therefore, this class holds 6 instances of `NNNCorrelation`, which in turn hold the
    information about triangles in each of the relevant configurations.  We name these:

    Attributes:
        n1n2n3:     Triangles where N1 is opposite d1, N2 is opposite d2, N3 is opposite d3.
        n1n3n2:     Triangles where N1 is opposite d1, N3 is opposite d2, N2 is opposite d3.
        n2n1n3:     Triangles where N2 is opposite d1, N1 is opposite d2, N3 is opposite d3.
        n2n3n1:     Triangles where N2 is opposite d1, N3 is opposite d2, N1 is opposite d3.
        n3n1n2:     Triangles where N3 is opposite d1, N1 is opposite d2, N2 is opposite d3.
        n3n2n1:     Triangles where N3 is opposite d1, N2 is opposite d2, N1 is opposite d3.

    If for instance N2 and N3 are the same field, then e.g. n1n2n3 and n1n3n2 will have
    the same values.

    Ojects of this class also hold the following attributes, which are identical in each of
    the above NNNCorrelation instances.

    Attributes:
        nbins:      The number of bins in logr where r = d2
        bin_size:   The size of the bins in logr
        min_sep:    The minimum separation being considered
        max_sep:    The maximum separation being considered
        nubins:     The number of bins in u where u = d3/d2
        ubin_size:  The size of the bins in u
        min_u:      The minimum u being considered
        max_u:      The maximum u being considered
        nvbins:     The number of bins in v where v = +-(d1-d2)/d3
        vbin_size:  The size of the bins in v
        min_v:      The minimum v being considered
        max_v:      The maximum v being considered
        logr1d:     The nominal centers of the nbins bins in log(r).
        u1d:        The nominal centers of the nubins bins in u.
        v1d:        The nominal centers of the nvbins bins in v.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_cross` directly,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `BinnedCorr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `BinnedCorr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    @depr_pos_kwargs
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NNNCrossCorrelation`.  See class doc for details.
        """
        BinnedCorr3.__init__(self, config, logger=logger, **kwargs)

        self._ro._d1 = 1  # NData
        self._ro._d2 = 1  # NData
        self._ro._d3 = 1  # NData

        self.n1n2n3 = NNNCorrelation(config, logger=logger, **kwargs)
        self.n1n3n2 = NNNCorrelation(config, logger=logger, **kwargs)
        self.n2n1n3 = NNNCorrelation(config, logger=logger, **kwargs)
        self.n2n3n1 = NNNCorrelation(config, logger=logger, **kwargs)
        self.n3n1n2 = NNNCorrelation(config, logger=logger, **kwargs)
        self.n3n2n1 = NNNCorrelation(config, logger=logger, **kwargs)
        self._all = [self.n1n2n3, self.n1n3n2, self.n2n1n3, self.n2n3n1, self.n3n1n2, self.n3n2n1]

        self.tot = 0.
        self.logger.debug('Finished building NNNCrossCorr')

    def __eq__(self, other):
        """Return whether two `NNNCrossCorrelation` instances are equal"""
        return (isinstance(other, NNNCrossCorrelation) and
                self.nbins == other.nbins and
                self.bin_size == other.bin_size and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.sep_units == other.sep_units and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nubins == other.nubins and
                self.ubin_size == other.ubin_size and
                self.min_v == other.min_v and
                self.max_v == other.max_v and
                self.nvbins == other.nvbins and
                self.vbin_size == other.vbin_size and
                self.coords == other.coords and
                self.bin_type == other.bin_type and
                self.bin_slop == other.bin_slop and
                self.xperiod == other.xperiod and
                self.yperiod == other.yperiod and
                self.zperiod == other.zperiod and
                self.n1n2n3 == other.n1n2n3 and
                self.n1n3n2 == other.n1n3n2 and
                self.n2n1n3 == other.n2n1n3 and
                self.n2n3n1 == other.n2n3n1 and
                self.n3n1n2 == other.n3n1n2 and
                self.n3n2n1 == other.n3n2n1)

    def copy(self):
        """Make a copy"""
        ret = NNNCrossCorrelation.__new__(NNNCrossCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, NNNCorrelation):
                ret.__dict__[key] = item.copy()
            else:
                ret.__dict__[key] = item
        # This needs to be the new list:
        ret._all = [ret.n1n2n3, ret.n1n3n2, ret.n2n1n3, ret.n2n3n1, ret.n3n1n2, ret.n3n2n1]
        return ret

    def _zero_copy(self, tot):
        # A minimal "copy" with zero for the weight array, and the given value for tot.
        ret = NNNCrossCorrelation.__new__(NNNCrossCorrelation)
        ret._ro = self._ro
        ret.n1n2n3 = self.n1n2n3._zero_copy(tot)
        ret.n1n3n2 = self.n1n3n2._zero_copy(tot)
        ret.n2n1n3 = self.n2n1n3._zero_copy(tot)
        ret.n2n3n1 = self.n2n3n1._zero_copy(tot)
        ret.n3n1n2 = self.n3n1n2._zero_copy(tot)
        ret.n3n2n1 = self.n3n2n1._zero_copy(tot)
        ret._all = [ret.n1n2n3, ret.n1n3n2, ret.n2n1n3, ret.n2n3n1, ret.n3n1n2, ret.n3n2n1]
        ret.tot = tot
        setattr(ret, '_nonzero', False)
        return ret

    def __repr__(self):
        return 'NNNCrossCorrelation(config=%r)'%self.config

    @depr_pos_kwargs
    def process_cross12(self, cat1, cat2, *, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        .. note::

            This only adds to the attributes n1n2n3, n2n1n3, n2n3n1, not the ones where
            3 comes before 2.  When running this via the regular `process` method, it will
            combine them at the end to make sure n1n2n3 == n1n3n2, etc. for a complete
            calculation of the 1-2 cross-correlation.

        Parameters:
            cat1 (Catalog):     The first catalog to process. (1 point in each triangle will come
                                from this catalog.)
            cat2 (Catalog):     The second catalog to process. (2 points in each triangle will come
                                from this catalog.)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NNN (1-2) cross-correlations')
        else:
            self.logger.info('Starting process NNN (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        for nnn in self._all:
            nnn._set_metric(self.metric, self.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        _lib.ProcessCross12(self.n1n2n3.corr, self.n2n1n3.corr, self.n2n3n1.corr,
                            f1.data, f2.data, self.output_dots,
                            f1._d, f2._d, self._coords,
                            self._bintype, self._metric)
        tot = cat1.sumw * cat2.sumw**2 / 2.
        self.n1n2n3.tot += tot
        self.n2n1n3.tot += tot
        self.n2n3n1.tot += tot
        self.tot += tot

    @depr_pos_kwargs
    def process_cross(self, cat1, cat2, cat3, *, metric=None, num_threads=None):
        """Process a set of three catalogs, accumulating the 3pt cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            cat3 (Catalog):     The third catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process NNN cross-correlations')
        else:
            self.logger.info('Starting process NNN cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        for nnn in self._all:
            nnn._set_metric(self.metric, self.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f3 = cat3.getNField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 3,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        _lib.ProcessCross3(self.n1n2n3.corr, self.n1n3n2.corr,
                           self.n2n1n3.corr, self.n2n3n1.corr,
                           self.n3n1n2.corr, self.n3n2n1.corr,
                           f1.data, f2.data, f3.data, self.output_dots,
                           f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)
        tot = cat1.sumw * cat2.sumw * cat3.sumw
        for nnn in self._all:
            nnn.tot += tot
        self.tot += tot

    def _finalize(self):
        for nnn in self._all:
            nnn._finalize()

    def finalize(self):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulate values in each bin, so they can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing by the total weight.
        """
        for nnn in self._all:
            nnn.finalize()

    @property
    def nonzero(self):
        """Return if there are any values accumulated yet.  (i.e. ntri > 0)
        """
        return any(nnn.nonzero for nnn in self._all)

    @lazy_property
    def _nonzero(self):
        # The lazy version when we can be sure the object isn't going to accumulate any more.
        return self.nonzero

    def _clear(self):
        """Clear the data vectors
        """
        for nnn in self._all:
            nnn._clear()
        self.tot = 0

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        self.tot = np.sum([c.tot for c in others])
        # Empty ones were only needed for tot.  Remove them now.
        others = [c for c in others if c._nonzero]
        other_all = zip(*[c._all for c in others]) # Transpose list of lists
        for nnn,o_nnn in zip(self._all, other_all):
            nnn._sum(o_nnn)

    def _add_tot(self, i, j, k, c1, c2, c3):
        tot = c1.sumw * c2.sumw * c3.sumw
        self.tot += tot
        for c in self._all:
            c.tot += tot
        self.results[(i,j,k)] = self._zero_copy(tot)

    def __iadd__(self, other):
        """Add a second `NNNCrossCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `NNNCrossCorrelation` objects should not have had
            `finalize` called yet.  Then, after adding them together, you should call `finalize`
            on the sum.
        """
        if not isinstance(other, NNNCrossCorrelation):
            raise TypeError("Can only add another NNNCrossCorrelation object")
        self.n1n2n3 += other.n1n2n3
        self.n1n3n2 += other.n1n3n2
        self.n2n1n3 += other.n2n1n3
        self.n2n3n1 += other.n2n3n1
        self.n3n1n2 += other.n3n1n2
        self.n3n2n1 += other.n3n2n1
        self.tot += other.tot
        return self

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        For NNNCrossCorrelation, this is always just 1.  We don't currently have any ability
        to automatically handle a random catalog for NNNCrossCorrelations, so we don't know
        what the correct weight would be for a given patch or set of patches.  This value
        is only used by the sample method of covariance estimation, so this limitation means
        that sample covariances may be expected to be less accurate than normal when used with
        NNNCorrelations.
        """
        return 1.

    @depr_pos_kwargs
    def process(self, cat1, cat2, cat3=None, *, metric=None, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True):
        """Accumulate the cross-correlation of the points in the given Catalogs: cat1, cat2, cat3.

        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first N field.
            cat2 (Catalog):     A catalog or list of catalogs for the second N field.
            cat3 (Catalog):     A catalog or list of catalogs for the third N field.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
            comm (mpi4py.Comm): If running MPI, an mpi4py Comm object to communicate between
                                processes.  If used, the rank=0 process will have the final
                                computation. This only works if using patches. (default: None)
            low_mem (bool):     Whether to sacrifice a little speed to try to reduce memory usage.
                                This only works if using patches. (default: False)
            initialize (bool):  Whether to begin the calculation with a call to
                                `BinnedCorr3.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
        """
        import math
        if initialize:
            self.clear()
            self._process12 = False

        if not isinstance(cat1,list): cat1 = cat1.get_patches()
        if not isinstance(cat2,list): cat2 = cat2.get_patches()
        if cat3 is not None and not isinstance(cat3,list): cat3 = cat3.get_patches()

        if cat3 is None:
            self._process12 = True
            self._process_all_cross12(cat1, cat2, metric, num_threads, comm, low_mem)
        else:
            self._process_all_cross(cat1, cat2, cat3, metric, num_threads, comm, low_mem)

        if finalize:
            if self._process12:
                # Then some of the processing involved a cross12 calculation.
                # This means that spots 2 and 3 should not be distinguished.
                # Combine the relevant arrays.
                self.n1n2n3 += self.n1n3n2
                self.n2n1n3 += self.n3n1n2
                self.n2n3n1 += self.n3n2n1
                # Copy back by doing clear and +=.
                self.n1n3n2.clear()
                self.n3n1n2.clear()
                self.n3n2n1.clear()
                self.n1n3n2 += self.n1n2n3
                self.n3n1n2 += self.n2n1n3
                self.n3n2n1 += self.n2n3n1

            self.finalize()

    @depr_pos_kwargs
    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False):
        r"""Write the correlation function to the file, file_name.

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
        """
        self.logger.info('Writing NNN cross-correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            names = [ 'n1n2n3', 'n1n3n2', 'n2n1n3', 'n2n3n1', 'n3n1n2', 'n3n2n1' ]
            for name, corr in zip(names, self._all):
                corr._write(writer, name, write_patch_results, zero_tot=True)

    @depr_pos_kwargs
    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `NNNCrossCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NNN cross-correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            names = [ 'n1n2n3', 'n1n3n2', 'n2n1n3', 'n2n3n1', 'n3n1n2', 'n3n2n1' ]
            for name, corr in zip(names, self._all):
                corr._read(reader, name)

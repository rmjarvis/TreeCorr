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
from .catalog import calculateVarG
from .binnedcorr3 import BinnedCorr3
from .util import double_ptr as dp
from .util import gen_read, gen_write, gen_multi_read, gen_multi_write


class GGGCorrelation(BinnedCorr3):
    r"""This class handles the calculation and storage of a 3-point shear-shear-shear correlation
    function.

    We use the "natural components" of the shear 3-point function described by Schneider &
    Lombardi (2003) [Astron.Astrophys. 397 (2003) 809-818].  In this paradigm, the shears
    are projected relative to some point defined by the geometry of the triangle.  They
    give several reasonable choices for this point.  We choose the triangle's centroid as the
    "most natural" point, as many simple shear fields have purely real :math:`\Gamma_0` using
    this definition.  It is also a fairly simple point to calculate in the code compared to
    some of the other options they offer, so projections relative to it are fairly efficient.

    There are 4 complex-valued 3-point shear corrletion functions defined for triples of shear
    values projected relative to the line joining the location of the shear to the cenroid of
    the triangle:

    .. math::

        \Gamma_0 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_1 &= \langle \gamma(\mathbf{x1})^* \gamma(\mathbf{x2}) \gamma(\mathbf{x3}) \rangle \\
        \Gamma_2 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2})^* \gamma(\mathbf{x3}) \rangle \\
        \Gamma_3 &= \langle \gamma(\mathbf{x1}) \gamma(\mathbf{x2}) \gamma(\mathbf{x3})^* \rangle \\

    where :math:`\mathbf{x1}, \mathbf{x2}, \mathbf{x3}` are the corners of the triange opposite
    sides d1, d2, d3 respectively, where d1 > d2 > d3, and :math:`{}^*` indicates complex
    conjugation.

    See the doc string of `BinnedCorr3` for a description of how the triangles
    are binned.

    This class only holds one set of these :math:`\Gamma` functions, which means that it is only
    directly applicable for computing auto-correlations.  To describe a cross-correlation of one
    shear field with another, you need three sets of these functions.  To describe a three-way
    cross-correlation of three different shear fields, you need six.  These use cases are
    enabled by the class `GGGCrossCorrelation` which holds six instances of this class to keep
    track of all the various triangles.  See that class for more details.

    Ojects of this class holds the following attributes:

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

    In addition, the following attributes are numpy arrays whose shape is (nbins, nubins, nvbins):

    Attributes:
        logr:       The nominal center of each bin in log(r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        u:          The nominal center of each bin in u.
        v:          The nominal center of each bin in v.
        meand1:     The (weighted) mean value of d1 for the triangles in each bin.
        meanlogd1:  The mean value of log(d1) for the triangles in each bin.
        meand2:     The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        meanlogd2:  The mean value of log(d2) for the triangles in each bin.
        meand2:     The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd2:  The mean value of log(d3) for the triangles in each bin.
        meanu:      The mean value of u for the triangles in each bin.
        meanv:      The mean value of v for the triangles in each bin.
        gam0:       The 0th "natural" correlation function, :math:`\Gamma_0(r,u,v)`.
        gam1:       The 1st "natural" correlation function, :math:`\Gamma_1(r,u,v)`.
        gam2:       The 2nd "natural" correlation function, :math:`\Gamma_2(r,u,v)`.
        gam3:       The 3rd "natural" correlation function, :math:`\Gamma_3(r,u,v)`.
        vargam0:    The variance of :math:`\Gamma_0`, only including the shot noise
                    propagated into the final correlation.  This (and the related values for
                    1,2,3) does not include sample variance, so it is always an underestimate
                    of the actual variance.
        vargam1:    The variance of :math:`\Gamma_1`.
        vargam2:    The variance of :math:`\Gamma_2`.
        vargam3:    The variance of :math:`\Gamma_3`.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_auto` and/or
        `process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
        the `finalize` function is called.

    The typical usage pattern is as follows::

        >>> ggg = treecorr.GGGCorrelation(config)
        >>> ggg.process(cat)              # For auto-correlation.
        >>> ggg.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> ggg.write(file_name)          # Write out to a file.
        >>> gam0 = ggg.gam0, etc.         # To access gamma values directly.
        >>> gam0r = ggg.gam0r             # You can also access real and imag parts separately.
        >>> gam0i = ggg.gam0i

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
    def __init__(self, config=None, logger=None, **kwargs):
        """Initialize `GGGCorrelation`.  See class doc for details.
        """
        BinnedCorr3.__init__(self, config, logger, **kwargs)

        self._ro._d1 = 3  # GData
        self._ro._d2 = 3  # GData
        self._ro._d3 = 3  # GData
        shape = self.logr.shape
        self.gam0r = np.zeros(shape, dtype=float)
        self.gam1r = np.zeros(shape, dtype=float)
        self.gam2r = np.zeros(shape, dtype=float)
        self.gam3r = np.zeros(shape, dtype=float)
        self.gam0i = np.zeros(shape, dtype=float)
        self.gam1i = np.zeros(shape, dtype=float)
        self.gam2i = np.zeros(shape, dtype=float)
        self.gam3i = np.zeros(shape, dtype=float)
        self.vargam0 = np.zeros(shape, dtype=float)
        self.vargam1 = np.zeros(shape, dtype=float)
        self.vargam2 = np.zeros(shape, dtype=float)
        self.vargam3 = np.zeros(shape, dtype=float)
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
        self.logger.debug('Finished building GGGCorr')

    @property
    def gam0(self):
        return self.gam0r + 1j * self.gam0i

    @property
    def gam1(self):
        return self.gam1r + 1j * self.gam1i

    @property
    def gam2(self):
        return self.gam2r + 1j * self.gam2i

    @property
    def gam3(self):
        return self.gam3r + 1j * self.gam3i

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _lib.BuildCorr3(
                    self._d1, self._d2, self._d3, self._bintype,
                    self._min_sep,self._max_sep,self.nbins,self._bin_size,self.b,
                    self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                    self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                    self.xperiod, self.yperiod, self.zperiod,
                    dp(self.gam0r), dp(self.gam0i), dp(self.gam1r), dp(self.gam1i),
                    dp(self.gam2r), dp(self.gam2i), dp(self.gam3r), dp(self.gam3i),
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
        """Return whether two `GGGCorrelation` instances are equal"""
        return (isinstance(other, GGGCorrelation) and
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
                np.array_equal(self.meand1, other.meand1) and
                np.array_equal(self.meanlogd1, other.meanlogd1) and
                np.array_equal(self.meand2, other.meand2) and
                np.array_equal(self.meanlogd2, other.meanlogd2) and
                np.array_equal(self.meand3, other.meand3) and
                np.array_equal(self.meanlogd3, other.meanlogd3) and
                np.array_equal(self.meanu, other.meanu) and
                np.array_equal(self.meanv, other.meanv) and
                np.array_equal(self.gam0r, other.gam0r) and
                np.array_equal(self.gam0i, other.gam0i) and
                np.array_equal(self.gam1r, other.gam1r) and
                np.array_equal(self.gam1i, other.gam1i) and
                np.array_equal(self.gam2r, other.gam2r) and
                np.array_equal(self.gam2i, other.gam2i) and
                np.array_equal(self.gam3r, other.gam3r) and
                np.array_equal(self.gam3i, other.gam3i) and
                np.array_equal(self.vargam0, other.vargam0) and
                np.array_equal(self.vargam1, other.vargam1) and
                np.array_equal(self.vargam2, other.vargam2) and
                np.array_equal(self.vargam3, other.vargam3) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.ntri, other.ntri))

    def copy(self):
        """Make a copy"""
        ret = GGGCorrelation.__new__(GGGCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to by deep copied.
                ret.__dict__[key] = item.copy()
            else:
                # For everything else, shallow copy is fine.
                # In particular don't deep copy config or logger
                # Most of the rest are scalars, which copy fine this way.
                ret.__dict__[key] = item
        ret._corr = None # We'll want to make a new one of these if we need it.
        return ret

    def __repr__(self):
        return 'GGGCorrelation(config=%r)'%self.config

    def process_auto(self, cat, metric=None, num_threads=None):
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
            self.logger.info('Starting process GGG auto-correlations')
        else:
            self.logger.info('Starting process GGG auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getGField(min_size, max_size, self.split_method,
                              bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        _lib.ProcessAuto3(self.corr, field.data, self.output_dots,
                                   field._d, self._coords, self._bintype, self._metric)

    def process_cross12(self, cat1, cat2, metric=None, num_threads=None):
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
            self.logger.info('Starting process GGG (1-2) cross-correlations')
        else:
            self.logger.info('Starting process GGG (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        _lib.ProcessCross12(self.corr, self.corr, self.corr,
                            f1.data, f2.data, self.output_dots,
                            f1._d, f2._d, self._coords,
                            self._bintype, self._metric)

    def process_cross(self, cat1, cat2, cat3, metric=None, num_threads=None):
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
            self.logger.info('Starting process GGG cross-correlations')
        else:
            self.logger.info('Starting process GGG cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f3 = cat3.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 6 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        _lib.ProcessCross3(self.corr, self.corr, self.corr,
                           self.corr, self.corr, self.corr,
                           f1.data, f2.data, f3.data, self.output_dots,
                           f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.gam0r[mask1] /= self.weight[mask1]
        self.gam0i[mask1] /= self.weight[mask1]
        self.gam1r[mask1] /= self.weight[mask1]
        self.gam1i[mask1] /= self.weight[mask1]
        self.gam2r[mask1] /= self.weight[mask1]
        self.gam2i[mask1] /= self.weight[mask1]
        self.gam3r[mask1] /= self.weight[mask1]
        self.gam3i[mask1] /= self.weight[mask1]
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
        self.meand1[mask2] = np.abs(self.v[mask2]) * self.meand3[mask2] + self.meand2[mask2]
        self.meanlogd1[mask2] = np.log(self.meand1[mask2])

    def finalize(self, varg1, varg2, varg3):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing by the total weight.

        Parameters:
            varg1 (float):  The shear variance for the first field.
            varg2 (float):  The shear variance for the second field.
            varg3 (float):  The shear variance for the third field.
        """
        self._finalize()
        mask1 = self.weight != 0
        mask2 = self.weight == 0
        self._var_num = varg1 * varg2 * varg3
        self.cov = self.estimate_cov(self.var_method)
        # Note: diagonal should be very close to pure real.  So ok to just copy real part.
        diag = self.cov.diagonal()
        # This should never trigger.  If you find this assert to fail, please post an
        # issue about it describing your use case that caused it to fail.
        assert np.sum(diag.imag**2) <= 1.e-8 * np.sum(diag.real**2)
        self.vargam0.ravel()[:] = self.cov.diagonal()[0:self._nbins].real
        self.vargam1.ravel()[:] = self.cov.diagonal()[self._nbins:2*self._nbins].real
        self.vargam2.ravel()[:] = self.cov.diagonal()[2*self._nbins:3*self._nbins].real
        self.vargam3.ravel()[:] = self.cov.diagonal()[3*self._nbins:4*self._nbins].real

    def _clear(self):
        """Clear the data vectors
        """
        self.gam0r[:,:,:] = 0.
        self.gam0i[:,:,:] = 0.
        self.gam1r[:,:,:] = 0.
        self.gam1i[:,:,:] = 0.
        self.gam2r[:,:,:] = 0.
        self.gam2i[:,:,:] = 0.
        self.gam3r[:,:,:] = 0.
        self.gam3i[:,:,:] = 0.
        self.vargam0[:,:,:] = 0.
        self.vargam1[:,:,:] = 0.
        self.vargam2[:,:,:] = 0.
        self.vargam3[:,:,:] = 0.
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

    def __iadd__(self, other):
        """Add a second `GGGCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `GGGCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, GGGCorrelation):
            raise TypeError("Can only add another GGGCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.nubins == other.nubins and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nvbins == other.nvbins and
                self.min_v == other.min_v and
                self.max_v == other.max_v):
            raise ValueError("GGGCorrelation to be added is not compatible with this one.")

        if not other.nonzero: return self
        self._set_metric(other.metric, other.coords, other.coords, other.coords)
        self.gam0r[:] += other.gam0r[:]
        self.gam0i[:] += other.gam0i[:]
        self.gam1r[:] += other.gam1r[:]
        self.gam1i[:] += other.gam1i[:]
        self.gam2r[:] += other.gam2r[:]
        self.gam2i[:] += other.gam2i[:]
        self.gam3r[:] += other.gam3r[:]
        self.gam3i[:] += other.gam3i[:]
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

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c.gam0r for c in others], axis=0, out=self.gam0r)
        np.sum([c.gam0i for c in others], axis=0, out=self.gam0i)
        np.sum([c.gam1r for c in others], axis=0, out=self.gam1r)
        np.sum([c.gam1i for c in others], axis=0, out=self.gam1i)
        np.sum([c.gam2r for c in others], axis=0, out=self.gam2r)
        np.sum([c.gam2i for c in others], axis=0, out=self.gam2i)
        np.sum([c.gam3r for c in others], axis=0, out=self.gam3r)
        np.sum([c.gam3i for c in others], axis=0, out=self.gam3i)
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

    def process(self, cat1, cat2=None, cat3=None, metric=None, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True):
        """Compute the 3pt correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        .. note::

            For a correlation of multiple catalogs, it typically matters which corner of the
            triangle comes from which catalog, which is not kept track of by this function.
            The final accumulation will have d1 > d2 > d3 regardless of which input catalog
            appears at each corner.  The class which keeps track of which catalog appears
            in each position in the triangle is `GGGCrossCorrelation`.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first G field.
            cat2 (Catalog):     A catalog or list of catalogs for the second G field.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third G field.
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

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if cat2 is not None and not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)
        if cat3 is not None and not isinstance(cat3,list):
            cat3 = cat3.get_patches(low_mem=low_mem)

        if cat2 is None:
            if cat3 is not None:
                raise ValueError("For two catalog case, use cat1,cat2, not cat1,cat3")
            self._process_all_auto(cat1, metric, num_threads, comm, low_mem)
        elif cat3 is None:
            self._process_all_cross12(cat1, cat2, metric, num_threads, comm, low_mem)
        else:
            self._process_all_cross(cat1, cat2, cat3, metric, num_threads, comm, low_mem)

        if finalize:
            if cat2 is None:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = varg1
                varg3 = varg1
                self.logger.info("varg = %f: sig_g = %f",varg1,math.sqrt(varg1))
            elif cat3 is None:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = calculateVarG(cat2, low_mem=low_mem)
                varg3 = varg2
                self.logger.info("varg1 = %f: sig_g = %f",varg1,math.sqrt(varg1))
                self.logger.info("varg2 = %f: sig_g = %f",varg2,math.sqrt(varg2))
            else:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = calculateVarG(cat2, low_mem=low_mem)
                varg3 = calculateVarG(cat3, low_mem=low_mem)
                self.logger.info("varg1 = %f: sig_g = %f",varg1,math.sqrt(varg1))
                self.logger.info("varg2 = %f: sig_g = %f",varg2,math.sqrt(varg2))
                self.logger.info("varg3 = %f: sig_g = %f",varg3,math.sqrt(varg3))
            self.finalize(varg1,varg2,varg3)

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, the concatenation of gam0.ravel(), gam1.ravel(), gam2.ravel(), gam3.ravel().

        .. note::

            This is a complex array, unlike most other statistics.
            The computed covariance matrix will be complex, although since it is Hermitian the
            diagonal is real, so the resulting vargam0, etc. will all be real arrays.
        """
        return np.concatenate([self.gam0.ravel(), self.gam1.ravel(),
                               self.gam2.ravel(), self.gam3.ravel()])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        In this case, 4 copies of self.weight.ravel().
        """
        return np.concatenate([self.weight.ravel()] * 4)

    def write(self, file_name, file_type=None, precision=None):
        r"""Write the correlation function to the file, file_name.

        As described in the doc string for `GGGCorrelation`, we use the "natural components" of
        the shear 3-point function described by Schneider & Lombardi (2003) using the triangle
        centroid as the projection point.  There are 4 complex-valued natural components, so there
        are 8 columns in the output file.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r = d2 where d1 > d2 > d3
        u_nom           The nominal center of the bin in u = d3/d2
        v_nom           The nominal center of the bin in v = +-(d1-d2)/d3
        meand1          The mean value :math:`\langle d1\rangle` of triangles that
                        fell into each bin
        meanlogd1       The mean value :math:`\langle \log(d1)\rangle` of triangles
                        that fell into each bin
        meand2          The mean value :math:`\langle d2\rangle` of triangles that
                        fell into each bin
        meanlogd2       The mean value :math:`\langle \log(d2)\rangle` of triangles
                        that fell into each bin
        meand3          The mean value :math:`\langle d3\rangle` of triangles that
                        fell into each bin
        meanlogd3       The mean value :math:`\langle \log(d3)\rangle` of triangles
                        that fell into each bin
        meanu           The mean value :math:`\langle u\rangle` of triangles that
                        fell into each bin
        meanv           The mean value :math:`\langle v\rangle` of triangles that
                        fell into each bi.
        gam0r           The real part of the estimator of :math:`\Gamma_0(r,u,v)`
        gam0i           The imag part of the estimator of :math:`\Gamma_0(r,u,v)`
        gam1r           The real part of the estimator of :math:`\Gamma_1(r,u,v)`
        gam1i           The imag part of the estimator of :math:`\Gamma_1(r,u,v)`
        gam2r           The real part of the estimator of :math:`\Gamma_2(r,u,v)`
        gam2i           The imag part of the estimator of :math:`\Gamma_2(r,u,v)`
        gam3r           The real part of the estimator of :math:`\Gamma_3(r,u,v)`
        gam3i           The imag part of the estimator of :math:`\Gamma_3(r,u,v)`
        sigma_gam0      The sqrt of the variance estimate of :math:`\Gamma_0`
        sigma_gam1      The sqrt of the variance estimate of :math:`\Gamma_1`
        sigma_gam2      The sqrt of the variance estimate of :math:`\Gamma_2`
        sigma_gam3      The sqrt of the variance estimate of :math:`\Gamma_3`
        weight          The total weight of triangles contributing to each bin.
        ntri            The number of triangles contributing to each bin.
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing GGG correlations to %s',file_name)

        col_names = [ 'r_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv',
                      'gam0r', 'gam0i', 'gam1r', 'gam1i', 'gam2r', 'gam2i', 'gam3r', 'gam3i',
                      'sigma_gam0', 'sigma_gam1', 'sigma_gam2', 'sigma_gam3', 'weight', 'ntri' ]
        columns = [ self.rnom, self.u, self.v,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv,
                    self.gam0r, self.gam0i, self.gam1r, self.gam1i,
                    self.gam2r, self.gam2i, self.gam3r, self.gam3i,
                    np.sqrt(self.vargam0), np.sqrt(self.vargam1), np.sqrt(self.vargam2),
                    np.sqrt(self.vargam3), self.weight, self.ntri ]

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        if precision is None:
            precision = self.config.get('precision', 4)

        gen_write(
            file_name, col_names, columns,
            params=params, precision=precision, file_type=file_type, logger=self.logger)

    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `GGGCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading GGG correlations from %s',file_name)

        data, params = gen_read(file_name, file_type=file_type, logger=self.logger)
        s = self.logr.shape
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self._ro.rnom = data['R_nom'].reshape(s)
        else:
            self._ro.rnom = data['r_nom'].reshape(s)
        self._ro.logr = np.log(self.rnom)
        self._ro.u = data['u_nom'].reshape(s)
        self._ro.v = data['v_nom'].reshape(s)
        self.meand1 = data['meand1'].reshape(s)
        self.meanlogd1 = data['meanlogd1'].reshape(s)
        self.meand2 = data['meand2'].reshape(s)
        self.meanlogd2 = data['meanlogd2'].reshape(s)
        self.meand3 = data['meand3'].reshape(s)
        self.meanlogd3 = data['meanlogd3'].reshape(s)
        self.meanu = data['meanu'].reshape(s)
        self.meanv = data['meanv'].reshape(s)
        self.gam0r = data['gam0r'].reshape(s)
        self.gam0i = data['gam0i'].reshape(s)
        self.gam1r = data['gam1r'].reshape(s)
        self.gam1i = data['gam1i'].reshape(s)
        self.gam2r = data['gam2r'].reshape(s)
        self.gam2i = data['gam2i'].reshape(s)
        self.gam3r = data['gam3r'].reshape(s)
        self.gam3i = data['gam3i'].reshape(s)
        # Read old output files without error.
        if 'sigma_gam' in data.dtype.names:  # pragma: no cover
            self.vargam0 = data['sigma_gam'].reshape(s)**2
            self.vargam1 = data['sigma_gam'].reshape(s)**2
            self.vargam2 = data['sigma_gam'].reshape(s)**2
            self.vargam3 = data['sigma_gam'].reshape(s)**2
        else:
            self.vargam0 = data['sigma_gam0'].reshape(s)**2
            self.vargam1 = data['sigma_gam1'].reshape(s)**2
            self.vargam2 = data['sigma_gam2'].reshape(s)**2
            self.vargam3 = data['sigma_gam3'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.ntri = data['ntri'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()

    @classmethod
    def _calculateT(cls, s, t, k1, k2, k3):
        # First calculate q values:
        q1 = (s+t)/3.
        q2 = q1-t
        q3 = q1-s

        # |qi|^2 shows up a lot, so save these.
        # The a stands for "absolute", and the ^2 part is implicit.
        a1 = np.abs(q1)**2
        a2 = np.abs(q2)**2
        a3 = np.abs(q3)**2
        a123 = a1*a2*a3

        # These combinations also appear multiple times.
        # The b doesn't stand for anything.  It's just the next letter after a.
        b1 = np.conjugate(q1)**2*q2*q3
        b2 = np.conjugate(q2)**2*q1*q3
        b3 = np.conjugate(q3)**2*q1*q2

        if k1==1 and k2==1 and k3==1:

            # Some factors we use multiple times
            expfactor = -np.exp(-(a1 + a2 + a3)/2)

            # JBJ Equation 51
            # Note that we actually accumulate the Gammas with a different choice for
            # alpha_i.  We accumulate the shears relative to the q vectors, not relative to s.
            # cf. JBJ Equation 41 and footnote 3.  The upshot is that we multiply JBJ's formulae
            # be (q1q2q3)^2 / |q1q2q3|^2 for T0 and (q1*q2q3)^2/|q1q2q3|^2 for T1.
            # Then T0 becomes
            # T0 = -(|q1 q2 q3|^2)/24 exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
            T0 = expfactor * a123 / 24

            # JBJ Equation 52
            # After the phase adjustment, T1 becomes:
            # T1 = -[(|q1 q2 q3|^2)/24
            #        - (q1*^2 q2 q3)/9
            #        + (q1*^4 q2^2 q3^2 + 2 |q2 q3|^2 q1*^2 q2 q3)/(|q1 q2 q3|^2)/27
            #       ] exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
            T1 = expfactor * (a123 / 24 - b1 / 9 + (b1**2 + 2*a2*a3*b1) / (a123 * 27))
            T2 = expfactor * (a123 / 24 - b2 / 9 + (b2**2 + 2*a1*a3*b2) / (a123 * 27))
            T3 = expfactor * (a123 / 24 - b3 / 9 + (b3**2 + 2*a1*a2*b3) / (a123 * 27))

        else:
            # SKL Equation 63:
            k1sq = k1*k1
            k2sq = k2*k2
            k3sq = k3*k3
            Theta2 = ((k1sq*k2sq + k1sq*k3sq + k2sq*k3sq)/3.)**0.5
            k1sq /= Theta2   # These are now what SKL calls theta_i^2 / Theta^2
            k2sq /= Theta2
            k3sq /= Theta2
            Theta4 = Theta2*Theta2
            Theta6 = Theta4*Theta2
            S = k1sq * k2sq * k3sq

            # SKL Equation 64:
            Z = ((2*k2sq + 2*k3sq - k1sq) * a1 +
                 (2*k3sq + 2*k1sq - k2sq) * a2 +
                 (2*k1sq + 2*k2sq - k3sq) * a3) / (6*Theta2)
            expfactor = -S * np.exp(-Z) / Theta4

            # SKL Equation 65:
            f1 = (k2sq+k3sq)/2 + (k2sq-k3sq)*(q2-q3)/(6*q1)
            f2 = (k3sq+k1sq)/2 + (k3sq-k1sq)*(q3-q1)/(6*q2)
            f3 = (k1sq+k2sq)/2 + (k1sq-k2sq)*(q1-q2)/(6*q3)
            f1c = np.conjugate(f1)
            f2c = np.conjugate(f2)
            f3c = np.conjugate(f3)

            # SKL Equation 69:
            g1 = k2sq*k3sq + (k3sq-k2sq)*k1sq*(q2-q3)/(3*q1)
            g2 = k3sq*k1sq + (k1sq-k3sq)*k2sq*(q3-q1)/(3*q2)
            g3 = k1sq*k2sq + (k2sq-k1sq)*k3sq*(q1-q2)/(3*q3)
            g1c = np.conjugate(g1)
            g2c = np.conjugate(g2)
            g3c = np.conjugate(g3)

            # SKL Equation 62:
            T0 = expfactor * a123 * f1c**2 * f2c**2 * f3c**2 / (24.*Theta6)

            # SKL Equation 68:
            T1 = expfactor * (
                a123 * f1**2 * f2c**2 * f3c**2 / (24*Theta6) -
                b1 * f1*f2c*f3c*g1c / (9*Theta4) +
                (b1**2 * g1c**2 + 2*k2sq*k3sq*a2*a3*b1 * f2c * f3c) / (a123 * 27*Theta2))
            T2 = expfactor * (
                a123 * f1c**2 * f2**2 * f3c**2 / (24*Theta6) -
                b2 * f1c*f2*f3c*g2c / (9*Theta4) +
                (b2**2 * g2c**2 + 2*k1sq*k3sq*a1*a3*b2 * f1c * f3c) / (a123 * 27*Theta2))
            T3 = expfactor * (
                a123 * f1c**2 * f2c**2 * f3**2 / (24*Theta6) -
                b3 * f1c*f2c*f3*g3c / (9*Theta4) +
                (b3**2 * g3c**2 + 2*k1sq*k2sq*a1*a2*b3 * f1c * f2c) / (a123 * 27*Theta2))

        return T0, T1, T2, T3

    def calculateMap3(self, R=None, k2=1, k3=1):
        r"""Calculate the skewness of the aperture mass from the correlation function.

        The equations for this come from Jarvis, Bernstein & Jain (2004, MNRAS, 352).
        See their section 3, especially equations 51 and 52 for the :math:`T_i` functions,
        equations 60 and 61 for the calculation of :math:`\langle \cal M^3 \rangle` and
        :math:`\langle \cal M^2 M^* \rangle`, and equations 55-58 for how to convert
        these to the return values.

        If k2 or k3 != 1, then this routine calculates the generalization of the skewness
        proposed by Schneider, Kilbinger & Lombardi (2005, A&A, 431):
        :math:`\langle M_{ap}^3(R, k_2 R, k_3 R)\rangle` and related values.

        If k2 = k3 = 1 (the default), then there are only 4 combinations of Map and Mx
        that are relevant:

        - map3 = :math:`\langle M_{ap}^3(R)\rangle`
        - map2mx = :math:`\langle M_{ap}^2(R) M_\times(R)\rangle`,
        - mapmx2 = :math:`\langle M_{ap}(R) M_\times(R)\rangle`
        - mx3 = :math:`\langle M_{\rm \times}^3(R)\rangle`

        However, if k2 or k3 != 1, then there are 8 combinations:

        - map3 = :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
        - mapmapmx = :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
        - mapmxmap = :math:`\langle M_{ap}(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmapmap = :math:`\langle M_\times(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmxmap = :math:`\langle M_\times(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
        - mxmapmx = :math:`\langle M_\times(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
        - mapmxmx = :math:`\langle M_{ap}(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
        - mx3 = :math:`\langle M_\times(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`

        To accommodate this full generality, we always return all 8 values, along with the
        estimated variance (which is equal for each), even when k2 = k3 = 1.

        .. note::

            The formulae for the ``m2_uform`` = 'Schneider' definition of the aperture mass,
            described in the documentation of `calculateMapSq`, are not known, so that is not an
            option here.  The calculations here use the definition that corresponds to
            ``m2_uform`` = 'Crittenden'.

        Parameters:
            R (array):      The R values at which to calculate the aperture mass statistics.
                            (default: None, which means use self.rnom1d)
            k2 (float):     If given, the ratio R2/R1 in the SKL formulae. (default: 1)
            k3 (float):     If given, the ratio R3/R1 in the SKL formulae. (default: 1)

        Returns:
            Tuple containing:

            - map3 = array of :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
            - mapmapmx = array of :math:`\langle M_{ap}(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
            - mapmxmap = array of :math:`\langle M_{ap}(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmapmap = array of :math:`\langle M_\times(R) M_{ap}(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmxmap = array of :math:`\langle M_\times(R) M_\times(k_2 R) M_{ap}(k_3 R)\rangle`
            - mxmapmx = array of :math:`\langle M_\times(R) M_{ap}(k_2 R) M_\times(k_3 R)\rangle`
            - mapmxmx = array of :math:`\langle M_{ap}(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
            - mx3 = array of :math:`\langle M_\times(R) M_\times(k_2 R) M_\times(k_3 R)\rangle`
            - varmap3 = array of variance estimates of the above values
        """
        # As in the calculateMapSq function, we Make s and t matrices, so we can eventually do the
        # integral by doing a matrix product.
        if R is None:
            R = self.rnom1d

        # Pick s = d2, so dlogs is bin_size
        s = d2 = np.outer(1./R, self.meand2.ravel())

        # We take t = d3, but we need the x and y components. (relative to s along x axis)
        # cf. Figure 1 in JBJ.
        # d1^2 = d2^2 + d3^2 - 2 d2 d3 cos(theta1)
        # tx = d3 cos(theta1) = (d2^2 + d3^2 - d1^2)/2d2
        # Simplify this using u=d3/d2 and v=(d1-d2)/d3
        #    = (d3^2 - (d1+d2)(d1-d2)) / 2d2
        #    = d3 (d3 - (d1+d2)v) / 2d2
        #    = d3 (u - (2+uv)v)/2
        #    = d3 (u - 2v - uv^2)/2
        #    = d3 (u(1-v^2)/2 - v)
        # Note that v here is really |v|.  We'll account for the sign of v in ty.
        d3 = np.outer(1./R, self.meand3.ravel())
        d1 = np.outer(1./R, self.meand1.ravel())
        u = self.meanu.ravel()
        v = self.meanv.ravel()
        tx = d3*(0.5*u*(1-v**2) - np.abs(v))
        # This form tends to be more stable near potentially degenerate triangles
        # than tx = (d2*d2 + d3*d3 - d1*d1) / (2*d2)
        # However, add a check to make sure.
        bad = (tx <= -d3) | (tx >= d3)
        if np.any(bad):  # pragma: no cover
            self.logger.warning("Warning: Detected some invalid triangles when computing Map^3")
            self.logger.warning("Excluding these triangles from the integral.")
            self.logger.debug("N bad points = %s",np.sum(bad))
            self.logger.debug("d1[bad] = %s",d1[bad])
            self.logger.debug("d2[bad] = %s",d2[bad])
            self.logger.debug("d3[bad] = %s",d3[bad])
            self.logger.debug("u[bad] = %s",u[bad])
            self.logger.debug("v[bad] = %s",v[bad])
            self.logger.debug("tx[bad] = %s",tx[bad])
        bad = np.where(bad)
        tx[bad] = 0  # for now to avoid nans
        ty = np.sqrt(d3**2 - tx**2)
        ty[:,self.meanv.ravel() > 0] *= -1.
        t = tx + 1j * ty

        # Next we need to construct the T values.
        T0, T1, T2, T3 = self._calculateT(s,t,1.,k2,k3)

        # Finally, account for the Jacobian in d^2t: jac = |J(tx, ty; u, v)|,
        # since our Gammas are accumulated in s, u, v, not s, tx, ty.
        # u = d3/d2, v = (d1-d2)/d3
        # tx = d3 (u - 2v - uv^2)/2
        #    = s/2 (u^2 - 2uv - u^2v^2)
        # dtx/du = s (u - v - uv^2)
        # dtx/dv = -us (1 + uv)
        # ty = sqrt(d3^2 - tx^2) = sqrt(u^2 s^2 - tx^2)
        # dty/du = s^2 u/2ty (1-v^2) (2 + 3uv - u^2 + u^2v^2)
        # dty/dv = s^2 u^2/2ty (1 + uv) (u - uv^2 - 2uv)
        #
        # After some algebra...
        #
        # J = s^3 u^2 (1+uv) / ty
        #   = d3^2 d1 / ty
        #
        jac = np.abs(d3*d3*d1/ty)
        jac[bad] = 0.  # Exclude any bad triangles from the integral.
        d2t = jac * self.ubin_size * self.vbin_size / (2.*np.pi)
        sds = s * s * self.bin_size  # Remember bin_size is dln(s)
        # Note: these are really d2t/2piR^2 and sds/R^2, which are what actually show up
        # in JBJ equations 45 and 50.

        T0 *= sds * d2t
        T1 *= sds * d2t
        T2 *= sds * d2t
        T3 *= sds * d2t

        # Now do the integral by taking the matrix products.
        gam0 = self.gam0.ravel()
        gam1 = self.gam1.ravel()
        gam2 = self.gam2.ravel()
        gam3 = self.gam3.ravel()
        vargam0 = self.vargam0.ravel()
        vargam1 = self.vargam1.ravel()
        vargam2 = self.vargam2.ravel()
        vargam3 = self.vargam3.ravel()
        mmm = T0.dot(gam0)
        mcmm = T1.dot(gam1)
        mmcm = T2.dot(gam2)
        mmmc = T3.dot(gam3)
        varmmm = (np.abs(T0)**2).dot(vargam0)
        varmcmm = (np.abs(T1)**2).dot(vargam1)
        varmmcm = (np.abs(T2)**2).dot(vargam2)
        varmmmc = (np.abs(T3)**2).dot(vargam3)

        if k2 == 1 and k3 == 1:
            mmm *= 6
            varmmm *= 6
            mcmm += mmcm
            mcmm += mmmc
            mcmm *= 2
            mmcm = mmmc = mcmm
            varmcmm += varmmcm
            varmcmm += varmmmc
            varmcmm *= 2
            varmmcm = varmmmc = varmcmm
        else:
            # Repeat the above for the other permutations
            for (_k1, _k2, _k3, _mcmm, _mmcm, _mmmc, _varmcmm, _varmmcm, _varmmmc) in [
                    (1,k3,k2,mcmm,mmmc,mmcm,varmcmm,varmmmc,varmmcm),
                    (k2,1,k3,mmcm,mcmm,mmmc,varmmcm,varmcmm,varmmmc),
                    (k2,k3,1,mmcm,mmmc,mcmm,varmmcm,varmmmc,varmcmm),
                    (k3,1,k2,mmmc,mcmm,mmcm,varmmmc,varmcmm,varmmcm),
                    (k3,k2,1,mmmc,mmcm,mcmm,varmmmc,varmmcm,varmcmm) ]:
                T0, T1, T2, T3 = self._calculateT(s,t,_k1,_k2,_k3)
                T0 *= sds * d2t
                T1 *= sds * d2t
                T2 *= sds * d2t
                T3 *= sds * d2t
                # Relies on numpy array overloading += to actually update in place.
                mmm += T0.dot(gam0)
                _mcmm += T1.dot(gam1)
                _mmcm += T2.dot(gam2)
                _mmmc += T3.dot(gam3)
                varmmm += (np.abs(T0)**2).dot(vargam0)
                _varmcmm += (np.abs(T1)**2).dot(vargam1)
                _varmmcm += (np.abs(T2)**2).dot(vargam2)
                _varmmmc += (np.abs(T3)**2).dot(vargam3)

        map3 = 0.25 * np.real(mcmm + mmcm + mmmc + mmm)
        mapmapmx = 0.25 * np.imag(mcmm + mmcm - mmmc + mmm)
        mapmxmap = 0.25 * np.imag(mcmm - mmcm + mmmc + mmm)
        mxmapmap = 0.25 * np.imag(-mcmm + mmcm + mmmc + mmm)
        mxmxmap = 0.25 * np.real(mcmm + mmcm - mmmc - mmm)
        mxmapmx = 0.25 * np.real(mcmm - mmcm + mmmc - mmm)
        mapmxmx = 0.25 * np.real(-mcmm + mmcm + mmmc - mmm)
        mx3 = 0.25 * np.imag(mcmm + mmcm + mmmc - mmm)

        var = (varmcmm + varmmcm + varmmmc + varmmm) / 16.

        return map3, mapmapmx, mapmxmap, mxmapmap, mxmxmap, mxmapmx, mapmxmx, mx3, var

    def writeMap3(self, file_name, R=None, file_type=None, precision=None):
        r"""Write the aperture mass skewness based on the correlation function to the
        file, file_name.

        The output file will include the following columns:

        ==========      ==========================================================
        Column          Description
        ==========      ==========================================================
        R               The aperture radius
        Map3            An estimate of :math:`\langle M_{ap}^3\rangle(R)`
                        (cf. `calculateMap3`)
        Map2Mx          An estimate of :math:`\langle M_{ap}^2 M_\times\rangle(R)`
        MapMx2          An estimate of :math:`\langle M_{ap} M_\times^2\rangle(R)`
        Mx3             An estimate of :math:`\langle M_\times^3\rangle(R)`
        sig_map         The sqrt of the variance estimate of each of these
        ==========      ==========================================================

        Parameters:
            file_name (str):    The name of the file to write to.
            R (array):          The R values at which to calculate the statistics.
                                (default: None, which means use self.rnom)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing Map^3 from GGG correlations to %s',file_name)

        if R is None:
            R = self.rnom1d
        stats = self.calculateMap3(R)
        if precision is None:
            precision = self.config.get('precision', 4)

        gen_write(
            file_name,
            ['R','Map3','Map2Mx', 'MapMx2', 'Mx3','sig_map'],
            [ R, stats[0], stats[1], stats[4], stats[7], np.sqrt(stats[8]) ],
            precision=precision, file_type=file_type, logger=self.logger)


class GGGCrossCorrelation(BinnedCorr3):
    r"""This class handles the calculation a 3-point shear-shear-shear cross-correlation
    function.

    For 3-point cross correlations, it matters which of the two or three fields falls on
    each corner of the triangle.  E.g. is field 1 on the corner opposite d1 (the longest
    size of the triangle) or is it field 2 (or 3) there?  This is in contrast to the 2-point
    correlation where the symmetry of the situation means that it doesn't matter which point
    is identified with each field.  This makes it significantly more complicated to keep track
    of all the relevant information for a 3-point cross correlation function.

    The `GGGCorrelation` class holds a single set of :math:`\Gamma` functions describing all
    possible triangles, parameterized according to their relative side lengths ordered as
    d1 > d2 > d3.

    For a cross-correlation of two fields: G1 - G1 - G2 (i.e. the G1 field is at two of the
    corners and G2 is at one corner), then we need three sets of these :math:`\Gamma` functions
    to capture all of the triangles, since the G2 points may be opposite d1 or d2 or d3.
    For a cross-correlation of three fields: G1 - G2 - G3, we need six sets, to account for
    all of the possible permutations relative to the triangle sides.

    Therefore, this class holds 6 instances of `GGGCorrelation`, which in turn hold the
    information about triangles in each of the relevant configurations.  We name these:

    Attributes:
        g1g2g3:     Triangles where G1 is opposite d1, G2 is opposite d2, G3 is opposite d3.
        g1g3g2:     Triangles where G1 is opposite d1, G3 is opposite d2, G2 is opposite d3.
        g2g1g3:     Triangles where G2 is opposite d1, G1 is opposite d2, G3 is opposite d3.
        g2g3g1:     Triangles where G2 is opposite d1, G3 is opposite d2, G1 is opposite d3.
        g3g1g2:     Triangles where G3 is opposite d1, G1 is opposite d2, G2 is opposite d3.
        g3g2g1:     Triangles where G3 is opposite d1, G2 is opposite d2, G1 is opposite d3.

    If for instance G2 and G3 are the same field, then e.g. g1g2g3 and g1g3g2 will have
    the same values.

    Ojects of this class also hold the following attributes, which are identical in each of
    the above GGGCorrelation instances.

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
    def __init__(self, config=None, logger=None, **kwargs):
        """Initialize `GGGCrossCorrelation`.  See class doc for details.
        """
        BinnedCorr3.__init__(self, config, logger, **kwargs)

        self._ro._d1 = 3  # GData
        self._ro._d2 = 3  # GData
        self._ro._d3 = 3  # GData

        self.g1g2g3 = GGGCorrelation(config, logger, **kwargs)
        self.g1g3g2 = GGGCorrelation(config, logger, **kwargs)
        self.g2g1g3 = GGGCorrelation(config, logger, **kwargs)
        self.g2g3g1 = GGGCorrelation(config, logger, **kwargs)
        self.g3g1g2 = GGGCorrelation(config, logger, **kwargs)
        self.g3g2g1 = GGGCorrelation(config, logger, **kwargs)
        self._all = [self.g1g2g3, self.g1g3g2, self.g2g1g3, self.g2g3g1, self.g3g1g2, self.g3g2g1]

        self.logger.debug('Finished building GGGCrossCorr')

    def __eq__(self, other):
        """Return whether two `GGGCrossCorrelation` instances are equal"""
        return (isinstance(other, GGGCrossCorrelation) and
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
                self.g1g2g3 == other.g1g2g3 and
                self.g1g3g2 == other.g1g3g2 and
                self.g2g1g3 == other.g2g1g3 and
                self.g2g3g1 == other.g2g3g1 and
                self.g3g1g2 == other.g3g1g2 and
                self.g3g2g1 == other.g3g2g1)

    def copy(self):
        """Make a copy"""
        ret = GGGCrossCorrelation.__new__(GGGCrossCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, GGGCorrelation):
                ret.__dict__[key] = item.copy()
            else:
                ret.__dict__[key] = item
        # This needs to be the new list:
        ret._all = [ret.g1g2g3, ret.g1g3g2, ret.g2g1g3, ret.g2g3g1, ret.g3g1g2, ret.g3g2g1]
        return ret

    def __repr__(self):
        return 'GGGCrossCorrelation(config=%r)'%self.config

    def process_cross12(self, cat1, cat2, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        .. note::

            This only adds to the attributes g1g2g3, g2g1g3, g2g3g1, not the ones where
            3 comes before 2.  When running this via the regular `process` method, it will
            combine them at the end to make sure g1g2g3 == g1g3g2, etc. for a complete
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
            self.logger.info('Starting process GGG (1-2) cross-correlations')
        else:
            self.logger.info('Starting process GGG (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        for ggg in self._all:
            ggg._set_metric(self.metric, self.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        _lib.ProcessCross12(self.g1g2g3.corr, self.g2g1g3.corr, self.g2g3g1.corr,
                            f1.data, f2.data, self.output_dots,
                            f1._d, f2._d, self._coords,
                            self._bintype, self._metric)

    def process_cross(self, cat1, cat2, cat3, metric=None, num_threads=None):
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
            self.logger.info('Starting process GGG cross-correlations')
        else:
            self.logger.info('Starting process GGG cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        for ggg in self._all:
            ggg._set_metric(self.metric, self.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f3 = cat3.getGField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        _lib.ProcessCross3(self.g1g2g3.corr, self.g1g3g2.corr,
                           self.g2g1g3.corr, self.g2g3g1.corr,
                           self.g3g1g2.corr, self.g3g2g1.corr,
                           f1.data, f2.data, f3.data, self.output_dots,
                           f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)

    def _finalize(self):
        for ggg in self._all:
            ggg._finalize()

    def finalize(self, varg1, varg2, varg3):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulate values in each bin, so they can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing by the total weight.

        Parameters:
            varg1 (float):  The shear variance for the first field that was correlated.
            varg2 (float):  The shear variance for the second field that was correlated.
            varg3 (float):  The shear variance for the third field that was correlated.
        """
        self.g1g2g3.finalize(varg1,varg2,varg3)
        self.g1g3g2.finalize(varg1,varg3,varg2)
        self.g2g1g3.finalize(varg2,varg1,varg3)
        self.g2g3g1.finalize(varg2,varg3,varg1)
        self.g3g1g2.finalize(varg3,varg1,varg2)
        self.g3g2g1.finalize(varg3,varg2,varg1)

    @property
    def nonzero(self):
        """Return if there are any values accumulated yet.  (i.e. ntri > 0)
        """
        return any([ggg.nonzero for ggg in self._all])

    def _clear(self):
        """Clear the data vectors
        """
        for ggg in self._all:
            ggg._clear()

    def __iadd__(self, other):
        """Add a second `GGGCrossCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `GGGCrossCorrelation` objects should not have had
            `finalize` called yet.  Then, after adding them together, you should call `finalize`
            on the sum.
        """
        if not isinstance(other, GGGCrossCorrelation):
            raise TypeError("Can only add another GGGCrossCorrelation object")
        self.g1g2g3 += other.g1g2g3
        self.g1g3g2 += other.g1g3g2
        self.g2g1g3 += other.g2g1g3
        self.g2g3g1 += other.g2g3g1
        self.g3g1g2 += other.g3g1g2
        self.g3g2g1 += other.g3g2g1
        return self

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        for i, ggg in enumerate(self._all):
            ggg._sum([c._all[i] for c in others])

    def process(self, cat1, cat2, cat3=None, metric=None, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True):
        """Accumulate the cross-correlation of the points in the given Catalogs: cat1, cat2, cat3.

        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first G field.
            cat2 (Catalog):     A catalog or list of catalogs for the second G field.
            cat3 (Catalog):     A catalog or list of catalogs for the third G field.
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
                self.g1g2g3 += self.g1g3g2
                self.g2g1g3 += self.g3g1g2
                self.g2g3g1 += self.g3g2g1
                # Copy back by doing clear and +=.
                # This makes sure the coords and metric are set properly.
                self.g1g3g2.clear()
                self.g3g1g2.clear()
                self.g3g2g1.clear()
                self.g1g3g2 += self.g1g2g3
                self.g3g1g2 += self.g2g1g3
                self.g3g2g1 += self.g2g3g1

            varg1 = calculateVarG(cat1, low_mem=low_mem)
            varg2 = calculateVarG(cat2, low_mem=low_mem)
            self.logger.info("varg1 = %f: sig_g = %f",varg1,math.sqrt(varg1))
            self.logger.info("varg2 = %f: sig_g = %f",varg2,math.sqrt(varg2))
            if cat3 is None:
                varg3 = varg2
            else:
                varg3 = calculateVarG(cat3, low_mem=low_mem)
                self.logger.info("varg3 = %f: sig_g = %f",varg3,math.sqrt(varg3))
            self.finalize(varg1,varg2,varg3)

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, the concatenation of zeta.ravel() for each combination in the following
        order: g1g2g3, g1g3g2, g2g1g3, g2g3g1, g3g1g2, g3g2g1.
        """
        return np.concatenate([ggg.getStat() for ggg in self._all])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        In this case, the concatenation of getWeight() for each combination in the following
        order: g1g2g3, g1g3g2, g2g1g3, g2g3g1, g3g1g2, g3g2g1.
        """
        return np.concatenate([ggg.getWeight() for ggg in self._all])

    def write(self, file_name, file_type=None, precision=None):
        r"""Write the cross-correlation functions to the file, file_name.

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing GGG cross-correlations to %s',file_name)

        col_names = [ 'r_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv',
                      'gam0r', 'gam0i', 'gam1r', 'gam1i', 'gam2r', 'gam2i', 'gam3r', 'gam3i',
                      'sigma_gam0', 'sigma_gam1', 'sigma_gam2', 'sigma_gam3', 'weight', 'ntri' ]
        group_names = [ 'g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1' ]
        columns = [ [ ggg.rnom, ggg.u, ggg.v,
                      ggg.meand1, ggg.meanlogd1, ggg.meand2, ggg.meanlogd2,
                      ggg.meand3, ggg.meanlogd3, ggg.meanu, ggg.meanv,
                      ggg.gam0r, ggg.gam0i, ggg.gam1r, ggg.gam1i,
                      ggg.gam2r, ggg.gam2i, ggg.gam3r, ggg.gam3i,
                      np.sqrt(ggg.vargam0), np.sqrt(ggg.vargam1), np.sqrt(ggg.vargam2),
                      np.sqrt(ggg.vargam3), ggg.weight, ggg.ntri ]
                    for ggg in self._all ]

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        if precision is None:
            precision = self.config.get('precision', 4)

        gen_multi_write(
            file_name, col_names, group_names, columns,
            params=params, precision=precision, file_type=file_type, logger=self.logger)

    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `GGGCrossCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading GGG cross-correlations from %s',file_name)

        group_names = [ 'g1g2g3', 'g1g3g2', 'g2g1g3', 'g2g3g1', 'g3g1g2', 'g3g2g1' ]

        groups = gen_multi_read(
                file_name, group_names, file_type=file_type, logger=self.logger)
        s = self.logr.shape
        for (data, params), name in zip(groups, group_names):
            ggg = getattr(self, name)
            ggg._ro.rnom = data['r_nom'].reshape(s)
            ggg._ro.logr = np.log(ggg.rnom)
            ggg._ro.u = data['u_nom'].reshape(s)
            ggg._ro.v = data['v_nom'].reshape(s)
            ggg.meand1 = data['meand1'].reshape(s)
            ggg.meanlogd1 = data['meanlogd1'].reshape(s)
            ggg.meand2 = data['meand2'].reshape(s)
            ggg.meanlogd2 = data['meanlogd2'].reshape(s)
            ggg.meand3 = data['meand3'].reshape(s)
            ggg.meanlogd3 = data['meanlogd3'].reshape(s)
            ggg.meanu = data['meanu'].reshape(s)
            ggg.meanv = data['meanv'].reshape(s)
            ggg.gam0r = data['gam0r'].reshape(s)
            ggg.gam0i = data['gam0i'].reshape(s)
            ggg.gam1r = data['gam1r'].reshape(s)
            ggg.gam1i = data['gam1i'].reshape(s)
            ggg.gam2r = data['gam2r'].reshape(s)
            ggg.gam2i = data['gam2i'].reshape(s)
            ggg.gam3r = data['gam3r'].reshape(s)
            ggg.gam3i = data['gam3i'].reshape(s)
            # Read old output files without error.
            ggg.vargam0 = data['sigma_gam0'].reshape(s)**2
            ggg.vargam1 = data['sigma_gam1'].reshape(s)**2
            ggg.vargam2 = data['sigma_gam2'].reshape(s)**2
            ggg.vargam3 = data['sigma_gam3'].reshape(s)**2
            ggg.weight = data['weight'].reshape(s)
            ggg.ntri = data['ntri'].reshape(s)
            ggg.coords = params['coords'].strip()
            ggg.metric = params['metric'].strip()
            ggg._ro.sep_units = params['sep_units'].strip()
            ggg._ro.bin_type = params['bin_type'].strip()

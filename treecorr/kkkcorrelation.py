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

from . import _treecorr
from .catalog import calculateVarK
from .corr3base import Corr3
from .util import make_writer, make_reader
from .config import make_minimal_config


class KKKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-scalar-scalar correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of the CMB temperature fluctuations, where "kappa" would really be
        :math:`\Delta T`.

    See the doc string of `Corr3` for a description of how the triangles are binned.

    Ojects of this class holds the following attributes:

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
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance of :math:`\zeta`, only including the shot noise propagated into
                    the final correlation.  This does not include sample variance, so it is always
                    an underestimate of the actual variance.
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

        If you separate out the steps of the `process` command and use `process_auto` and/or
        `process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
        the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> kkk = treecorr.KKKCorrelation(config)
        >>> kkk.process(cat)              # For auto-correlation.
        >>> kkk.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> kkk.write(file_name)          # Write out to a file.
        >>> zeta = kkk.zeta               # To access zeta directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `KKKCorrelation`.  See class doc for details.
        """
        Corr3.__init__(self, config, logger=logger, **kwargs)

        shape = self.data_shape
        self.zetar = np.zeros(shape, dtype=float)
        self.weightr = np.zeros(shape, dtype=float)
        if self.bin_type == 'LogMultipole':
            self.weighti = np.zeros(shape, dtype=float)
            self.zetai = np.zeros(shape, dtype=float)
        else:
            self.weighti = np.array([])
            self.zetai = np.array([])
        self.ntri = np.zeros(shape, dtype=float)
        self._varzeta = None
        self._cov = None
        self._var_num = 0
        self.logger.debug('Finished building KKKCorr')

    @property
    def weight(self):
        if self.weighti.size:
            return self.weightr + 1j * self.weighti
        else:
            return self.weightr

    @property
    def zeta(self):
        if self.zetai.size:
            return self.zetar + 1j * self.zetai
        else:
            return self.zetar

    @property
    def corr(self):
        if self._corr is None:
            x = np.array([])
            self._corr = _treecorr.KKKCorr(
                    self._bintype,
                    self._min_sep, self._max_sep, self.nbins, self._bin_size, self.b,
                    self.angle_slop,
                    self._ro.min_u,self._ro.max_u,self._ro.nubins,self._ro.ubin_size,self.bu,
                    self._ro.min_v,self._ro.max_v,self._ro.nvbins,self._ro.vbin_size,self.bv,
                    self.xperiod, self.yperiod, self.zperiod,
                    self.zetar, self.zetai, x, x, x, x, x, x,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv,
                    self.weightr, self.weighti, self.ntri)
        return self._corr

    def __eq__(self, other):
        """Return whether two `KKKCorrelation` instances are equal"""
        return (isinstance(other, KKKCorrelation) and
                self._equal_binning(other) and
                self._equal_bin_data(other) and
                np.array_equal(self.zeta, other.zeta) and
                np.array_equal(self.varzeta, other.varzeta) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.ntri, other.ntri))

    def copy(self):
        """Make a copy"""
        ret = KKKCorrelation.__new__(KKKCorrelation)
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
        return f'KKKCorrelation({self._repr_kwargs})'

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
            self.logger.info('Starting process KKK auto-correlations')
        else:
            self.logger.info('Starting process KKK auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getKField(min_size=min_size, max_size=max_size,
                              split_method=self.split_method, brute=bool(self.brute),
                              min_top=self.min_top, max_top=self.max_top,
                              coords=self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        self.corr.processAuto(field.data, self.output_dots, self._metric)

    def process_cross12(self, cat1, cat2, *, metric=None, ordered=True, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs as part of a larger
        auto- or cross-correlation calculation.  E.g. when splitting up a large catalog into
        patches, this is appropriate to use for the cross correlation between different patches
        as part of the complete auto-correlation of the full catalog.

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
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process KKK (1-2) cross-correlations')
        else:
            self.logger.info('Starting process KKK (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getKField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        self.corr.processCross12(f1.data, f2.data, (1 if ordered else 0),
                                 self.output_dots, self._metric)

    def process_cross(self, cat1, cat2, cat3, *, metric=None, ordered=True, num_threads=None):
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
        """
        if cat1.name == '' and cat2.name == '' and cat3.name == '':
            self.logger.info('Starting process KKK cross-correlations')
        else:
            self.logger.info('Starting process KKK cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getKField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f3 = cat3.getKField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 3,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 6 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        self.corr.processCross(f1.data, f2.data, f3.data,
                               (3 if ordered is True else 1 if ordered == 1 else 0),
                               self.output_dots, self._metric)

    def _finalize(self):
        mask1 = self.weightr != 0
        mask2 = self.weightr == 0

        self.meand2[mask1] /= self.weightr[mask1]
        self.meanlogd2[mask1] /= self.weightr[mask1]
        self.meand3[mask1] /= self.weightr[mask1]
        self.meanlogd3[mask1] /= self.weightr[mask1]
        if self.bin_type != 'LogMultipole':
            self.zetar[mask1] /= self.weightr[mask1]
            self.meand1[mask1] /= self.weightr[mask1]
            self.meanlogd1[mask1] /= self.weightr[mask1]
            self.meanu[mask1] /= self.weightr[mask1]
        if self.bin_type == 'LogRUV':
            self.meanv[mask1] /= self.weightr[mask1]

        # Update the units
        self._apply_units(mask1)

        # Set to nominal when no triangles in bin.
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
                self.meanu[mask2] = 0.
                self.meand1[mask2] = 0.
                self.meanlogd1[mask2] = 0.

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

    def finalize(self, vark1, vark2, vark3):
        """Finalize the calculation of the correlation function.

        The `process_auto`, `process_cross12` and `process_cross` commands accumulate values in
        each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing by the total weight.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
            vark3 (float):  The variance of the third scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2 * vark3

        # I don't really understand why the variance is coming out 2x larger than the normal
        # formula for LogSAS.  But with just Gaussian noise, I need to multiply the numerator
        # by two to get the variance estimates to come out right.
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = np.zeros(self.data_shape)
            if self._var_num != 0:
                self._varzeta.ravel()[:] = self.cov_diag
        return self._varzeta

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
        if self.bin_type == 'LogRUV':
            self.meanv[:,:,:] = 0.
        self.zetar[:,:,:] = 0.
        self.weightr[:,:,:] = 0.
        if self.bin_type == 'LogMultipole':
            self.zetai[:,:,:] = 0.
            self.weighti[:,:,:] = 0.
        self.ntri[:,:,:] = 0.
        self._varzeta = None
        self._cov = None

    def __iadd__(self, other):
        """Add a second `KKKCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `KKKCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, KKKCorrelation):
            raise TypeError("Can only add another KKKCorrelation object")
        if not self._equal_binning(other, brief=True):
            raise ValueError("KKKCorrelation to be added is not compatible with this one.")

        if not other.nonzero: return self
        self._set_metric(other.metric, other.coords, other.coords, other.coords)
        self.meand1[:] += other.meand1[:]
        self.meanlogd1[:] += other.meanlogd1[:]
        self.meand2[:] += other.meand2[:]
        self.meanlogd2[:] += other.meanlogd2[:]
        self.meand3[:] += other.meand3[:]
        self.meanlogd3[:] += other.meanlogd3[:]
        self.meanu[:] += other.meanu[:]
        if self.bin_type == 'LogRUV':
            self.meanv[:] += other.meanv[:]
        self.zetar[:] += other.zetar[:]
        self.weightr[:] += other.weightr[:]
        if self.bin_type == 'LogMultipole':
            self.zetai[:] += other.zetai[:]
            self.weighti[:] += other.weighti[:]
        self.ntri[:] += other.ntri[:]
        return self

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c.meand1 for c in others], axis=0, out=self.meand1)
        np.sum([c.meanlogd1 for c in others], axis=0, out=self.meanlogd1)
        np.sum([c.meand2 for c in others], axis=0, out=self.meand2)
        np.sum([c.meanlogd2 for c in others], axis=0, out=self.meanlogd2)
        np.sum([c.meand3 for c in others], axis=0, out=self.meand3)
        np.sum([c.meanlogd3 for c in others], axis=0, out=self.meanlogd3)
        np.sum([c.meanu for c in others], axis=0, out=self.meanu)
        if self.bin_type == 'LogRUV':
            np.sum([c.meanv for c in others], axis=0, out=self.meanv)
        np.sum([c.zetar for c in others], axis=0, out=self.zetar)
        np.sum([c.weightr for c in others], axis=0, out=self.weightr)
        if self.bin_type == 'LogMultipole':
            np.sum([c.zetai for c in others], axis=0, out=self.zetai)
            np.sum([c.weighti for c in others], axis=0, out=self.weighti)
        np.sum([c.ntri for c in others], axis=0, out=self.ntri)

    def process(self, cat1, cat2=None, cat3=None, *, metric=None, ordered=True, num_threads=None,
                comm=None, low_mem=False, initialize=True, finalize=True,
                patch_method=None, algo=None, max_n=None):
        """Compute the 3pt correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        For cross correlations, the default behavior is to use cat1 for the first vertex (P1),
        cat2 for the second vertex (P2), and cat3 for the third vertex (P3).  If only two
        catalogs are given, vertices P2 and P3 both come from cat2.  The sides d1, d2, d3,
        used to define the binning, are taken to be opposte P1, P2, P3 respectively.

        However, if you want to accumulate triangles where objects from each catalog can take
        any position in the triangles, you can set ``ordered=False``.  In this case,
        triangles will be formed where P1, P2 and P3 can come any input catalog, so long as there
        is one from cat1, one from cat2, and one from cat3 (or two from cat2 if cat3 is None).

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first K field.
            cat2 (Catalog):     A catalog or list of catalogs for the second K field.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third K field.
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
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
            patch_method (str): Which patch method to use. (default is to use 'local' if
                                bin_type=LogMultipole, and 'global' otherwise)
            algo (str):         Which accumulation algorithm to use. (options are 'triangle' or
                                'multipole'; default is 'multipole' unless bin_type is 'LogRUV',
                                which can only use 'triangle')
            max_n (int):        If using the multpole algorithm, and this is not directly using
                                bin_type='LogMultipole', then this is the value of max_n to use
                                for the multipole part of the calculation. (default is to use
                                2pi/phi_bin_size; this value can also be given in the constructor
                                in the config dict.)
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
            corr = KKKCorrelation(config, max_n=max_n)
            corr.process(cat1, cat2, cat3,
                         metric=metric, ordered=ordered, num_threads=num_threads,
                         comm=comm, low_mem=low_mem, initialize=initialize, finalize=finalize,
                         patch_method=patch_method, algo='multipole')
            corr.toSAS(target=self)
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

        if cat2 is None:
            if cat3 is not None:
                raise ValueError("For two catalog case, use cat1,cat2, not cat1,cat3")
            self._process_all_auto(cat1, metric, num_threads, comm, low_mem, local)
        elif cat3 is None:
            self._process_all_cross12(cat1, cat2, metric, ordered, num_threads, comm, low_mem, local)
        else:
            self._process_all_cross(cat1, cat2, cat3, metric, ordered, num_threads, comm, low_mem, local)

        if finalize:
            if cat2 is None:
                vark1 = calculateVarK(cat1, low_mem=low_mem)
                vark2 = vark1
                vark3 = vark1
                self.logger.info("vark = %f: sig_k = %f",vark1,math.sqrt(vark1))
            elif cat3 is None:
                vark1 = calculateVarK(cat1, low_mem=low_mem)
                vark2 = calculateVarK(cat2, low_mem=low_mem)
                vark3 = vark2
                self.logger.info("vark1 = %f: sig_k = %f",vark1,math.sqrt(vark1))
                self.logger.info("vark2 = %f: sig_k = %f",vark2,math.sqrt(vark2))
            else:
                vark1 = calculateVarK(cat1, low_mem=low_mem)
                vark2 = calculateVarK(cat2, low_mem=low_mem)
                vark3 = calculateVarK(cat3, low_mem=low_mem)
                self.logger.info("vark1 = %f: sig_k = %f",vark1,math.sqrt(vark1))
                self.logger.info("vark2 = %f: sig_k = %f",vark2,math.sqrt(vark2))
                self.logger.info("vark3 = %f: sig_k = %f",vark3,math.sqrt(vark3))
            self.finalize(vark1,vark2,vark3)

    def toSAS(self, *, target=None, **kwargs):
        """Convert a multipole-binned correlation to the corresponding SAS binning.

        This is only valid for bin_type == LogMultipole.

        Keyword Arguments:
            target:     A target KKKCorrelation object with LogSAS binning to write to.
                        If this is not given, a new object will be created based on the
                        configuration paramters of the current object. (default: None)
            **kwargs:   Any kwargs that you want to use to configure the returned object.
                        Typically, might include min_phi, max_phi, nphi_bins, phi_bin_size.
                        The default phi binning is [0,pi] with nphi_bins = self.max_n.

        Returns:
            sas:        A KKKCorrelation object with bin_type=LogSAS containing the
                        same information as this object, but with the SAS binning.
        """
        if self.bin_type != 'LogMultipole':
            raise TypeError("toSAS is invalid for bin_type = %s"%self.bin_type)

        if target is None:
            config = self.config.copy()
            config['bin_type'] = 'LogSAS'
            max_n = config.pop('max_n')
            if 'nphi_bins' not in kwargs and 'phi_bin_size' not in kwargs:
                config['nphi_bins'] = max_n
            sas = KKKCorrelation(config, **kwargs)
        else:
            sas = target
            sas.clear()
        if not np.array_equal(sas.rnom1d, self.rnom1d):
            raise ValueError("toSAS cannot change sep parameters")

        # Copy these over
        sas.meand2[:,:,:] = self.meand2[:,:,0][:,:,None]
        sas.meanlogd2[:,:,:] = self.meanlogd2[:,:,0][:,:,None]
        sas.meand3[:,:,:] = self.meand3[:,:,0][:,:,None]
        sas.meanlogd3[:,:,:] = self.meanlogd3[:,:,0][:,:,None]
        sas._var_num = self._var_num
        sas.npatch1 = self.npatch1
        sas.npatch2 = self.npatch2
        sas.npatch3 = self.npatch3
        sas.coords = self.coords
        sas.metric = self.metric

        # Use nominal for meanphi
        sas.meanu[:] = sas.phi / sas._phi_units
        # Compute d1 from actual d2,d3 and nominal phi
        sas.meand1[:] = np.sqrt(sas.meand2**2 + sas.meand3**2
                                - 2*sas.meand2 * sas.meand3 * np.cos(sas.phi))
        sas.meanlogd1[:] = np.log(sas.meand1)

        # Z(d2,d3,phi) = 1/2pi sum_n Z_n(d2,d3) exp(i n phi)
        expiphi = np.exp(1j * self.n1d[:,None] * sas.phi1d)
        sas.weightr[:] = np.real(self.weight.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size
        sas.zetar[:] = np.real(self.zeta.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size

        # We leave zeta unnormalized in the Multipole class, so after the FT,
        # we still need to divide by weight.
        mask = sas.weightr != 0
        sas.zetar[mask] /= sas.weightr[mask]

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
            temp.zetar[:] = np.real(v.zeta.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size

            # Undo the normalization of the d arrays.
            temp.meand1 *= temp.weightr
            temp.meand2 *= temp.weightr
            temp.meand3 *= temp.weightr
            temp.meanlogd1 *= temp.weightr
            temp.meanlogd2 *= temp.weightr
            temp.meanlogd3 *= temp.weightr
            temp.meanu *= temp.weightr

            sas.results[k] = temp

        return sas

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False):
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
        n               The multipole index n (from -max_n .. max_n)
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
                        fell into each bin
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zeta_re and zeta_im.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
                        (if rrr is given)
        weight          The total weight of triangles contributing to each bin.
                        (For LogMultipole, this is split into real and imaginary parts,
                        weight_re and weight_im.)
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
        """
        self.logger.info('Writing KKK correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, name, write_patch_results)

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
        if self.bin_type == 'LogMultipole':
            col_names += ['zeta_re', 'zeta_im', 'sigma_zeta', 'weight_re', 'weight_im', 'ntri']
        else:
            col_names += ['zeta', 'sigma_zeta', 'weight', 'ntri']
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
        if self.bin_type == 'LogMultipole':
            data += [ self.zetar, self.zetai, np.sqrt(self.varzeta),
                      self.weightr, self.weighti, self.ntri ]
        else:
            data += [ self.zeta, np.sqrt(self.varzeta), self.weight, self.ntri ]
        data = [ col.flatten() for col in data ]
        return data

    @property
    def _write_params(self):
        params = make_minimal_config(self.config, Corr3._valid_params)
        # Add in a couple other things we want to preserve that aren't construction kwargs.
        params['coords'] = self.coords
        params['metric'] = self.metric
        return params

    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `KKKCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading KKK correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

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
            self.zetar = data['zeta_re'].reshape(s)
            self.zetai = data['zeta_im'].reshape(s)
            self.weightr = data['weight_re'].reshape(s)
            self.weighti = data['weight_im'].reshape(s)
        else:
            self.zetar = data['zeta'].reshape(s)
            self.weightr = data['weight'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2
        self.ntri = data['ntri'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)
        self.npatch3 = params.get('npatch3', 1)

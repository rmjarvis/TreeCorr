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

import treecorr
import numpy as np


class KKKCorrelation(treecorr.BinnedCorr3):
    r"""This class handles the calculation and storage of a 3-point kappa-kappa-kappa correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of the CMB temperature fluctuations, where "kappa" would really be
        :math:`\Delta T`.

    See the doc string of `BinnedCorr3` for a description of how the triangles are binned.

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
        zeta:       The correlation function, :math:`\zeta(r,u,v)`.
        varzeta:    The variance of :math:`\zeta`, only including the shot noise propagated into
                    the final correlation.  This does not include sample variance, so it is always
                    an underestimate of the actual variance.
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
                        in `BinnedCorr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `BinnedCorr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        """Initialize `KKKCorrelation`.  See class doc for details.
        """
        treecorr.BinnedCorr3.__init__(self, config, logger, **kwargs)

        self._ro._d1 = 2  # KData
        self._ro._d2 = 2  # KData
        self._ro._d3 = 2  # KData
        shape = self.logr.shape
        self.zeta = np.zeros(shape, dtype=float)
        self.varzeta = np.zeros(shape, dtype=float)
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
        self.logger.debug('Finished building KKKCorr')

    @property
    def corr(self):
        if self._corr is None:
            from treecorr.util import double_ptr as dp
            self._corr = treecorr._lib.BuildCorr3(
                    self._d1, self._d2, self._d3, self._bintype,
                    self._min_sep,self._max_sep,self.nbins,self._bin_size,self.b,
                    self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                    self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                    self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                    dp(self.zeta), dp(None), dp(None), dp(None),
                    dp(None), dp(None), dp(None), dp(None),
                    dp(self.meand1), dp(self.meanlogd1), dp(self.meand2), dp(self.meanlogd2),
                    dp(self.meand3), dp(self.meanlogd3), dp(self.meanu), dp(self.meanv),
                    dp(self.weight), dp(self.ntri))
        return self._corr

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if self._corr is not None:
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyCorr3(self.corr, self._d1, self._d2, self._d3, self._bintype)

    def __eq__(self, other):
        """Return whether two `KKKCorrelation` instances are equal"""
        return (isinstance(other, KKKCorrelation) and
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
                self.min_rpar == other.min_rpar and
                self.max_rpar == other.max_rpar and
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
        return 'KKKCorrelation(config=%r)'%self.config

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
            self.logger.info('Starting process KKK auto-correlations')
        else:
            self.logger.info('Starting process KKK auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getKField(min_size, max_size, self.split_method,
                              bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        treecorr._lib.ProcessAuto3(self.corr, field.data, self.output_dots,
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
            self.logger.info('Starting process KKK (1-2) cross-correlations')
        else:
            self.logger.info('Starting process KKK (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        treecorr._lib.ProcessCross12(self.corr, self.corr, self.corr,
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
            self.logger.info('Starting process KKK cross-correlations')
        else:
            self.logger.info('Starting process KKK cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f3 = cat3.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 6 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        treecorr._lib.ProcessCross3(self.corr, self.corr, self.corr,
                                    self.corr, self.corr, self.corr,
                                    f1.data, f2.data, f3.data, self.output_dots,
                                    f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.zeta[mask1] /= self.weight[mask1]
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

    def finalize(self, vark1, vark2, vark3):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing by the total weight.

        Parameters:
            vark1 (float):  The kappa variance for the first field.
            vark2 (float):  The kappa variance for the second field.
            vark3 (float):  The kappa variance for the third field.
        """
        self._finalize()
        mask1 = self.weight != 0
        mask2 = self.weight == 0
        self.varzeta[mask1] = vark1 * vark2 * vark3 / self.weight[mask1]
        self.varzeta[mask2] = 0.

    def clear(self):
        """Clear the data vectors
        """
        self.zeta[:,:,:] = 0.
        self.varzeta[:,:,:] = 0.
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
        """Add a second `KKKCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `KKKCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, KKKCorrelation):
            raise TypeError("Can only add another KKKCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.nubins == other.nubins and
                self.min_u == other.min_u and
                self.max_u == other.max_u and
                self.nvbins == other.nvbins and
                self.min_v == other.min_v and
                self.max_v == other.max_v):
            raise ValueError("KKKCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords)
        self.zeta[:] += other.zeta[:]
        self.varzeta[:] += other.varzeta[:]
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

    def process(self, cat1, cat2=None, cat3=None, metric=None, num_threads=None):
        """Accumulate the 3pt correlation of the points in the given Catalog(s).

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
            in each position in the triangle is `KKKCrossCorrelation`.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first K field.
            cat2 (Catalog):     A catalog or list of catalogs for the second K field.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third K field.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        import math
        self.clear()
        self.results.clear()
        if not isinstance(cat1,list): cat1 = cat1.get_patches()
        if cat2 is not None and not isinstance(cat2,list): cat2 = cat2.get_patches()
        if cat3 is not None and not isinstance(cat3,list): cat3 = cat3.get_patches()

        if cat2 is None:
            if cat3 is not None:
                raise ValueError("For two catalog case, use cat1,cat2, not cat1,cat3")
            vark1 = treecorr.calculateVarK(cat1)
            vark2 = vark1
            vark3 = vark1
            self.logger.info("vark = %f: sig_k = %f",vark1,math.sqrt(vark1))
            self._process_all_auto(cat1, metric, num_threads)
        elif cat3 is None:
            vark1 = treecorr.calculateVarK(cat1)
            vark2 = treecorr.calculateVarK(cat2)
            vark3 = vark2
            self.logger.info("vark1 = %f: sig_k = %f",vark1,math.sqrt(vark1))
            self.logger.info("vark2 = %f: sig_k = %f",vark2,math.sqrt(vark2))
            self._process_all_cross12(cat1, cat2, metric, num_threads)
        else:
            vark1 = treecorr.calculateVarK(cat1)
            vark2 = treecorr.calculateVarK(cat2)
            vark3 = treecorr.calculateVarK(cat3)
            self.logger.info("vark1 = %f: sig_k = %f",vark1,math.sqrt(vark1))
            self.logger.info("vark2 = %f: sig_k = %f",vark2,math.sqrt(vark2))
            self.logger.info("vark3 = %f: sig_k = %f",vark3,math.sqrt(vark3))
            self._process_all_cross(cat1, cat2, cat3, metric, num_threads)
        self.finalize(vark1,vark2,vark3)

    def write(self, file_name, file_type=None, precision=None):
        r"""Write the correlation function to the file, file_name.

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
                        fell into each bin
        zeta            The estimator of :math:`\zeta(r,u,v)`
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
        weight          The total weight of triangles contributing to each bin
        ntri            The number of triangles contributing to each bin
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
        self.logger.info('Writing KKK correlations to %s',file_name)

        col_names = [ 'r_nom', 'u_nom', 'v_nom',
                      'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv',
                      'zeta', 'sigma_zeta', 'weight', 'ntri' ]
        columns = [ self.rnom, self.u, self.v,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv,
                    self.zeta, np.sqrt(self.varzeta), self.weight, self.ntri ]

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        if precision is None:
            precision = self.config.get('precision', 4)

        treecorr.util.gen_write(
            file_name, col_names, columns,
            params=params, precision=precision, file_type=file_type, logger=self.logger)

    def read(self, file_name, file_type=None):
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

        data, params = treecorr.util.gen_read(file_name, file_type=file_type, logger=self.logger)
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
        self.zeta = data['zeta'].reshape(s)
        self.varzeta = data['sigma_zeta'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.ntri = data['ntri'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()


class KKKCrossCorrelation(treecorr.BinnedCorr3):
    r"""This class handles the calculation a 3-point kappa-kappa-kappa cross-correlation
    function.

    For 3-point cross correlations, it matters which of the two or three fields falls on
    each corner of the triangle.  E.g. is field 1 on the corner opposite d1 (the longest
    size of the triangle) or is it field 2 (or 3) there?  This is in contrast to the 2-point
    correlation where the symmetry of the situation means that it doesn't matter which point
    is identified with each field.  This makes it significantly more complicated to keep track
    of all the relevant information for a 3-point cross correlation function.

    The `KKKCorrelation` class holds a single :math:`\zeta` functions describing all
    possible triangles, parameterized according to their relative side lengths ordered as
    d1 > d2 > d3.

    For a cross-correlation of two fields: K1 - K1 - K2 (i.e. the K1 field is at two of the
    corners and K2 is at one corner), then we need three these :math:`\zeta` functions
    to capture all of the triangles, since the K2 points may be opposite d1 or d2 or d3.
    For a cross-correlation of three fields: K1 - K2 - K3, we need six sets, to account for
    all of the possible permutations relative to the triangle sides.

    Therefore, this class holds 6 instances of `KKKCorrelation`, which in turn hold the
    information about triangles in each of the relevant configurations.  We name these:

    Attribute:
        k1k2k3:     Triangles where K1 is opposite d1, K2 is opposite d2, K3 is opposite d3.
        k1k3k2:     Triangles where K1 is opposite d1, K3 is opposite d2, K2 is opposite d3.
        k2k1k3:     Triangles where K2 is opposite d1, K1 is opposite d2, K3 is opposite d3.
        k2k3k1:     Triangles where K2 is opposite d1, K3 is opposite d2, K1 is opposite d3.
        k3k1k2:     Triangles where K3 is opposite d1, K1 is opposite d2, K2 is opposite d3.
        k3k2k1:     Triangles where K3 is opposite d1, K2 is opposite d2, K1 is opposite d3.

    If for instance K2 and K3 are the same field, then e.g. k1k2k3 and k1k3k2 will have
    the same values.

    Ojects of this class also hold the following attributes, which are identical in each of
    the above KKKCorrelation instances.

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
        """Initialize `KKKCrossCorrelation`.  See class doc for details.
        """
        treecorr.BinnedCorr3.__init__(self, config, logger, **kwargs)

        self._ro._d1 = 2  # KData
        self._ro._d2 = 2  # KData
        self._ro._d3 = 2  # KData

        self.k1k2k3 = KKKCorrelation(config, logger, **kwargs)
        self.k1k3k2 = KKKCorrelation(config, logger, **kwargs)
        self.k2k1k3 = KKKCorrelation(config, logger, **kwargs)
        self.k2k3k1 = KKKCorrelation(config, logger, **kwargs)
        self.k3k1k2 = KKKCorrelation(config, logger, **kwargs)
        self.k3k2k1 = KKKCorrelation(config, logger, **kwargs)

        self.logger.debug('Finished building KKKCrossCorr')

    def __eq__(self, other):
        """Return whether two `KKKCrossCorrelation` instances are equal"""
        return (isinstance(other, KKKCrossCorrelation) and
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
                self.min_rpar == other.min_rpar and
                self.max_rpar == other.max_rpar and
                self.xperiod == other.xperiod and
                self.yperiod == other.yperiod and
                self.zperiod == other.zperiod and
                self.k1k2k3 == other.k1k2k3 and
                self.k1k3k2 == other.k1k3k2 and
                self.k2k1k3 == other.k2k1k3 and
                self.k2k3k1 == other.k2k3k1 and
                self.k3k1k2 == other.k3k1k2 and
                self.k3k2k1 == other.k3k2k1)

    def copy(self):
        """Make a copy"""
        ret = KKKCrossCorrelation.__new__(KKKCrossCorrelation)
        for key, item in self.__dict__.items():
            if isinstance(item, KKKCorrelation):
                ret.__dict__[key] = item.copy()
            else:
                ret.__dict__[key] = item
        return ret

    def __repr__(self):
        return 'KKKCrossCorrelation(config=%r)'%self.config

    def process_cross12(self, cat1, cat2, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where one of the
        points in each triangle come from the first catalog, and two come from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        .. note::

            This only adds to the attributes k1k2k3, k2k1k3, k2k3k1, not the ones where
            3 comes before 2.  When running this via the regular `process` method, it will
            combine them at the end to make sure k1k2k3 == k1k3k2, etc. for a complete
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
            self.logger.info('Starting process KKK (1-2) cross-correlations')
        else:
            self.logger.info('Starting process KKK (1-2) cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self.k1k2k3._set_metric(self.metric, self.coords)
        self.k2k1k3._set_metric(self.metric, self.coords)
        self.k2k3k1._set_metric(self.metric, self.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        # Note: all 3 correlation objects are the same.  Thus, all triangles will be placed
        # into self.corr, whichever way the three catalogs are permuted for each triangle.
        treecorr._lib.ProcessCross12(self.k1k2k3.corr, self.k2k1k3.corr, self.k2k3k1.corr,
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
            self.logger.info('Starting process KKK cross-correlations')
        else:
            self.logger.info('Starting process KKK cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)
        self.k1k2k3._set_metric(self.metric, self.coords)
        self.k1k3k2._set_metric(self.metric, self.coords)
        self.k2k1k3._set_metric(self.metric, self.coords)
        self.k2k3k1._set_metric(self.metric, self.coords)
        self.k3k1k2._set_metric(self.metric, self.coords)
        self.k3k2k1._set_metric(self.metric, self.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f3 = cat3.getKField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        treecorr._lib.ProcessCross3(self.k1k2k3.corr, self.k1k3k2.corr,
                                    self.k2k1k3.corr, self.k2k3k1.corr,
                                    self.k3k1k2.corr, self.k3k2k1.corr,
                                    f1.data, f2.data, f3.data, self.output_dots,
                                    f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)

    def finalize(self, vark1, vark2, vark3):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulate values in each bin, so they can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing by the total weight.

        Parameters:
            vark1 (float):  The kappa variance for the first field that was correlated.
            vark2 (float):  The kappa variance for the second field that was correlated.
            vark3 (float):  The kappa variance for the third field that was correlated.
        """
        self.k1k2k3.finalize(vark1,vark2,vark3)
        self.k1k3k2.finalize(vark1,vark3,vark2)
        self.k2k1k3.finalize(vark2,vark1,vark3)
        self.k2k3k1.finalize(vark2,vark3,vark1)
        self.k3k1k2.finalize(vark3,vark1,vark2)
        self.k3k2k1.finalize(vark3,vark2,vark1)

    def clear(self):
        """Clear the data vectors
        """
        self.k1k2k3.clear()
        self.k1k3k2.clear()
        self.k2k1k3.clear()
        self.k2k3k1.clear()
        self.k3k1k2.clear()
        self.k3k2k1.clear()

    def __iadd__(self, other):
        """Add a second `KKKCrossCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `KKKCrossCorrelation` objects should not have had
            `finalize` called yet.  Then, after adding them together, you should call `finalize`
            on the sum.
        """
        if not isinstance(other, KKKCrossCorrelation):
            raise TypeError("Can only add another KKKCrossCorrelation object")
        self.k1k2k3 += other.k1k2k3
        self.k1k3k2 += other.k1k3k2
        self.k2k1k3 += other.k2k1k3
        self.k2k3k1 += other.k2k3k1
        self.k3k1k2 += other.k3k1k2
        self.k3k2k1 += other.k3k2k1
        return self

    def process(self, cat1, cat2, cat3=None, metric=None, num_threads=None):
        """Accumulate the cross-correlation of the points in the given Catalogs: cat1, cat2, cat3.

        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking one corner of the triangles, and the second taking two corners.
        - If 3 arguments are given, then compute a three-way cross-correlation function.

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first K field.
            cat2 (Catalog):     A catalog or list of catalogs for the second K field.
            cat3 (Catalog):     A catalog or list of catalogs for the third K field.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        import math
        self.clear()
        if not isinstance(cat1,list): cat1 = cat1.get_patches()
        if not isinstance(cat2,list): cat2 = cat2.get_patches()
        if cat3 is not None and not isinstance(cat3,list): cat3 = cat3.get_patches()

        vark1 = treecorr.calculateVarK(cat1)
        vark2 = treecorr.calculateVarK(cat2)
        self.logger.info("vark1 = %f: sig_k = %f",vark1,math.sqrt(vark1))
        self.logger.info("vark2 = %f: sig_k = %f",vark2,math.sqrt(vark2))

        if cat3 is None:
            vark3 = vark2
            self._process_all_cross12(cat1, cat2, metric, num_threads)
            # The k1k2k3 and k1k3k2 are equivalent, but process_all_cross12 only added things to
            # one or the other of these (only k1k2k3 if no lists are involved).  So add them
            # together and copy, so they are equal.
            # Likewise the other pairs that are symmetric between 2,3.
            if np.any(self.k1k3k2.ntri != 0):
                self.k1k2k3 += self.k1k3k2
            if np.any(self.k3k1k2.ntri != 0):
                self.k2k1k3 += self.k3k1k2
            if np.any(self.k3k2k1.ntri != 0):
                self.k2k3k1 += self.k3k2k1
            # Copy back by doing clear and +=.
            # This makes sure the coords and metric are set properly.
            self.k1k3k2.clear()
            self.k3k1k2.clear()
            self.k3k2k1.clear()
            self.k1k3k2 += self.k1k2k3
            self.k3k1k2 += self.k2k1k3
            self.k3k2k1 += self.k2k3k1
        else:
            vark3 = treecorr.calculateVarK(cat3)
            self.logger.info("vark3 = %f: sig_k = %f",vark3,math.sqrt(vark3))
            self._process_all_cross(cat1, cat2, cat3, metric, num_threads)
        self.finalize(vark1,vark2,vark3)

    def write(self, file_name, file_type=None, precision=None):
        r"""Write the correlation function to the file, file_name.

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing KKK cross-correlations to %s',file_name)

        col_names = [ 'r_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv',
                      'zeta', 'sigma_zeta', 'weight', 'ntri' ]
        group_names = [ 'k1k2k3', 'k1k3k2', 'k2k1k3', 'k2k3k1', 'k3k1k2', 'k3k2k1' ]
        columns = [
                    [ kkk.rnom, kkk.u, kkk.v,
                      kkk.meand1, kkk.meanlogd1, kkk.meand2, kkk.meanlogd2,
                      kkk.meand3, kkk.meanlogd3, kkk.meanu, kkk.meanv,
                      kkk.zeta, np.sqrt(kkk.varzeta), kkk.weight, kkk.ntri ]
                    for kkk in [ self.k1k2k3, self.k1k3k2, self.k2k1k3,
                                 self.k2k3k1, self.k3k1k2, self.k3k2k1 ]
                  ]

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        if precision is None:
            precision = self.config.get('precision', 4)

        treecorr.util.gen_multi_write(
            file_name, col_names, group_names, columns,
            params=params, precision=precision, file_type=file_type, logger=self.logger)

    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `KKKCrossCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading KKK cross-correlations from %s',file_name)

        group_names = [ 'k1k2k3', 'k1k3k2', 'k2k1k3', 'k2k3k1', 'k3k1k2', 'k3k2k1' ]

        groups = treecorr.util.gen_multi_read(
                file_name, group_names, file_type=file_type, logger=self.logger)
        s = self.logr.shape
        for (data, params), name in zip(groups, group_names):
            kkk = getattr(self, name)
            kkk._ro.rnom = data['r_nom'].reshape(s)
            kkk._ro.logr = np.log(kkk.rnom)
            kkk._ro.u = data['u_nom'].reshape(s)
            kkk._ro.v = data['v_nom'].reshape(s)
            kkk.meand1 = data['meand1'].reshape(s)
            kkk.meanlogd1 = data['meanlogd1'].reshape(s)
            kkk.meand2 = data['meand2'].reshape(s)
            kkk.meanlogd2 = data['meanlogd2'].reshape(s)
            kkk.meand3 = data['meand3'].reshape(s)
            kkk.meanlogd3 = data['meanlogd3'].reshape(s)
            kkk.meanu = data['meanu'].reshape(s)
            kkk.meanv = data['meanv'].reshape(s)
            kkk.zeta = data['zeta'].reshape(s)
            kkk.varzeta = data['sigma_zeta'].reshape(s)**2
            kkk.weight = data['weight'].reshape(s)
            kkk.ntri = data['ntri'].reshape(s)
            kkk.coords = params['coords'].strip()
            kkk.metric = params['metric'].strip()
            kkk._ro.sep_units = params['sep_units'].strip()
            kkk._ro.bin_type = params['bin_type'].strip()

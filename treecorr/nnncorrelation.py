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


class NNNCorrelation(treecorr.BinnedCorr3):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    See the doc string of `BinnedCorr3` for a description of how the triangles are binned.

    Ojects of this class holds the following attributes:

    Attributes:
        logr:      The nominal center of the bin in log(r) (the natural logarithm of r).
        nbins:     The number of bins in logr where r = d2
        bin_size:  The size of the bins in logr
        min_sep:   The minimum separation being considered
        max_sep:   The maximum separation being considered
        nubins:    The number of bins in u where u = d3/d2
        ubin_size: The size of the bins in u
        min_u:     The minimum u being considered
        max_u:     The maximum u being considered
        nvbins:    The number of bins in v where v = +-(d1-d2)/d3
        vbin_size: The size of the bins in v
        min_v:     The minimum v being considered
        max_v:     The maximum v being considered
        logr1d:    The nominal centers of the nbins bins in log(r).
        u1d:       The nominal centers of the nubins bins in u.
        v1d:       The nominal centers of the nvbins bins in v.

    In addition, the following attributes are numpy arrays whose shape is (nbins, nubins, nvbins):

    Attributes:
        logr:      The nominal center of the bin in log(r).
        rnom:      The nominal center of the bin converted to regular distance.
                   i.e. r = exp(logr).
        u:         The nominal center of the bin in u.
        v:         The nominal center of the bin in v.
        meand1:    The (weighted) mean value of d1 for the triangles in each bin.
        meanlogd1: The mean value of log(d1) for the triangles in each bin.
        meand2:    The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        meanlogd2: The mean value of log(d2) for the triangles in each bin.
        meand2:    The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd2: The mean value of log(d3) for the triangles in each bin.
        meanu:     The mean value of u for the triangles in each bin.
        meanv:     The mean value of v for the triangles in each bin.
        weight:    The total weight in each bin.
        ntri:      The number of triangles in each bin.
        tot:       The total number of triangles processed, which is used to normalize
                   the randoms if they have a different number of triangles.

    If **sep_units** are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.  Note however, that if you separate out the steps of the
    `process` command and use `process_auto` and/or `process_cross`, then the units will not be
    applied to **meanr** or **meanlogr** until the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> nnn = treecorr.NNNCorrelation(config)
        >>> nnn.process(cat)         # For auto-correlation.
        >>> nnn.process(cat1,cat2,cat3)   # For cross-correlation.
        >>> rrr.process...           # Likewise for random-random correlations
        >>> drr.process...           # If desired, also do data-random correlations
        >>> rdr.process...           # ... in all three
        >>> rrd.process...           # ... permutations
        >>> rdd.process...           # Also with two data and one random
        >>> drd.process...           # ... in all three
        >>> ddr.process...           # ... permutations
        >>> nn.write(file_name,rrr,drr,...)  # Write out to a file.
        >>> zeta,varzeta = nn.calculateZeta(rrr,drr,...)  # Or get the 3pt function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries in addition to those listed
                        in `BinnedCorr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    See the documentation for `BinnedCorr3` for the list of other allowed kwargs, which may be
    passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr3.__init__(self, config, logger, **kwargs)

        self._d1 = 1  # NData
        self._d2 = 1  # NData
        self._d3 = 1  # NData
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
        self._build_corr()
        self.logger.debug('Finished building NNNCorr')

    def _build_corr(self):
        from treecorr.util import double_ptr as dp
        self.corr = treecorr._lib.BuildCorr3(
                self._d1, self._d2, self._d3, self._bintype,
                self._min_sep,self._max_sep,self.nbins,self._bin_size,self.b,
                self.min_u,self.max_u,self.nubins,self.ubin_size,self.bu,
                self.min_v,self.max_v,self.nvbins,self.vbin_size,self.bv,
                self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                dp(None), dp(None), dp(None), dp(None),
                dp(None), dp(None), dp(None), dp(None),
                dp(self.meand1), dp(self.meanlogd1), dp(self.meand2), dp(self.meanlogd2),
                dp(self.meand3), dp(self.meanlogd3), dp(self.meanu), dp(self.meanv),
                dp(self.weight), dp(self.ntri));

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        # In case __init__ failed to get that far
        if hasattr(self,'corr'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyCorr3(self.corr, self._d1, self._d2, self._d3, self._bintype)

    def __eq__(self, other):
        """Return whether two NNNCorrelations are equal"""
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
                self.min_rpar == other.min_rpar and
                self.xperiod == other.xperiod and
                self.yperiod == other.yperiod and
                self.zperiod == other.zperiod and
                self.max_rpar == other.max_rpar and
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
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['corr']
        del d['logger']  # Oh well.  This is just lost in the copy.  Can't be pickled.
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._build_corr()
        self.logger = treecorr.config.setup_logger(
                treecorr.config.get(self.config,'verbose',int,1),
                self.config.get('log_file',None))

    def __repr__(self):
        return 'NNNCorrelation(config=%r)'%self.config

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
            self.logger.info('Starting process NNN auto-correlations')
        else:
            self.logger.info('Starting process NNN auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)

        self._set_num_threads(num_threads)

        min_size, max_size = self._get_minmax_size()

        field = cat.getNField(min_size, max_size, self.split_method,
                              bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        treecorr._lib.ProcessAuto3(self.corr, field.data, self.output_dots,
                                   field._d, self._coords, self._bintype, self._metric)
        self.tot += (1./6.) * cat.sumw**3

    def process_cross21(self, cat1, cat2, metric=None, num_threads=None):
        """Process two catalogs, accumulating the 3pt cross-correlation, where two of the
        points in each triangle come from the first catalog, and one from the second.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meand1, meanlogd1, etc.

        .. warning::

            This is not implemented yet.

        Parameters:
            cat1 (Catalog):     The first catalog to process
            cat2 (Catalog):     The second catalog to process
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        raise NotImplementedError("No partial cross NNN yet.")


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
            self.logger.info('Starting process NNN cross-correlations')
        else:
            self.logger.info('Starting process NNN cross-correlations for cats %s, %s, %s.',
                             cat1.name, cat2.name, cat3.name)

        self._set_metric(metric, cat1.coords, cat2.coords, cat3.coords)

        self._set_num_threads(num_threads)

        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f2 = cat2.getNField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)
        f3 = cat3.getNField(min_size, max_size, self.split_method,
                            bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        treecorr._lib.ProcessCross3(self.corr, f1.data, f2.data, f3.data, self.output_dots,
                                    f1._d, f2._d, f3._d, self._coords, self._bintype, self._metric)
        self.tot += cat1.sumw * cat2.sumw * cat3.sumw / 6.0


    def finalize(self):
        """Finalize the calculation of meand1, meanlogd1, etc.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanlogr, meanu, meanv by dividing by the total weight.
        """
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


    def clear(self):
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

    def __iadd__(self, other):
        """Add a second NNNCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        `process_auto` and/or `process_cross`, and they should not have had `finalize` called yet.
        Then, after adding them together, you should call `finalize` on the sum.
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

        self._set_metric(other.metric, other.coords)
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
        self.tot += other.tot
        return self


    def process(self, cat1, cat2=None, cat3=None, metric=None, num_threads=None):
        """Accumulate the number of triangles of points between cat1, cat2, and cat3.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function with the
          first catalog taking two corners of the triangles. (Not implemented yet.)
        - If 3 arguments are given, then compute a cross-correlation function.

        All arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Note: For a correlation of multiple catalogs, it matters which corner of the
        triangle comes from which catalog.  The final accumulation will have
        d1 > d2 > d3 where d1 is between two points in cat2,cat3; d2 is between
        points in cat1,cat3; and d3 is between points in cat1,cat2.  To accumulate
        all the possible triangles between three catalogs, you should call this
        multiple times with the different catalogs in different positions.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first N field.
            cat2 (Catalog):     A catalog or list of catalogs for the second N field, if any.
                                (default: None)
            cat3 (Catalog):     A catalog or list of catalogs for the third N field, if any.
                                (default: None)
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if cat3 is not None and not isinstance(cat3,list): cat3 = [cat3]

        if cat2 is None and cat3 is None:
            self._process_all_auto(cat1, metric, num_threads)
        elif (cat2 is None) != (cat3 is None):
            raise NotImplementedError("No partial cross NNN yet.")
        else:
            assert cat2 is not None and cat3 is not None
            self._process_all_cross(cat1,cat2,cat3, metric, num_threads)
        self.finalize()

    def calculateZeta(self, rrr, drr=None, rdr=None, rrd=None,
                      ddr=None, drd=None, rdd=None):
        """Calculate the 3pt function given another 3pt function of random
        points using the same mask, and possibly cross correlations of the data and random.

        There are two possible formulae that are currently supported.

        1. The simplest formula to use is :math:`\\zeta^\\prime = (DDD-RRR)/RRR`.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.
           However, note that in this case, the return value is not normally called :math:`\\zeta`.
           Rather, this is an estimator of

           .. math::
               \\zeta^\\prime(d_1,d_2,d_3) = \\zeta(d_1,d_2,d_3) + \\xi(d_1) + \\xi(d_2) + \\xi(d_3)

           where :math:`\\xi` is the two-point correlation function for each leg of the triangle.
           You would typically want to calculate that separately and subtract off the
           two-point contributions.

        2. For auto-correlations, a better formula is
           :math:`\\zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR`.
           In this case, DDR, etc. calculate the triangles where two points come from either
           the data or the randoms.  In the case of DDR for instance, points x1 and x2 on the
           triangle have data and x3 uses randoms, where points x1, x2, x3 are opposite sides
           d1, d2, d3 with d1 > d2 > d3 as usual; RDR has randoms at 1,3 and data at 2, etc.

        - If only rrr is provided, the first formula will be used.
        - If all of rrr, drr, rdr, rrd, ddr, drd, rdd are provided then the second will be used.

        Parameters:
            rrr (NNNCorrelation):   The auto-correlation of the random field (RRR)
            drr (NNNCorrelation):   DRR if desired. (default: None)
            rdr (NNNCorrelation):   RDR if desired. (default: None)
            rrd (NNNCorrelation):   RRD if desired. (default: None)
            ddr (NNNCorrelation):   DDR if desired. (default: None)
            drd (NNNCorrelation):   DRD if desired. (default: None)
            rdd (NNNCorrelation):   RDD if desired. (default: None)

        Returns:
            Tuple containing

                - zeta = array of :math:`\\zeta(d_1,d_2,d_3)`
                - varzeta = array of variance estimates of :math:`\\zeta(d_1,d_2,d_3)`
        """
        # Each random ntri value needs to be rescaled by the ratio of total possible tri.
        if rrr.tot == 0:
            raise ValueError("rrr has tot=0.")

        if (drr is not None or rdr is not None or rrd is not None or
            ddr is not None or drd is not None or rdd is not None):
            if (drr is None or rdr is None or rrd is None or
                ddr is None or drd is None or rdd is None):
                raise TypeError("Must provide all 6 combinations rdr, drr, etc.")

        rrrw = self.tot / rrr.tot
        if drr is None:
            zeta = (self.weight - rrr.weight * rrrw)
        else:
            if rrd.tot == 0:
                raise ValueError("rrd has tot=0.")
            if ddr.tot == 0:
                raise ValueError("ddr has tot=0.")
            if rdr.tot == 0:
                raise ValueError("rdr has tot=0.")
            if drr.tot == 0:
                raise ValueError("drr has tot=0.")
            if drd.tot == 0:
                raise ValueError("drd has tot=0.")
            if rdd.tot == 0:
                raise ValueError("rdd has tot=0.")
            rrdw = self.tot / rrd.tot
            ddrw = self.tot / ddr.tot
            rdrw = self.tot / rdr.tot
            drrw = self.tot / drr.tot
            drdw = self.tot / drd.tot
            rddw = self.tot / rdd.tot
            zeta = (self.weight + rrd.weight * rrdw + rdr.weight * rdrw + drr.weight * drrw
                    - ddr.weight * ddrw - drd.weight * drdw - rdd.weight * rddw - rrr.weight * rrrw)
        if np.any(rrr.weight == 0):
            self.logger.warning("Warning: Some bins for the randoms had no triangles.")
        mask1 = rrr.weight != 0
        mask2 = rrr.weight == 0
        zeta[mask1] /= (rrr.weight[mask1] * rrrw)
        zeta[mask2] = 0

        varzeta = np.zeros_like(rrr.weight)
        varzeta[mask1] = 1./ (rrr.weight[mask1] * rrrw)

        return zeta, varzeta


    def write(self, file_name, rrr=None, drr=None, rdr=None, rrd=None,
              ddr=None, drd=None, rdd=None, file_type=None, precision=None):
        """Write the correlation function to the file, file_name.

        Normally, at least rrr should be provided, but if this is None, then only the
        basic accumulated number of triangles are output (along with the columns parametrizing
        the size and shape of the triangles).

        If at least rrr is given, then it will output an estimate of the final 3pt correlation
        function, :math:`\\zeta`. There are two possible formulae that are currently supported.

        1. The simplest formula to use is :math:`\\zeta^\\prime = (DDD-RRR)/RRR`.
           In this case, only rrr needs to be given, the NNNCorrelation of a random field.
           However, note that in this case, the return value is not what is normally called
           :math:`\\zeta`.  Rather, this is an estimator of

           .. math::
               \\zeta^\\prime(d_1,d_2,d_3) = \\zeta(d_1,d_2,d_3) + \\xi(d_1) + \\xi(d_2) + \\xi(d_3)

           where :math:`\\xi` is the two-point correlation function for each leg of the triangle.
           You would typically want to calculate that separately and subtract off the
           two-point contributions.

        2. For auto-correlations, a better formula is
           :math:`\\zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR`.
           In this case, DDR, etc. calculate the triangles where two points come from either
           the data or the randoms.  In the case of DDR for instance, points x1 and x2 on the
           triangle have data and x3 uses randoms, where points x1, x2, x3 are opposite sides
           d1, d2, d3 with d1 > d2 > d3 as usual; RDR has randoms at 1,3 and data at 2, etc.
           For this case, all the combinations rrr, drr, rdr, rrd, ddr, drd, rdd must be
           provided.

        The output file will include the following columns:

        ==========      ===============================================================
        Column          Description
        ==========      ===============================================================
        r_nom           The nominal center of the bin in r = d2 where d1 > d2 > d3
        u_nom           The nominal center of the bin in u = d3/d2
        v_nom           The nominal center of the bin in v = +-(d1-d2)/d3
        meand1          The mean value <d1> of triangles that fell into each bin
        meanlogd1       The mean value <logd1> of triangles that fell into each bin
        meand2          The mean value <d2> of triangles that fell into each bin
        meanlogd2       The mean value <logd2> of triangles that fell into each bin
        meand3          The mean value <d3> of triangles that fell into each bin
        meanlogd3       The mean value <logd3> of triangles that fell into each bin
        meanu           The mean value <u> of triangles that fell into each bin
        meanv           The mean value <v> of triangles that fell into each bin
        zeta            The estimator zeta (if rrr is given)
        sigma_zeta      The sqrt of the variance estimate of zeta (if rrr is given)
        DDD             The total weight of DDD triangles in each bin
        RRR             The total weight of RRR triangles in each bin (if rrr is given)
        DRR             The total weight of DRR triangles in each bin (if drr is given)
        RDR             The total weight of RDR triangles in each bin (if rdr is given)
        RRD             The total weight of RRD triangles in each bin (if rrd is given)
        DDR             The total weight of DDR triangles in each bin (if ddr is given)
        DRD             The total weight of DRD triangles in each bin (if drd is given)
        RDD             The total weight of RDD triangles in each bin (if rdd is given)
        ntri            The number of triangles contributing to each bin
        ==========      ===============================================================

        If **sep_units** was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):        The name of the file to write to.
            rrr (NNNCorrelation):   The auto-correlation of the random field (RRR)
            drr (NNNCorrelation):   DRR if desired. (default: None)
            rdr (NNNCorrelation):   RDR if desired. (default: None)
            rrd (NNNCorrelation):   RRD if desired. (default: None)
            ddr (NNNCorrelation):   DDR if desired. (default: None)
            drd (NNNCorrelation):   DRD if desired. (default: None)
            rdd (NNNCorrelation):   RDD if desired. (default: None)
            file_type (str):        The type of file to write ('ASCII' or 'FITS').
                                    (default: determine the type automatically from the extension
                                    of file_name.)
            precision (int):        For ASCII output catalogs, the desired precision. (default: 4;
                                    this value can also be given in the constructor in the config
                                    dict.)
        """
        self.logger.info('Writing NNN correlations to %s',file_name)

        col_names = [ 'r_nom', 'u_nom', 'v_nom', 'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                      'meand3', 'meanlogd3', 'meanu', 'meanv' ]
        columns = [ self.rnom, self.u, self.v,
                    self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                    self.meand3, self.meanlogd3, self.meanu, self.meanv ]
        if rrr is None:
            if (drr is not None or rdr is not None or rrd is not None or
                ddr is not None or drd is not None or rdd is not None):
                raise TypeError("rrr must be provided if other combinations are not None")
            col_names += [ 'DDD', 'ntri' ]
            columns += [ self.weight, self.ntri ]
        else:
            # This will check for other invalid combinations of rrr, drr, etc.
            zeta, varzeta = self.calculateZeta(rrr,drr,rdr,rrd,ddr,drd,rdd)

            col_names += [ 'zeta','sigma_zeta','DDD','RRR' ]
            columns += [ zeta, np.sqrt(varzeta),
                         self.weight, rrr.weight * (self.tot/rrr.tot) ]

            if drr is not None:
                col_names += ['DRR','RDR','RRD','DDR','DRD','RDD']
                columns += [ drr.weight * (self.tot/drr.tot), rdr.weight * (self.tot/rdr.tot),
                             rrd.weight * (self.tot/rrd.tot), ddr.weight * (self.tot/ddr.tot),
                             drd.weight * (self.tot/drd.tot), rdd.weight * (self.tot/rdd.tot) ]
            col_names += [ 'ntri' ]
            columns += [ self.ntri ]

        if precision is None:
            precision = self.config.get('precision', 4)

        params = { 'tot' : self.tot, 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        treecorr.util.gen_write(
            file_name, col_names, columns,
            params=params, precision=precision, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The NNNCorrelation object should be constructed with the same configuration
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NNN correlations from %s',file_name)

        data, params = treecorr.util.gen_read(file_name, file_type=file_type, logger=self.logger)
        s = self.logr.shape
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self.rnom = data['R_nom'].reshape(s)
        else:
            self.rnom = data['r_nom'].reshape(s)
        self.logr = np.log(self.rnom)
        self.u = data['u_nom'].reshape(s)
        self.v = data['v_nom'].reshape(s)
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
        self.tot = params['tot']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.sep_units = params['sep_units'].strip()
        self.bin_type = params['bin_type'].strip()
        self._build_corr()



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
.. module:: nnncorrelation
"""

import numpy as np

from . import _treecorr
from .corr3base import Corr3
from .util import make_writer, make_reader, lazy_property
from .config import make_minimal_config


class NNNCorrelation(Corr3):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    See the doc string of `Corr3` for a description of how the triangles can be binned.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:      The number of bins in logr where r = d2
        bin_size:   The size of the bins in logr
        min_sep:    The minimum separation being considered
        max_sep:    The maximum separation being considered
        logr1d:     The nominal centers of the nbins bins in log(r).
        tot:        The total number of triangles processed, which is used to normalize
                    the randoms if they have a different number of triangles.

    If the bin_type is LogRUV, then it will have these attributes:

    Attributes:
        nubins:     The number of bins in u where u = d3/d2
        ubin_size:  The size of the bins in u
        min_u:      The minimum u being considered
        max_u:      The maximum u being considered
        nvbins:     The number of bins in v where v = +-(d1-d2)/d3
        vbin_size:  The size of the bins in v
        min_v:      The minimum v being considered
        max_v:      The maximum v being considered
        u1d:        The nominal centers of the nubins bins in u.
        v1d:        The nominal centers of the nvbins bins in v.

    If the bin_type is LogSAS, then it will have these attributes:

    Attributes:
        nphi_bins:  The number of bins in phi where v = +-(d1-d2)/d3.
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
        meanu:      The mean value of u for the triangles in each bin.
        meanv:      The mean value of v for the triangles in each bin.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).

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
        meanlogd1:  The mean value of log(d1) for the triangles in each bin.
        meand2:     The (weighted) mean value of d2 for the triangles in each bin.
        meanlogd2:  The mean value of log(d2) for the triangles in each bin.
        meand3:     The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd3:  The mean value of log(d3) for the triangles in each bin.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr3.process` command and use `process_auto` and/or
        `Corr3.process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr``
        until the `finalize` function is called.

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
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NNNCorrelation'
    _letter1 = 'N'
    _letter2 = 'N'
    _letter3 = 'N'
    _letters = 'NNN'
    _builder = _treecorr.NNNCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = lambda *args, **kwargs: None
    _calculateVar3 = lambda *args, **kwargs: None
    _sig1 = None
    _sig2 = None
    _sig3 = None
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NNNCorrelation`.  See class doc for details.
        """
        Corr3.__init__(self, config, logger=logger, **kwargs)

        self.tot = 0.
        self._rrr_weight = None
        self._rrr = None
        self._drr = None
        self._rdd = None
        self._write_rrr = None
        self._write_drr = None
        self._write_rdd = None
        self._write_patch_results = False
        self.zeta = None
        x = np.array([])
        self._z1 = self._z2 = self._z3 = self._z4 = x
        self._z5 = self._z6 = self._z7 = self._z8 = x
        self.logger.debug('Finished building NNNCorr')

    def copy(self):
        """Make a copy"""
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rrr is not None and self._rrr is not True:
            ret._rrr = self._rrr.copy()
        if self._drr is not None and self._drr is not True:
            ret._drr = self._drr.copy()
        if self._rdd is not None and self._rdd is not True:
            ret._rdd = self._rdd.copy()
        return ret

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
        ret.weightr = self._zero_array
        if self.bin_type == 'LogMultipole':
            ret.weighti = self._zero_array
        else:
            ret.weighti = np.array([])
        ret.ntri = self._zero_array
        ret.tot = tot
        ret._corr = None
        ret._rrr = ret._drr = ret._rdd = None
        ret._write_rrr = ret._write_drr = ret._write_rdd = None
        ret._write_patch_results = False
        ret._cov = None
        ret._logger_name = None
        # This override is really the main advantage of using this:
        setattr(ret, '_nonzero', False)
        return ret

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
        super()._process_auto(cat, metric, num_threads)
        self.tot += (1./6.) * cat.sumw**3

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
        super()._process_cross12(cat1, cat2, metric, ordered, num_threads)
        self.tot += 0.5 * cat1.sumw * cat2.sumw**2

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
        super().process_cross(cat1, cat2, cat3, metric=metric, ordered=ordered,
                              num_threads=num_threads)
        self.tot += cat1.sumw * cat2.sumw * cat3.sumw

    def finalize(self):
        """Finalize the calculation of meand1, meanlogd1, etc.

        The `process_auto`, `process_cross12` and `process_cross` commands accumulate values in
        each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meand1, meand2, etc. by dividing by the total weight.
        """
        self._finalize()

    @lazy_property
    def _nonzero(self):
        # The lazy version when we can be sure the object isn't going to accumulate any more.
        return self.nonzero

    def _clear(self):
        super()._clear()
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
            if self.bin_type == 'LogRUV':
                np.sum([c.meanv for c in others], axis=0, out=self.meanv)
            np.sum([c.ntri for c in others], axis=0, out=self.ntri)
            np.sum([c.weightr for c in others], axis=0, out=self.weightr)
            if self.bin_type == 'LogMultipole':
                np.sum([c.weighti for c in others], axis=0, out=self.weighti)
        self.tot = tot

    def _add_tot(self, ijk, c1, c2, c3):
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
        self.results[ijk] = self._zero_copy(tot)

    def __iadd__(self, other):
        """Add a second Correlation object's data to this one.

        .. note::

            For this to make sense, both objects should not have had `finalize` called yet.
            Then, after adding them together, you should call `finalize` on the sum.
        """
        super().__iadd__(other)
        self.tot += other.tot
        return self

    def _mean_weight(self):
        mean_ntri = np.mean(self.ntri)
        return 1 if mean_ntri == 0 else np.mean(self.weight)/mean_ntri

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        This raises a RuntimeError if calculateZeta has not been run yet.
        """
        if self.zeta is None:
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

    def toSAS(self, *, target=None, **kwargs):
        """Convert a multipole-binned correlation to the corresponding SAS binning.

        This is only valid for bin_type == LogMultipole.

        Keyword Arguments:
            target:     A target NNNCorrelation object with LogSAS binning to write to.
                        If this is not given, a new object will be created based on the
                        configuration paramters of the current object. (default: None)
            **kwargs:   Any kwargs that you want to use to configure the returned object.
                        Typically, might include min_phi, max_phi, nphi_bins, phi_bin_size.
                        The default phi binning is [0,pi] with nphi_bins = self.max_n.

        Returns:
            An NNNCorrelation object with bin_type=LogSAS containing the
            same information as this object, but with the SAS binning.
        """
        sas = super().toSAS(target=target, **kwargs)

        sas.tot = self.tot
        for k,v in self.results.items():
            sas.results[k].tot = v.tot

        return sas

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

        .. note::

            This method is not valid for bin_type='LogMultipole'. I don't think there is
            a straightforward way to go directly from the multipole expoansion of DDD and
            RRR to Zeta.  Normally one would instead convert both to LogSAS binning
            (cf. `toSAS`) and then call `calculateZeta` with those.

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

        if self.bin_type == 'LogMultipole':
            raise TypeError("calculateZeta is not valid for LogMultipole binning.")

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
        self._rrr_weight = np.abs(rrr.weight) * rrrf

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
        self._cov = self.estimate_cov(self.var_method)
        self.varzeta = self.cov_diag.reshape(self.zeta.shape)
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
            if self._rdd.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
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

    def write(self, file_name, *, rrr=None, drr=None, rdd=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
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
        weight_re       The real part of the complex weight.
        weight_im       The imaginary part of the complex weight.
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
        zeta            The estimator :math:`\zeta` (if rrr is given, or zeta was
                        already computed)
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
            write_cov (bool):       Whether to write the covariance matrix as well. (default: False)
        """
        self.logger.info('Writing NNN correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        self._write_rrr = rrr
        self._write_drr = drr
        self._write_rdd = rdd
        self._write_patch_results = write_patch_results
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov, zero_tot=True)
            if write_patch_results:
                # Also write out rr, dr, rd, so covariances can be computed on round trip.
                rrr = rrr or self._rrr
                drr = drr or self._drr
                rdd = rdd or self._rdd
                if rrr:
                    rrr._write(writer, '_rrr', write_patch_results, zero_tot=True)
                if drr:
                    drr._write(writer, '_drr', write_patch_results, zero_tot=True)
                if rdd:
                    rdd._write(writer, '_rdd', write_patch_results, zero_tot=True)
        self._write_rrr = None
        self._write_drr = None
        self._write_rdd = None
        self._write_patch_results = False

    @property
    def _write_col_names(self):
        rrr = self._write_rrr
        drr = self._write_drr
        rdd = self._write_rdd
        if self.bin_type == 'LogRUV':
            col_names = ['r_nom', 'u_nom', 'v_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanu', 'meanv']
        elif self.bin_type == 'LogSAS':
            col_names = ['d2_nom', 'd3_nom', 'phi_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanphi']
        else:
            # LogMultipole
            col_names = ['d2_nom', 'd3_nom', 'n',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3']
        if rrr is None:
            if self.zeta is not None:
                col_names += [ 'zeta', 'sigma_zeta' ]
            if self.weighti.size:
                col_names += [ 'weight_re', 'weight_im' ]
            col_names += [ 'DDD', 'ntri' ]
        else:
            col_names += [ 'zeta','sigma_zeta','DDD','RRR' ]
            if drr is not None:
                col_names += ['DRR','RDD']
            col_names += [ 'ntri' ]
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
            # LogMultipole
            data = [ self.d2nom, self.d3nom, self.n,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3 ]
        rrr = self._write_rrr
        drr = self._write_drr
        rdd = self._write_rdd
        if rrr is None:
            if drr is not None or rdd is not None:
                raise TypeError("rrr must be provided if other combinations are not None")
            if self.zeta is not None:
                data += [ self.zeta, np.sqrt(self.varzeta) ]
            if self.weighti.size:
                data += [ self.weightr, self.weighti ]
                weight = np.abs(self.weight)
            else:
                weight = self.weightr
            data += [ weight, self.ntri ]
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
        params = super()._write_params
        params['tot'] = self.tot
        if self._write_patch_results:
            params['_rrr'] = bool(self._rrr)
            params['_drr'] = bool(self._drr)
            params['_rdd'] = bool(self._rdd)
        return params

    @classmethod
    def from_file(cls, file_name, *, file_type=None, logger=None, rng=None):
        """Create an NNNCorrelation instance from an output file.

        This should be a file that was written by TreeCorr.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII', 'FITS', or 'HDF').  (default: determine
                                the type automatically from the extension of file_name.)
            logger (Logger):    If desired, a logger object to use for logging. (default: None)
            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                                random number generation. (default: None)

        Returns:
            An NNNCorrelation object, constructed from the information in the file.
        """
        if logger:
            logger.info('Building NNNCorrelation from %s',file_name)
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
            corr.logger.info('Reading NNN correlations from %s',file_name)
            corr._do_read(reader, name=name, params=params)
        return corr

    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS or HDF5 file, so
        there is no loss of information.

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
            self._do_read(reader)

    def _do_read(self, reader, name=None, params=None):
        self._read(reader, name, params)
        if self._rrr:
            rrr = self.copy()
            rrr._read(reader, name='_rrr')
            self._rrr = rrr
        if self._drr:
            drr = self.copy()
            drr._read(reader, name='_drr')
            self._drr = drr
        if self._rdd:
            rdd = self.copy()
            rdd._read(reader, name='_rdd')
            self._rdd = rdd

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        if 'zeta' in data.dtype.names:
            self.zeta = data['zeta'].reshape(s)
            self.varzeta = data['sigma_zeta'].reshape(s)**2
        if self.bin_type != 'LogMultipole':
            self.weightr = data['DDD'].reshape(s)
        self.tot = params['tot']
        # Note: "or None" turns False -> None
        self._rrr = params.get('_rrr', None) or None
        self._drr = params.get('_drr', None) or None
        self._rdd = params.get('_rdd', None) or None

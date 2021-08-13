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
.. module:: nncorrelation
"""

import numpy as np

from . import _lib, _ffi
from .binnedcorr2 import BinnedCorr2
from .util import double_ptr as dp
from .util import gen_read, gen_write, lazy_property

class NNCorrelation(BinnedCorr2):
    r"""This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:     The number of bins in logr
        bin_size:  The size of the bins in logr
        min_sep:   The minimum separation being considered
        max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:
        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        meanr:      The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:   The mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        tot:        The total number of pairs processed, which is used to normalize
                    the randoms if they have a different number of pairs.

    If `calculateXi` has been called, then the following will also be available:

    Attributes:
        xi:         The correlation function, :math:`\xi(r)`
        varxi:      An estimate of the variance of :math:`\xi`
        cov:        An estimate of the full covariance matrix.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_auto` and/or
        `process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
        the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> nn = treecorr.NNCorrelation(config)
        >>> nn.process(cat)         # For auto-correlation.
        >>> nn.process(cat1,cat2)   # For cross-correlation.
        >>> rr.process...           # Likewise for random-random correlations
        >>> dr.process...           # If desired, also do data-random correlations
        >>> rd.process...           # For cross-correlations, also do the reverse.
        >>> nn.write(file_name,rr,dr,rd)         # Write out to a file.
        >>> xi,varxi = nn.calculateXi(rr,dr,rd)  # Or get the correlation function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `BinnedCorr2`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `BinnedCorr2` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        """Initialize `NNCorrelation`.  See class doc for details.
        """
        BinnedCorr2.__init__(self, config, logger, **kwargs)

        self._ro._d1 = 1  # NData
        self._ro._d2 = 1  # NData
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self.tot = 0.
        self._rr_weight = None  # Marker that calculateXi hasn't been called yet.
        self._rr = None
        self._dr = None
        self._rd = None
        self.logger.debug('Finished building NNCorr')

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _lib.BuildCorr2(
                    self._d1, self._d2, self._bintype,
                    self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                    self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                    dp(None), dp(None), dp(None), dp(None),
                    dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs))
        return self._corr

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if self._corr is not None:
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        """Return whether two `NNCorrelation` instances are equal"""
        return (isinstance(other, NNCorrelation) and
                self.nbins == other.nbins and
                self.bin_size == other.bin_size and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.sep_units == other.sep_units and
                self.coords == other.coords and
                self.bin_type == other.bin_type and
                self.bin_slop == other.bin_slop and
                self.min_rpar == other.min_rpar and
                self.max_rpar == other.max_rpar and
                self.xperiod == other.xperiod and
                self.yperiod == other.yperiod and
                self.zperiod == other.zperiod and
                self.tot == other.tot and
                np.array_equal(self.meanr, other.meanr) and
                np.array_equal(self.meanlogr, other.meanlogr) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.npairs, other.npairs))

    def copy(self):
        """Make a copy"""
        ret = NNCorrelation.__new__(NNCorrelation)
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
        if self._rd is not None:
            ret._rd = self._rd.copy()
        if self._dr is not None:
            ret._dr = self._dr.copy()
        if self._rr is not None:
            ret._rr = self._rr.copy()
        return ret

    @lazy_property
    def _zero_array(self):
        # An array of all zeros with the same shape as self.weight (and other data arrays)
        z = np.zeros_like(self.weight)
        z.flags.writeable=False  # Just to make sure we get an error if we try to change it.
        return z

    def _zero_copy(self, tot):
        # A minimal "copy" with zero for the weight array, and the given value for tot.
        ret = NNCorrelation.__new__(NNCorrelation)
        ret._ro = self._ro
        ret.config = self.config
        ret.npairs = self._zero_array
        ret.weight = self._zero_array
        ret.tot = tot
        ret._corr = None
        # This override is really the main advantage of using this:
        setattr(ret, '_nonzero', False)
        return ret

    def __repr__(self):
        return 'NNCorrelation(config=%r)'%self.config

    def process_auto(self, cat, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meanr, meanlogr.

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
            self.logger.info('Starting process NN auto-correlations')
        else:
            self.logger.info('Starting process NN auto-correlations for cat %s.', cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getNField(min_size, max_size, self.split_method,
                              bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        _lib.ProcessAuto2(self.corr, field.data, self.output_dots,
                          field._d, self._coords, self._bintype, self._metric)
        self.tot += 0.5 * cat.sumw**2


    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation of meanr, meanlogr.

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
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NN cross-correlations')
        else:
            self.logger.info('Starting process NN cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute == 1,
                            self.min_top, self.max_top, self.coords)
        f2 = cat2.getNField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute == 2,
                            self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        _lib.ProcessCross2(self.corr, f1.data, f2.data, self.output_dots,
                           f1._d, f2._d, self._coords, self._bintype, self._metric)
        self.tot += cat1.sumw*cat2.sumw


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the sums into the bins, but does not finalize the calculation.
        After calling this function as often as desired, the `finalize` command will
        finish the calculation.

        .. warning::

            .. deprecated:: 4.1

                This function is deprecated and slated to be removed.
                If you have a need for it, please open an issue to describe your use case.

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
        import warnings
        warnings.warn("The process_pairwise function is slated to be removed in a future version. "+
                      "If you are actually using this function usefully, please "+
                      "open an issue to describe your use case.", FutureWarning)
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NN pairwise-correlations')
        else:
            self.logger.info('Starting process NN pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField()
        f2 = cat2.getNSimpleField()

        _lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                         f1._d, f2._d, self._coords, self._bintype, self._metric)
        self.tot += (cat1.sumw+cat2.sumw)/2.

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]

    def finalize(self):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanr, meanlogr by dividing by the total weight.
        """
        self._finalize()

    @lazy_property
    def _nonzero(self):
        # The lazy version when we can be sure the object isn't going to accumulate any more.
        return self.nonzero

    def _clear(self):
        """Clear the data vectors
        """
        self.meanr.ravel()[:] = 0.
        self.meanlogr.ravel()[:] = 0.
        self.weight.ravel()[:] = 0.
        self.npairs.ravel()[:] = 0.
        self.tot = 0.

    def __iadd__(self, other):
        """Add a second `NNCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `NNCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, NNCorrelation):
            raise TypeError("Can only add another NNCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NNCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords, other.coords)
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        self.tot += other.tot
        return self

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
            np.sum([c.meanr for c in others], axis=0, out=self.meanr)
            np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
            np.sum([c.weight for c in others], axis=0, out=self.weight)
            np.sum([c.npairs for c in others], axis=0, out=self.npairs)
        self.tot = tot

    def _add_tot(self, i, j, c1, c2):
        # When storing results from a patch-based run, tot needs to be accumulated even if
        # the total weight being accumulated comes out to be zero.
        # This only applies to NNCorrelation.  For the other ones, this is a no op.
        tot = c1.sumw * c2.sumw
        self.tot += tot
        # We also have to keep all pairs in the results dict, otherwise the tot calculation
        # gets messed up.  We need to accumulate the tot value of all pairs, even if
        # the resulting weight is zero.  But use a minimal copy with just the necessary fields
        # to save some time.
        self.results[(i,j)] = self._zero_copy(tot)

    def process(self, cat1, cat2=None, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True):
        """Compute the correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first N field.
            cat2 (Catalog):     A catalog or list of catalogs for the second N field, if any.
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
                                `BinnedCorr2.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
        """
        if initialize:
            self.clear()

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if cat2 is not None and not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)

        if cat2 is None or len(cat2) == 0:
            self._process_all_auto(cat1, metric, num_threads, comm, low_mem)
        else:
            self._process_all_cross(cat1, cat2, metric, num_threads, comm, low_mem)

        if finalize:
            self.finalize()

    def _mean_weight(self):
        mean_np = np.mean(self.npairs)
        return 1 if mean_np == 0 else np.mean(self.weight)/mean_np

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        This raises a RuntimeError if calculateXi has not been run yet.
        """
        if self._rr is None:
            raise RuntimeError("You need to call calculateXi before calling estimate_cov.")
        return self.xi.ravel()

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        This is the weight array corresponding to `getStat`.  In this case, it is the denominator
        RR from the calculation done by calculateXi().
        """
        if self._rr_weight is not None:
            return self._rr_weight.ravel()
        else:
            return self.tot

    def calculateXi(self, rr, dr=None, rd=None):
        r"""Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross correlations of the data and random.

        The rr value is the `NNCorrelation` function for random points.
        For a signal that involves a cross correlations, there should be two random
        cross-correlations: data-random and random-data, given as dr and rd.

        - If dr is None, the simple correlation function :math:`\xi = (DD/RR - 1)` is used.
        - if dr is given and rd is None, then :math:`\xi = (DD - 2DR + RR)/RR` is used.
        - If dr and rd are both given, then :math:`\xi = (DD - DR - RD + RR)/RR` is used.

        where DD is the data NN correlation function, which is the current object.

        .. note::

            The default method for estimating the variance is 'shot', which only includes the
            shot noise propagated into the final correlation.  This does not include sample
            variance, so it is always an underestimate of the actual variance.  To get better
            estimates, you need to set ``var_method`` to something else and use patches in the
            input catalog(s).  cf. `Covariance Estimates`.

        After calling this method, you can use the `BinnedCorr2.estimate_cov` method or use this
        correlation object in the `estimate_multi_cov` function.  Also, the calculated xi and
        varxi returned from this function will be available as attributes.

        Parameters:
            rr (NNCorrelation):     The auto-correlation of the random field (RR)
            dr (NNCorrelation):     The cross-correlation of the data with randoms (DR), if
                                    desired, in which case the Landy-Szalay estimator will be
                                    calculated.  (default: None)
            rd (NNCorrelation):     The cross-correlation of the randoms with data (RD), if
                                    desired. (default: None, which means use rd=dr)

        Returns:
            Tuple containing:

            - xi = array of :math:`\xi(r)`
            - varxi = an estimate of the variance of :math:`\xi(r)`
        """
        # Each random weight value needs to be rescaled by the ratio of total possible pairs.
        if rr.tot == 0:
            raise ValueError("rr has tot=0.")

        # rrf is the factor to scale rr weights to get something commensurate to the dd density.
        rrf = self.tot / rr.tot

        # Likewise for the other two potential randoms:
        if dr is not None:
            if dr.tot == 0:
                raise ValueError("dr has tot=0.")
            drf = self.tot / dr.tot
        if rd is not None:
            if rd.tot == 0:
                raise ValueError("rd has tot=0.")
            rdf = self.tot / rd.tot

        # Calculate xi based on which randoms are provided.
        denom = rr.weight * rrf
        if dr is None and rd is None:
            self.xi = self.weight - denom
        elif rd is not None and dr is None:
            self.xi = self.weight - 2.*rd.weight * rdf + denom
        elif dr is not None and rd is None:
            self.xi = self.weight - 2.*dr.weight * drf + denom
        else:
            self.xi = self.weight - rd.weight * rdf - dr.weight * drf + denom

        # Divide by RR in all cases.
        if np.any(rr.weight == 0):
            self.logger.warning("Warning: Some bins for the randoms had no pairs.")
            denom[rr.weight==0] = 1.  # guard against division by 0.
        self.xi /= denom

        # Set up necessary info for estimate_cov

        # First the bits needed for shot noise covariance:
        ddw = self._mean_weight()
        rrw = rr._mean_weight()
        if dr is not None:
            drw = dr._mean_weight()
        if rd is not None:
            rdw = rd._mean_weight()

        # Note: The use of varxi_factor for the shot noise varxi is semi-empirical.
        #       It gives the increase in the variance over the case where RR >> DD.
        #       I don't have a good derivation that this is the right factor to apply
        #       when the random catalog is not >> larger than the data.
        #       When I tried to derive this from first principles, I get the below formula,
        #       but without the **2.  So I'm not sure why this factor needs to be squared.
        #       It seems at least plausible that I missed something in the derivation that
        #       leads to this getting squared, but I can't really justify it.
        #       But it's also possible that this is wrong...
        #       Anyway, it seems to give good results compared to the empirical variance.
        #       cf. test_nn.py:test_varxi
        if dr is None and rd is None:
            varxi_factor = 1 + rrf*rrw/ddw
        elif rd is not None and dr is None:
            varxi_factor = 1 + 2*rdf*rdw/ddw + rrf*rrw/ddw
        elif dr is not None and rd is None:
            varxi_factor = 1 + 2*drf*drw/ddw + rrf*rrw/ddw
        else:
            varxi_factor = 1 + drf*drw/ddw + rdf*rdw/ddw + rrf*rrw/ddw
        self._var_num = ddw * varxi_factor**2
        self._rr_weight = rr.weight * rrf

        # Now set up the bits needed for patch-based covariance
        self._rr = rr
        self._dr = dr
        self._rd = rd

        if len(self.results) > 0:
            # Check that rr,dr,rd use the same patches as dd
            if rr.npatch1 != 1 and rr.npatch2 != 1:
                if rr.npatch1 != self.npatch1 or rr.npatch2 != self.npatch2:
                    raise RuntimeError("If using patches, RR must be run with the same patches "
                                       "as DD")

            if dr is not None and (len(dr.results) == 0 or dr.npatch1 != self.npatch1 or
                                   dr.npatch2 not in (self.npatch2, 1)):
                raise RuntimeError("DR must be run with the same patches as DD")
            if rd is not None and (len(rd.results) == 0 or rd.npatch2 != self.npatch2 or
                                   rd.npatch1 not in (self.npatch1, 1)):
                raise RuntimeError("RD must be run with the same patches as DD")

            # If there are any rr,rd,dr patch pairs that aren't in results (because dr is a cross
            # correlation, and dd,rr may be auto-correlations, or because the d catalogs has some
            # patches with no items), then we need to add some dummy results to make sure all the
            # right pairs are computed when we make the vectors for the covariance matrix.
            add_ij = set()
            if rr.npatch1 != 1 and rr.npatch2 != 1:
                for ij in rr.results:
                    if ij not in self.results:
                        add_ij.add(ij)

            if dr is not None and dr.npatch2 != 1:
                for ij in dr.results:
                    if ij not in self.results:
                        add_ij.add(ij)

            if rd is not None and rd.npatch1 != 1:
                for ij in rd.results:
                    if ij not in self.results:
                        add_ij.add(ij)

            if len(add_ij) > 0:
                for ij in add_ij:
                    self.results[ij] = self._zero_copy(0)
                self.__dict__.pop('_ok',None)  # If it was already made, it will need to be redone.

        # Now that it's all set up, calculate the covariance and set varxi to the diagonal.
        self.cov = self.estimate_cov(self.var_method)
        self.varxi = self.cov.diagonal()
        return self.xi, self.varxi

    def _calculate_xi_from_pairs(self, pairs):
        self._sum([self.results[ij] for ij in pairs])
        self._finalize()
        if self._rr is None:
            return
        dd = self.weight
        if len(self._rr.results) > 0:
            # This is the usual case.  R has patches just like D.
            # Calculate rr and rrf in the normal way based on the same pairs as used for DD.
            pairs1 = [ij for ij in pairs if self._rr._ok[ij[0],ij[1]]]
            self._rr._sum([self._rr.results[ij] for ij in pairs1])
            dd_tot = self.tot
        else:
            # In this case, R was not run with patches.
            # This is not necessarily much worse in practice it turns out.
            # We just need to scale RR down by the relative area.
            # The approximation we'll use is that tot in the auto-correlations is
            # proportional to area**2.
            # So the sum of tot**0.5 when i==j gives an estimate of the fraction of the total area.
            area_frac = np.sum([self.results[ij].tot**0.5 for ij in pairs if ij[0] == ij[1]])
            area_frac /= np.sum([cij.tot**0.5 for ij,cij in self.results.items() if ij[0] == ij[1]])
            # First figure out the original total for all DD that had the same footprint as RR.
            dd_tot = np.sum([self.results[ij].tot for ij in self.results])
            # The rrf we want will be a factor of area_frac smaller than the original
            # dd_tot/rr_tot.  We can effect this by multiplying the full dd_tot by area_frac
            # and use that value normally below.  (Also for drf and rdf.)
            dd_tot *= area_frac

        rr = self._rr.weight
        rrf = dd_tot / self._rr.tot

        if self._dr is not None:
            if self._dr.npatch2 == 1:
                # If r doesn't have patches, then convert all (i,i) pairs to (i,0).
                pairs2 = [(ij[0],0) for ij in pairs if ij[0] == ij[1]]
            else:
                pairs2 = [ij for ij in pairs if self._dr._ok[ij[0],ij[1]]]
            self._dr._sum([self._dr.results[ij] for ij in pairs2])
            dr = self._dr.weight
            drf = dd_tot / self._dr.tot
        if self._rd is not None:
            if self._rd.npatch1 == 1:
                # If r doesn't have patches, then convert all (i,i) pairs to (0,i).
                pairs3 = [(0,ij[1]) for ij in pairs if ij[0] == ij[1]]
            else:
                pairs3 = [ij for ij in pairs if self._rd._ok[ij[0],ij[1]]]
            self._rd._sum([self._rd.results[ij] for ij in pairs3])
            rd = self._rd.weight
            rdf = dd_tot / self._rd.tot
        denom = rr * rrf
        if self._dr is None and self._rd is None:
            xi = dd - denom
        elif self._rd is not None and self._dr is None:
            xi = dd - 2.*rd * rdf + denom
        elif self._dr is not None and self._rd is None:
            xi = dd - 2.*dr * drf + denom
        else:
            xi = dd - rd * rdf - dr * drf + denom
        denom[denom == 0] = 1  # Guard against division by zero.
        self.xi = xi / denom
        self._rr_weight = denom

    def write(self, file_name, rr=None, dr=None, rd=None, file_type=None, precision=None):
        r"""Write the correlation function to the file, file_name.

        rr is the `NNCorrelation` function for random points.
        If dr is None, the simple correlation function :math:`\xi = (DD - RR)/RR` is used.
        if dr is given and rd is None, then :math:`\xi = (DD - 2DR + RR)/RR` is used.
        If dr and rd are both given, then :math:`\xi = (DD - DR - RD + RR)/RR` is used.

        Normally, at least rr should be provided, but if this is also None, then only the
        basic accumulated number of pairs are output (along with the separation columns).

        The output file will include the following columns:

        ==========      ==========================================================
        Column          Description
        ==========      ==========================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r\rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r)\rangle` of pairs that
                        fell into each bin
        xi              The estimator :math:`\xi` (if rr is given, or calculateXi
                        has been called)
        sigma_xi        The sqrt of the variance estimate of xi (if rr is given
                        or calculateXi has been called)
        DD              The total weight of pairs in each bin.
        RR              The total weight of RR pairs in each bin (if rr is given)
        DR              The total weight of DR pairs in each bin (if dr is given)
        RD              The total weight of RD pairs in each bin (if rd is given)
        npairs          The total number of pairs in each bin
        ==========      ==========================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):        The name of the file to write to.
            rr (NNCorrelation):     The auto-correlation of the random field (RR)
            dr (NNCorrelation):     The cross-correlation of the data with randoms (DR), if
                                    desired. (default: None)
            rd (NNCorrelation):     The cross-correlation of the randoms with data (RD), if
                                    desired. (default: None, which means use rd=dr)
            file_type (str):        The type of file to write ('ASCII' or 'FITS').
                                    (default: determine the type automatically from the extension
                                    of file_name.)
            precision (int):        For ASCII output catalogs, the desired precision. (default: 4;
                                    this value can also be given in the constructor in the config
                                    dict.)
        """
        self.logger.info('Writing NN correlations to %s',file_name)

        col_names = [ 'r_nom','meanr','meanlogr' ]
        columns = [ self.rnom, self.meanr, self.meanlogr ]
        if rr is None:
            if hasattr(self, 'xi'):
                col_names += [ 'xi','sigma_xi' ]
                columns += [ self.xi, np.sqrt(self.varxi) ]
            col_names += [ 'DD', 'npairs' ]
            columns += [ self.weight, self.npairs ]
            if dr is not None:
                raise TypeError("rr must be provided if dr is not None")
            if rd is not None:
                raise TypeError("rr must be provided if rd is not None")
        else:
            xi, varxi = self.calculateXi(rr,dr,rd)

            col_names += [ 'xi','sigma_xi','DD','RR' ]
            columns += [ xi, np.sqrt(varxi),
                         self.weight, rr.weight * (self.tot/rr.tot) ]

            if dr is not None and rd is not None:
                col_names += ['DR','RD']
                columns += [ dr.weight * (self.tot/dr.tot), rd.weight * (self.tot/rd.tot) ]
            elif dr is not None or rd is not None:
                if dr is None: dr = rd
                col_names += ['DR']
                columns += [ dr.weight * (self.tot/dr.tot) ]
            col_names += [ 'npairs' ]
            columns += [ self.npairs ]

        if precision is None:
            precision = self.config.get('precision', 4)

        params = { 'tot' : self.tot, 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        gen_write(
            file_name, col_names, columns, params=params,
            precision=precision, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `NNCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):   The name of the file to read in.
            file_type (str):   The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NN correlations from %s',file_name)

        data, params = gen_read(file_name, file_type=file_type, logger=self.logger)
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self._ro.rnom = data['R_nom']
            self.meanr = data['meanR']
            self.meanlogr = data['meanlogR']
        else:
            self._ro.rnom = data['r_nom']
            self.meanr = data['meanr']
            self.meanlogr = data['meanlogr']
        self._ro.logr = np.log(self.rnom)
        self.weight = data['DD']
        self.npairs = data['npairs']
        self.tot = params['tot']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()
        if 'xi' in data.dtype.names:
            self.xi = data['xi']
            self.varxi = data['sigma_xi']**2

    def calculateNapSq(self, rr, R=None, dr=None, rd=None, m2_uform=None):
        r"""Calculate the corrollary to the aperture mass statistics for counts.

        .. math::

            \langle N_{ap}^2 \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{2R^2}
            \left [ T_+\left(\frac{r}{R}\right) \xi(r) \right] \\

        The ``m2_uform`` parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If ``m2_uform`` is 'Crittenden':

        .. math::

            U(r) &= \frac{1}{2\pi} (1-r^2) \exp(-r^2/2) \\
            T_+(s) &= \frac{s^4 - 16s^2 + 32}{128} \exp(-s^2/4) \\
            rmax &= \infty

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If ``m2_uform`` is 'Schneider':

        .. math::

            U(r) &= \frac{9}{\pi} (1-r^2) (1/3-r^2) \\
            T_+(s) &= \frac{12}{5\pi} (2-15s^2) \arccos(s/2) \\
            &\qquad + \frac{1}{100\pi} s \sqrt{4-s^2} (120 + 2320s^2 - 754s^4 + 132s^6 - 9s^8) \\
            rmax &= 2R

        cf. Schneider, et al (2002): A&A, 389, 729

        This is used by `NGCorrelation.writeNorm`.  See that function and also
        `GGCorrelation.calculateMapSq` for more details.

        Parameters:
            rr (NNCorrelation): The auto-correlation of the random field (RR)
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            dr (NNCorrelation): The cross-correlation of the data with randoms (DR), if
                                desired. (default: None)
            rd (NNCorrelation): The cross-correlation of the randoms with data (RD), if
                                desired. (default: None, which means use rd=dr)
            m2_uform (str):     Which form to use for the aperture mass.  (default: 'Crittenden';
                                this value can also be given in the constructor in the config dict.)

        Returns:
            Tuple containing

                - nsq = array of :math:`\langle N_{ap}^2 \rangle(R)`
                - varnsq = array of variance estimates of this value
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform', 'Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")
        if R is None:
            R = self.rnom

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        s = np.outer(1./R, self.meanr)
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = np.exp(-ssq/4.)
            Tp = (32. + ssq*(-16. + ssq)) / 128. * exp_factor
        else:
            Tp = np.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tp[s<2.] = 12./(5.*np.pi) * (2.-15.*ssqa) * np.arccos(sa/2.)
            Tp[s<2.] += 1./(100.*np.pi) * sa * np.sqrt(4.-ssqa) * (
                        120. + ssqa*(2320. + ssqa*(-754. + ssqa*(132. - 9.*ssqa))))
        Tp *= ssq

        xi, varxi = self.calculateXi(rr,dr,rd)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxi = Tp.dot(xi)
        nsq = Tpxi * self.bin_size
        varnsq = (Tp**2).dot(varxi) * self.bin_size**2

        return nsq, varnsq

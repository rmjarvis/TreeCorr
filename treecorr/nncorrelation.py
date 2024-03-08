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
.. module:: nncorrelation
"""

import numpy as np

from . import _treecorr
from .corr2base import Corr2
from .util import make_writer, make_reader, lazy_property
from .config import make_minimal_config


class NNCorrelation(Corr2):
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

        If you separate out the steps of the `Corr2.process` command and use `process_auto`
        and/or `Corr2.process_cross`, then the units will not be applied to ``meanr`` or
        ``meanlogr`` until the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> nn = treecorr.NNCorrelation(config)
        >>> nn.process(cat)         # For auto-correlation.
        >>> nn.process(cat1,cat2)   # For cross-correlation.
        >>> rr.process...           # Likewise for random-random correlations
        >>> dr.process...           # If desired, also do data-random correlations
        >>> rd.process...           # For cross-correlations, also do the reverse.
        >>> nn.write(file_name,rr=rr,dr=dr,rd=rd)         # Write out to a file.
        >>> xi,varxi = nn.calculateXi(rr=rr,dr=dr,rd=rd)  # Or get correlation function directly.

    Parameters:
        config (dict):  A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries besides those listed
                        in `Corr2`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr2` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'NNCorrelation'
    _letter1 = 'N'
    _letter2 = 'N'
    _letters = 'NN'
    _builder = _treecorr.NNCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = lambda *args, **kwargs: None
    _sig1 = None
    _sig2 = None
    # The angles are not important for accuracy of NN correlations.
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NNCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)
        self.tot = 0.
        self._xi1 = self._xi2 = self._xi3 = self._xi4 = np.array([])
        self._rr_weight = None  # Marker that calculateXi hasn't been called yet.
        self._rr = None
        self._dr = None
        self._rd = None
        self._write_rr = None
        self._write_dr = None
        self._write_rd = None
        self._write_patch_results = False
        self.logger.debug('Finished building NNCorr')

    def copy(self):
        """Make a copy"""
        ret = super().copy()
        # True is possible during read before we finish reading in these attributes.
        if self._rr is not None and self._rr is not True:
            ret._rr = self._rr.copy()
        if self._dr is not None and self._dr is not True:
            ret._dr = self._dr.copy()
        if self._rd is not None and self._rd is not True:
            ret._rd = self._rd.copy()
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
        ret.coords = self.coords
        ret.metric = self.metric
        ret.config = self.config
        ret.meanr = self._zero_array
        ret.meanlogr = self._zero_array
        ret.weight = self._zero_array
        ret.npairs = self._zero_array
        ret.tot = tot
        ret._corr = None
        ret._rr = ret._dr = ret._rd = None
        ret._write_rr = ret._write_dr = ret._write_rd = None
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
        super()._process_auto(cat, metric, num_threads)
        self.tot += 0.5 * cat.sumw**2

    def process_cross(self, cat1, cat2, *, metric=None, num_threads=None):
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
        super().process_cross(cat1, cat2, metric=metric, num_threads=num_threads)
        self.tot += cat1.sumw * cat2.sumw

    def finalize(self):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `Corr2.process_cross` commands accumulate values in each bin,
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
        super()._clear()
        self.tot = 0.

    def __iadd__(self, other):
        """Add a second Correlation object's data to this one.

        .. note::

            For this to make sense, both objects should not have had `finalize` called yet.
            Then, after adding them together, you should call `finalize` on the sum.
        """
        super().__iadd__(other)
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

    def _add_tot(self, ij, c1, c2):
        # When storing results from a patch-based run, tot needs to be accumulated even if
        # the total weight being accumulated comes out to be zero.
        # This only applies to NNCorrelation.  For the other ones, this is a no op.
        tot = c1.sumw * c2.sumw
        self.tot += tot
        # We also have to keep all pairs in the results dict, otherwise the tot calculation
        # gets messed up.  We need to accumulate the tot value of all pairs, even if
        # the resulting weight is zero.  But use a minimal copy with just the necessary fields
        # to save some time.
        self.results[ij] = self._zero_copy(tot)

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

    def calculateXi(self, *, rr, dr=None, rd=None):
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

        After calling this method, you can use the `Corr2.estimate_cov` method or use this
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
            self.xi = self.weight - dr.weight * drf - rd.weight * rdf + denom

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

            # If there are any rr,dr,rd patch pairs that aren't in results (because dr is a cross
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
        self._cov = self.estimate_cov(self.var_method)
        self.varxi = self.cov_diag
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
            if self._rd.npatch1 == 1 and not all([p[0] == 0 for p in pairs]):
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
            xi = dd - dr * drf - rd * rdf + denom
        denom[denom == 0] = 1  # Guard against division by zero.
        self.xi = xi / denom
        self._rr_weight = denom

    def write(self, file_name, *, rr=None, dr=None, rd=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
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
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):       Whether to write the covariance matrix as well. (default: False)
        """
        self.logger.info('Writing NN correlations to %s',file_name)
        # Temporary attributes, so the helper functions can access them.
        precision = self.config.get('precision', 4) if precision is None else precision
        self._write_rr = rr
        self._write_dr = dr
        self._write_rd = rd
        self._write_patch_results = write_patch_results
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov, zero_tot=True)
            if write_patch_results:
                # Also write out rr, dr, rd, so covariances can be computed on round trip.
                rr = rr or self._rr
                dr = dr or self._dr
                rd = rd or self._rd
                if rr:
                    rr._write(writer, '_rr', write_patch_results, zero_tot=True)
                if dr:
                    dr._write(writer, '_dr', write_patch_results, zero_tot=True)
                if rd:
                    rd._write(writer, '_rd', write_patch_results, zero_tot=True)
        self._write_rr = None
        self._write_dr = None
        self._write_rd = None
        self._write_patch_results = False

    @property
    def _write_col_names(self):
        col_names = [ 'r_nom','meanr','meanlogr' ]
        rr = self._write_rr
        dr = self._write_dr
        rd = self._write_rd
        if rr is None:
            if hasattr(self, 'xi'):
                col_names += [ 'xi','sigma_xi' ]
            col_names += [ 'DD', 'npairs' ]
        else:
            col_names += [ 'xi','sigma_xi','DD','RR' ]
            if dr is not None and rd is not None:
                col_names += ['DR','RD']
            elif dr is not None or rd is not None:
                col_names += ['DR']
            col_names += [ 'npairs' ]
        return col_names

    @property
    def _write_data(self):
        data = [ self.rnom, self.meanr, self.meanlogr ]
        rr = self._write_rr
        dr = self._write_dr
        rd = self._write_rd
        if rr is None:
            if hasattr(self, 'xi'):
                data += [ self.xi, np.sqrt(self.varxi) ]
            data += [ self.weight, self.npairs ]
            if dr is not None:
                raise TypeError("rr must be provided if dr is not None")
            if rd is not None:
                raise TypeError("rr must be provided if rd is not None")
        else:
            xi, varxi = self.calculateXi(rr=rr, dr=dr, rd=rd)
            data += [ xi, np.sqrt(varxi),
                      self.weight, rr.weight * (self.tot/rr.tot) ]
            if dr is not None and rd is not None:
                data += [ dr.weight * (self.tot/dr.tot), rd.weight * (self.tot/rd.tot) ]
            elif dr is not None or rd is not None:
                if dr is None: dr = rd
                data += [ dr.weight * (self.tot/dr.tot) ]
            data += [ self.npairs ]
        data = [ col.flatten() for col in data ]
        return data

    @property
    def _write_params(self):
        params = super()._write_params
        params['tot'] = self.tot
        if self._write_patch_results:
            params['_rr'] = bool(self._rr)
            params['_dr'] = bool(self._dr)
            params['_rd'] = bool(self._rd)
        return params

    @classmethod
    def from_file(cls, file_name, *, file_type=None, logger=None, rng=None):
        """Create an NNCorrelation instance from an output file.

        This should be a file that was written by TreeCorr.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII', 'FITS', or 'HDF').  (default: determine
                                the type automatically from the extension of file_name.)
            logger (Logger):    If desired, a logger object to use for logging. (default: None)
            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                                random number generation. (default: None)

        Returns:
            An NNCorrelation object, constructed from the information in the file.
        """
        if logger:
            logger.info('Building NNCorrelation from %s',file_name)
        with make_reader(file_name, file_type, logger) as reader:
            name = 'main' if 'main' in reader else None
            params = reader.read_params(ext=name)
            letters = params.get('corr', None)
            if letters not in Corr2._lookup_dict:
                raise OSError("%s does not seem to be a valid treecorr output file."%file_name)
            if params['corr'] != cls._letters:
                raise OSError("Trying to read a %sCorrelation output file with %s"%(
                                params['corr'], cls.__name__))
            kwargs = make_minimal_config(params, Corr2._valid_params)
            corr = cls(**kwargs, logger=logger, rng=rng)
            corr.logger.info('Reading NN correlations from %s',file_name)
            corr._do_read(reader, name=name, params=params)
        return corr

    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS or HDF5 file, so
        there is no loss of information.

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
        with make_reader(file_name, file_type, self.logger) as reader:
            self._do_read(reader)

    def _do_read(self, reader, name=None, params=None):
        self._read(reader, name, params)
        if self._rr:
            rr = self.copy()
            rr._read(reader, name='_rr')
            self._rr = rr
        if self._dr:
            dr = self.copy()
            dr._read(reader, name='_dr')
            self._dr = dr
        if self._rd:
            rd = self.copy()
            rd._read(reader, name='_rd')
            self._rd = rd

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.logr.shape
        self.weight = data['DD'].reshape(s)
        self.tot = params['tot']
        if 'xi' in data.dtype.names:
            self.xi = data['xi'].reshape(s)
            self.varxi = data['sigma_xi'].reshape(s)**2
        # Note: "or None" turns False -> None
        self._rr = params.get('_rr', None) or None
        self._dr = params.get('_dr', None) or None
        self._rd = params.get('_rd', None) or None

    def calculateNapSq(self, *, rr, R=None, dr=None, rd=None, m2_uform=None):
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

        xi, varxi = self.calculateXi(rr=rr, dr=dr, rd=rd)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxi = Tp.dot(xi)
        nsq = Tpxi * self.bin_size
        varnsq = (Tp**2).dot(varxi) * self.bin_size**2

        return nsq, varnsq

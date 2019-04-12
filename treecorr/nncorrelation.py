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

import treecorr
import numpy as np


class NNCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:     The number of bins in logr
        bin_size:  The size of the bins in logr
        min_sep:   The minimum separation being considered
        max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:
        logr:      The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:      The nominal center of the bin converted to regular distance.
                   i.e. r = exp(logr).
        meanr:     The (weighted) mean value of r for the pairs in each bin.
                   If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:  The mean value of log(r) for the pairs in each bin.
                   If there are no pairs in a bin, then logr will be used instead.
        weight:    The total weight in each bin.
        npairs:    The number of pairs in each bin.
        tot:       The total number of pairs processed, which is used to normalize
                   the randoms if they have a different number of pairs.

    If **sep_units** are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.  Note however, that if you separate out the steps of the
    `process` command and use `process_auto` and/or `process_cross`, then the units will not be
    applied to **meanr** or **meanlogr** until the `finalize` function is called.

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
                        This dict is allowed to have addition entries in addition to those listed
                        in `BinnedCorr2`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    See the documentation for `BinnedCorr2` for the list of other allowed kwargs, which may be
    passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self._d1 = 1  # NData
        self._d2 = 1  # NData
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self.tot = 0.
        self._build_corr()
        self.logger.debug('Finished building NNCorr')

    def _build_corr(self):
        from treecorr.util import double_ptr as dp
        self.corr = treecorr._lib.BuildCorr2(
                self._d1, self._d2, self._bintype,
                self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                dp(None), dp(None), dp(None), dp(None),
                dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs));

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        # In case __init__ failed to get that far
        if hasattr(self,'corr'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        """Return whether two NNCorrelations are equal"""
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
        treecorr._lib.ProcessAuto2(self.corr, field.data, self.output_dots,
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
                            self.brute is True or self.brute is 1,
                            self.min_top, self.max_top, self.coords)
        f2 = cat2.getNField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute is 2,
                            self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        treecorr._lib.ProcessCross2(self.corr, f1.data, f2.data, self.output_dots,
                                    f1._d, f2._d, self._coords, self._bintype, self._metric)
        self.tot += cat1.sumw*cat2.sumw


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the sums into the bins, but does not finalize the calculation.
        After calling this function as often as desired, the `finalize` command will
        finish the calculation.

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
            self.logger.info('Starting process NN pairwise-correlations')
        else:
            self.logger.info('Starting process NN pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField()
        f2 = cat2.getNSimpleField()

        treecorr._lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                                  f1._d, f2._d, self._coords, self._bintype, self._metric)
        self.tot += (cat1.sumw+cat2.sumw)/2.


    def finalize(self):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanr, meanlogr by dividing by the total weight.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]


    def clear(self):
        """Clear the data vectors
        """
        self.meanr.ravel()[:] = 0.
        self.meanlogr.ravel()[:] = 0.
        self.weight.ravel()[:] = 0.
        self.npairs.ravel()[:] = 0.
        self.tot = 0.

    def __iadd__(self, other):
        """Add a second NNCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        `process_auto` and/or `process_cross`, and they should not have had `finalize` called yet.
        Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, NNCorrelation):
            raise TypeError("Can only add another NNCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NNCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords)
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        self.tot += other.tot
        return self


    def process(self, cat1, cat2=None, metric=None, num_threads=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

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
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]

        if cat2 is None or len(cat2) == 0:
            self._process_all_auto(cat1,metric,num_threads)
        else:
            self._process_all_cross(cat1,cat2,metric,num_threads)
        self.finalize()

    def _mean_weight(self):
        mean_np = np.mean(self.npairs)
        return 1 if mean_np == 0 else np.mean(self.weight)/mean_np

    def calculateXi(self, rr, dr=None, rd=None):
        """Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross correlations of the data and random.

        The rr value is the NNCorrelation function for random points.
        For a signal that involves a cross correlations, there should be two random
        cross-correlations: data-random and random-data, given as dr and rd.

        - If dr is None, the simple correlation function :math:`\\xi = (DD/RR - 1)` is used.
        - if dr is given and rd is None, then :math:`\\xi = (DD - 2DR + RR)/RR` is used.
        - If dr and rd are both given, then :math:`\\xi = (DD - DR - RD + RR)/RR` is used.

        where DD is the data NN correlation function, which is the current object.

        Parameters:
            rr (NNCorrelation):     The auto-correlation of the random field (RR)
            dr (NNCorrelation):     The cross-correlation of the data with randoms (DR), if
                                    desired, in which case the Landy-Szalay estimator will be
                                    calculated.  (default: None)
            rd (NNCorrelation):     The cross-correlation of the randoms with data (RD), if
                                    desired. (default: None, which means use rd=dr)

        Returns:
            Tuple containing

                - xi = array of :math:`\\xi(r)`
                - varxi = array of variance estimates of :math:`\\xi(r)`
        """
        # Each random weight value needs to be rescaled by the ratio of total possible pairs.
        if rr.tot == 0:
            raise ValueError("rr has tot=0.")

        # rrf is the factor to scale rr weights to get something commensurate to the dd density.
        rrf = self.tot / rr.tot

        # ddw and rrw are the mean weight of the dd and rr pairs.
        # This is only needed for the variance estimate, since there is shot noise on the
        # number of pairs, not the weight.
        ddw = self._mean_weight()
        rrw = rr._mean_weight()

        if dr is None:
            if rd is None:
                xi = (self.weight - rr.weight * rrf)
                varxi_factor = 1 + rrf*rrw/ddw
            else:
                if rd.tot == 0:
                    raise ValueError("rd has tot=0.")
                rdf = self.tot / rd.tot
                rdw = rd._mean_weight()
                xi = (self.weight - 2.*rd.weight * rdf + rr.weight * rrf)
                varxi_factor = 1 + 2*rdf*rdw/ddw + rrf*rrw/ddw
        else:
            if dr.tot == 0:
                raise ValueError("dr has tot=0.")
            drf = self.tot / dr.tot
            drw = dr._mean_weight()
            if rd is None:
                xi = (self.weight - 2.*dr.weight * drf + rr.weight * rrf)
                varxi_factor = 1 + 2*drf*drw/ddw + rrf*rrw/ddw
            else:
                if rd.tot == 0:
                    raise ValueError("rd has tot=0.")
                rdf = self.tot / rd.tot
                rdw = rd._mean_weight()
                xi = (self.weight - rd.weight * rdf - dr.weight * drf + rr.weight * rrf)
                varxi_factor = 1 + drf*drw/ddw + rdf*rdw/ddw + rrf*rrw/ddw
        if np.any(rr.weight == 0):
            self.logger.warning("Warning: Some bins for the randoms had no pairs.")
        mask1 = rr.weight != 0
        mask2 = rr.weight == 0
        xi[mask1] /= (rr.weight[mask1] * rrf)
        xi[mask2] = 0

        varxi = np.zeros_like(rr.weight)
        # Note: The use of varxi_factor here is semi-empirical.
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
        varxi[mask1] = ddw * varxi_factor**2 / (rr.weight[mask1] * rrf)

        return xi, varxi


    def write(self, file_name, rr=None, dr=None, rd=None, file_type=None, precision=None):
        """Write the correlation function to the file, file_name.

        rr is the NNCorrelation function for random points.
        If dr is None, the simple correlation function :math:`\\xi = (DD - RR)/RR` is used.
        if dr is given and rd is None, then :math:`\\xi = (DD - 2DR + RR)/RR` is used.
        If dr and rd are both given, then :math:`\\xi = (DD - DR - RD + RR)/RR` is used.

        Normally, at least rr should be provided, but if this is also None, then only the
        basic accumulated number of pairs are output (along with the separation columns).

        The output file will include the following columns:

        ==========      =========================================================
        Column          Description
        ==========      =========================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value <r> of pairs that fell into each bin
        meanlogr        The mean value <log(r)> of pairs that fell into each bin
        xi              The estimator xi (if rr is given)
        sigma_xi        The sqrt of the variance estimate of xi (if rr is given)
        DD              The total weight of pairs in each bin.
        RR              The total weight of RR pairs in each bin (if rr is given)
        DR              The total weight of DR pairs in each bin (if dr is given)
        RD              The total weight of RD pairs in each bin (if rd is given)
        npairs          The total number of pairs in each bin
        ==========      =========================================================

        If **sep_units** was given at construction, then the distances will all be in these units.
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

        treecorr.util.gen_write(
            file_name, col_names, columns, params=params,
            precision=precision, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The NNCorrelation object should be constructed with the same configuration
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        Parameters:
            file_name (str):   The name of the file to read in.
            file_type (str):   The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NN correlations from %s',file_name)

        data, params = treecorr.util.gen_read(file_name, file_type=file_type, logger=self.logger)
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self.rnom = data['R_nom']
            self.meanr = data['meanR']
            self.meanlogr = data['meanlogR']
        else:
            self.rnom = data['r_nom']
            self.meanr = data['meanr']
            self.meanlogr = data['meanlogr']
        self.logr = np.log(self.rnom)
        self.weight = data['DD']
        self.npairs = data['npairs']
        self.tot = params['tot']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.sep_units = params['sep_units'].strip()
        self.bin_type = params['bin_type'].strip()
        self._build_corr()


    def calculateNapSq(self, rr, R=None, dr=None, rd=None, m2_uform=None):
        """Calculate the correlary to the aperture mass statistics for counts.

        .. math::

            \\langle N_{ap}^2 \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{2R^2}
            \\left [ T_+\\left(\\frac{r}{R}\\right) \\xi(r) \\right] \\\\

        The **m2_uform** parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If **m2_uform** is 'Crittenden':

        .. math::

            U(r) &= \\frac{1}{2\\pi} (1-r^2) \\exp(-r^2/2) \\\\
            T_+(s) &= \\frac{s^4 - 16s^2 + 32}{128} \\exp(-s^2/4) \\\\
            rmax &= \\infty

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If **m2_uform** is 'Schneider':

        .. math::

            U(r) &= \\frac{9}{\\pi} (1-r^2) (1/3-r^2) \\\\
            T_+(s) &= \\frac{12}{5\\pi} (2-15s^2) \\arccos(s/2) \\\\
            &\\qquad + \\frac{1}{100\\pi} s \\sqrt{4-s^2} (120 + 2320s^2 - 754s^4 + 132s^6 - 9s^8) \\\\
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

                - nsq = array of :math:`\\langle N_{ap}^2 \\rangle(R)`
                - varnsq = array of variance estimates of this value
        """
        if m2_uform is None:
            m2_uform = treecorr.config.get(self.config,'m2_uform',str,'Crittenden')
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



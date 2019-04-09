# Copyright (c) 2003-2015 by Mike Jarvis
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
.. module:: ggcorrelation
"""

import treecorr
import numpy as np


class GGCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    Ojects of this class holds the following attributes:

        :nbins:     The number of bins in logr
        :bin_size:  The size of the bins in logr
        :min_sep:   The minimum separation being considered
        :max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

        :logr:      The nominal center of the bin in log(r) (the natural logarithm of r).
        :rnom:      The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        :meanr:     The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        :meanlogr:  The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        :xip:       The correlation function, :math:`\\xi_+(r)`.
        :xim:       The correlation funciton, :math:`\\xi_-(r)`.
        :xip_im:    The imaginary part of :math:`\\xi_+(r)`.
        :xim_im:    The imaginary part of :math:`\\xi_-(r)`.
        :varxi:     The variance of xip and xim, only including the shape noise propagated
                    into the final correlation.  This does not include sample variance, so
                    it is always an underestimate of the actual variance.
        :weight:    The total weight in each bin.
        :npairs:    The number of pairs going into each bin.

    If `sep_units` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.  Note however, that if you separate out the steps of the
    :func:`process` command and use :func:`process_auto` and/or :func:`process_cross`, then the
    units will not be applied to :meanr: or :meanlogr: until the :func:`finalize` function is
    called.

    The typical usage pattern is as follows:

        >>> gg = treecorr.GGCorrelation(config)
        >>> gg.process(cat)         # For auto-correlation.
        >>> gg.process(cat1,cat2)   # For cross-correlation.
        >>> gg.write(file_name)     # Write out to a file.
        >>> xip = gg.xip            # Or access the correlation function directly.

    :param config:      A configuration dict that can be used to pass in kwargs if desired.
                        This dict is allowed to have addition entries in addition to those listed
                        in :class:`~treecorr.BinnedCorr2`, which are ignored here. (default: None)
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    See the documentation for :class:`~treecorr.BinnedCorr2` for the list of other allowed kwargs,
    which may be passed either directly or in the config dict.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self._d1 = 3  # GData
        self._d2 = 3  # GData
        self.xip = np.zeros_like(self.rnom, dtype=float)
        self.xim = np.zeros_like(self.rnom, dtype=float)
        self.xip_im = np.zeros_like(self.rnom, dtype=float)
        self.xim_im = np.zeros_like(self.rnom, dtype=float)
        self.varxi = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self._build_corr()
        self.logger.debug('Finished building GGCorr')

    def _build_corr(self):
        from treecorr.util import double_ptr as dp
        self.corr = treecorr._lib.BuildCorr2(
                self._d1, self._d2, self._bintype,
                self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                dp(self.xip),dp(self.xip_im),dp(self.xim),dp(self.xim_im),
                dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs))

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        # In case __init__ failed to get that far
        if hasattr(self,'corr'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        return (isinstance(other, GGCorrelation) and
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
                np.array_equal(self.meanr, other.meanr) and
                np.array_equal(self.meanlogr, other.meanlogr) and
                np.array_equal(self.xip, other.xip) and
                np.array_equal(self.xim, other.xim) and
                np.array_equal(self.xip_im, other.xip_im) and
                np.array_equal(self.xim_im, other.xim_im) and
                np.array_equal(self.varxi, other.varxi) and
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
        return 'GGCorrelation(config=%r)'%self.config

    def process_auto(self, cat, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat:         The catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.GGCorrelation.process` for
                            details.  (default: 'Euclidean'; this value can also be given in the
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler is clang prior to version 3.7.
        """
        if cat.name == '':
            self.logger.info('Starting process GG auto-correlations')
        else:
            self.logger.info('Starting process GG auto-correlations for cat %s.',cat.name)

        self._set_metric(metric, cat.coords)

        self._set_num_threads(num_threads)

        min_size, max_size = self._get_minmax_size()

        field = cat.getGField(min_size, max_size, self.split_method,
                              bool(self.brute), self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        treecorr._lib.ProcessAuto2(self.corr, field.data, self.output_dots,
                                   field._d, self._coords, self._bintype, self._metric)


    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.GGCorrelation.process` for
                            details.  (default: 'Euclidean'; this value can also be given in the
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process GG cross-correlations')
        else:
            self.logger.info('Starting process GG cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute is 1,
                            self.min_top, self.max_top, self.coords)
        f2 = cat2.getGField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute is 2,
                            self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        treecorr._lib.ProcessCross2(self.corr, f1.data, f2.data, self.output_dots,
                                    f1._d, f2._d, self._coords, self._bintype, self._metric)


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.GGCorrelation.process` for
                            details.  (default: 'Euclidean'; this value can also be given in the
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process GG pairwise-correlations')
        else:
            self.logger.info('Starting process GG pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        f1 = cat1.getGSimpleField()
        f2 = cat2.getGSimpleField()

        treecorr._lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                                  f1._d, f2._d, self._coords, self._bintype, self._metric)


    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        :param varg1:   The shear variance per component for the first field.
        :param varg2:   The shear variance per component for the second field.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.xip[mask1] /= self.weight[mask1]
        self.xim[mask1] /= self.weight[mask1]
        self.xip_im[mask1] /= self.weight[mask1]
        self.xim_im[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = 2 * varg1 * varg2 / self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.


    def clear(self):
        """Clear the data vectors
        """
        self.xip.ravel()[:] = 0
        self.xim.ravel()[:] = 0
        self.xip_im.ravel()[:] = 0
        self.xim_im.ravel()[:] = 0
        self.meanr.ravel()[:] = 0
        self.meanlogr.ravel()[:] = 0
        self.weight.ravel()[:] = 0
        self.npairs.ravel()[:] = 0


    def __iadd__(self, other):
        """Add a second GGCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        process_auto and/or process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, GGCorrelation):
            raise TypeError("Can only add another GGCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("GGCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords)
        self.xip.ravel()[:] += other.xip.ravel()[:]
        self.xim.ravel()[:] += other.xim.ravel()[:]
        self.xip_im.ravel()[:] += other.xip_im.ravel()[:]
        self.xim_im.ravel()[:] += other.xim_im.ravel()[:]
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        return self


    def process(self, cat1, cat2=None, metric=None, num_threads=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        :param cat1:    A catalog or list of catalogs for the first G field.
        :param cat2:    A catalog or list of catalogs for the second G field, if any.
                        (default: None)
        :param metric:  Which metric to use for distance measurements.  Options are given
                        in the doc string of :class:`~treecorr.BinnedCorr2`.
                        (default: 'Euclidean'; this value can also be given in the constructor
                        in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler is clang prior to version 3.7.
        """
        import math
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]

        if cat2 is None:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = varg1
            self.logger.info("varg = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
            self._process_all_auto(cat1, metric, num_threads)
        else:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = treecorr.calculateVarG(cat2)
            self.logger.info("varg1 = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
            self.logger.info("varg2 = %f: sig_sn (per component) = %f",varg2,math.sqrt(varg2))
            self._process_all_cross(cat1,cat2, metric, num_threads)
        self.finalize(varg1,varg2)


    def write(self, file_name, file_type=None, precision=None):
        """Write the correlation function to the file, file_name.

        The output file will include the following columns:

            :R_nom:     The nominal center of the bin in R.
            :meanR:     The mean value :math:`\\langle R\\rangle` of pairs that fell into each bin.
            :meanlogR:  The mean value :math:`\\langle logR\\rangle` of pairs that fell into each
                        bin.
            :xip:       The real part of the :math:`\\xi_+` correlation function.
            :xim:       The real part of the :math:`\\xi_-` correlation function.
            :xip_im:    The imag part of the :math:`\\xi_+` correlation function.
            :xim_im:    The imag part of the :math:`\\xi_-` correlation function.
            :sigma_xi:  The sqrt of the variance estimate of :math:`\\xi_+`, :math:`\\xi_-`.
            :weight:    The total weight contributing to each bin.
            :npairs:    The number of pairs contributing ot each bin.

        If `sep_units` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        :param file_name:   The name of the file to write to.
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param precision:   For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing GG correlations to %s',file_name)

        if precision is None:
            precision = treecorr.config.get(self.config,'precision',int,4)

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        treecorr.util.gen_write(
            file_name,
            ['R_nom','meanR','meanlogR','xip','xim','xip_im','xim_im','sigma_xi','weight','npairs'],
            [ self.rnom, self.meanr, self.meanlogr,
              self.xip, self.xim, self.xip_im, self.xim_im, np.sqrt(self.varxi),
              self.weight, self.npairs ],
            params=params, precision=precision, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The GGCorrelation object should be constructed with the same configuration
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        :param file_name:   The name of the file to read in.
        :param file_type:   The type of file ('ASCII' or 'FITS').  (default: determine the type
                            automatically from the extension of file_name.)
        """
        self.logger.info('Reading GG correlations from %s',file_name)

        data, params = treecorr.util.gen_read(file_name, file_type=file_type, logger=self.logger)
        self.rnom = data['R_nom']
        self.logr = np.log(data['R_nom'])
        self.meanr = data['meanR']
        self.meanlogr = data['meanlogR']
        self.xip = data['xip']
        self.xim = data['xim']
        self.xip_im = data['xip_im']
        self.xim_im = data['xim_im']
        self.varxi = data['sigma_xi']**2
        self.weight = data['weight']
        self.npairs = data['npairs']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.sep_units = params['sep_units'].strip()
        self.bin_type = params['bin_type'].strip()
        self._build_corr()


    def calculateMapSq(self, m2_uform=None):
        """Calculate the aperture mass statistics from the correlation function.

        .. math::

            \\langle M_{ap}^2 \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{2R^2}
            \\left [ T_+\\left(\\frac{r}{R}\\right) \\xi_+(r) +
            T_-\\left(\\frac{r}{R}\\right) \\xi_-(r) \\right] \\\\
            \\langle M_\\times^2 \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{2R^2}
            \\left [ T_+\\left(\\frac{r}{R}\\right) \\xi_+(r) -
            T_-\\left(\\frac{r}{R}\\right) \\xi_-(r) \\right]

        The m2_uform parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If m2_uform == 'Crittenden':

        .. math::

            U(r) &= \\frac{1}{2\\pi} (1-r^2) \\exp(-r^2/2) \\\\
            Q(r) &= \\frac{1}{4\\pi} r^2 \\exp(-r^2/2) \\\\
            T_+(s) &= \\frac{s^4 - 16s^2 + 32}{128} \\exp(-s^2/4) \\\\
            T_-(s) &= \\frac{s^4}{128} \\exp(-s^2/4) \\\\
            rmax &= \\infty

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If m2_uform == 'Schneider':

        .. math::

            U(r) &= \\frac{9}{\\pi} (1-r^2) (1/3-r^2) \\\\
            Q(r) &= \\frac{6}{\\pi} r^2 (1-r^2) \\\\
            T_+(s) &= \\frac{12}{5\\pi} (2-15s^2) \\arccos(s/2)
            + \\frac{1}{100\\pi} s \\sqrt{4-s^2} (120 + 2320s^2 - 754s^4 + 132s^6 - 9s^8) \\\\
            T_-(s) &= \\frac{3}{70\\pi} s^3 (4-s^2)^{7/2} \\\\
            rmax &= 2R

        cf. Schneider, et al (2002): A&A, 389, 729

        .. note::

            This function is only implemented for Log binning.


        :param m2_uform:    Which form to use for the aperture mass, as described above.
                            (default: 'Crittenden'; this value can also be given in the
                            constructor in the config dict.)

        :returns:           (mapsq, mapsq_im, mxsq, mxsq_im, varmapsq) as a tuple
        """
        if m2_uform is None:
            m2_uform = treecorr.config.get(self.config,'m2_uform',str,'Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")
        if self.bin_type is not 'Log':
            raise ValueError("calculateMapSq requires Log binning.")

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        r = self.rnom
        s = np.outer(1./r, self.meanr)
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = np.exp(-ssq/4.)
            Tp = (32. + ssq*(-16. + ssq)) / 128. * exp_factor
            Tm = ssq * ssq / 128. * exp_factor
        else:
            Tp = np.zeros_like(s)
            Tm = np.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tp[s<2.] = 12./(5.*np.pi) * (2.-15.*ssqa) * np.arccos(sa/2.)
            Tp[s<2.] += 1./(100.*np.pi) * sa * np.sqrt(4.-ssqa) * (
                        120. + ssqa*(2320. + ssqa*(-754. + ssqa*(132. - 9.*ssqa))))
            Tm[s<2.] = 3./(70.*np.pi) * sa * ssqa * (4.-ssqa)**3.5
        Tp *= ssq
        Tm *= ssq

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxip = Tp.dot(self.xip)
        Tmxim = Tm.dot(self.xim)
        mapsq = (Tpxip + Tmxim) * 0.5 * self.bin_size
        mxsq = (Tpxip - Tmxim) * 0.5 * self.bin_size
        Tpxip_im = Tp.dot(self.xip_im)
        Tmxim_im = Tm.dot(self.xim_im)
        mapsq_im = (Tpxip_im + Tmxim_im) * 0.5 * self.bin_size
        mxsq_im = (Tpxip_im - Tmxim_im) * 0.5 * self.bin_size

        # The variance of each of these is
        # Var(<Map^2>(R)) = int_r=0..2R [1/4 s^4 dlogr^2 (T+(s)^2 + T-(s)^2) Var(xi)]
        varmapsq = (Tp**2 + Tm**2).dot(self.varxi) * 0.25 * self.bin_size**2

        return mapsq, mapsq_im, mxsq, mxsq_im, varmapsq


    def calculateGamSq(self, eb=False):
        """Calculate the tophat shear variance from the correlation function.

        .. math::

            \\langle \\gamma^2 \\rangle(R) &= \\int_0^{2R} \\frac{r dr}{R^2} S_+(s) \\xi_+(r) \\\\
            \\langle \\gamma^2 \\rangle_E(R) &= \\int_0^{2R} \\frac{r dr}{2 R^2}
            \\left [ S_+\\left(\\frac{r}{R}\\right) \\xi_+(r) +
            S_-\\left(\\frac{r}{R}\\right) \\xi_-(r) \\right ] \\\\
            \\langle \\gamma^2 \\rangle_B(R) &= \\int_0^{2R} \\frac{r dr}{2 R^2}
            \\left [ S_+\\left(\\frac{r}{R}\\right) \\xi_+(r) -
            S_-\\left(\\frac{r}{R}\\right) \\xi_-(r) \\right ] \\\\

            S_+(s) &= \\frac{1}{\\pi} \\left(4 \\arccos(s/2) - s \\sqrt{4-s^2} \\right) \\\\
            S_-(s) &= \\begin{cases}
            s<=2, & [ s \\sqrt{4-s^2} (6-s^2) - 8(3-s^2) \\arcsin(s/2) ] / (\\pi s^4) \\\\
            s>=2, & 4(s^2-3)/(s^4)
            \\end{cases}

        cf. Schneider, et al (2002): A&A, 389, 729

        The default behavior is not to compute the E/B versions.  They are calculated if
        eb is set to True.

        .. note::

            This function is only implemented for Log binning.


        :param eb:  Whether to include the E/B decomposition as well as the total
                    :math:`\\langle \\gamma^2\\rangle`.  (default: False)

        :returns:   (gamsq, vargamsq) if `eb == False` or
                    (gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e)  if `eb == True`
        """
        if self.bin_type is not 'Log':
            raise ValueError("calculateGamSq requires Log binning.")

        r = self.rnom
        s = np.outer(1./r, self.meanr)
        ssq = s*s
        Sp = np.zeros_like(s)
        sa = s[s<2]
        ssqa = ssq[s<2]
        Sp[s<2.] = 1./np.pi * ssqa * (4.*np.arccos(sa/2.) - sa*np.sqrt(4.-ssqa))

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Spxip = Sp.dot(self.xip)
        gamsq = Spxip * self.bin_size
        vargamsq = (Sp**2).dot(self.varxi) * self.bin_size**2

        # Stop here if eb == False
        if not eb: return gamsq, vargamsq

        Sm = np.empty_like(s)
        Sm[s<2.] = 1./(ssqa*np.pi) * (sa*np.sqrt(4.-ssqa)*(6.-ssqa)
                                                 -8.*(3.-ssqa)*np.arcsin(sa/2.))
        Sm[s>=2.] = 4.*(ssq[s>=2]-3.)/ssq[s>=2]
        # This already includes the extra ssq factor.

        Smxim = Sm.dot(self.xim)
        gamsq_e = (Spxip + Smxim) * 0.5 * self.bin_size
        gamsq_b = (Spxip - Smxim) * 0.5 * self.bin_size
        vargamsq_e = (Sp**2 + Sm**2).dot(self.varxi) * 0.25 * self.bin_size**2

        return gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e


    def writeMapSq(self, file_name, m2_uform=None, file_type=None, precision=None):
        """Write the aperture mass statistics based on the correlation function to the
        file, file_name.

        See :meth:`~treecorr.GGCorrelation.calculateMapSq` for an explanation of the m2_uform
        parameter.

        The output file will include the following columns:

            :R:         The aperture radius
            :Mapsq:     The real part of :math:`\\langle M_{ap}^2\\rangle`.
                        cf. :meth:`~treecorr.GGCorrelation.calculateMapSq`.
            :Mxsq:      The real part of :math:`\\langle M_x^2\\rangle`.
            :MMxa:      The imag part of :math:`\\langle M_{ap}^2\\rangle`.
                        This is one of two estimators of :math:`\\langle M_{ap} M_x\\rangle`.
            :MMxb:      The imag part of :math:`-\\langle M_x^2\\rangle`.
                        This is the second estimator of :math:`\\langle M_{ap} M_x\\rangle`.
            :sig_map:   The sqrt of the variance estimate of :math:`\\langle M_{ap}^2\\rangle`
                        (which is equal to the variance of :math:`\\langle M_x^2\\rangle` as well).
            :Gamsq:     The tophat shear variance :math:`\\langle \\gamma^2\\rangle`.
                        cf. :meth:`~treecorr.GGCorrelation.calculateGamSq`.
            :sig_gam:   The sqrt of the variance estimate of :math:`\\langle \\gamma^2\\rangle`


        :param file_name:   The name of the file to write to.
        :param m2_uform:    Which form to use for the aperture mass.  (default: 'Crittenden';
                            this value can also be given in the constructor in the config dict.)
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param precision:   For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing Map^2 from GG correlations to %s',file_name)

        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = self.calculateMapSq(m2_uform=m2_uform)
        gamsq, vargamsq = self.calculateGamSq()
        if precision is None:
            precision = treecorr.config.get(self.config,'precision',int,4)

        treecorr.util.gen_write(
            file_name,
            ['R','Mapsq','Mxsq','MMxa','MMxb','sig_map','Gamsq','sig_gam'],
            [ self.rnom,
              mapsq, mxsq, mapsq_im, -mxsq_im, np.sqrt(varmapsq),
              gamsq, np.sqrt(vargamsq) ],
            precision=precision, file_type=file_type, logger=self.logger)



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
.. module:: ngcorrelation
"""

import treecorr
import numpy as np


class NGCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-shear correlation
    function.  This is the tangential shear profile around lenses, commonly referred to as
    galaxy-galaxy lensing.

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
        meanlogr:  The (weighted) mean value of log(r) for the pairs in each bin.
                   If there are no pairs in a bin, then logr will be used instead.
        xi:        The correlation function, :math:`\\xi(r) = \\langle \\gamma_T\\rangle`.
        xi_im:     The imaginary part of :math:`\\xi(r)`.
        varxi:     The variance of :math:`\\xi`, only including the shape noise propagated into
                   the final correlation.  This does not include sample variance, so it is
                   always an underestimate of the actual variance.
        weight:    The total weight in each bin.
        npairs:    The number of pairs going into each bin.

    If **sep_units** are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.  Note however, that if you separate out the steps of the
    `process` command and use `process_cross`, then the units will not be applied to **meanr** or
    **meanlogr** until the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> ng = treecorr.NGCorrelation(config)
        >>> ng.process(cat1,cat2)   # Compute the cross-correlation.
        >>> ng.write(file_name)     # Write out to a file.
        >>> xi = gg.xi              # Or access the correlation function directly.

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
        self._d2 = 3  # GData
        self.xi = np.zeros_like(self.rnom, dtype=float)
        self.xi_im = np.zeros_like(self.rnom, dtype=float)
        self.varxi = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self._build_corr()
        self.logger.debug('Finished building NGCorr')

    def _build_corr(self):
        from treecorr.util import double_ptr as dp
        self.corr = treecorr._lib.BuildCorr2(
                self._d1, self._d2, self._bintype,
                self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                dp(self.xi),dp(self.xi_im), dp(None), dp(None),
                dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs));

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        # In case __init__ failed to get that far
        if hasattr(self,'corr'):  # pragma: no branch
            if not treecorr._ffi._lock.locked(): # pragma: no branch
                treecorr._lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        """Return whether two NGCorrelations are equal"""
        return (isinstance(other, NGCorrelation) and
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
                np.array_equal(self.xi, other.xi) and
                np.array_equal(self.xi_im, other.xi_im) and
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
        return 'NGCorrelation(config=%r)'%self.config

    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the `finalize` command will
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
            self.logger.info('Starting process NG cross-correlations')
        else:
            self.logger.info('Starting process NG cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size, max_size, self.split_method,
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
        calling this function as often as desired, the `finalize` command will
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
            self.logger.info('Starting process NG pairwise-correlations')
        else:
            self.logger.info('Starting process NG pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField()
        f2 = cat2.getGSimpleField()

        treecorr._lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                                  f1._d, f2._d, self._coords, self._bintype, self._metric)


    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            varg (float):   The shear variance per component for the second field.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.xi[mask1] /= self.weight[mask1]
        self.xi_im[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = varg / self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.


    def clear(self):
        """Clear the data vectors
        """
        self.xi.ravel()[:] = 0
        self.xi_im.ravel()[:] = 0
        self.varxi.ravel()[:] = 0
        self.meanr.ravel()[:] = 0
        self.meanlogr.ravel()[:] = 0
        self.weight.ravel()[:] = 0
        self.npairs.ravel()[:] = 0

    def __iadd__(self, other):
        """Add a second NGCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        `process_cross`, and they should not have had `finalize` called yet.
        Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, NGCorrelation):
            raise TypeError("Can only add another NGCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NGCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords)
        self.xi.ravel()[:] += other.xi.ravel()[:]
        self.xi_im.ravel()[:] += other.xi_im.ravel()[:]
        self.varxi.ravel()[:] += other.varxi.ravel()[:]
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        return self


    def process(self, cat1, cat2, metric=None, num_threads=None):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the N field.
            cat2 (Catalog):     A catalog or list of catalogs for the G field.
            metric (str):       Which metric to use.  See `Metrics` for details.
                                (default: 'Euclidean'; this value can also be given in the
                                constructor in the config dict.)
            num_threads (int):  How many OpenMP threads to use during the calculation.
                                (default: use the number of cpu cores; this value can also be given
                                in the constructor in the config dict.)
        """
        import math
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if not isinstance(cat2,list): cat2 = [cat2]

        varg = treecorr.calculateVarG(cat2)
        self.logger.info("varg = %f: sig_sn (per component) = %f",varg,math.sqrt(varg))
        self._process_all_cross(cat1,cat2,metric,num_threads)
        self.finalize(varg)


    def calculateXi(self, rg=None):
        """Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rg is None, the simple correlation function :math:`\\langle \\gamma_T\\rangle` is
          returned.
        - If rg is not None, then a compensated calculation is done:
          :math:`\\langle \\gamma_T\\rangle = (DG - RG)`

        Parameters:
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\\xi(R)`
                - xi_im = array of the imaginary part of :math:`\\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        if rg is None:
            return self.xi, self.xi_im, self.varxi
        else:
            return (self.xi - rg.xi), (self.xi_im - rg.xi_im), (self.varxi + rg.varxi)


    def write(self, file_name, rg=None, file_type=None, precision=None):
        """Write the correlation function to the file, file_name.

        - If rg is None, the simple correlation function :math:`\\langle \\gamma_T\\rangle` is used.
        - If rg is not None, then a compensated calculation is done:
          :math:`\\langle \\gamma_T\\rangle = (DG - RG)`

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value <r> of pairs that fell into each bin
        meanlogr        The mean value <log(r)> of pairs that fell into each bin
        gamT            The real part of the mean tangential shear <gamma_T>(r)
        gamX            The imag part of the mean tangential shear <gamma_X>(r)
        sigma           The sqrt of the variance estimate of either of these
        weight          The total weight contributing to each bin
        npairs          The number of pairs contributing ot each bin
        ==========      =============================================================

        If **sep_units** was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing NG correlations to %s',file_name)

        xi, xi_im, varxi = self.calculateXi(rg)
        if precision is None:
            precision = self.config.get('precision', 4)

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        treecorr.util.gen_write(
            file_name,
            ['r_nom','meanr','meanlogr','gamT','gamX','sigma','weight','npairs'],
            [ self.rnom, self.meanr, self.meanlogr,
              xi, xi_im, np.sqrt(varxi), self.weight, self.npairs ],
            params=params, precision=precision, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The NGCorrelation object should be constructed with the same configuration
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading NG correlations from %s',file_name)

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
        self.xi = data['gamT']
        self.xi_im = data['gamX']
        self.varxi = data['sigma']**2
        self.weight = data['weight']
        self.npairs = data['npairs']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.sep_units = params['sep_units'].strip()
        self.bin_type = params['bin_type'].strip()
        self._build_corr()


    def calculateNMap(self, R=None, rg=None, m2_uform=None):
        """Calculate the aperture mass statistics from the correlation function.

        .. math::

            \\langle N M_{ap} \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{R^2}
            T_\\times\\left(\\frac{r}{R}\\right) \\Re\\xi(r) \\\\
            \\langle N M_{\\times} \\rangle(R) &= \\int_{0}^{rmax} \\frac{r dr}{R^2}
            T_\\times\\left(\\frac{r}{R}\\right) \\Im\\xi(r)

        The **m2_uform** parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If **m2_uform** is 'Crittenden':

        .. math::

            U(r) &= \\frac{1}{2\\pi} (1-r^2) \\exp(-r^2/2) \\\\
            T_\\times(s) &= \\frac{s^2}{128} (12-s^2) \\exp(-s^2/4)

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If **m2_uform** is 'Schneider':

        .. math::

            U(r) &= \\frac{9}{\\pi} (1-r^2) (1/3-r^2) \\\\
            T_\\times(s) &= \\frac{18}{\\pi} s^2 \\arccos(s/2) \\\\
            &\qquad - \\frac{3}{40\\pi} s^3 \\sqrt{4-s^2} (196 - 74s^2 + 14s^4 - s^6)

        cf. Schneider, et al (2002): A&A, 389, 729

        In neither case is this formula in the above papers, but the derivation is similar
        to the derivations of :math:`T_+` and :math:`T_-` in Schneider et al. (2002).

        Parameters:
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            m2_uform (str):     Which form to use for the aperture mass, as described above.
                                (default: 'Crittenden'; this value can also be given in the
                                constructor in the config dict.)

        Returns:
            Tuple containing

                - nmap = array of :math:`\\langle N M_{ap} \\rangle(R)`
                - nmx = array of :math:`\\langle N M_{\\times} \\rangle(R)`
                - varnmap = array of variance estimates of the above values
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
            Tx = ssq * (12. - ssq) / 128. * exp_factor
        else:
            Tx = np.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tx[s<2.] = 196. + ssqa*(-74. + ssqa*(14. - ssqa))
            Tx[s<2.] *= -3./(40.*np.pi) * sa * ssqa * np.sqrt(4.-sa**2)
            Tx[s<2.] += 18./np.pi * ssqa * np.arccos(sa/2.)
        Tx *= ssq

        xi, xi_im, varxi = self.calculateXi(rg)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Txxi = Tx.dot(xi)
        Txxi_im = Tx.dot(xi_im)
        nmap = Txxi * self.bin_size
        nmx = Txxi_im * self.bin_size

        # The variance of each of these is
        # Var(<NMap>(R)) = int_r=0..2R [s^4 dlogr^2 Tx(s)^2 Var(xi)]
        varnmap = (Tx**2).dot(varxi) * self.bin_size**2

        return nmap, nmx, varnmap


    def writeNMap(self, file_name, R=None, rg=None, m2_uform=None, file_type=None, precision=None):
        """Write the cross correlation of the foreground galaxy counts with the aperture mass
        based on the correlation function to the file, file_name.

        If rg is provided, the compensated calculation will be used for :math:`\\xi`.

        See `calculateNMap` for an explanation of the **m2_uform** parameter.

        The output file will include the following columns:

        ==========      ====================================================
        Column          Description
        ==========      ====================================================
        R               The radius of the aperture.
        NMap            The mean value <N_ap M_ap>(R)
        NMx             The mean value <N_ap M_x>(R)
        sig_nmap        The sqrt of the variance estimate of either of these
        ==========      ====================================================


        Parameters:
            file_name (str):    The name of the file to write to.
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            m2_uform (str):     Which form to use for the aperture mass.  (default: 'Crittenden';
                                this value can also be given in the constructor in the config dict.)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing NMap from NG correlations to %s',file_name)
        if R is None:
            R = self.rnom

        nmap, nmx, varnmap = self.calculateNMap(R=R, rg=rg, m2_uform=m2_uform)
        if precision is None:
            precision = self.config.get('precision', 4)

        treecorr.util.gen_write(
            file_name,
            ['R','NMap','NMx','sig_nmap'],
            [ R, nmap, nmx, np.sqrt(varnmap) ],
            precision=precision, file_type=file_type, logger=self.logger)


    def writeNorm(self, file_name, gg, dd, rr, R=None, dr=None, rg=None,
                  m2_uform=None, file_type=None, precision=None):
        """Write the normalized aperture mass cross-correlation to the file, file_name.

        The combination :math:`\\langle N M_{ap}\\rangle^2 / \\langle M_{ap}^2\\rangle
        \\langle N_{ap}^2\\rangle` is related to :math:`r`, the galaxy-mass correlation
        coefficient.  Similarly, :math:`\\langle N_{ap}^2\\rangle / \\langle M_{ap}^2\\rangle`
        is related to :math:`b`, the galaxy bias parameter.  cf. Hoekstra et al, 2002:
        http://adsabs.harvard.edu/abs/2002ApJ...577..604H

        This function computes these combinations and outputs them to a file.

        - if rg is provided, the compensated calculation will be used for
          :math:`\\langle N_{ap} M_{ap} \\rangle`.
        - if dr is provided, the compensated calculation will be used for
          :math:`\\langle N_{ap}^2 \\rangle`.

        See `calculateNMap` for an explanation of the **m2_uform** parameter.

        The output file will include the following columns:

        ==========      ====================================================
        Column          Description
        ==========      ====================================================
        R               The radius of the aperture
        NMap            An estimate of <N_ap M_ap>(R)
        NMx             An estimate of <N_ap M_x>(R)
        sig_nmap        The sqrt of the variance estimate of either of these
        Napsq           An estimate of <N_ap^2>(R)
        sig_napsq       The sqrt of the variance estimate of <N_ap^2>
        Mapsq           An estimate of <M_ap^2>(R)
        sig_mapsq       The sqrt of the variance estimate of <M_ap^2>
        NMap_norm       The ratio <N_ap M_ap>^2 / <N_ap^2> <M_{ap}^2>
        sig_norm        The sqrt of the variance estimate of this ratio
        Nsq_Mapsq       The ratio <N_ap^2>/<M_ap^2>
        sig_nn_mm       The sqrt of the variance estimate of this ratio
        ==========      ====================================================

        Parameters:
            file_name (str):    The name of the file to write to.
            gg (GGCorrelation): The auto-correlation of the shear field
            dd (NNCorrelation): The auto-correlation of the lens counts (DD)
            rr (NNCorrelation): The auto-correlation of the random field (RR)
            R (array):          The R values at which to calculate the aperture mass statistics.
                                (default: None, which means use self.rnom)
            dr (NNCorrelation): The cross-correlation of the data with randoms (DR), if
                                desired, in which case the Landy-Szalay estimator will be
                                calculated.  (default: None)
            rd (NNCorrelation): The cross-correlation of the randoms with data (RD), if
                                desired. (default: None, which means use rd=dr)
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)
            m2_uform (str):     Which form to use for the aperture mass.  (default: 'Crittenden';
                                this value can also be given in the constructor in the config dict.)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing Norm from NG correlations to %s',file_name)
        if R is None:
            R = self.rnom

        nmap, nmx, varnmap = self.calculateNMap(R=R, rg=rg, m2_uform=m2_uform)
        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(R=R, m2_uform=m2_uform)
        nsq, varnsq = dd.calculateNapSq(R=R, rr=rr, dr=dr, m2_uform=m2_uform)

        nmnorm = nmap**2 / (nsq * mapsq)
        varnmnorm = nmnorm**2 * (4. * varnmap / nmap**2 + varnsq / nsq**2 + varmapsq / mapsq**2)
        nnnorm = nsq / mapsq
        varnnnorm = nnnorm**2 * (varnsq / nsq**2 + varmapsq / mapsq**2)
        if precision is None:
            precision = self.config.get('precision', 4)

        treecorr.util.gen_write(
            file_name,
            [ 'R',
              'NMap','NMx','sig_nmap',
              'Napsq','sig_napsq','Mapsq','sig_mapsq',
              'NMap_norm','sig_norm','Nsq_Mapsq','sig_nn_mm' ],
            [ R,
              nmap, nmx, np.sqrt(varnmap),
              nsq, np.sqrt(varnsq), mapsq, np.sqrt(varmapsq),
              nmnorm, np.sqrt(varnmnorm), nnnorm, np.sqrt(varnnnorm) ],
            precision=precision, file_type=file_type, logger=self.logger)


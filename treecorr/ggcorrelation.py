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
.. module:: ggcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarG
from .zzcorrelation import BaseZZCorrelation
from .util import make_writer, make_reader
from .config import make_minimal_config


class GGCorrelation(BaseZZCorrelation):
    r"""This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:      The number of bins in logr
        bin_size:   The size of the bins in logr
        min_sep:    The minimum separation being considered
        max_sep:    The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

    Attributes:

        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        meanr:      The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:   The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xip:        The correlation function, :math:`\xi_+(r)`.
        xim:        The correlation function, :math:`\xi_-(r)`.
        xip_im:     The imaginary part of :math:`\xi_+(r)`.
        xim_im:     The imaginary part of :math:`\xi_-(r)`.
        varxip:     An estimate of the variance of :math:`\xi_+(r)`
        varxim:     An estimate of the variance of :math:`\xi_-(r)`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix for the data vector with
                    :math:`\xi_+` first and then :math:`\xi_-`.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxip``,
        ``varxim``, and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr2.process` command and use
        `BaseZZCorrelation.process_auto` and/or `Corr2.process_cross`, then the units will not be
        applied to ``meanr`` or ``meanlogr`` until the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> gg = treecorr.GGCorrelation(config)
        >>> gg.process(cat)         # For auto-correlation.
        >>> gg.process(cat1,cat2)   # For cross-correlation.
        >>> gg.write(file_name)     # Write out to a file.
        >>> xip = gg.xip            # Or access the correlation function directly.

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
    _cls = 'GGCorrelation'
    _letter1 = 'G'
    _letter2 = 'G'
    _letters = 'GG'
    _builder = _treecorr.GGCorr
    _calculateVar1 = staticmethod(calculateVarG)
    _calculateVar2 = staticmethod(calculateVarG)

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `GGCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        The `BaseZZCorrelation.process_auto` and `Corr2.process_cross` commands accumulate values
        in each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            varg1 (float):  The variance per component of the first shear field.
            varg2 (float):  The variance per component of the second shear field.
        """
        super().finalize(varg1, varg2)

    def calculateMapSq(self, *, R=None, m2_uform=None):
        r"""Calculate the aperture mass statistics from the correlation function.

        .. math::

            \langle M_{ap}^2 \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{2R^2}
            \left [ T_+\left(\frac{r}{R}\right) \xi_+(r) +
            T_-\left(\frac{r}{R}\right) \xi_-(r) \right] \\
            \langle M_\times^2 \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{2R^2}
            \left[ T_+\left(\frac{r}{R}\right) \xi_+(r) -
            T_-\left(\frac{r}{R}\right) \xi_-(r) \right]

        The ``m2_uform`` parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If ``m2_uform`` is 'Crittenden':

        .. math::

            U(r) &= \frac{1}{2\pi} (1-r^2) \exp(-r^2/2) \\
            Q(r) &= \frac{1}{4\pi} r^2 \exp(-r^2/2) \\
            T_+(s) &= \frac{s^4 - 16s^2 + 32}{128} \exp(-s^2/4) \\
            T_-(s) &= \frac{s^4}{128} \exp(-s^2/4) \\
            rmax &= \infty

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If ``m2_uform`` is 'Schneider':

        .. math::

            U(r) &= \frac{9}{\pi} (1-r^2) (1/3-r^2) \\
            Q(r) &= \frac{6}{\pi} r^2 (1-r^2) \\
            T_+(s) &= \frac{12}{5\pi} (2-15s^2) \arccos(s/2) \\
            &\qquad + \frac{1}{100\pi} s \sqrt{4-s^2} (120 + 2320s^2 - 754s^4 + 132s^6 - 9s^8) \\
            T_-(s) &= \frac{3}{70\pi} s^3 (4-s^2)^{7/2} \\
            rmax &= 2R

        cf. Schneider, et al (2002): A&A, 389, 729

        .. note::

            This function is only implemented for Log binning.


        Parameters:
            R (array):      The R values at which to calculate the aperture mass statistics.
                            (default: None, which means use self.rnom)
            m2_uform (str): Which form to use for the aperture mass, as described above.
                            (default: 'Crittenden'; this value can also be given in the
                            constructor in the config dict.)

        Returns:
            Tuple containing

                - mapsq = array of :math:`\langle M_{ap}^2 \rangle(R)`
                - mapsq_im = the imaginary part of mapsq, which is an estimate of
                  :math:`\langle M_{ap} M_\times \rangle(R)`
                - mxsq = array of :math:`\langle M_\times^2 \rangle(R)`
                - mxsq_im = the imaginary part of mxsq, which is an estimate of
                  :math:`\langle M_{ap} M_\times \rangle(R)`
                - varmapsq = array of the variance estimate of either mapsq or mxsq
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform', 'Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")
        if self.bin_type != 'Log':
            raise ValueError("calculateMapSq requires Log binning.")
        if R is None:
            R = self.rnom

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        s = np.outer(1./R, self.meanr)
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
        varmapsq = (Tp**2).dot(self.varxip) + (Tm**2).dot(self.varxim)
        varmapsq *= 0.25 * self.bin_size**2

        return mapsq, mapsq_im, mxsq, mxsq_im, varmapsq

    def calculateGamSq(self, *, R=None, eb=False):
        r"""Calculate the tophat shear variance from the correlation function.

        .. math::

            \langle \gamma^2 \rangle(R) &= \int_0^{2R} \frac{r dr}{R^2} S_+(s) \xi_+(r) \\
            \langle \gamma^2 \rangle_E(R) &= \int_0^{2R} \frac{r dr}{2 R^2}
            \left[ S_+\left(\frac{r}{R}\right) \xi_+(r) +
            S_-\left(\frac{r}{R}\right) \xi_-(r) \right] \\
            \langle \gamma^2 \rangle_B(R) &= \int_0^{2R} \frac{r dr}{2 R^2}
            \left[ S_+\left(\frac{r}{R}\right) \xi_+(r) -
            S_-\left(\frac{r}{R}\right) \xi_-(r) \right] \\

            S_+(s) &= \frac{1}{\pi} \left(4 \arccos(s/2) - s \sqrt{4-s^2} \right) \\
            S_-(s) &= \begin{cases}
            s<=2, & \frac{1}{\pi s^4} \left(s \sqrt{4-s^2} (6-s^2) - 8(3-s^2) \arcsin(s/2)\right)\\
            s>=2, & \frac{1}{s^4} \left(4(s^2-3)\right)
            \end{cases}

        cf. Schneider, et al (2002): A&A, 389, 729

        The default behavior is not to compute the E/B versions.  They are calculated if
        eb is set to True.

        .. note::

            This function is only implemented for Log binning.


        Parameters:
            R (array):  The R values at which to calculate the shear variance.
                        (default: None, which means use self.rnom)
            eb (bool):  Whether to include the E/B decomposition as well as the total
                        :math:`\langle \gamma^2\rangle`.  (default: False)

        Returns:
            Tuple containing

                - gamsq = array of :math:`\langle \gamma^2 \rangle(R)`
                - vargamsq = array of the variance estimate of gamsq
                - gamsq_e  (Only if eb is True) = array of :math:`\langle \gamma^2 \rangle_E(R)`
                - gamsq_b  (Only if eb is True) = array of :math:`\langle \gamma^2 \rangle_B(R)`
                - vargamsq_e  (Only if eb is True) = array of the variance estimate of
                  gamsq_e or gamsq_b
        """
        if self.bin_type != 'Log':
            raise ValueError("calculateGamSq requires Log binning.")

        if R is None:
            R = self.rnom
        s = np.outer(1./R, self.meanr)
        ssq = s*s
        Sp = np.zeros_like(s)
        sa = s[s<2]
        ssqa = ssq[s<2]
        Sp[s<2.] = 1./np.pi * ssqa * (4.*np.arccos(sa/2.) - sa*np.sqrt(4.-ssqa))

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Spxip = Sp.dot(self.xip)
        gamsq = Spxip * self.bin_size
        vargamsq = (Sp**2).dot(self.varxip) * self.bin_size**2

        # Stop here if eb is False
        if not eb: return gamsq, vargamsq

        Sm = np.empty_like(s)
        Sm[s<2.] = 1./(ssqa*np.pi) * (sa*np.sqrt(4.-ssqa)*(6.-ssqa)
                                      -8.*(3.-ssqa)*np.arcsin(sa/2.))
        Sm[s>=2.] = 4.*(ssq[s>=2]-3.)/ssq[s>=2]
        # This already includes the extra ssq factor.

        Smxim = Sm.dot(self.xim)
        gamsq_e = (Spxip + Smxim) * 0.5 * self.bin_size
        gamsq_b = (Spxip - Smxim) * 0.5 * self.bin_size
        vargamsq_e = (Sp**2).dot(self.varxip) + (Sm**2).dot(self.varxim)
        vargamsq_e *= 0.25 * self.bin_size**2

        return gamsq, vargamsq, gamsq_e, gamsq_b, vargamsq_e

    def writeMapSq(self, file_name, *, R=None, m2_uform=None, file_type=None, precision=None):
        r"""Write the aperture mass statistics based on the correlation function to the
        file, file_name.

        See `calculateMapSq` for an explanation of the ``m2_uform`` parameter.

        The output file will include the following columns:

        =========       ==========================================================
        Column          Description
        =========       ==========================================================
        R               The aperture radius
        Mapsq           The real part of :math:`\langle M_{ap}^2\rangle`
                         (cf. `calculateMapSq`)
        Mxsq            The real part of :math:`\langle M_\times^2\rangle`
        MMxa            The imag part of :math:`\langle M_{ap}^2\rangle`:
                         an estimator of :math:`\langle M_{ap} M_\times\rangle`
        MMxa            The imag part of :math:`\langle M_\times^2\rangle`:
                         an estimator of :math:`\langle M_{ap} M_\times\rangle`
        sig_map         The sqrt of the variance estimate of
                         :math:`\langle M_{ap}^2\rangle`
        Gamsq           The tophat shear variance :math:`\langle \gamma^2\rangle`
                         (cf. `calculateGamSq`)
        sig_gam         The sqrt of the variance estimate of
                         :math:`\langle \gamma^2\rangle`
        =========       ==========================================================

        Parameters:
            file_name (str):    The name of the file to write to.
            R (array):          The R values at which to calculate the statistics.
                                (default: None, which means use self.rnom)
            m2_uform (str):     Which form to use for the aperture mass.  (default: 'Crittenden';
                                this value can also be given in the constructor in the config dict.)
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing Map^2 from GG correlations to %s',file_name)

        if R is None:
            R = self.rnom
        mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = self.calculateMapSq(R=R, m2_uform=m2_uform)
        gamsq, vargamsq = self.calculateGamSq(R=R)
        if precision is None:
            precision = self.config.get('precision', 4)

        col_names = ['R','Mapsq','Mxsq','MMxa','MMxb','sig_map','Gamsq','sig_gam']
        columns = [ R,
                    mapsq, mxsq, mapsq_im, -mxsq_im, np.sqrt(varmapsq),
                    gamsq, np.sqrt(vargamsq) ]
        with make_writer(file_name, precision, file_type, logger=self.logger) as writer:
            writer.write(col_names, columns)

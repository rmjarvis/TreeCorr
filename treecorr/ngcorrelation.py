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
.. module:: ngcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarG
from .util import make_writer, make_reader
from .config import make_minimal_config
from .nzcorrelation import BaseNZCorrelation


class NGCorrelation(BaseNZCorrelation):
    r"""This class handles the calculation and storage of a 2-point count-shear correlation
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
        logr:       The nominal center of the bin in log(r) (the natural logarithm of r).
        rnom:       The nominal center of the bin converted to regular distance.
                    i.e. r = exp(logr).
        meanr:      The (weighted) mean value of r for the pairs in each bin.
                    If there are no pairs in a bin, then exp(logr) will be used instead.
        meanlogr:   The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xi:         The correlation function, :math:`\xi(r) = \langle \gamma_T\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.
        raw_xi:     The raw value of xi, uncorrected by an RG calculation. cf. `calculateXi`
        raw_xi_im:  The raw value of xi_im, uncorrected by an RG calculation. cf. `calculateXi`
        raw_varxi:  The raw value of varxi, uncorrected by an RG calculation. cf. `calculateXi`

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into
        the final correlation.  This does not include sample variance, so it is always an
        underestimate of the actual variance.  To get better estimates, you need to set
        ``var_method`` to something else and use patches in the input catalog(s).
        cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr2.process` command and use `Corr2.process_cross`,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    The typical usage pattern is as follows:

        >>> ng = treecorr.NGCorrelation(config)
        >>> ng.process(cat1,cat2)   # Compute the cross-correlation.
        >>> ng.write(file_name)     # Write out to a file.
        >>> xi = ng.xi              # Or access the correlation function directly.

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
    _cls = 'NGCorrelation'
    _letter1 = 'N'
    _letter2 = 'G'
    _letters = 'NG'
    _builder = _treecorr.NGCorr
    _calculateVar1 = lambda *args, **kwargs: None
    _calculateVar2 = staticmethod(calculateVarG)
    _zreal = 'gamT'
    _zimag = 'gamX'

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `NGCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        The `Corr2.process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            varg (float):   The variance per component of the shear field.
        """
        super().finalize(varg)

    def calculateXi(self, *, rg=None):
        r"""Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rg is None, the simple correlation function :math:`\langle \gamma_T\rangle` is
          returned.
        - If rg is not None, then a compensated calculation is done:
          :math:`\langle \gamma_T\rangle = (DG - RG)`, where DG represents the mean shear
          around the lenses and RG represents the mean shear around random points.

        After calling this function, the attributes ``xi``, ``xi_im``, ``varxi``, and ``cov`` will
        correspond to the compensated values (if rg is provided).  The raw, uncompensated values
        are available as ``rawxi``, ``raw_xi_im``, and ``raw_varxi``.

        Parameters:
            rg (NGCorrelation): The cross-correlation using random locations as the lenses
                                (RG), if desired.  (default: None)

        Returns:
            Tuple containing

                - xi = array of the real part of :math:`\xi(R)`
                - xi_im = array of the imaginary part of :math:`\xi(R)`
                - varxi = array of the variance estimates of the above values
        """
        return super().calculateXi(rz=rg)

    def write(self, file_name, *, rg=None, file_type=None, precision=None,
              write_patch_results=False, write_cov=False):
        r"""Write the correlation function to the file, file_name.

        - If rg is None, the simple correlation function :math:`\langle \gamma_T\rangle` is used.
        - If rg is not None, then a compensated calculation is done:
          :math:`\langle \gamma_T\rangle = (DG - RG)`, where DG represents the mean shear
          around the lenses and RG represents the mean shear around random points.

        The output file will include the following columns:

        ==========      =============================================================
        Column          Description
        ==========      =============================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that fell
                        into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs that
                        fell into each bin
        gamT            The real part of the mean tangential shear,
                        :math:`\langle \gamma_T \rangle(r)`
        gamX            The imag part of the mean tangential shear,
                        :math:`\langle \gamma_\times \rangle(r)`
        sigma           The sqrt of the variance estimate of either of these
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      =============================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
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
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        super().write(file_name, rg, file_type, precision, write_patch_results, write_cov)

    def calculateNMap(self, *, R=None, rg=None, m2_uform=None):
        r"""Calculate the aperture mass statistics from the correlation function.

        .. math::

            \langle N M_{ap} \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{R^2}
            T_\times\left(\frac{r}{R}\right) \Re\xi(r) \\
            \langle N M_{\times} \rangle(R) &= \int_{0}^{rmax} \frac{r dr}{R^2}
            T_\times\left(\frac{r}{R}\right) \Im\xi(r)

        The ``m2_uform`` parameter sets which definition of the aperture mass to use.
        The default is to use 'Crittenden'.

        If ``m2_uform`` is 'Crittenden':

        .. math::

            U(r) &= \frac{1}{2\pi} (1-r^2) \exp(-r^2/2) \\
            T_\times(s) &= \frac{s^2}{128} (12-s^2) \exp(-s^2/4)

        cf. Crittenden, et al (2002): ApJ, 568, 20

        If ``m2_uform`` is 'Schneider':

        .. math::

            U(r) &= \frac{9}{\pi} (1-r^2) (1/3-r^2) \\
            T_\times(s) &= \frac{18}{\pi} s^2 \arccos(s/2) \\
            &\qquad - \frac{3}{40\pi} s^3 \sqrt{4-s^2} (196 - 74s^2 + 14s^4 - s^6)

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

                - nmap = array of :math:`\langle N M_{ap} \rangle(R)`
                - nmx = array of :math:`\langle N M_{\times} \rangle(R)`
                - varnmap = array of variance estimates of the above values
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform','Crittenden')
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

        xi, xi_im, varxi = self.calculateXi(rg=rg)

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

    def writeNMap(self, file_name, *, R=None, rg=None, m2_uform=None, file_type=None,
                  precision=None):
        r"""Write the cross correlation of the foreground galaxy counts with the aperture mass
        based on the correlation function to the file, file_name.

        If rg is provided, the compensated calculation will be used for :math:`\xi`.

        See `calculateNMap` for an explanation of the ``m2_uform`` parameter.

        The output file will include the following columns:

        ==========      =========================================================
        Column          Description
        ==========      =========================================================
        R               The radius of the aperture.
        NMap            An estimate of :math:`\langle N_{ap} M_{ap} \rangle(R)`
        NMx             An estimate of :math:`\langle N_{ap} M_\times \rangle(R)`
        sig_nmap        The sqrt of the variance estimate of either of these
        ==========      =========================================================


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

        col_names = ['R','NMap','NMx','sig_nmap']
        columns = [ R, nmap, nmx, np.sqrt(varnmap) ]
        writer = make_writer(file_name, precision, file_type, logger=self.logger)
        with writer:
            writer.write(col_names, columns)

    def writeNorm(self, file_name, *, gg, dd, rr, R=None, dr=None, rg=None,
                  m2_uform=None, file_type=None, precision=None):
        r"""Write the normalized aperture mass cross-correlation to the file, file_name.

        The combination :math:`\langle N M_{ap}\rangle^2 / \langle M_{ap}^2\rangle
        \langle N_{ap}^2\rangle` is related to :math:`r`, the galaxy-mass correlation
        coefficient.  Similarly, :math:`\langle N_{ap}^2\rangle / \langle M_{ap}^2\rangle`
        is related to :math:`b`, the galaxy bias parameter.  cf. Hoekstra et al, 2002:
        http://adsabs.harvard.edu/abs/2002ApJ...577..604H

        This function computes these combinations and outputs them to a file.

        - if rg is provided, the compensated calculation will be used for
          :math:`\langle N_{ap} M_{ap} \rangle`.
        - if dr is provided, the compensated calculation will be used for
          :math:`\langle N_{ap}^2 \rangle`.

        See `calculateNMap` for an explanation of the ``m2_uform`` parameter.

        The output file will include the following columns:

        ==========      =====================================================================
        Column          Description
        ==========      =====================================================================
        R               The radius of the aperture
        NMap            An estimate of :math:`\langle N_{ap} M_{ap} \rangle(R)`
        NMx             An estimate of :math:`\langle N_{ap} M_\times \rangle(R)`
        sig_nmap        The sqrt of the variance estimate of either of these
        Napsq           An estimate of :math:`\langle N_{ap}^2 \rangle(R)`
        sig_napsq       The sqrt of the variance estimate of :math:`\langle N_{ap}^2 \rangle`
        Mapsq           An estimate of :math:`\langle M_{ap}^2 \rangle(R)`
        sig_mapsq       The sqrt of the variance estimate of :math:`\langle M_{ap}^2 \rangle`
        NMap_norm       The ratio :math:`\langle N_{ap} M_{ap} \rangle^2 /`
                        :math:`\langle N_{ap}^2 \rangle \langle M_{ap}^2 \rangle`
        sig_norm        The sqrt of the variance estimate of this ratio
        Nsq_Mapsq       The ratio :math:`\langle N_{ap}^2 \rangle / \langle M_{ap}^2 \rangle`
        sig_nn_mm       The sqrt of the variance estimate of this ratio
        ==========      =====================================================================

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

        col_names = [ 'R',
                      'NMap','NMx','sig_nmap',
                      'Napsq','sig_napsq','Mapsq','sig_mapsq',
                      'NMap_norm','sig_norm','Nsq_Mapsq','sig_nn_mm' ]
        columns = [ R,
                    nmap, nmx, np.sqrt(varnmap),
                    nsq, np.sqrt(varnsq), mapsq, np.sqrt(varmapsq),
                    nmnorm, np.sqrt(varnmnorm), nnnorm, np.sqrt(varnnnorm) ]
        writer = make_writer(file_name, precision, file_type, logger=self.logger)
        with writer:
            writer.write(col_names, columns)

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
from .catalog import calculateVarK
from .corr3base import Corr3
from .util import make_writer, make_reader
from .config import make_minimal_config


class KKKCorrelation(Corr3):
    r"""This class handles the calculation and storage of a 3-point scalar-scalar-scalar correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of the CMB temperature fluctuations, where "kappa" would really be
        :math:`\Delta T`.

    See the doc string of `Corr3` for a description of how the triangles are binned.

    Ojects of this class holds the following attributes:

    Attributes:
        nbins:      The number of bins in logr where r = d2.
        bin_size:   The size of the bins in logr.
        min_sep:    The minimum separation being considered.
        max_sep:    The maximum separation being considered.
        logr1d:     The nominal centers of the nbins bins in log(r).

    If the bin_type is LogRUV, then it will have these attributes:

    Attributes:
        nubins:     The number of bins in u where u = d3/d2.
        ubin_size:  The size of the bins in u.
        min_u:      The minimum u being considered.
        max_u:      The maximum u being considered.
        nvbins:     The number of bins in v where v = +-(d1-d2)/d3.
        vbin_size:  The size of the bins in v.
        min_v:      The minimum v being considered.
        max_v:      The maximum v being considered.
        u1d:        The nominal centers of the nubins bins in u.
        v1d:        The nominal centers of the nvbins bins in v.

    If the bin_type is LogSAS, then it will have these attributes:

    Attributes:
        nphi_bins:  The number of bins in phi.
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
        meanu:      The (weighted) mean value of u for the triangles in each bin.
        meanv:      The (weighted) mean value of v for the triangles in each bin.

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
        zeta:       The correlation function, :math:`\zeta`.
        varzeta:    The variance of :math:`\zeta`, only including the shot noise propagated into
                    the final correlation.  This does not include sample variance, so it is always
                    an underestimate of the actual variance.
        meand1:     The (weighted) mean value of d1 for the triangles in each bin.
        meanlogd1:  The (weighted) mean value of log(d1) for the triangles in each bin.
        meand2:     The (weighted) mean value of d2 (aka r) for the triangles in each bin.
        meanlogd2:  The (weighted) mean value of log(d2) for the triangles in each bin.
        meand3:     The (weighted) mean value of d3 for the triangles in each bin.
        meanlogd3:  The (weighted) mean value of log(d3) for the triangles in each bin.
        weight:     The total weight in each bin.
        ntri:       The number of triangles going into each bin (including those where one or
                    more objects have w=0).

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr3.process` command and use `process_auto` and/or
        `Corr3.process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
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
                        in `Corr3`, which are ignored here. (default: None)
        logger:         If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)

    Keyword Arguments:
        **kwargs:       See the documentation for `Corr3` for the list of allowed keyword
                        arguments, which may be passed either directly or in the config dict.
    """
    _cls = 'KKKCorrelation'
    _letter1 = 'K'
    _letter2 = 'K'
    _letter3 = 'K'
    _letters = 'KKK'
    _builder = _treecorr.KKKCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarK)
    _calculateVar3 = staticmethod(calculateVarK)
    _sig1 = 'sig_k'
    _sig2 = 'sig_k'
    _sig3 = 'sig_k'
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `KKKCorrelation`.  See class doc for details.
        """
        Corr3.__init__(self, config, logger=logger, **kwargs)

        shape = self.data_shape
        self._z1 = np.zeros(shape, dtype=float)
        x = np.array([])
        if self.bin_type == 'LogMultipole':
            self._z2 = np.zeros(shape, dtype=float)
        else:
            self._z2 = x
        self._varzeta = None
        self._z3 = self._z4 = self._z5 = self._z6 = self._z7 = self._z8 = x
        self.logger.debug('Finished building KKKCorr')

    @property
    def zeta(self):
        if self._z2.size:
            return self._z1 + 1j * self._z2
        else:
            return self._z1

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

    def finalize(self, vark1, vark2, vark3):
        """Finalize the calculation of the correlation function.

        The `process_auto`, `process_cross12` and `Corr3.process_cross` commands accumulate values
        in each bin, so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing by the total weight.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
            vark3 (float):  The variance of the third scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2 * vark3

        # I don't really understand why the variance is coming out 2x larger than the normal
        # formula for LogSAS.  But with just Gaussian noise, I need to multiply the numerator
        # by two to get the variance estimates to come out right.
        if self.bin_type in ['LogSAS', 'LogMultipole']:
            self._var_num *= 2

    @property
    def varzeta(self):
        if self._varzeta is None:
            self._varzeta = np.zeros(self.data_shape)
            if self._var_num != 0:
                self._varzeta.ravel()[:] = self.cov_diag
        return self._varzeta

    def _clear(self):
        super()._clear()
        self._varzeta = None

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c.meand1 for c in others], axis=0, out=self.meand1)
        np.sum([c.meanlogd1 for c in others], axis=0, out=self.meanlogd1)
        np.sum([c.meand2 for c in others], axis=0, out=self.meand2)
        np.sum([c.meanlogd2 for c in others], axis=0, out=self.meanlogd2)
        np.sum([c.meand3 for c in others], axis=0, out=self.meand3)
        np.sum([c.meanlogd3 for c in others], axis=0, out=self.meanlogd3)
        np.sum([c.meanu for c in others], axis=0, out=self.meanu)
        if self.bin_type == 'LogRUV':
            np.sum([c.meanv for c in others], axis=0, out=self.meanv)
        np.sum([c._z1 for c in others], axis=0, out=self._z1)
        np.sum([c.weightr for c in others], axis=0, out=self.weightr)
        if self.bin_type == 'LogMultipole':
            np.sum([c._z2 for c in others], axis=0, out=self._z2)
            np.sum([c.weighti for c in others], axis=0, out=self.weighti)
        np.sum([c.ntri for c in others], axis=0, out=self.ntri)

    def toSAS(self, *, target=None, **kwargs):
        """Convert a multipole-binned correlation to the corresponding SAS binning.

        This is only valid for bin_type == LogMultipole.

        Keyword Arguments:
            target:     A target KKKCorrelation object with LogSAS binning to write to.
                        If this is not given, a new object will be created based on the
                        configuration paramters of the current object. (default: None)
            **kwargs:   Any kwargs that you want to use to configure the returned object.
                        Typically, might include min_phi, max_phi, nphi_bins, phi_bin_size.
                        The default phi binning is [0,pi] with nphi_bins = self.max_n.

        Returns:
            A KKKCorrelation object with bin_type=LogSAS containing the
            same information as this object, but with the SAS binning.
        """
        sas = super().toSAS(target=target, **kwargs)

        sas._var_num = self._var_num

        # Z(d2,d3,phi) = 1/2pi sum_n Z_n(d2,d3) exp(i n phi)
        expiphi = np.exp(1j * self.n1d[:,None] * sas.phi1d)
        sas._z1[:] = np.real(self.zeta.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size
        mask = sas.weightr != 0
        sas._z1[mask] /= sas.weightr[mask]

        for k,v in self.results.items():
            sas.results[k]._z1[:] = np.real(v.zeta.dot(expiphi)) / (2*np.pi) * sas.phi_bin_size

        return sas

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        r"""Write the correlation function to the file, file_name.

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
        n               The multipole index n (from -max_n .. max_n)
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
        zeta            The estimator of :math:`\zeta` (For LogMultipole, this is split
                        into real and imaginary parts, zeta_re and zeta_im.)
        sigma_zeta      The sqrt of the variance estimate of :math:`\zeta`
                        (if rrr is given)
        weight          The total weight of triangles contributing to each bin.
                        (For LogMultipole, this is split into real and imaginary parts,
                        weight_re and weight_im.)
        ntri            The number of triangles contributing to each bin
        ==========      ================================================================

        If ``sep_units`` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        Parameters:
            file_name (str):    The name of the file to write to.
            file_type (str):    The type of file to write ('ASCII' or 'FITS').  (default: determine
                                the type automatically from the extension of file_name.)
            precision (int):    For ASCII output catalogs, the desired precision. (default: 4;
                                this value can also be given in the constructor in the config dict.)
            write_patch_results (bool): Whether to write the patch-based results as well.
                                        (default: False)
            write_cov (bool):   Whether to write the covariance matrix as well. (default: False)
        """
        self.logger.info('Writing KKK correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov)

    @property
    def _write_col_names(self):
        if self.bin_type == 'LogRUV':
            col_names = ['r_nom', 'u_nom', 'v_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanu', 'meanv']
        elif self.bin_type == 'LogSAS':
            col_names = ['d2_nom', 'd3_nom', 'phi_nom',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3', 'meanphi']
        else:
            col_names = ['d2_nom', 'd3_nom', 'n',
                         'meand1', 'meanlogd1', 'meand2', 'meanlogd2',
                         'meand3', 'meanlogd3']
        if self.bin_type == 'LogMultipole':
            col_names += ['zeta_re', 'zeta_im', 'sigma_zeta', 'weight_re', 'weight_im', 'ntri']
        else:
            col_names += ['zeta', 'sigma_zeta', 'weight', 'ntri']
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
            data = [ self.d2nom, self.d3nom, self.n,
                     self.meand1, self.meanlogd1, self.meand2, self.meanlogd2,
                     self.meand3, self.meanlogd3 ]
        if self.bin_type == 'LogMultipole':
            data += [ self._z1, self._z2, np.sqrt(self.varzeta),
                      self.weightr, self.weighti, self.ntri ]
        else:
            data += [ self.zeta, np.sqrt(self.varzeta), self.weight, self.ntri ]
        data = [ col.flatten() for col in data ]
        return data

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.data_shape
        if self.bin_type == 'LogMultipole':
            self._z1 = data['zeta_re'].reshape(s)
            self._z2 = data['zeta_im'].reshape(s)
        else:
            self._z1 = data['zeta'].reshape(s)
        self._varzeta = data['sigma_zeta'].reshape(s)**2

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
from .catalog import calculateVarZ
from .corr2base import Corr2
from .util import make_writer
from .config import make_minimal_config


class BaseZZCorrelation(Corr2):
    """This class is a base class for all the ??Correlation classes, where both ?'s are one of the
    complex fields of varying spin.

    A lot of the implementation is shared among those types, so whenever possible the shared
    implementation is done in this class.
    """
    _sig1 = 'sig_sn (per component)'
    _sig2 = 'sig_sn (per component)'

    def __init__(self, config=None, *, logger=None, **kwargs):
        super().__init__(config, logger=logger, **kwargs)

        self._xi1 = np.zeros_like(self.rnom, dtype=float)
        self._xi2 = np.zeros_like(self.rnom, dtype=float)
        self._xi3 = np.zeros_like(self.rnom, dtype=float)
        self._xi4 = np.zeros_like(self.rnom, dtype=float)
        self._varxip = None
        self._varxim = None
        self.logger.debug('Finished building %s', self._cls)

    @property
    def xip(self):
        return self._xi1

    @property
    def xip_im(self):
        return self._xi2

    @property
    def xim(self):
        return self._xi3

    @property
    def xim_im(self):
        return self._xi4

    def getStat(self):
        """The standard statistic for the current correlation object as a 1-d array.

        In this case, this is the concatenation of self.xip and self.xim (raveled if necessary).
        """
        return np.concatenate([self.xip.ravel(), self.xim.ravel()])

    def getWeight(self):
        """The weight array for the current correlation object as a 1-d array.

        This is the weight array corresponding to `getStat`. In this case, the weight is
        duplicated to account for both xip and xim returned as part of getStat().
        """
        return np.concatenate([self.weight.ravel(), self.weight.ravel()])

    def process_auto(self, cat, *, metric=None, num_threads=None):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the `finalize` command will
        finish the calculation.

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

    def finalize(self, varz1, varz2):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `Corr2.process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            varz1 (float):  The variance per component of the first field.
            varz2 (float):  The variance per component of the second field.
        """
        self._finalize()
        self._var_num = 2. * varz1 * varz2

    @property
    def varxip(self):
        if self._varxip is None:
            self._varxip = np.zeros_like(self.rnom, dtype=float)
            if self._var_num != 0:
                self._varxip.ravel()[:] = self.cov_diag[:self._nbins]
        return self._varxip

    @property
    def varxim(self):
        if self._varxim is None:
            self._varxim = np.zeros_like(self.rnom, dtype=float)
            if self._var_num != 0:
                self._varxim.ravel()[:] = self.cov_diag[self._nbins:]
        return self._varxim

    def _clear(self):
        super()._clear()
        self._varxip = None
        self._varxim = None

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c._xi1 for c in others], axis=0, out=self._xi1)
        np.sum([c._xi2 for c in others], axis=0, out=self._xi2)
        np.sum([c._xi3 for c in others], axis=0, out=self._xi3)
        np.sum([c._xi4 for c in others], axis=0, out=self._xi4)
        np.sum([c.meanr for c in others], axis=0, out=self.meanr)
        np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
        np.sum([c.weight for c in others], axis=0, out=self.weight)
        np.sum([c.npairs for c in others], axis=0, out=self.npairs)
        self._varxip = None
        self._varxim = None
        self._cov = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        r"""Write the correlation function to the file, file_name.

        The output file will include the following columns:

        =========       ========================================================
        Column          Description
        =========       ========================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that
                        fell into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs
                        that fell into each bin
        xip             The real part of the :math:`\xi_+` correlation function
        xim             The real part of the :math:`\xi_-` correlation function
        xip_im          The imag part of the :math:`\xi_+` correlation function
        xim_im          The imag part of the :math:`\xi_-` correlation function
        sigma_xip       The sqrt of the variance estimate of :math:`\xi_+`
        sigma_xim       The sqrt of the variance estimate of :math:`\xi_-`
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        =========       ========================================================

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
        self.logger.info(f'Writing {self._letters} correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov)

    @property
    def _write_col_names(self):
        return ['r_nom', 'meanr', 'meanlogr', 'xip', 'xim', 'xip_im', 'xim_im',
                'sigma_xip', 'sigma_xim', 'weight', 'npairs']

    @property
    def _write_data(self):
        data = [ self.rnom, self.meanr, self.meanlogr,
                 self.xip, self.xim, self.xip_im, self.xim_im,
                 np.sqrt(self.varxip), np.sqrt(self.varxim),
                 self.weight, self.npairs ]
        data = [ col.flatten() for col in data ]
        return data

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.logr.shape
        self._xi1 = data['xip'].reshape(s)
        self._xi2 = data['xip_im'].reshape(s)
        self._xi3 = data['xim'].reshape(s)
        self._xi4 = data['xim_im'].reshape(s)
        self._varxip = data['sigma_xip'].reshape(s)**2
        self._varxim = data['sigma_xim'].reshape(s)**2


class ZZCorrelation(BaseZZCorrelation):
    r"""This class handles the calculation and storage of a 2-point correlation function
    of two complex spin-0 fields.  If either spin-0 field is real, you should instead use
    `KZCorrelation` as it will be faster, and if both are real, you should use `KKCorrelation`.
    This class is intended for correlations of scalar fields with a complex values that
    don't change with orientation.

    To be consistent with the other spin correlation functions, we compute two quantities:

    .. math::

        \xi_+ = \langle z_1 z_2^* \rangle
        \xi_- = \langle z_1 z_2 \rangle

    There is no projection along the line connecting the two points as there is for the other
    complex fields, since the field values don't change with orientation.

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
        applied to ``meanr`` or ``meanlogr`` until the `BaseZZCorrelation.finalize` function is
        called.

    The typical usage pattern is as follows:

        >>> zz = treecorr.ZZCorrelation(config)
        >>> zz.process(cat)         # For auto-correlation.
        >>> zz.process(cat1,cat2)   # For cross-correlation.
        >>> zz.write(file_name)     # Write out to a file.
        >>> xip = zz.xip            # Or access the correlation function directly.

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
    _cls = 'ZZCorrelation'
    _letter1 = 'Z'
    _letter2 = 'Z'
    _letters = 'ZZ'
    _builder = _treecorr.ZZCorr
    _calculateVar1 = staticmethod(calculateVarZ)
    _calculateVar2 = staticmethod(calculateVarZ)

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `ZZCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

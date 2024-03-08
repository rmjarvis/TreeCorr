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
.. module:: kkcorrelation
"""

import numpy as np

from . import _treecorr
from .catalog import calculateVarK
from .corr2base import Corr2
from .util import make_writer
from .config import make_minimal_config


class KKCorrelation(Corr2):
    r"""This class handles the calculation and storage of a 2-point scalar-scalar correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of the CMB temperature fluctuations, where "kappa" would really be
        :math:`\Delta T`.

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
        xi:         The correlation function, :math:`\xi(r)`
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shot noise propagated into the final
        correlation.  This does not include sample variance, so it is always an underestimate of
        the actual variance.  To get better estimates, you need to set ``var_method`` to something
        else and use patches in the input catalog(s).  cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `Corr2.process` command and use `process_auto`
        and/or `Corr2.process_cross`, then the units will not be applied to ``meanr`` or
        ``meanlogr`` until the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> kk = treecorr.KKCorrelation(config)
        >>> kk.process(cat)         # For auto-correlation.
        >>> kk.process(cat1,cat2)   # For cross-correlation.
        >>> kk.write(file_name)     # Write out to a file.
        >>> xi = kk.xi              # Or access the correlation function directly.

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
    _cls = 'KKCorrelation'
    _letter1 = 'K'
    _letter2 = 'K'
    _letters = 'KK'
    _builder = _treecorr.KKCorr
    _calculateVar1 = staticmethod(calculateVarK)
    _calculateVar2 = staticmethod(calculateVarK)
    _sig1 = 'sig_k'
    _sig2 = 'sig_k'
    # The angles are not important for accuracy of KK correlations.
    _default_angle_slop = 1

    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `KKCorrelation`.  See class doc for details.
        """
        super().__init__(config, logger=logger, **kwargs)

        self._xi1 = np.zeros_like(self.rnom, dtype=float)
        self._xi2 = self._xi3 = self._xi4 = np.array([])
        self._varxi = None
        self.logger.debug('Finished building KKCorr')

    @property
    def xi(self):
        return self._xi1

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

    def finalize(self, vark1, vark2):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `Corr2.process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            vark1 (float):  The variance of the first scalar field.
            vark2 (float):  The variance of the second scalar field.
        """
        self._finalize()
        self._var_num = vark1 * vark2

    @property
    def varxi(self):
        if self._varxi is None:
            self._varxi = np.zeros_like(self.rnom, dtype=float)
            if self._var_num != 0:
                self._varxi.ravel()[:] = self.cov_diag
        return self._varxi

    def _clear(self):
        """Clear the data vectors
        """
        super()._clear()
        self._varxi = None

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c._xi1 for c in others], axis=0, out=self._xi1)
        np.sum([c.meanr for c in others], axis=0, out=self.meanr)
        np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
        np.sum([c.weight for c in others], axis=0, out=self.weight)
        np.sum([c.npairs for c in others], axis=0, out=self.npairs)
        self._varxi = None
        self._cov = None

    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False,
              write_cov=False):
        r"""Write the correlation function to the file, file_name.

        The output file will include the following columns:

        ==========      ========================================================
        Column          Description
        ==========      ========================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r \rangle` of pairs that
                        fell into each bin
        meanlogr        The mean value :math:`\langle \log(r) \rangle` of pairs
                        that fell into each bin
        xi              The estimate of the correlation function xi(r)
        sigma_xi        The sqrt of the variance estimate of xi(r)
        weight          The total weight contributing to each bin
        npairs          The total number of pairs in each bin
        ==========      ========================================================

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
        self.logger.info('Writing KK correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, None, write_patch_results, write_cov=write_cov)

    @property
    def _write_col_names(self):
        return ['r_nom','meanr','meanlogr','xi','sigma_xi','weight','npairs']

    @property
    def _write_data(self):
        data = [ self.rnom, self.meanr, self.meanlogr,
                 self.xi, np.sqrt(self.varxi), self.weight, self.npairs ]
        data = [ col.flatten() for col in data ]
        return data

    def _read_from_data(self, data, params):
        super()._read_from_data(data, params)
        s = self.logr.shape
        self._xi1 = data['xi'].reshape(s)
        self._varxi = data['sigma_xi'].reshape(s)**2

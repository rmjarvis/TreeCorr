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
.. module:: kgcorrelation
"""

import numpy as np

from . import _lib, _ffi
from .catalog import calculateVarG, calculateVarK
from .binnedcorr2 import BinnedCorr2
from .util import double_ptr as dp
from .util import gen_read, gen_write


class KGCorrelation(BinnedCorr2):
    r"""This class handles the calculation and storage of a 2-point kappa-shear correlation
    function.

    .. note::

        While we use the term kappa (:math:`\kappa`) here and the letter K in various places,
        in fact any scalar field will work here.  For example, you can use this to compute
        correlations of some survey property, such as seeing, with shear, where "kappa" would
        really be the measured property, e.g. the observed sizes of the stars.

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
        xi:         The correlation function, :math:`\xi(r) = \langle \kappa\, \gamma_T\rangle`.
        xi_im:      The imaginary part of :math:`\xi(r)`.
        varxi:      An estimate of the variance of :math:`\xi`
        weight:     The total weight in each bin.
        npairs:     The number of pairs going into each bin (including pairs where one or
                    both objects have w=0).
        cov:        An estimate of the full covariance matrix.

    .. note::

        The default method for estimating the variance and covariance attributes (``varxi``,
        and ``cov``) is 'shot', which only includes the shape noise propagated into the final
        correlation.  This does not include sample variance, so it is always an underestimate of
        the actual variance.  To get better estimates, you need to set ``var_method`` to something
        else and use patches in the input catalog(s).  cf. `Covariance Estimates`.

    If ``sep_units`` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.

    .. note::

        If you separate out the steps of the `process` command and use `process_cross`,
        then the units will not be applied to ``meanr`` or ``meanlogr`` until the `finalize`
        function is called.

    The typical usage pattern is as follows:

        >>> kg = treecorr.KGCorrelation(config)
        >>> kg.process(cat1,cat2)   # Calculate the cross-correlation
        >>> kg.write(file_name)     # Write out to a file.
        >>> xi = kg.xi              # Or access the correlation function directly.

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
        """Initialize `KGCorrelation`.  See class doc for details.
        """
        BinnedCorr2.__init__(self, config, logger, **kwargs)

        self._ro._d1 = 2  # KData
        self._ro._d2 = 3  # GData
        self.xi = np.zeros_like(self.rnom, dtype=float)
        self.xi_im = np.zeros_like(self.rnom, dtype=float)
        self.varxi = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self.logger.debug('Finished building KGCorr')

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _lib.BuildCorr2(
                    self._d1, self._d2, self._bintype,
                    self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                    self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                    dp(self.xi),dp(self.xi_im), dp(None), dp(None),
                    dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs))
        return self._corr

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if self._corr is not None:
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        """Return whether two `KGCorrelation` instances are equal"""
        return (isinstance(other, KGCorrelation) and
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
        ret = KGCorrelation.__new__(KGCorrelation)
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
        return ret

    def __repr__(self):
        return 'KGCorrelation(config=%r)'%self.config

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
            self.logger.info('Starting process KG cross-correlations')
        else:
            self.logger.info('Starting process KG cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getKField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute == 1,
                            self.min_top, self.max_top, self.coords)
        f2 = cat2.getGField(min_size, max_size, self.split_method,
                            self.brute is True or self.brute == 2,
                            self.min_top, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        _lib.ProcessCross2(self.corr, f1.data, f2.data, self.output_dots,
                           f1._d, f2._d, self._coords, self._bintype, self._metric)


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the `finalize` command will
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
            self.logger.info('Starting process KG pairwise-correlations')
        else:
            self.logger.info('Starting process KG pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)

        f1 = cat1.getKSimpleField()
        f2 = cat2.getGSimpleField()

        _lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                         f1._d, f2._d, self._coords, self._bintype, self._metric)

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.xi[mask1] /= self.weight[mask1]
        self.xi_im[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]

    def finalize(self, vark, varg):
        """Finalize the calculation of the correlation function.

        The `process_cross` command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        Parameters:
            vark (float):   The kappa variance for the first field.
            varg (float):   The shear variance per component for the second field.
        """
        self._finalize()
        self._var_num = vark * varg
        self.cov = self.estimate_cov(self.var_method)
        self.varxi.ravel()[:] = self.cov.diagonal()

    def _clear(self):
        """Clear the data vectors
        """
        self.xi.ravel()[:] = 0
        self.xi_im.ravel()[:] = 0
        self.meanr.ravel()[:] = 0
        self.meanlogr.ravel()[:] = 0
        self.weight.ravel()[:] = 0
        self.npairs.ravel()[:] = 0

    def __iadd__(self, other):
        """Add a second `KGCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `KGCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, KGCorrelation):
            raise TypeError("Can only add another KGCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("KGCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords, other.coords)
        self.xi.ravel()[:] += other.xi.ravel()[:]
        self.xi_im.ravel()[:] += other.xi_im.ravel()[:]
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        return self

    def _sum(self, others):
        # Equivalent to the operation of:
        #     self._clear()
        #     for other in others:
        #         self += other
        # but no sanity checks and use numpy.sum for faster calculation.
        np.sum([c.xi for c in others], axis=0, out=self.xi)
        np.sum([c.xi_im for c in others], axis=0, out=self.xi_im)
        np.sum([c.meanr for c in others], axis=0, out=self.meanr)
        np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
        np.sum([c.weight for c in others], axis=0, out=self.weight)
        np.sum([c.npairs for c in others], axis=0, out=self.npairs)

    def process(self, cat1, cat2, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the K field.
            cat2 (Catalog):     A catalog or list of catalogs for the G field.
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
        import math
        if initialize:
            self.clear()

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)

        self._process_all_cross(cat1, cat2, metric, num_threads, comm, low_mem)

        if finalize:
            vark = calculateVarK(cat1, low_mem=low_mem)
            varg = calculateVarG(cat2, low_mem=low_mem)
            self.logger.info("vark = %f: sig_k = %f",vark,math.sqrt(vark))
            self.logger.info("varg = %f: sig_sn (per component) = %f",varg,math.sqrt(varg))
            self.finalize(vark,varg)

    def write(self, file_name, file_type=None, precision=None):
        r"""Write the correlation function to the file, file_name.

        The output file will include the following columns:

        ==========      ========================================================
        Column          Description
        ==========      ========================================================
        r_nom           The nominal center of the bin in r
        meanr           The mean value :math:`\langle r\rangle` of pairs that
                        fell into each bin
        meanlogr        The mean value :math:`\langle \log(r)\rangle` of pairs
                        that fell into each bin
        kgamT           The real part of correlation function,
                        :math:`\langle \kappa\, \gamma_T\rangle`
        kgamX           The imag part of correlation function,
                        :math:`\langle \kappa\, \gamma_\times\rangle`
        sigma           The sqrt of the variance estimate of both of these
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
        """
        self.logger.info('Writing KG correlations to %s',file_name)
        if precision is None:
            precision = self.config.get('precision', 4)

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        gen_write(
            file_name,
            ['r_nom','meanr','meanlogr','kgamT','kgamX','sigma','weight','npairs'],
            [ self.rnom, self.meanr, self.meanlogr,
              self.xi, self.xi_im, np.sqrt(self.varxi),
              self.weight, self.npairs ],
            params=params, precision=precision, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `KGCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading KG correlations from %s',file_name)

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
        self.xi = data['kgamT']
        self.xi_im = data['kgamX']
        self.varxi = data['sigma']**2
        self.weight = data['weight']
        self.npairs = data['npairs']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()

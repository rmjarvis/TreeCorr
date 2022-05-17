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
.. module:: ggcorrelation
"""

import numpy as np

from . import _lib, _ffi
from .catalog import calculateVarG
from .binnedcorr2 import BinnedCorr2
from .util import double_ptr as dp
from .util import make_writer, make_reader
from .util import depr_pos_kwargs


class GGCorrelation(BinnedCorr2):
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

        If you separate out the steps of the `process` command and use `process_auto` and/or
        `process_cross`, then the units will not be applied to ``meanr`` or ``meanlogr`` until
        the `finalize` function is called.

    The typical usage pattern is as follows:

        >>> gg = treecorr.GGCorrelation(config)
        >>> gg.process(cat)         # For auto-correlation.
        >>> gg.process(cat1,cat2)   # For cross-correlation.
        >>> gg.write(file_name)     # Write out to a file.
        >>> xip = gg.xip            # Or access the correlation function directly.

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
    @depr_pos_kwargs
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `GGCorrelation`.  See class doc for details.
        """
        BinnedCorr2.__init__(self, config, logger=logger, **kwargs)

        self._ro._d1 = 3  # GData
        self._ro._d2 = 3  # GData
        self.xip = np.zeros_like(self.rnom, dtype=float)
        self.xim = np.zeros_like(self.rnom, dtype=float)
        self.xip_im = np.zeros_like(self.rnom, dtype=float)
        self.xim_im = np.zeros_like(self.rnom, dtype=float)
        self.varxip = np.zeros_like(self.rnom, dtype=float)
        self.varxim = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self.logger.debug('Finished building GGCorr')

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _lib.BuildCorr2(
                    self._d1, self._d2, self._bintype,
                    self._min_sep,self._max_sep,self._nbins,self._bin_size,self.b,
                    self.min_rpar, self.max_rpar, self.xperiod, self.yperiod, self.zperiod,
                    dp(self.xip),dp(self.xip_im),dp(self.xim),dp(self.xim_im),
                    dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs))
        return self._corr

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if self._corr is not None:
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyCorr2(self.corr, self._d1, self._d2, self._bintype)

    def __eq__(self, other):
        """Return whether two `GGCorrelation` instances are equal"""
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
                np.array_equal(self.varxip, other.varxip) and
                np.array_equal(self.varxim, other.varxim) and
                np.array_equal(self.weight, other.weight) and
                np.array_equal(self.npairs, other.npairs))

    def copy(self):
        """Make a copy"""
        ret = GGCorrelation.__new__(GGCorrelation)
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
        return 'GGCorrelation(config=%r)'%self.config

    @depr_pos_kwargs
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
        if cat.name == '':
            self.logger.info('Starting process GG auto-correlations')
        else:
            self.logger.info('Starting process GG auto-correlations for cat %s.',cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getGField(min_size=min_size, max_size=max_size,
                              split_method=self.split_method,
                              brute=bool(self.brute),
                              min_top=self.min_top, max_top=self.max_top,
                              coords=self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        _lib.ProcessAuto2(self.corr, field.data, self.output_dots,
                          field._d, self._coords, self._bintype, self._metric)


    @depr_pos_kwargs
    def process_cross(self, cat1, cat2, *, metric=None, num_threads=None):
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
            self.logger.info('Starting process GG cross-correlations')
        else:
            self.logger.info('Starting process GG cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getGField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getGField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        _lib.ProcessCross2(self.corr, f1.data, f2.data, self.output_dots,
                           f1._d, f2._d, self._coords, self._bintype, self._metric)


    @depr_pos_kwargs
    def process_pairwise(self, cat1, cat2, *, metric=None, num_threads=None):
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
            metric (str):       Which metric to use.  See `process` for
                                details.  (default: 'Euclidean'; this value can also be given in the
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
            self.logger.info('Starting process GG pairwise-correlations')
        else:
            self.logger.info('Starting process GG pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)

        f1 = cat1.getGSimpleField()
        f2 = cat2.getGSimpleField()

        _lib.ProcessPair(self.corr, f1.data, f2.data, self.output_dots,
                         f1._d, f2._d, self._coords, self._bintype, self._metric)

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

    def _finalize(self):
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.xip[mask1] /= self.weight[mask1]
        self.xim[mask1] /= self.weight[mask1]
        self.xip_im[mask1] /= self.weight[mask1]
        self.xim_im[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]

        # Update the units of meanr, meanlogr
        self._apply_units(mask1)

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = self.rnom[mask2]
        self.meanlogr[mask2] = self.logr[mask2]

    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            varg1 (float):  The shear variance per component for the first field.
            varg2 (float):  The shear variance per component for the second field.
        """
        self._finalize()
        self._var_num = 2. * varg1 * varg2
        self.cov = self.estimate_cov(self.var_method)
        self.varxip.ravel()[:] = self.cov.diagonal()[:self._nbins]
        self.varxim.ravel()[:] = self.cov.diagonal()[self._nbins:]

    def _clear(self):
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
        """Add a second `GGCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `GGCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, GGCorrelation):
            raise TypeError("Can only add another GGCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("GGCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords, other.coords)
        self.xip.ravel()[:] += other.xip.ravel()[:]
        self.xim.ravel()[:] += other.xim.ravel()[:]
        self.xip_im.ravel()[:] += other.xip_im.ravel()[:]
        self.xim_im.ravel()[:] += other.xim_im.ravel()[:]
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
        np.sum([c.xip for c in others], axis=0, out=self.xip)
        np.sum([c.xim for c in others], axis=0, out=self.xim)
        np.sum([c.xip_im for c in others], axis=0, out=self.xip_im)
        np.sum([c.xim_im for c in others], axis=0, out=self.xim_im)
        np.sum([c.meanr for c in others], axis=0, out=self.meanr)
        np.sum([c.meanlogr for c in others], axis=0, out=self.meanlogr)
        np.sum([c.weight for c in others], axis=0, out=self.weight)
        np.sum([c.npairs for c in others], axis=0, out=self.npairs)

    @depr_pos_kwargs
    def process(self, cat1, cat2=None, *, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True):
        """Compute the correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first G field.
            cat2 (Catalog):     A catalog or list of catalogs for the second G field, if any.
                                (default: None)
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
        if cat2 is not None and not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)

        if cat2 is None:
            self._process_all_auto(cat1, metric, num_threads, comm, low_mem)
        else:
            self._process_all_cross(cat1, cat2, metric, num_threads, comm, low_mem)

        if finalize:
            if cat2 is None:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = varg1
                self.logger.info("varg = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
            else:
                varg1 = calculateVarG(cat1, low_mem=low_mem)
                varg2 = calculateVarG(cat2, low_mem=low_mem)
                self.logger.info("varg1 = %f: sig_sn (per component) = %f",varg1,math.sqrt(varg1))
                self.logger.info("varg2 = %f: sig_sn (per component) = %f",varg2,math.sqrt(varg2))
            self.finalize(varg1,varg2)


    @depr_pos_kwargs
    def write(self, file_name, *, file_type=None, precision=None, write_patch_results=False):
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
        """
        self.logger.info('Writing GG correlations to %s',file_name)
        precision = self.config.get('precision', 4) if precision is None else precision
        name = 'main' if write_patch_results else None
        with make_writer(file_name, precision, file_type, self.logger) as writer:
            self._write(writer, name, write_patch_results)

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

    @property
    def _write_params(self):
        return { 'coords' : self.coords, 'metric' : self.metric,
                 'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

    @depr_pos_kwargs
    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        .. warning::

            The `GGCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading GG correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

    # Helper function used by _read
    def _read_from_data(self, data, params):
        s = self.logr.shape
        if 'R_nom' in data.dtype.names:  # pragma: no cover
            self._ro.rnom = data['R_nom'].reshape(s)
            self.meanr = data['meanR'].reshape(s)
            self.meanlogr = data['meanlogR'].reshape(s)
        else:
            self._ro.rnom = data['r_nom'].reshape(s)
            self.meanr = data['meanr'].reshape(s)
            self.meanlogr = data['meanlogr'].reshape(s)
        self.xip = data['xip'].reshape(s)
        self.xim = data['xim'].reshape(s)
        self.xip_im = data['xip_im'].reshape(s)
        self.xim_im = data['xim_im'].reshape(s)
        # Read old output files without error.
        if 'sigma_xi' in data.dtype.names:  # pragma: no cover
            self.varxip = data['sigma_xi'].reshape(s)**2
            self.varxim = data['sigma_xi'].reshape(s)**2
        else:
            self.varxip = data['sigma_xip'].reshape(s)**2
            self.varxim = data['sigma_xim'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.npairs = data['npairs'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self._ro.sep_units = params['sep_units'].strip()
        self._ro.bin_type = params['bin_type'].strip()
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)

    @depr_pos_kwargs
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


    @depr_pos_kwargs
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


    @depr_pos_kwargs
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

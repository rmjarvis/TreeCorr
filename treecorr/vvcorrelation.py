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

from . import _treecorr
from .catalog import calculateVarV
from .corr2base import Corr2
from .util import make_writer, make_reader
from .config import make_minimal_config


class VVCorrelation(Corr2):
    r"""This class handles the calculation and storage of a 2-point vector-vector correlation
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

        >>> vv = treecorr.VVCorrelation(config)
        >>> vv.process(cat)         # For auto-correlation.
        >>> vv.process(cat1,cat2)   # For cross-correlation.
        >>> vv.write(file_name)     # Write out to a file.
        >>> xip = vv.xip            # Or access the correlation function directly.

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
    def __init__(self, config=None, *, logger=None, **kwargs):
        """Initialize `VVCorrelation`.  See class doc for details.
        """
        Corr2.__init__(self, config, logger=logger, **kwargs)

        self.xip = np.zeros_like(self.rnom, dtype=float)
        self.xim = np.zeros_like(self.rnom, dtype=float)
        self.xip_im = np.zeros_like(self.rnom, dtype=float)
        self.xim_im = np.zeros_like(self.rnom, dtype=float)
        self.meanr = np.zeros_like(self.rnom, dtype=float)
        self.meanlogr = np.zeros_like(self.rnom, dtype=float)
        self.weight = np.zeros_like(self.rnom, dtype=float)
        self.npairs = np.zeros_like(self.rnom, dtype=float)
        self._varxip = None
        self._varxim = None
        self._cov = None
        self._var_num = 0
        self._processed_cats1 = []
        self._processed_cats2 = []
        self.logger.debug('Finished building VVCorr')

    @property
    def corr(self):
        if self._corr is None:
            self._corr = _treecorr.VVCorr(self._bintype, self._min_sep, self._max_sep, self._nbins,
                                          self._bin_size, self.b, self.angle_slop,
                                          self.min_rpar, self.max_rpar,
                                          self.xperiod, self.yperiod, self.zperiod,
                                          self.xip, self.xip_im, self.xim, self.xim_im,
                                          self.meanr, self.meanlogr, self.weight, self.npairs)
        return self._corr

    def __eq__(self, other):
        """Return whether two `VVCorrelation` instances are equal"""
        return (isinstance(other, VVCorrelation) and
                self.nbins == other.nbins and
                self.bin_size == other.bin_size and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep and
                self.sep_units == other.sep_units and
                self.coords == other.coords and
                self.bin_type == other.bin_type and
                self.bin_slop == other.bin_slop and
                self.angle_slop == other.angle_slop and
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
        ret = VVCorrelation.__new__(VVCorrelation)
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
        return f'VVCorrelation({self._repr_kwargs})'

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
            self.logger.info('Starting process VV auto-correlations')
        else:
            self.logger.info('Starting process VV auto-correlations for cat %s.',cat.name)

        self._set_metric(metric, cat.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        field = cat.getVField(min_size=min_size, max_size=max_size,
                              split_method=self.split_method,
                              brute=bool(self.brute),
                              min_top=self.min_top, max_top=self.max_top,
                              coords=self.coords)

        self.logger.info('Starting %d jobs.',field.nTopLevelNodes)
        self.corr.processAuto(field.data, self.output_dots, self._metric)


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
            self.logger.info('Starting process VV cross-correlations')
        else:
            self.logger.info('Starting process VV cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)
        self._set_num_threads(num_threads)
        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getVField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 1,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)
        f2 = cat2.getVField(min_size=min_size, max_size=max_size,
                            split_method=self.split_method,
                            brute=self.brute is True or self.brute == 2,
                            min_top=self.min_top, max_top=self.max_top,
                            coords=self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        self.corr.processCross(f1.data, f2.data, self.output_dots, self._metric)

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

    def finalize(self, varv1, varv2):
        """Finalize the calculation of the correlation function.

        The `process_auto` and `process_cross` commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        Parameters:
            varv1 (float):  The variance per component of the first vector field.
            varv2 (float):  The variance per component of the second vector field.
        """
        self._finalize()
        self._var_num = 2. * varv1 * varv2

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
        self._varxip = None
        self._varxim = None
        self._cov = None

    def __iadd__(self, other):
        """Add a second `VVCorrelation`'s data to this one.

        .. note::

            For this to make sense, both `VVCorrelation` objects should not have had `finalize`
            called yet.  Then, after adding them together, you should call `finalize` on the sum.
        """
        if not isinstance(other, VVCorrelation):
            raise TypeError("Can only add another VVCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("VVCorrelation to be added is not compatible with this one.")

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

    def process(self, cat1, cat2=None, *, metric=None, num_threads=None, comm=None, low_mem=False,
                initialize=True, finalize=True, patch_method='global'):
        """Compute the correlation function.

        - If only 1 argument is given, then compute an auto-correlation function.
        - If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        Parameters:
            cat1 (Catalog):     A catalog or list of catalogs for the first V field.
            cat2 (Catalog):     A catalog or list of catalogs for the second V field, if any.
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
                                `Corr2.clear`.  (default: True)
            finalize (bool):    Whether to complete the calculation with a call to `finalize`.
                                (default: True)
            patch_method (str): Which patch method to use. (default: 'global')
        """
        import math
        if initialize:
            self.clear()
            self._processed_cats1.clear()
            self._processed_cats2.clear()

        if patch_method not in ['local', 'global']:
            raise ValueError("Invalid patch_method %s"%patch_method)
        local = patch_method == 'local'

        if not isinstance(cat1,list):
            cat1 = cat1.get_patches(low_mem=low_mem)
        if cat2 is not None and not isinstance(cat2,list):
            cat2 = cat2.get_patches(low_mem=low_mem)

        if cat2 is None:
            self._process_all_auto(cat1, metric, num_threads, comm, low_mem, local)
        else:
            self._process_all_cross(cat1, cat2, metric, num_threads, comm, low_mem, local)

        self._processed_cats1.extend(cat1)
        if cat2 is not None:
            self._processed_cats2.extend(cat2)
        if finalize:
            if cat2 is None:
                varv1 = calculateVarV(self._processed_cats1, low_mem=low_mem)
                varv2 = varv1
                self.logger.info("varv = %f: sig_sn (per component) = %f",varv1,math.sqrt(varv1))
            else:
                varv1 = calculateVarV(self._processed_cats1, low_mem=low_mem)
                varv2 = calculateVarV(self._processed_cats2, low_mem=low_mem)
                self.logger.info("varv1 = %f: sig_sn (per component) = %f",varv1,math.sqrt(varv1))
                self.logger.info("varv2 = %f: sig_sn (per component) = %f",varv2,math.sqrt(varv2))
            self.finalize(varv1,varv2)
            self._processed_cats1.clear()
            self._processed_cats2.clear()

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
        self.logger.info('Writing VV correlations to %s',file_name)
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

    @property
    def _write_params(self):
        params = make_minimal_config(self.config, Corr2._valid_params)
        # Add in a couple other things we want to preserve that aren't construction kwargs.
        params['coords'] = self.coords
        params['metric'] = self.metric
        return params

    @classmethod
    def from_file(cls, file_name, *, file_type=None, logger=None, rng=None):
        """Create a VVCorrelation instance from an output file.

        This should be a file that was written by TreeCorr.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII', 'FITS', or 'HDF').  (default: determine
                                the type automatically from the extension of file_name.)
            logger (Logger):    If desired, a logger object to use for logging. (default: None)
            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for bootstrap
                                random number generation. (default: None)

        Returns:
            corr: A VVCorrelation object, constructed from the information in the file.
        """
        if logger:
            logger.info('Building VVCorrelation from %s',file_name)
        with make_reader(file_name, file_type, logger) as reader:
            name = 'main' if 'main' in reader else None
            params = reader.read_params(ext=name)
            kwargs = make_minimal_config(params, Corr2._valid_params)
            corr = cls(**kwargs, logger=logger, rng=rng)
            corr.logger.info('Reading VV correlations from %s',file_name)
            corr._read(reader, name=name, params=params)
        return corr

    def read(self, file_name, *, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS or HDF5 file, so
        there is no loss of information.

        .. warning::

            The `VVCorrelation` object should be constructed with the same configuration
            parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
            checked by the read function.

        Parameters:
            file_name (str):    The name of the file to read in.
            file_type (str):    The type of file ('ASCII' or 'FITS').  (default: determine the type
                                automatically from the extension of file_name.)
        """
        self.logger.info('Reading VV correlations from %s',file_name)
        with make_reader(file_name, file_type, self.logger) as reader:
            self._read(reader)

    # Helper function used by _read
    def _read_from_data(self, data, params):
        s = self.logr.shape
        self.meanr = data['meanr'].reshape(s)
        self.meanlogr = data['meanlogr'].reshape(s)
        self.xip = data['xip'].reshape(s)
        self.xim = data['xim'].reshape(s)
        self.xip_im = data['xip_im'].reshape(s)
        self.xim_im = data['xim_im'].reshape(s)
        # Read old output files without error.
        self._varxip = data['sigma_xip'].reshape(s)**2
        self._varxim = data['sigma_xim'].reshape(s)**2
        self.weight = data['weight'].reshape(s)
        self.npairs = data['npairs'].reshape(s)
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.npatch1 = params.get('npatch1', 1)
        self.npatch2 = params.get('npatch2', 1)

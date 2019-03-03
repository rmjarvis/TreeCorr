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
.. module:: nkcorrelation
"""

import treecorr
import numpy


class NKCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-kappa correlation
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
        :xi:        The correlation function, xi(r).
        :varxi:     The variance of xi, only including the shot noise propagated into the
                    final correlation.  This does not include sample variance, so it is
                    always an underestimate of the actual variance.
        :weight:    The total weight in each bin.
        :npairs:    The number of pairs going into each bin.

    If `sep_units` are given (either in the config dict or as a named kwarg) then the distances
    will all be in these units.  Note however, that if you separate out the steps of the
    :func:`process` command and use :func:`process_auto` and/or :func:`process_cross`, then the
    units will not be applied to :meanr: or :meanlogr: until the :func:`finalize` function is
    called.

    The typical usage pattern is as follows:

        >>> nk = treecorr.NKCorrelation(config)
        >>> nk.process(cat1,cat2)   # Compute the cross-correlation function.
        >>> nk.write(file_name)     # Write out to a file.
        >>> xi = nk.xi              # Or access the correlation function directly.

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

        self.xi = numpy.zeros_like(self.rnom, dtype=float)
        self.varxi = numpy.zeros_like(self.rnom, dtype=float)
        self.meanr = numpy.zeros_like(self.rnom, dtype=float)
        self.meanlogr = numpy.zeros_like(self.rnom, dtype=float)
        self.weight = numpy.zeros_like(self.rnom, dtype=float)
        self.npairs = numpy.zeros_like(self.rnom, dtype=float)
        self._build_corr()
        self.logger.debug('Finished building NKCorr')

    def _build_corr(self):
        from treecorr.util import double_ptr as dp
        self.corr = treecorr._lib.BuildNKCorr(
                self._bintype,
                self._min_sep,self._max_sep,self._nbins,self.bin_size,self.b,
                self.min_rpar, self.max_rpar,
                dp(self.xi),
                dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs));

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'corr'):    # In case __init__ failed to get that far
            treecorr._lib.DestroyNKCorr(self.corr, self._bintype)

    def copy(self):
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
                treecorr.config.get(self.config,'verbose',int,0),
                self.config.get('log_file',None))

    def __repr__(self):
        return 'NKCorrelation(config=%r)'%self.config

    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NKCorrelation.process` for
                            details.  (default: 'Euclidean'; this value can also be given in the
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NK cross-correlations')
        else:
            self.logger.info('Starting process NK cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        min_size, max_size = self._get_minmax_size()

        f1 = cat1.getNField(min_size, max_size, self.split_method, self.max_top, self.coords)
        f2 = cat2.getKField(min_size, max_size, self.split_method, self.max_top, self.coords)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        treecorr._lib.ProcessCrossNK(self.corr, f1.data, f2.data, self.output_dots,
                                     self._coords, self._bintype, self._metric)


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NKCorrelation.process` for
                            details.  (default: 'Euclidean'; this value can also be given in the
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NK pairwise-correlations')
        else:
            self.logger.info('Starting process NK pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        self._set_metric(metric, cat1.coords, cat2.coords)

        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField()
        f2 = cat2.getKSimpleField()

        treecorr._lib.ProcessPairNK(self.corr, f1.data, f2.data, self.output_dots,
                                    self._coords, self._bintype, self._metric)


    def finalize(self, vark):
        """Finalize the calculation of the correlation function.

        The process_cross command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        :param vark:    The kappa variance for the second field.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.xi[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = vark / self.weight[mask1]

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
        self.meanr.ravel()[:] = 0
        self.meanlogr.ravel()[:] = 0
        self.weight.ravel()[:] = 0
        self.npairs.ravel()[:] = 0

    def __iadd__(self, other):
        """Add a second NKCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using
        process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, NKCorrelation):
            raise AttributeError("Can only add another NKCorrelation object")
        if not (self._nbins == other._nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NKCorrelation to be added is not compatible with this one.")

        self._set_metric(other.metric, other.coords)
        self.xi.ravel()[:] += other.xi.ravel()[:]
        self.meanr.ravel()[:] += other.meanr.ravel()[:]
        self.meanlogr.ravel()[:] += other.meanlogr.ravel()[:]
        self.weight.ravel()[:] += other.weight.ravel()[:]
        self.npairs.ravel()[:] += other.npairs.ravel()[:]
        return self


    def process(self, cat1, cat2, metric=None, num_threads=None):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used
        for that element of the correlation.

        :param cat1:    A catalog or list of catalogs for the N field.
        :param cat2:    A catalog or list of catalogs for the K field.
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
        if not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")
        if len(cat2) == 0:
            raise ValueError("No catalogs provided for cat2")

        vark = treecorr.calculateVarK(cat2)
        self.logger.info("vark = %f: sig_k = %f",vark,math.sqrt(vark))
        self._process_all_cross(cat1,cat2,metric,num_threads)
        self.finalize(vark)


    def calculateXi(self, rk=None):
        """Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rk is None, the simple correlation function :math:`\\langle \\kappa \\rangle` is
          returned.
        - If rk is not None, then a compensated calculation is done:
          :math:`\\langle \\kappa \\rangle = (dk - rk)`

        :param rk:          An NKCorrelation using random locations as the lenses, if desired.
                            (default: None)

        :returns:           (xi, varxi) as a tuple
        """
        if rk is None:
            return self.xi, self.varxi
        else:
            return self.xi - rk.xi, self.varxi + rk.varxi


    def write(self, file_name, rk=None, file_type=None, prec=None):
        """Write the correlation function to the file, file_name.

        - If rk is None, the simple correlation function :math:`\\langle \\kappa \\rangle(R)` is
          used.
        - If rk is not None, then a compensated calculation is done:
          :math:`\\langle \\kappa \\rangle(R) = (dk - rk)`.

        The output file will include the following columns:

            :R_nom:     The nominal center of the bin in R.
            :meanR:     The mean value :math:`\\langle R\\rangle` of pairs that fell into each bin.
            :meanlogR:  The mean value :math:`\\langle logR\\rangle` of pairs that fell into each
                        bin.
            :kappa:     The mean value of :math:`\\langle \\kappa \\rangle(R)`.
            :sigma:     The sqrt of the variance estimate of :math:`\\langle \\kappa \\rangle`.
            :weight:    The total weight contributing to each bin.
            :npairs:    The number of pairs contributing ot each bin.

        If `sep_units` was given at construction, then the distances will all be in these units.
        Otherwise, they will be in either the same units as x,y,z (for flat or 3d coordinates) or
        radians (for spherical coordinates).

        :param file_name:   The name of the file to write to.
        :param rk:          An NKCorrelation using random locations as the lenses, if desired.
                            (default: None)
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param prec:        For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing NK correlations to %s',file_name)

        xi, varxi = self.calculateXi(rk)
        if prec is None:
            prec = self.config.get('precision', 4)

        params = { 'coords' : self.coords, 'metric' : self.metric,
                   'sep_units' : self.sep_units, 'bin_type' : self.bin_type }

        treecorr.util.gen_write(
            file_name,
            ['R_nom','meanR','meanlogR','kappa','sigma','weight','npairs'],
            [ self.rnom, self.meanr, self.meanlogr,
              xi, numpy.sqrt(varxi), self.weight, self.npairs ],
            params=params, prec=prec, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The NKCorrelation object should be constructed with the same configuration
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        :param file_name:   The name of the file to read in.
        :param file_type:   The type of file ('ASCII' or 'FITS').  (default: determine the type
                            automatically from the extension of file_name.)
        """
        self.logger.info('Reading NK correlations from %s',file_name)

        data, params = treecorr.util.gen_read(file_name, file_type=file_type)
        self.rnom = data['R_nom']
        self.logr = numpy.log(self.rnom)
        self.meanr = data['meanR']
        self.meanlogr = data['meanlogR']
        self.xi = data['kappa']
        self.varxi = data['sigma']**2
        self.weight = data['weight']
        self.npairs = data['npairs']
        self.coords = params['coords'].strip()
        self.metric = params['metric'].strip()
        self.sep_units = params['sep_units'].strip()
        self.bin_type = params['bin_type'].strip()
        self._build_corr()



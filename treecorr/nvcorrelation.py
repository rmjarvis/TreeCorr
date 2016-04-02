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
.. module:: nvcorrelation
"""

import treecorr
import numpy


class NVCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-vector correlation
    function.

    Ojects of this class holds the following attributes:

        :nbins:     The number of bins in logr
        :bin_size:  The size of the bins in logr
        :min_sep:   The minimum separation being considered
        :max_sep:   The maximum separation being considered

    In addition, the following attributes are numpy arrays of length (nbins):

        :logr:      The nominal center of the bin in log(r) (the natural logarithm of r).
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

    If sep_units are given (either in the config dict or as a named kwarg) then logr and meanlogr
    both take r to be in these units.  i.e. exp(logr) will have R in units of sep_units.

    The usage pattern is as follows:

        >>> nv = treecorr.NVCorrelation(config)
        >>> nv.process(cat1,cat2)   # Compute the cross-correlation function.
        >>> nv.write(file_name)     # Write out to a file.
        >>> xi = nv.xi              # Or access the correlation function directly.

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

        self.xi = numpy.zeros(self.nbins, dtype=float)
        self.varxi = numpy.zeros(self.nbins, dtype=float)
        self.meanr = numpy.zeros(self.nbins, dtype=float)
        self.meanlogr = numpy.zeros(self.nbins, dtype=float)
        self.weight = numpy.zeros(self.nbins, dtype=float)
        self.npairs = numpy.zeros(self.nbins, dtype=float)
        self._build_corr()
        self.logger.debug('Finished building NVCorr')

    def _build_corr(self):
        from treecorr.util import double_ptr as dp
        self.corr = treecorr._lib.BuildNVCorr(
                self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                dp(self.xi),
                dp(self.meanr),dp(self.meanlogr),dp(self.weight),dp(self.npairs));

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'corr'):    # In case __init__ failed to get that far
            treecorr._lib.DestroyNVCorr(self.corr)

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
        return 'NVCorrelation(config=%r)'%self.config

    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NVCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NV cross-correlations')
        else:
            self.logger.info('Starting process NV cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")
        if metric == 'Rperp' and cat1.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        self._set_num_threads(num_threads)

        min_size = self.min_sep * self.b / (2.+3.*self.b);
        if metric == 'Rperp': min_size /= 2.
        max_size = self.max_sep * self.b

        f1 = cat1.getNField(min_size,max_size,self.split_method,self.max_top)
        f2 = cat2.getVField(min_size,max_size,self.split_method,self.max_top)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        treecorr._lib.ProcessCrossNV3D(self.corr, f1.data, f2.data, self.output_dots)


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.NVCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process NV pairwise-correlations')
        else:
            self.logger.info('Starting process NV pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")
        if metric == 'Rperp' and cat1.coords != '3d':
            raise ValueError("Rperp metric is only valid for catalogs with 3d positions.")

        self._set_num_threads(num_threads)

        f1 = cat1.getNSimpleField()
        f2 = cat2.getVSimpleField()

        treecorr._lib.ProcessPairwiseNV3D(self.corr, f1.data, f2.data, self.output_dots)


    def finalize(self):
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

        # Update the units of meanr, meanlogr
        self.meanr[mask1] /= self.sep_units
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanr, meanlogr when available, but set to nominal when no pairs in bin.
        self.meanr[mask2] = numpy.exp(self.logr[mask2])
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.


    def clear(self):
        """Clear the data vectors
        """
        self.xi[:] = 0
        self.meanr[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0

    def __iadd__(self, other):
        """Add a second NVCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using 
        process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, NVCorrelation()):
            raise AttributeError("Can only add another NVCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("NVCorrelation to be added is not compatible with this one.")

        self.xi[:] += other.xi[:]
        self.meanr[:] += other.meanr[:]
        self.meanlogr[:] += other.meanlogr[:]
        self.weight[:] += other.weight[:]
        self.npairs[:] += other.npairs[:]
        return self


    def process(self, cat1, cat2, metric=None, num_threads=None):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:        A catalog or list of catalogs for the N field.
        :param cat2:        A catalog or list of catalogs for the K field.
        :param metric:      Which metric to use for distance measurements.  Options are:

                            - 'Euclidean' = straight line Euclidean distance between two points.
                              For spherical coordinates (ra,dec without r), this is the chord
                              distance between points on the unit sphere.
                            - 'Rperp' = the perpendicular component of the distance. For two points
                              with distance from Earth `r1, r2`, if `d` is the normal Euclidean 
                              distance and :math:`Rparallel = |r1-r2|`, then we define
                              :math:`Rperp^2 = d^2 - Rparallel^2`.

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
        
        self._process_all_cross(cat1,cat2,metric,num_threads)
        self.finalize()


    def calculateXi(self):
        """Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        - If rk is None, the simple correlation function :math:`\\langle \\kappa \\rangle` is
          returned.


        :returns:           xi
        """
        return self.xi - 1.0/3.
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
.. module:: kgcorrelation
"""

import treecorr
import numpy

# Start by loading up the relevant C functions using ctypes
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))

# some useful aliases
cint = ctypes.c_int
cdouble = ctypes.c_double
cdouble_ptr = ctypes.POINTER(cdouble)
cvoid_ptr = ctypes.c_void_p

_treecorr.BuildKGCorr.restype = cvoid_ptr
_treecorr.BuildKGCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyKGCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessCrossKGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossKG3D.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossKGPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseKGFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseKG3D.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseKGPerp.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class KGCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point kappa-shear correlation
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
        :xi:        The correlation function, :math:`\\xi(r) = \\langle \\kappa \\gamma_T\\rangle`.
        :xi_im:     The imaginary part of :math:`\\xi(r)`.
        :varxi:     The variance of :math:`\\xi`, only including the shape noise propagated into the
                    final correlation.  This does not include sample variance, so it is always
                    an underestimate of the actual variance.
        :weight:    The total weight in each bin.
        :npairs:    The number of pairs going into each bin.

    If sep_units are given (either in the config dict or as a named kwarg) then logr and meanlogr
    both take r to be in these units.  i.e. exp(logr) will have R in units of sep_units.

    The usage pattern is as follows:

        >>> kg = treecorr.KGCorrelation(config)
        >>> kg.process(cat1,cat2)   # Calculate the cross-correlation
        >>> kg.write(file_name)     # Write out to a file.
        >>> xi = kg.xi              # Or access the correlation function directly.

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
        self.xi_im = numpy.zeros(self.nbins, dtype=float)
        self.varxi = numpy.zeros(self.nbins, dtype=float)
        self.meanr = numpy.zeros(self.nbins, dtype=float)
        self.meanlogr = numpy.zeros(self.nbins, dtype=float)
        self.weight = numpy.zeros(self.nbins, dtype=float)
        self.npairs = numpy.zeros(self.nbins, dtype=float)
        self._build_corr()
        self.logger.debug('Finished building KGCorr')

    def _build_corr(self):
        xi = self.xi.ctypes.data_as(cdouble_ptr)
        xi_im = self.xi_im.ctypes.data_as(cdouble_ptr)
        meanr = self.meanr.ctypes.data_as(cdouble_ptr)
        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        weight = self.weight.ctypes.data_as(cdouble_ptr)
        npairs = self.npairs.ctypes.data_as(cdouble_ptr)
        self.corr = _treecorr.BuildKGCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          xi,xi_im,meanr,meanlogr,weight,npairs);

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'corr'):    # In case __init__ failed to get that far
            _treecorr.DestroyKGCorr(self.corr)

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
        return 'KGCorrelation(config=%r)'%self.config

    def process_cross(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.KGCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process KG cross-correlations')
        else:
            self.logger.info('Starting process KG cross-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        self._set_num_threads(num_threads)

        f1 = cat1.getKField(self.min_sep,self.max_sep,self.b,self.split_method,metric,self.max_top)
        f2 = cat2.getGField(self.min_sep,self.max_sep,self.b,self.split_method,metric,self.max_top)

        self.logger.info('Starting %d jobs.',f1.nTopLevelNodes)
        if f1.flat:
            _treecorr.ProcessCrossKGFlat(self.corr, f1.data, f2.data, self.output_dots)
        elif f1.perp:
            _treecorr.ProcessCrossKGPerp(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossKG3D(self.corr, f1.data, f2.data, self.output_dots)


    def process_pairwise(self, cat1, cat2, metric=None, num_threads=None):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:        The first catalog to process
        :param cat2:        The second catalog to process
        :param metric:      Which metric to use.  See :meth:`~treecorr.KGCorrelation.process` for 
                            details.  (default: 'Euclidean'; this value can also be given in the 
                            constructor in the config dict.)
        :param num_threads: How many OpenMP threads to use during the calculation.  
                            (default: use the number of cpu cores; this value can also be given in
                            the constructor in the config dict.) Note that this won't work if the 
                            system's C compiler is clang prior to version 3.7.
        """
        if cat1.name == '' and cat2.name == '':
            self.logger.info('Starting process KG pairwise-correlations')
        else:
            self.logger.info('Starting process KG pairwise-correlations for cats %s, %s.',
                             cat1.name, cat2.name)

        if metric is None:
            metric = treecorr.config.get(self.config,'metric',str,'Euclidean')
        if metric not in ['Euclidean', 'Rperp']:
            raise ValueError("Invalid metric.")
        if cat1.coords != cat2.coords:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        self._set_num_threads(num_threads)

        f1 = cat1.getKSimpleField(metric)
        f2 = cat2.getGSimpleField(metric)

        if f1.flat:
            _treecorr.ProcessPairwiseKGFlat(self.corr, f1.data, f2.data, self.output_dots)
        elif f1.perp:
            _treecorr.ProcessPairwiseKGPerp(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessPairwiseKG3D(self.corr, f1.data, f2.data, self.output_dots)


    def finalize(self, vark, varg):
        """Finalize the calculation of the correlation function.

        The process_cross command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.

        :param vark:    The kappa variance for the first field.
        :param varg:    The shear variance per component for the second field.
        """
        mask1 = self.weight != 0
        mask2 = self.weight == 0

        self.xi[mask1] /= self.weight[mask1]
        self.xi_im[mask1] /= self.weight[mask1]
        self.meanr[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = vark * varg / self.weight[mask1]

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
        self.xi_im[:] = 0
        self.meanr[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0

    def __iadd__(self, other):
        """Add a second GGCorrelation's data to this one.

        Note: For this to make sense, both Correlation objects should have been using 
        process_cross, and they should not have had finalize called yet.
        Then, after adding them together, you should call finalize on the sum.
        """
        if not isinstance(other, KGCorrelation):
            raise AttributeError("Can only add another KGCorrelation object")
        if not (self.nbins == other.nbins and
                self.min_sep == other.min_sep and
                self.max_sep == other.max_sep):
            raise ValueError("KGCorrelation to be added is not compatible with this one.")

        self.xi[:] += other.xi[:]
        self.xi_im[:] += other.xi_im[:]
        self.meanr[:] += other.meanr[:]
        self.meanlogr[:] += other.meanlogr[:]
        self.weight[:] += other.weight[:]
        self.npairs[:] += other.npairs[:]
        return self


    def process(self, cat1, cat2, metric=None, num_threads=None):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:        A catalog or list of catalogs for the K field.
        :param cat2:        A catalog or list of catalogs for the G field.
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

        vark = treecorr.calculateVarK(cat1)
        varg = treecorr.calculateVarG(cat2)
        self.logger.info("vark = %f: sig_k = %f",vark,math.sqrt(vark))
        self.logger.info("varg = %f: sig_sn (per component) = %f",varg,math.sqrt(varg))
        self._process_all_cross(cat1,cat2,metric,num_threads)
        self.finalize(vark,varg)


    def write(self, file_name, file_type=None, prec=None):
        """Write the correlation function to the file, file_name.

        The output file will include the following columns:

            :R_nom:     The nominal center of the bin in R.
            :meanR:     The mean value :math:`\\langle R\\rangle` of pairs that fell into each bin.
            :meanlogR:  The mean value :math:`\\langle logR\\rangle` of pairs that fell into each
                        bin.
            :kgamT:     The real part of correlation function 
                        :math:`\\xi = \\langle \\kappa \\gamma_T \\rangle`.
            :kgamX:     The imag part of correlation function 
                        :math:`\\xi = \\langle \\kappa \\gamma_T \\rangle`.
            :sigma:     The sqrt of the variance estimate of :math:`\\xi`.
            :weight:    The total weight contributing to each bin.
            :npairs:    The number of pairs contributing ot each bin.


        :param file_name:   The name of the file to write to.
        :param file_type:   The type of file to write ('ASCII' or 'FITS').  (default: determine
                            the type automatically from the extension of file_name.)
        :param prec:        For ASCII output catalogs, the desired precision. (default: 4;
                            this value can also be given in the constructor in the config dict.)
        """
        self.logger.info('Writing KG correlations to %s',file_name)
        if prec is None:
            prec = self.config.get('precision', 4)
        
        treecorr.util.gen_write(
            file_name,
            ['R_nom','meanR','meanlogR','kgamT','kgamX','sigma','weight','npairs'],
            [ numpy.exp(self.logr), self.meanr, self.meanlogr,
              self.xi, self.xi_im, numpy.sqrt(self.varxi),
              self.weight, self.npairs ],
            prec=prec, file_type=file_type, logger=self.logger)


    def read(self, file_name, file_type=None):
        """Read in values from a file.

        This should be a file that was written by TreeCorr, preferably a FITS file, so there
        is no loss of information.

        Warning: The KGCorrelation object should be constructed with the same configuration 
        parameters as the one being read.  e.g. the same min_sep, max_sep, etc.  This is not
        checked by the read function.

        :param file_name:   The name of the file to read in.
        :param file_type:   The type of file ('ASCII' or 'FITS').  (default: determine the type
                            automatically from the extension of file_name.)
        """
        self.logger.info('Reading KG correlations from %s',file_name)

        data = treecorr.util.gen_read(file_name, file_type=file_type)
        self.logr = numpy.log(data['R_nom'])
        self.meanr = data['meanR']
        self.meanlogr = data['meanlogR']
        self.xi = data['kgamT']
        self.xi_im = data['kgamX']
        self.varxi = data['sigma']**2
        self.weight = data['weight']
        self.npairs = data['npairs']



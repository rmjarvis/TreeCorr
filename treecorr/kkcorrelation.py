# Copyright (c) 2003-2014 by Mike Jarvis
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

_treecorr.BuildKKCorr.restype = cvoid_ptr
_treecorr.BuildKKCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr, cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyKKCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoKKSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoKKFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossKKSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossKKFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseKKSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseKKFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class KKCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point kappa-kappa correlation
    function.

    Note: while we use the term kappa here and the letter K in various places, in fact
    any scalar field will work here.  For example, you can use this to compute correlations
    of the CMB temperature fluctuations, where "kappa" would really be delta T.

    It holds the following attributes:

        :logr:      The nominal center of the bin in log(r).
        :meanlogr:  The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        :xi:        The correlation function, xi(r).
        :varxi:     The variance of xi, only including the shot noise propagated into the
                    final correlation.  This does not include sample variance, so it is always
                    an underestimate of the actual variance.
        :weight:    The total weight in each bin.
        :npairs:    The number of pairs going into each bin.

    The usage pattern is as follows:

        >>> kk = treecorr.KKCorrelation(config)
        >>> kk.process(cat)         # For auto-correlation.
        >>> kk.process(cat1,cat2)   # For cross-correlation.
        >>> kk.write(file_name)     # Write out to a file.
        >>> xi = kk.xi              # Or access the correlation function directly.

    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.xi = numpy.zeros(self.nbins, dtype=float)

        xi = self.xi.ctypes.data_as(cdouble_ptr)
        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        weight = self.weight.ctypes.data_as(cdouble_ptr)
        npairs = self.npairs.ctypes.data_as(cdouble_ptr)

        self.corr = _treecorr.BuildKKCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          xi,meanlogr,weight,npairs);
        self.logger.debug('Finished building KKCorr')
 

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyKKCorr(self.corr)


    def process_auto(self, cat):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat:     The catalog to process
        """
        self.logger.info('Starting process KK auto-correlations for cat %s.',cat.name)
        field = cat.getKField(self.min_sep,self.max_sep,self.b,self.split_method)

        if field.sphere:
            _treecorr.ProcessAutoKKSphere(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoKKFlat(self.corr, field.data, self.output_dots)


    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        """
        self.logger.info('Starting process KK cross-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getKField(self.min_sep,self.max_sep,self.b,self.split_method)
        f2 = cat2.getKField(self.min_sep,self.max_sep,self.b,self.split_method)

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessCrossKKSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossKKFlat(self.corr, f1.data, f2.data, self.output_dots)


    def process_pairwise(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation, only using
        the corresponding pairs of objects in each catalog.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        """
        self.logger.info('Starting process KK pairwise-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getKSimpleField()
        f2 = cat2.getKSimpleField()

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessPairwiseKKSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessPairwiseKKFlat(self.corr, f1.data, f2.data, self.output_dots)


    def finalize(self, vark1, vark2):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.

        :param vark1:   The kappa variance for the first field.
        :param vark2:   The kappa variance for the second field.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.xi[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = vark1 * vark2 / self.npairs[mask1]

        # Update the units of meanlogr
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.


    def clear(self):
        """Clear the data vectors
        """
        self.xi[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0


    def process(self, cat1, cat2=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:    A catalog or list of catalogs for the first K field.
        :param cat2:    A catalog or list of catalogs for the second K field, if any.
                        (default: None)
        """
        import math
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")

        if cat2 is None or len(cat2) == 0:
            vark1 = treecorr.calculateVarK(cat1)
            vark2 = vark1
            self.logger.info("vark = %f: sig_k = %f",vark1,math.sqrt(vark1))
            self._process_all_auto(cat1)
        else:
            vark1 = treecorr.calculateVarK(cat1)
            vark2 = treecorr.calculateVarK(cat2)
            self.logger.info("vark1 = %f: sig_k = %f",vark1,math.sqrt(vark1))
            self.logger.info("vark2 = %f: sig_k = %f",vark2,math.sqrt(vark2))
            self._process_all_cross(cat1,cat2)
        self.finalize(vark1,vark2)


    def write(self, file_name):
        """Write the correlation function to the file, file_name.

        :param file_name:   The name of the file to write to.
        """
        self.logger.info('Writing KK correlations to %s',file_name)
         
        self.gen_write(
            file_name,
            ['R_nom','<R>','xi','sigma_xi','weight','npairs'],
            [ numpy.exp(self.logr), numpy.exp(self.meanlogr),
              self.xi, numpy.sqrt(self.varxi), self.weight, self.npairs ] )


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

_treecorr.BuildNNCorr.restype = cvoid_ptr
_treecorr.BuildNNCorr.argtypes = [
    cdouble, cdouble, cint, cdouble, cdouble,
    cdouble_ptr, cdouble_ptr ]
_treecorr.DestroyNNCorr.argtypes = [ cvoid_ptr ]
_treecorr.ProcessAutoNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessAutoNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessCrossNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNNSphere.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]
_treecorr.ProcessPairwiseNNFlat.argtypes = [ cvoid_ptr, cvoid_ptr, cvoid_ptr, cint ]


class NNCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point count-count correlation
    function.  i.e. the regular density correlation function.

    It holds the following attributes:

        :logr:      The nominal center of the bin in log(r).
        :meanlogr:  The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        :npairs:    The number of pairs going into each bin.
        :tot:       The total number of pairs processed, which is used to normalize
                    the randoms if they have a different number of pairs.

    The usage pattern is as follows:

        >>> nn = treecorr.NNCorrelation(config)
        >>> nn.process(cat)         # For auto-correlation.
        >>> nn.process(cat1,cat2)   # For cross-correlation.
        >>> rr.process...           # Likewise for random-random correlations
        >>> dr.process...           # If desired, also do data-random correlations
        >>> rd.process...           # For cross-correlations, also do the reverse.
        >>> nn.write(file_name,rr,dr,rd)         # Write out to a file.
        >>> xi,varxi = nn.calculateXi(rr,dr,rd)  # Or get the correlation function directly.

    :param config:      The configuration dict which defines attributes about how to read the file.
                        Any kwargs that are not those listed here will be added to the config, 
                        so you can even omit the config dict and just enter all parameters you
                        want as kwargs.  (default: None) 
    :param logger:      If desired, a logger object for logging. (default: None, in which case
                        one will be built according to the config dict's verbose level.)
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.tot = 0.

        meanlogr = self.meanlogr.ctypes.data_as(cdouble_ptr)
        npairs = self.npairs.ctypes.data_as(cdouble_ptr)

        self.corr = _treecorr.BuildNNCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          meanlogr,npairs);
        self.logger.debug('Finished building NNCorr')


    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyNNCorr(self.corr)


    def process_auto(self, cat):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.

        :param cat:      The catalog to process
        """
        self.logger.info('Starting process NN auto-correlations for cat %s.',cat.name)
        field = cat.getNField(self.min_sep,self.max_sep,self.b,self.split_method)

        if field.sphere:
            _treecorr.ProcessAutoNNSphere(self.corr, field.data, self.output_dots)
        else:
            _treecorr.ProcessAutoNNFlat(self.corr, field.data, self.output_dots)
        self.tot += 0.5 * cat.nobj**2


    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.

        :param cat1:     The first catalog to process
        :param cat2:     The second catalog to process
        """
        self.logger.info('Starting process NN cross-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getNField(self.min_sep,self.max_sep,self.b,self.split_method)
        f2 = cat2.getNField(self.min_sep,self.max_sep,self.b,self.split_method)

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessCrossNNSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNNFlat(self.corr, f1.data, f2.data, self.output_dots)
        self.tot += cat1.nobj*cat2.nobj


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
        self.logger.info('Starting process NN pairwise-correlations for cats %s, %s.',
                         cat1.name, cat2.name)
        f1 = cat1.getNSimpleField()
        f2 = cat2.getNSimpleField()

        if f1.sphere != f2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if f1.sphere:
            _treecorr.ProcessPairwiseNNSphere(self.corr, f1.data, f2.data, self.output_dots)
        else:
            _treecorr.ProcessPairwiseNNFlat(self.corr, f1.data, f2.data, self.output_dots)
        self.tot += cat1.nobj


    def finalize(self):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation of meanlogr by dividing each column by the total weight.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.meanlogr[mask1] /= self.npairs[mask1]

        # Update the units of meanlogr
        self.meanlogr[mask1] -= self.log_sep_units

        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]


    def clear(self):
        """Clear the data vectors
        """
        self.meanlogr[:] = 0.
        self.npairs[:] = 0.
        self.tot = 0.


    def process(self, cat1, cat2=None):
        """Compute the correlation function.

        If only 1 argument is given, then compute an auto-correlation function.
        If 2 arguments are given, then compute a cross-correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.

        :param cat1:    A catalog or list of catalogs for the first N field.
        :param cat2:    A catalog or list of catalogs for the second N field, if any.
                        (default: None)
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")

        if cat2 is None or len(cat2) == 0:
            self._process_all_auto(cat1)
        else:
            self._process_all_cross(cat1,cat2)
        self.finalize()


    def calculateXi(self, rr, dr=None, rd=None):
        """Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross correlations of the data and random.

        For a signal that involves a cross correlations, there should be two random
        cross-correlations: data-random and random-data, given as dr and rd.

        rr is the NNCorrelation function for random points.
        If dr is None, the simple correlation function (nn/rr - 1) is used.
        if dr is given and rd is None, then (nn - 2dr + rr)/rr is used.
        If dr and rd are both given, then (nn - dr - rd + rr)/rr is used.

        :param rr:          An NNCorrelation object for the random-random pairs.
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rd:          An NNCorrelation object for the random-data pairs, if desired and 
                            different from dr.  (default: None, which mean use rd=dr)
                        
        :returns:           (xi, varxi) as a tuple
        """
        # Each random npairs value needs to be rescaled by the ratio of total possible pairs.
        if rr.tot == 0:
            raise RuntimeError("rr has tot=0.")

        rrw = self.tot / rr.tot
        if dr is None:
            if rd is None:
                xi = (self.npairs - rr.npairs * rrw)
            else:
                rdw = self.tot / rd.tot
                xi = (self.npairs - 2.*rd.npairs * rdw + rr.npairs * rrw)
        else:
            drw = self.tot / dr.tot
            if rd is None:
                xi = (self.npairs - 2.*dr.npairs * drw + rr.npairs * rrw)
            else:
                rdw = self.tot / rd.tot
                xi = (self.npairs - rd.npairs * rdw - dr.npairs * drw + rr.npairs * rrw)
        if any(rr.npairs == 0):
            self.logger.warn("Warning: Some bins for the randoms had no pairs.")
            self.logger.warn("         Probably max_sep is larger than your field.")
        mask1 = rr.npairs != 0
        mask2 = rr.npairs == 0
        xi[mask1] /= (rr.npairs[mask1] * rrw)
        xi[mask2] = 0

        varxi = numpy.zeros_like(rr.npairs)
        varxi[mask1] = 1./ (rr.npairs[mask1] * rrw)

        return xi, varxi


    def write(self, file_name, rr, dr=None, rd=None):
        """Write the correlation function to the file, file_name.

        rr is the NNCorrelation function for random points.
        If dr is None, the simple correlation function (nn - rr)/rr is used.
        if dr is given and rd is None, then (nn - 2dr + rr)/rr is used.
        If dr and rd are both given, then (nn - dr - rd + rr)/rr is used.

        :param file_name:   The name of the file to write to.
        :param rr:          An NNCorrelation object for the random-random pairs.
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rd:          An NNCorrelation object for the random-data pairs, if desired and 
                            different from dr.  (default: None, which mean use rd=dr)
        """
        self.logger.info('Writing NN correlations to %s',file_name)
        
        xi, varxi = self.calculateXi(rr,dr,rd)

        headers = ['R_nom','<R>','xi','sigma_xi','DD','RR']
        columns = [ numpy.exp(self.logr), numpy.exp(self.meanlogr),
                    xi, numpy.sqrt(varxi),
                    self.npairs, rr.npairs * (self.tot/rr.tot) ]

        if dr is not None or rd is not None:
            if dr is None: dr = rd
            if rd is None: rd = dr
            if dr.tot == 0:
                raise RuntimeError("dr has tot=0.")
            if rd.tot == 0:
                raise RuntimeError("rd has tot=0.")
            headers += ['DR','RD']
            columns += [ dr.npairs * (self.tot/dr.tot), rd.npairs * (self.tot/rd.tot) ]

        self.gen_write(file_name, headers, columns)


    def calculateNapSq(self, rr, dr=None, rd=None, m2_uform=None):
        """Calculate the correlary to the aperture mass statistics for counts.

        This is used by NGCorrelation.writeNorm.  See that function and also 
        GGCorrelation.calculateMapSq() for more details.

        :param rr:          An NNCorrelation object for the random-random pairs.
        :param dr:          An NNCorrelation object for the data-random pairs, if desired, in which
                            case the Landy-Szalay estimator will be calculated.  (default: None)
        :param rd:          An NNCorrelation object for the random-data pairs, if desired and 
                            different from dr.  (default: None, which mean use rd=dr)
        :param m2_uform:    Which form to use for the aperture mass.  (default: None)

        :returns: (nsq, varnsq)
        """
        if m2_uform is None:
            m2_uform = self.config.get('m2_uform','Crittenden')
        if m2_uform not in ['Crittenden', 'Schneider']:
            raise ValueError("Invalid m2_uform")

        # Make s a matrix, so we can eventually do the integral by doing a matrix product.
        r = numpy.exp(self.logr)
        meanr = numpy.exp(self.meanlogr) # Use the actual mean r for each bin
        s = numpy.outer(1./r, meanr)  
        ssq = s*s
        if m2_uform == 'Crittenden':
            exp_factor = numpy.exp(-ssq/4.)
            Tp = (32. + ssq*(-16. + ssq)) / 128. * exp_factor
        else:
            Tp = numpy.zeros_like(s)
            sa = s[s<2.]
            ssqa = ssq[s<2.]
            Tp[s<2.] = 12./(5.*numpy.pi) * (2.-15.*ssqa) * numpy.arccos(sa/2.)
            Tp[s<2.] += 1./(100.*numpy.pi) * sa * numpy.sqrt(4.-ssqa) * (
                        120. + ssqa*(2320. + ssqa*(-754. + ssqa*(132. - 9.*ssqa))))
        Tp *= ssq

        xi, varxi = self.calculateXi(rr,dr,rd)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxi = Tp.dot(xi)
        nsq = Tpxi * self.bin_size
        varnsq = (Tp**2).dot(varxi) * self.bin_size**2

        return nsq, varnsq



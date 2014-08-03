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
# 3. Neither the name of the {organization} nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.


import treecorr
import numpy

# Start by loading up the relevant C functions using ctypes
import ctypes
import os

# The numpy version of this function tries to be more portable than the native
# ctypes.cdll.LoadLibary or cdtypes.CDLL functions.
_treecorr = numpy.ctypeslib.load_library('_treecorr',os.path.dirname(__file__))


class N2Correlation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    It holds the following attributes:

        logr        The nominal center of the bin in log(r).
        meanlogr    The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        npairs      The number of pairs going into each bin.
        tot         The total number of pairs processed, which is used to normalize
                    the randoms if they have a different number of pairs.

    The usage pattern is as follows:

        nn = treecorr.N2Correlation(config)
        nn.process(cat1)        # For auto-correlation.
        nn.process(cat1,cat2)   # For cross-correlation.
        rr.process...           # Likewise for random-random correlations
        nr.process...           # If desired, also do data-random correlations
        rn.process...           # For cross-correlations, also do the reverse.
        nn.write(file_name,rr,nr,rn)         # Write out to a file.
        xi,varxi = nn.calculateXi(rr,nr,rn)  # Or get the calculated correlation function directly.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger, **kwargs)

        self.xi = numpy.zeros(self.nbins, dtype=float)
        self.varxi = numpy.zeros(self.nbins, dtype=float)
        self.tot = 0.

        # an alias
        double_ptr = ctypes.POINTER(ctypes.c_double)

        meanlogr = self.meanlogr.ctypes.data_as(double_ptr)
        npairs = self.npairs.ctypes.data_as(double_ptr)

        _treecorr.BuildNNCorr.restype = ctypes.c_void_p
        _treecorr.BuildNNCorr.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            double_ptr, double_ptr ]

        self.corr = _treecorr.BuildNNCorr(self.min_sep,self.max_sep,self.nbins,self.bin_size,self.b,
                                          meanlogr,npairs);
        self.logger.debug('Finished building NNCorr')

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.
        if hasattr(self,'data'):    # In case __init__ failed to get that far
            _treecorr.DestroyNNCorr.argtypes = [ ctypes.c_void_p ]
            _treecorr.DestroyNNCorr(self.corr)


    def process_auto(self, cat1):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the auto-correlation for the given catalog.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.
        """
        self.logger.info('Starting process N2 auto-correlations for cat %s.',cat1.file_name)
        nfield = cat1.getNField(self.min_sep,self.max_sep,self.b)

        if nfield.sphere:
            _treecorr.ProcessAutoNNSphere.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int ]
            _treecorr.ProcessAutoNNSphere(self.corr, nfield.data, self.output_dots)
        else:
            _treecorr.ProcessAutoNNFlat.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int ]
            _treecorr.ProcessAutoNNFlat(self.corr, nfield.data, self.output_dots)

    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the cross-correlation for the given catalogs.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation of meanlogr.
        """
        self.logger.info('Starting process N2 cross-correlations for cats %s, %s.',
                         cat1.file_name, cat2.file_name)
        nfield1 = cat1.getNField(self.min_sep,self.max_sep,self.b)
        nfield2 = cat2.getNField(self.min_sep,self.max_sep,self.b)

        if nfield1.sphere != nfield2.sphere:
            raise AttributeError("Cannot correlate catalogs with different coordinate systems.")

        if nfield1.sphere:
            _treecorr.ProcessCrossNNSphere.argtypes = [ 
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_voidp, ctypes.c_int ]
            _treecorr.ProcessCrossNNSphere(self.corr, nfield1.data, nfield2.data, self.output_dots)
        else:
            _treecorr.ProcessCrossNNFlat.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_voidp, ctypes.c_int ]
            _treecorr.ProcessCrossNNFlat(self.corr, nfield1.data, nfield2.data, self.output_dots)


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
        """
        self.clear()
        if not isinstance(cat1,list): cat1 = [cat1]
        if cat2 is not None and not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")

        if cat2 is None or len(cat2) == 0:
            if self.config.get('do_auto_corr',False) or len(cat1) == 1:
                for c1 in cat1:
                    self.process_auto(c1)
                    self.tot += 0.5*c1.nobj**2
            if self.config.get('do_cross_corr',True):
                for i,c1 in enumerate(cat1):
                    for c2 in cat1[i+1:]:
                        self.process_cross(c1,c2)
                        self.tot += c1.nobj*c2.nobj
        else:
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2)
                    self.tot += c1.nobj*c2.nobj
        self.finalize()


    def calculateXi(self, rr, nr=None, rn=None):
        """Calculate the correlation function given another correlation function of random
        points using the same mask, and possibly cross correlations of the data and random.

        For a signal that involves a cross correlations, there should be two random
        cross-correlations: data-random and random-data, given as nr and rn.

        rr is the N2Correlation function for random points.
        If nr is None, the simple correlation function (nn/rr - 1) is used.
        if nr is given and rn is None, then (nn - 2nr + rr)/rr is used.
        If nr and rn are both given, then (nn - nr - rn + rr)/rr is used.

        returns (xi, varxi)
        """
        # Each random npairs value needs to be rescaled by the ratio of total possible pairs.
        rrw = self.tot / rr.tot
        if nr is None:
            if rn is None:
                xi = (self.npairs - rr.npairs * rrw)
            else:
                rnw = self.tot / rn.tot
                xi = (self.npairs - 2.*rn.npairs * rnw + rr.npairs * rrw)
        else:
            nrw = self.tot / nr.tot
            if rn is None:
                xi = (self.npairs - 2.*nr.npairs * nrw + rr.npairs * rrw)
            else:
                rnw = self.tot / rn.tot
                xi = (self.npairs - rn.npairs * rnw - nr.npairs * nrw + rr.npairs * rrw)
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

    def write(self, file_name, rr, nr=None, rn=None):
        """Write the correlation function to the file, file_name.

        rr is the N2Correlation function for random points.
        If nr is None, the simple correlation function (nn - rr)/rr is used.
        if nr is given and rn is None, then (nn - 2nr + rr)/rr is used.
        If nr and rn are both given, then (nn - nr - rn + rr)/rr is used.
        """
        self.logger.info('Writing N2 correlations to %s',file_name)
        
        xi, varxi = self.calculateXi(rr,nr,rn)

        headers = ['R_nom','<R>','xi','sigma_xi','NN','RR']
        columns = [ numpy.exp(self.logr), numpy.exp(self.meanlogr),
                    xi, numpy.sqrt(varxi),
                    self.npairs, rr.npairs * (self.tot/rr.tot) ]

        if nr is not None or rn is not None:
            if nr is None: nr = rn
            if rn is None: rn = nr
            headers += ['NR','RN']
            columns += [ nr.npairs * (self.tot/nr.tot), rn.npairs * (self.tot/rn.tot) ]

        self.gen_write(file_name, headers, columns)

    def calculateNapSq(self, rr, nr=None, rn=None, m2_uform=None):
        """Calculate the correlary to the aperture mass statistics for counts.

        This is used by NGCorrelation.writeNorm.  See that function and also 
        G2Correlation.calculateMapSq() for more details.

        returns (nsq, varnsq)
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

        xi, varxi = self.calculateXi(rr,nr,rn)

        # Now do the integral by taking the matrix products.
        # Note that dlogr = bin_size
        Tpxi = Tp.dot(xi)
        nsq = Tpxi * self.bin_size
        varnsq = (Tp**2).dot(varxi) * self.bin_size**2

        return nsq, varnsq




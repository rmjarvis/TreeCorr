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

class NGCorrelation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    It holds the following attributes:

        logr        The nominal center of the bin in log(r).
        meanlogr    The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xi          The correlation function, xi(r) = <gamma_T>.
        xi_im       The imaginary part of xi(r).
        varxi       The variance of xi, only including the shape noise propagated into the
                    final correlation.  This does not include sample variance, so it is
                    always an underestimate of the actual variance.
        weight      The total weight in each bin.
        npairs      The number of pairs going into each bin.

    The usage pattern is as follows:

        ng = treecorr.NGCorrelation(config)
        ng.process(cat1,cat2)   # Compute the cross-correlation.
        ng.write(file_name)     # Write out to a file.
        xi = gg.xi              # Or access the correlation function directly.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger=None, **kwargs)

        self.xi = numpy.zeros(self.nbins, dtype=float)
        self.xi_im = numpy.zeros(self.nbins, dtype=float)

    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Process NG cross-correlations for cats %s,%s...  or not.',
                         cat1.file_name, cat2.file_name)

    def finalize(self, varg):
        """Finalize the calculation of the correlation function.

        The process_cross command accumulates values in each bin, so it can be called
        multiple times if appropriate.  Afterwards, this command finishes the calculation
        by dividing each column by the total weight.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.xi[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = varg / self.npairs[mask1]

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

    def process(self, cat1, cat2):
        """Compute the correlation function.

        Both arguments may be lists, in which case all items in the list are used 
        for that element of the correlation.
        """
        self.clear()

        if not isinstance(cat1,list): cat1 = [cat1]
        if not isinstance(cat2,list): cat2 = [cat2]
        if len(cat1) == 0:
            raise ValueError("No catalogs provided for cat1")
        if len(cat2) == 0:
            raise ValueError("No catalogs provided for cat2")

        varg = treecorr.calculateVarG(cat2)
        for c1 in cat1:
            for c2 in cat2:
                self.process_cross(c1,c2)
        self.finalize(varg)

    def calculateXi(self, rg=None):
        """Calculate the correlation function possibly given another correlation function
        that uses random points for the foreground objects.

        If rg is None, the simple correlation function <gamma_T> is returned.
        If rg is not None, then a compensated calculation is done: <gamma_T> = (ng - rg)
        """
        if rg is None:
            return self.xi, self.xi_im
        else:
            return self.xi - rg.xi, self.xi_im - rg.xi_im

    def write(self, file_name, rg=None):
        """Write the correlation function to the file, file_name.

        If rg is None, the simple correlation function <gamma_T> is used.
        If rg is not None, then a compensated calculation is done: <gamma_T> = (ng - rg)
        """
        self.logger.info('Writing NG correlations to %s',file_name)
    
        xi, xi_im = self.calculateXi(rg)
        
        output = numpy.empty( (self.nbins, 7) )
        output[:,0] = numpy.exp(self.logr)
        output[:,1] = numpy.exp(self.meanlogr)
        output[:,2] = xi
        output[:,3] = xi_im
        output[:,4] = numpy.sqrt(self.varxi)
        output[:,5] = self.weight
        output[:,6] = self.npairs

        prec = self.config.get('precision',3)
        width = prec+8
        header_form = 6*("{:^%d}."%width) + "{:^%d}"%width
        header = header_form.format('R_nom','<R>','real(xi)','imag(xi)','sigma_xi','weight',
                                    'npairs')
        fmt = '%%%d.%de'%(width,prec)
        numpy.savetxt(file_name, output, fmt=fmt, header=header)

    def writeNMap(self, file_name, rg=None):
        self.logger.info('Writing NMap from NG correlations to %s',file_name)

    def writeNorm(self, file_name, gg, nn, rr, nr=None, rg=None):
        self.logger.info('Writing Norm from NG correlations to %s',file_name)


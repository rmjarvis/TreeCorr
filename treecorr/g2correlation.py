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

class G2Correlation(treecorr.BinnedCorr2):
    """This class handles the calculation and storage of a 2-point shear-shear correlation
    function.

    It holds the following attributes:

        logr        The nominal center of the bin in log(r).
        meanlogr    The (weighted) mean value of log(r) for the pairs in each bin.
                    If there are no pairs in a bin, then logr will be used instead.
        xip         The correlation function, xi_plus(r).
        xim         The correlation funciton, xi_minus(r).
        xip_im      The imaginary part of xi_plus(r).
        xim_im      The imaginary part of xi_plus(r).
        varxi       The variance of xip and xim, only including the shape noise propagated
                    into the final correlation.  This does not include sample variance, so
                    it is always an underestimate of the actual variance.
        weight      The total weight in each bin.
        npairs      The number of pairs going into each bin.

    The usage pattern is as follows:

        gg = treecorr.G2Correlation(config)
        gg.process(cat1)        # For auto-correlation.
        gg.process(cat1,cat2)   # For cross-correlation.
        gg.write(file_name)     # Write out to a file.
        xip = gg.xip            # Or access the correlation function directly.
    """
    def __init__(self, config=None, logger=None, **kwargs):
        treecorr.BinnedCorr2.__init__(self, config, logger=None, **kwargs)

        self.xip = numpy.zeros(self.nbins, dtype=float)
        self.xim = numpy.zeros(self.nbins, dtype=float)
        self.xip_im = numpy.zeros(self.nbins, dtype=float)
        self.xim_im = numpy.zeros(self.nbins, dtype=float)

    def process_auto(self, cat1):
        """Process a single catalog, accumulating the auto-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Process G2 auto-correlations for cat %s...  or not.',cat1.file_name)
        gfield = cat1.getGField(self.min_sep,self.max_sep,self.b)

    def process_cross(self, cat1, cat2):
        """Process a single pair of catalogs, accumulating the cross-correlation.

        This accumulates the weighted sums into the bins, but does not finalize
        the calculation by dividing by the total weight at the end.  After
        calling this function as often as desired, the finalize() command will
        finish the calculation.
        """
        self.logger.info('Process G2 cross-correlations for cats %s,%s...  or not.',
                         cat1.file_name, cat2.file_name)

    def finalize(self, varg1, varg2):
        """Finalize the calculation of the correlation function.

        The process_auto and process_cross commands accumulate values in each bin,
        so they can be called multiple times if appropriate.  Afterwards, this command
        finishes the calculation by dividing each column by the total weight.
        """
        mask1 = self.npairs != 0
        mask2 = self.npairs == 0

        self.xip[mask1] /= self.weight[mask1]
        self.xim[mask1] /= self.weight[mask1]
        self.xip_im[mask1] /= self.weight[mask1]
        self.xim_im[mask1] /= self.weight[mask1]
        self.meanlogr[mask1] /= self.weight[mask1]
        self.varxi[mask1] = varg1 * varg2 / self.npairs[mask1]

        # Use meanlogr when available, but set to nominal when no pairs in bin.
        self.meanlogr[mask2] = self.logr[mask2]
        self.varxi[mask2] = 0.

    def clear(self):
        """Clear the data vectors
        """
        self.xip[:] = 0
        self.xim[:] = 0
        self.xip_im[:] = 0
        self.xim_im[:] = 0
        self.meanlogr[:] = 0
        self.weight[:] = 0
        self.npairs[:] = 0

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
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = varg1

            if self.config.get('do_auto_corr',False) or len(cat1) == 1:
                for c1 in cat1:
                    self.process_auto(c1)

            if self.config.get('do_cross_corr',True):
                for i,c1 in enumerate(cat1):
                    for c2 in cat1[i+1:]:
                        self.process_cross(c1,c2)
        else:
            varg1 = treecorr.calculateVarG(cat1)
            varg2 = treecorr.calculateVarG(cat2)
            for c1 in cat1:
                for c2 in cat2:
                    self.process_cross(c1,c2)

        self.finalize(varg1,varg2)


    def write(self, file_name):
        """Write the correlation function to the file, file_name.
        """
        self.logger.info('Writing G2 correlations to %s',file_name)
        
        output = numpy.empty( (self.nbins, 9) )
        output[:,0] = numpy.exp(self.logr)
        output[:,1] = numpy.exp(self.meanlogr)
        output[:,2] = self.xip
        output[:,3] = self.xip_im
        output[:,4] = self.xim
        output[:,5] = self.xim_im
        output[:,6] = numpy.sqrt(self.varxi)
        output[:,7] = self.weight
        output[:,8] = self.npairs

        prec = self.config.get('precision',3)
        width = prec+8
        header_form = 8*("{:^%d}."%width) + "{:^%d}"%width
        header = header_form.format('R_nom','<R>','real(xi+)','imag(xi+)','real(xi-)','imag(xi-)',
                                   'sigma_xi','weight','npairs')
        fmt = '%%%d.%de'%(width,prec)
        numpy.savetxt(file_name, output, fmt=fmt, header=header)

    def calculateMapSq(self):
        """Calculate the aperture mass statistics from the correlation function.
        """

    def writeMapSq(self, file_name):
        """Write the aperture mass statistics based on the correlation function to the
        file, file_name.
        """
        self.logger.info('Writing Map^2 from G2 correlations to %s',file_name)


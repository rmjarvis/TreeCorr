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


import numpy
import treecorr
import os

from test_helper import get_aardvark

def test_aardvark():

    # Eric Suchyta did a brute force calculation of the Aardvark catalog, so it is useful to
    # compare the output from my code with that.

    get_aardvark()
    file_name = os.path.join('data','Aardvark.fit')
    config = treecorr.read_config('Aardvark.params')
    cat1 = treecorr.Catalog(file_name, config)
    gg = treecorr.G2Correlation(config)
    gg.process(cat1)

    direct_file_name = os.path.join('data','Aardvark.direct.dat')
    direct_data = numpy.loadtxt(direct_file_name)
    direct_xip = direct_data[:,3]
    direct_xim = direct_data[:,4]

    #print 'gg.xip = ',gg.xip
    #print 'direct.xip = ',direct_xip

    xip_err = gg.xip - direct_xip
    print 'xip_err = ',xip_err
    print 'max = ',max(xip_err)
    assert max(xip_err) < 2.e-7

    xim_err = gg.xim - direct_xim
    print 'xim_err = ',xim_err
    print 'max = ',max(xim_err)
    assert max(xim_err) < 1.e-7

    # However, after some back and forth about the calculation, we concluded that Eric hadn't
    # done the spherical trig correctly to get the shears relative to the great circle joining
    # the two positions.  So let's compare with my own brute force calculation (i.e. using
    # bin_slop = 0):
    # This also has the advantage that the radial bins are done the same way -- uniformly 
    # spaced in log of the chord distance, rather than the great circle distance.

    bs0_file_name = os.path.join('data','Aardvark.bs0')
    bs0_data = numpy.loadtxt(bs0_file_name)
    bs0_xip = bs0_data[:,2]
    bs0_xim = bs0_data[:,3]

    #print 'gg.xip = ',gg.xip
    #print 'bs0.xip = ',bs0_xip

    xip_err = gg.xip - bs0_xip
    print 'xip_err = ',xip_err
    print 'max = ',max(xip_err)
    assert max(xip_err) < 1.e-7

    xim_err = gg.xim - bs0_xim
    print 'xim_err = ',xim_err
    print 'max = ',max(xim_err)
    assert max(xim_err) < 5.e-8

    # As bin_slop decreases, the agreement should get even better.
    if __name__ == '__main__':
        # This test is slow, so only do it if running test_g2.py directly.
        config['bin_slop'] = 0.2
        gg = treecorr.G2Correlation(config)
        gg.process(cat1)

        #print 'gg.xip = ',gg.xip
        #print 'bs0.xip = ',bs0_xip

        xip_err = gg.xip - bs0_xip
        print 'xip_err = ',xip_err
        print 'max = ',max(xip_err)
        assert max(xip_err) < 1.e-8

        xim_err = gg.xim - bs0_xim
        print 'xim_err = ',xim_err
        print 'max = ',max(xim_err)
        assert max(xim_err) < 1.e-8

 
if __name__ == '__main__':
    test_aardvark()

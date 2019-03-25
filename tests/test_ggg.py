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

from __future__ import print_function
import numpy as np
import treecorr
import os
import coord

from test_helper import get_script_name, do_pickle

def test_direct():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if bin_slop=0.

    ngal = 100
    s = 10.
    np.random.seed(8675309)
    x = np.random.normal(0,s, (ngal,) )
    y = np.random.normal(0,s, (ngal,) )
    w = np.random.random(ngal)
    g1 = np.random.normal(0,0.2, (ngal,) )
    g2 = np.random.normal(0,0.2, (ngal,) )

    cat = treecorr.Catalog(x=x, y=y, w=w, g1=g1, g2=g2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    nubins = 5
    nvbins = 10
    max_sep = min_sep * np.exp(nrbins * bin_size)
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins, bin_slop=0.)
    ggg.process(cat, num_threads=2)

    true_ntri = np.zeros((nrbins, nubins, nvbins), dtype=int)
    true_weight = np.zeros((nrbins, nubins, nvbins), dtype=float)
    true_gam0 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    true_gam1 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    true_gam2 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    true_gam3 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                d12 = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                d23 = np.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2)
                d31 = np.sqrt((x[k]-x[i])**2 + (y[k]-y[i])**2)

                d3, d2, d1 = sorted([d12, d23, d31])
                rindex = np.floor(np.log(d2/min_sep) / bin_size).astype(int)
                if rindex < 0 or rindex >= nrbins: continue

                if [d1, d2, d3] == [d23, d31, d12]: ii,jj,kk = i,j,k
                elif [d1, d2, d3] == [d23, d12, d31]: ii,jj,kk = i,k,j
                elif [d1, d2, d3] == [d31, d12, d23]: ii,jj,kk = j,k,i
                elif [d1, d2, d3] == [d31, d23, d12]: ii,jj,kk = j,i,k
                elif [d1, d2, d3] == [d12, d23, d31]: ii,jj,kk = k,i,j
                elif [d1, d2, d3] == [d12, d31, d23]: ii,jj,kk = k,j,i
                else: assert False
                # Now use ii, jj, kk rather than i,j,k, to get the indices
                # that correspond to the points in the right order.

                u = d3/d2
                v = (d1-d2)/d3
                if (x[jj]-x[ii])*(y[kk]-y[ii]) < (x[kk]-x[ii])*(y[jj]-y[ii]):
                    v = -v

                uindex = np.floor(u / bin_size).astype(int)
                assert 0 <= uindex < nubins
                vindex = np.floor((v+1) / bin_size).astype(int)
                assert 0 <= vindex < nvbins

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x[i] + x[j] + x[k])/3.
                ceny = (y[i] + y[j] + y[k])/3.

                expmialpha1 = (x[ii]-cenx) - 1j*(y[ii]-ceny)
                expmialpha1 /= abs(expmialpha1)
                expmialpha2 = (x[jj]-cenx) - 1j*(y[jj]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x[kk]-cenx) - 1j*(y[kk]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w[i] * w[j] * w[k]
                g1p = (g1[ii] + 1j*g2[ii]) * expmialpha1**2
                g2p = (g1[jj] + 1j*g2[jj]) * expmialpha2**2
                g3p = (g1[kk] + 1j*g2[kk]) * expmialpha3**2
                gam0 = www * g1p * g2p * g3p
                gam1 = www * np.conjugate(g1p) * g2p * g3p
                gam2 = www * g1p * np.conjugate(g2p) * g3p
                gam3 = www * g1p * g2p * np.conjugate(g3p)

                true_ntri[rindex,uindex,vindex] += 1
                true_weight[rindex,uindex,vindex] += www
                true_gam0[rindex,uindex,vindex] += gam0
                true_gam1[rindex,uindex,vindex] += gam1
                true_gam2[rindex,uindex,vindex] += gam2
                true_gam3[rindex,uindex,vindex] += gam3

    pos = true_weight > 0
    true_gam0[pos] /= true_weight[pos]
    true_gam1[pos] /= true_weight[pos]
    true_gam2[pos] /= true_weight[pos]
    true_gam3[pos] /= true_weight[pos]

    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0, true_gam0, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1, true_gam1, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2, true_gam2, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3, true_gam3, rtol=1.e-5, atol=1.e-8)

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # Check that running via the corr3 script works correctly.
    config = treecorr.config.read_config('configs/ggg_direct.yaml')
    cat.write(config['file_name'])
    treecorr.corr3(config)
    data = fitsio.read(config['ggg_file_name'])
    np.testing.assert_allclose(data['R_nom'], ggg.rnom.flatten())
    np.testing.assert_allclose(data['u_nom'], ggg.u.flatten())
    np.testing.assert_allclose(data['v_nom'], ggg.v.flatten())
    np.testing.assert_allclose(data['ntri'], ggg.ntri.flatten())
    np.testing.assert_allclose(data['weight'], ggg.weight.flatten())
    np.testing.assert_allclose(data['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Also check the "cross" calculation.  (Real cross doesn't work, but this should.)
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins, bin_slop=0.)
    ggg.process(cat, cat, cat, num_threads=2)
    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    config['file_name2'] = config['file_name']
    config['file_name3'] = config['file_name']
    treecorr.corr3(config)
    data = fitsio.read(config['ggg_file_name'])
    np.testing.assert_allclose(data['R_nom'], ggg.rnom.flatten())
    np.testing.assert_allclose(data['u_nom'], ggg.u.flatten())
    np.testing.assert_allclose(data['v_nom'], ggg.v.flatten())
    np.testing.assert_allclose(data['ntri'], ggg.ntri.flatten())
    np.testing.assert_allclose(data['weight'], ggg.weight.flatten())
    np.testing.assert_allclose(data['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  bin_slop=1.e-16, max_top=0)
    ggg.process(cat)
    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-3, atol=1.e-4)

    # Check a few basic operations with a GGCorrelation object.
    do_pickle(ggg)

    ggg2 = ggg.copy()
    ggg2 += ggg
    np.testing.assert_allclose(ggg2.ntri, 2*ggg.ntri)
    np.testing.assert_allclose(ggg2.weight, 2*ggg.weight)
    np.testing.assert_allclose(ggg2.meand1, 2*ggg.meand1)
    np.testing.assert_allclose(ggg2.meand2, 2*ggg.meand2)
    np.testing.assert_allclose(ggg2.meand3, 2*ggg.meand3)
    np.testing.assert_allclose(ggg2.meanlogd1, 2*ggg.meanlogd1)
    np.testing.assert_allclose(ggg2.meanlogd2, 2*ggg.meanlogd2)
    np.testing.assert_allclose(ggg2.meanlogd3, 2*ggg.meanlogd3)
    np.testing.assert_allclose(ggg2.meanu, 2*ggg.meanu)
    np.testing.assert_allclose(ggg2.meanv, 2*ggg.meanv)
    np.testing.assert_allclose(ggg2.gam0r, 2*ggg.gam0r)
    np.testing.assert_allclose(ggg2.gam0i, 2*ggg.gam0i)
    np.testing.assert_allclose(ggg2.gam1r, 2*ggg.gam1r)
    np.testing.assert_allclose(ggg2.gam1i, 2*ggg.gam1i)
    np.testing.assert_allclose(ggg2.gam2r, 2*ggg.gam2r)
    np.testing.assert_allclose(ggg2.gam2i, 2*ggg.gam2i)
    np.testing.assert_allclose(ggg2.gam3r, 2*ggg.gam3r)
    np.testing.assert_allclose(ggg2.gam3i, 2*ggg.gam3i)

    ggg2.clear()
    ggg2 += ggg
    np.testing.assert_allclose(ggg2.ntri, ggg.ntri)
    np.testing.assert_allclose(ggg2.weight, ggg.weight)
    np.testing.assert_allclose(ggg2.meand1, ggg.meand1)
    np.testing.assert_allclose(ggg2.meand2, ggg.meand2)
    np.testing.assert_allclose(ggg2.meand3, ggg.meand3)
    np.testing.assert_allclose(ggg2.meanlogd1, ggg.meanlogd1)
    np.testing.assert_allclose(ggg2.meanlogd2, ggg.meanlogd2)
    np.testing.assert_allclose(ggg2.meanlogd3, ggg.meanlogd3)
    np.testing.assert_allclose(ggg2.meanu, ggg.meanu)
    np.testing.assert_allclose(ggg2.meanv, ggg.meanv)
    np.testing.assert_allclose(ggg2.gam0r, ggg.gam0r)
    np.testing.assert_allclose(ggg2.gam0i, ggg.gam0i)
    np.testing.assert_allclose(ggg2.gam1r, ggg.gam1r)
    np.testing.assert_allclose(ggg2.gam1i, ggg.gam1i)
    np.testing.assert_allclose(ggg2.gam2r, ggg.gam2r)
    np.testing.assert_allclose(ggg2.gam2i, ggg.gam2i)
    np.testing.assert_allclose(ggg2.gam3r, ggg.gam3r)
    np.testing.assert_allclose(ggg2.gam3i, ggg.gam3i)

    ascii_name = 'output/ggg_ascii.txt'
    ggg.write(ascii_name, precision=16)
    ggg3 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins)
    ggg3.read(ascii_name)
    np.testing.assert_allclose(ggg3.ntri, ggg.ntri)
    np.testing.assert_allclose(ggg3.weight, ggg.weight)
    np.testing.assert_allclose(ggg3.meand1, ggg.meand1)
    np.testing.assert_allclose(ggg3.meand2, ggg.meand2)
    np.testing.assert_allclose(ggg3.meand3, ggg.meand3)
    np.testing.assert_allclose(ggg3.meanlogd1, ggg.meanlogd1)
    np.testing.assert_allclose(ggg3.meanlogd2, ggg.meanlogd2)
    np.testing.assert_allclose(ggg3.meanlogd3, ggg.meanlogd3)
    np.testing.assert_allclose(ggg3.meanu, ggg.meanu)
    np.testing.assert_allclose(ggg3.meanv, ggg.meanv)
    np.testing.assert_allclose(ggg3.gam0r, ggg.gam0r)
    np.testing.assert_allclose(ggg3.gam0i, ggg.gam0i)
    np.testing.assert_allclose(ggg3.gam1r, ggg.gam1r)
    np.testing.assert_allclose(ggg3.gam1i, ggg.gam1i)
    np.testing.assert_allclose(ggg3.gam2r, ggg.gam2r)
    np.testing.assert_allclose(ggg3.gam2i, ggg.gam2i)
    np.testing.assert_allclose(ggg3.gam3r, ggg.gam3r)
    np.testing.assert_allclose(ggg3.gam3i, ggg.gam3i)

    fits_name = 'output/ggg_fits.fits'
    ggg.write(fits_name)
    ggg4 = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins)
    ggg4.read(fits_name)
    np.testing.assert_allclose(ggg4.ntri, ggg.ntri)
    np.testing.assert_allclose(ggg4.weight, ggg.weight)
    np.testing.assert_allclose(ggg4.meand1, ggg.meand1)
    np.testing.assert_allclose(ggg4.meand2, ggg.meand2)
    np.testing.assert_allclose(ggg4.meand3, ggg.meand3)
    np.testing.assert_allclose(ggg4.meanlogd1, ggg.meanlogd1)
    np.testing.assert_allclose(ggg4.meanlogd2, ggg.meanlogd2)
    np.testing.assert_allclose(ggg4.meanlogd3, ggg.meanlogd3)
    np.testing.assert_allclose(ggg4.meanu, ggg.meanu)
    np.testing.assert_allclose(ggg4.meanv, ggg.meanv)
    np.testing.assert_allclose(ggg4.gam0r, ggg.gam0r)
    np.testing.assert_allclose(ggg4.gam0i, ggg.gam0i)
    np.testing.assert_allclose(ggg4.gam1r, ggg.gam1r)
    np.testing.assert_allclose(ggg4.gam1i, ggg.gam1i)
    np.testing.assert_allclose(ggg4.gam2r, ggg.gam2r)
    np.testing.assert_allclose(ggg4.gam2i, ggg.gam2i)
    np.testing.assert_allclose(ggg4.gam3r, ggg.gam3r)
    np.testing.assert_allclose(ggg4.gam3i, ggg.gam3i)

def test_direct_spherical():
    # Repeat in spherical coords

    ngal = 50
    s = 10.
    np.random.seed(8675309)
    x = np.random.normal(0,s, (ngal,) )
    y = np.random.normal(0,s, (ngal,) ) + 200  # Put everything at large y, so small angle on sky
    z = np.random.normal(0,s, (ngal,) )
    w = np.random.random(ngal)
    g1 = np.random.normal(0,0.2, (ngal,) )
    g2 = np.random.normal(0,0.2, (ngal,) )
    w = np.ones_like(w)

    ra, dec = coord.CelestialCoord.xyz_to_radec(x,y,z)

    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='rad', dec_units='rad', w=w, g1=g1, g2=g2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    nubins = 5
    nvbins = 10
    max_sep = min_sep * np.exp(nrbins * bin_size)
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  sep_units='deg', bin_slop=0.)
    ggg.process(cat)

    r = np.sqrt(x**2 + y**2 + z**2)
    x /= r;  y /= r;  z /= r
    north_pole = coord.CelestialCoord(0*coord.radians, 90*coord.degrees)

    true_ntri = np.zeros((nrbins, nubins, nvbins), dtype=int)
    true_weight = np.zeros((nrbins, nubins, nvbins), dtype=float)
    true_gam0 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    true_gam1 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    true_gam2 = np.zeros((nrbins, nubins, nvbins), dtype=complex)
    true_gam3 = np.zeros((nrbins, nubins, nvbins), dtype=complex)

    rad_min_sep = min_sep * coord.degrees / coord.radians
    rad_max_sep = max_sep * coord.degrees / coord.radians
    c = [coord.CelestialCoord(r*coord.radians, d*coord.radians) for (r,d) in zip(ra, dec)]
    for i in range(ngal):
        for j in range(i+1,ngal):
            for k in range(j+1,ngal):
                d12 = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
                d23 = np.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2 + (z[j]-z[k])**2)
                d31 = np.sqrt((x[k]-x[i])**2 + (y[k]-y[i])**2 + (z[k]-z[i])**2)

                d3, d2, d1 = sorted([d12, d23, d31])
                rindex = np.floor(np.log(d2/rad_min_sep) / bin_size).astype(int)
                if rindex < 0 or rindex >= nrbins: continue

                if [d1, d2, d3] == [d23, d31, d12]: ii,jj,kk = i,j,k
                elif [d1, d2, d3] == [d23, d12, d31]: ii,jj,kk = i,k,j
                elif [d1, d2, d3] == [d31, d12, d23]: ii,jj,kk = j,k,i
                elif [d1, d2, d3] == [d31, d23, d12]: ii,jj,kk = j,i,k
                elif [d1, d2, d3] == [d12, d23, d31]: ii,jj,kk = k,i,j
                elif [d1, d2, d3] == [d12, d31, d23]: ii,jj,kk = k,j,i
                else: assert False
                # Now use ii, jj, kk rather than i,j,k, to get the indices
                # that correspond to the points in the right order.

                u = d3/d2
                v = (d1-d2)/d3
                if ( ((x[jj]-x[ii])*(y[kk]-y[ii]) - (x[kk]-x[ii])*(y[jj]-y[ii])) * z[ii] +
                     ((y[jj]-y[ii])*(z[kk]-z[ii]) - (y[kk]-y[ii])*(z[jj]-z[ii])) * x[ii] +
                     ((z[jj]-z[ii])*(x[kk]-x[ii]) - (z[kk]-z[ii])*(x[jj]-x[ii])) * y[ii] ) > 0:
                    v = -v

                uindex = np.floor(u / bin_size).astype(int)
                assert 0 <= uindex < nubins
                vindex = np.floor((v+1) / bin_size).astype(int)
                assert 0 <= vindex < nvbins

                # Rotate shears to coordinates where line connecting to center is horizontal.
                # Original orientation is where north is up.
                cenx = (x[i] + x[j] + x[k])/3.
                ceny = (y[i] + y[j] + y[k])/3.
                cenz = (z[i] + z[j] + z[k])/3.
                cen = coord.CelestialCoord.from_xyz(cenx,ceny,cenz)

                theta1 = 90*coord.degrees - c[ii].angleBetween(north_pole, cen)
                theta2 = 90*coord.degrees - c[jj].angleBetween(north_pole, cen)
                theta3 = 90*coord.degrees - c[kk].angleBetween(north_pole, cen)
                exp2theta1 = np.cos(2*theta1) + 1j * np.sin(2*theta1)
                exp2theta2 = np.cos(2*theta2) + 1j * np.sin(2*theta2)
                exp2theta3 = np.cos(2*theta3) + 1j * np.sin(2*theta3)

                www = w[i] * w[j] * w[k]
                g1p = (g1[ii] + 1j*g2[ii]) * exp2theta1
                g2p = (g1[jj] + 1j*g2[jj]) * exp2theta2
                g3p = (g1[kk] + 1j*g2[kk]) * exp2theta3
                gam0 = www * g1p * g2p * g3p
                gam1 = www * np.conjugate(g1p) * g2p * g3p
                gam2 = www * g1p * np.conjugate(g2p) * g3p
                gam3 = www * g1p * g2p * np.conjugate(g3p)

                true_ntri[rindex,uindex,vindex] += 1
                true_weight[rindex,uindex,vindex] += www
                true_gam0[rindex,uindex,vindex] += gam0
                true_gam1[rindex,uindex,vindex] += gam1
                true_gam2[rindex,uindex,vindex] += gam2
                true_gam3[rindex,uindex,vindex] += gam3

    pos = true_weight > 0
    true_gam0[pos] /= true_weight[pos]
    true_gam1[pos] /= true_weight[pos]
    true_gam2[pos] /= true_weight[pos]
    true_gam3[pos] /= true_weight[pos]

    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-5, atol=1.e-8)

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # Check that running via the corr3 script works correctly.
    config = treecorr.config.read_config('configs/ggg_direct_spherical.yaml')
    cat.write(config['file_name'])
    treecorr.corr3(config)
    data = fitsio.read(config['ggg_file_name'])
    np.testing.assert_allclose(data['R_nom'], ggg.rnom.flatten())
    np.testing.assert_allclose(data['u_nom'], ggg.u.flatten())
    np.testing.assert_allclose(data['v_nom'], ggg.v.flatten())
    np.testing.assert_allclose(data['ntri'], ggg.ntri.flatten())
    np.testing.assert_allclose(data['weight'], ggg.weight.flatten())
    np.testing.assert_allclose(data['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(data['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    # Repeat with binslop not precisely 0, since the code flow is different for bin_slop == 0.
    # And don't do any top-level recursion so we actually test not going to the leaves.
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  sep_units='deg', bin_slop=1.e-16, max_top=0)
    ggg.process(cat)
    np.testing.assert_array_equal(ggg.ntri, true_ntri)
    np.testing.assert_allclose(ggg.weight, true_weight, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0r, true_gam0.real, rtol=1.e-5, atol=1.e-8)
    np.testing.assert_allclose(ggg.gam0i, true_gam0.imag, rtol=1.e-5, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1r, true_gam1.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam1i, true_gam1.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2r, true_gam2.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam2i, true_gam2.imag, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3r, true_gam3.real, rtol=1.e-3, atol=1.e-4)
    np.testing.assert_allclose(ggg.gam3i, true_gam3.imag, rtol=1.e-3, atol=1.e-4)


def test_ggg():
    # Use gamma_t(r) = gamma0 r^2/r0^2 exp(-r^2/2r0^2)
    # i.e. gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2
    #
    # Rather than go through the bispectrum, I found it easier to just directly do the
    # integral:
    #
    # Gamma0 = int(int( g(x+x1,y+y1) g(x+x2,y+y2) g(x-x1-x2,y-y1-y2) (x1-Iy1)^2/(x1^2+y1^2)
    #                       (x2-Iy2)^2/(x2^2+y2^2) (x1+x2-I(y1+y2))^2/((x1+x2)^2+(y1+y2)^2)))
    #
    # where the positions are measured relative to the centroid (x,y).
    # If we call the positions relative to the centroid:
    #    q1 = x1 + I y1
    #    q2 = x2 + I y2
    #    q3 = -(x1+x2) - I (y1+y2)
    # then the result comes out as
    #
    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    #
    # The other three are a bit more complicated.
    #
    # Gamma1 = int(int( g(x+x1,y+y1)* g(x+x2,y+y2) g(x-x1-x2,y-y1-y2) (x1+Iy1)^2/(x1^2+y1^2)
    #                       (x2-Iy2)^2/(x2^2+y2^2) (x1+x2-I(y1+y2))^2/((x1+x2)^2+(y1+y2)^2)))
    #
    #        = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    #
    # Gamm2 and Gamma3 are found from cyclic rotations of q1,q2,q3.

    gamma0 = 0.05
    r0 = 10.
    if __name__ == '__main__':
        ngal = 200000
        L = 30.*r0  # Not infinity, so this introduces some error.  Our integrals were to infinity.
        tol_factor = 1
    else:
        # Looser tests that don't take so long to run.
        ngal = 10000
        L = 20.*r0
        tol_factor = 5
    np.random.seed(8675309)
    x = (np.random.random_sample(ngal)-0.5) * L
    y = (np.random.random_sample(ngal)-0.5) * L
    r2 = (x**2 + y**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x**2-y**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x*y)/r0**2

    min_sep = 11.
    max_sep = 15.
    nbins = 3
    min_u = 0.7
    max_u = 1.0
    nubins = 3
    min_v = -0.1
    max_v = 0.3
    nvbins = 4

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    ggg = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                  nubins=nubins, nvbins=nvbins,
                                  sep_units='arcmin', verbose=1)
    ggg.process(cat)

    # log(<d>) != <logd>, but it should be close:
    print('meanlogd1 - log(meand1) = ',ggg.meanlogd1 - np.log(ggg.meand1))
    print('meanlogd2 - log(meand2) = ',ggg.meanlogd2 - np.log(ggg.meand2))
    print('meanlogd3 - log(meand3) = ',ggg.meanlogd3 - np.log(ggg.meand3))
    print('meand3 / meand2 = ',ggg.meand3 / ggg.meand2)
    print('meanu = ',ggg.meanu)
    print('max diff = ',np.max(np.abs(ggg.meand3/ggg.meand2 -ggg.meanu)))
    print('max rel diff = ',np.max(np.abs((ggg.meand3/ggg.meand2 -ggg.meanu)/ggg.meanu)))
    print('(meand1 - meand2)/meand3 = ',(ggg.meand1-ggg.meand2) / ggg.meand3)
    print('meanv = ',ggg.meanv)
    print('max diff = ',np.max(np.abs((ggg.meand1-ggg.meand2)/ggg.meand3 -np.abs(ggg.meanv))))
    print('max rel diff = ',np.max(np.abs(((ggg.meand1-ggg.meand2)/ggg.meand3-np.abs(ggg.meanv))/ggg.meanv)))
    np.testing.assert_allclose(ggg.meanlogd1, np.log(ggg.meand1), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meanlogd2, np.log(ggg.meand2), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meanlogd3, np.log(ggg.meand3), rtol=1.e-3)
    np.testing.assert_allclose(ggg.meand3/ggg.meand2, ggg.meanu, rtol=1.e-5 * tol_factor)
    np.testing.assert_allclose((ggg.meand1-ggg.meand2)/ggg.meand3, np.abs(ggg.meanv),
                                  rtol=1.e-5 * tol_factor, atol=1.e-5 * tol_factor)
    np.testing.assert_allclose(ggg.meanlogd3-ggg.meanlogd2, np.log(ggg.meanu),
                                  atol=1.e-3 * tol_factor)
    np.testing.assert_allclose(np.log(ggg.meand1-ggg.meand2)-ggg.meanlogd3,
                                  np.log(np.abs(ggg.meanv)), atol=2.e-3 * tol_factor)

    d1 = ggg.meand1
    d2 = ggg.meand2
    d3 = ggg.meand3
    #print('rnom = ',np.exp(ggg.logr))
    #print('unom = ',ggg.u)
    #print('vnom = ',ggg.v)
    #print('d1 = ',d1)
    #print('d2 = ',d2)
    #print('d3 = ',d3)

    # For q1,q2,q3, we can choose an orientation where c1 is at the origin, and d2 is horizontal.
    # Then let s be the "complex vector" from c1 to c3, which is just real.
    s = d2
    # And let t be from c1 to c2. t = |t| e^Iphi
    # |t| = d3
    # cos(phi) = (d2^2+d3^2-d1^2)/(2d2 d3)
    # |t| cos(phi) = (d2^2+d3^2-d1^2)/2d2
    # |t| sin(phi) = sqrt(|t|^2 - (|t|cos(phi))^2)
    tx = (d2**2 + d3**2 - d1**2)/(2.*d2)
    ty = np.sqrt(d3**2 - tx**2)
    # As arranged, if ty > 0, points 1,2,3 are clockwise, which is negative v.
    # So for bins with positive v, we need to flip the direction of ty.
    ty[ggg.meanv > 0] *= -1.
    t = tx + 1j * ty

    q1 = (s + t)/3.
    q2 = q1 - t
    q3 = q1 - s
    nq1 = np.abs(q1)**2
    nq2 = np.abs(q2)**2
    nq3 = np.abs(q3)**2
    #print('q1 = ',q1)
    #print('q2 = ',q2)
    #print('q3 = ',q3)

    # The L^2 term in the denominator of true_zeta is the area over which the integral is done.
    # Since the centers of the triangles don't go to the edge of the box, we approximate the
    # correct area by subtracting off 2*mean(qi) from L, which should give a slightly better
    # estimate of the correct area to use here.  (We used 2d2 for the kkk calculation, but this
    # is probably a slightly better estimate of the distance the triangles can get to the edges.)
    L = L - 2.*(q1 + q2 + q3)/3.

    # Gamma0 = -2/3 gamma0^3/L^2r0^4 Pi |q1|^2 |q2|^2 |q3|^2 exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2)
    true_gam0 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) * (nq1*nq2*nq3) )

    # Gamma1 = -2/3 gamma0^3/L^2r0^4 Pi exp(-(|q1|^2+|q2|^2+|q3|^2)/2r0^2) *
    #             ( |q1|^2 |q2|^2 |q3|^2 - 8/3 r0^2 q1^2 q2* q3*
    #               + 8/9 r0^4 (q1^2 q2*^2 q3*^2)/(|q1|^2 |q2|^2 |q3|^2) (2q1^2-q2^2-q3^2) )
    true_gam1 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * q1**2*nq2*nq3/(q2*q3)
                     + (8./9. * r0**4 * (q1**2 * nq2 * nq3)/(nq1 * q2**2 * q3**2) *
                         (2.*q1**2 - q2**2 - q3**2)) ))

    true_gam2 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*q2**2*nq3/(q1*q3)
                     + (8./9. * r0**4 * (nq1 * q2**2 * nq3)/(q1**2 * nq2 * q3**2) *
                         (2.*q2**2 - q1**2 - q3**2)) ))

    true_gam3 = ((-2.*np.pi * gamma0**3)/(3. * L**2 * r0**4) *
                    np.exp(-(nq1+nq2+nq3)/(2.*r0**2)) *
                    (nq1*nq2*nq3 - 8./3. * r0**2 * nq1*nq2*q3**2/(q1*q2)
                     + (8./9. * r0**4 * (nq1 * nq2 * q3**2)/(q1**2 * q2**2 * nq3) *
                         (2.*q3**2 - q1**2 - q2**2)) ))

    print('ntri = ',ggg.ntri)
    print('gam0 = ',ggg.gam0)
    print('true_gam0 = ',true_gam0)
    print('ratio = ',ggg.gam0 / true_gam0)
    print('diff = ',ggg.gam0 - true_gam0)
    print('max rel diff = ',np.max(np.abs((ggg.gam0 - true_gam0)/true_gam0)))
    # The Gamma0 term is a bit worse than the others.  The accurracy improves as I increase the
    # number of objects, so I think it's just because of the smallish number of galaxies being
    # not super accurate.
    np.testing.assert_allclose(ggg.gam0, true_gam0, rtol=0.2 * tol_factor, atol=1.e-7)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam0)),
                                  np.log(np.abs(true_gam0)), atol=0.2 * tol_factor)

    print('gam1 = ',ggg.gam1)
    print('true_gam1 = ',true_gam1)
    print('ratio = ',ggg.gam1 / true_gam1)
    print('diff = ',ggg.gam1 - true_gam1)
    print('max rel diff = ',np.max(np.abs((ggg.gam1 - true_gam1)/true_gam1)))
    np.testing.assert_allclose(ggg.gam1, true_gam1, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam1)),
                                  np.log(np.abs(true_gam1)), atol=0.1 * tol_factor)

    print('gam2 = ',ggg.gam2)
    print('true_gam2 = ',true_gam2)
    print('ratio = ',ggg.gam2 / true_gam2)
    print('diff = ',ggg.gam2 - true_gam2)
    print('max rel diff = ',np.max(np.abs((ggg.gam2 - true_gam2)/true_gam2)))
    np.testing.assert_allclose(ggg.gam2, true_gam2, rtol=0.1 * tol_factor)
    print('max rel diff for log = ',np.max(np.abs((ggg.gam2 - true_gam2)/true_gam2)))
    np.testing.assert_allclose(np.log(np.abs(ggg.gam2)),
                                  np.log(np.abs(true_gam2)), atol=0.1 * tol_factor)

    print('gam3 = ',ggg.gam3)
    print('true_gam3 = ',true_gam3)
    print('ratio = ',ggg.gam3 / true_gam3)
    print('diff = ',ggg.gam3 - true_gam3)
    print('max rel diff = ',np.max(np.abs((ggg.gam3 - true_gam3)/true_gam3)))
    np.testing.assert_allclose(ggg.gam3, true_gam3, rtol=0.1 * tol_factor)
    np.testing.assert_allclose(np.log(np.abs(ggg.gam3)),
                                  np.log(np.abs(true_gam3)), atol=0.1 * tol_factor)

    # Check that we get the same result using the corr3 function:
    cat.write(os.path.join('data','ggg_data.dat'))
    config = treecorr.config.read_config('configs/ggg.yaml')
    config['verbose'] = 0
    treecorr.corr3(config)
    corr3_output = np.genfromtxt(os.path.join('output','ggg.out'), names=True, skip_header=1)
    np.testing.assert_allclose(corr3_output['gam0r'], ggg.gam0r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam0i'], ggg.gam0i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam1r'], ggg.gam1r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam1i'], ggg.gam1i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2r'], ggg.gam2r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2i'], ggg.gam2i.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam3r'], ggg.gam3r.flatten(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam3i'], ggg.gam3i.flatten(), rtol=1.e-3)

    try:
        import fitsio
    except ImportError:
        print('Skipping FITS tests, since fitsio is not installed')
        return

    # Check the fits write option
    out_file_name1 = os.path.join('output','ggg_out1.fits')
    ggg.write(out_file_name1)
    data = fitsio.read(out_file_name1)
    np.testing.assert_almost_equal(data['R_nom'], np.exp(ggg.logr).flatten())
    np.testing.assert_almost_equal(data['u_nom'], ggg.u.flatten())
    np.testing.assert_almost_equal(data['v_nom'], ggg.v.flatten())
    np.testing.assert_almost_equal(data['meand1'], ggg.meand1.flatten())
    np.testing.assert_almost_equal(data['meanlogd1'], ggg.meanlogd1.flatten())
    np.testing.assert_almost_equal(data['meand2'], ggg.meand2.flatten())
    np.testing.assert_almost_equal(data['meanlogd2'], ggg.meanlogd2.flatten())
    np.testing.assert_almost_equal(data['meand3'], ggg.meand3.flatten())
    np.testing.assert_almost_equal(data['meanlogd3'], ggg.meanlogd3.flatten())
    np.testing.assert_almost_equal(data['meanu'], ggg.meanu.flatten())
    np.testing.assert_almost_equal(data['meanv'], ggg.meanv.flatten())
    np.testing.assert_almost_equal(data['gam0r'], ggg.gam0.real.flatten())
    np.testing.assert_almost_equal(data['gam1r'], ggg.gam1.real.flatten())
    np.testing.assert_almost_equal(data['gam2r'], ggg.gam2.real.flatten())
    np.testing.assert_almost_equal(data['gam3r'], ggg.gam3.real.flatten())
    np.testing.assert_almost_equal(data['gam0i'], ggg.gam0.imag.flatten())
    np.testing.assert_almost_equal(data['gam1i'], ggg.gam1.imag.flatten())
    np.testing.assert_almost_equal(data['gam2i'], ggg.gam2.imag.flatten())
    np.testing.assert_almost_equal(data['gam3i'], ggg.gam3.imag.flatten())
    np.testing.assert_almost_equal(data['sigma_gam'], np.sqrt(ggg.vargam.flatten()))
    np.testing.assert_almost_equal(data['weight'], ggg.weight.flatten())
    np.testing.assert_almost_equal(data['ntri'], ggg.ntri.flatten())

    # Check the read function
    # Note: These don't need the flatten. The read function should reshape them to the right shape.
    ggg2 = treecorr.GGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                   min_u=min_u, max_u=max_u, min_v=min_v, max_v=max_v,
                                   nubins=nubins, nvbins=nvbins,
                                   sep_units='arcmin', verbose=1)
    ggg2.read(out_file_name1)
    np.testing.assert_almost_equal(ggg2.logr, ggg.logr)
    np.testing.assert_almost_equal(ggg2.u, ggg.u)
    np.testing.assert_almost_equal(ggg2.v, ggg.v)
    np.testing.assert_almost_equal(ggg2.meand1, ggg.meand1)
    np.testing.assert_almost_equal(ggg2.meanlogd1, ggg.meanlogd1)
    np.testing.assert_almost_equal(ggg2.meand2, ggg.meand2)
    np.testing.assert_almost_equal(ggg2.meanlogd2, ggg.meanlogd2)
    np.testing.assert_almost_equal(ggg2.meand3, ggg.meand3)
    np.testing.assert_almost_equal(ggg2.meanlogd3, ggg.meanlogd3)
    np.testing.assert_almost_equal(ggg2.meanu, ggg.meanu)
    np.testing.assert_almost_equal(ggg2.meanv, ggg.meanv)
    np.testing.assert_almost_equal(ggg2.gam0, ggg.gam0)
    np.testing.assert_almost_equal(ggg2.gam1, ggg.gam1)
    np.testing.assert_almost_equal(ggg2.gam2, ggg.gam2)
    np.testing.assert_almost_equal(ggg2.gam3, ggg.gam3)
    np.testing.assert_almost_equal(ggg2.vargam, ggg.vargam)
    np.testing.assert_almost_equal(ggg2.weight, ggg.weight)
    np.testing.assert_almost_equal(ggg2.ntri, ggg.ntri)
    assert ggg2.coords == ggg.coords
    assert ggg2.metric == ggg.metric
    assert ggg2.sep_units == ggg.sep_units
    assert ggg2.bin_type == ggg.bin_type

if __name__ == '__main__':
    test_ggg()

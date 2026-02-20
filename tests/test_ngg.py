# Copyright (c) 2003-2024 by Mike Jarvis
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

import numpy as np
import treecorr
import os
import coord
import time

from test_helper import do_pickle, assert_raises, timer, is_ccw, is_ccw_3d
from test_helper import get_from_wiki, CaptureLog
from test_patch3pt import generate_shear_field


@timer
def test_direct_logruv_cross():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 50
    s = 10.
    sig_gam = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,sig_gam, (ngal,) )
    g2_2 = rng.normal(0,sig_gam, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal)
    g1_3 = rng.normal(0,sig_gam, (ngal,) )
    g2_3 = rng.normal(0,sig_gam, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    ngg = treecorr.NGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gng = treecorr.GNGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    ggn = treecorr.GGNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_132 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_213 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_231 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_312 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_321 = np.zeros((nrbins, nubins, 2*nvbins))
    true_gam0_123 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_132 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_213 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_231 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_312 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_321 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_123 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_132 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_213 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_231 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_312 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_321 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_weight_123 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_132 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_213 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_231 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_312 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_321 = np.zeros((nrbins, nubins, 2*nvbins))
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    max_sep = min_sep * np.exp(nrbins*bin_size)
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                dij = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                dik = np.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                djk = np.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = is_ccw(x1[i],y1[i],x2[j],y2[j],x3[k],y3[k])

                # Rotate shear to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x3[k])/3.
                ceny = (y1[i] + y2[j] + y3[k])/3.

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                www = w1[i] * w2[j] * w3[k]
                gam0 = www * g2p * g3p
                gam1 = www * np.conjugate(g2p) * g3p

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_123
                        true_gam0 = true_gam0_123
                        true_gam1 = true_gam2_123
                        true_weight = true_weight_123
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_213
                        true_gam0 = true_gam0_213
                        true_gam1 = true_gam1_213
                        true_weight = true_weight_213
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_231
                        true_gam0 = true_gam0_231
                        true_gam1 = true_gam1_231
                        true_weight = true_weight_231
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_132
                        true_gam0 = true_gam0_132
                        true_gam1 = true_gam2_132
                        true_weight = true_weight_132
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_312
                        true_gam0 = true_gam0_312
                        true_gam1 = true_gam1_312
                        true_weight = true_weight_312
                        gam1 = np.conjugate(gam1)
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_321
                        true_gam0 = true_gam0_321
                        true_gam1 = true_gam1_321
                        true_weight = true_weight_321
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)

                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if r < min_sep or r >= max_sep: continue
                if u < min_u or u >= max_u: continue
                if v < min_v or v >= max_v: continue
                if not ccw:
                    v = -v
                kr = int(np.floor( (np.log(r)-log_min_sep) / bin_size ))
                ku = int(np.floor( (u-min_u) / ubin_size ))
                if v > 0:
                    kv = int(np.floor( (v-min_v) / vbin_size )) + nvbins
                else:
                    kv = int(np.floor( (v-(-max_v)) / vbin_size ))
                assert 0 <= kr < nrbins
                assert 0 <= ku < nubins
                assert 0 <= kv < 2*nvbins

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_gam0[kr,ku,kv] += gam0
                true_gam1[kr,ku,kv] += gam1

    true_ntri_sum1 = true_ntri_123 + true_ntri_132
    true_weight_sum1 = true_weight_123 + true_weight_132
    true_gam0_sum1 = true_gam0_123 + true_gam0_132
    true_gam1_sum1 = true_gam2_123 + true_gam2_132
    true_ntri_sum2 = true_ntri_213 + true_ntri_312
    true_weight_sum2 = true_weight_213 + true_weight_312
    true_gam0_sum2 = true_gam0_213 + true_gam0_312
    true_gam1_sum2 = true_gam1_213 + true_gam1_312
    true_ntri_sum3 = true_ntri_231 + true_ntri_321
    true_weight_sum3 = true_weight_231 + true_weight_321
    true_gam0_sum3 = true_gam0_231 + true_gam0_321
    true_gam1_sum3 = true_gam1_231 + true_gam1_321
    pos = true_weight_sum1 > 0
    true_gam0_sum1[pos] /= true_weight_sum1[pos]
    true_gam1_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_gam0_sum2[pos] /= true_weight_sum2[pos]
    true_gam1_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_gam0_sum3[pos] /= true_weight_sum3[pos]
    true_gam1_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    g0_list = [true_gam0_123, true_gam0_132, true_gam0_213, true_gam0_231,
               true_gam0_312, true_gam0_321]
    g1_list = [true_gam2_123, true_gam2_132, true_gam1_213, true_gam1_231,
               true_gam1_312, true_gam1_321]
    for w,g0,g1 in zip(w_list, g0_list, g1_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]

    ngg.process(cat1, cat2, cat3)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)

    ngg.process(cat1, cat3, cat2)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_132)
    np.testing.assert_allclose(ngg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_132, rtol=1.e-5)

    gng.process(cat2, cat1, cat3)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    gng.process(cat3, cat1, cat2)
    np.testing.assert_array_equal(gng.ntri, true_ntri_312)
    np.testing.assert_allclose(gng.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_312, rtol=1.e-5)

    ggn.process(cat2, cat3, cat1)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)
    ggn.process(cat3, cat2, cat1)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_321)
    np.testing.assert_allclose(ggn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 1
    ngg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)

    gng.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # Check bin_slop=0
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    ngg.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=True)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=True)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # And again with no top-level recursion
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    ngg.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=True)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=True)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=False)
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # With these, ordered=False is equivalent to the K vertex being fixed.
    ngg.process(cat1, cat2, cat3, ordered=1)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=2)
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=3)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # Compute Gamma using randoms
    # We save these values to compare to later ones with patches.
    xr = rng.uniform(-3*s,3*s, (2*ngal,) )
    yr = rng.normal(-3*s,3*s, (2*ngal,) )
    rcat = treecorr.Catalog(x=xr, y=yr)
    rgg = ngg.copy()
    rgg.process(rcat, cat2, cat3, ordered=False)
    gam0_ngg_r, gam2_ngg_r, _, _ = ngg.calculateGam(rgg=rgg)
    grg = gng.copy()
    grg.process(cat2, rcat, cat3, ordered=False)
    gam0_gng_r, gam1_gng_r, _, _ = gng.calculateGam(grg=grg)
    ggr = ggn.copy()
    ggr.process(cat2, cat3, rcat, ordered=False)
    gam0_ggn_r, gam1_ggn_r, _, _ = ggn.calculateGam(ggr=ggr)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        ngg.process(cat1, cat3=cat3)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    ngg.process(cat1p, cat2, cat3)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2p, cat1, cat3)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2p, cat3, cat1)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1, cat2p, cat3)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2, cat1p, cat3)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2, cat3p, cat1)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1, cat2, cat3p)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2, cat1, cat3p)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1p)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    # Now all three patched
    ngg.process(cat1p, cat2p, cat3p)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p)
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    # Unordered
    ngg.process(cat1p, cat2p, cat3p, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=False)
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=1)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=2)
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=3)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # Check using randoms with and without patches
    rcatp = treecorr.Catalog(x=xr, y=yr, patch_centers=cat1p.patch_centers)
    rgg.process(rcat, cat2p, cat3p, ordered=False)
    ngg.calculateGam(rgg=rgg)
    ngg.estimate_cov('jackknife')  # No accuracy check.  Just make sure it runs without error.
    np.testing.assert_allclose(ngg.gam0, gam0_ngg_r, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, gam2_ngg_r, rtol=1.e-5)
    rgg.process(rcatp, cat2p, cat3p, ordered=False)
    ngg.calculateGam(rgg=rgg)
    ngg.estimate_cov('jackknife')
    np.testing.assert_allclose(ngg.gam0, gam0_ngg_r, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, gam2_ngg_r, rtol=1.e-5)
    grg.process(cat2p, rcat, cat3p, ordered=False)
    gng.calculateGam(grg=grg)
    gng.estimate_cov('jackknife')
    np.testing.assert_allclose(gng.gam0, gam0_gng_r, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, gam1_gng_r, rtol=1.e-5)
    grg.process(cat2p, rcatp, cat3p, ordered=False)
    gng.calculateGam(grg=grg)
    gng.estimate_cov('jackknife')
    np.testing.assert_allclose(gng.gam0, gam0_gng_r, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, gam1_gng_r, rtol=1.e-5)
    ggr.process(cat2p, cat3p, rcat, ordered=False)
    ggn.calculateGam(ggr=ggr)
    ggn.estimate_cov('jackknife')
    np.testing.assert_allclose(ggn.gam0, gam0_ggn_r, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, gam1_ggn_r, rtol=1.e-5)
    ggr.process(cat2p, cat3p, rcatp, ordered=False)
    ggn.calculateGam(ggr=ggr)
    ggn.estimate_cov('jackknife')
    np.testing.assert_allclose(ggn.gam0, gam0_ggn_r, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, gam1_ggn_r, rtol=1.e-5)

    # Check when some patches have no objects
    rcatpx = treecorr.Catalog(x=xr, y=yr, npatch=20, rng=rng)
    cat1px = treecorr.Catalog(x=x1, y=y1, w=w1, patch_centers=rcatpx.patch_centers)
    cat2px = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2,
                              patch_centers=rcatpx.patch_centers)
    cat3px = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3,
                              patch_centers=rcatpx.patch_centers)
    ngg.process(cat1px, cat2px, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        ngg.calculateGam(rgg=rgg)
    rgg.process(rcatpx, cat2px, cat3px, ordered=False)
    ngg.calculateGam(rgg=rgg)
    np.testing.assert_allclose(ngg.gam0, gam0_ngg_r, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, gam2_ngg_r, rtol=1.e-5)
    gng.process(cat2px, cat1px, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        gng.calculateGam(grg=grg)
    grg.process(cat2px, rcatpx, cat3px, ordered=False)
    gng.calculateGam(grg=grg)
    np.testing.assert_allclose(gng.gam0, gam0_gng_r, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, gam1_gng_r, rtol=1.e-5)
    ggn.process(cat2px, cat3px, cat1px, ordered=False)
    with assert_raises(RuntimeError):
        ggn.calculateGam(ggr=ggr)
    ggr.process(cat2px, cat3px, rcatpx, ordered=False)
    ggn.calculateGam(ggr=ggr)
    np.testing.assert_allclose(ggn.gam0, gam0_ggn_r, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, gam1_ggn_r, rtol=1.e-5)

    # patch_method=local
    ngg.process(cat1p, cat2p, cat3p, patch_method='local')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, patch_method='local')
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, patch_method='local')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=1, patch_method='local')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=2, patch_method='local')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=3, patch_method='local')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    with assert_raises(ValueError):
        ngg.process(cat1p, cat2p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        gng.process(cat2p, cat1p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        ggn.process(cat2p, cat3p, cat1p, patch_method='nonlocal')


@timer
def test_direct_logruv_cross12():
    # Check the 1-2 cross correlation

    ngal = 50
    s = 10.
    sig_gam = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,sig_gam, (ngal,) )
    g2_2 = rng.normal(0,sig_gam, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    ngg = treecorr.NGGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gng = treecorr.GNGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    ggn = treecorr.GGNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    true_ntri_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_221 = np.zeros((nrbins, nubins, 2*nvbins))
    true_gam0_122 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_212 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam0_221 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam2_122 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_212 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_gam1_221 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_weight_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_weight_221 = np.zeros((nrbins, nubins, 2*nvbins))
    ubin_size = (max_u-min_u) / nubins
    vbin_size = (max_v-min_v) / nvbins
    max_sep = min_sep * np.exp(nrbins*bin_size)
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(j+1,ngal):
                dij = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                dik = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                djk = np.sqrt((x2[j]-x2[k])**2 + (y2[j]-y2[k])**2)
                if dij == 0.: continue
                if dik == 0.: continue
                if djk == 0.: continue
                ccw = is_ccw(x1[i],y1[i],x2[j],y2[j],x2[k],y2[k])

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x2[k])/3.
                ceny = (y1[i] + y2[j] + y2[k])/3.

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x2[k]-cenx) - 1j*(y2[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2

                www = w1[i] * w2[j] * w2[k]
                gam0 = www * g2p * g3p
                gam1 = www * np.conjugate(g2p) * g3p

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_122
                        true_gam0 = true_gam0_122
                        true_gam1 = true_gam2_122
                        true_weight = true_weight_122
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_212
                        true_gam0 = true_gam0_212
                        true_gam1 = true_gam1_212
                        true_weight = true_weight_212
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_221
                        true_gam0 = true_gam0_221
                        true_gam1 = true_gam1_221
                        true_weight = true_weight_221
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_122
                        true_gam0 = true_gam0_122
                        true_gam1 = true_gam2_122
                        true_weight = true_weight_122
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_212
                        true_gam0 = true_gam0_212
                        true_gam1 = true_gam1_212
                        true_weight = true_weight_212
                        gam1 = np.conjugate(gam1)
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_221
                        true_gam0 = true_gam0_221
                        true_gam1 = true_gam1_221
                        true_weight = true_weight_221
                        ccw = not ccw
                        gam1 = np.conjugate(gam1)

                r = d2
                u = d3/d2
                v = (d1-d2)/d3
                if r < min_sep or r >= max_sep: continue
                if u < min_u or u >= max_u: continue
                if v < min_v or v >= max_v: continue
                if not ccw:
                    v = -v
                kr = int(np.floor( (np.log(r)-log_min_sep) / bin_size ))
                ku = int(np.floor( (u-min_u) / ubin_size ))
                if v > 0:
                    kv = int(np.floor( (v-min_v) / vbin_size )) + nvbins
                else:
                    kv = int(np.floor( (v-(-max_v)) / vbin_size ))
                assert 0 <= kr < nrbins
                assert 0 <= ku < nubins
                assert 0 <= kv < 2*nvbins

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_gam0[kr,ku,kv] += gam0
                true_gam1[kr,ku,kv] += gam1

    pos = true_weight_221 > 0
    true_gam0_221[pos] /= true_weight_221[pos]
    true_gam1_221[pos] /= true_weight_221[pos]
    pos = true_weight_212 > 0
    true_gam0_212[pos] /= true_weight_212[pos]
    true_gam1_212[pos] /= true_weight_212[pos]
    pos = true_weight_122 > 0
    true_gam0_122[pos] /= true_weight_122[pos]
    true_gam2_122[pos] /= true_weight_122[pos]

    ngg.process(cat1, cat2, cat2)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2, cat2, cat1)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: GNG doesn't have a two-argument version.
    ngg.process(cat1, cat2)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2, cat1)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    with assert_raises(ValueError):
        gng.process(cat1, cat2)
    with assert_raises(ValueError):
        gng.process(cat2, cat1)
    with assert_raises(ValueError):
        ngg.process(cat1)
    with assert_raises(ValueError):
        ngg.process(cat2)
    with assert_raises(ValueError):
        gng.process(cat1)
    with assert_raises(ValueError):
        gng.process(cat2)
    with assert_raises(ValueError):
        ggn.process(cat1)
    with assert_raises(ValueError):
        ggn.process(cat2)

    # ordered=False doesn't do anything different, since there is no other valid order.
    ngg.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    # Repeat with binslop = 0
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')

    ngg.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    # And again with no top-level recursion
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    ngg.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)

    ngg.process(cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2p, cat1p, ordered=True)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2p, cat1p, ordered=False)
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2p, cat1p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)


@timer
def test_vargam_logruv():
    # Test that the shot noise estimate of vargam0 is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(1234)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 30
    nsource = 5000

    file_name = 'data/test_vargam_ngg_logruv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nggs = []
        all_gngs = []
        all_ggns = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise
            # from the random x,y positions.
            x1 = (rng.random_sample(nlens)-0.5) * L
            y1 = (rng.random_sample(nlens)-0.5) * L
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5
            r2 = (x2**2 + y2**2)/r0**2
            g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
            g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
            g1 += rng.normal(0, 0.3, size=nsource)
            g2 += rng.normal(0, 0.3, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1)
            gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
            ngg = treecorr.NGGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          nubins=2, nvbins=2, bin_type='LogRUV')
            gng = treecorr.GNGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          nubins=2, nvbins=2, bin_type='LogRUV')
            ggn = treecorr.GGNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          nubins=2, nvbins=2, bin_type='LogRUV')
            ngg.process(ncat, gcat)
            gng.process(gcat, ncat, gcat)
            ggn.process(gcat, ncat)
            all_nggs.append(ngg)
            all_gngs.append(gng)
            all_ggns.append(ggn)

        mean_ngg_gam0r = np.mean([ngg.gam0r for ngg in all_nggs], axis=0)
        mean_ngg_gam0i = np.mean([ngg.gam0i for ngg in all_nggs], axis=0)
        mean_ngg_gam2r = np.mean([ngg.gam2r for ngg in all_nggs], axis=0)
        mean_ngg_gam2i = np.mean([ngg.gam2i for ngg in all_nggs], axis=0)
        var_ngg_gam0r = np.var([ngg.gam0r for ngg in all_nggs], axis=0)
        var_ngg_gam0i = np.var([ngg.gam0i for ngg in all_nggs], axis=0)
        var_ngg_gam2r = np.var([ngg.gam2r for ngg in all_nggs], axis=0)
        var_ngg_gam2i = np.var([ngg.gam2i for ngg in all_nggs], axis=0)
        mean_ngg_vargam0 = np.mean([ngg.vargam0 for ngg in all_nggs], axis=0)
        mean_ngg_vargam2 = np.mean([ngg.vargam2 for ngg in all_nggs], axis=0)
        mean_gng_gam0r = np.mean([gng.gam0r for gng in all_gngs], axis=0)
        mean_gng_gam0i = np.mean([gng.gam0i for gng in all_gngs], axis=0)
        mean_gng_gam1r = np.mean([gng.gam1r for gng in all_gngs], axis=0)
        mean_gng_gam1i = np.mean([gng.gam1i for gng in all_gngs], axis=0)
        var_gng_gam0r = np.var([gng.gam0r for gng in all_gngs], axis=0)
        var_gng_gam0i = np.var([gng.gam0i for gng in all_gngs], axis=0)
        var_gng_gam1r = np.var([gng.gam1r for gng in all_gngs], axis=0)
        var_gng_gam1i = np.var([gng.gam1i for gng in all_gngs], axis=0)
        mean_gng_vargam0 = np.mean([gng.vargam0 for gng in all_gngs], axis=0)
        mean_gng_vargam1 = np.mean([gng.vargam1 for gng in all_gngs], axis=0)
        mean_ggn_gam0r = np.mean([ggn.gam0r for ggn in all_ggns], axis=0)
        mean_ggn_gam0i = np.mean([ggn.gam0i for ggn in all_ggns], axis=0)
        mean_ggn_gam1r = np.mean([ggn.gam1r for ggn in all_ggns], axis=0)
        mean_ggn_gam1i = np.mean([ggn.gam1i for ggn in all_ggns], axis=0)
        var_ggn_gam0r = np.var([ggn.gam0r for ggn in all_ggns], axis=0)
        var_ggn_gam0i = np.var([ggn.gam0i for ggn in all_ggns], axis=0)
        var_ggn_gam1r = np.var([ggn.gam1r for ggn in all_ggns], axis=0)
        var_ggn_gam1i = np.var([ggn.gam1i for ggn in all_ggns], axis=0)
        mean_ggn_vargam0 = np.mean([ggn.vargam0 for ggn in all_ggns], axis=0)
        mean_ggn_vargam1 = np.mean([ggn.vargam1 for ggn in all_ggns], axis=0)

        np.savez(file_name,
                 mean_ngg_gam0r=mean_ngg_gam0r,
                 mean_ngg_gam0i=mean_ngg_gam0i,
                 mean_ngg_gam2r=mean_ngg_gam2r,
                 mean_ngg_gam2i=mean_ngg_gam2i,
                 var_ngg_gam0r=var_ngg_gam0r,
                 var_ngg_gam0i=var_ngg_gam0i,
                 var_ngg_gam2r=var_ngg_gam2r,
                 var_ngg_gam2i=var_ngg_gam2i,
                 mean_ngg_vargam0=mean_ngg_vargam0,
                 mean_ngg_vargam2=mean_ngg_vargam2,
                 mean_gng_gam0r=mean_gng_gam0r,
                 mean_gng_gam0i=mean_gng_gam0i,
                 mean_gng_gam1r=mean_gng_gam1r,
                 mean_gng_gam1i=mean_gng_gam1i,
                 var_gng_gam0r=var_gng_gam0r,
                 var_gng_gam0i=var_gng_gam0i,
                 var_gng_gam1r=var_gng_gam1r,
                 var_gng_gam1i=var_gng_gam1i,
                 mean_gng_vargam0=mean_gng_vargam0,
                 mean_gng_vargam1=mean_gng_vargam1,
                 mean_ggn_gam0r=mean_ggn_gam0r,
                 mean_ggn_gam0i=mean_ggn_gam0i,
                 mean_ggn_gam1r=mean_ggn_gam1r,
                 mean_ggn_gam1i=mean_ggn_gam1i,
                 var_ggn_gam0r=var_ggn_gam0r,
                 var_ggn_gam0i=var_ggn_gam0i,
                 var_ggn_gam1r=var_ggn_gam1r,
                 var_ggn_gam1i=var_ggn_gam1i,
                 mean_ggn_vargam0=mean_ggn_vargam0,
                 mean_ggn_vargam1=mean_ggn_vargam1)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_ngg_gam0r = data['mean_ngg_gam0r']
    mean_ngg_gam0i = data['mean_ngg_gam0i']
    mean_ngg_gam2r = data['mean_ngg_gam2r']
    mean_ngg_gam2i = data['mean_ngg_gam2i']
    var_ngg_gam0r = data['var_ngg_gam0r']
    var_ngg_gam0i = data['var_ngg_gam0i']
    var_ngg_gam2r = data['var_ngg_gam2r']
    var_ngg_gam2i = data['var_ngg_gam2i']
    mean_ngg_vargam0 = data['mean_ngg_vargam0']
    mean_ngg_vargam2 = data['mean_ngg_vargam2']
    mean_gng_gam0r = data['mean_gng_gam0r']
    mean_gng_gam0i = data['mean_gng_gam0i']
    mean_gng_gam1r = data['mean_gng_gam1r']
    mean_gng_gam1i = data['mean_gng_gam1i']
    var_gng_gam0r = data['var_gng_gam0r']
    var_gng_gam0i = data['var_gng_gam0i']
    var_gng_gam1r = data['var_gng_gam1r']
    var_gng_gam1i = data['var_gng_gam1i']
    mean_gng_vargam0 = data['mean_gng_vargam0']
    mean_gng_vargam1 = data['mean_gng_vargam1']
    mean_ggn_gam0r = data['mean_ggn_gam0r']
    mean_ggn_gam0i = data['mean_ggn_gam0i']
    mean_ggn_gam1r = data['mean_ggn_gam1r']
    mean_ggn_gam1i = data['mean_ggn_gam1i']
    var_ggn_gam0r = data['var_ggn_gam0r']
    var_ggn_gam0i = data['var_ggn_gam0i']
    var_ggn_gam1r = data['var_ggn_gam1r']
    var_ggn_gam1i = data['var_ggn_gam1i']
    mean_ggn_vargam0 = data['mean_ggn_vargam0']
    mean_ggn_vargam1 = data['mean_ggn_vargam1']

    print('var_ngg_gam0r = ',var_ngg_gam0r)
    print('mean ngg_vargam0 = ',mean_ngg_vargam0)
    print('ratio = ',var_ngg_gam0r.ravel() / mean_ngg_vargam0.ravel())
    print('var_gng_gam0r = ',var_gng_gam0r)
    print('mean gng_vargam0 = ',mean_gng_vargam0)
    print('ratio = ',var_gng_gam0r.ravel() / mean_gng_vargam0.ravel())
    print('var_ggn_gam0r = ',var_ggn_gam0r)
    print('mean ggn_vargam0 = ',mean_ggn_vargam0)
    print('ratio = ',var_ggn_gam0r.ravel() / mean_ggn_vargam0.ravel())

    print('max relerr for ngg gam0r = ',
          np.max(np.abs((var_ngg_gam0r - mean_ngg_vargam0)/var_ngg_gam0r)))
    print('max relerr for ngg gam0i = ',
          np.max(np.abs((var_ngg_gam0i - mean_ngg_vargam0)/var_ngg_gam0i)))
    np.testing.assert_allclose(mean_ngg_vargam0, var_ngg_gam0r, rtol=0.03)
    np.testing.assert_allclose(mean_ngg_vargam0, var_ngg_gam0i, rtol=0.03)
    np.testing.assert_allclose(mean_ngg_vargam2, var_ngg_gam2r, rtol=0.03)
    np.testing.assert_allclose(mean_ngg_vargam2, var_ngg_gam2i, rtol=0.03)

    print('max relerr for gng gam0r = ',
          np.max(np.abs((var_gng_gam0r - mean_gng_vargam0)/var_gng_gam0r)))
    print('max relerr for gng gam0i = ',
          np.max(np.abs((var_gng_gam0i - mean_gng_vargam0)/var_gng_gam0i)))
    np.testing.assert_allclose(mean_gng_vargam0, var_gng_gam0r, rtol=0.03)
    np.testing.assert_allclose(mean_gng_vargam0, var_gng_gam0i, rtol=0.03)
    np.testing.assert_allclose(mean_gng_vargam1, var_gng_gam1r, rtol=0.03)
    np.testing.assert_allclose(mean_gng_vargam1, var_gng_gam1i, rtol=0.03)

    print('max relerr for ggn gam0r = ',
          np.max(np.abs((var_ggn_gam0r - mean_ggn_vargam0)/var_ggn_gam0r)))
    print('max relerr for ggn gam0i = ',
          np.max(np.abs((var_ggn_gam0i - mean_ggn_vargam0)/var_ggn_gam0i)))
    np.testing.assert_allclose(mean_ggn_vargam0, var_ggn_gam0r, rtol=0.06)
    np.testing.assert_allclose(mean_ggn_vargam0, var_ggn_gam0i, rtol=0.06)
    np.testing.assert_allclose(mean_ggn_vargam1, var_ggn_gam1r, rtol=0.06)
    np.testing.assert_allclose(mean_ggn_vargam1, var_ggn_gam1i, rtol=0.06)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5
    r2 = (x2**2 + y2**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
    g1 += rng.normal(0, 0.3, size=nsource)
    g2 += rng.normal(0, 0.3, size=nsource)

    ncat = treecorr.Catalog(x=x1, y=y1)
    gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
    ngg = treecorr.NGGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  nubins=2, nvbins=2, bin_type='LogRUV')
    gng = treecorr.GNGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  nubins=2, nvbins=2, bin_type='LogRUV')
    ggn = treecorr.GGNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  nubins=2, nvbins=2, bin_type='LogRUV')

    # Before running process, vargam0 and cov are allowed, but all 0.
    np.testing.assert_array_equal(ngg.cov, 0)
    np.testing.assert_array_equal(ngg.vargam0, 0)
    ngg.clear()
    np.testing.assert_array_equal(ngg.vargam2, 0)
    np.testing.assert_array_equal(gng.cov, 0)
    np.testing.assert_array_equal(gng.vargam0, 0)
    gng.clear()
    np.testing.assert_array_equal(gng.vargam1, 0)
    np.testing.assert_array_equal(ggn.cov, 0)
    np.testing.assert_array_equal(ggn.vargam0, 0)
    ggn.clear()
    np.testing.assert_array_equal(ggn.vargam1, 0)

    ngg.process(ncat, gcat)
    print('NGG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ngg.vargam0 - var_ngg_gam0r)/var_ngg_gam0r)))
    print('ratio = ',ngg.vargam0 / var_ngg_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ngg.vargam0 - var_ngg_gam0i)/var_ngg_gam0i)))
    print('ratio = ',ngg.vargam0 / var_ngg_gam0i)
    print('var_num = ',ngg._var_num)
    print('ntri = ',ngg.ntri)
    np.testing.assert_allclose(ngg.vargam0, var_ngg_gam0r, rtol=0.3)
    np.testing.assert_allclose(ngg.vargam0, var_ngg_gam0i, rtol=0.3)
    np.testing.assert_allclose(ngg.vargam2, var_ngg_gam2r, rtol=0.3)
    np.testing.assert_allclose(ngg.vargam2, var_ngg_gam2i, rtol=0.3)
    n = ngg.vargam0.size
    np.testing.assert_allclose(ngg.cov.diagonal()[0:n], ngg.vargam0.ravel())
    np.testing.assert_allclose(ngg.cov.diagonal()[n:2*n], ngg.vargam2.ravel())

    gng.process(gcat, ncat, gcat)
    print('GNG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((gng.vargam0 - var_gng_gam0r)/var_gng_gam0r)))
    print('ratio = ',gng.vargam0 / var_gng_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((gng.vargam0 - var_gng_gam0i)/var_gng_gam0i)))
    print('ratio = ',gng.vargam0 / var_gng_gam0i)
    np.testing.assert_allclose(gng.vargam0, var_gng_gam0r, rtol=0.3)
    np.testing.assert_allclose(gng.vargam0, var_gng_gam0i, rtol=0.3)
    np.testing.assert_allclose(gng.vargam1, var_gng_gam1r, rtol=0.3)
    np.testing.assert_allclose(gng.vargam1, var_gng_gam1i, rtol=0.3)
    np.testing.assert_allclose(gng.cov.diagonal()[0:n], gng.vargam0.ravel())
    np.testing.assert_allclose(gng.cov.diagonal()[n:2*n], gng.vargam1.ravel())

    ggn.process(gcat, ncat)
    print('GGN single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ggn.vargam0 - var_ggn_gam0r)/var_ggn_gam0r)))
    print('ratio = ',ggn.vargam0 / var_ggn_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ggn.vargam0 - var_ggn_gam0i)/var_ggn_gam0i)))
    print('ratio = ',ggn.vargam0 / var_ggn_gam0i)
    np.testing.assert_allclose(ggn.vargam0, var_ggn_gam0r, rtol=0.3)
    np.testing.assert_allclose(ggn.vargam0, var_ggn_gam0i, rtol=0.3)
    np.testing.assert_allclose(ggn.vargam1, var_ggn_gam1r, rtol=0.3)
    np.testing.assert_allclose(ggn.vargam1, var_ggn_gam1i, rtol=0.3)
    np.testing.assert_allclose(ggn.cov.diagonal()[0:n], ggn.vargam0.ravel())
    np.testing.assert_allclose(ggn.cov.diagonal()[n:2*n], ggn.vargam1.ravel())

    # Check valid aliases
    # NGG: gam0 = gam1, gam2 = conj(gam3)
    # GNG: gam0 = gam2, gam1 = conj(gam3)
    # GGN: gam0 = gam3, gam1 = conj(gam2)
    np.testing.assert_array_equal(ngg.gam0r, np.real(ngg.gam0))
    np.testing.assert_array_equal(ngg.gam0i, np.imag(ngg.gam0))
    np.testing.assert_array_equal(ngg.gam1r, np.real(ngg.gam1))
    np.testing.assert_array_equal(ngg.gam1i, np.imag(ngg.gam1))
    np.testing.assert_array_equal(ngg.gam2r, np.real(ngg.gam2))
    np.testing.assert_array_equal(ngg.gam2i, np.imag(ngg.gam2))
    np.testing.assert_array_equal(ngg.gam3r, np.real(ngg.gam3))
    np.testing.assert_array_equal(ngg.gam3i, np.imag(ngg.gam3))
    np.testing.assert_array_equal(ngg.gam0, ngg.gam1)
    np.testing.assert_array_equal(ngg.gam0r, ngg.gam1r)
    np.testing.assert_array_equal(ngg.gam0i, ngg.gam1i)
    np.testing.assert_array_equal(ngg.gam2, np.conjugate(ngg.gam3))
    np.testing.assert_array_equal(ngg.gam2r, ngg.gam3r)
    np.testing.assert_array_equal(ngg.gam2i, -ngg.gam3i)
    np.testing.assert_array_equal(ngg.vargam0, ngg.vargam1)
    np.testing.assert_array_equal(ngg.vargam2, ngg.vargam3)

    np.testing.assert_array_equal(gng.gam0r, np.real(gng.gam0))
    np.testing.assert_array_equal(gng.gam0i, np.imag(gng.gam0))
    np.testing.assert_array_equal(gng.gam1r, np.real(gng.gam1))
    np.testing.assert_array_equal(gng.gam1i, np.imag(gng.gam1))
    np.testing.assert_array_equal(gng.gam2r, np.real(gng.gam2))
    np.testing.assert_array_equal(gng.gam2i, np.imag(gng.gam2))
    np.testing.assert_array_equal(gng.gam3r, np.real(gng.gam3))
    np.testing.assert_array_equal(gng.gam3i, np.imag(gng.gam3))
    np.testing.assert_array_equal(gng.gam0, gng.gam2)
    np.testing.assert_array_equal(gng.gam0r, gng.gam2r)
    np.testing.assert_array_equal(gng.gam0i, gng.gam2i)
    np.testing.assert_array_equal(gng.gam1, np.conjugate(gng.gam3))
    np.testing.assert_array_equal(gng.gam1r, gng.gam3r)
    np.testing.assert_array_equal(gng.gam1i, -gng.gam3i)
    np.testing.assert_array_equal(gng.vargam0, gng.vargam2)
    np.testing.assert_array_equal(gng.vargam1, gng.vargam3)

    np.testing.assert_array_equal(ggn.gam0r, np.real(ggn.gam0))
    np.testing.assert_array_equal(ggn.gam0i, np.imag(ggn.gam0))
    np.testing.assert_array_equal(ggn.gam1r, np.real(ggn.gam1))
    np.testing.assert_array_equal(ggn.gam1i, np.imag(ggn.gam1))
    np.testing.assert_array_equal(ggn.gam2r, np.real(ggn.gam2))
    np.testing.assert_array_equal(ggn.gam2i, np.imag(ggn.gam2))
    np.testing.assert_array_equal(ggn.gam3r, np.real(ggn.gam3))
    np.testing.assert_array_equal(ggn.gam3i, np.imag(ggn.gam3))
    np.testing.assert_array_equal(ggn.gam0, ggn.gam3)
    np.testing.assert_array_equal(ggn.gam0r, ggn.gam3r)
    np.testing.assert_array_equal(ggn.gam0i, ggn.gam3i)
    np.testing.assert_array_equal(ggn.gam1, np.conjugate(ggn.gam2))
    np.testing.assert_array_equal(ggn.gam1r, ggn.gam2r)
    np.testing.assert_array_equal(ggn.gam1i, -ggn.gam2i)
    np.testing.assert_array_equal(ggn.vargam0, ggn.vargam3)
    np.testing.assert_array_equal(ggn.vargam1, ggn.vargam2)


@timer
def test_direct_logsas_cross():
    # If the catalogs are small enough, we can do a direct calculation to see if comes out right.
    # This should exactly match the treecorr result if brute=True.

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal).astype(np.float32)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal).astype(np.float32)
    g1_2 = rng.normal(0,0.2, (ngal,) )
    g2_2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.random_sample(ngal).astype(np.float32)
    g1_3 = rng.normal(0,0.2, (ngal,) )
    g2_3 = rng.normal(0,0.2, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 3

    # In this test set, we use the slow triangle algorithm.
    # We'll test the multipole algorithm below.
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_132 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_213 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_231 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_312 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_321 = np.zeros((nbins, nbins, nphi_bins))
    true_gam0_123 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_132 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_213 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_231 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_312 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam0_321 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam2_123 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam2_132 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_213 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_231 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_312 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_gam1_321 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_weight_123 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_132 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_213 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_231 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_312 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_321 = np.zeros((nbins, nbins, nphi_bins))
    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    bin_size = (log_max_sep - log_min_sep) / nbins
    phi_bin_size = np.pi / nphi_bins
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                d1 = np.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                d2 = np.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x3[k])/3.
                ceny = (y1[i] + y2[j] + y3[k])/3.

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w3[k]
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                gam0 = www * g2p * g3p
                gam1 = www * np.conjugate(g2p) * g3p

                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    # 123
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_123[kr2,kr3,kphi] += 1
                        true_weight_123[kr2,kr3,kphi] += www
                        true_gam0_123[kr2,kr3,kphi] += gam0
                        true_gam2_123[kr2,kr3,kphi] += gam1

                    # 132
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_132[kr3,kr2,kphi] += 1
                        true_weight_132[kr3,kr2,kphi] += www
                        true_gam0_132[kr3,kr2,kphi] += gam0
                        true_gam2_132[kr3,kr2,kphi] += np.conjugate(gam1)

                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr3 < nbins
                    # 231
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_231[kr3,kr1,kphi] += 1
                        true_weight_231[kr3,kr1,kphi] += www
                        true_gam0_231[kr3,kr1,kphi] += gam0
                        true_gam1_231[kr3,kr1,kphi] += gam1

                    # 213
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_213[kr1,kr3,kphi] += 1
                        true_weight_213[kr1,kr3,kphi] += www
                        true_gam0_213[kr1,kr3,kphi] += gam0
                        true_gam1_213[kr1,kr3,kphi] += gam1

                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    # 312
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_312[kr1,kr2,kphi] += 1
                        true_weight_312[kr1,kr2,kphi] += www
                        true_gam0_312[kr1,kr2,kphi] += gam0
                        true_gam1_312[kr1,kr2,kphi] += np.conjugate(gam1)

                    # 321
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_321[kr2,kr1,kphi] += 1
                        true_weight_321[kr2,kr1,kphi] += www
                        true_gam0_321[kr2,kr1,kphi] += gam0
                        true_gam1_321[kr2,kr1,kphi] += np.conjugate(gam1)

    true_ntri_sum1 = true_ntri_123 + true_ntri_132
    true_weight_sum1 = true_weight_123 + true_weight_132
    true_gam0_sum1 = true_gam0_123 + true_gam0_132
    true_gam1_sum1 = true_gam2_123 + true_gam2_132
    true_ntri_sum2 = true_ntri_213 + true_ntri_312
    true_weight_sum2 = true_weight_213 + true_weight_312
    true_gam0_sum2 = true_gam0_213 + true_gam0_312
    true_gam1_sum2 = true_gam1_213 + true_gam1_312
    true_ntri_sum3 = true_ntri_231 + true_ntri_321
    true_weight_sum3 = true_weight_231 + true_weight_321
    true_gam0_sum3 = true_gam0_231 + true_gam0_321
    true_gam1_sum3 = true_gam1_231 + true_gam1_321
    pos = true_weight_sum1 > 0
    true_gam0_sum1[pos] /= true_weight_sum1[pos]
    true_gam1_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_gam0_sum2[pos] /= true_weight_sum2[pos]
    true_gam1_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_gam0_sum3[pos] /= true_weight_sum3[pos]
    true_gam1_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    g0_list = [true_gam0_123, true_gam0_132, true_gam0_213, true_gam0_231,
               true_gam0_312, true_gam0_321]
    g1_list = [true_gam2_123, true_gam2_132, true_gam1_213, true_gam1_231,
               true_gam1_312, true_gam1_321]
    for w,g0,g1 in zip(w_list, g0_list, g1_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]

    ngg.process(cat1, cat2, cat3, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)

    ngg.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_132)
    np.testing.assert_allclose(ngg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_132, rtol=1.e-5)

    gng.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)

    gng.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_312)
    np.testing.assert_allclose(gng.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_312, rtol=1.e-5)

    ggn.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ggn.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_321)
    np.testing.assert_allclose(ggn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 1
    ngg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-4)

    gng.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)

    ggn.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # Check binslop = 0
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    ngg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # And again with no top-level recursion
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    ngg.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2, cat1, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    ngg.process(cat1, cat2, cat3, ordered=1, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-4)
    gng.process(cat2, cat1, cat3, ordered=2, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2, cat3, cat1, ordered=3, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        ngg.process(cat1, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        gng.process(cat1, cat3=cat1, algo='triangle')
    with assert_raises(ValueError):
        ggn.process(cat3, cat3=cat1, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    ngg.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=1, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=2, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=3, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_123)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_213)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_231)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    ngg.process(cat1p, cat2p, cat3p, ordered=1, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_sum1)
    np.testing.assert_allclose(ngg.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_sum1, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam1_sum1, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat3p, ordered=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_sum2)
    np.testing.assert_allclose(gng.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_sum2, rtol=1.e-5)
    ggn.process(cat2p, cat3p, cat1p, ordered=3, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_sum3)
    np.testing.assert_allclose(ggn.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_sum3, rtol=1.e-5)

    with assert_raises(ValueError):
        ngg.process(cat1p, cat2p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        gng.process(cat2p, cat1p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        ggn.process(cat2p, cat3p, cat1p, patch_method='nonlocal', algo='triangle')


@timer
def test_direct_logsas_cross12():
    # Check the 1-2 cross correlation

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    g1_2 = rng.normal(0,0.2, (ngal,) )
    g2_2 = rng.normal(0,0.2, (ngal,) )
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 7

    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_122 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_212 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_221 = np.zeros((nbins, nbins, nphi_bins))
    true_gam0_122 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam0_212 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam0_221 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam2_122 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam1_212 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_gam1_221 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_weight_122 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_212 = np.zeros((nbins, nbins, nphi_bins))
    true_weight_221 = np.zeros((nbins, nbins, nphi_bins))
    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    bin_size = (log_max_sep - log_min_sep) / nbins
    phi_bin_size = np.pi / nphi_bins
    log_min_sep = np.log(min_sep)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                if j == k: continue
                d1 = np.sqrt((x2[j]-x2[k])**2 + (y2[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                # Rotate shears to coordinates where line connecting to center is horizontal.
                cenx = (x1[i] + x2[j] + x2[k])/3.
                ceny = (y1[i] + y2[j] + y2[k])/3.

                expmialpha2 = (x2[j]-cenx) - 1j*(y2[j]-ceny)
                expmialpha2 /= abs(expmialpha2)
                expmialpha3 = (x2[k]-cenx) - 1j*(y2[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w2[k]
                g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                gam0 = www * g2p * g3p
                gam1 = www * np.conjugate(g2p) * g3p

                # 123
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                        gam1 = np.conjugate(gam1)
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_122[kr2,kr3,kphi] += 1
                        true_weight_122[kr2,kr3,kphi] += www
                        true_gam0_122[kr2,kr3,kphi] += gam0
                        true_gam2_122[kr2,kr3,kphi] += gam1

                # 231
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_221[kr3,kr1,kphi] += 1
                        true_weight_221[kr3,kr1,kphi] += www
                        true_gam0_221[kr3,kr1,kphi] += gam0
                        true_gam1_221[kr3,kr1,kphi] += gam1

                # 312
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    gam1 = np.conjugate(gam1)
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_212[kr1,kr2,kphi] += 1
                        true_weight_212[kr1,kr2,kphi] += www
                        true_gam0_212[kr1,kr2,kphi] += gam0
                        true_gam1_212[kr1,kr2,kphi] += gam1

    w_list = [true_weight_122, true_weight_212, true_weight_221]
    g0_list = [true_gam0_122, true_gam0_212, true_gam0_221]
    g1_list = [true_gam2_122, true_gam1_212, true_gam1_221]
    for w,g0,g1 in zip(w_list, g0_list, g1_list):
        pos = w > 0
        g0[pos] /= w[pos]
        g1[pos] /= w[pos]

    ngg.process(cat1, cat2, cat2, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-4, atol=1.e-6)
    ngg.process(cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-4, atol=1.e-6)

    gng.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-4, atol=1.e-6)

    ggn.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4, atol=1.e-6)
    ggn.process(cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4, atol=1.e-6)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4, atol=1.e-6)

    with assert_raises(ValueError):
        gng.process(cat1, cat2)
    with assert_raises(ValueError):
        gng.process(cat2, cat1)
    with assert_raises(ValueError):
        ngg.process(cat1)
    with assert_raises(ValueError):
        ngg.process(cat2)
    with assert_raises(ValueError):
        gng.process(cat1)
    with assert_raises(ValueError):
        gng.process(cat2)
    with assert_raises(ValueError):
        ggn.process(cat1)
    with assert_raises(ValueError):
        ggn.process(cat2)

    # With ordered=False, doesn't do anything different, since there is no other valid order.
    ngg.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)

    ggn.process(cat2, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)

    ngg.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    gng.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gng.ntri, true_ntri_212)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-5)
    ggn.process(cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)

    ngg.process(cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngg.ntri, true_ntri_122)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-5)
    ggn.process(cat2p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ggn.ntri, true_ntri_221)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-5)


@timer
def test_direct_logmultipole_cross():
    # Check the cross correlation with LogMultipole
    if __name__ == '__main__':
        ngal = 100
    else:
        ngal = 30

    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.uniform(1,3, (ngal,))
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(1,3, (ngal,))
    g1_2 = rng.normal(0,0.2, (ngal,))
    g2_2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)
    x3 = rng.normal(0,s, (ngal,) )
    y3 = rng.normal(0,s, (ngal,) )
    w3 = rng.uniform(1,3, (ngal,))
    g1_3 = rng.normal(0,0.2, (ngal,))
    g2_3 = rng.normal(0,0.2, (ngal,))
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3)

    min_sep = 1.
    max_sep = 30.
    nbins = 5
    max_n = 10

    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_gam0_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam2_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_123 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam2_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_132 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_213 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_231 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_312 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_321 = np.zeros((nbins, nbins, 2*max_n+1))
    bin_size = (log_max_sep - log_min_sep) / nbins
    n1d = np.arange(-max_n, max_n+1)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                d1 = np.sqrt((x2[j]-x3[k])**2 + (y2[j]-y3[k])**2)
                d2 = np.sqrt((x1[i]-x3[k])**2 + (y1[i]-y3[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                www = w1[i] * w2[j] * w3[k]

                # 123, 132
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g2 is projected to the line from c1 to c2
                    # g3 is projected to the line from c1 to c3.
                    expmialpha2 = (x1[i]-x2[j]) - 1j*(y1[i]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = (x1[i]-x3[k]) - 1j*(y1[i]-y3[k])
                    expmialpha3 /= abs(expmialpha3)
                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                    gam0 = www * g2p * g3p
                    gam1 = www * np.conjugate(g2p) * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi

                    true_gam0_123[kr2,kr3,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam2_123[kr2,kr3,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_weight_123[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_123[kr2,kr3,:] += 1
                    true_gam0_132[kr3,kr2,:] += gam0 * np.exp(1j * n1d * phi)
                    true_gam2_132[kr3,kr2,:] += np.conjugate(gam1) * np.exp(1j * n1d * phi)
                    true_weight_132[kr3,kr2,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_132[kr3,kr2,:] += 1

                # 231, 213
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c2 to c3.
                    # g2 is projected to the average of the two lines.
                    expmialpha3 = (x2[j]-x3[k]) - 1j*(y2[j]-y3[k])
                    expmialpha3 /= abs(expmialpha3)
                    expmialpha1 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = expmialpha1 + expmialpha3
                    expmialpha2 /= abs(expmialpha2)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                    gam0 = www * g2p * g3p
                    gam1 = www * np.conjugate(g2p) * g3p

                    assert 0 <= kr3 < nbins
                    assert 0 <= kr1 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi

                    true_gam0_231[kr3,kr1,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_231[kr3,kr1,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_weight_231[kr3,kr1,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_231[kr3,kr1,:] += 1
                    true_gam0_213[kr1,kr3,:] += gam0 * np.exp(1j * n1d * phi)
                    true_gam1_213[kr1,kr3,:] += gam1 * np.exp(1j * n1d * phi)
                    true_weight_213[kr1,kr3,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_213[kr1,kr3,:] += 1

                # 312, 321
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g2 is projected to the line from c3 to c2.
                    # g3 is projected to the average of the two lines.
                    expmialpha1 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x3[k]-x2[j]) - 1j*(y3[k]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                    gam0 = www * g3p * g2p
                    gam1 = www * np.conjugate(g3p) * g2p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi

                    true_gam0_312[kr1,kr2,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_312[kr1,kr2,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_weight_312[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_312[kr1,kr2,:] += 1
                    true_gam0_321[kr2,kr1,:] += gam0 * np.exp(1j * n1d * phi)
                    true_gam1_321[kr2,kr1,:] += gam1 * np.exp(1j * n1d * phi)
                    true_weight_321[kr2,kr1,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_321[kr2,kr1,:] += 1

    ngg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(ngg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-4)
    ngg.process(cat1, cat3, cat2)
    np.testing.assert_allclose(ngg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_132, rtol=1.e-4)
    gng.process(cat2, cat1, cat3)
    np.testing.assert_allclose(gng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-4)
    gng.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gng.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_312, rtol=1.e-4)
    ggn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-4)
    ggn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_321, rtol=1.e-4)

    # Repeat with binslop = 0
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ngg.process(cat1, cat2, cat3)
    np.testing.assert_allclose(ngg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-4)
    ngg.process(cat1, cat3, cat2)
    np.testing.assert_allclose(ngg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_132, rtol=1.e-4)

    gng.process(cat2, cat1, cat3)
    np.testing.assert_allclose(gng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-4)
    gng.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gng.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_312, rtol=1.e-4)

    ggn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-4)
    ggn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_321, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    ngg.process(cat1p, cat2, cat3)
    np.testing.assert_allclose(ngg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-4)
    ngg.process(cat1p, cat3, cat2)
    np.testing.assert_allclose(ngg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_132, rtol=1.e-4)

    gng.process(cat2p, cat1, cat3)
    np.testing.assert_allclose(gng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-4)
    gng.process(cat3p, cat1, cat2)
    np.testing.assert_allclose(gng.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_312, rtol=1.e-4)

    ggn.process(cat2p, cat3, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-4)
    ggn.process(cat3p, cat2, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_321, rtol=1.e-4)

    # Now use all three patched
    ngg.process(cat1p, cat2p, cat3p)
    np.testing.assert_allclose(ngg.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_123, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_123, rtol=1.e-4)
    ngg.process(cat1p, cat3p, cat2p)
    np.testing.assert_allclose(ngg.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_132, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_132, rtol=1.e-4)

    gng.process(cat2p, cat1p, cat3p)
    np.testing.assert_allclose(gng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_213, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_213, rtol=1.e-4)
    gng.process(cat3p, cat1p, cat2p)
    np.testing.assert_allclose(gng.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_312, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_312, rtol=1.e-4)

    ggn.process(cat2p, cat3p, cat1p)
    np.testing.assert_allclose(ggn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_231, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_231, rtol=1.e-4)
    ggn.process(cat3p, cat2p, cat1p)
    np.testing.assert_allclose(ggn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_321, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_321, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        ngg.process(cat1p, cat2p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        gng.process(cat2p, cat1p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        ggn.process(cat2p, cat3p, cat1p, patch_method='global')

    # Test I/O
    for name, corr in zip(['ngg', 'gng', 'ggn'], [ngg, gng, ggn]):
        ascii_name = 'output/'+name+'_ascii_logmultipole.txt'
        corr.write(ascii_name, precision=16)
        corr2 = treecorr.Corr3.from_file(ascii_name)
        np.testing.assert_allclose(corr2.ntri, corr.ntri)
        np.testing.assert_allclose(corr2.weight, corr.weight)
        np.testing.assert_allclose(corr2.gam0, corr.gam0)
        np.testing.assert_allclose(corr2.gam1, corr.gam1)
        np.testing.assert_allclose(corr2.gam2, corr.gam2)
        np.testing.assert_allclose(corr2.gam3, corr.gam3)
        np.testing.assert_allclose(corr2.meand1, corr.meand1)
        np.testing.assert_allclose(corr2.meand2, corr.meand2)
        np.testing.assert_allclose(corr2.meand3, corr.meand3)
        np.testing.assert_allclose(corr2.meanlogd1, corr.meanlogd1)
        np.testing.assert_allclose(corr2.meanlogd2, corr.meanlogd2)
        np.testing.assert_allclose(corr2.meanlogd3, corr.meanlogd3)


@timer
def test_direct_logmultipole_cross12():
    # Check the cross correlation with LogMultipole
    if __name__ == '__main__':
        ngal = 100
    else:
        ngal = 30

    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.uniform(1,3, (ngal,))
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.uniform(1,3, (ngal,))
    g1_2 = rng.normal(0,0.2, (ngal,))
    g2_2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2)

    min_sep = 10.
    max_sep = 30.
    nbins = 5
    max_n = 8

    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_gam0_122 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam2_122 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_122 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_122 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_212 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_212 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_212 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_212 = np.zeros((nbins, nbins, 2*max_n+1))
    true_gam0_221 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_gam1_221 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_221 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_221 = np.zeros((nbins, nbins, 2*max_n+1))
    bin_size = (log_max_sep - log_min_sep) / nbins
    n1d = np.arange(-max_n, max_n+1)
    for i in range(ngal):
        for j in range(ngal):
            for k in range(ngal):
                if k == j: continue
                d1 = np.sqrt((x2[j]-x2[k])**2 + (y2[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                www = w1[i] * w2[j] * w2[k]

                # 122
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # Rotate shears according to the x projection.  See Porth et al, Figure 1.
                    # g2 is projected to the line from c1 to c2.
                    # g3 is projected to the line from c1 to c3.
                    expmialpha2 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha3 /= abs(expmialpha3)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    gam0 = www * g2p * g3p
                    gam2 = www * np.conjugate(g2p) * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = -phi
                    true_weight_122[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_gam0_122[kr2,kr3,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam2_122[kr2,kr3,:] += gam2 * np.exp(-1j * n1d * phi)
                    true_ntri_122[kr2,kr3,:] += 1.

                # 221
                if d3 >= min_sep and d3 < max_sep and d1 >= min_sep and d1 < max_sep:
                    # g3 is projected to the line from c2 to c3.
                    # g2 is projected to the average of the two lines.
                    expmialpha3 = (x2[j]-x2[k]) - 1j*(y2[j]-y2[k])
                    expmialpha3 /= abs(expmialpha3)
                    expmialpha1 = (x2[j]-x1[i]) - 1j*(y2[j]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = expmialpha1 + expmialpha3
                    expmialpha2 /= abs(expmialpha2)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    gam0 = www * g2p * g3p
                    gam1 = www * np.conjugate(g2p) * g3p

                    assert 0 <= kr3 < nbins
                    assert 0 <= kr1 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = -phi
                    true_weight_221[kr3,kr1,:] += www * np.exp(-1j * n1d * phi)
                    true_gam0_221[kr3,kr1,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_221[kr3,kr1,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_ntri_221[kr3,kr1,:] += 1.

                # 212
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g2 is projected to the line from c3 to c2.
                    # g3 is projected to the average of the two lines.
                    expmialpha1 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x2[k]-x2[j]) - 1j*(y2[k]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)

                    g2p = (g1_2[j] + 1j*g2_2[j]) * expmialpha2**2
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    gam0 = www * g3p * g2p
                    gam1 = www * np.conjugate(g3p) * g2p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = -phi
                    true_weight_212[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_gam0_212[kr1,kr2,:] += gam0 * np.exp(-1j * n1d * phi)
                    true_gam1_212[kr1,kr2,:] += gam1 * np.exp(-1j * n1d * phi)
                    true_ntri_212[kr1,kr2,:] += 1.

    gng.process(cat2, cat1, cat2)
    np.testing.assert_allclose(gng.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-4)
    ggn.process(cat2, cat2, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4)
    # 3 arg version doesn't work for ngg because the ngg processing doesn't know cat2 and cat3
    # are actually the same, so it doesn't subtract off the duplicates.

    # 2 arg version
    ngg.process(cat1, cat2)
    np.testing.assert_allclose(ngg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-4)
    ggn.process(cat2, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4)

    # Repeat with binslop = 0
    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ngg.process(cat1, cat2)
    np.testing.assert_allclose(ngg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-4)

    gng.process(cat2, cat1, cat2)
    np.testing.assert_allclose(gng.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-4)

    ggn.process(cat2, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    ngg.process(cat1p, cat2)
    np.testing.assert_allclose(ngg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-4)

    gng.process(cat2p, cat1, cat2)
    np.testing.assert_allclose(gng.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-4)

    ggn.process(cat2p, cat1)
    np.testing.assert_allclose(ggn.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4)

    # Now use both patched
    ngg.process(cat1p, cat2p)
    np.testing.assert_allclose(ngg.ntri, true_ntri_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(ngg.gam0, true_gam0_122, rtol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2_122, rtol=1.e-4)

    gng.process(cat2p, cat1p, cat2p)
    np.testing.assert_allclose(gng.ntri, true_ntri_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(gng.gam0, true_gam0_212, rtol=1.e-4)
    np.testing.assert_allclose(gng.gam1, true_gam1_212, rtol=1.e-4)

    ggn.process(cat2p, cat1p)
    np.testing.assert_allclose(ggn.ntri, true_ntri_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(ggn.gam0, true_gam0_221, rtol=1.e-4)
    np.testing.assert_allclose(ggn.gam1, true_gam1_221, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        ngg.process(cat1p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        gng.process(cat1p, cat2p, cat1p, patch_method='global')
    with assert_raises(ValueError):
        ggn.process(cat2p, cat1p, patch_method='global')


@timer
def test_ngg_logsas():
    # Use gaussian tangential shear around "lens" centers
    #
    # gamma(r) = -gamma0 exp(-r^2/2r0^2) (x+iy)^2 / r0^2

    gamma0 = 0.05
    r0 = 10.
    if __name__ == '__main__':
        nlens = 300
        nsource = 1000000
        L = 100. * r0
        tol_factor = 1
    else:
        nlens = 20
        nsource = 50000
        L = 20. * r0
        tol_factor = 4

    rng = np.random.RandomState(8675309)
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L

    dx = x2[:,None]-x1[None,:]
    dy = y2[:,None]-y1[None,:]
    r = np.sqrt(dx**2 + dy**2) / r0

    g1 = -gamma0 * np.exp(-r**2) * (dx**2-dy**2)/r0**2
    g2 = -gamma0 * np.exp(-r**2) * (2*dx*dy)/r0**2
    g1 = np.sum(g1, axis=1)
    g2 = np.sum(g2, axis=1)
    print('mean gamma = ',np.mean(g1),np.mean(g2))

    ncat = treecorr.Catalog(x=x1, y=y1)
    gcat = treecorr.Catalog(x=x2, y=y2, g1=g1, g2=g2)

    nrand = 10*nlens
    x3 = (rng.random_sample(nrand)-0.5) * L
    y3 = (rng.random_sample(nrand)-0.5) * L
    rcat = treecorr.Catalog(x=x3, y=y3)

    min_sep = 4.
    max_sep = 9.
    nbins = 15
    min_phi = 0.5
    max_phi = 2.5
    nphi_bins = 20

    ngg = treecorr.NGGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  bin_type='LogSAS')
    gng = treecorr.GNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  bin_type='LogSAS')
    ggn = treecorr.GGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  bin_type='LogSAS')

    for name, corr in zip(['ngg', 'gng', 'ggn'], [ngg, gng, ggn]):
        t0 = time.time()
        if name == 'ngg':
            corr.process(ncat, gcat, algo='triangle')
        elif name == 'gng':
            corr.process(gcat, ncat, gcat, algo='triangle', num_threads=1)
        else:
            corr.process(gcat, ncat, algo='triangle')
        t1 = time.time()
        print(name,'process time = ',t1-t0)

        # Use r1,r2 = distance from N vertex to two G vertices
        if name == 'ngg':
            r1 = corr.meand2/r0
            r2 = corr.meand3/r0
            r3 = corr.meand1/r0
            #phi = np.arccos((r1**2+r2**2-r3**2)/(2*r1*r2))
            phi = corr.meanphi
            g1 = lambda ngg: ngg.gam2
            cls = treecorr.NGGCorrelation
        elif name == 'gng':
            r1 = corr.meand3/r0
            r2 = corr.meand1/r0
            r3 = corr.meand2/r0
            phi = np.arccos((r1**2+r2**2-r3**2)/(2*r1*r2))
            g1 = lambda gng: gng.gam3
            cls = treecorr.GNGCorrelation
        else:
            r1 = corr.meand1/r0
            r2 = corr.meand2/r0
            r3 = corr.meand3/r0
            phi = np.arccos((r1**2+r2**2-r3**2)/(2*r1*r2))
            g1 = lambda ggn: ggn.gam1
            cls = treecorr.GGNCorrelation

        # Use coordinates where r1 is horizontal, N is at the origin.
        s = r1 * (1 + 0j)
        t = r2 * np.exp(1j * phi)
        q3 = (s + t)/3.
        q1 = q3 - t
        q2 = q3 - s

        gamma1 = -gamma0 * np.exp(-r2**2) * t**2
        gamma1 *= np.conjugate(q1)**2 / np.abs(q1**2)
        gamma2 = -gamma0 * np.exp(-r1**2) * s**2
        gamma2 *= np.conjugate(q2)**2 / np.abs(q2**2)

        true_gam0 = gamma1 * gamma2
        true_gam1 = np.conjugate(gamma1) * gamma2

        # We use small bins so the projection angles are well estimated.  But this means that
        # some bins end up with rather few triangles.  Limit the tests to those with a
        # reasonably large number of triangles.
        m = np.where(corr.ntri > np.max(corr.ntri)/2.)
        #print('gam0 ratio = ',(corr.gam0[m] / true_gam0[m]).ravel())
        print('max rel diff = ',np.max(np.abs((corr.gam0[m] - true_gam0[m])/true_gam0[m])))
        #print('gam1 ratio = ',(g1(corr)[m] / true_gam1[m]).ravel())
        print('max rel diff = ',np.max(np.abs((g1(corr)[m] - true_gam1[m])/true_gam1[m])))

        np.testing.assert_allclose(corr.gam0[m], true_gam0[m], rtol=0.2 * tol_factor, atol=1.e-7)
        np.testing.assert_allclose(np.log(np.abs(corr.gam0[m])),
                                   np.log(np.abs(true_gam0[m])), atol=0.2 * tol_factor)
        np.testing.assert_allclose(g1(corr)[m], true_gam1[m], rtol=0.3 * tol_factor, atol=1.e-7)
        np.testing.assert_allclose(np.log(np.abs(g1(corr)[m])),
                                   np.log(np.abs(true_gam1[m])), atol=0.3 * tol_factor)

        # Repeat this using Multipole and then convert to SAS:
        corrm = cls(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=80,
                    bin_type='LogMultipole')
        t0 = time.time()
        if name == 'ngg':
            corrm.process(ncat, gcat)
        elif name == 'gng':
            corrm.process(gcat, ncat, gcat)
        else:
            corrm.process(gcat, ncat)
        corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins)
        t1 = time.time()
        print('time for multipole corr:', t1-t0)

        print('gam0 mean ratio = ',np.mean(corrs.gam0[m] / corr.gam0[m]))
        print('gam0 mean diff = ',np.mean(corrs.gam0[m] - corr.gam0[m]))
        print('gam1 mean ratio = ',np.mean(g1(corrs)[m] / g1(corr)[m]))
        print('gam1 mean diff = ',np.mean(g1(corrs)[m] - g1(corr)[m]))
        # Some of the individual values are a little ratty, but on average, they are quite close.
        np.testing.assert_allclose(corrs.gam0[m], corr.gam0[m], rtol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.gam1[m], corr.gam1[m], rtol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.gam2[m], corr.gam2[m], rtol=0.2*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.gam0[m] / corr.gam0[m]), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.gam1[m] / corr.gam1[m]), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.gam2[m] / corr.gam2[m]), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.std(corrs.gam0[m] / corr.gam0[m]), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(np.std(corrs.gam1[m] / corr.gam1[m]), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(np.std(corrs.gam2[m] / corr.gam2[m]), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd1[m], corr.meanlogd1[m], atol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd2[m], corr.meanlogd2[m], atol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd3[m], corr.meanlogd3[m], atol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanphi[m], corr.meanphi[m], rtol=0.1*tol_factor)

        # Error to try to change sep binning with toSAS
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins, min_sep=5)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins, max_sep=25)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins, nbins=20)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                bin_size=0.01, nbins=None)
        # Error if non-Multipole calls toSAS
        with assert_raises(TypeError):
            corrs.toSAS()

        # All of the above is the default algorithm if process doesn't set algo='triangle'.
        # Check the automatic use of the multipole algorithm from LogSAS.
        corr3 = corr.copy()
        if name == 'ngg':
            corr3.process(ncat, gcat, max_n=80)
        elif name == 'gng':
            corr3.process(gcat, ncat, gcat, max_n=80)
        else:
            corr3.process(gcat, ncat, max_n=80)
        np.testing.assert_allclose(corr3.weight, corrs.weight)
        np.testing.assert_allclose(corr3.gam0, corrs.gam0)
        np.testing.assert_allclose(corr3.gam1, corrs.gam1)
        np.testing.assert_allclose(corr3.gam2, corrs.gam2)

        # Now use randoms
        corr2 = corr.copy()
        rand = corr.copy()
        if name == 'ngg':
            rand.process(rcat, gcat, algo='triangle')
            czkwargs = dict(rgg=rand)
        elif name == 'gng':
            rand.process(gcat, rcat, gcat, algo='triangle')
            czkwargs = dict(grg=rand)
        else:
            rand.process(gcat, rcat, algo='triangle')
            czkwargs = dict(ggr=rand)
        gam0, gam1, vargam0, vargam1 = corr2.calculateGam(**czkwargs)
        if name == 'gng':
            # gng uses gam3 for our nominal gam1, so need to conjugate the return value of
            # calculateGam
            gam1 = np.conjugate(gam1)
        print('with rand gam0 mean ratio = ',np.mean(gam0[m]/true_gam0[m]))
        print('with rand gam1 mean ratio = ',np.mean(gam1[m]/true_gam1[m]))
        np.testing.assert_allclose(gam0[m], true_gam0[m], rtol=0.15*tol_factor)
        np.testing.assert_allclose(gam1[m], true_gam1[m], rtol=0.15*tol_factor)

        corr2x = corr2.copy()
        np.testing.assert_allclose(corr2x.gam0, corr2.gam0)
        np.testing.assert_allclose(corr2x.gam1, corr2.gam1)
        np.testing.assert_allclose(corr2x.gam2, corr2.gam2)
        np.testing.assert_allclose(corr2x.vargam0, corr2.vargam0)
        np.testing.assert_allclose(corr2x.vargam1, corr2.vargam1)
        np.testing.assert_allclose(corr2x.vargam2, corr2.vargam2)
        np.testing.assert_allclose(corr.calculateGam()[0], corr.gam0)
        np.testing.assert_allclose(corr.calculateGam()[2], corr.vargam0)
        np.testing.assert_allclose(corrs.calculateGam()[0], corrs.gam0)
        np.testing.assert_allclose(corrs.calculateGam()[2], corrs.vargam0)

        with assert_raises(TypeError):
            corrm.calculateGam(**czkwargs)

        # Check that we get the same result using the corr3 functin:
        # (This implicitly uses the multipole algorithm.)
        ncat.write(os.path.join('data',name+'_ndata_logsas.dat'))
        gcat.write(os.path.join('data',name+'_gdata_logsas.dat'))
        config = treecorr.config.read_config('configs/'+name+'_logsas.yaml')
        config['verbose'] = 0
        treecorr.corr3(config)
        corr3_output = np.genfromtxt(os.path.join('output',name+'_logsas.out'),
                                     names=True, skip_header=1)
        np.testing.assert_allclose(corr3_output['gam0r'], corr3.gam0r.flatten(), rtol=1.e-3, atol=0)
        np.testing.assert_allclose(corr3_output['gam0i'], corr3.gam0i.flatten(), rtol=1.e-3, atol=0)
        if name == 'ngg':
            np.testing.assert_allclose(corr3_output['gam2r'], corr3.gam2r.flatten(),
                                       rtol=1.e-3, atol=0)
            np.testing.assert_allclose(corr3_output['gam2i'], corr3.gam2i.flatten(),
                                       rtol=1.e-3, atol=0)
        else:
            np.testing.assert_allclose(corr3_output['gam1r'], corr3.gam1r.flatten(),
                                       rtol=1.e-3, atol=0)
            np.testing.assert_allclose(corr3_output['gam1i'], corr3.gam1i.flatten(),
                                       rtol=1.e-3, atol=0)

        if name == 'gng':
            # Invalid to omit file_name2
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)
        else:
            # Invalid to call cat2 file_name3 rather than file_name2
            config['file_name3'] = config['file_name2']
            del config['file_name2']
            if name == 'ngg':
                config['g1_col'] = [0,0,3]
                config['g2_col'] = [0,0,4]
            else:
                config['g1_col'] = [3,0,0]
                config['g2_col'] = [4,0,0]
            with assert_raises(TypeError):
                treecorr.corr3(config)

        # Check the fits write option
        try:
            import fitsio
        except ImportError:
            pass
        else:
            out_file_name = os.path.join('output','corr_ngg_logsas.fits')
            corr.write(out_file_name)
            data = fitsio.read(out_file_name)
            np.testing.assert_allclose(data['d2_nom'], np.exp(corr.logd2).flatten())
            np.testing.assert_allclose(data['d3_nom'], np.exp(corr.logd3).flatten())
            np.testing.assert_allclose(data['phi_nom'], corr.phi.flatten())
            np.testing.assert_allclose(data['meand1'], corr.meand1.flatten())
            np.testing.assert_allclose(data['meanlogd1'], corr.meanlogd1.flatten())
            np.testing.assert_allclose(data['meand2'], corr.meand2.flatten())
            np.testing.assert_allclose(data['meanlogd2'], corr.meanlogd2.flatten())
            np.testing.assert_allclose(data['meand3'], corr.meand3.flatten())
            np.testing.assert_allclose(data['meanlogd3'], corr.meanlogd3.flatten())
            np.testing.assert_allclose(data['meanphi'], corr.meanphi.flatten())
            np.testing.assert_allclose(data['gam0r'], corr.gam0.real.flatten())
            np.testing.assert_allclose(data['gam0i'], corr.gam0.imag.flatten())
            np.testing.assert_allclose(data['sigma_gam0'], np.sqrt(corr.vargam0.flatten()))
            if name == 'ngg':
                np.testing.assert_allclose(data['gam2r'], corr.gam2.real.flatten())
                np.testing.assert_allclose(data['gam2i'], corr.gam2.imag.flatten())
                np.testing.assert_allclose(data['sigma_gam2'], np.sqrt(corr.vargam2.flatten()))
            else:
                np.testing.assert_allclose(data['gam1r'], corr.gam1.real.flatten())
                np.testing.assert_allclose(data['gam1i'], corr.gam1.imag.flatten())
                np.testing.assert_allclose(data['sigma_gam1'], np.sqrt(corr.vargam1.flatten()))
            np.testing.assert_allclose(data['weight'], corr.weight.flatten())
            np.testing.assert_allclose(data['ntri'], corr.ntri.flatten())

            # Check the read function
            # Note: These don't need the flatten.
            # The read function should reshape them to the right shape.
            corr2 = treecorr.Corr3.from_file(out_file_name)
            np.testing.assert_allclose(corr2.logd2, corr.logd2)
            np.testing.assert_allclose(corr2.logd3, corr.logd3)
            np.testing.assert_allclose(corr2.phi, corr.phi)
            np.testing.assert_allclose(corr2.meand1, corr.meand1)
            np.testing.assert_allclose(corr2.meanlogd1, corr.meanlogd1)
            np.testing.assert_allclose(corr2.meand2, corr.meand2)
            np.testing.assert_allclose(corr2.meanlogd2, corr.meanlogd2)
            np.testing.assert_allclose(corr2.meand3, corr.meand3)
            np.testing.assert_allclose(corr2.meanlogd3, corr.meanlogd3)
            np.testing.assert_allclose(corr2.meanphi, corr.meanphi)
            np.testing.assert_allclose(corr2.gam0, corr.gam0)
            np.testing.assert_allclose(corr2.gam1, corr.gam1)
            np.testing.assert_allclose(corr2.gam2, corr.gam2)
            np.testing.assert_allclose(corr2.vargam0, corr.vargam0)
            np.testing.assert_allclose(corr2.vargam1, corr.vargam1)
            np.testing.assert_allclose(corr2.vargam2, corr.vargam2)
            np.testing.assert_allclose(corr2.weight, corr.weight)
            np.testing.assert_allclose(corr2.ntri, corr.ntri)
            assert corr2.coords == corr.coords
            assert corr2.metric == corr.metric
            assert corr2.sep_units == corr.sep_units
            assert corr2.bin_type == corr.bin_type

@timer
def test_vargam():
    # Test that the shot noise estimate of vargam0 is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 100
    nsource = 10000

    file_name = 'data/test_vargam_ngg.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nggs = []
        all_gngs = []
        all_ggns = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise
            # from the random x,y positions.
            x1 = (rng.random_sample(nlens)-0.5) * L
            y1 = (rng.random_sample(nlens)-0.5) * L
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            # Varied weights are hard, but at least check that non-unit weights work correctly.
            w = np.ones_like(x2) * 5
            r2 = (x2**2 + y2**2)/r0**2
            g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
            g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
            g1 += rng.normal(0, 0.3, size=nsource)
            g2 += rng.normal(0, 0.3, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1)
            gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
            ngg = treecorr.NGGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            gng = treecorr.GNGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            ggn = treecorr.GGNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            ngg.process(ncat, gcat)
            gng.process(gcat, ncat, gcat)
            ggn.process(gcat, ncat)
            all_nggs.append(ngg)
            all_gngs.append(gng)
            all_ggns.append(ggn)

        mean_ngg_gam0r = np.mean([ngg.gam0r for ngg in all_nggs], axis=0)
        mean_ngg_gam0i = np.mean([ngg.gam0i for ngg in all_nggs], axis=0)
        mean_ngg_gam2r = np.mean([ngg.gam2r for ngg in all_nggs], axis=0)
        mean_ngg_gam2i = np.mean([ngg.gam2i for ngg in all_nggs], axis=0)
        var_ngg_gam0r = np.var([ngg.gam0r for ngg in all_nggs], axis=0)
        var_ngg_gam0i = np.var([ngg.gam0i for ngg in all_nggs], axis=0)
        var_ngg_gam2r = np.var([ngg.gam2r for ngg in all_nggs], axis=0)
        var_ngg_gam2i = np.var([ngg.gam2i for ngg in all_nggs], axis=0)
        mean_ngg_vargam0 = np.mean([ngg.vargam0 for ngg in all_nggs], axis=0)
        mean_ngg_vargam2 = np.mean([ngg.vargam2 for ngg in all_nggs], axis=0)
        mean_gng_gam0r = np.mean([gng.gam0r for gng in all_gngs], axis=0)
        mean_gng_gam0i = np.mean([gng.gam0i for gng in all_gngs], axis=0)
        mean_gng_gam1r = np.mean([gng.gam1r for gng in all_gngs], axis=0)
        mean_gng_gam1i = np.mean([gng.gam1i for gng in all_gngs], axis=0)
        var_gng_gam0r = np.var([gng.gam0r for gng in all_gngs], axis=0)
        var_gng_gam0i = np.var([gng.gam0i for gng in all_gngs], axis=0)
        var_gng_gam1r = np.var([gng.gam1r for gng in all_gngs], axis=0)
        var_gng_gam1i = np.var([gng.gam1i for gng in all_gngs], axis=0)
        mean_gng_vargam0 = np.mean([gng.vargam0 for gng in all_gngs], axis=0)
        mean_gng_vargam1 = np.mean([gng.vargam1 for gng in all_gngs], axis=0)
        mean_ggn_gam0r = np.mean([ggn.gam0r for ggn in all_ggns], axis=0)
        mean_ggn_gam0i = np.mean([ggn.gam0i for ggn in all_ggns], axis=0)
        mean_ggn_gam1r = np.mean([ggn.gam1r for ggn in all_ggns], axis=0)
        mean_ggn_gam1i = np.mean([ggn.gam1i for ggn in all_ggns], axis=0)
        var_ggn_gam0r = np.var([ggn.gam0r for ggn in all_ggns], axis=0)
        var_ggn_gam0i = np.var([ggn.gam0i for ggn in all_ggns], axis=0)
        var_ggn_gam1r = np.var([ggn.gam1r for ggn in all_ggns], axis=0)
        var_ggn_gam1i = np.var([ggn.gam1i for ggn in all_ggns], axis=0)
        mean_ggn_vargam0 = np.mean([ggn.vargam0 for ggn in all_ggns], axis=0)
        mean_ggn_vargam1 = np.mean([ggn.vargam1 for ggn in all_ggns], axis=0)

        np.savez(file_name,
                 mean_ngg_gam0r=mean_ngg_gam0r,
                 mean_ngg_gam0i=mean_ngg_gam0i,
                 mean_ngg_gam2r=mean_ngg_gam2r,
                 mean_ngg_gam2i=mean_ngg_gam2i,
                 var_ngg_gam0r=var_ngg_gam0r,
                 var_ngg_gam0i=var_ngg_gam0i,
                 var_ngg_gam2r=var_ngg_gam2r,
                 var_ngg_gam2i=var_ngg_gam2i,
                 mean_ngg_vargam0=mean_ngg_vargam0,
                 mean_ngg_vargam2=mean_ngg_vargam2,
                 mean_gng_gam0r=mean_gng_gam0r,
                 mean_gng_gam0i=mean_gng_gam0i,
                 mean_gng_gam1r=mean_gng_gam1r,
                 mean_gng_gam1i=mean_gng_gam1i,
                 var_gng_gam0r=var_gng_gam0r,
                 var_gng_gam0i=var_gng_gam0i,
                 var_gng_gam1r=var_gng_gam1r,
                 var_gng_gam1i=var_gng_gam1i,
                 mean_gng_vargam0=mean_gng_vargam0,
                 mean_gng_vargam1=mean_gng_vargam1,
                 mean_ggn_gam0r=mean_ggn_gam0r,
                 mean_ggn_gam0i=mean_ggn_gam0i,
                 mean_ggn_gam1r=mean_ggn_gam1r,
                 mean_ggn_gam1i=mean_ggn_gam1i,
                 var_ggn_gam0r=var_ggn_gam0r,
                 var_ggn_gam0i=var_ggn_gam0i,
                 var_ggn_gam1r=var_ggn_gam1r,
                 var_ggn_gam1i=var_ggn_gam1i,
                 mean_ggn_vargam0=mean_ggn_vargam0,
                 mean_ggn_vargam1=mean_ggn_vargam1)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_ngg_gam0r = data['mean_ngg_gam0r']
    mean_ngg_gam0i = data['mean_ngg_gam0i']
    mean_ngg_gam2r = data['mean_ngg_gam2r']
    mean_ngg_gam2i = data['mean_ngg_gam2i']
    var_ngg_gam0r = data['var_ngg_gam0r']
    var_ngg_gam0i = data['var_ngg_gam0i']
    var_ngg_gam2r = data['var_ngg_gam2r']
    var_ngg_gam2i = data['var_ngg_gam2i']
    mean_ngg_vargam0 = data['mean_ngg_vargam0']
    mean_ngg_vargam2 = data['mean_ngg_vargam2']
    mean_gng_gam0r = data['mean_gng_gam0r']
    mean_gng_gam0i = data['mean_gng_gam0i']
    mean_gng_gam1r = data['mean_gng_gam1r']
    mean_gng_gam1i = data['mean_gng_gam1i']
    var_gng_gam0r = data['var_gng_gam0r']
    var_gng_gam0i = data['var_gng_gam0i']
    var_gng_gam1r = data['var_gng_gam1r']
    var_gng_gam1i = data['var_gng_gam1i']
    mean_gng_vargam0 = data['mean_gng_vargam0']
    mean_gng_vargam1 = data['mean_gng_vargam1']
    mean_ggn_gam0r = data['mean_ggn_gam0r']
    mean_ggn_gam0i = data['mean_ggn_gam0i']
    mean_ggn_gam1r = data['mean_ggn_gam1r']
    mean_ggn_gam1i = data['mean_ggn_gam1i']
    var_ggn_gam0r = data['var_ggn_gam0r']
    var_ggn_gam0i = data['var_ggn_gam0i']
    var_ggn_gam1r = data['var_ggn_gam1r']
    var_ggn_gam1i = data['var_ggn_gam1i']
    mean_ggn_vargam0 = data['mean_ggn_vargam0']
    mean_ggn_vargam1 = data['mean_ggn_vargam1']

    print('var_ngg_gam0r = ',var_ngg_gam0r)
    print('mean ngg_vargam0 = ',mean_ngg_vargam0)
    print('ratio = ',var_ngg_gam0r.ravel() / mean_ngg_vargam0.ravel())
    print('var_gng_gam0r = ',var_gng_gam0r)
    print('mean gng_vargam0 = ',mean_gng_vargam0)
    print('ratio = ',var_gng_gam0r.ravel() / mean_gng_vargam0.ravel())
    print('var_ggn_gam0r = ',var_ggn_gam0r)
    print('mean ggn_vargam0 = ',mean_ggn_vargam0)
    print('ratio = ',var_ggn_gam0r.ravel() / mean_ggn_vargam0.ravel())

    print('max relerr for ngg gam0r = ',
          np.max(np.abs((var_ngg_gam0r - mean_ngg_vargam0)/var_ngg_gam0r)))
    print('max relerr for ngg gam0i = ',
          np.max(np.abs((var_ngg_gam0i - mean_ngg_vargam0)/var_ngg_gam0i)))
    print('max relerr for ngg gam2r = ',
          np.max(np.abs((var_ngg_gam2r - mean_ngg_vargam2)/var_ngg_gam2r)))
    print('max relerr for ngg gam2i = ',
          np.max(np.abs((var_ngg_gam2i - mean_ngg_vargam2)/var_ngg_gam2i)))
    np.testing.assert_allclose(mean_ngg_vargam0, var_ngg_gam0r, rtol=0.1)
    np.testing.assert_allclose(mean_ngg_vargam0, var_ngg_gam0i, rtol=0.1)
    np.testing.assert_allclose(mean_ngg_vargam2, var_ngg_gam2r, rtol=0.15)
    np.testing.assert_allclose(mean_ngg_vargam2, var_ngg_gam2i, rtol=0.15)

    print('max relerr for gng gam0r = ',
          np.max(np.abs((var_gng_gam0r - mean_gng_vargam0)/var_gng_gam0r)))
    print('max relerr for gng gam0i = ',
          np.max(np.abs((var_gng_gam0i - mean_gng_vargam0)/var_gng_gam0i)))
    print('max relerr for gng gam1r = ',
          np.max(np.abs((var_gng_gam1r - mean_gng_vargam1)/var_gng_gam1r)))
    print('max relerr for gng gam0i = ',
          np.max(np.abs((var_gng_gam1i - mean_gng_vargam1)/var_gng_gam1i)))
    np.testing.assert_allclose(mean_gng_vargam0, var_gng_gam0r, rtol=0.1)
    np.testing.assert_allclose(mean_gng_vargam0, var_gng_gam0i, rtol=0.1)
    np.testing.assert_allclose(mean_gng_vargam1, var_gng_gam1r, rtol=0.1)
    np.testing.assert_allclose(mean_gng_vargam1, var_gng_gam1i, rtol=0.1)

    print('max relerr for ggn gam0r = ',
          np.max(np.abs((var_ggn_gam0r - mean_ggn_vargam0)/var_ggn_gam0r)))
    print('max relerr for ggn gam0i = ',
          np.max(np.abs((var_ggn_gam0i - mean_ggn_vargam0)/var_ggn_gam0i)))
    print('max relerr for ggn gam1r = ',
          np.max(np.abs((var_ggn_gam1r - mean_ggn_vargam1)/var_ggn_gam1r)))
    print('max relerr for ggn gam0i = ',
          np.max(np.abs((var_ggn_gam1i - mean_ggn_vargam1)/var_ggn_gam1i)))
    np.testing.assert_allclose(mean_ggn_vargam0, var_ggn_gam0r, rtol=0.1)
    np.testing.assert_allclose(mean_ggn_vargam0, var_ggn_gam0i, rtol=0.1)
    np.testing.assert_allclose(mean_ggn_vargam1, var_ggn_gam1r, rtol=0.1)
    np.testing.assert_allclose(mean_ggn_vargam1, var_ggn_gam1i, rtol=0.1)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5
    r2 = (x2**2 + y2**2)/r0**2
    g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
    g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
    g1 += rng.normal(0, 0.3, size=nsource)
    g2 += rng.normal(0, 0.3, size=nsource)

    ncat = treecorr.Catalog(x=x1, y=y1)
    gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
    ngg = treecorr.NGGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)
    gng = treecorr.GNGCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)
    ggn = treecorr.GGNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)

    # Before running process, vargam0 and cov are allowed, but all 0.
    np.testing.assert_array_equal(ngg.cov, 0)
    np.testing.assert_array_equal(ngg.vargam0, 0)
    np.testing.assert_array_equal(ngg.vargam2, 0)
    np.testing.assert_array_equal(gng.cov, 0)
    np.testing.assert_array_equal(gng.vargam0, 0)
    np.testing.assert_array_equal(gng.vargam1, 0)
    np.testing.assert_array_equal(ggn.cov, 0)
    np.testing.assert_array_equal(ggn.vargam0, 0)
    np.testing.assert_array_equal(ggn.vargam1, 0)

    ngg.process(ncat, gcat)
    print('NGG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ngg.vargam0 - var_ngg_gam0r)/var_ngg_gam0r)))
    print('ratio = ',ngg.vargam0 / var_ngg_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ngg.vargam0 - var_ngg_gam0i)/var_ngg_gam0i)))
    print('ratio = ',ngg.vargam0 / var_ngg_gam0i)
    np.testing.assert_allclose(ngg.vargam0, var_ngg_gam0r, rtol=0.2)
    np.testing.assert_allclose(ngg.vargam0, var_ngg_gam0i, rtol=0.2)
    np.testing.assert_allclose(ngg.vargam2, var_ngg_gam2r, rtol=0.2)
    np.testing.assert_allclose(ngg.vargam2, var_ngg_gam2i, rtol=0.2)
    n = ngg.vargam0.size
    np.testing.assert_allclose(ngg.cov.diagonal()[0:n], ngg.vargam0.ravel())
    np.testing.assert_allclose(ngg.cov.diagonal()[n:2*n], ngg.vargam2.ravel())

    gng.process(gcat, ncat, gcat)
    print('GNG single run:')
    print('max relerr for gam0r = ',np.max(np.abs((gng.vargam0 - var_gng_gam0r)/var_gng_gam0r)))
    print('ratio = ',gng.vargam0 / var_gng_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((gng.vargam0 - var_gng_gam0i)/var_gng_gam0i)))
    print('ratio = ',gng.vargam0 / var_gng_gam0i)
    np.testing.assert_allclose(gng.vargam0, var_gng_gam0r, rtol=0.2)
    np.testing.assert_allclose(gng.vargam0, var_gng_gam0i, rtol=0.2)
    np.testing.assert_allclose(gng.vargam1, var_gng_gam1r, rtol=0.2)
    np.testing.assert_allclose(gng.vargam1, var_gng_gam1i, rtol=0.2)
    np.testing.assert_allclose(gng.cov.diagonal()[0:n], gng.vargam0.ravel())
    np.testing.assert_allclose(gng.cov.diagonal()[n:2*n], gng.vargam1.ravel())

    ggn.process(gcat, ncat)
    print('GGN single run:')
    print('max relerr for gam0r = ',np.max(np.abs((ggn.vargam0 - var_ggn_gam0r)/var_ggn_gam0r)))
    print('ratio = ',ggn.vargam0 / var_ggn_gam0r)
    print('max relerr for gam0i = ',np.max(np.abs((ggn.vargam0 - var_ggn_gam0i)/var_ggn_gam0i)))
    print('ratio = ',ggn.vargam0 / var_ggn_gam0i)
    np.testing.assert_allclose(ggn.vargam0, var_ggn_gam0r, rtol=0.2)
    np.testing.assert_allclose(ggn.vargam0, var_ggn_gam0i, rtol=0.2)
    np.testing.assert_allclose(ggn.vargam1, var_ggn_gam1r, rtol=0.2)
    np.testing.assert_allclose(ggn.vargam1, var_ggn_gam1i, rtol=0.2)
    np.testing.assert_allclose(ggn.cov.diagonal()[0:n], ggn.vargam0.ravel())
    np.testing.assert_allclose(ggn.cov.diagonal()[n:2*n], ggn.vargam1.ravel())

@timer
def test_ngg_logsas_jk():
    # Test jackknife covariance estimates for ngg correlations with LogSAS binning.

    # Skip this test on windows, since it is vv slow.
    if os.name == 'nt': return

    if __name__ == '__main__':
        nhalo = 50
        nsource = 400000
        npatch = 12
        tol_factor = 1
    else:
        nhalo = 50
        nsource = 100000
        npatch = 10
        tol_factor = 4

    nbins = 2
    min_sep = 10
    max_sep = 16
    nphi_bins = 5
    min_phi = 30
    max_phi = 90

    file_name = 'data/test_ngg_logsas_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_ngg0 = []
        all_ngg2 = []
        all_gng0 = []
        all_gng1 = []
        all_ggn0 = []
        all_ggn1 = []
        all_ngg0_r = []
        all_ngg2_r = []
        all_gng0_r = []
        all_gng1_r = []
        all_ggn0_r = []
        all_ggn1_r = []
        rng1 = np.random.default_rng()
        for run in range(nruns):
            # It doesn't work as well if the same points are in both, so make a single field
            # with both k and g, but take half the points for k, and the other half for g.
            x, y, g1, g2, _, xh, yh = generate_shear_field(nsource, nhalo, rng1, return_halos=True)
            xr = rng1.uniform(0, 1000, size=10*nhalo)
            yr = rng1.uniform(0, 1000, size=10*nhalo)
            print(run,': ',np.std(g1),np.std(g2))
            gcat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            ncat = treecorr.Catalog(x=xh, y=yh)
            rcat = treecorr.Catalog(x=xr, y=yr)
            ngg = treecorr.NGGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            ngg.process(ncat, gcat)
            all_ngg0.append(ngg.gam0.ravel())
            all_ngg2.append(ngg.gam2.ravel())
            rgg = ngg.copy()
            rgg.process(rcat, gcat)
            gam0, gam1, _, _ = ngg.calculateGam(rgg=rgg)
            all_ngg0_r.append(gam0.ravel())
            all_ngg2_r.append(gam1.ravel())

            gng = treecorr.GNGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            gng.process(gcat, ncat, gcat)
            all_gng0.append(gng.gam0.ravel())
            all_gng1.append(gng.gam1.ravel())
            grg = gng.copy()
            grg.process(gcat, rcat, gcat)
            gam0, gam1, _, _ = gng.calculateGam(grg=grg)
            all_gng0_r.append(gam0.ravel())
            all_gng1_r.append(gam1.ravel())

            ggn = treecorr.GGNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            ggn.process(gcat, ncat)
            all_ggn0.append(ggn.gam0.ravel())
            all_ggn1.append(ggn.gam1.ravel())
            ggr = ggn.copy()
            ggr.process(gcat, rcat)
            gam0, gam1, _, _ = ggn.calculateGam(ggr=ggr)
            all_ggn0_r.append(gam0.ravel())
            all_ggn1_r.append(gam1.ravel())

        mean_ngg0 = np.mean(all_ngg0, axis=0)
        mean_ngg2 = np.mean(all_ngg2, axis=0)
        var_ngg0 = np.var(all_ngg0, axis=0)
        var_ngg2 = np.var(all_ngg2, axis=0)
        mean_gng0 = np.mean(all_gng0, axis=0)
        mean_gng1 = np.mean(all_gng1, axis=0)
        var_gng0 = np.var(all_gng0, axis=0)
        var_gng1 = np.var(all_gng1, axis=0)
        mean_ggn0 = np.mean(all_ggn0, axis=0)
        mean_ggn1 = np.mean(all_ggn1, axis=0)
        var_ggn0 = np.var(all_ggn0, axis=0)
        var_ggn1 = np.var(all_ggn1, axis=0)
        mean_ngg0_r = np.mean(all_ngg0, axis=0)
        mean_ngg2_r = np.mean(all_ngg2, axis=0)
        var_ngg0_r = np.var(all_ngg0, axis=0)
        var_ngg2_r = np.var(all_ngg2, axis=0)
        mean_gng0_r = np.mean(all_gng0, axis=0)
        mean_gng1_r = np.mean(all_gng1, axis=0)
        var_gng0_r = np.var(all_gng0, axis=0)
        var_gng1_r = np.var(all_gng1, axis=0)
        mean_ggn0_r = np.mean(all_ggn0, axis=0)
        mean_ggn1_r = np.mean(all_ggn1, axis=0)
        var_ggn0_r = np.var(all_ggn0, axis=0)
        var_ggn1_r = np.var(all_ggn1, axis=0)

        np.savez(file_name,
                 mean_ngg0=mean_ngg0, var_ngg0=var_ngg0,
                 mean_ngg2=mean_ngg2, var_ngg2=var_ngg2,
                 mean_gng0=mean_gng0, var_gng0=var_gng0,
                 mean_gng1=mean_gng1, var_gng1=var_gng1,
                 mean_ggn0=mean_ggn0, var_ggn0=var_ggn0,
                 mean_ggn1=mean_ggn1, var_ggn1=var_ggn1,
                 mean_ngg0_r=mean_ngg0_r, var_ngg0_r=var_ngg0_r,
                 mean_ngg2_r=mean_ngg2_r, var_ngg2_r=var_ngg2_r,
                 mean_gng0_r=mean_gng0_r, var_gng0_r=var_gng0_r,
                 mean_gng1_r=mean_gng1_r, var_gng1_r=var_gng1_r,
                 mean_ggn0_r=mean_ggn0_r, var_ggn0_r=var_ggn0_r,
                 mean_ggn1_r=mean_ggn1_r, var_ggn1_r=var_ggn1_r)

    data = np.load(file_name)
    mean_ngg0 = data['mean_ngg0']
    mean_ngg2 = data['mean_ngg2']
    var_ngg0 = data['var_ngg0']
    var_ngg2 = data['var_ngg2']
    mean_gng0 = data['mean_gng0']
    mean_gng1 = data['mean_gng1']
    var_gng0 = data['var_gng0']
    var_gng1 = data['var_gng1']
    mean_ggn0 = data['mean_ggn0']
    mean_ggn1 = data['mean_ggn1']
    var_ggn0 = data['var_ggn0']
    var_ggn1 = data['var_ggn1']
    mean_ngg0_r = data['mean_ngg0_r']
    mean_ngg2_r = data['mean_ngg2_r']
    var_ngg0_r = data['var_ngg0_r']
    var_ngg2_r = data['var_ngg2_r']
    mean_gng0_r = data['mean_gng0_r']
    mean_gng1_r = data['mean_gng1_r']
    var_gng0_r = data['var_gng0_r']
    var_gng1_r = data['var_gng1_r']
    mean_ggn0_r = data['mean_ggn0_r']
    mean_ggn1_r = data['mean_ggn1_r']
    var_ggn0_r = data['var_ggn0_r']
    var_ggn1_r = data['var_ggn1_r']
    print('mean ngg0 = ',mean_ngg0)
    print('mean ngg2 = ',mean_ngg2)
    print('var ngg0 = ',var_ngg0)
    print('var ngg2 = ',var_ngg2)
    print('mean gng0 = ',mean_gng0)
    print('mean gng1 = ',mean_gng1)
    print('var gng0 = ',var_gng0)
    print('var gng1 = ',var_gng1)
    print('mean ggn0 = ',mean_ggn0)
    print('mean ggn1 = ',mean_ggn1)
    print('var ggn0 = ',var_ggn0)
    print('var ggn1 = ',var_ggn1)

    rng = np.random.default_rng(123)
    x, y, g1, g2, _, xh, yh = generate_shear_field(2*nsource, nhalo, rng, return_halos=True)
    xr = rng.uniform(0, 1000, size=10*nhalo)
    yr = rng.uniform(0, 1000, size=10*nhalo)
    gcat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch, rng=rng)
    ncat = treecorr.Catalog(x=xh, y=yh, rng=rng, patch_centers=gcat.patch_centers)
    rcat = treecorr.Catalog(x=xr, y=yr, rng=rng, patch_centers=gcat.patch_centers)

    # First check calculate_xi with all pairs in results dict.
    ngg = treecorr.NGGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    ngg.process(ncat, gcat)
    ngg2 = ngg.copy()
    ngg2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngg.results.keys()], False)
    np.testing.assert_allclose(ngg2.ntri, ngg.ntri, rtol=0.01)
    np.testing.assert_allclose(ngg2.gam0, ngg.gam0, rtol=0.01)
    np.testing.assert_allclose(ngg2.gam2, ngg.gam2, rtol=0.01)
    np.testing.assert_allclose(ngg2.vargam0, ngg.vargam0, rtol=0.01)
    np.testing.assert_allclose(ngg2.vargam2, ngg.vargam2, rtol=0.01)
    rgg = ngg.copy()
    rgg.process(rcat, gcat)
    ngg_r = ngg.copy()
    gam0_rgg, gam2_rgg, vargam0_rgg, vargam2_rgg = ngg_r.calculateGam(rgg=rgg)
    ngg2 = ngg_r.copy()
    ngg2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngg_r.results.keys()], False)
    np.testing.assert_allclose(ngg2.gam0, ngg_r.gam0, rtol=0.01)
    np.testing.assert_allclose(ngg2.gam2, ngg_r.gam2, rtol=0.01)
    np.testing.assert_allclose(ngg2.vargam0, ngg_r.vargam0, rtol=0.01)
    np.testing.assert_allclose(ngg2.vargam2, ngg_r.vargam2, rtol=0.01)
    np.testing.assert_allclose(ngg2.gam0, gam0_rgg, rtol=0.01)
    np.testing.assert_allclose(ngg2.gam2, gam2_rgg, rtol=0.01)
    np.testing.assert_allclose(ngg2.vargam0, vargam0_rgg, rtol=0.01)
    np.testing.assert_allclose(ngg2.vargam2, vargam2_rgg, rtol=0.01)

    gng = treecorr.GNGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    gng.process(gcat, ncat, gcat)
    gng2 = gng.copy()
    gng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gng.results.keys()], False)
    np.testing.assert_allclose(gng2.ntri, gng.ntri, rtol=0.01)
    np.testing.assert_allclose(gng2.gam0, gng.gam0, rtol=0.01)
    np.testing.assert_allclose(gng2.gam1, gng.gam1, rtol=0.01)
    np.testing.assert_allclose(gng2.vargam0, gng.vargam0, rtol=0.01)
    np.testing.assert_allclose(gng2.vargam1, gng.vargam1, rtol=0.01)
    grg = gng.copy()
    grg.process(gcat, rcat, gcat)
    gng_r = gng.copy()
    gam0_grg, gam1_grg, vargam0_grg, vargam1_grg = gng_r.calculateGam(grg=grg)
    gng2 = gng_r.copy()
    gng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gng_r.results.keys()], False)
    np.testing.assert_allclose(gng2.gam0, gng_r.gam0, rtol=0.01)
    np.testing.assert_allclose(gng2.gam1, gng_r.gam1, rtol=0.01)
    np.testing.assert_allclose(gng2.vargam0, gng_r.vargam0, rtol=0.01)
    np.testing.assert_allclose(gng2.vargam1, gng_r.vargam1, rtol=0.01)
    np.testing.assert_allclose(gng2.gam0, gam0_grg, rtol=0.01)
    np.testing.assert_allclose(gng2.gam1, gam1_grg, rtol=0.01)
    np.testing.assert_allclose(gng2.vargam0, vargam0_grg, rtol=0.01)
    np.testing.assert_allclose(gng2.vargam1, vargam1_grg, rtol=0.01)

    ggn = treecorr.GGNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    ggn.process(gcat, ncat)
    ggn2 = ggn.copy()
    ggn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ggn.results.keys()], False)
    np.testing.assert_allclose(ggn2.ntri, ggn.ntri, rtol=0.01)
    np.testing.assert_allclose(ggn2.gam0, ggn.gam0, rtol=0.01)
    np.testing.assert_allclose(ggn2.gam1, ggn.gam1, rtol=0.01)
    np.testing.assert_allclose(ggn2.vargam0, ggn.vargam0, rtol=0.01)
    np.testing.assert_allclose(ggn2.vargam1, ggn.vargam1, rtol=0.01)
    ggr = gng.copy()
    ggr.process(gcat, rcat, gcat)
    ggn_r = ggn.copy()
    gam0_ggr, gam1_ggr, vargam0_ggr, vargam1_ggr = ggn_r.calculateGam(ggr=ggr)
    ggn2 = ggn_r.copy()
    ggn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ggn_r.results.keys()], False)
    np.testing.assert_allclose(ggn2.gam0, ggn_r.gam0, rtol=0.01)
    np.testing.assert_allclose(ggn2.gam1, ggn_r.gam1, rtol=0.01)
    np.testing.assert_allclose(ggn2.vargam0, ggn_r.vargam0, rtol=0.01)
    np.testing.assert_allclose(ggn2.vargam1, ggn_r.vargam1, rtol=0.01)
    np.testing.assert_allclose(ggn2.gam0, gam0_ggr, rtol=0.01)
    np.testing.assert_allclose(ggn2.gam1, gam1_ggr, rtol=0.01)
    np.testing.assert_allclose(ggn2.vargam0, vargam0_ggr, rtol=0.01)
    np.testing.assert_allclose(ggn2.vargam1, vargam1_ggr, rtol=0.01)

    # Next check jackknife covariance estimate
    cov_ngg = ngg.estimate_cov('jackknife')
    n = ngg.vargam0.size
    print('ngg gam0 var ratio = ',np.diagonal(cov_ngg)[0:n]/var_ngg0)
    print('ngg0 max log(ratio) = ',
          np.max(np.abs(np.log(np.diagonal(cov_ngg)[0:n]) - np.log(var_ngg0))))
    print('ngg gam2 var ratio = ',np.diagonal(cov_ngg)[n:2*n]/var_ngg2)
    print('ngg2 max log(ratio) = ',
          np.max(np.abs(np.log(np.diagonal(cov_ngg)[n:2*n]) - np.log(var_ngg2))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_ngg)[0:n]), np.log(var_ngg0), atol=0.4*tol_factor)
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_ngg)[n:2*n]), np.log(var_ngg2), atol=0.4*tol_factor)

    cov_gng = gng.estimate_cov('jackknife')
    print('gng gam0 var ratio = ',np.diagonal(cov_gng)[0:n]/var_gng0)
    print('gng0 max log(ratio) = ',
          np.max(np.abs(np.log(np.diagonal(cov_gng)[0:n]) - np.log(var_gng0))))
    print('gng gam1 var ratio = ',np.diagonal(cov_gng)[n:2*n]/var_gng1)
    print('gng1 max log(ratio) = ',
          np.max(np.abs(np.log(np.diagonal(cov_gng)[n:2*n]) - np.log(var_gng1))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_gng)[0:n]), np.log(var_gng0), atol=0.4*tol_factor)
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_gng)[n:2*n]), np.log(var_gng1), atol=0.5*tol_factor)

    cov_ggn = ggn.estimate_cov('jackknife')
    print('ggn gam0 var ratio = ',np.diagonal(cov_ggn)[0:n]/var_ggn0)
    print('ggn0 max log(ratio) = ',
          np.max(np.abs(np.log(np.diagonal(cov_ggn)[0:n]) - np.log(var_ggn0))))
    print('ggn gam1 var ratio = ',np.diagonal(cov_ggn)[n:2*n]/var_ggn1)
    print('ggn1 max log(ratio) = ',
          np.max(np.abs(np.log(np.diagonal(cov_ggn)[n:2*n]) - np.log(var_ggn1))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_ggn)[0:n]), np.log(var_ggn0), atol=0.5*tol_factor)
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_ggn)[n:2*n]), np.log(var_ggn1), atol=0.5*tol_factor)

    # Check that these still work after roundtripping through a file.
    file_name = os.path.join('output','test_write_results_ngg.dat')
    ngg.write(file_name, write_patch_results=True)
    ngg2 = treecorr.Corr3.from_file(file_name)
    cov2 = ngg2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_ngg)

    file_name = os.path.join('output','test_write_results_gng.dat')
    gng.write(file_name, write_patch_results=True)
    gng2 = treecorr.Corr3.from_file(file_name)
    cov2 = gng2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_gng)

    file_name = os.path.join('output','test_write_results_ggn.dat')
    ggn.write(file_name, write_patch_results=True)
    ggn2 = treecorr.Corr3.from_file(file_name)
    cov2 = ggn2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_ggn)

    # Check jackknife using LogMultipole
    print('Using LogMultipole:')
    nggm = treecorr.NGGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    nggm.process(ncat, gcat)
    fm0 = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins).gam0.ravel()
    fm1 = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins).gam1.ravel()
    fm2 = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins).gam2.ravel()
    cov = nggm.estimate_cov('jackknife', func=fm0)
    print('ngg0 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ngg0))))
    np.testing.assert_allclose(np.diagonal(cov), var_ngg0, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ngg0), atol=0.5*tol_factor)
    cov = nggm.estimate_cov('jackknife', func=fm2)
    print('ngg1 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ngg2))))
    np.testing.assert_allclose(np.diagonal(cov), var_ngg2, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ngg2), atol=0.5*tol_factor)

    gngm = treecorr.GNGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    gngm.process(gcat, ncat, gcat)
    cov = gngm.estimate_cov('jackknife', func=fm0)
    print('gng0 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_gng0))))
    np.testing.assert_allclose(np.diagonal(cov), var_gng0, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_gng0), atol=0.5*tol_factor)
    cov = gngm.estimate_cov('jackknife', func=fm1)
    print('gng1 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_gng1))))
    np.testing.assert_allclose(np.diagonal(cov), var_gng1, rtol=0.4 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_gng1), atol=0.5*tol_factor)

    ggnm = treecorr.GGNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    ggnm.process(gcat, ncat)
    cov = ggnm.estimate_cov('jackknife', func=fm0)
    print('ggn0 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggn0))))
    np.testing.assert_allclose(np.diagonal(cov), var_ggn0, rtol=0.6 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggn0), atol=0.7*tol_factor)
    cov = ggnm.estimate_cov('jackknife', func=fm1)
    print('ggn1 max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ggn1))))
    np.testing.assert_allclose(np.diagonal(cov), var_ggn1, rtol=0.5 * tol_factor)
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ggn1), atol=0.7*tol_factor)

@timer
def test_ngg_rlens():
    # Similar to test_ng_rlens, but with two shears
    # Use gaussian tangential shear around lens centers
    # gamma_t(r) = gamma0 exp(-R^2/2R0^2)

    nlens = 300
    nsource = 100000
    gamma0 = 0.05
    r0 = 10.
    L = 100. * r0
    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L  # -500 < x < 500
    zl = (rng.random_sample(nlens)-0.5) * L  # -500 < y < 500
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 10000 < z < 14000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    xs = (rng.random_sample(nsource)-0.5) * 10*L   # -5000 < x < 5000
    zs = (rng.random_sample(nsource)-0.5) * 10*L   # -5000 < y < 5000
    ys = rng.random_sample(nsource) * 40*L + 100*L  # 100000 < z < 140000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    for x,y,z,r in zip(xl,yl,zl,rl):
        # Rlens = |r1 x r2| / |r2|
        xcross = ys * z - zs * y
        ycross = zs * x - xs * z
        zcross = xs * y - ys * x
        Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / rs

        gammat = gamma0 * np.exp(-0.5*Rlens**2/r0**2)
        # For the rotation, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = xs/rs-x/r
        dz = zs/rs-z/r
        drsq = dx**2 + dz**2
        g1 += -gammat * (dx**2-dz**2)/drsq
        g2 += -gammat * (2.*dx*dz)/drsq

    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)

    bin_size = 0.2
    min_sep = 12
    max_sep = 16
    min_phi = 0.5
    max_phi = 2.5
    nphi_bins = 20

    ngg = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  metric='Rlens')

    def predict(ngg):
        # Prediction is the same as in test_ngg_logsas above.
        r2 = ngg.meand2
        r3 = ngg.meand3
        phi = ngg.meanphi

        # Use coordinates where r2 is horizontal, N is at the origin.
        s = r2 * (1 + 0j) / r0
        t = r3 * np.exp(1j * phi) / r0
        q1 = (s + t)/3.
        q2 = q1 - t
        q3 = q1 - s

        gamma2 = -gamma0 * np.exp(-0.5*r3**2/r0**2) * t**2 / np.abs(t)**2
        gamma2 *= np.conjugate(q2)**2 / np.abs(q2**2)
        gamma3 = -gamma0 * np.exp(-0.5*r2**2/r0**2) * s**2 / np.abs(s)**2
        gamma3 *= np.conjugate(q3)**2 / np.abs(q3**2)

        true_gam0 = gamma2 * gamma3
        true_gam2 = np.conjugate(gamma2) * gamma3
        return true_gam0, true_gam2

    if __name__ == '__main__':
        t0 = time.time()
        ngg.process(lens_cat, source_cat, algo='triangle')
        t1 = time.time()
        print('time for algo=triangle process = ',t1-t0)

        true_gam0, true_gam2 = predict(ngg)

        #print('ngg.gam0 = ',ngg.gam0.ravel())
        #print('theory = ',true_gam0.ravel())
        #print('ratio = ',ngg.gam0.ravel() / true_gam0.ravel())
        #print('diff = ',ngg.gam0.ravel() - true_gam0.ravel())
        print('gam0 max diff = ',np.max(np.abs(ngg.gam0 - true_gam0)))
        print('max rel diff = ',np.max(np.abs(ngg.gam0 / true_gam0 - 1)))
        np.testing.assert_allclose(ngg.gam0, true_gam0, rtol=0.3)

        #print('ngg.gam2 = ',ngg.gam2.ravel())
        #print('theory = ',true_gam2.ravel())
        #print('ratio = ',ngg.gam2.ravel() / true_gam2.ravel())
        #print('diff = ',ngg.gam2.ravel() - true_gam2.ravel())
        print('gam2 max diff = ',np.max(np.abs(ngg.gam2 - true_gam2)))
        print('max rel diff = ',np.max(np.abs(ngg.gam2 / true_gam2 - 1)))
        np.testing.assert_allclose(ngg.gam2, true_gam2, rtol=0.3)

    # Now use multipole algorithm
    t0 = time.time()
    ngg.process(lens_cat, source_cat)
    t1 = time.time()
    print('time for algo=multipole process = ',t1-t0)

    true_gam0, true_gam2 = predict(ngg)

    print('Results using multipole')
    #print('ngg.gam0 = ',ngg.gam0.ravel())
    #print('theory = ',true_gam0.ravel())
    #print('ratio = ',ngg.gam0.ravel() / true_gam0.ravel())
    #print('diff = ',ngg.gam0.ravel() - true_gam0.ravel())
    print('gam0 max diff = ',np.max(np.abs(ngg.gam0 - true_gam0)))
    print('max rel diff = ',np.max(np.abs(ngg.gam0 / true_gam0 - 1)))
    np.testing.assert_allclose(ngg.gam0, true_gam0, rtol=0.3)

    #print('ngg.gam2 = ',ngg.gam2.ravel())
    #print('theory = ',true_gam2.ravel())
    #print('ratio = ',ngg.gam2.ravel() / true_gam2.ravel())
    #print('diff = ',ngg.gam2.ravel() - true_gam2.ravel())
    print('gam2 max diff = ',np.max(np.abs(ngg.gam2 - true_gam2)))
    print('max rel diff = ',np.max(np.abs(ngg.gam2 / true_gam2 - 1)))
    np.testing.assert_allclose(ngg.gam2, true_gam2, rtol=0.3)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ngg_rlens_lens.dat'))
    source_cat.write(os.path.join('data','ngg_rlens_source.dat'))
    config = treecorr.read_config('configs/ngg_rlens.yaml')
    config['verbose'] = 0
    treecorr.corr3(config)
    corr3_output = np.genfromtxt(os.path.join('output','ngg_rlens.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr3_output['gam0r'], ngg.gam0r.ravel(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam0i'], ngg.gam0i.ravel(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2r'], ngg.gam2r.ravel(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2i'], ngg.gam2i.ravel(), rtol=1.e-3)

    # Repeat with the sources being given as RA/Dec only.
    ral, decl = coord.CelestialCoord.xyz_to_radec(xl,yl,zl)
    ras, decs = coord.CelestialCoord.xyz_to_radec(xs,ys,zs)
    lens_cat = treecorr.Catalog(ra=ral, dec=decl, ra_units='radians', dec_units='radians', r=rl)
    source_cat = treecorr.Catalog(ra=ras, dec=decs, ra_units='radians', dec_units='radians',
                                  g1=g1, g2=g2)

    t0 = time.time()
    ngg.process(lens_cat, source_cat)
    t1 = time.time()
    print('time for algo=multipole process with ra/dec cats = ',t1-t0)

    print('Results when sources have no radius information')
    #print('ngg.gam0 = ',ngg.gam0)
    #print('theory = ',true_gam0)
    #print('ratio = ',ngg.gam0 / true_gam0)
    #print('diff = ',ngg.gam0 - true_gam0)
    print('gam0 max diff = ',np.max(np.abs(ngg.gam0 - true_gam0)))
    print('max rel diff = ',np.max(np.abs(ngg.gam0 / true_gam0 - 1)))
    np.testing.assert_allclose(ngg.gam0, true_gam0, rtol=0.3)

    #print('ngg.gam2 = ',ngg.gam2)
    #print('theory = ',true_gam2)
    #print('ratio = ',ngg.gam2 / true_gam2)
    #print('diff = ',ngg.gam2 - true_gam2)
    print('gam2 max diff = ',np.max(np.abs(ngg.gam2 - true_gam2)))
    print('max rel diff = ',np.max(np.abs(ngg.gam2 / true_gam2 - 1)))
    np.testing.assert_allclose(ngg.gam2, true_gam2, rtol=0.3)


@timer
def test_ngg_rlens_bkg():
    # Same as above, except limit the sources to be in the background of the lens.

    nlens = 300
    nsource = 100000
    gamma0 = 0.05
    r0 = 10.
    L = 100. * r0

    rng = np.random.RandomState(8675309)
    xl = (rng.random_sample(nlens)-0.5) * L  # -500 < x < 500
    zl = (rng.random_sample(nlens)-0.5) * L  # -500 < y < 500
    yl = rng.random_sample(nlens) * 4*L + 10*L  # 10000 < z < 14000
    rl = np.sqrt(xl**2 + yl**2 + zl**2)
    xs = (rng.random_sample(nsource)-0.5) * L   # -500 < x < 500
    zs = (rng.random_sample(nsource)-0.5) * L   # -500 < y < 500
    ys = rng.random_sample(nsource) * 10*L + 8*L  # 8000 < z < 18000
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    print('xl = ',np.min(xl),np.max(xl))
    print('yl = ',np.min(yl),np.max(yl))
    print('zl = ',np.min(zl),np.max(zl))
    print('xs = ',np.min(xs),np.max(xs))
    print('ys = ',np.min(ys),np.max(ys))
    print('zs = ',np.min(zs),np.max(zs))
    g1 = np.zeros( (nsource,) )
    g2 = np.zeros( (nsource,) )
    bin_size = 0.1
    print('Making shear vectors')
    for x,y,z,r in zip(xl,yl,zl,rl):
        # This time, only give the true shear to the background galaxies.
        bkg = (rs > r)

        # Rlens = |r1 x r2| / |r2|
        xcross = ys[bkg] * z - zs[bkg] * y
        ycross = zs[bkg] * x - xs[bkg] * z
        zcross = xs[bkg] * y - ys[bkg] * x
        Rlens = np.sqrt(xcross**2 + ycross**2 + zcross**2) / (rs[bkg])

        gammat = gamma0 * np.exp(-0.5*Rlens**2/r0**2)
        # For the rotation, approximate that the x,z coords are approx the perpendicular plane.
        # So just normalize back to the unit sphere and do the 2d projection calculation.
        # It's not exactly right, but it should be good enough for this unit test.
        dx = (xs/rs)[bkg]-x/r
        dz = (zs/rs)[bkg]-z/r
        drsq = dx**2 + dz**2

        g1[bkg] += -gammat * (dx**2-dz**2)/drsq
        g2[bkg] += -gammat * (2.*dx*dz)/drsq

    lens_cat = treecorr.Catalog(x=xl, y=yl, z=zl)
    source_cat = treecorr.Catalog(x=xs, y=ys, z=zs, g1=g1, g2=g2)

    bin_size = 0.2
    min_sep = 12
    max_sep = 16
    min_phi = 0.5
    max_phi = 2.5
    nphi_bins = 20

    ngg = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                  metric='Rlens', min_rpar=0)

    # Prediction is the same as above.
    def predict(ngg):
        r2 = ngg.meand2
        r3 = ngg.meand3
        phi = ngg.meanphi
        s = r2 * (1 + 0j) / r0
        t = r3 * np.exp(1j * phi) / r0
        q1 = (s + t)/3.
        q2 = q1 - t
        q3 = q1 - s
        gamma2 = -gamma0 * np.exp(-0.5*r3**2/r0**2) * t**2 / np.abs(t)**2
        gamma2 *= np.conjugate(q2)**2 / np.abs(q2**2)
        gamma3 = -gamma0 * np.exp(-0.5*r2**2/r0**2) * s**2 / np.abs(s)**2
        gamma3 *= np.conjugate(q3)**2 / np.abs(q3**2)
        true_gam0 = gamma2 * gamma3
        true_gam2 = np.conjugate(gamma2) * gamma3
        return true_gam0, true_gam2

    if __name__ == '__main__':
        t0 = time.time()
        ngg.process(lens_cat, source_cat, algo='triangle')
        t1 = time.time()
        print('time for algo=triangle process = ',t1-t0)

        true_gam0, true_gam2 = predict(ngg)

        print('ngg.gam0 = ',ngg.gam0.ravel())
        print('theory = ',true_gam0.ravel())
        print('ratio = ',ngg.gam0.ravel() / true_gam0.ravel())
        print('diff = ',ngg.gam0.ravel() - true_gam0.ravel())
        print('gam0 max diff = ',np.max(np.abs(ngg.gam0 - true_gam0)))
        print('max rel diff = ',np.max(np.abs(ngg.gam0 / true_gam0 - 1)))
        np.testing.assert_allclose(ngg.gam0, true_gam0, rtol=0.3)

        print('ngg.gam2 = ',ngg.gam2.ravel())
        print('theory = ',true_gam2.ravel())
        print('ratio = ',ngg.gam2.ravel() / true_gam2.ravel())
        print('diff = ',ngg.gam2.ravel() - true_gam2.ravel())
        print('gam2 max diff = ',np.max(np.abs(ngg.gam2 - true_gam2)))
        print('max rel diff = ',np.max(np.abs(ngg.gam2 / true_gam2 - 1)))
        np.testing.assert_allclose(ngg.gam2, true_gam2, rtol=0.3)

    ngg.process(lens_cat, source_cat)
    true_gam0, true_gam2 = predict(ngg)

    #print('ngg.gam0 = ',ngg.gam0.ravel())
    #print('theory = ',true_gam0.ravel())
    #print('diff = ',ngg.gam0.ravel() - true_gam0.ravel())
    print('gam0 ratio = ',ngg.gam0.ravel() / true_gam0.ravel())
    print('mean ratio = ',np.mean(ngg.gam0 / true_gam0))
    print('gam0 max diff = ',np.max(np.abs(ngg.gam0 - true_gam0)))
    print('max rel diff = ',np.max(np.abs(ngg.gam0 / true_gam0 - 1)))
    np.testing.assert_allclose(ngg.gam0, true_gam0, rtol=0.3)

    #print('ngg.gam2 = ',ngg.gam2.ravel())
    #print('theory = ',true_gam2.ravel())
    #print('diff = ',ngg.gam2.ravel() - true_gam2.ravel())
    print('gam2 ratio = ',ngg.gam2.ravel() / true_gam2.ravel())
    print('mean ratio = ',np.mean(ngg.gam2 / true_gam2))
    print('gam2 max diff = ',np.max(np.abs(ngg.gam2 - true_gam2)))
    print('max rel diff = ',np.max(np.abs(ngg.gam2 / true_gam2 - 1)))
    np.testing.assert_allclose(ngg.gam2, true_gam2, rtol=0.3)

    # Establish the right order of magnitude for the later tests without min_rpar.
    np.testing.assert_allclose(np.mean(ngg.gam0/true_gam0), 1., atol=0.15)
    np.testing.assert_allclose(np.mean(ngg.gam2/true_gam2), 1., atol=0.15)
    np.testing.assert_allclose(ngg.gam0, true_gam0, atol=1.e-4)
    np.testing.assert_allclose(ngg.gam2, true_gam2, atol=1.e-4)

    # Check that we get the same result using the corr2 function:
    lens_cat.write(os.path.join('data','ngg_rlens_bkg_lens.dat'))
    source_cat.write(os.path.join('data','ngg_rlens_bkg_source.dat'))
    config = treecorr.read_config('configs/ngg_rlens_bkg.yaml')
    config['verbose'] = 0
    treecorr.corr3(config)
    corr3_output = np.genfromtxt(os.path.join('output','ngg_rlens_bkg.out'), names=True,
                                 skip_header=1)
    np.testing.assert_allclose(corr3_output['gam0r'], ngg.gam0r.ravel(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam0i'], ngg.gam0i.ravel(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2r'], ngg.gam2r.ravel(), rtol=1.e-3)
    np.testing.assert_allclose(corr3_output['gam2i'], ngg.gam2i.ravel(), rtol=1.e-3)

    # Without min_rpar, this should fail.
    ngg2 = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                   min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                   metric='Rlens')
    ngg2.process(lens_cat, source_cat)

    print('Results without min_rpar')
    print('gam0 ratio = ',ngg2.gam0/true_gam0)
    print('mean ratio = ',np.mean(ngg2.gam0 / true_gam0))
    print('gam2 ratio = ',ngg2.gam2/true_gam2)
    print('mean ratio = ',np.mean(ngg2.gam2 / true_gam2))
    print('gam0 max diff = ',np.max(np.abs(ngg2.gam0 - true_gam0)))
    print('max rel diff = ',np.max(np.abs(ngg2.gam0 / true_gam0 - 1)))
    print('gam2 max diff = ',np.max(np.abs(ngg2.gam2 - true_gam2)))
    print('max rel diff = ',np.max(np.abs(ngg2.gam2 / true_gam2 - 1)))
    assert np.abs(np.mean(ngg2.gam0/true_gam0)) < 0.7
    assert np.abs(np.mean(ngg2.gam2/true_gam2)) < 0.7
    assert np.max(np.abs(ngg2.gam0 - true_gam0)) > 2.e-4
    assert np.max(np.abs(ngg2.gam2 - true_gam2)) > 2.e-4

    # Repeat with Arc metric
    min_sep /= 12*L
    max_sep /= 12*L
    ngg3 = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                   min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                   metric='Arc', min_rpar=0)
    ngg3.process(lens_cat, source_cat)

    print('Results for Arc')
    print('gam0 ratio = ',ngg3.gam0/true_gam0)
    print('mean ratio = ',np.mean(ngg3.gam0 / true_gam0))
    print('gam2 ratio = ',ngg3.gam2/true_gam2)
    print('mean ratio = ',np.mean(ngg3.gam2 / true_gam2))
    print('gam0 max diff = ',np.max(np.abs(ngg3.gam0 - true_gam0)))
    print('max rel diff = ',np.max(np.abs(ngg3.gam0 / true_gam0 - 1)))
    print('gam2 max diff = ',np.max(np.abs(ngg3.gam2 - true_gam2)))
    print('max rel diff = ',np.max(np.abs(ngg3.gam2 / true_gam2 - 1)))
    np.testing.assert_allclose(np.mean(ngg.gam0/true_gam0), 1., atol=0.2)
    np.testing.assert_allclose(np.mean(ngg.gam2/true_gam2), 1., atol=0.2)
    np.testing.assert_allclose(ngg3.gam0, true_gam0, rtol=0.5)
    np.testing.assert_allclose(ngg3.gam2, true_gam2, rtol=0.4)

    # Without min_rpar, this should fail.
    ngg4 = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                   min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                   metric='Arc')
    ngg4.process(lens_cat, source_cat)

    print('Results without min_rpar')
    print('gam0 ratio = ',ngg4.gam0/true_gam0)
    print('mean ratio = ',np.mean(ngg4.gam0 / true_gam0))
    print('gam2 ratio = ',ngg4.gam2/true_gam2)
    print('mean ratio = ',np.mean(ngg4.gam2 / true_gam2))
    print('gam0 max diff = ',np.max(np.abs(ngg4.gam0 - true_gam0)))
    print('max rel diff = ',np.max(np.abs(ngg4.gam0 / true_gam0 - 1)))
    print('gam2 max diff = ',np.max(np.abs(ngg4.gam2 - true_gam2)))
    print('max rel diff = ',np.max(np.abs(ngg4.gam2 / true_gam2 - 1)))
    assert np.abs(np.mean(ngg4.gam0/true_gam0)) < 0.7
    assert np.abs(np.mean(ngg4.gam2/true_gam2)) < 0.7
    assert np.max(np.abs(ngg4.gam0 - true_gam0)) > 2.e-4
    assert np.max(np.abs(ngg4.gam2 - true_gam2)) > 2.e-4

    # Can't have max_rpar < min_rpar
    with assert_raises(ValueError):
        treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                metric='Rlens', min_rpar=10, max_rpar=0)

    # Can't use min/max rpar with non-3d coords
    ral, decl = coord.CelestialCoord.xyz_to_radec(xl,yl,zl)
    ras, decs = coord.CelestialCoord.xyz_to_radec(xs,ys,zs)
    lens_cat_sph = treecorr.Catalog(ra=ral, dec=decl, ra_units='radians', dec_units='radians', r=rl)
    source_cat_sph = treecorr.Catalog(ra=ras, dec=decs, ra_units='radians', dec_units='radians',
                                      g1=g1, g2=g2)
    with assert_raises(ValueError):
        ngg.process(lens_cat_sph, source_cat_sph)
    ngg5 = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                   min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                   metric='Rlens', max_rpar=0)
    with assert_raises(ValueError):
        ngg5.process(lens_cat_sph, source_cat_sph)

    # sep units invalid with normal 3d coords (non-Arc metric)
    ngg6 = treecorr.NGGCorrelation(bin_size=bin_size, min_sep=min_sep, max_sep=max_sep,
                                   min_phi=min_phi, max_phi=max_phi, nphi_bins=nphi_bins,
                                   metric='Rlens', min_rpar=0, sep_units='arcmin')
    with assert_raises(ValueError):
        ngg6.process(lens_cat, source_cat)


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross12()
    test_vargam_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross12()
    test_direct_logmultipole_cross()
    test_direct_logmultipole_cross12()
    test_ngg_logsas()
    test_vargam()
    test_ngg_logsas_jk()
    test_ngg_rlens()
    test_ngg_rlens_bkg()

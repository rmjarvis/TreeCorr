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
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)
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

    nng = treecorr.NNGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    ngn = treecorr.NGNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gnn = treecorr.GNNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
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
    true_zeta_123 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_132 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_213 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_231 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_312 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_321 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
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

                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_123
                        true_zeta = true_zeta_123
                        true_weight = true_weight_123
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_213
                        true_zeta = true_zeta_213
                        true_weight = true_weight_213
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_231
                        true_zeta = true_zeta_231
                        true_weight = true_weight_231
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_132
                        true_zeta = true_zeta_132
                        true_weight = true_weight_132
                        ccw = not ccw
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_312
                        true_zeta = true_zeta_312
                        true_weight = true_weight_312
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_321
                        true_zeta = true_zeta_321
                        true_weight = true_weight_321
                        ccw = not ccw

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

                www = w1[i] * w2[j] * w3[k]
                zeta = www * g3p

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    true_ntri_sum3 = true_ntri_123 + true_ntri_213
    true_weight_sum3 = true_weight_123 + true_weight_213
    true_zeta_sum3 = true_zeta_123 + true_zeta_213
    true_ntri_sum2 = true_ntri_132 + true_ntri_231
    true_weight_sum2 = true_weight_132 + true_weight_231
    true_zeta_sum2 = true_zeta_132 + true_zeta_231
    true_ntri_sum1 = true_ntri_312 + true_ntri_321
    true_weight_sum1 = true_weight_312 + true_weight_321
    true_zeta_sum1 = true_zeta_312 + true_zeta_321
    pos = true_weight_sum1 > 0
    true_zeta_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_zeta_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_zeta_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    z_list = [true_zeta_123, true_zeta_132, true_zeta_213, true_zeta_231,
              true_zeta_312, true_zeta_321]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    nng.process(cat1, cat2, cat3)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zetar, true_zeta_123.real, rtol=1.e-5)
    np.testing.assert_allclose(nng.zetai, true_zeta_123.imag, rtol=1.e-5)

    nng.process(cat2, cat1, cat3)
    np.testing.assert_array_equal(nng.ntri, true_ntri_213)
    np.testing.assert_allclose(nng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_213, rtol=1.e-5)

    ngn.process(cat1, cat3, cat2)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zetar, true_zeta_132.real, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zetai, true_zeta_132.imag, rtol=1.e-5)
    ngn.process(cat2, cat3, cat1)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_231)
    np.testing.assert_allclose(ngn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_231, rtol=1.e-5)

    gnn.process(cat3, cat1, cat2)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zetar, true_zeta_312.real, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zetai, true_zeta_312.imag, rtol=1.e-5)
    gnn.process(cat3, cat2, cat1)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_321)
    np.testing.assert_allclose(gnn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where G is in 3
    nng.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)

    ngn.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check bin_slop=0
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nng.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=True)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # And again with no top-level recursion
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nng.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=True)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # With these, ordered=False is equivalent to the G vertex being fixed.
    nng.process(cat1, cat2, cat3, ordered=3)
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=2)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=1)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng2 = nng.copy()
    nng2.process(cat1, cat2, cat3, ordered=False, corr_only=True)
    np.testing.assert_allclose(nng2.weight, nng.weight)
    np.testing.assert_allclose(nng2.zeta, nng.zeta)
    np.testing.assert_allclose(nng2.ntri, nng.weight / (np.mean(w1)*np.mean(w2)*np.mean(w3)))
    np.testing.assert_allclose(nng2.meand2, nng.rnom)
    np.testing.assert_allclose(nng2.meanlogd2, nng.logr)
    np.testing.assert_allclose(nng2.meanu, nng.u)
    np.testing.assert_allclose(nng2.meanv, nng.v)
    np.testing.assert_allclose(nng2.meand3/nng2.meand2, nng.u)
    np.testing.assert_allclose((nng2.meand1 - nng2.meand2)/nng2.meand3, np.abs(nng.v))
    np.testing.assert_allclose(nng2.meanlogd1, np.log(nng2.meand1))
    np.testing.assert_allclose(nng2.meanlogd3, np.log(nng2.meand3))

    ngn2 = ngn.copy()
    ngn2.process(cat1, cat3, cat2, ordered=False, corr_only=True)
    np.testing.assert_allclose(ngn2.weight, ngn.weight)
    np.testing.assert_allclose(ngn2.zeta, ngn.zeta)
    np.testing.assert_allclose(ngn2.ntri, ngn.weight / (np.mean(w1)*np.mean(w2)*np.mean(w3)))
    np.testing.assert_allclose(ngn2.meand2, ngn.rnom)
    np.testing.assert_allclose(ngn2.meanlogd2, ngn.logr)
    np.testing.assert_allclose(ngn2.meanu, ngn.u)
    np.testing.assert_allclose(ngn2.meanv, ngn.v)
    np.testing.assert_allclose(ngn2.meand3/ngn2.meand2, ngn.u)
    np.testing.assert_allclose((ngn2.meand1 - ngn2.meand2)/ngn2.meand3, np.abs(ngn.v))
    np.testing.assert_allclose(ngn2.meanlogd1, np.log(ngn2.meand1))
    np.testing.assert_allclose(ngn2.meanlogd3, np.log(ngn2.meand3))

    gnn2 = gnn.copy()
    gnn2.process(cat3, cat1, cat2, ordered=False, corr_only=True)
    np.testing.assert_allclose(gnn2.weight, gnn.weight)
    np.testing.assert_allclose(gnn2.zeta, gnn.zeta)
    np.testing.assert_allclose(gnn2.ntri, gnn.weight / (np.mean(w1)*np.mean(w2)*np.mean(w3)))
    np.testing.assert_allclose(gnn2.meand2, gnn.rnom)
    np.testing.assert_allclose(gnn2.meanlogd2, gnn.logr)
    np.testing.assert_allclose(gnn2.meanu, gnn.u)
    np.testing.assert_allclose(gnn2.meanv, gnn.v)
    np.testing.assert_allclose(gnn2.meand3/gnn2.meand2, gnn.u)
    np.testing.assert_allclose((gnn2.meand1 - gnn2.meand2)/gnn2.meand3, np.abs(gnn.v))
    np.testing.assert_allclose(gnn2.meanlogd1, np.log(gnn2.meand1))
    np.testing.assert_allclose(gnn2.meanlogd3, np.log(gnn2.meand3))

    # With no randoms, calculateZeta just returns zeta.
    np.testing.assert_allclose(nng.calculateZeta()[0], true_zeta_sum3, rtol=1.e-5)
    np.testing.assert_allclose(ngn.calculateZeta()[0], true_zeta_sum2, rtol=1.e-5)
    np.testing.assert_allclose(gnn.calculateZeta()[0], true_zeta_sum1, rtol=1.e-5)

    # Compute zeta using randoms
    # We save these values to compare to later ones with patches.
    xr = rng.uniform(-3*s,3*s, (2*ngal,) )
    yr = rng.normal(-3*s,3*s, (2*ngal,) )
    rcat = treecorr.Catalog(x=xr, y=yr)

    rrg = nng.copy()
    rrg.process(rcat, rcat, cat3, ordered=3)
    diff = nng.raw_zeta - rrg.zeta
    zeta_nng_rr, _ = nng.calculateZeta(rrg=rrg)
    np.testing.assert_allclose(zeta_nng_rr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, diff, rtol=1.e-5)
    drg = nng.copy()
    drg.process(cat1, rcat, cat3, ordered=3)
    diff = nng.raw_zeta - 2*drg.zeta + rrg.zeta
    zeta_nng_dr, _ = nng.calculateZeta(rrg=rrg, drg=drg)
    np.testing.assert_allclose(zeta_nng_dr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, diff, rtol=1.e-5)
    rdg = nng.copy()
    rdg.process(rcat, cat2, cat3, ordered=3)
    diff = nng.raw_zeta - 2*rdg.zeta + rrg.zeta
    zeta_nng_rd, _ = nng.calculateZeta(rrg=rrg, rdg=rdg)
    np.testing.assert_allclose(nng.zeta, diff, rtol=1.e-5)
    diff = nng.raw_zeta - drg.zeta - rdg.zeta + rrg.zeta
    zeta_nng_rdr, _ = nng.calculateZeta(rrg=rrg, rdg=rdg, drg=drg)
    np.testing.assert_allclose(nng.zeta, diff, rtol=1.e-5)

    rgr = ngn.copy()
    rgr.process(rcat, cat3, rcat, ordered=2)
    diff = ngn.raw_zeta - rgr.zeta
    zeta_ngn_rr, _ = ngn.calculateZeta(rgr=rgr)
    np.testing.assert_allclose(zeta_ngn_rr, diff, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, diff, rtol=1.e-5)
    dgr = ngn.copy()
    dgr.process(cat1, cat3, rcat, ordered=2)
    diff = ngn.raw_zeta - 2*dgr.zeta + rgr.zeta
    zeta_ngn_dr, _ = ngn.calculateZeta(rgr=rgr, dgr=dgr)
    np.testing.assert_allclose(zeta_ngn_dr, diff, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, diff, rtol=1.e-5)
    rgd = ngn.copy()
    rgd.process(rcat, cat3, cat2, ordered=2)
    diff = ngn.raw_zeta - 2*rgd.zeta + rgr.zeta
    zeta_ngn_rd, _ = ngn.calculateZeta(rgr=rgr, rgd=rgd)
    np.testing.assert_allclose(zeta_ngn_rd, diff, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, diff, rtol=1.e-5)
    diff = ngn.raw_zeta - dgr.zeta - rgd.zeta + rgr.zeta
    zeta_ngn_rdr, _ = ngn.calculateZeta(rgr=rgr, rgd=rgd, dgr=dgr)
    np.testing.assert_allclose(zeta_ngn_rdr, diff, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, diff, rtol=1.e-5)

    grr = gnn.copy()
    grr.process(cat3, rcat, rcat, ordered=1)
    diff = gnn.raw_zeta - grr.zeta
    zeta_gnn_rr, _ = gnn.calculateZeta(grr=grr)
    np.testing.assert_allclose(zeta_gnn_rr, diff, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, diff, rtol=1.e-5)
    gdr = gnn.copy()
    gdr.process(cat3, cat1, rcat, ordered=1)
    diff = gnn.raw_zeta - 2*gdr.zeta + grr.zeta
    zeta_gnn_dr, _ = gnn.calculateZeta(grr=grr, gdr=gdr)
    np.testing.assert_allclose(zeta_gnn_dr, diff, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, diff, rtol=1.e-5)
    grd = gnn.copy()
    grd.process(cat3, rcat, cat2, ordered=1)
    diff = gnn.raw_zeta - 2*grd.zeta + grr.zeta
    zeta_gnn_rd, _ = gnn.calculateZeta(grr=grr, grd=grd)
    np.testing.assert_allclose(zeta_gnn_rd, diff, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, diff, rtol=1.e-5)
    diff = gnn.raw_zeta - gdr.zeta - grd.zeta + grr.zeta
    zeta_gnn_rdr, _ = gnn.calculateZeta(grr=grr, grd=grd, gdr=gdr)
    np.testing.assert_allclose(zeta_gnn_rdr, diff, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, diff, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        nng.process(cat1, cat3=cat3)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    nng.process(cat1p, cat2, cat3)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1p, cat3, cat2)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3p, cat1, cat2)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1, cat2p, cat3)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1, cat3p, cat2)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3, cat1p, cat2)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1, cat2, cat3p)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2p)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2p)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    # Now all three patched
    nng.process(cat1p, cat2p, cat3p)
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng2.process(cat1p, cat2p, cat3p, corr_only=True)
    np.testing.assert_allclose(nng2.weight, nng.weight)
    np.testing.assert_allclose(nng2.zeta, nng.zeta)
    np.testing.assert_allclose(nng2.ntri, nng.weight / (np.mean(w1)*np.mean(w2)*np.mean(w3)))
    np.testing.assert_allclose(nng2.meand2, nng.rnom)
    np.testing.assert_allclose(nng2.meanlogd2, nng.logr)
    np.testing.assert_allclose(nng2.meanu, nng.u)
    np.testing.assert_allclose(nng2.meanv, nng.v)
    np.testing.assert_allclose(nng2.meand3/nng2.meand2, nng.u)
    np.testing.assert_allclose((nng2.meand1 - nng2.meand2)/nng2.meand3, np.abs(nng.v))
    np.testing.assert_allclose(nng2.meanlogd1, np.log(nng2.meand1))
    np.testing.assert_allclose(nng2.meanlogd3, np.log(nng2.meand3))

    ngn2.process(cat1p, cat3p, cat2p, corr_only=True)
    np.testing.assert_allclose(ngn2.weight, ngn.weight)
    np.testing.assert_allclose(ngn2.zeta, ngn.zeta)
    np.testing.assert_allclose(ngn2.ntri, ngn.weight / (np.mean(w1)*np.mean(w2)*np.mean(w3)))
    np.testing.assert_allclose(ngn2.meand2, ngn.rnom)
    np.testing.assert_allclose(ngn2.meanlogd2, ngn.logr)
    np.testing.assert_allclose(ngn2.meanu, ngn.u)
    np.testing.assert_allclose(ngn2.meanv, ngn.v)
    np.testing.assert_allclose(ngn2.meand3/ngn2.meand2, ngn.u)
    np.testing.assert_allclose((ngn2.meand1 - ngn2.meand2)/ngn2.meand3, np.abs(ngn.v))
    np.testing.assert_allclose(ngn2.meanlogd1, np.log(ngn2.meand1))
    np.testing.assert_allclose(ngn2.meanlogd3, np.log(ngn2.meand3))

    gnn2.process(cat3p, cat1p, cat2p, corr_only=True)
    np.testing.assert_allclose(gnn2.weight, gnn.weight)
    np.testing.assert_allclose(gnn2.zeta, gnn.zeta)
    np.testing.assert_allclose(gnn2.ntri, gnn.weight / (np.mean(w1)*np.mean(w2)*np.mean(w3)))
    np.testing.assert_allclose(gnn2.meand2, gnn.rnom)
    np.testing.assert_allclose(gnn2.meanlogd2, gnn.logr)
    np.testing.assert_allclose(gnn2.meanu, gnn.u)
    np.testing.assert_allclose(gnn2.meanv, gnn.v)
    np.testing.assert_allclose(gnn2.meand3/gnn2.meand2, gnn.u)
    np.testing.assert_allclose((gnn2.meand1 - gnn2.meand2)/gnn2.meand3, np.abs(gnn.v))
    np.testing.assert_allclose(gnn2.meanlogd1, np.log(gnn2.meand1))
    np.testing.assert_allclose(gnn2.meanlogd3, np.log(gnn2.meand3))

    # Unordered
    nng.process(cat1p, cat2p, cat3p, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=False)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=3)
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=2)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=1)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check using randoms with and without patches
    rcatp = treecorr.Catalog(x=xr, y=yr, patch_centers=cat1p.patch_centers)
    rrg.process(rcat, rcat, cat3p, ordered=False)
    nng.calculateZeta(rrg=rrg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_rr, rtol=1.e-5)
    rrg.process(rcatp, rcatp, cat3p, ordered=False)
    nng.calculateZeta(rrg=rrg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_rr, rtol=1.e-5)
    drg.process(cat1p, rcat, cat3p, ordered=False)
    nng.calculateZeta(rrg=rrg, drg=drg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_dr, rtol=1.e-5)
    drg.process(cat1p, rcatp, cat3p, ordered=False)
    nng.calculateZeta(rrg=rrg, drg=drg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_dr, rtol=1.e-5)
    rdg.process(rcat, cat2p, cat3p, ordered=False)
    nng.calculateZeta(rrg=rrg, rdg=rdg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_rd, rtol=1.e-5)
    rdg.process(rcatp, cat2p, cat3p, ordered=False)
    nng.calculateZeta(rrg=rrg, rdg=rdg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_rd, rtol=1.e-5)
    nng.calculateZeta(rrg=rrg, rdg=rdg, drg=drg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_rdr, rtol=1.e-5)
    nng.calculateZeta(rrg=rrg, rdg=rdg, drg=drg)
    nng.estimate_cov('jackknife')
    np.testing.assert_allclose(nng.zeta, zeta_nng_rdr, rtol=1.e-5)

    rgr.process(rcat, cat3p, rcat, ordered=False)
    ngn.calculateZeta(rgr=rgr)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rr, rtol=1.e-5)
    rgr.process(rcatp, cat3p, rcatp, ordered=False)
    ngn.calculateZeta(rgr=rgr)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rr, rtol=1.e-5)
    dgr.process(cat1p, cat3p, rcat, ordered=False)
    ngn.calculateZeta(rgr=rgr, dgr=dgr)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_dr, rtol=1.e-5)
    dgr.process(cat1p, cat3p, rcatp, ordered=False)
    ngn.calculateZeta(rgr=rgr, dgr=dgr)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_dr, rtol=1.e-5)
    rgd.process(rcat, cat3p, cat2p, ordered=False)
    ngn.calculateZeta(rgr=rgr, rgd=rgd)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rd, rtol=1.e-5)
    rgd.process(rcatp, cat3p, cat2p, ordered=False)
    ngn.calculateZeta(rgr=rgr, rgd=rgd)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rd, rtol=1.e-5)
    ngn.calculateZeta(rgr=rgr, rgd=rgd, dgr=dgr)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rdr, rtol=1.e-5)
    ngn.calculateZeta(rgr=rgr, rgd=rgd, dgr=dgr)
    ngn.estimate_cov('jackknife')
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rdr, rtol=1.e-5)

    grr.process(cat3p, rcat, rcat, ordered=False)
    gnn.calculateZeta(grr=grr)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rr, rtol=1.e-5)
    grr.process(cat3p, rcatp, rcatp, ordered=False)
    gnn.calculateZeta(grr=grr)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rr, rtol=1.e-5)
    gdr.process(cat3p, cat1p, rcat, ordered=False)
    gnn.calculateZeta(grr=grr, gdr=gdr)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_dr, rtol=1.e-5)
    gdr.process(cat3p, cat1p, rcatp, ordered=False)
    gnn.calculateZeta(grr=grr, gdr=gdr)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_dr, rtol=1.e-5)
    grd.process(cat3p, rcat, cat2p, ordered=False)
    gnn.calculateZeta(grr=grr, grd=grd)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rd, rtol=1.e-5)
    grd.process(cat3p, rcatp, cat2p, ordered=False)
    gnn.calculateZeta(grr=grr, grd=grd)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rd, rtol=1.e-5)
    gnn.calculateZeta(grr=grr, grd=grd, gdr=gdr)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rdr, rtol=1.e-5)
    gnn.calculateZeta(grr=grr, grd=grd, gdr=gdr)
    gnn.estimate_cov('jackknife')
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rdr, rtol=1.e-5)

    # Check when some patches have no objects
    rcatpx = treecorr.Catalog(x=xr, y=yr, npatch=20, rng=rng)
    cat1px = treecorr.Catalog(x=x1, y=y1, w=w1, patch_centers=rcatpx.patch_centers)
    cat2px = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=rcatpx.patch_centers)
    cat3px = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3,
                              patch_centers=rcatpx.patch_centers)
    nng.process(cat1px, cat2px, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        nng.calculateZeta(rrg=rrg)
    rrg.process(rcatpx, rcatpx, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        nng.calculateZeta(rrg=rrg, rdg=rdg)
    with assert_raises(RuntimeError):
        nng.calculateZeta(rrg=rrg, drg=drg)
    rdg.process(rcatpx, cat2px, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        nng.calculateZeta(rrg=rrg, drg=drg, rdg=rdg)
    drg.process(cat1px, rcatpx, cat3px, ordered=False)
    with assert_raises(TypeError):
        nng.calculateZeta(drg=drg)
    with assert_raises(TypeError):
        nng.calculateZeta(rdg=rdg)
    nng.calculateZeta(rrg=rrg)
    np.testing.assert_allclose(nng.zeta, zeta_nng_rr, rtol=1.e-5)
    nng.calculateZeta(rrg=rrg, rdg=rdg)
    np.testing.assert_allclose(nng.zeta, zeta_nng_rd, rtol=1.e-5)
    nng.calculateZeta(rrg=rrg, drg=drg)
    np.testing.assert_allclose(nng.zeta, zeta_nng_dr, rtol=1.e-5)
    nng.calculateZeta(rrg=rrg, drg=drg, rdg=rdg)
    np.testing.assert_allclose(nng.zeta, zeta_nng_rdr, rtol=1.e-5)

    ngn.process(cat1px, cat3px, cat2px, ordered=False)
    with assert_raises(RuntimeError):
        ngn.calculateZeta(rgr=rgr)
    rgr.process(rcatpx, cat3px, rcatpx, ordered=False)
    with assert_raises(RuntimeError):
        ngn.calculateZeta(rgr=rgr, rgd=rgd)
    with assert_raises(RuntimeError):
        ngn.calculateZeta(rgr=rgr, dgr=dgr)
    rgd.process(rcatpx, cat3px, cat2px, ordered=False)
    with assert_raises(RuntimeError):
        ngn.calculateZeta(rgr=rgr, dgr=dgr, rgd=rgd)
    dgr.process(cat1px, cat3px, rcatpx, ordered=False)
    with assert_raises(TypeError):
        ngn.calculateZeta(dgr=dgr)
    with assert_raises(TypeError):
        ngn.calculateZeta(rgd=rgd)
    ngn.calculateZeta(rgr=rgr)
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rr, rtol=1.e-5)
    ngn.calculateZeta(rgr=rgr, dgr=dgr)
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_dr, rtol=1.e-5)
    ngn.calculateZeta(rgr=rgr, rgd=rgd)
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rd, rtol=1.e-5)
    ngn.calculateZeta(rgr=rgr, dgr=dgr, rgd=rgd)
    np.testing.assert_allclose(ngn.zeta, zeta_ngn_rdr, rtol=1.e-5)

    gnn.process(cat3px, cat1px, cat2px, ordered=False)
    with assert_raises(RuntimeError):
        gnn.calculateZeta(grr=grr)
    grr.process(cat3px, rcatpx, rcatpx, ordered=False)
    with assert_raises(RuntimeError):
        gnn.calculateZeta(grr=grr, gdr=gdr)
    with assert_raises(RuntimeError):
        gnn.calculateZeta(grr=grr, grd=grd)
    gdr.process(cat3px, cat1px, rcatpx, ordered=False)
    with assert_raises(RuntimeError):
        gnn.calculateZeta(grr=grr, gdr=gdr, grd=grd)
    grd.process(cat3px, rcatpx, cat2px, ordered=False)
    with assert_raises(TypeError):
        gnn.calculateZeta(gdr=gdr)
    with assert_raises(TypeError):
        gnn.calculateZeta(grd=grd)
    gnn.calculateZeta(grr=grr)
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rr, rtol=1.e-5)
    gnn.calculateZeta(grr=grr, gdr=gdr)
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_dr, rtol=1.e-5)
    gnn.calculateZeta(grr=grr, grd=grd)
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rd, rtol=1.e-5)
    gnn.calculateZeta(grr=grr, gdr=gdr, grd=grd)
    np.testing.assert_allclose(gnn.zeta, zeta_gnn_rdr, rtol=1.e-5)

    # patch_method=local
    nng.process(cat1p, cat2p, cat3p, patch_method='local')
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, patch_method='local')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, patch_method='local')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=3, patch_method='local')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=2, patch_method='local')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=1, patch_method='local')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    with assert_raises(ValueError):
        nng.process(cat1p, cat2p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        ngn.process(cat1p, cat3p, cat2p, patch_method='nonlocal')
    with assert_raises(ValueError):
        gnn.process(cat3p, cat1p, cat2p, patch_method='nonlocal')


@timer
def test_direct_logruv_cross21():
    # Check the 2-1 cross correlation

    ngal = 50
    s = 10.
    sig_gam = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    g1_1 = rng.normal(0,sig_gam, (ngal,) )
    g2_1 = rng.normal(0,sig_gam, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    nng = treecorr.NNGCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    ngn = treecorr.NGNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    gnn = treecorr.GNNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    true_ntri_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_221 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_122 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_212 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
    true_zeta_221 = np.zeros((nrbins, nubins, 2*nvbins), dtype=complex)
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

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)

                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2

                if dij < dik:
                    if dik < djk:
                        d3 = dij; d2 = dik; d1 = djk
                        true_ntri = true_ntri_122
                        true_zeta = true_zeta_122
                        true_weight = true_weight_122
                    elif dij < djk:
                        d3 = dij; d2 = djk; d1 = dik
                        true_ntri = true_ntri_212
                        true_zeta = true_zeta_212
                        true_weight = true_weight_212
                        ccw = not ccw
                    else:
                        d3 = djk; d2 = dij; d1 = dik
                        true_ntri = true_ntri_221
                        true_zeta = true_zeta_221
                        true_weight = true_weight_221
                else:
                    if dij < djk:
                        d3 = dik; d2 = dij; d1 = djk
                        true_ntri = true_ntri_122
                        true_zeta = true_zeta_122
                        true_weight = true_weight_122
                        ccw = not ccw
                    elif dik < djk:
                        d3 = dik; d2 = djk; d1 = dij
                        true_ntri = true_ntri_212
                        true_zeta = true_zeta_212
                        true_weight = true_weight_212
                    else:
                        d3 = djk; d2 = dik; d1 = dij
                        true_ntri = true_ntri_221
                        true_zeta = true_zeta_221
                        true_weight = true_weight_221
                        ccw = not ccw

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

                www = w1[i] * w2[j] * w2[k]
                zeta = www * g1p

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    pos = true_weight_221 > 0
    true_zeta_221[pos] /= true_weight_221[pos]
    pos = true_weight_212 > 0
    true_zeta_212[pos] /= true_weight_212[pos]
    pos = true_weight_122 > 0
    true_zeta_122[pos] /= true_weight_122[pos]

    nng.process(cat2, cat2, cat1)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1, cat2, cat2)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: NGN doesn't have a two-argument version.
    nng.process(cat2, cat1)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1, cat2)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    with assert_raises(ValueError):
        ngn.process(cat2, cat1)
    with assert_raises(ValueError):
        ngn.process(cat1, cat2)
    with assert_raises(ValueError):
        nng.process(cat1)
    with assert_raises(ValueError):
        nng.process(cat2)
    with assert_raises(ValueError):
        ngn.process(cat1)
    with assert_raises(ValueError):
        ngn.process(cat2)
    with assert_raises(ValueError):
        gnn.process(cat1)
    with assert_raises(ValueError):
        gnn.process(cat2)

    # ordered=False doesn't do anything different, since there is no other valid order.
    nng.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with binslop = 0
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')

    nng.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    # And again with no top-level recursion
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nng.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng2 = nng.copy()
    nng2.process(cat2, cat1, corr_only=True)
    np.testing.assert_allclose(nng2.weight, nng.weight)
    np.testing.assert_allclose(nng2.zeta, nng.zeta)
    np.testing.assert_allclose(nng2.ntri, nng.weight / (np.mean(w1)*np.mean(w2)**2))
    np.testing.assert_allclose(nng2.meand2, nng.rnom)
    np.testing.assert_allclose(nng2.meanlogd2, nng.logr)
    np.testing.assert_allclose(nng2.meanu, nng.u)
    np.testing.assert_allclose(nng2.meanv, nng.v)
    np.testing.assert_allclose(nng2.meand3/nng2.meand2, nng.u)
    np.testing.assert_allclose((nng2.meand1 - nng2.meand2)/nng2.meand3, np.abs(nng.v))
    np.testing.assert_allclose(nng2.meanlogd1, np.log(nng2.meand1))
    np.testing.assert_allclose(nng2.meanlogd3, np.log(nng2.meand3))

    ngn2 = ngn.copy()
    ngn2.process(cat2, cat1, cat2, corr_only=True)
    np.testing.assert_allclose(ngn2.weight, ngn.weight)
    np.testing.assert_allclose(ngn2.zeta, ngn.zeta)
    np.testing.assert_allclose(ngn2.ntri, ngn.weight / (np.mean(w1)*np.mean(w2)**2))
    np.testing.assert_allclose(ngn2.meand2, ngn.rnom)
    np.testing.assert_allclose(ngn2.meanlogd2, ngn.logr)
    np.testing.assert_allclose(ngn2.meanu, ngn.u)
    np.testing.assert_allclose(ngn2.meanv, ngn.v)
    np.testing.assert_allclose(ngn2.meand3/ngn2.meand2, ngn.u)
    np.testing.assert_allclose((ngn2.meand1 - ngn2.meand2)/ngn2.meand3, np.abs(ngn.v))
    np.testing.assert_allclose(ngn2.meanlogd1, np.log(ngn2.meand1))
    np.testing.assert_allclose(ngn2.meanlogd3, np.log(ngn2.meand3))

    gnn2 = gnn.copy()
    gnn2.process(cat1, cat2, corr_only=True)
    np.testing.assert_allclose(gnn2.weight, gnn.weight)
    np.testing.assert_allclose(gnn2.zeta, gnn.zeta)
    np.testing.assert_allclose(gnn2.ntri, gnn.weight / (np.mean(w1)*np.mean(w2)**2))
    np.testing.assert_allclose(gnn2.meand2, gnn.rnom)
    np.testing.assert_allclose(gnn2.meanlogd2, gnn.logr)
    np.testing.assert_allclose(gnn2.meanu, gnn.u)
    np.testing.assert_allclose(gnn2.meanv, gnn.v)
    np.testing.assert_allclose(gnn2.meand3/gnn2.meand2, gnn.u)
    np.testing.assert_allclose((gnn2.meand1 - gnn2.meand2)/gnn2.meand3, np.abs(gnn.v))
    np.testing.assert_allclose(gnn2.meanlogd1, np.log(gnn2.meand1))
    np.testing.assert_allclose(gnn2.meanlogd3, np.log(gnn2.meand3))

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)

    nng.process(cat2p, cat1p, ordered=True)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2p, cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2p, cat1p, ordered=False)
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2p, cat1p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2p, cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng2.process(cat2p, cat1p, corr_only=True)
    np.testing.assert_allclose(nng2.weight, nng.weight)
    np.testing.assert_allclose(nng2.zeta, nng.zeta)
    np.testing.assert_allclose(nng2.ntri, nng.weight / (np.mean(w1)*np.mean(w2)**2))
    np.testing.assert_allclose(nng2.meand2, nng.rnom)
    np.testing.assert_allclose(nng2.meanlogd2, nng.logr)
    np.testing.assert_allclose(nng2.meanu, nng.u)
    np.testing.assert_allclose(nng2.meanv, nng.v)
    np.testing.assert_allclose(nng2.meand3/nng2.meand2, nng.u)
    np.testing.assert_allclose((nng2.meand1 - nng2.meand2)/nng2.meand3, np.abs(nng.v))
    np.testing.assert_allclose(nng2.meanlogd1, np.log(nng2.meand1))
    np.testing.assert_allclose(nng2.meanlogd3, np.log(nng2.meand3))

    ngn2.process(cat2p, cat1p, cat2p, corr_only=True)
    np.testing.assert_allclose(ngn2.weight, ngn.weight)
    np.testing.assert_allclose(ngn2.zeta, ngn.zeta)
    np.testing.assert_allclose(ngn2.ntri, ngn.weight / (np.mean(w1)*np.mean(w2)**2))
    np.testing.assert_allclose(ngn2.meand2, ngn.rnom)
    np.testing.assert_allclose(ngn2.meanlogd2, ngn.logr)
    np.testing.assert_allclose(ngn2.meanu, ngn.u)
    np.testing.assert_allclose(ngn2.meanv, ngn.v)
    np.testing.assert_allclose(ngn2.meand3/ngn2.meand2, ngn.u)
    np.testing.assert_allclose((ngn2.meand1 - ngn2.meand2)/ngn2.meand3, np.abs(ngn.v))
    np.testing.assert_allclose(ngn2.meanlogd1, np.log(ngn2.meand1))
    np.testing.assert_allclose(ngn2.meanlogd3, np.log(ngn2.meand3))

    gnn2.process(cat1p, cat2p, corr_only=True)
    np.testing.assert_allclose(gnn2.weight, gnn.weight)
    np.testing.assert_allclose(gnn2.zeta, gnn.zeta)
    np.testing.assert_allclose(gnn2.ntri, gnn.weight / (np.mean(w1)*np.mean(w2)**2))
    np.testing.assert_allclose(gnn2.meand2, gnn.rnom)
    np.testing.assert_allclose(gnn2.meanlogd2, gnn.logr)
    np.testing.assert_allclose(gnn2.meanu, gnn.u)
    np.testing.assert_allclose(gnn2.meanv, gnn.v)
    np.testing.assert_allclose(gnn2.meand3/gnn2.meand2, gnn.u)
    np.testing.assert_allclose((gnn2.meand1 - gnn2.meand2)/gnn2.meand3, np.abs(gnn.v))
    np.testing.assert_allclose(gnn2.meanlogd1, np.log(gnn2.meand1))
    np.testing.assert_allclose(gnn2.meanlogd3, np.log(gnn2.meand3))


@timer
def test_varzeta_logruv():
    # Test that the shot noise estimate of varzeta is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675309)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 30
    nsource = 100000

    file_name = 'data/test_varzeta_nng_logruv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nngs = []
        all_ngns = []
        all_gnns = []

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

            ncat = treecorr.Catalog(x=x1, y=y1, x_units='arcmin', y_units='arcmin')
            gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2,
                                    x_units='arcmin', y_units='arcmin')
            nng = treecorr.NNGCorrelation(nbins=2, min_sep=80., max_sep=100.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          nubins=2, nvbins=2,
                                          sep_units='arcmin', bin_type='LogRUV')
            ngn = treecorr.NGNCorrelation(nbins=2, min_sep=80., max_sep=100.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          nubins=2, nvbins=2,
                                          sep_units='arcmin', bin_type='LogRUV')
            gnn = treecorr.GNNCorrelation(nbins=2, min_sep=80., max_sep=100.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          nubins=2, nvbins=2,
                                          sep_units='arcmin', bin_type='LogRUV')
            nng.process(ncat, gcat)
            ngn.process(ncat, gcat, ncat)
            gnn.process(gcat, ncat)
            all_nngs.append(nng)
            all_ngns.append(ngn)
            all_gnns.append(gnn)

        mean_nng_zetar = np.mean([nng.zetar for nng in all_nngs], axis=0)
        mean_nng_zetai = np.mean([nng.zetai for nng in all_nngs], axis=0)
        var_nng_zetar = np.var([nng.zetar for nng in all_nngs], axis=0)
        var_nng_zetai = np.var([nng.zetai for nng in all_nngs], axis=0)
        mean_nng_varzeta = np.mean([nng.varzeta for nng in all_nngs], axis=0)
        mean_ngn_zetar = np.mean([ngn.zetar for ngn in all_ngns], axis=0)
        mean_ngn_zetai = np.mean([ngn.zetai for ngn in all_ngns], axis=0)
        var_ngn_zetar = np.var([ngn.zetar for ngn in all_ngns], axis=0)
        var_ngn_zetai = np.var([ngn.zetai for ngn in all_ngns], axis=0)
        mean_ngn_varzeta = np.mean([ngn.varzeta for ngn in all_ngns], axis=0)
        mean_gnn_zetar = np.mean([gnn.zetar for gnn in all_gnns], axis=0)
        mean_gnn_zetai = np.mean([gnn.zetai for gnn in all_gnns], axis=0)
        var_gnn_zetar = np.var([gnn.zetar for gnn in all_gnns], axis=0)
        var_gnn_zetai = np.var([gnn.zetai for gnn in all_gnns], axis=0)
        mean_gnn_varzeta = np.mean([gnn.varzeta for gnn in all_gnns], axis=0)

        np.savez(file_name,
                 mean_nng_zetar=mean_nng_zetar,
                 mean_nng_zetai=mean_nng_zetai,
                 var_nng_zetar=var_nng_zetar,
                 var_nng_zetai=var_nng_zetai,
                 mean_nng_varzeta=mean_nng_varzeta,
                 mean_ngn_zetar=mean_ngn_zetar,
                 mean_ngn_zetai=mean_ngn_zetai,
                 var_ngn_zetar=var_ngn_zetar,
                 var_ngn_zetai=var_ngn_zetai,
                 mean_ngn_varzeta=mean_ngn_varzeta,
                 mean_gnn_zetar=mean_gnn_zetar,
                 mean_gnn_zetai=mean_gnn_zetai,
                 var_gnn_zetar=var_gnn_zetar,
                 var_gnn_zetai=var_gnn_zetai,
                 mean_gnn_varzeta=mean_gnn_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_nng_zetar = data['mean_nng_zetar']
    mean_nng_zetai = data['mean_nng_zetai']
    var_nng_zetar = data['var_nng_zetar']
    var_nng_zetai = data['var_nng_zetai']
    mean_nng_varzeta = data['mean_nng_varzeta']
    mean_ngn_zetar = data['mean_ngn_zetar']
    mean_ngn_zetai = data['mean_ngn_zetai']
    var_ngn_zetar = data['var_ngn_zetar']
    var_ngn_zetai = data['var_ngn_zetai']
    mean_ngn_varzeta = data['mean_ngn_varzeta']
    mean_gnn_zetar = data['mean_gnn_zetar']
    mean_gnn_zetai = data['mean_gnn_zetai']
    var_gnn_zetar = data['var_gnn_zetar']
    var_gnn_zetai = data['var_gnn_zetai']
    mean_gnn_varzeta = data['mean_gnn_varzeta']

    print('var_nng_zetar = ',var_nng_zetar)
    print('mean nng_varzeta = ',mean_nng_varzeta)
    print('ratio = ',var_nng_zetar.ravel() / mean_nng_varzeta.ravel())
    print('var_ngn_zetar = ',var_ngn_zetar)
    print('mean ngn_varzeta = ',mean_ngn_varzeta)
    print('ratio = ',var_ngn_zetar.ravel() / mean_ngn_varzeta.ravel())
    print('var_gnn_zetar = ',var_gnn_zetar)
    print('mean gnn_varzeta = ',mean_gnn_varzeta)
    print('ratio = ',var_gnn_zetar.ravel() / mean_gnn_varzeta.ravel())

    print('max relerr for nng zetar = ',
          np.max(np.abs((var_nng_zetar - mean_nng_varzeta)/var_nng_zetar)))
    print('max relerr for nng zetai = ',
          np.max(np.abs((var_nng_zetai - mean_nng_varzeta)/var_nng_zetai)))
    np.testing.assert_allclose(mean_nng_varzeta, var_nng_zetar, rtol=0.15)
    np.testing.assert_allclose(mean_nng_varzeta, var_nng_zetai, rtol=0.15)

    print('max relerr for ngn zetar = ',
          np.max(np.abs((var_ngn_zetar - mean_ngn_varzeta)/var_ngn_zetar)))
    print('max relerr for ngn zetai = ',
          np.max(np.abs((var_ngn_zetai - mean_ngn_varzeta)/var_ngn_zetai)))
    np.testing.assert_allclose(mean_ngn_varzeta, var_ngn_zetar, rtol=0.15)
    np.testing.assert_allclose(mean_ngn_varzeta, var_ngn_zetai, rtol=0.15)

    print('max relerr for gnn zetar = ',
          np.max(np.abs((var_gnn_zetar - mean_gnn_varzeta)/var_gnn_zetar)))
    print('max relerr for gnn zetai = ',
          np.max(np.abs((var_gnn_zetai - mean_gnn_varzeta)/var_gnn_zetai)))
    np.testing.assert_allclose(mean_gnn_varzeta, var_gnn_zetar, rtol=0.20)
    np.testing.assert_allclose(mean_gnn_varzeta, var_gnn_zetai, rtol=0.20)

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

    ncat = treecorr.Catalog(x=x1, y=y1, x_units='arcmin', y_units='arcmin')
    gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2, x_units='arcmin', y_units='arcmin')
    nng = treecorr.NNGCorrelation(nbins=2, min_sep=80., max_sep=100.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  nubins=2, nvbins=2,
                                  sep_units='arcmin', bin_type='LogRUV')
    ngn = treecorr.NGNCorrelation(nbins=2, min_sep=80., max_sep=100.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  nubins=2, nvbins=2,
                                  sep_units='arcmin', bin_type='LogRUV')
    gnn = treecorr.GNNCorrelation(nbins=2, min_sep=80., max_sep=100.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  nubins=2, nvbins=2,
                                  sep_units='arcmin', bin_type='LogRUV')

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(nng.cov, 0)
    np.testing.assert_array_equal(nng.varzeta, 0)
    np.testing.assert_array_equal(ngn.cov, 0)
    np.testing.assert_array_equal(ngn.varzeta, 0)
    np.testing.assert_array_equal(gnn.cov, 0)
    np.testing.assert_array_equal(gnn.varzeta, 0)

    nng.process(ncat, gcat)
    print('NNG single run:')
    print('max relerr for zetar = ',np.max(np.abs((nng.varzeta - var_nng_zetar)/var_nng_zetar)))
    print('ratio = ',nng.varzeta / var_nng_zetar)
    print('max relerr for zetai = ',np.max(np.abs((nng.varzeta - var_nng_zetai)/var_nng_zetai)))
    print('ratio = ',nng.varzeta / var_nng_zetai)
    print('var_num = ',nng._var_num)
    print('ntri = ',nng.ntri)
    np.testing.assert_allclose(nng.varzeta, var_nng_zetar, rtol=0.7)
    np.testing.assert_allclose(nng.varzeta, var_nng_zetai, rtol=0.7)
    np.testing.assert_allclose(nng.cov.diagonal(), nng.varzeta.ravel())

    ngn.process(ncat, gcat, ncat)
    print('NGN single run:')
    print('max relerr for zetar = ',np.max(np.abs((ngn.varzeta - var_ngn_zetar)/var_ngn_zetar)))
    print('ratio = ',ngn.varzeta / var_ngn_zetar)
    print('max relerr for zetai = ',np.max(np.abs((ngn.varzeta - var_ngn_zetai)/var_ngn_zetai)))
    print('ratio = ',ngn.varzeta / var_ngn_zetai)
    np.testing.assert_allclose(ngn.varzeta, var_ngn_zetar, rtol=0.7)
    np.testing.assert_allclose(ngn.varzeta, var_ngn_zetai, rtol=0.7)
    np.testing.assert_allclose(ngn.cov.diagonal(), ngn.varzeta.ravel())

    gnn.process(gcat, ncat)
    print('GNN single run:')
    print('max relerr for zetar = ',np.max(np.abs((gnn.varzeta - var_gnn_zetar)/var_gnn_zetar)))
    print('ratio = ',gnn.varzeta / var_gnn_zetar)
    print('max relerr for zetai = ',np.max(np.abs((gnn.varzeta - var_gnn_zetai)/var_gnn_zetai)))
    print('ratio = ',gnn.varzeta / var_gnn_zetai)
    np.testing.assert_allclose(gnn.varzeta, var_gnn_zetar, rtol=0.7)
    np.testing.assert_allclose(gnn.varzeta, var_gnn_zetai, rtol=0.7)
    np.testing.assert_allclose(gnn.cov.diagonal(), gnn.varzeta.ravel())



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
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)
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
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_132 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_213 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_231 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_312 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_321 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_123 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_132 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_213 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_231 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_312 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
    true_zeta_321 = np.zeros((nbins, nbins, nphi_bins), dtype=complex )
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

                expmialpha3 = (x3[k]-cenx) - 1j*(y3[k]-ceny)
                expmialpha3 /= abs(expmialpha3)

                www = w1[i] * w2[j] * w3[k]
                g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                zeta = www * g3p

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
                        true_zeta_123[kr2,kr3,kphi] += zeta

                    # 132
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_132[kr3,kr2,kphi] += 1
                        true_weight_132[kr3,kr2,kphi] += www
                        true_zeta_132[kr3,kr2,kphi] += zeta

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
                        true_zeta_231[kr3,kr1,kphi] += zeta

                    # 213
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_213[kr1,kr3,kphi] += 1
                        true_weight_213[kr1,kr3,kphi] += www
                        true_zeta_213[kr1,kr3,kphi] += zeta

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
                        true_zeta_312[kr1,kr2,kphi] += zeta

                    # 321
                    phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_321[kr2,kr1,kphi] += 1
                        true_weight_321[kr2,kr1,kphi] += www
                        true_zeta_321[kr2,kr1,kphi] += zeta

    true_ntri_sum3 = true_ntri_123 + true_ntri_213
    true_weight_sum3 = true_weight_123 + true_weight_213
    true_zeta_sum3 = true_zeta_123 + true_zeta_213
    true_ntri_sum2 = true_ntri_132 + true_ntri_231
    true_weight_sum2 = true_weight_132 + true_weight_231
    true_zeta_sum2 = true_zeta_132 + true_zeta_231
    true_ntri_sum1 = true_ntri_312 + true_ntri_321
    true_weight_sum1 = true_weight_312 + true_weight_321
    true_zeta_sum1 = true_zeta_312 + true_zeta_321
    pos = true_weight_sum1 > 0
    true_zeta_sum1[pos] /= true_weight_sum1[pos]
    pos = true_weight_sum2 > 0
    true_zeta_sum2[pos] /= true_weight_sum2[pos]
    pos = true_weight_sum3 > 0
    true_zeta_sum3[pos] /= true_weight_sum3[pos]

    # Now normalize each one individually.
    w_list = [true_weight_123, true_weight_132, true_weight_213, true_weight_231,
              true_weight_312, true_weight_321]
    z_list = [true_zeta_123, true_zeta_132, true_zeta_213, true_zeta_231,
              true_zeta_312, true_zeta_321]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    nng.process(cat1, cat2, cat3, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)

    nng.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_213)
    np.testing.assert_allclose(nng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_213, rtol=1.e-5)

    ngn.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)

    ngn.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_231)
    np.testing.assert_allclose(ngn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_231, rtol=1.e-5)

    gnn.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    gnn.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_321)
    np.testing.assert_allclose(gnn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where G is in 3
    nng.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-4)

    ngn.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)

    gnn.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check binslop = 0
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    nng.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # And again with no top-level recursion
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    nng.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng.process(cat1, cat2, cat3, ordered=3, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-4)
    ngn.process(cat1, cat3, cat2, ordered=2, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3, cat1, cat2, ordered=1, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        nng.process(cat1, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        ngn.process(cat1, cat3=cat1, algo='triangle')
    with assert_raises(ValueError):
        gnn.process(cat3, cat3=cat1, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1p.patch_centers)

    nng.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=3, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=2, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=1, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_123)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_132)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_312)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    nng.process(cat1p, cat2p, cat3p, ordered=3, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nng.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_sum3, rtol=1.e-5)
    ngn.process(cat1p, cat3p, cat2p, ordered=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(ngn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_sum2, rtol=1.e-5)
    gnn.process(cat3p, cat1p, cat2p, ordered=1, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(gnn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_sum1, rtol=1.e-5)

    with assert_raises(ValueError):
        nng.process(cat1p, cat2p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        ngn.process(cat1p, cat3p, cat2p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        gnn.process(cat3p, cat1p, cat2p, patch_method='nonlocal', algo='triangle')


@timer
def test_direct_logsas_cross21():
    # Check the 2-1 cross correlation

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    g1_1 = rng.normal(0,0.2, (ngal,) )
    g2_1 = rng.normal(0,0.2, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 7

    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_122 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_212 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_221 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_122 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_zeta_212 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
    true_zeta_221 = np.zeros((nbins, nbins, nphi_bins), dtype=complex)
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

                expmialpha1 = (x1[i]-cenx) - 1j*(y1[i]-ceny)
                expmialpha1 /= abs(expmialpha1)

                www = w1[i] * w2[j] * w2[k]
                g1p = (g1_1[i] + 1j*g2_1[i]) * expmialpha1**2
                zeta = www * g1p

                # 123
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_122[kr2,kr3,kphi] += 1
                        true_weight_122[kr2,kr3,kphi] += www
                        true_zeta_122[kr2,kr3,kphi] += zeta

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
                        true_zeta_221[kr3,kr1,kphi] += zeta

                # 312
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x2[j],y2[j]):
                        phi = 2*np.pi - phi
                    if phi >= 0 and phi < np.pi:
                        kphi = int(np.floor( phi / phi_bin_size ))
                        assert 0 <= kphi < nphi_bins
                        true_ntri_212[kr1,kr2,kphi] += 1
                        true_weight_212[kr1,kr2,kphi] += www
                        true_zeta_212[kr1,kr2,kphi] += zeta

    w_list = [true_weight_122, true_weight_212, true_weight_221]
    z_list = [true_zeta_122, true_zeta_212, true_zeta_221]
    for w,z in zip(w_list, z_list):
        pos = w > 0
        z[pos] /= w[pos]

    nng.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)
    nng.process(cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    ngn.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)

    gnn.process(cat1, cat2, cat2, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    gnn.process(cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)

    with assert_raises(ValueError):
        ngn.process(cat2, cat1)
    with assert_raises(ValueError):
        ngn.process(cat1, cat2)
    with assert_raises(ValueError):
        nng.process(cat1)
    with assert_raises(ValueError):
        nng.process(cat2)
    with assert_raises(ValueError):
        ngn.process(cat1)
    with assert_raises(ValueError):
        ngn.process(cat2)
    with assert_raises(ValueError):
        gnn.process(cat1)
    with assert_raises(ValueError):
        gnn.process(cat2)

    # With ordered=False, doesn't do anything different, since there is no other valid order.
    nng.process(cat2, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)

    gnn.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, g1=g1_1, g2=g2_1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)

    nng.process(cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    ngn.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(ngn.ntri, true_ntri_212)
    np.testing.assert_allclose(ngn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_212, rtol=1.e-5)
    gnn.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)

    nng.process(cat2p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nng.ntri, true_ntri_221)
    np.testing.assert_allclose(nng.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_221, rtol=1.e-5)
    gnn.process(cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(gnn.ntri, true_ntri_122)
    np.testing.assert_allclose(gnn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_122, rtol=1.e-5)


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
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)
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

    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_zeta_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_123 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_123 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_132 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_132 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_213 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_213 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_231 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_231 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_312 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_312 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_321 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
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
                    # g3 is projected to the line from c1 to c3.
                    expmialpha3 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                    zeta = www * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi
                    true_zeta_123[kr2,kr3,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_123[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_123[kr2,kr3,:] += 1
                    true_zeta_132[kr3,kr2,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_132[kr3,kr2,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_132[kr3,kr2,:] += 1

                # 213, 231
                if d1 >= min_sep and d1 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c2 to c3.
                    expmialpha3 = (x3[k]-x2[j]) - 1j*(y3[k]-y2[j])
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                    zeta = www * g3p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d1**2 + d3**2 - d2**2)/(2*d1*d3))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi
                    true_zeta_231[kr3,kr1,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_231[kr3,kr1,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_231[kr3,kr1,:] += 1
                    true_zeta_213[kr1,kr3,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_213[kr1,kr3,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_213[kr1,kr3,:] += 1

                # 312, 321
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g3 is projected to the average of lines from c3 to c1 and c3 to c2.
                    expmialpha1 = (x3[k]-x1[i]) - 1j*(y3[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x3[k]-x2[j]) - 1j*(y3[k]-y2[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_3[k] + 1j*g2_3[k]) * expmialpha3**2
                    zeta = www * g3p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x3[k],y3[k],x2[j],y2[j]):
                        phi = -phi
                    true_zeta_312[kr1,kr2,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_312[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_312[kr1,kr2,:] += 1
                    true_zeta_321[kr2,kr1,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_321[kr2,kr1,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_321[kr2,kr1,:] += 1

    nng.process(cat1, cat2, cat3)
    np.testing.assert_allclose(nng.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-4)
    nng.process(cat2, cat1, cat3)
    np.testing.assert_allclose(nng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_213, rtol=1.e-4)
    ngn.process(cat1, cat3, cat2)
    np.testing.assert_allclose(ngn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-4)
    ngn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(ngn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_231, rtol=1.e-4)
    gnn.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gnn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-4)
    gnn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(gnn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_321, rtol=1.e-4)

    # Repeat with binslop = 0
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nng.process(cat1, cat2, cat3)
    np.testing.assert_allclose(nng.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-4)
    nng.process(cat2, cat1, cat3)
    np.testing.assert_allclose(nng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_213, rtol=1.e-4)

    ngn.process(cat1, cat3, cat2)
    np.testing.assert_allclose(ngn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-4)
    ngn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(ngn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_231, rtol=1.e-4)

    gnn.process(cat3, cat1, cat2)
    np.testing.assert_allclose(gnn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-4)
    gnn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(gnn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_321, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, g1=g1_3, g2=g2_3, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    nng.process(cat1p, cat2, cat3)
    np.testing.assert_allclose(nng.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-4)
    nng.process(cat2p, cat1, cat3)
    np.testing.assert_allclose(nng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_213, rtol=1.e-4)

    ngn.process(cat1p, cat3, cat2)
    np.testing.assert_allclose(ngn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-4)
    ngn.process(cat2p, cat3, cat1)
    np.testing.assert_allclose(ngn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_231, rtol=1.e-4)

    gnn.process(cat3p, cat1, cat2)
    np.testing.assert_allclose(gnn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-4)
    gnn.process(cat3p, cat2, cat1)
    np.testing.assert_allclose(gnn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_321, rtol=1.e-4)

    # Now use all three patched
    nng.process(cat1p, cat2p, cat3p)
    np.testing.assert_allclose(nng.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_123, rtol=1.e-4)
    nng.process(cat2p, cat1p, cat3p)
    np.testing.assert_allclose(nng.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_213, rtol=1.e-4)

    ngn.process(cat1p, cat3p, cat2p)
    np.testing.assert_allclose(ngn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_132, rtol=1.e-4)
    ngn.process(cat2p, cat3p, cat1p)
    np.testing.assert_allclose(ngn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_231, rtol=1.e-4)

    gnn.process(cat3p, cat1p, cat2p)
    np.testing.assert_allclose(gnn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_312, rtol=1.e-4)
    gnn.process(cat3p, cat2p, cat1p)
    np.testing.assert_allclose(gnn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_321, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        nng.process(cat1p, cat2p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        ngn.process(cat1p, cat3p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        gnn.process(cat3p, cat1p, cat2p, patch_method='global')

    # Test I/O
    for name, corr in zip(['nng', 'ngn', 'gnn'], [nng, ngn, gnn]):
        ascii_name = 'output/'+name+'_ascii_logmultipole.txt'
        corr.write(ascii_name, precision=16)
        corr2 = treecorr.Corr3.from_file(ascii_name)
        np.testing.assert_allclose(corr2.ntri, corr.ntri)
        np.testing.assert_allclose(corr2.weight, corr.weight)
        np.testing.assert_allclose(corr2.zeta, corr.zeta)
        np.testing.assert_allclose(corr2.meand1, corr.meand1)
        np.testing.assert_allclose(corr2.meand2, corr.meand2)
        np.testing.assert_allclose(corr2.meand3, corr.meand3)
        np.testing.assert_allclose(corr2.meanlogd1, corr.meanlogd1)
        np.testing.assert_allclose(corr2.meanlogd2, corr.meanlogd2)
        np.testing.assert_allclose(corr2.meanlogd3, corr.meanlogd3)


@timer
def test_direct_logmultipole_cross21():
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

    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')

    log_min_sep = np.log(min_sep)
    log_max_sep = np.log(max_sep)
    true_zeta_112 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_112 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_112 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_121 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_121 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_121 = np.zeros((nbins, nbins, 2*max_n+1))
    true_zeta_211 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_weight_211 = np.zeros((nbins, nbins, 2*max_n+1), dtype=complex)
    true_ntri_211 = np.zeros((nbins, nbins, 2*max_n+1))
    bin_size = (log_max_sep - log_min_sep) / nbins
    n1d = np.arange(-max_n, max_n+1)
    for i in range(ngal):
        for j in range(ngal):
            if i == j: continue
            for k in range(ngal):
                d1 = np.sqrt((x1[j]-x2[k])**2 + (y1[j]-y2[k])**2)
                d2 = np.sqrt((x1[i]-x2[k])**2 + (y1[i]-y2[k])**2)
                d3 = np.sqrt((x1[i]-x1[j])**2 + (y1[i]-y1[j])**2)
                if d1 == 0.: continue
                if d2 == 0.: continue
                if d3 == 0.: continue

                kr1 = int(np.floor( (np.log(d1)-log_min_sep) / bin_size ))
                kr2 = int(np.floor( (np.log(d2)-log_min_sep) / bin_size ))
                kr3 = int(np.floor( (np.log(d3)-log_min_sep) / bin_size ))

                www = w1[i] * w1[j] * w2[k]

                # 112, 121
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
                    # g3 is projected to the line from c1 to c3.
                    expmialpha3 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    zeta = www * g3p

                    assert 0 <= kr2 < nbins
                    assert 0 <= kr3 < nbins
                    phi = np.arccos((d2**2 + d3**2 - d1**2)/(2*d2*d3))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x1[j],y1[j]):
                        phi = -phi
                    true_zeta_112[kr2,kr3,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_112[kr2,kr3,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_112[kr2,kr3,:] += 1
                    true_zeta_121[kr3,kr2,:] += zeta * np.exp(1j * n1d * phi)
                    true_weight_121[kr3,kr2,:] += www * np.exp(1j * n1d * phi)
                    true_ntri_121[kr3,kr2,:] += 1

                # 211
                if d1 >= min_sep and d1 < max_sep and d2 >= min_sep and d2 < max_sep:
                    # g3 is projected to the average of lines from c3 to c1 and c3 to c2.
                    expmialpha1 = (x2[k]-x1[i]) - 1j*(y2[k]-y1[i])
                    expmialpha1 /= abs(expmialpha1)
                    expmialpha2 = (x2[k]-x1[j]) - 1j*(y2[k]-y1[j])
                    expmialpha2 /= abs(expmialpha2)
                    expmialpha3 = expmialpha1 + expmialpha2
                    expmialpha3 /= abs(expmialpha3)
                    g3p = (g1_2[k] + 1j*g2_2[k]) * expmialpha3**2
                    zeta = www * g3p

                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x1[j],y1[j]):
                        phi = -phi
                    true_zeta_211[kr1,kr2,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_211[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_211[kr1,kr2,:] += 1

    nng.process(cat1, cat1, cat2)
    np.testing.assert_allclose(nng.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_112, rtol=1.e-4)
    ngn.process(cat1, cat2, cat1)
    np.testing.assert_allclose(ngn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_121, rtol=1.e-4)
    # 3 arg version doesn't work for gnn because the gnn processing doesn't know cat2 and cat3
    # are actually the same, so it doesn't subtract off the duplicates.

    # 2 arg version
    nng.process(cat1, cat2)
    np.testing.assert_allclose(nng.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_112, rtol=1.e-4)
    gnn.process(cat2, cat1)
    np.testing.assert_allclose(gnn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_211, rtol=1.e-4)

    # Repeat with binslop = 0
    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nng.process(cat1, cat2)
    np.testing.assert_allclose(nng.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_112, rtol=1.e-4)

    ngn.process(cat1, cat2, cat1)
    np.testing.assert_allclose(ngn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_121, rtol=1.e-4)

    gnn.process(cat2, cat1)
    np.testing.assert_allclose(gnn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_211, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, g1=g1_2, g2=g2_2, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    nng.process(cat1p, cat2)
    np.testing.assert_allclose(nng.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_112, rtol=1.e-4)

    ngn.process(cat1p, cat2, cat1)
    np.testing.assert_allclose(ngn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_121, rtol=1.e-4)

    gnn.process(cat2p, cat1)
    np.testing.assert_allclose(gnn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_211, rtol=1.e-4)

    # Now use both patched
    nng.process(cat1p, cat2p)
    np.testing.assert_allclose(nng.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nng.zeta, true_zeta_112, rtol=1.e-4)

    ngn.process(cat1p, cat2p, cat1p)
    np.testing.assert_allclose(ngn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(ngn.zeta, true_zeta_121, rtol=1.e-4)

    gnn.process(cat2p, cat1p)
    np.testing.assert_allclose(gnn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(gnn.zeta, true_zeta_211, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        nng.process(cat1p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        ngn.process(cat1p, cat2p, cat1p, patch_method='global')
    with assert_raises(ValueError):
        gnn.process(cat2p, cat1p, patch_method='global')


@timer
def test_nng_logsas():
    # For this test, we need coherent pattern that gives an NNG signal.
    # We take pairs of "lens" points and put a constant shear field in the circle between them.
    #
    # This is similar to a predicted pattern between pairs of lenses by Bernardeau et al, 2003.
    # cf. https://arxiv.org/pdf/astro-ph/0201029 esp. Figure 2 bottom panel.

    r0 = 10.
    if __name__ == '__main__':
        nlens = 2000
        nsource = 2000000
        L = 100. * r0
        tol_factor = 1
    else:
        # Looser tests that don't take so long to run.
        nlens = 400
        nsource = 100000
        L = 30. * r0
        tol_factor = 4

    rng = np.random.RandomState(8675309)
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    phi = rng.random_sample(nlens) * 2*np.pi
    x2 = x1 + r0 * np.cos(phi)
    y2 = y1 + r0 * np.sin(phi)
    xmid = (x1+x2)/2
    ymid = (y1+y2)/2
    g = -0.07 * np.exp(2j * phi)

    x3 = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    y3 = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    gamma = np.zeros(nsource, dtype=complex)
    for k in range(nlens):
        dx = x3-xmid[k]
        dy = y3-ymid[k]
        r = np.sqrt(dx**2 + dy**2) / r0
        gamma[r <= r0/2] += g[k]

    ncat = treecorr.Catalog(x=np.concatenate([x1,x2]), y=np.concatenate([y1,y2]))
    gcat = treecorr.Catalog(x=x3, y=y3, g1=np.real(gamma), g2=np.imag(gamma))

    min_sep = 4
    max_sep = 12
    nbins = 15
    nphi_bins = 30

    nrand = 10*nlens
    xr = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    yr = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    rcat = treecorr.Catalog(x=xr, y=yr)

    nng = treecorr.NNGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nphi_bins=nphi_bins, bin_type='LogSAS')
    ngn = treecorr.NGNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nphi_bins=nphi_bins, bin_type='LogSAS')
    gnn = treecorr.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nphi_bins=nphi_bins, bin_type='LogSAS')

    for name, corr in zip(['nng', 'ngn', 'gnn'], [nng, ngn, gnn]):
        print(name)
        t0 = time.time()
        if name == 'nng':
            corr.process(ncat, gcat, algo='triangle')
        elif name == 'ngn':
            corr.process(ncat, gcat, ncat, algo='triangle')
        else:
            corr.process(gcat, ncat, algo='triangle')
        t1 = time.time()
        print(name,'process time = ',t1-t0)

        # Compute true zeta based on measured d1,d2,d3 in correlation
        # Take d1 to be the distance between the two N vertices.
        if name == 'nng':
            d1 = corr.meand3
            d2 = corr.meand1
            d3 = corr.meand2
        elif name == 'ngn':
            d1 = corr.meand2
            d2 = corr.meand3
            d3 = corr.meand1
        else:
            d1 = corr.meand1
            d2 = corr.meand2
            d3 = corr.meand3

        # Expect signal when d1 ~= 10 and opening angle at G point is > 90 degrees.
        good_shape = (d1>9.5) & (d1<10.5) & (d2**2+d3**2 < d1**2)
        i = np.where(corr.ntri == np.max(corr.ntri))
        print('Most populated bin is d1,d2,d3 = ',d1[i][0], d2[i][0], d3[i][0])
        print('overall max ntri = ',np.max(corr.ntri))
        print('max ntri with good shape = ',np.max(corr.ntri[good_shape]))

        # Use bins with at least 80% of this value, so we get decent statistics.
        min_ntri = np.max(corr.ntri[good_shape]) * 0.8
        m = np.where(good_shape & (corr.ntri>min_ntri))

        # Expect g to be perpendicular to line d1.
        # But zeta projection is relative to centroid.  Need to project.
        theta2 = np.arccos((d1**2 + d3**2 - d2**2) / (2*d1*d3))
        theta1 = np.arccos((d2**2 + d3**2 - d1**2) / (2*d2*d3))
        s = d1
        t = d3 * np.exp(1j * theta2)
        q2 = (s + t)/3.
        q1 = q2 - t
        q1 /= np.abs(q1)
        true_zeta = np.zeros_like(corr.zeta)
        true_zeta[m] = -0.07 * q1[m]**2

        print('ntri = ',corr.ntri[m])
        print('zeta/0.07 = ',corr.zeta[m]/0.07)
        print('expected zeta/0.07 = ',true_zeta[m]/0.07)
        print('ratio = ',corr.zeta[m]/true_zeta[m])
        print('max rel diff (m) = ',np.max(np.abs((corr.zeta[m] - true_zeta[m])/true_zeta[m])))
        print('max diff (everywhere) = ',np.max(np.abs((corr.zeta - true_zeta))))

        np.testing.assert_allclose(corr.zeta[m], true_zeta[m], rtol=0.3 * tol_factor)
        np.testing.assert_allclose(np.log(np.abs(corr.zeta[m])),
                                   np.log(np.abs(true_zeta[m])), atol=0.3 * tol_factor)
        np.testing.assert_allclose(corr.zeta, true_zeta, atol=0.2 * tol_factor)

        # Repeat this using Multipole and then convert to SAS:
        if name == 'nng':
            cls = treecorr.NNGCorrelation
        elif name == 'ngn':
            cls = treecorr.NGNCorrelation
        else:
            cls = treecorr.GNNCorrelation
        corrm = cls(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=80,
                    bin_type='LogMultipole')
        t0 = time.time()
        if name == 'nng':
            corrm.process(ncat, gcat)
        elif name == 'ngn':
            corrm.process(ncat, gcat, ncat)
        else:
            corrm.process(gcat, ncat)
        corrs = corrm.toSAS(nphi_bins=nphi_bins)
        t1 = time.time()
        print('time for multipole corr:', t1-t0)

        print('ratio = ',corrs.zeta[m] / corr.zeta[m])
        print('zeta mean ratio = ',np.mean(corrs.zeta[m] / corr.zeta[m]))
        print('zeta mean diff = ',np.mean(corrs.zeta[m] - corr.zeta[m]))
        # Some of the individual values are a little ratty, but on average, they are quite close.
        rtol = 0.25 if name == 'gnn' else 0.1
        np.testing.assert_allclose(corrs.zeta[m], corr.zeta[m], rtol=rtol*tol_factor)
        rtol = 0.1 if name == 'gnn' else 0.05
        np.testing.assert_allclose(np.mean(corrs.zeta[m] / corr.zeta[m]), 1., rtol=rtol*tol_factor)
        np.testing.assert_allclose(corrs.ntri[m], corr.ntri[m], rtol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd1[m], corr.meanlogd1[m],
                                   rtol=0.1*tol_factor, atol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd2[m], corr.meanlogd2[m],
                                   rtol=0.1*tol_factor, atol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd3[m], corr.meanlogd3[m],
                                   rtol=0.1*tol_factor, atol=0.2*tol_factor)
        np.testing.assert_allclose(corrs.meanphi[m], corr.meanphi[m],
                                   rtol=0.1*tol_factor, atol=0.2*tol_factor)

        # Error to try to change sep binning with toSAS
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, min_sep=1)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, max_sep=25)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, nbins=50)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, bin_size=0.01, nbins=None)
        # Error if non-Multipole calls toSAS
        with assert_raises(TypeError):
            corrs.toSAS()

        # All of the above is the default algorithm if process doesn't set algo='triangle'.
        # Check the automatic use of the multipole algorithm from LogSAS.
        corr3 = corr.copy()
        if name == 'nng':
            corr3.process(ncat, gcat, algo='multipole', max_n=80)
        elif name == 'ngn':
            corr3.process(ncat, gcat, ncat, algo='multipole', max_n=80)
        else:
            corr3.process(gcat, ncat, algo='multipole', max_n=80)
        np.testing.assert_allclose(corr3.weight, corrs.weight)
        np.testing.assert_allclose(corr3.zeta, corrs.zeta)

        # Now use randoms
        corr2 = corr.copy()
        rr = corr.copy()
        if name == 'nng':
            rr.process(rcat, gcat)
            czkwargs = dict(rrg=rr)
        elif name == 'ngn':
            rr.process(rcat, gcat, rcat)
            czkwargs = dict(rgr=rr)
        else:
            rr.process(gcat, rcat)
            czkwargs = dict(grr=rr)
        zeta, varzeta = corr2.calculateZeta(**czkwargs)

        print('with rand mean ratio = ',np.mean(corr2.zeta[m]/true_zeta[m]))
        np.testing.assert_allclose(corr2.zeta[m], true_zeta[m], rtol=0.3*tol_factor)

        corr2x = corr2.copy()
        np.testing.assert_allclose(corr2x.zeta, corr2.zeta)
        np.testing.assert_allclose(corr2x.zetar, corr2.zeta.real)
        np.testing.assert_allclose(corr2x.zetai, corr2.zeta.imag)
        np.testing.assert_allclose(corr2x.varzeta, corr2.varzeta)
        np.testing.assert_allclose(corr2x.raw_zeta, corr2.raw_zeta)
        np.testing.assert_allclose(corr2x.raw_varzeta, corr2.raw_varzeta)
        np.testing.assert_allclose(corr.calculateZeta()[0], corr.zeta)
        np.testing.assert_allclose(corr.calculateZeta()[1], corr.varzeta)
        np.testing.assert_allclose(corrs.calculateZeta()[0], corrs.zeta)
        np.testing.assert_allclose(corrs.calculateZeta()[1], corrs.varzeta)

        with assert_raises(TypeError):
            corrm.calculateZeta(**czkwargs)

        # Check that we get the same result using the corr3 functin:
        # (This implicitly uses the multipole algorithm.)
        ncat.write(os.path.join('data',name+'_ndata_logsas.dat'))
        gcat.write(os.path.join('data',name+'_gdata_logsas.dat'))
        config = treecorr.config.read_config('configs/'+name+'_logsas.yaml')
        config['verbose'] = 0
        treecorr.corr3(config)
        corr3_output = np.genfromtxt(os.path.join('output',name+'_logsas.out'),
                                     names=True, skip_header=1)
        np.testing.assert_allclose(corr3_output['zetar'], corr3.zetar.flatten(), rtol=1.e-3, atol=0)
        np.testing.assert_allclose(corr3_output['zetai'], corr3.zetai.flatten(), rtol=1.e-3,
                                   atol=0)

        if name == 'ngn':
            # Invalid to omit file_name2
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)
        else:
            # Invalid to call cat2 file_name3 rather than file_name2
            config['file_name3'] = config['file_name2']
            if name == 'nng':
                config['g1_col'] = [0,0,3]
                config['g2_col'] = [0,0,4]
            else:
                config['g1_col'] = [3,0,0]
                config['g2_col'] = [4,0,0]
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)

        # Check the fits write option
        try:
            import fitsio
        except ImportError:
            pass
        else:
            out_file_name = os.path.join('output','corr_nng_logsas.fits')
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
            np.testing.assert_allclose(data['zetar'], corr.zeta.real.flatten())
            np.testing.assert_allclose(data['zetai'], corr.zeta.imag.flatten())
            np.testing.assert_allclose(data['sigma_zeta'], np.sqrt(corr.varzeta.flatten()))
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
            np.testing.assert_allclose(corr2.zeta, corr.zeta)
            np.testing.assert_allclose(corr2.varzeta, corr.varzeta)
            np.testing.assert_allclose(corr2.weight, corr.weight)
            np.testing.assert_allclose(corr2.ntri, corr.ntri)
            assert corr2.coords == corr.coords
            assert corr2.metric == corr.metric
            assert corr2.sep_units == corr.sep_units
            assert corr2.bin_type == corr.bin_type

@timer
def test_varzeta():
    # Test that the shot noise estimate of varzeta is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for g1,g2, but this pattern doesn't have much 3pt correlation.
    gamma0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(8675308)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 100
    nsource = 50000

    file_name = 'data/test_varzeta_nng.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nngs = []
        all_ngns = []
        all_gnns = []

        for run in range(nruns):
            print(f'{run}/{nruns}')
            # In addition to the shape noise below, there is shot noise
            # from the random x,y positions.
            x1 = (rng.random_sample(nlens)-0.5) * L
            y1 = (rng.random_sample(nlens)-0.5) * L
            x2 = (rng.random_sample(nsource)-0.5) * L
            y2 = (rng.random_sample(nsource)-0.5) * L
            w = np.ones_like(x2) * 5
            r2 = (x2**2 + y2**2)/r0**2
            g1 = -gamma0 * np.exp(-r2/2.) * (x2**2-y2**2)/r0**2
            g2 = -gamma0 * np.exp(-r2/2.) * (2.*x2*y2)/r0**2
            g1 += rng.normal(0, 0.3, size=nsource)
            g2 += rng.normal(0, 0.3, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1)
            gcat = treecorr.Catalog(x=x2, y=y2, w=w, g1=g1, g2=g2)
            nng = treecorr.NNGCorrelation(nbins=2, min_sep=50., max_sep=70., nphi_bins=20)
            ngn = treecorr.NGNCorrelation(nbins=2, min_sep=50., max_sep=70., nphi_bins=20)
            gnn = treecorr.GNNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.4, nphi_bins=20)
            nng.process(ncat, gcat)
            ngn.process(ncat, gcat, ncat)
            gnn.process(gcat, ncat)
            all_nngs.append(nng)
            all_ngns.append(ngn)
            all_gnns.append(gnn)

        mean_nng_zetar = np.mean([nng.zetar for nng in all_nngs], axis=0)
        mean_nng_zetai = np.mean([nng.zetai for nng in all_nngs], axis=0)
        var_nng_zetar = np.var([nng.zetar for nng in all_nngs], axis=0)
        var_nng_zetai = np.var([nng.zetai for nng in all_nngs], axis=0)
        mean_nng_varzeta = np.mean([nng.varzeta for nng in all_nngs], axis=0)
        mean_ngn_zetar = np.mean([ngn.zetar for ngn in all_ngns], axis=0)
        mean_ngn_zetai = np.mean([ngn.zetai for ngn in all_ngns], axis=0)
        var_ngn_zetar = np.var([ngn.zetar for ngn in all_ngns], axis=0)
        var_ngn_zetai = np.var([ngn.zetai for ngn in all_ngns], axis=0)
        mean_ngn_varzeta = np.mean([ngn.varzeta for ngn in all_ngns], axis=0)
        mean_gnn_zetar = np.mean([gnn.zetar for gnn in all_gnns], axis=0)
        mean_gnn_zetai = np.mean([gnn.zetai for gnn in all_gnns], axis=0)
        var_gnn_zetar = np.var([gnn.zetar for gnn in all_gnns], axis=0)
        var_gnn_zetai = np.var([gnn.zetai for gnn in all_gnns], axis=0)
        mean_gnn_varzeta = np.mean([gnn.varzeta for gnn in all_gnns], axis=0)

        np.savez(file_name,
                 mean_nng_zetar=mean_nng_zetar,
                 mean_nng_zetai=mean_nng_zetai,
                 var_nng_zetar=var_nng_zetar,
                 var_nng_zetai=var_nng_zetai,
                 mean_nng_varzeta=mean_nng_varzeta,
                 mean_ngn_zetar=mean_ngn_zetar,
                 mean_ngn_zetai=mean_ngn_zetai,
                 var_ngn_zetar=var_ngn_zetar,
                 var_ngn_zetai=var_ngn_zetai,
                 mean_ngn_varzeta=mean_ngn_varzeta,
                 mean_gnn_zetar=mean_gnn_zetar,
                 mean_gnn_zetai=mean_gnn_zetai,
                 var_gnn_zetar=var_gnn_zetar,
                 var_gnn_zetai=var_gnn_zetai,
                 mean_gnn_varzeta=mean_gnn_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_nng_zetar = data['mean_nng_zetar']
    mean_nng_zetai = data['mean_nng_zetai']
    var_nng_zetar = data['var_nng_zetar']
    var_nng_zetai = data['var_nng_zetai']
    mean_nng_varzeta = data['mean_nng_varzeta']
    mean_ngn_zetar = data['mean_ngn_zetar']
    mean_ngn_zetai = data['mean_ngn_zetai']
    var_ngn_zetar = data['var_ngn_zetar']
    var_ngn_zetai = data['var_ngn_zetai']
    mean_ngn_varzeta = data['mean_ngn_varzeta']
    mean_gnn_zetar = data['mean_gnn_zetar']
    mean_gnn_zetai = data['mean_gnn_zetai']
    var_gnn_zetar = data['var_gnn_zetar']
    var_gnn_zetai = data['var_gnn_zetai']
    mean_gnn_varzeta = data['mean_gnn_varzeta']

    print('var_nng_zetar = ',var_nng_zetar)
    print('mean nng_varzeta = ',mean_nng_varzeta)
    print('ratio = ',var_nng_zetar.ravel() / mean_nng_varzeta.ravel())
    print('var_ngn_zetar = ',var_ngn_zetar)
    print('mean ngn_varzeta = ',mean_ngn_varzeta)
    print('ratio = ',var_ngn_zetar.ravel() / mean_ngn_varzeta.ravel())
    print('var_gnn_zetar = ',var_gnn_zetar)
    print('mean gnn_varzeta = ',mean_gnn_varzeta)
    print('ratio = ',var_gnn_zetar.ravel() / mean_gnn_varzeta.ravel())

    print('max relerr for nng zetar = ',
          np.max(np.abs((var_nng_zetar - mean_nng_varzeta)/var_nng_zetar)))
    print('max relerr for nng zetai = ',
          np.max(np.abs((var_nng_zetai - mean_nng_varzeta)/var_nng_zetai)))
    np.testing.assert_allclose(mean_nng_varzeta, var_nng_zetar, rtol=0.15)
    np.testing.assert_allclose(mean_nng_varzeta, var_nng_zetai, rtol=0.15)

    print('max relerr for ngn zetar = ',
          np.max(np.abs((var_ngn_zetar - mean_ngn_varzeta)/var_ngn_zetar)))
    print('max relerr for ngn zetai = ',
          np.max(np.abs((var_ngn_zetai - mean_ngn_varzeta)/var_ngn_zetai)))
    np.testing.assert_allclose(mean_ngn_varzeta, var_ngn_zetar, rtol=0.15)
    np.testing.assert_allclose(mean_ngn_varzeta, var_ngn_zetai, rtol=0.15)

    print('max relerr for gnn zetar = ',
          np.max(np.abs((var_gnn_zetar - mean_gnn_varzeta)/var_gnn_zetar)))
    print('max relerr for gnn zetai = ',
          np.max(np.abs((var_gnn_zetai - mean_gnn_varzeta)/var_gnn_zetai)))
    np.testing.assert_allclose(mean_gnn_varzeta, var_gnn_zetar, rtol=0.15)
    np.testing.assert_allclose(mean_gnn_varzeta, var_gnn_zetai, rtol=0.15)

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
    nng = treecorr.NNGCorrelation(nbins=2, min_sep=50., max_sep=70., nphi_bins=20)
    ngn = treecorr.NGNCorrelation(nbins=2, min_sep=50., max_sep=70., nphi_bins=20)
    gnn = treecorr.GNNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.4, nphi_bins=20)

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(nng.cov, 0)
    np.testing.assert_array_equal(nng.varzeta, 0)
    np.testing.assert_array_equal(ngn.cov, 0)
    np.testing.assert_array_equal(ngn.varzeta, 0)
    np.testing.assert_array_equal(gnn.cov, 0)
    np.testing.assert_array_equal(gnn.varzeta, 0)

    nng.process(ncat, gcat)
    print('NNG single run:')
    print('max relerr for zetar = ',np.max(np.abs((nng.varzeta - var_nng_zetar)/var_nng_zetar)))
    print('ratio = ',nng.varzeta / var_nng_zetar)
    print('max relerr for zetai = ',np.max(np.abs((nng.varzeta - var_nng_zetai)/var_nng_zetai)))
    print('ratio = ',nng.varzeta / var_nng_zetai)
    print('var_num = ',nng._var_num)
    print('ntri = ',nng.ntri)
    np.testing.assert_allclose(nng.varzeta, var_nng_zetar, rtol=0.4)
    np.testing.assert_allclose(nng.varzeta, var_nng_zetai, rtol=0.4)
    np.testing.assert_allclose(nng.cov.diagonal(), nng.varzeta.ravel())

    ngn.process(ncat, gcat, ncat)
    print('NGN single run:')
    print('max relerr for zetar = ',np.max(np.abs((ngn.varzeta - var_ngn_zetar)/var_ngn_zetar)))
    print('ratio = ',ngn.varzeta / var_ngn_zetar)
    print('max relerr for zetai = ',np.max(np.abs((ngn.varzeta - var_ngn_zetai)/var_ngn_zetai)))
    print('ratio = ',ngn.varzeta / var_ngn_zetai)
    np.testing.assert_allclose(ngn.varzeta, var_ngn_zetar, rtol=0.4)
    np.testing.assert_allclose(ngn.varzeta, var_ngn_zetai, rtol=0.4)
    np.testing.assert_allclose(ngn.cov.diagonal(), ngn.varzeta.ravel())

    gnn.process(gcat, ncat)
    print('GNN single run:')
    print('max relerr for zetar = ',np.max(np.abs((gnn.varzeta - var_gnn_zetar)/var_gnn_zetar)))
    print('ratio = ',gnn.varzeta / var_gnn_zetar)
    print('max relerr for zetai = ',np.max(np.abs((gnn.varzeta - var_gnn_zetai)/var_gnn_zetai)))
    print('ratio = ',gnn.varzeta / var_gnn_zetai)
    np.testing.assert_allclose(gnn.varzeta, var_gnn_zetar, rtol=0.4)
    np.testing.assert_allclose(gnn.varzeta, var_gnn_zetai, rtol=0.4)
    np.testing.assert_allclose(gnn.cov.diagonal(), gnn.varzeta.ravel())


@timer
def test_nng_logsas_jk():
    # Test jackknife covariance estimates for nng correlations with LogSAS binning.

    # Skip this test on windows, since it is vv slow.
    if os.name == 'nt': return

    if __name__ == '__main__':
        nhalo = 2000
        nsource = 50000
        npatch = 300
        tol_factor = 1
    else:
        nhalo = 2000
        nsource = 50000
        npatch = 12
        tol_factor = 2

    nhalo = 2000
    nsource = 50000
    npatch = 16
    tol_factor = 2

    nbins = 2
    min_sep = 12
    max_sep = 16
    nphi_bins = 5
    min_phi = 30
    max_phi = 90

    file_name = 'data/test_nng_logsas_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_nng = []
        all_ngn = []
        all_gnn = []
        all_nng_rr = []
        all_ngn_rr = []
        all_gnn_rr = []
        all_nng_dr = []
        all_ngn_dr = []
        all_gnn_dr = []
        rng1 = np.random.default_rng()
        for run in range(nruns):
            x, y, g1, g2, _, xh, yh = generate_shear_field(nsource, nhalo, rng1, return_halos=True)
            xr = rng1.uniform(0, 1000, size=10*nhalo)
            yr = rng1.uniform(0, 1000, size=10*nhalo)
            print(run,': ',np.std(g1),np.std(g2))
            gcat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)
            ncat = treecorr.Catalog(x=xh, y=yh)
            rcat = treecorr.Catalog(x=xr, y=yr)
            nng = treecorr.NNGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            nng.process(ncat, gcat)
            all_nng.append(nng.zeta.ravel())
            rrg = nng.copy()
            rrg.process(rcat, gcat)
            drg = nng.copy()
            drg.process(ncat, rcat, gcat, ordered=3)
            zeta, _ = nng.calculateZeta(rrg=rrg)
            all_nng_rr.append(zeta.ravel())
            zeta, _ = nng.calculateZeta(rrg=rrg, drg=drg)
            all_nng_dr.append(zeta.ravel())

            ngn = treecorr.NGNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            ngn.process(ncat, gcat, ncat)
            all_ngn.append(ngn.zeta.ravel())
            rgr = ngn.copy()
            rgr.process(rcat, gcat, rcat)
            dgr = ngn.copy()
            dgr.process(ncat, gcat, rcat, ordered=2)
            zeta, _ = ngn.calculateZeta(rgr=rgr)
            all_ngn_rr.append(zeta.ravel())
            zeta, _ = ngn.calculateZeta(rgr=rgr, dgr=dgr)
            all_ngn_dr.append(zeta.ravel())

            gnn = treecorr.GNNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            gnn.process(gcat, ncat)
            all_gnn.append(gnn.zeta.ravel())
            grr = gnn.copy()
            grr.process(gcat, rcat)
            gdr = gnn.copy()
            gdr.process(gcat, ncat, rcat, ordered=1)
            zeta, _ = gnn.calculateZeta(grr=grr)
            all_gnn_rr.append(zeta.ravel())
            zeta, _ = gnn.calculateZeta(grr=grr, gdr=gdr)
            all_gnn_dr.append(zeta.ravel())

        mean_nng = np.mean(all_nng, axis=0)
        var_nng = np.var(all_nng, axis=0)
        mean_ngn = np.mean(all_ngn, axis=0)
        var_ngn = np.var(all_ngn, axis=0)
        mean_gnn = np.mean(all_gnn, axis=0)
        var_gnn = np.var(all_gnn, axis=0)
        mean_nng_rr = np.mean(all_nng_rr, axis=0)
        var_nng_rr = np.var(all_nng_rr, axis=0)
        mean_ngn_rr = np.mean(all_ngn_rr, axis=0)
        var_ngn_rr = np.var(all_ngn_rr, axis=0)
        mean_gnn_rr = np.mean(all_gnn_rr, axis=0)
        var_gnn_rr = np.var(all_gnn_rr, axis=0)
        mean_nng_dr = np.mean(all_nng_dr, axis=0)
        var_nng_dr = np.var(all_nng_dr, axis=0)
        mean_ngn_dr = np.mean(all_ngn_dr, axis=0)
        var_ngn_dr = np.var(all_ngn_dr, axis=0)
        mean_gnn_dr = np.mean(all_gnn_dr, axis=0)
        var_gnn_dr = np.var(all_gnn_dr, axis=0)

        np.savez(file_name,
                 mean_nng=mean_nng, var_nng=var_nng,
                 mean_ngn=mean_ngn, var_ngn=var_ngn,
                 mean_gnn=mean_gnn, var_gnn=var_gnn,
                 mean_nng_rr=mean_nng_rr, var_nng_rr=var_nng_rr,
                 mean_ngn_rr=mean_ngn_rr, var_ngn_rr=var_ngn_rr,
                 mean_gnn_rr=mean_gnn_rr, var_gnn_rr=var_gnn_rr,
                 mean_nng_dr=mean_nng_dr, var_nng_dr=var_nng_dr,
                 mean_ngn_dr=mean_ngn_dr, var_ngn_dr=var_ngn_dr,
                 mean_gnn_dr=mean_gnn_dr, var_gnn_dr=var_gnn_dr)

    data = np.load(file_name)
    mean_nng = data['mean_nng']
    var_nng = data['var_nng']
    mean_ngn = data['mean_ngn']
    var_ngn = data['var_ngn']
    mean_gnn = data['mean_gnn']
    var_gnn = data['var_gnn']
    mean_nng_rr = data['mean_nng_rr']
    var_nng_rr = data['var_nng_rr']
    mean_ngn_rr = data['mean_ngn_rr']
    var_ngn_rr = data['var_ngn_rr']
    mean_gnn_rr = data['mean_gnn_rr']
    var_gnn_rr = data['var_gnn_rr']
    mean_nng_dr = data['mean_nng_dr']
    var_nng_dr = data['var_nng_dr']
    mean_ngn_dr = data['mean_ngn_dr']
    var_ngn_dr = data['var_ngn_dr']
    mean_gnn_dr = data['mean_gnn_dr']
    var_gnn_dr = data['var_gnn_dr']
    print('mean nng = ',mean_nng)
    print('var nng = ',var_nng)
    print('mean ngn = ',mean_ngn)
    print('var ngn = ',var_ngn)
    print('mean gnn = ',mean_gnn)
    print('var gnn = ',var_gnn)

    rng = np.random.default_rng(1234)
    x, y, g1, g2, _, xh, yh = generate_shear_field(nsource, nhalo, rng, return_halos=True)
    xr = rng.uniform(0, 1000, size=10*nhalo)
    yr = rng.uniform(0, 1000, size=10*nhalo)
    gcat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2, npatch=npatch, rng=rng)
    ncat = treecorr.Catalog(x=xh, y=yh, rng=rng, patch_centers=gcat.patch_centers)
    rcat = treecorr.Catalog(x=xr, y=yr, rng=rng, patch_centers=gcat.patch_centers)

    # First check calculate_xi with all pairs in results dict.
    nng = treecorr.NNGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    nng.process(ncat, gcat)
    nng2 = nng.copy()
    nng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nng.results.keys()], False)
    np.testing.assert_allclose(nng2.ntri, nng.ntri, rtol=0.01)
    np.testing.assert_allclose(nng2.zeta, nng.zeta, rtol=0.01)
    np.testing.assert_allclose(nng2.varzeta, nng.varzeta, rtol=0.01)
    rrg = nng.copy()
    rrg.process(rcat, gcat)
    nng_rr = nng.copy()
    zeta_rrg, varzeta_rrg = nng_rr.calculateZeta(rrg=rrg)
    nng2 = nng_rr.copy()
    nng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nng.results.keys()], False)
    np.testing.assert_allclose(nng2.ntri, nng_rr.ntri, rtol=0.01)
    np.testing.assert_allclose(nng2.zeta, nng_rr.zeta, rtol=0.01)
    np.testing.assert_allclose(nng2.varzeta, nng_rr.varzeta, rtol=0.01)
    np.testing.assert_allclose(nng2.zeta, zeta_rrg, rtol=0.01)
    np.testing.assert_allclose(nng2.varzeta, varzeta_rrg, rtol=0.01)
    drg = nng.copy()
    drg.process(ncat, rcat, gcat, ordered=3)
    nng_dr = nng.copy()
    zeta_drg, varzeta_drg = nng_dr.calculateZeta(rrg=rrg, drg=drg)
    nng2 = nng_dr.copy()
    nng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nng.results.keys()], False)
    np.testing.assert_allclose(nng2.ntri, nng_dr.ntri, rtol=0.01)
    np.testing.assert_allclose(nng2.zeta, nng_dr.zeta, rtol=0.01)
    np.testing.assert_allclose(nng2.varzeta, nng_dr.varzeta, rtol=0.01)
    np.testing.assert_allclose(nng2.zeta, zeta_drg, rtol=0.01)
    np.testing.assert_allclose(nng2.varzeta, varzeta_drg, rtol=0.01)
    nng2.calculateZeta(rrg=rrg, rdg=drg)
    nng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nng.results.keys()], True)
    np.testing.assert_allclose(nng2.zeta, nng_dr.zeta, rtol=0.01)
    nng2.calculateZeta(rrg=rrg, rdg=drg, drg=drg)
    nng2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nng.results.keys()], True)
    np.testing.assert_allclose(nng2.zeta, nng_dr.zeta, rtol=0.01)

    ngn = treecorr.NGNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    ngn.process(ncat, gcat, ncat)
    ngn2 = ngn.copy()
    ngn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngn.results.keys()], False)
    np.testing.assert_allclose(ngn2.ntri, ngn.ntri, rtol=0.01)
    np.testing.assert_allclose(ngn2.zeta, ngn.zeta, rtol=0.01)
    np.testing.assert_allclose(ngn2.varzeta, ngn.varzeta, rtol=0.01)
    rgr = ngn.copy()
    rgr.process(rcat, gcat, rcat)
    ngn_rr = ngn.copy()
    zeta_rgr, varzeta_rgr = ngn_rr.calculateZeta(rgr=rgr)
    ngn2 = ngn_rr.copy()
    ngn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngn.results.keys()], False)
    np.testing.assert_allclose(ngn2.ntri, ngn_rr.ntri, rtol=0.01)
    np.testing.assert_allclose(ngn2.zeta, ngn_rr.zeta, rtol=0.01)
    np.testing.assert_allclose(ngn2.varzeta, ngn_rr.varzeta, rtol=0.01)
    np.testing.assert_allclose(ngn2.zeta, zeta_rgr, rtol=0.01)
    np.testing.assert_allclose(ngn2.varzeta, varzeta_rgr, rtol=0.01)
    dgr = ngn.copy()
    dgr.process(ncat, gcat, rcat)
    ngn_dr = ngn.copy()
    zeta_dgr, varzeta_dgr = ngn_dr.calculateZeta(rgr=rgr, dgr=dgr)
    ngn2 = ngn_dr.copy()
    ngn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngn.results.keys()], False)
    np.testing.assert_allclose(ngn2.ntri, ngn_dr.ntri, rtol=0.01)
    np.testing.assert_allclose(ngn2.zeta, ngn_dr.zeta, rtol=0.01)
    np.testing.assert_allclose(ngn2.varzeta, ngn_dr.varzeta, rtol=0.01)
    np.testing.assert_allclose(ngn2.zeta, zeta_dgr, rtol=0.01)
    np.testing.assert_allclose(ngn2.varzeta, varzeta_dgr, rtol=0.01)
    ngn2.calculateZeta(rgr=rgr, rgd=dgr)
    ngn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngn.results.keys()], True)
    np.testing.assert_allclose(ngn2.zeta, ngn_dr.zeta, rtol=0.01)
    ngn2.calculateZeta(rgr=rgr, rgd=dgr, dgr=dgr)
    ngn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in ngn.results.keys()], True)
    np.testing.assert_allclose(ngn2.zeta, ngn_dr.zeta, rtol=0.01)

    gnn = treecorr.GNNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    gnn.process(gcat, ncat)
    gnn2 = gnn.copy()
    gnn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gnn.results.keys()], False)
    np.testing.assert_allclose(gnn2.ntri, gnn.ntri, rtol=0.01)
    np.testing.assert_allclose(gnn2.zeta, gnn.zeta, rtol=0.01)
    np.testing.assert_allclose(gnn2.varzeta, gnn.varzeta, rtol=0.01)
    grr = gnn.copy()
    grr.process(gcat, rcat)
    gnn_rr = gnn.copy()
    zeta_grr, varzeta_grr = gnn_rr.calculateZeta(grr=grr)
    gnn2 = gnn_rr.copy()
    gnn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gnn.results.keys()], False)
    np.testing.assert_allclose(gnn2.ntri, gnn_rr.ntri, rtol=0.01)
    np.testing.assert_allclose(gnn2.zeta, gnn_rr.zeta, rtol=0.01)
    np.testing.assert_allclose(gnn2.varzeta, gnn_rr.varzeta, rtol=0.01)
    np.testing.assert_allclose(gnn2.zeta, zeta_grr, rtol=0.01)
    np.testing.assert_allclose(gnn2.varzeta, varzeta_grr, rtol=0.01)
    gdr = gnn.copy()
    gdr.process(gcat, ncat, rcat)
    gnn_dr = gnn.copy()
    zeta_gdr, varzeta_gdr= gnn_dr.calculateZeta(grr=grr, gdr=gdr)
    gnn2 = gnn_dr.copy()
    gnn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gnn.results.keys()], False)
    np.testing.assert_allclose(gnn2.ntri, gnn_dr.ntri, rtol=0.01)
    np.testing.assert_allclose(gnn2.zeta, gnn_dr.zeta, rtol=0.01)
    np.testing.assert_allclose(gnn2.varzeta, gnn_dr.varzeta, rtol=0.01)
    np.testing.assert_allclose(gnn2.zeta, zeta_gdr, rtol=0.01)
    np.testing.assert_allclose(gnn2.varzeta, varzeta_gdr, rtol=0.01)
    gnn2.calculateZeta(grr=grr, grd=gdr)
    gnn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gnn.results.keys()], True)
    np.testing.assert_allclose(gnn2.zeta, gnn_dr.zeta, rtol=0.01)
    gnn2.calculateZeta(grr=grr, gdr=gdr, grd=gdr)
    gnn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in gnn.results.keys()], True)
    np.testing.assert_allclose(gnn2.zeta, gnn_dr.zeta, rtol=0.01)

    # Next check jackknife covariance estimate
    cov_nng = nng.estimate_cov('jackknife')
    n = nng.varzeta.size
    print('nng cov diag = ',np.diagonal(cov_nng))
    print('var_nng = ',var_nng)
    print('nng zeta var ratio = ',np.diagonal(cov_nng)/var_nng)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nng))-np.log(var_nng))))
    np.testing.assert_allclose(np.log(np.diagonal(cov_nng)), np.log(var_nng), atol=0.6*tol_factor)

    cov_ngn = ngn.estimate_cov('jackknife')
    print('ngn var ratio = ',np.diagonal(cov_ngn)/var_ngn)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_ngn))-np.log(var_ngn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov_ngn)), np.log(var_ngn), atol=0.6*tol_factor)

    cov_gnn = gnn.estimate_cov('jackknife')
    print('gnn var ratio = ',np.diagonal(cov_gnn)/var_gnn)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_gnn))-np.log(var_gnn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov_gnn)), np.log(var_gnn), atol=0.6*tol_factor)

    # Check that these still work after roundtripping through a file.
    file_name = os.path.join('output','test_write_results_nng.dat')
    nng.write(file_name, write_patch_results=True)
    nng2 = treecorr.Corr3.from_file(file_name)
    cov2 = nng2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_nng)

    file_name = os.path.join('output','test_write_results_ngn.dat')
    ngn.write(file_name, write_patch_results=True)
    ngn2 = treecorr.Corr3.from_file(file_name)
    cov2 = ngn2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_ngn)

    file_name = os.path.join('output','test_write_results_gnn.dat')
    gnn.write(file_name, write_patch_results=True)
    gnn2 = treecorr.Corr3.from_file(file_name)
    cov2 = gnn2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_gnn)

    # Check jackknife using LogMultipole
    print('Using LogMultipole:')
    nngm = treecorr.NNGCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    nngm.process(ncat, gcat)
    fm = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                 nphi_bins=nphi_bins).zeta.ravel()
    cov = nngm.estimate_cov('jackknife', func=fm)
    print('nng max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nng))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nng), atol=0.6*tol_factor)

    ngnm = treecorr.NGNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    ngnm.process(ncat, gcat, ncat)
    cov = ngnm.estimate_cov('jackknife', func=fm)
    print('ngn max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_ngn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_ngn), atol=0.6*tol_factor)

    gnnm = treecorr.GNNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    gnnm.process(gcat, ncat)
    cov = gnnm.estimate_cov('jackknife', func=fm)
    print('gnn max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_gnn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_gnn), atol=0.6*tol_factor)

    # Check with randoms
    cov_nng = nng_rr.estimate_cov('jackknife')
    print('with rand nng cov diag = ',np.diagonal(cov_nng))
    print('nng zeta var ratio = ',np.diagonal(cov_nng)/var_nng_rr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nng))-np.log(var_nng_rr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_nng)), np.log(var_nng_rr), atol=0.6*tol_factor)
    cov_nng = nng_dr.estimate_cov('jackknife')
    print('with rand/dr nng cov diag = ',np.diagonal(cov_nng))
    print('nng zeta var ratio = ',np.diagonal(cov_nng)/var_nng_dr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nng))-np.log(var_nng_dr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_nng)), np.log(var_nng_dr), atol=0.6*tol_factor)

    cov_ngn = ngn_rr.estimate_cov('jackknife')
    print('with rand ngn cov diag = ',np.diagonal(cov_ngn))
    print('ngn zeta var ratio = ',np.diagonal(cov_ngn)/var_ngn_rr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_ngn))-np.log(var_ngn_rr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_ngn)), np.log(var_ngn_rr), atol=0.6*tol_factor)
    cov_ngn = ngn_dr.estimate_cov('jackknife')
    print('with rand/dr ngn cov diag = ',np.diagonal(cov_ngn))
    print('ngn zeta var ratio = ',np.diagonal(cov_ngn)/var_ngn_dr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_ngn))-np.log(var_ngn_dr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_ngn)), np.log(var_ngn_dr), atol=0.6*tol_factor)

    cov_gnn = gnn_rr.estimate_cov('jackknife')
    print('with rand gnn cov diag = ',np.diagonal(cov_gnn))
    print('gnn zeta var ratio = ',np.diagonal(cov_gnn)/var_gnn_rr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_gnn))-np.log(var_gnn_rr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_gnn)), np.log(var_gnn_rr), atol=0.6*tol_factor)
    cov_gnn = gnn_dr.estimate_cov('jackknife')
    print('with rand/dr gnn cov diag = ',np.diagonal(cov_gnn))
    print('gnn zeta var ratio = ',np.diagonal(cov_gnn)/var_gnn_dr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_gnn))-np.log(var_gnn_dr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_gnn)), np.log(var_gnn_dr), atol=0.7*tol_factor)


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross21()
    test_varzeta_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross21()
    test_direct_logmultipole_cross()
    test_direct_logmultipole_cross21()
    test_nng_logsas()
    test_varzeta()
    test_nng_logsas_jk()

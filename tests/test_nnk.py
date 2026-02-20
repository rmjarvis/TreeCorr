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
    sig_kap = 0.2
    rng = np.random.RandomState(8675308)
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
    k3 = rng.normal(0,sig_kap, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3)

    min_sep = 1.
    bin_size = 0.2
    nrbins = 10
    min_u = 0.13
    max_u = 0.89
    nubins = 5
    min_v = 0.13
    max_v = 0.59
    nvbins = 5

    nnk = treecorr.NNKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    nkn = treecorr.NKNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    knn = treecorr.KNNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
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
    true_zeta_123 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_132 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_213 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_231 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_312 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_321 = np.zeros((nrbins, nubins, 2*nvbins))
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
                zeta = www * k3[k]

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

    nnk.process(cat1, cat2, cat3)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)

    nnk.process(cat2, cat1, cat3)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_213)
    np.testing.assert_allclose(nnk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_213, rtol=1.e-5)

    nkn.process(cat1, cat3, cat2)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    nkn.process(cat2, cat3, cat1)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_231)
    np.testing.assert_allclose(nkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_231, rtol=1.e-5)

    knn.process(cat3, cat1, cat2)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)
    knn.process(cat3, cat2, cat1)
    np.testing.assert_array_equal(knn.ntri, true_ntri_321)
    np.testing.assert_allclose(knn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 3
    nnk.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)

    nkn.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check bin_slop=0
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nnk.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=True)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # And again with no top-level recursion
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nnk.process(cat1, cat2, cat3, ordered=True)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=True)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1, cat2, cat3, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=False)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # With these, ordered=False is equivalent to the K vertex being fixed.
    nnk.process(cat1, cat2, cat3, ordered=3)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=2)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=1)
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # With no randoms, calculateZeta just returns zeta.
    np.testing.assert_allclose(nnk.calculateZeta()[0], true_zeta_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nkn.calculateZeta()[0], true_zeta_sum2, rtol=1.e-5)
    np.testing.assert_allclose(knn.calculateZeta()[0], true_zeta_sum1, rtol=1.e-5)

    # Compute zeta using randoms
    # We save these values to compare to later ones with patches.
    xr = rng.uniform(-3*s,3*s, (2*ngal,) )
    yr = rng.normal(-3*s,3*s, (2*ngal,) )
    rcat = treecorr.Catalog(x=xr, y=yr)

    rrk = nnk.copy()
    rrk.process(rcat, rcat, cat3, ordered=3)
    diff = nnk.raw_zeta - rrk.zeta
    zeta_nnk_rr, _ = nnk.calculateZeta(rrk=rrk)
    np.testing.assert_allclose(zeta_nnk_rr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, diff, rtol=1.e-5)
    drk = nnk.copy()
    drk.process(cat1, rcat, cat3, ordered=3)
    diff = nnk.raw_zeta - 2*drk.zeta + rrk.zeta
    zeta_nnk_dr, _ = nnk.calculateZeta(rrk=rrk, drk=drk)
    np.testing.assert_allclose(zeta_nnk_dr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, diff, rtol=1.e-5)
    rdk = nnk.copy()
    rdk.process(rcat, cat2, cat3, ordered=3)
    diff = nnk.raw_zeta - 2*rdk.zeta + rrk.zeta
    zeta_nnk_rd, _ = nnk.calculateZeta(rrk=rrk, rdk=rdk)
    np.testing.assert_allclose(nnk.zeta, diff, rtol=1.e-5)
    diff = nnk.raw_zeta - drk.zeta - rdk.zeta + rrk.zeta
    zeta_nnk_rdr, _ = nnk.calculateZeta(rrk=rrk, rdk=rdk, drk=drk)
    np.testing.assert_allclose(nnk.zeta, diff, rtol=1.e-5)

    rkr = nkn.copy()
    rkr.process(rcat, cat3, rcat, ordered=2)
    diff = nkn.raw_zeta - rkr.zeta
    zeta_nkn_rr, _ = nkn.calculateZeta(rkr=rkr)
    np.testing.assert_allclose(zeta_nkn_rr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, diff, rtol=1.e-5)
    dkr = nkn.copy()
    dkr.process(cat1, cat3, rcat, ordered=2)
    diff = nkn.raw_zeta - 2*dkr.zeta + rkr.zeta
    zeta_nkn_dr, _ = nkn.calculateZeta(rkr=rkr, dkr=dkr)
    np.testing.assert_allclose(zeta_nkn_dr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, diff, rtol=1.e-5)
    rkd = nkn.copy()
    rkd.process(rcat, cat3, cat2, ordered=2)
    diff = nkn.raw_zeta - 2*rkd.zeta + rkr.zeta
    zeta_nkn_rd, _ = nkn.calculateZeta(rkr=rkr, rkd=rkd)
    np.testing.assert_allclose(zeta_nkn_rd, diff, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, diff, rtol=1.e-5)
    diff = nkn.raw_zeta - dkr.zeta - rkd.zeta + rkr.zeta
    zeta_nkn_rdr, _ = nkn.calculateZeta(rkr=rkr, rkd=rkd, dkr=dkr)
    np.testing.assert_allclose(zeta_nkn_rdr, diff, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, diff, rtol=1.e-5)

    krr = knn.copy()
    krr.process(cat3, rcat, rcat, ordered=1)
    diff = knn.raw_zeta - krr.zeta
    zeta_knn_rr, _ = knn.calculateZeta(krr=krr)
    np.testing.assert_allclose(zeta_knn_rr, diff, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, diff, rtol=1.e-5)
    kdr = knn.copy()
    kdr.process(cat3, cat1, rcat, ordered=1)
    diff = knn.raw_zeta - 2*kdr.zeta + krr.zeta
    zeta_knn_dr, _ = knn.calculateZeta(krr=krr, kdr=kdr)
    np.testing.assert_allclose(zeta_knn_dr, diff, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, diff, rtol=1.e-5)
    krd = knn.copy()
    krd.process(cat3, rcat, cat2, ordered=1)
    diff = knn.raw_zeta - 2*krd.zeta + krr.zeta
    zeta_knn_rd, _ = knn.calculateZeta(krr=krr, krd=krd)
    np.testing.assert_allclose(zeta_knn_rd, diff, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, diff, rtol=1.e-5)
    diff = knn.raw_zeta - kdr.zeta - krd.zeta + krr.zeta
    zeta_knn_rdr, _ = knn.calculateZeta(krr=krr, krd=krd, kdr=kdr)
    np.testing.assert_allclose(zeta_knn_rdr, diff, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, diff, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        nnk.process(cat1, cat3=cat3)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=cat1p.patch_centers)

    # First test with just one catalog using patches
    nnk.process(cat1p, cat2, cat3)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1p, cat3, cat2)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3p, cat1, cat2)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1, cat2p, cat3)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1, cat3p, cat2)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3, cat1p, cat2)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1, cat2, cat3p)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2p)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3, cat1, cat2p)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    # Now all three patched
    nnk.process(cat1p, cat2p, cat3p)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p)
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    # Unordered
    nnk.process(cat1p, cat2p, cat3p, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=False)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=3)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=2)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=1)
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check using randoms with and without patches
    rcatp = treecorr.Catalog(x=xr, y=yr, patch_centers=cat1p.patch_centers)
    rrk.process(rcat, rcat, cat3p, ordered=False)
    nnk.calculateZeta(rrk=rrk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rr, rtol=1.e-5)
    rrk.process(rcatp, rcatp, cat3p, ordered=False)
    nnk.calculateZeta(rrk=rrk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rr, rtol=1.e-5)
    drk.process(cat1p, rcat, cat3p, ordered=False)
    nnk.calculateZeta(rrk=rrk, drk=drk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_dr, rtol=1.e-5)
    drk.process(cat1p, rcatp, cat3p, ordered=False)
    nnk.calculateZeta(rrk=rrk, drk=drk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_dr, rtol=1.e-5)
    rdk.process(rcat, cat2p, cat3p, ordered=False)
    nnk.calculateZeta(rrk=rrk, rdk=rdk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rd, rtol=1.e-5)
    rdk.process(rcatp, cat2p, cat3p, ordered=False)
    nnk.calculateZeta(rrk=rrk, rdk=rdk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rd, rtol=1.e-5)
    nnk.calculateZeta(rrk=rrk, rdk=rdk, drk=drk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rdr, rtol=1.e-5)
    nnk.calculateZeta(rrk=rrk, rdk=rdk, drk=drk)
    nnk.estimate_cov('jackknife')
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rdr, rtol=1.e-5)

    rkr.process(rcat, cat3p, rcat, ordered=False)
    nkn.calculateZeta(rkr=rkr)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rr, rtol=1.e-5)
    rkr.process(rcatp, cat3p, rcatp, ordered=False)
    nkn.calculateZeta(rkr=rkr)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rr, rtol=1.e-5)
    dkr.process(cat1p, cat3p, rcat, ordered=False)
    nkn.calculateZeta(rkr=rkr, dkr=dkr)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_dr, rtol=1.e-5)
    dkr.process(cat1p, cat3p, rcatp, ordered=False)
    nkn.calculateZeta(rkr=rkr, dkr=dkr)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_dr, rtol=1.e-5)
    rkd.process(rcat, cat3p, cat2p, ordered=False)
    nkn.calculateZeta(rkr=rkr, rkd=rkd)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rd, rtol=1.e-5)
    rkd.process(rcatp, cat3p, cat2p, ordered=False)
    nkn.calculateZeta(rkr=rkr, rkd=rkd)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rd, rtol=1.e-5)
    nkn.calculateZeta(rkr=rkr, rkd=rkd, dkr=dkr)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rdr, rtol=1.e-5)
    nkn.calculateZeta(rkr=rkr, rkd=rkd, dkr=dkr)
    nkn.estimate_cov('jackknife')
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rdr, rtol=1.e-5)

    krr.process(cat3p, rcat, rcat, ordered=False)
    knn.calculateZeta(krr=krr)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_rr, rtol=1.e-5)
    krr.process(cat3p, rcatp, rcatp, ordered=False)
    knn.calculateZeta(krr=krr)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_rr, rtol=1.e-5)
    kdr.process(cat3p, cat1p, rcat, ordered=False)
    knn.calculateZeta(krr=krr, kdr=kdr)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_dr, rtol=1.e-5)
    kdr.process(cat3p, cat1p, rcatp, ordered=False)
    knn.calculateZeta(krr=krr, kdr=kdr)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_dr, rtol=1.e-5)
    krd.process(cat3p, rcat, cat2p, ordered=False)
    knn.calculateZeta(krr=krr, krd=krd)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_rd, rtol=1.e-5)
    krd.process(cat3p, rcatp, cat2p, ordered=False)
    knn.calculateZeta(krr=krr, krd=krd)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_rd, rtol=1.e-5)
    knn.calculateZeta(krr=krr, krd=krd, kdr=kdr)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_rdr, rtol=1.e-5)
    knn.calculateZeta(krr=krr, krd=krd, kdr=kdr)
    knn.estimate_cov('jackknife')
    np.testing.assert_allclose(knn.zeta, zeta_knn_rdr, rtol=1.e-5)

    # Check when some patches have no objects
    rcatpx = treecorr.Catalog(x=xr, y=yr, npatch=20, rng=rng)
    cat1px = treecorr.Catalog(x=x1, y=y1, w=w1, patch_centers=rcatpx.patch_centers)
    cat2px = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=rcatpx.patch_centers)
    cat3px = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=rcatpx.patch_centers)
    nnk.process(cat1px, cat2px, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        nnk.calculateZeta(rrk=rrk)
    rrk.process(rcatpx, rcatpx, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        nnk.calculateZeta(rrk=rrk, rdk=rdk)
    with assert_raises(RuntimeError):
        nnk.calculateZeta(rrk=rrk, drk=drk)
    rdk.process(rcatpx, cat2px, cat3px, ordered=False)
    with assert_raises(RuntimeError):
        nnk.calculateZeta(rrk=rrk, drk=drk, rdk=rdk)
    drk.process(cat1px, rcatpx, cat3px, ordered=False)
    with assert_raises(TypeError):
        nnk.calculateZeta(drk=drk)
    with assert_raises(TypeError):
        nnk.calculateZeta(rdk=rdk)
    nnk.calculateZeta(rrk=rrk)
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rr, rtol=1.e-5)
    nnk.calculateZeta(rrk=rrk, rdk=rdk)
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rd, rtol=1.e-5)
    nnk.calculateZeta(rrk=rrk, drk=drk)
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_dr, rtol=1.e-5)
    nnk.calculateZeta(rrk=rrk, drk=drk, rdk=rdk)
    np.testing.assert_allclose(nnk.zeta, zeta_nnk_rdr, rtol=1.e-5)

    nkn.process(cat1px, cat3px, cat2px, ordered=False)
    with assert_raises(RuntimeError):
        nkn.calculateZeta(rkr=rkr)
    rkr.process(rcatpx, cat3px, rcatpx, ordered=False)
    with assert_raises(RuntimeError):
        nkn.calculateZeta(rkr=rkr, rkd=rkd)
    with assert_raises(RuntimeError):
        nkn.calculateZeta(rkr=rkr, dkr=dkr)
    rkd.process(rcatpx, cat3px, cat2px, ordered=False)
    with assert_raises(RuntimeError):
        nkn.calculateZeta(rkr=rkr, dkr=dkr, rkd=rkd)
    dkr.process(cat1px, cat3px, rcatpx, ordered=False)
    with assert_raises(TypeError):
        nkn.calculateZeta(dkr=dkr)
    with assert_raises(TypeError):
        nkn.calculateZeta(rkd=rkd)
    nkn.calculateZeta(rkr=rkr)
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rr, rtol=1.e-5)
    nkn.calculateZeta(rkr=rkr, dkr=dkr)
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_dr, rtol=1.e-5)
    nkn.calculateZeta(rkr=rkr, rkd=rkd)
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rd, rtol=1.e-5)
    nkn.calculateZeta(rkr=rkr, dkr=dkr, rkd=rkd)
    np.testing.assert_allclose(nkn.zeta, zeta_nkn_rdr, rtol=1.e-5)

    knn.process(cat3px, cat1px, cat2px, ordered=False)
    with assert_raises(RuntimeError):
        knn.calculateZeta(krr=krr)
    krr.process(cat3px, rcatpx, rcatpx, ordered=False)
    with assert_raises(RuntimeError):
        knn.calculateZeta(krr=krr, kdr=kdr)
    with assert_raises(RuntimeError):
        knn.calculateZeta(krr=krr, krd=krd)
    kdr.process(cat3px, cat1px, rcatpx, ordered=False)
    with assert_raises(RuntimeError):
        knn.calculateZeta(krr=krr, kdr=kdr, krd=krd)
    krd.process(cat3px, rcatpx, cat2px, ordered=False)
    with assert_raises(TypeError):
        knn.calculateZeta(kdr=kdr)
    with assert_raises(TypeError):
        knn.calculateZeta(krd=krd)
    knn.calculateZeta(krr=krr)
    np.testing.assert_allclose(knn.zeta, zeta_knn_rr, rtol=1.e-5)
    knn.calculateZeta(krr=krr, kdr=kdr)
    np.testing.assert_allclose(knn.zeta, zeta_knn_dr, rtol=1.e-5)
    knn.calculateZeta(krr=krr, krd=krd)
    np.testing.assert_allclose(knn.zeta, zeta_knn_rd, rtol=1.e-5)
    knn.calculateZeta(krr=krr, kdr=kdr, krd=krd)
    np.testing.assert_allclose(knn.zeta, zeta_knn_rdr, rtol=1.e-5)

    # patch_method=local
    nnk.process(cat1p, cat2p, cat3p, patch_method='local')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, patch_method='local')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, patch_method='local')
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=3, patch_method='local')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=2, patch_method='local')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=1, patch_method='local')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    with assert_raises(ValueError):
        nnk.process(cat1p, cat2p, cat3p, patch_method='nonlocal')
    with assert_raises(ValueError):
        nkn.process(cat1p, cat3p, cat2p, patch_method='nonlocal')
    with assert_raises(ValueError):
        knn.process(cat3p, cat1p, cat2p, patch_method='nonlocal')


@timer
def test_direct_logruv_cross21():
    # Check the 2-1 cross correlation

    ngal = 50
    s = 10.
    sig_kap = 0.2
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    k1 = rng.normal(0,sig_kap, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
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

    nnk = treecorr.NNKCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    nkn = treecorr.NKNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    knn = treecorr.KNNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  brute=True, bin_type='LogRUV')

    true_ntri_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_ntri_221 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_122 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_212 = np.zeros((nrbins, nubins, 2*nvbins))
    true_zeta_221 = np.zeros((nrbins, nubins, 2*nvbins))
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
                zeta = www * k1[i]

                true_ntri[kr,ku,kv] += 1
                true_weight[kr,ku,kv] += www
                true_zeta[kr,ku,kv] += zeta

    pos = true_weight_221 > 0
    true_zeta_221[pos] /= true_weight_221[pos]
    pos = true_weight_212 > 0
    true_zeta_212[pos] /= true_weight_212[pos]
    pos = true_weight_122 > 0
    true_zeta_122[pos] /= true_weight_122[pos]

    nnk.process(cat2, cat2, cat1)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2, cat1, cat2)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=1.e-5)
    knn.process(cat1, cat2, cat2)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with only 2 cat arguments
    # Note: NKN doesn't have a two-argument version.
    nnk.process(cat2, cat1)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1, cat2)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    with assert_raises(ValueError):
        nkn.process(cat2, cat1)
    with assert_raises(ValueError):
        nkn.process(cat1, cat2)
    with assert_raises(ValueError):
        nnk.process(cat1)
    with assert_raises(ValueError):
        nnk.process(cat2)
    with assert_raises(ValueError):
        nkn.process(cat1)
    with assert_raises(ValueError):
        nkn.process(cat2)
    with assert_raises(ValueError):
        knn.process(cat1)
    with assert_raises(ValueError):
        knn.process(cat2)

    # ordered=False doesn't do anything different, since there is no other valid order.
    nnk.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    # Repeat with binslop = 0
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, bin_type='LogRUV')

    nnk.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=1.e-5)
    knn.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    # And again with no top-level recursion
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nrbins,
                                  min_u=min_u, max_u=max_u, nubins=nubins,
                                  min_v=min_v, max_v=max_v, nvbins=nvbins,
                                  bin_slop=0, verbose=1, max_top=0, bin_type='LogRUV')

    nnk.process(cat2, cat1, ordered=True)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2, cat1, cat2, ordered=True)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=1.e-5)
    knn.process(cat1, cat2, ordered=True)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2, cat1, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1, cat2, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)

    nnk.process(cat2p, cat1p, ordered=True)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2p, cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=1.e-5)
    knn.process(cat1p, cat2p, ordered=True)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2p, cat1p, ordered=False)
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1p, cat2p, ordered=False)
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2p, cat1p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2p, cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=1.e-5)
    knn.process(cat1p, cat2p, ordered=True, patch_method='local')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2p, cat1p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1p, cat2p, ordered=False, patch_method='local')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)


@timer
def test_varzeta_logruv():
    # Test that the shot noise estimate of varzeta is close based on actual variance of many runs
    # when there is no real signal.  So should be just shot noise.

    # Put in a nominal pattern for k, but this pattern doesn't have much 3pt correlation.
    kappa0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(12345)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 30
    nsource = 50000

    file_name = 'data/test_varzeta_nnk_logruv.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nnks = []
        all_nkns = []
        all_knns = []

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
            k = kappa0 * np.exp(-r2/2.)
            k += rng.normal(0, 0.2, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1, x_units='arcmin', y_units='arcmin')
            kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k,
                                    x_units='arcmin', y_units='arcmin')
            nnk = treecorr.NNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                          bin_type='LogRUV')
            nkn = treecorr.NKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                          bin_type='LogRUV')
            knn = treecorr.KNNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                          sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                          bin_type='LogRUV')
            nnk.process(ncat, kcat)
            nkn.process(ncat, kcat, ncat)
            knn.process(kcat, ncat)
            all_nnks.append(nnk)
            all_nkns.append(nkn)
            all_knns.append(knn)

        mean_nnk_zeta = np.mean([nnk.zeta for nnk in all_nnks], axis=0)
        var_nnk_zeta = np.var([nnk.zeta for nnk in all_nnks], axis=0)
        mean_nnk_varzeta = np.mean([nnk.varzeta for nnk in all_nnks], axis=0)
        mean_nkn_zeta = np.mean([nkn.zeta for nkn in all_nkns], axis=0)
        var_nkn_zeta = np.var([nkn.zeta for nkn in all_nkns], axis=0)
        mean_nkn_varzeta = np.mean([nkn.varzeta for nkn in all_nkns], axis=0)
        mean_knn_zeta = np.mean([knn.zeta for knn in all_knns], axis=0)
        var_knn_zeta = np.var([knn.zeta for knn in all_knns], axis=0)
        mean_knn_varzeta = np.mean([knn.varzeta for knn in all_knns], axis=0)

        np.savez(file_name,
                 mean_nnk_zeta=mean_nnk_zeta,
                 var_nnk_zeta=var_nnk_zeta,
                 mean_nnk_varzeta=mean_nnk_varzeta,
                 mean_nkn_zeta=mean_nkn_zeta,
                 var_nkn_zeta=var_nkn_zeta,
                 mean_nkn_varzeta=mean_nkn_varzeta,
                 mean_knn_zeta=mean_knn_zeta,
                 var_knn_zeta=var_knn_zeta,
                 mean_knn_varzeta=mean_knn_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_nnk_zeta = data['mean_nnk_zeta']
    var_nnk_zeta = data['var_nnk_zeta']
    mean_nnk_varzeta = data['mean_nnk_varzeta']
    mean_nkn_zeta = data['mean_nkn_zeta']
    var_nkn_zeta = data['var_nkn_zeta']
    mean_nkn_varzeta = data['mean_nkn_varzeta']
    mean_knn_zeta = data['mean_knn_zeta']
    var_knn_zeta = data['var_knn_zeta']
    mean_knn_varzeta = data['mean_knn_varzeta']

    print('var_nnk_zeta = ',var_nnk_zeta)
    print('mean nnk_varzeta = ',mean_nnk_varzeta)
    print('ratio = ',var_nnk_zeta.ravel() / mean_nnk_varzeta.ravel())
    print('var_nkn_zeta = ',var_nkn_zeta)
    print('mean nkn_varzeta = ',mean_nkn_varzeta)
    print('ratio = ',var_nkn_zeta.ravel() / mean_nkn_varzeta.ravel())
    print('var_knn_zeta = ',var_knn_zeta)
    print('mean knn_varzeta = ',mean_knn_varzeta)
    print('ratio = ',var_knn_zeta.ravel() / mean_knn_varzeta.ravel())

    print('max relerr for nnk zeta = ',
          np.max(np.abs((var_nnk_zeta - mean_nnk_varzeta)/var_nnk_zeta)))
    np.testing.assert_allclose(mean_nnk_varzeta, var_nnk_zeta, rtol=0.05)

    print('max relerr for nkn zeta = ',
          np.max(np.abs((var_nkn_zeta - mean_nkn_varzeta)/var_nkn_zeta)))
    np.testing.assert_allclose(mean_nkn_varzeta, var_nkn_zeta, rtol=0.05)

    print('max relerr for knn zeta = ',
          np.max(np.abs((var_knn_zeta - mean_knn_varzeta)/var_knn_zeta)))
    np.testing.assert_allclose(mean_knn_varzeta, var_knn_zeta, rtol=0.05)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5
    r2 = (x2**2 + y2**2)/r0**2
    k = kappa0 * np.exp(-r2/2.)
    k += rng.normal(0, 0.2, size=nsource)

    ncat = treecorr.Catalog(x=x1, y=y1, x_units='arcmin', y_units='arcmin')
    kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k, x_units='arcmin', y_units='arcmin')
    nnk = treecorr.NNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                  bin_type='LogRUV')
    nkn = treecorr.NKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                  bin_type='LogRUV')
    knn = treecorr.KNNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_u=0.2, max_u=0.5, min_v=0.2, max_v=0.6,
                                  sep_units='arcmin', nubins=2, nvbins=2, verbose=1,
                                  bin_type='LogRUV')

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(nnk.cov, 0)
    np.testing.assert_array_equal(nnk.varzeta, 0)
    np.testing.assert_array_equal(nkn.cov, 0)
    np.testing.assert_array_equal(nkn.varzeta, 0)
    np.testing.assert_array_equal(knn.cov, 0)
    np.testing.assert_array_equal(knn.varzeta, 0)

    nnk.process(ncat, kcat)
    print('NNK single run:')
    print('max relerr for zeta = ',np.max(np.abs((nnk.varzeta - var_nnk_zeta)/var_nnk_zeta)))
    print('ratio = ',nnk.varzeta / var_nnk_zeta)
    np.testing.assert_allclose(nnk.varzeta, var_nnk_zeta, rtol=0.7)
    np.testing.assert_allclose(nnk.cov.diagonal(), nnk.varzeta.ravel())

    nkn.process(ncat, kcat, ncat)
    print('NKN single run:')
    print('max relerr for zeta = ',np.max(np.abs((nkn.varzeta - var_nkn_zeta)/var_nkn_zeta)))
    print('ratio = ',nkn.varzeta / var_nkn_zeta)
    np.testing.assert_allclose(nkn.varzeta, var_nkn_zeta, rtol=0.7)
    np.testing.assert_allclose(nkn.cov.diagonal(), nkn.varzeta.ravel())

    knn.process(kcat, ncat)
    print('KNN single run:')
    print('max relerr for zeta = ',np.max(np.abs((knn.varzeta - var_knn_zeta)/var_knn_zeta)))
    print('ratio = ',knn.varzeta / var_knn_zeta)
    np.testing.assert_allclose(knn.varzeta, var_knn_zeta, rtol=0.7)
    np.testing.assert_allclose(knn.cov.diagonal(), knn.varzeta.ravel())



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
    k3 = rng.normal(0,0.2, (ngal,) )
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 3

    # In this test set, we use the slow triangle algorithm.
    # We'll test the multipole algorithm below.
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  brute=True, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_123 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_132 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_213 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_231 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_312 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_321 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_123 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_132 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_213 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_231 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_312 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_321 = np.zeros((nbins, nbins, nphi_bins))
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

                www = w1[i] * w2[j] * w3[k]
                zeta = www * k3[k]

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

    nnk.process(cat1, cat2, cat3, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)

    nnk.process(cat2, cat1, cat3, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_213)
    np.testing.assert_allclose(nnk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_213, rtol=1.e-5)

    nkn.process(cat1, cat3, cat2, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)

    nkn.process(cat2, cat3, cat1, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_231)
    np.testing.assert_allclose(nkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_231, rtol=1.e-5)

    knn.process(cat3, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    knn.process(cat3, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_321)
    np.testing.assert_allclose(knn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_321, rtol=1.e-5)

    # With ordered=False, we end up with the sum of both versions where K is in 3
    nnk.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-4)

    nkn.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)

    knn.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Check binslop = 0
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    nnk.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # And again with no top-level recursion
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, max_top=0, bin_type='LogSAS')

    nnk.process(cat1, cat2, cat3, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=True, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1, cat2, cat3, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1, cat3, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    nnk.process(cat1, cat2, cat3, ordered=3, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-4)
    nkn.process(cat1, cat3, cat2, ordered=2, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3, cat1, cat2, ordered=1, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    # Error to have cat3, but not cat2
    with assert_raises(ValueError):
        nnk.process(cat1, cat3=cat3, algo='triangle')
    with assert_raises(ValueError):
        nkn.process(cat1, cat3=cat1, algo='triangle')
    with assert_raises(ValueError):
        knn.process(cat3, cat3=cat1, algo='triangle')

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=3, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=cat1p.patch_centers)

    nnk.process(cat1p, cat2p, cat3p, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=3, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=2, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=1, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_123)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_132)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_312)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    nnk.process(cat1p, cat2p, cat3p, ordered=3, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_sum3)
    np.testing.assert_allclose(nnk.weight, true_weight_sum3, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_sum3, rtol=1.e-5)
    nkn.process(cat1p, cat3p, cat2p, ordered=2, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_sum2)
    np.testing.assert_allclose(nkn.weight, true_weight_sum2, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_sum2, rtol=1.e-5)
    knn.process(cat3p, cat1p, cat2p, ordered=1, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_sum1)
    np.testing.assert_allclose(knn.weight, true_weight_sum1, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_sum1, rtol=1.e-5)

    with assert_raises(ValueError):
        nnk.process(cat1p, cat2p, cat3p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        nkn.process(cat1p, cat3p, cat2p, patch_method='nonlocal', algo='triangle')
    with assert_raises(ValueError):
        knn.process(cat3p, cat1p, cat2p, patch_method='nonlocal', algo='triangle')


@timer
def test_direct_logsas_cross21():
    # Check the 2-1 cross correlation

    ngal = 50
    s = 10.
    rng = np.random.RandomState(8675309)
    x1 = rng.normal(0,s, (ngal,) )
    y1 = rng.normal(0,s, (ngal,) )
    w1 = rng.random_sample(ngal)
    k1 = rng.normal(0,0.2, (ngal,) )
    cat1 = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1)
    x2 = rng.normal(0,s, (ngal,) )
    y2 = rng.normal(0,s, (ngal,) )
    w2 = rng.random_sample(ngal)
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2)

    min_sep = 1.
    max_sep = 10.
    nbins = 5
    nphi_bins = 7

    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                  nbins=nbins, nphi_bins=nphi_bins,
                                  bin_slop=0, bin_type='LogSAS')

    # Figure out the correct answer for each permutation
    true_ntri_122 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_212 = np.zeros((nbins, nbins, nphi_bins))
    true_ntri_221 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_122 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_212 = np.zeros((nbins, nbins, nphi_bins))
    true_zeta_221 = np.zeros((nbins, nbins, nphi_bins))
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

                www = w1[i] * w2[j] * w2[k]
                zeta = www * k1[i]

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

    nnk.process(cat2, cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)
    nnk.process(cat2, cat1, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-4, atol=1.e-6)

    nkn.process(cat2, cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=1.e-4, atol=1.e-6)

    knn.process(cat1, cat2, cat2, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)
    knn.process(cat1, cat2, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-4, atol=1.e-6)

    with assert_raises(ValueError):
        nkn.process(cat2, cat1)
    with assert_raises(ValueError):
        nkn.process(cat1, cat2)
    with assert_raises(ValueError):
        nnk.process(cat1)
    with assert_raises(ValueError):
        nnk.process(cat2)
    with assert_raises(ValueError):
        nkn.process(cat1)
    with assert_raises(ValueError):
        nkn.process(cat2)
    with assert_raises(ValueError):
        knn.process(cat1)
    with assert_raises(ValueError):
        knn.process(cat2)

    # With ordered=False, doesn't do anything different, since there is no other valid order.
    nnk.process(cat2, cat1, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)

    knn.process(cat1, cat2, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, k=k1, npatch=4, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1p.patch_centers)

    nnk.process(cat2p, cat1p, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2p, cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=2.e-5)
    knn.process(cat1p, cat2p, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2p, cat1p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1p, cat2p, ordered=False, algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2p, cat1p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    nkn.process(cat2p, cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nkn.ntri, true_ntri_212)
    np.testing.assert_allclose(nkn.weight, true_weight_212, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_212, rtol=2.e-5)
    knn.process(cat1p, cat2p, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)

    nnk.process(cat2p, cat1p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(nnk.ntri, true_ntri_221)
    np.testing.assert_allclose(nnk.weight, true_weight_221, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_221, rtol=1.e-5)
    knn.process(cat1p, cat2p, ordered=False, patch_method='local', algo='triangle')
    np.testing.assert_array_equal(knn.ntri, true_ntri_122)
    np.testing.assert_allclose(knn.weight, true_weight_122, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_122, rtol=1.e-5)


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
    k3 = rng.normal(0,0.2, (ngal,))
    cat3 = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3)

    min_sep = 1.
    max_sep = 30.
    nbins = 5
    max_n = 10

    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
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
                zeta = www * k3[k]

                # 123, 132
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
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

    nnk.process(cat1, cat2, cat3)
    np.testing.assert_allclose(nnk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-4)
    nnk.process(cat2, cat1, cat3)
    np.testing.assert_allclose(nnk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_213, rtol=1.e-4)
    nkn.process(cat1, cat3, cat2)
    np.testing.assert_allclose(nkn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-4)
    nkn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(nkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_231, rtol=1.e-4)
    knn.process(cat3, cat1, cat2)
    np.testing.assert_allclose(knn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-4)
    knn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(knn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_321, rtol=1.e-4)

    # Repeat with binslop = 0
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nnk.process(cat1, cat2, cat3)
    np.testing.assert_allclose(nnk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-4)
    nnk.process(cat2, cat1, cat3)
    np.testing.assert_allclose(nnk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_213, rtol=1.e-4)

    nkn.process(cat1, cat3, cat2)
    np.testing.assert_allclose(nkn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-4)
    nkn.process(cat2, cat3, cat1)
    np.testing.assert_allclose(nkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_231, rtol=1.e-4)

    knn.process(cat3, cat1, cat2)
    np.testing.assert_allclose(knn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-4)
    knn.process(cat3, cat2, cat1)
    np.testing.assert_allclose(knn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_321, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, patch_centers=cat1.patch_centers)
    cat3p = treecorr.Catalog(x=x3, y=y3, w=w3, k=k3, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    nnk.process(cat1p, cat2, cat3)
    np.testing.assert_allclose(nnk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-4)
    nnk.process(cat2p, cat1, cat3)
    np.testing.assert_allclose(nnk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_213, rtol=1.e-4)

    nkn.process(cat1p, cat3, cat2)
    np.testing.assert_allclose(nkn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-4)
    nkn.process(cat2p, cat3, cat1)
    np.testing.assert_allclose(nkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_231, rtol=1.e-4)

    knn.process(cat3p, cat1, cat2)
    np.testing.assert_allclose(knn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-4)
    knn.process(cat3p, cat2, cat1)
    np.testing.assert_allclose(knn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_321, rtol=1.e-4)

    # Now use all three patched
    nnk.process(cat1p, cat2p, cat3p)
    np.testing.assert_allclose(nnk.ntri, true_ntri_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_123, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_123, rtol=1.e-4)
    nnk.process(cat2p, cat1p, cat3p)
    np.testing.assert_allclose(nnk.ntri, true_ntri_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_213, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_213, rtol=1.e-4)

    nkn.process(cat1p, cat3p, cat2p)
    np.testing.assert_allclose(nkn.ntri, true_ntri_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_132, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_132, rtol=1.e-4)
    nkn.process(cat2p, cat3p, cat1p)
    np.testing.assert_allclose(nkn.ntri, true_ntri_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_231, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_231, rtol=1.e-4)

    knn.process(cat3p, cat1p, cat2p)
    np.testing.assert_allclose(knn.ntri, true_ntri_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_312, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_312, rtol=1.e-4)
    knn.process(cat3p, cat2p, cat1p)
    np.testing.assert_allclose(knn.ntri, true_ntri_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_321, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_321, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        nnk.process(cat1p, cat2p, cat3p, patch_method='global')
    with assert_raises(ValueError):
        nkn.process(cat1p, cat3p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        knn.process(cat3p, cat1p, cat2p, patch_method='global')

    # Test I/O
    for name, corr in zip(['nnk', 'nkn', 'knn'], [nnk, nkn, knn]):
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
    k2 = rng.normal(0,0.2, (ngal,))
    cat2 = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2)

    min_sep = 10.
    max_sep = 30.
    nbins = 5
    max_n = 8

    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  brute=True, bin_type='LogMultipole')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
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
                zeta = www * k2[k]

                # 112, 121
                if d2 >= min_sep and d2 < max_sep and d3 >= min_sep and d3 < max_sep:
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
                    assert 0 <= kr1 < nbins
                    assert 0 <= kr2 < nbins
                    phi = np.arccos((d1**2 + d2**2 - d3**2)/(2*d1*d2))
                    if not is_ccw(x1[i],y1[i],x2[k],y2[k],x1[j],y1[j]):
                        phi = -phi
                    true_zeta_211[kr1,kr2,:] += zeta * np.exp(-1j * n1d * phi)
                    true_weight_211[kr1,kr2,:] += www * np.exp(-1j * n1d * phi)
                    true_ntri_211[kr1,kr2,:] += 1

    nnk.process(cat1, cat1, cat2)
    np.testing.assert_allclose(nnk.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_112, rtol=1.e-4)
    nkn.process(cat1, cat2, cat1)
    np.testing.assert_allclose(nkn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_121, rtol=1.e-4)
    # 3 arg version doesn't work for knn because the knn processing doesn't know cat2 and cat3
    # are actually the same, so it doesn't subtract off the duplicates.

    # 2 arg version
    nnk.process(cat1, cat2)
    np.testing.assert_allclose(nnk.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_112, rtol=1.e-4)
    knn.process(cat2, cat1)
    np.testing.assert_allclose(knn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_211, rtol=1.e-4)

    # Repeat with binslop = 0
    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, max_n=max_n,
                                  bin_slop=0, angle_slop=0, max_top=2, bin_type='LogMultipole')
    nnk.process(cat1, cat2)
    np.testing.assert_allclose(nnk.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_112, rtol=1.e-4)

    nkn.process(cat1, cat2, cat1)
    np.testing.assert_allclose(nkn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_121, rtol=1.e-4)

    knn.process(cat2, cat1)
    np.testing.assert_allclose(knn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_211, rtol=1.e-4)

    # Split into patches to test the list-based version of the code.
    cat1p = treecorr.Catalog(x=x1, y=y1, w=w1, npatch=2, rng=rng)
    cat2p = treecorr.Catalog(x=x2, y=y2, w=w2, k=k2, patch_centers=cat1.patch_centers)

    # First test with just one catalog using patches
    nnk.process(cat1p, cat2)
    np.testing.assert_allclose(nnk.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_112, rtol=1.e-4)

    nkn.process(cat1p, cat2, cat1)
    np.testing.assert_allclose(nkn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_121, rtol=1.e-4)

    knn.process(cat2p, cat1)
    np.testing.assert_allclose(knn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_211, rtol=1.e-4)

    # Now use both patched
    nnk.process(cat1p, cat2p)
    np.testing.assert_allclose(nnk.ntri, true_ntri_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.weight, true_weight_112, rtol=1.e-5)
    np.testing.assert_allclose(nnk.zeta, true_zeta_112, rtol=1.e-4)

    nkn.process(cat1p, cat2p, cat1p)
    np.testing.assert_allclose(nkn.ntri, true_ntri_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.weight, true_weight_121, rtol=1.e-5)
    np.testing.assert_allclose(nkn.zeta, true_zeta_121, rtol=1.e-4)

    knn.process(cat2p, cat1p)
    np.testing.assert_allclose(knn.ntri, true_ntri_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.weight, true_weight_211, rtol=1.e-5)
    np.testing.assert_allclose(knn.zeta, true_zeta_211, rtol=1.e-4)

    # local is default, and here global is not allowed.
    with assert_raises(ValueError):
        nnk.process(cat1p, cat2p, patch_method='global')
    with assert_raises(ValueError):
        nkn.process(cat1p, cat2p, cat1p, patch_method='global')
    with assert_raises(ValueError):
        knn.process(cat2p, cat1p, patch_method='global')


@timer
def test_nnk_logsas():
    # For this test, we need coherent pattern that gives an NNK signal.
    # We take pairs of "lens" points and construct two dipoles that point towards each
    # other, so the space between the two points has positive kappa, and there is compensating
    # negative values around it.
    #
    # (x1,y1) = random
    # phi = random
    # (x2,y2) = x1 + 2r0 cos(phi), y1 + 2r0 sin(phia)
    #
    # kappa = k0 [exp(-r1^2/2r0^2) r1 cos(theta1-phi) - exp(-r2^2/2r0^2) r2 cos(theta2-phi)]
    #
    # where r1, r2 are distances from points p1, p2 respectively and theta1, theta2 are the
    # azimuthal angles around p1, p2.
    #

    r0 = 10.
    if __name__ == '__main__':
        nlens = 200
        nsource = 400000
        L = 40. * r0
        min_ntri = 500
        tol_factor = 1
    else:
        # Looser tests that don't take so long to run.
        nlens = 100
        nsource = 100000
        L = 30. * r0
        min_ntri = 90
        tol_factor = 3

    rng = np.random.RandomState(8675309)
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    phi = rng.random_sample(nlens) * 2*np.pi
    x2 = x1 + r0 * np.cos(phi)
    y2 = y1 + r0 * np.sin(phi)

    x3 = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    y3 = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    dx1 = x3[:,None]-x1[None,:]
    dy1 = y3[:,None]-y1[None,:]
    theta1 = np.arctan2(dy1,dx1)
    r1 = np.sqrt(dx1**2 + dy1**2) / r0
    dx2 = x3[:,None]-x2[None,:]
    dy2 = y3[:,None]-y2[None,:]
    theta2 = np.arctan2(dy2,dx2)
    r2 = np.sqrt(dx2**2 + dy2**2) / r0

    # kappa = k0 [exp(-r1^2/2r0^2) r1 cos(theta1-phi) - exp(-r2^2/2r0^2) r2 cos(theta2-phi)]
    k0 = 0.07
    kappa = k0 * (np.exp(-r1**2/2) * r1 * np.cos(theta1-phi)
                  - np.exp(-r2**2/2) * r2 * np.cos(theta2-phi))
    kappa = np.sum(kappa, axis=1)

    ncat = treecorr.Catalog(x=np.concatenate([x1,x2]), y=np.concatenate([y1,y2]))
    kcat = treecorr.Catalog(x=x3, y=y3, k=kappa)

    nrand = 10*nlens
    xr = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    yr = (rng.random_sample(nsource)-0.5) * (L + 2*r0)
    rcat = treecorr.Catalog(x=xr, y=yr)

    min_sep = 8
    max_sep = 12
    nbins = 5
    nphi_bins = 30

    nnk = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nphi_bins=nphi_bins, bin_type='LogSAS')
    nkn = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nphi_bins=nphi_bins, bin_type='LogSAS')
    knn = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                  nphi_bins=nphi_bins, bin_type='LogSAS')

    for name, corr in zip(['nnk', 'nkn', 'knn'], [nnk, nkn, knn]):

        t0 = time.time()
        if name == 'nnk':
            corr.process(ncat, kcat, algo='triangle')
        elif name == 'nkn':
            corr.process(ncat, kcat, ncat, algo='triangle')
        else:
            corr.process(kcat, ncat, algo='triangle')
        t1 = time.time()
        print(name,'process time = ',t1-t0)

        if name == 'nnk':
            # Use this as the canonical definition of d1,d2,d3.
            # Others will need to permute their names to match so 1,2 have N and 3 has K.
            d1 = corr.meand1
            d2 = corr.meand2
            d3 = corr.meand3
            theta2 = corr.meanphi
            theta1 = np.arccos( (d1**2 + d3**2 - d2**2)/(2*d1*d3) )
        elif name == 'nkn':
            d1 = corr.meand1
            d2 = corr.meand3
            d3 = corr.meand2
            theta2 = corr.meanphi
            theta1 = np.arccos( (d1**2 + d3**2 - d2**2)/(2*d1*d3) )
        else:
            d1 = corr.meand3
            d2 = corr.meand2
            d3 = corr.meand1
            theta2 = np.arccos( (d2**2 + d3**2 - d1**2)/(2*d2*d3) )
            theta1 = np.arccos( (d1**2 + d3**2 - d2**2)/(2*d1*d3) )

        # Note: different sign here and no -phi because of different way of calculating theta1,2.
        true_zeta = k0 * (np.exp(-d1**2/(2*r0**2)) * d1/r0 * np.cos(theta1)
                          + np.exp(-d2**2/(2*r0**2)) * d2/r0 * np.cos(theta2))

        # Expect signal when d3 ~= r0 = 10 and theta < ~pi/2  (and not too close to 0)
        if name == 'knn':
            # Because knn doesn't bin directly on d3 (which it calls d1, opposite the phi vertex),
            # some of the bins that nominally should be good don't have many triangles.  These
            # bins don't match nearly as well as the ones with at least 500 triangles, so limit
            # to those bins.
            m = np.where((d3 > 9.7) & (d3 < 10.3) & (theta2 > 0.1) & (theta2 < 1.6) &
                         (corr.ntri > min_ntri))
        else:
            m = np.where((d3 > 9.7) & (d3 < 10.3) & (theta2 > 0.1) & (theta2 < 1.6))
        # For the other two, they already have ntri > 500
        print('ntri = ',corr.ntri[m])
        assert np.all(corr.ntri[m] > min_ntri)
        print('max diff = ',np.max(np.abs(corr.zeta[m] - true_zeta[m])))
        print('max rel diff = ',np.max(np.abs((corr.zeta[m] - true_zeta[m])/true_zeta[m])))
        np.testing.assert_allclose(corr.zeta[m], true_zeta[m], rtol=0.2*tol_factor)
        print('mean diff = ',np.mean(corr.zeta[m] - true_zeta[m]))
        print('mean zeta = ',np.mean(true_zeta[m]))
        assert np.abs(np.mean(corr.zeta[m] - true_zeta[m])) < 0.1 * np.mean(true_zeta[m])

        # Repeat this using Multipole and then convert to SAS:
        t0 = time.time()
        if name == 'nnk':
            corrm = treecorr.NNKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                            max_n=120, bin_type='LogMultipole')
            corrm.process(ncat, kcat)
        elif name == 'nkn':
            corrm = treecorr.NKNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                            max_n=120, bin_type='LogMultipole')
            corrm.process(ncat, kcat, ncat)
        else:
            corrm = treecorr.KNNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
                                            max_n=120, bin_type='LogMultipole')
            corrm.process(kcat, ncat)
        corrs = corrm.toSAS(nphi_bins=nphi_bins)
        t1 = time.time()
        print('time for multipole corr:', t1-t0)

        print('"true" zeta = ',true_zeta[m])
        print('zeta from multipole = ',corrs.zeta[m])
        print('zeta from triangle = ',corr.zeta[m])
        print('zeta mean ratio = ',np.mean(corrs.zeta[m] / corr.zeta[m]))
        print('zeta mean diff = ',np.mean(corrs.zeta[m] - corr.zeta[m]))
        np.testing.assert_allclose(corrs.zeta[m], corr.zeta[m], rtol=0.2*tol_factor)
        np.testing.assert_allclose(np.mean(corrs.zeta[m] / corr.zeta[m]), 1., rtol=0.02*tol_factor)
        np.testing.assert_allclose(np.std(corrs.zeta[m] / corr.zeta[m]), 0., atol=0.08*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd1[m], corr.meanlogd1[m], rtol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd2[m], corr.meanlogd2[m], rtol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanlogd3[m], corr.meanlogd3[m], rtol=0.1*tol_factor)
        np.testing.assert_allclose(corrs.meanphi[m], corr.meanphi[m], rtol=0.1*tol_factor)

        # Error to try to change sep binning with toSAS
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, min_sep=5)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, max_sep=25)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, nbins=20)
        with assert_raises(ValueError):
            corrs = corrm.toSAS(nphi_bins=nphi_bins, bin_size=0.01, nbins=None)
        # Error if non-Multipole calls toSAS
        with assert_raises(TypeError):
            corrs.toSAS()

        # All of the above is the default algorithm if process doesn't set algo='triangle'.
        # Check the automatic use of the multipole algorithm from LogSAS.
        corr3 = corr.copy()
        t0 = time.time()
        if name == 'nnk':
            corr3.process(ncat, kcat, max_n=120)
        elif name == 'nkn':
            corr3.process(ncat, kcat, ncat, max_n=120)
        else:
            corr3.process(kcat, ncat, max_n=120)
        t1 = time.time()
        print(name,'normal process time = ',t1-t0)

        np.testing.assert_allclose(corr3.weight, corrs.weight)
        np.testing.assert_allclose(corr3.zeta, corrs.zeta)

        # Now use randoms
        corr2 = corr.copy()
        rr = corr.copy()
        if name == 'nnk':
            rr.process(rcat, kcat)
            czkwargs = dict(rrk=rr)
        elif name == 'nkn':
            rr.process(rcat, kcat, rcat)
            czkwargs = dict(rkr=rr)
        else:
            rr.process(kcat, rcat)
            czkwargs = dict(krr=rr)
        zeta, varzeta = corr2.calculateZeta(**czkwargs)

        print('with rand mean ratio = ',np.mean(corr2.zeta[m]/true_zeta[m]))
        np.testing.assert_allclose(corr2.zeta[m], true_zeta[m], rtol=0.2*tol_factor)

        corr2x = corr2.copy()
        np.testing.assert_allclose(corr2x.zeta, corr2.zeta)
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
        kcat.write(os.path.join('data',name+'_kdata_logsas.dat'))
        config = treecorr.config.read_config('configs/'+name+'_logsas.yaml')
        config['verbose'] = 0
        treecorr.corr3(config)
        corr3_output = np.genfromtxt(os.path.join('output',name+'_logsas.out'),
                                     names=True, skip_header=1)
        np.testing.assert_allclose(corr3_output['zeta'], corr3.zeta.flatten(), rtol=1.e-3, atol=0)

        if name == 'nkn':
            # Invalid to omit file_name2
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)
        else:
            # Invalid to call cat2 file_name3 rather than file_name2
            config['file_name3'] = config['file_name2']
            if name == 'nnk':
                config['k_col'] = [0,0,3]
            else:
                config['k_col'] = [3,0,0]
            del config['file_name2']
            with assert_raises(TypeError):
                treecorr.corr3(config)

        # Check the fits write option
        try:
            import fitsio
        except ImportError:
            pass
        else:
            out_file_name = os.path.join('output','corr_nnk_logsas.fits')
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
            np.testing.assert_allclose(data['zeta'], corr.zeta.real.flatten())
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

    # Put in a nominal pattern for k, but this pattern doesn't have much 3pt correlation.
    kappa0 = 0.05
    r0 = 10.
    L = 50.*r0
    rng = np.random.RandomState(1234)

    # Note: to get a good estimate of var(xi), you need a lot of runs.  The number of
    # runs matters much more than the number of galaxies for getting this to pass.
    nruns = 50000

    nlens = 100
    nsource = 100000

    file_name = 'data/test_varzeta_nnk.npz'
    print(file_name)
    if not os.path.isfile(file_name):
        all_nnks = []
        all_nkns = []
        all_knns = []

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
            k = kappa0 * np.exp(-r2/2.)
            k += rng.normal(0, 0.2, size=nsource)

            ncat = treecorr.Catalog(x=x1, y=y1)
            kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k)
            nnk = treecorr.NNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            nkn = treecorr.NKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            knn = treecorr.KNNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                          min_phi=0.3, max_phi=2.8, nphi_bins=20)
            nnk.process(ncat, kcat)
            nkn.process(ncat, kcat, ncat)
            knn.process(kcat, ncat)
            all_nnks.append(nnk)
            all_nkns.append(nkn)
            all_knns.append(knn)

        mean_nnk_zeta = np.mean([nnk.zeta for nnk in all_nnks], axis=0)
        var_nnk_zeta = np.var([nnk.zeta for nnk in all_nnks], axis=0)
        mean_nnk_varzeta = np.mean([nnk.varzeta for nnk in all_nnks], axis=0)
        mean_nkn_zeta = np.mean([nkn.zeta for nkn in all_nkns], axis=0)
        var_nkn_zeta = np.var([nkn.zeta for nkn in all_nkns], axis=0)
        mean_nkn_varzeta = np.mean([nkn.varzeta for nkn in all_nkns], axis=0)
        mean_knn_zeta = np.mean([knn.zeta for knn in all_knns], axis=0)
        var_knn_zeta = np.var([knn.zeta for knn in all_knns], axis=0)
        mean_knn_varzeta = np.mean([knn.varzeta for knn in all_knns], axis=0)

        np.savez(file_name,
                 mean_nnk_zeta=mean_nnk_zeta,
                 var_nnk_zeta=var_nnk_zeta,
                 mean_nnk_varzeta=mean_nnk_varzeta,
                 mean_nkn_zeta=mean_nkn_zeta,
                 var_nkn_zeta=var_nkn_zeta,
                 mean_nkn_varzeta=mean_nkn_varzeta,
                 mean_knn_zeta=mean_knn_zeta,
                 var_knn_zeta=var_knn_zeta,
                 mean_knn_varzeta=mean_knn_varzeta)

    data = np.load(file_name)
    print('nruns = ',nruns)

    mean_nnk_zeta = data['mean_nnk_zeta']
    var_nnk_zeta = data['var_nnk_zeta']
    mean_nnk_varzeta = data['mean_nnk_varzeta']
    mean_nkn_zeta = data['mean_nkn_zeta']
    var_nkn_zeta = data['var_nkn_zeta']
    mean_nkn_varzeta = data['mean_nkn_varzeta']
    mean_knn_zeta = data['mean_knn_zeta']
    var_knn_zeta = data['var_knn_zeta']
    mean_knn_varzeta = data['mean_knn_varzeta']

    print('var_nnk_zeta = ',var_nnk_zeta)
    print('mean nnk_varzeta = ',mean_nnk_varzeta)
    print('ratio = ',var_nnk_zeta.ravel() / mean_nnk_varzeta.ravel())
    print('var_nkn_zeta = ',var_nkn_zeta)
    print('mean nkn_varzeta = ',mean_nkn_varzeta)
    print('ratio = ',var_nkn_zeta.ravel() / mean_nkn_varzeta.ravel())
    print('var_knn_zeta = ',var_knn_zeta)
    print('mean knn_varzeta = ',mean_knn_varzeta)
    print('ratio = ',var_knn_zeta.ravel() / mean_knn_varzeta.ravel())

    print('max relerr for nnk zeta = ',
          np.max(np.abs((var_nnk_zeta - mean_nnk_varzeta)/var_nnk_zeta)))
    np.testing.assert_allclose(mean_nnk_varzeta, var_nnk_zeta, rtol=0.15)

    print('max relerr for nkn zeta = ',
          np.max(np.abs((var_nkn_zeta - mean_nkn_varzeta)/var_nkn_zeta)))
    np.testing.assert_allclose(mean_nkn_varzeta, var_nkn_zeta, rtol=0.15)

    print('max relerr for knn zeta = ',
          np.max(np.abs((var_knn_zeta - mean_knn_varzeta)/var_knn_zeta)))
    np.testing.assert_allclose(mean_knn_varzeta, var_knn_zeta, rtol=0.20)

    # Now the actual test that's based on current code, not just from the saved file.
    # There is a bit more noise on a singe run, so the tolerance needs to be somewhat higher.
    x1 = (rng.random_sample(nlens)-0.5) * L
    y1 = (rng.random_sample(nlens)-0.5) * L
    x2 = (rng.random_sample(nsource)-0.5) * L
    y2 = (rng.random_sample(nsource)-0.5) * L
    # Varied weights are hard, but at least check that non-unit weights work correctly.
    w = np.ones_like(x2) * 5
    r2 = (x2**2 + y2**2)/r0**2
    k = kappa0 * np.exp(-r2/2.)
    k += rng.normal(0, 0.2, size=nsource)

    ncat = treecorr.Catalog(x=x1, y=y1)
    kcat = treecorr.Catalog(x=x2, y=y2, w=w, k=k)
    nnk = treecorr.NNKCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)
    nkn = treecorr.NKNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)
    knn = treecorr.KNNCorrelation(nbins=2, min_sep=50., max_sep=70.,
                                  min_phi=0.3, max_phi=2.8, nphi_bins=20)

    # Before running process, varzeta and cov are allowed, but all 0.
    np.testing.assert_array_equal(nnk.cov, 0)
    np.testing.assert_array_equal(nnk.varzeta, 0)
    np.testing.assert_array_equal(nkn.cov, 0)
    np.testing.assert_array_equal(nkn.varzeta, 0)
    np.testing.assert_array_equal(knn.cov, 0)
    np.testing.assert_array_equal(knn.varzeta, 0)

    nnk.process(ncat, kcat)
    print('NNK single run:')
    print('max relerr for zeta = ',np.max(np.abs((nnk.varzeta - var_nnk_zeta)/var_nnk_zeta)))
    print('ratio = ',nnk.varzeta / var_nnk_zeta)
    np.testing.assert_allclose(nnk.varzeta, var_nnk_zeta, rtol=0.5)
    np.testing.assert_allclose(nnk.cov.diagonal(), nnk.varzeta.ravel())

    nkn.process(ncat, kcat, ncat)
    print('NKN single run:')
    print('max relerr for zeta = ',np.max(np.abs((nkn.varzeta - var_nkn_zeta)/var_nkn_zeta)))
    print('ratio = ',nkn.varzeta / var_nkn_zeta)
    np.testing.assert_allclose(nkn.varzeta, var_nkn_zeta, rtol=0.5)
    np.testing.assert_allclose(nkn.cov.diagonal(), nkn.varzeta.ravel())

    knn.process(kcat, ncat)
    print('KNN single run:')
    print('max relerr for zeta = ',np.max(np.abs((knn.varzeta - var_knn_zeta)/var_knn_zeta)))
    print('ratio = ',knn.varzeta / var_knn_zeta)
    np.testing.assert_allclose(knn.varzeta, var_knn_zeta, rtol=0.6)
    np.testing.assert_allclose(knn.cov.diagonal(), knn.varzeta.ravel())

@timer
def test_nnk_logsas_jk():
    # Test jackknife covariance estimates for nnk correlations with LogSAS binning.

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
        npatch = 16
        tol_factor = 3

    nbins = 2
    min_sep = 12
    max_sep = 16
    nphi_bins = 5
    min_phi = 30
    max_phi = 90

    file_name = 'data/test_nnk_logsas_jk_{}.npz'.format(nsource)
    print(file_name)
    if not os.path.isfile(file_name):
        nruns = 1000
        all_nnk = []
        all_nkn = []
        all_knn = []
        all_nnk_rr = []
        all_nkn_rr = []
        all_knn_rr = []
        all_nnk_dr = []
        all_nkn_dr = []
        all_knn_dr = []
        rng1 = np.random.default_rng()
        for run in range(nruns):
            x, y, _, _, k, xh, yh = generate_shear_field(nsource, nhalo, rng1, return_halos=True)
            k -= np.mean(k)
            xr = rng1.uniform(0, 1000, size=10*nhalo)
            yr = rng1.uniform(0, 1000, size=10*nhalo)
            print(run,': ',np.std(k))
            kcat = treecorr.Catalog(x=x, y=y, k=k)
            ncat = treecorr.Catalog(x=xh, y=yh)
            rcat = treecorr.Catalog(x=xr, y=yr)
            nnk = treecorr.NNKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            nnk.process(ncat, kcat)
            all_nnk.append(nnk.zeta.ravel())
            rrk = nnk.copy()
            rrk.process(rcat, kcat)
            drk = nnk.copy()
            drk.process(ncat, rcat, kcat, ordered=3)
            zeta, _ = nnk.calculateZeta(rrk=rrk)
            all_nnk_rr.append(zeta.ravel())
            zeta, _ = nnk.calculateZeta(rrk=rrk, drk=drk)
            all_nnk_dr.append(zeta.ravel())

            nkn = treecorr.NKNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            nkn.process(ncat, kcat, ncat)
            all_nkn.append(nkn.zeta.ravel())
            rkr = nkn.copy()
            rkr.process(rcat, kcat, rcat)
            dkr = nkn.copy()
            dkr.process(ncat, kcat, rcat, ordered=2)
            zeta, _ = nkn.calculateZeta(rkr=rkr)
            all_nkn_rr.append(zeta.ravel())
            zeta, _ = nkn.calculateZeta(rkr=rkr, dkr=dkr)
            all_nkn_dr.append(zeta.ravel())

            knn = treecorr.KNNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                          min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                          nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
            knn.process(kcat, ncat)
            all_knn.append(knn.zeta.ravel())
            krr = knn.copy()
            krr.process(kcat, rcat)
            kdr = knn.copy()
            kdr.process(kcat, ncat, rcat, ordered=1)
            zeta, _ = knn.calculateZeta(krr=krr)
            all_knn_rr.append(zeta.ravel())
            zeta, _ = knn.calculateZeta(krr=krr, kdr=kdr)
            all_knn_dr.append(zeta.ravel())

        mean_nnk = np.mean(all_nnk, axis=0)
        var_nnk = np.var(all_nnk, axis=0)
        mean_nkn = np.mean(all_nkn, axis=0)
        var_nkn = np.var(all_nkn, axis=0)
        mean_knn = np.mean(all_knn, axis=0)
        var_knn = np.var(all_knn, axis=0)
        mean_nnk_rr = np.mean(all_nnk_rr, axis=0)
        var_nnk_rr = np.var(all_nnk_rr, axis=0)
        mean_nkn_rr = np.mean(all_nkn_rr, axis=0)
        var_nkn_rr = np.var(all_nkn_rr, axis=0)
        mean_knn_rr = np.mean(all_knn_rr, axis=0)
        var_knn_rr = np.var(all_knn_rr, axis=0)
        mean_nnk_dr = np.mean(all_nnk_dr, axis=0)
        var_nnk_dr = np.var(all_nnk_dr, axis=0)
        mean_nkn_dr = np.mean(all_nkn_dr, axis=0)
        var_nkn_dr = np.var(all_nkn_dr, axis=0)
        mean_knn_dr = np.mean(all_knn_dr, axis=0)
        var_knn_dr = np.var(all_knn_dr, axis=0)

        np.savez(file_name,
                 mean_nnk=mean_nnk, var_nnk=var_nnk,
                 mean_nkn=mean_nkn, var_nkn=var_nkn,
                 mean_knn=mean_knn, var_knn=var_knn,
                 mean_nnk_rr=mean_nnk_rr, var_nnk_rr=var_nnk_rr,
                 mean_nkn_rr=mean_nkn_rr, var_nkn_rr=var_nkn_rr,
                 mean_knn_rr=mean_knn_rr, var_knn_rr=var_knn_rr,
                 mean_nnk_dr=mean_nnk_dr, var_nnk_dr=var_nnk_dr,
                 mean_nkn_dr=mean_nkn_dr, var_nkn_dr=var_nkn_dr,
                 mean_knn_dr=mean_knn_dr, var_knn_dr=var_knn_dr)

    data = np.load(file_name)
    mean_nnk = data['mean_nnk']
    var_nnk = data['var_nnk']
    mean_nkn = data['mean_nkn']
    var_nkn = data['var_nkn']
    mean_knn = data['mean_knn']
    var_knn = data['var_knn']
    mean_nnk_rr = data['mean_nnk_rr']
    var_nnk_rr = data['var_nnk_rr']
    mean_nkn_rr = data['mean_nkn_rr']
    var_nkn_rr = data['var_nkn_rr']
    mean_knn_rr = data['mean_knn_rr']
    var_knn_rr = data['var_knn_rr']
    mean_nnk_dr = data['mean_nnk_dr']
    var_nnk_dr = data['var_nnk_dr']
    mean_nkn_dr = data['mean_nkn_dr']
    var_nkn_dr = data['var_nkn_dr']
    mean_knn_dr = data['mean_knn_dr']
    var_knn_dr = data['var_knn_dr']
    print('mean nnk = ',mean_nnk)
    print('var nnk = ',var_nnk)
    print('mean nkn = ',mean_nkn)
    print('var nkn = ',var_nkn)
    print('mean knn = ',mean_knn)
    print('var knn = ',var_knn)

    rng = np.random.default_rng(1234)
    x, y, _, _, k, xh, yh = generate_shear_field(nsource, nhalo, rng, return_halos=True)
    k -= np.mean(k)
    xr = rng.uniform(0, 1000, size=10*nhalo)
    yr = rng.uniform(0, 1000, size=10*nhalo)
    kcat = treecorr.Catalog(x=x, y=y, k=k, npatch=npatch, rng=rng)
    ncat = treecorr.Catalog(x=xh, y=yh, rng=rng, patch_centers=kcat.patch_centers)
    rcat = treecorr.Catalog(x=xr, y=yr, rng=rng, patch_centers=kcat.patch_centers)

    # First check calculate_xi with all pairs in results dict.
    nnk = treecorr.NNKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    nnk.process(ncat, kcat)
    nnk2 = nnk.copy()
    nnk2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nnk.results.keys()], False)
    np.testing.assert_allclose(nnk2.ntri, nnk.ntri, rtol=0.01)
    np.testing.assert_allclose(nnk2.zeta, nnk.zeta, rtol=0.01)
    np.testing.assert_allclose(nnk2.varzeta, nnk.varzeta, rtol=0.01)
    rrk = nnk.copy()
    rrk.process(rcat, kcat)
    nnk_rr = nnk.copy()
    zeta_rrk, varzeta_rrk = nnk_rr.calculateZeta(rrk=rrk)
    nnk2 = nnk_rr.copy()
    nnk2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nnk.results.keys()], False)
    np.testing.assert_allclose(nnk2.ntri, nnk_rr.ntri, rtol=0.01)
    np.testing.assert_allclose(nnk2.zeta, nnk_rr.zeta, rtol=0.01)
    np.testing.assert_allclose(nnk2.varzeta, nnk_rr.varzeta, rtol=0.01)
    np.testing.assert_allclose(nnk2.zeta, zeta_rrk, rtol=0.01)
    np.testing.assert_allclose(nnk2.varzeta, varzeta_rrk, rtol=0.01)
    drk = nnk.copy()
    drk.process(ncat, rcat, kcat, ordered=3)
    nnk_dr = nnk.copy()
    zeta_drk, varzeta_drk = nnk_dr.calculateZeta(rrk=rrk, drk=drk)
    nnk2 = nnk_dr.copy()
    nnk2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nnk.results.keys()], False)
    np.testing.assert_allclose(nnk2.ntri, nnk_dr.ntri, rtol=0.01)
    np.testing.assert_allclose(nnk2.zeta, nnk_dr.zeta, rtol=0.01)
    np.testing.assert_allclose(nnk2.varzeta, nnk_dr.varzeta, rtol=0.01)
    np.testing.assert_allclose(nnk2.zeta, zeta_drk, rtol=0.01)
    np.testing.assert_allclose(nnk2.varzeta, varzeta_drk, rtol=0.01)
    nnk2.calculateZeta(rrk=rrk, rdk=drk)
    nnk2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nnk.results.keys()], True)
    np.testing.assert_allclose(nnk2.zeta, nnk_dr.zeta, rtol=0.01)
    nnk2.calculateZeta(rrk=rrk, rdk=drk, drk=drk)
    nnk2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nnk.results.keys()], True)
    np.testing.assert_allclose(nnk2.zeta, nnk_dr.zeta, rtol=0.01)

    nkn = treecorr.NKNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    nkn.process(ncat, kcat, ncat)
    nkn2 = nkn.copy()
    nkn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nkn.results.keys()], False)
    np.testing.assert_allclose(nkn2.ntri, nkn.ntri, rtol=0.01)
    np.testing.assert_allclose(nkn2.zeta, nkn.zeta, rtol=0.01)
    np.testing.assert_allclose(nkn2.varzeta, nkn.varzeta, rtol=0.01)
    rkr = nkn.copy()
    rkr.process(rcat, kcat, rcat)
    nkn_rr = nkn.copy()
    zeta_rkr, varzeta_rkr = nkn_rr.calculateZeta(rkr=rkr)
    nkn2 = nkn_rr.copy()
    nkn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nkn.results.keys()], False)
    np.testing.assert_allclose(nkn2.ntri, nkn_rr.ntri, rtol=0.01)
    np.testing.assert_allclose(nkn2.zeta, nkn_rr.zeta, rtol=0.01)
    np.testing.assert_allclose(nkn2.varzeta, nkn_rr.varzeta, rtol=0.01)
    np.testing.assert_allclose(nkn2.zeta, zeta_rkr, rtol=0.01)
    np.testing.assert_allclose(nkn2.varzeta, varzeta_rkr, rtol=0.01)
    dkr = nkn.copy()
    dkr.process(ncat, kcat, rcat)
    nkn_dr = nkn.copy()
    zeta_dkr, varzeta_dkr = nkn_dr.calculateZeta(rkr=rkr, dkr=dkr)
    nkn2 = nkn_dr.copy()
    nkn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nkn.results.keys()], False)
    np.testing.assert_allclose(nkn2.ntri, nkn_dr.ntri, rtol=0.01)
    np.testing.assert_allclose(nkn2.zeta, nkn_dr.zeta, rtol=0.01)
    np.testing.assert_allclose(nkn2.varzeta, nkn_dr.varzeta, rtol=0.01)
    np.testing.assert_allclose(nkn2.zeta, zeta_dkr, rtol=0.01)
    np.testing.assert_allclose(nkn2.varzeta, varzeta_dkr, rtol=0.01)
    nkn2.calculateZeta(rkr=rkr, rkd=dkr)
    nkn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nkn.results.keys()], True)
    np.testing.assert_allclose(nkn2.zeta, nkn_dr.zeta, rtol=0.01)
    nkn2.calculateZeta(rkr=rkr, rkd=dkr, dkr=dkr)
    nkn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in nkn.results.keys()], True)
    np.testing.assert_allclose(nkn2.zeta, nkn_dr.zeta, rtol=0.01)

    knn = treecorr.KNNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                                  min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                  nphi_bins=nphi_bins, bin_type='LogSAS', max_n=40)
    knn.process(kcat, ncat)
    knn2 = knn.copy()
    knn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in knn.results.keys()], False)
    np.testing.assert_allclose(knn2.ntri, knn.ntri, rtol=0.01)
    np.testing.assert_allclose(knn2.zeta, knn.zeta, rtol=0.01)
    np.testing.assert_allclose(knn2.varzeta, knn.varzeta, rtol=0.01)
    krr = knn.copy()
    krr.process(kcat, rcat)
    knn_rr = knn.copy()
    zeta_krr, varzeta_krr = knn_rr.calculateZeta(krr=krr)
    knn2 = knn_rr.copy()
    knn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in knn.results.keys()], False)
    np.testing.assert_allclose(knn2.ntri, knn_rr.ntri, rtol=0.01)
    np.testing.assert_allclose(knn2.zeta, knn_rr.zeta, rtol=0.01)
    np.testing.assert_allclose(knn2.varzeta, knn_rr.varzeta, rtol=0.01)
    np.testing.assert_allclose(knn2.zeta, zeta_krr, rtol=0.01)
    np.testing.assert_allclose(knn2.varzeta, varzeta_krr, rtol=0.01)
    kdr = knn.copy()
    kdr.process(kcat, ncat, rcat)
    knn_dr = knn.copy()
    zeta_kdr, varzeta_kdr= knn_dr.calculateZeta(krr=krr, kdr=kdr)
    knn2 = knn_dr.copy()
    knn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in knn.results.keys()], False)
    np.testing.assert_allclose(knn2.ntri, knn_dr.ntri, rtol=0.01)
    np.testing.assert_allclose(knn2.zeta, knn_dr.zeta, rtol=0.01)
    np.testing.assert_allclose(knn2.varzeta, knn_dr.varzeta, rtol=0.01)
    np.testing.assert_allclose(knn2.zeta, zeta_kdr, rtol=0.01)
    np.testing.assert_allclose(knn2.varzeta, varzeta_kdr, rtol=0.01)
    knn2.calculateZeta(krr=krr, krd=kdr)
    knn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in knn.results.keys()], True)
    np.testing.assert_allclose(knn2.zeta, knn_dr.zeta, rtol=0.01)
    knn2.calculateZeta(krr=krr, kdr=kdr, krd=kdr)
    knn2._calculate_xi_from_pairs([(i,j,k,1) for i,j,k in knn.results.keys()], True)
    np.testing.assert_allclose(knn2.zeta, knn_dr.zeta, rtol=0.01)

    # Next check jackknife covariance estimate
    cov_nnk = nnk.estimate_cov('jackknife')
    n = nnk.varzeta.size
    print('nnk cov diag = ',np.diagonal(cov_nnk))
    print('var_nnk = ',var_nnk)
    print('nnk zeta var ratio = ',np.diagonal(cov_nnk)/var_nnk)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nnk))-np.log(var_nnk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov_nnk)), np.log(var_nnk), atol=0.8*tol_factor)

    cov_nkn = nkn.estimate_cov('jackknife')
    print('nkn var ratio = ',np.diagonal(cov_nkn)/var_nkn)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nkn))-np.log(var_nkn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov_nkn)), np.log(var_nkn), atol=0.7*tol_factor)

    cov_knn = knn.estimate_cov('jackknife')
    print('knn var ratio = ',np.diagonal(cov_knn)/var_knn)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_knn))-np.log(var_knn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov_knn)), np.log(var_knn), atol=0.7*tol_factor)

    # Check that these still work after roundtripping through a file.
    file_name = os.path.join('output','test_write_results_nnk.dat')
    nnk.write(file_name, write_patch_results=True)
    nnk2 = treecorr.Corr3.from_file(file_name)
    cov2 = nnk2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_nnk)

    file_name = os.path.join('output','test_write_results_nkn.dat')
    nkn.write(file_name, write_patch_results=True)
    nkn2 = treecorr.Corr3.from_file(file_name)
    cov2 = nkn2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_nkn)

    file_name = os.path.join('output','test_write_results_knn.dat')
    knn.write(file_name, write_patch_results=True)
    knn2 = treecorr.Corr3.from_file(file_name)
    cov2 = knn2.estimate_cov('jackknife')
    np.testing.assert_allclose(cov2, cov_knn)

    # Check jackknife using LogMultipole
    print('Using LogMultipole:')
    nnkm = treecorr.NNKCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    nnkm.process(ncat, kcat)
    fm = lambda corr: corr.toSAS(min_phi=min_phi, max_phi=max_phi, phi_units='deg',
                                 nphi_bins=nphi_bins).zeta.ravel()
    cov = nnkm.estimate_cov('jackknife', func=fm)
    print('nnk max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nnk))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nnk), atol=0.8*tol_factor)

    nknm = treecorr.NKNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    nknm.process(ncat, kcat, ncat)
    cov = nknm.estimate_cov('jackknife', func=fm)
    print('nkn max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_nkn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_nkn), atol=0.7*tol_factor)

    knnm = treecorr.KNNCorrelation(nbins=nbins, min_sep=min_sep, max_sep=max_sep, max_n=40,
                                   rng=rng, bin_type='LogMultipole')
    knnm.process(kcat, ncat)
    cov = knnm.estimate_cov('jackknife', func=fm)
    print('knn max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov))-np.log(var_knn))))
    np.testing.assert_allclose(np.log(np.diagonal(cov)), np.log(var_knn), atol=0.7*tol_factor)

    # Check with randoms
    cov_nnk = nnk_rr.estimate_cov('jackknife')
    print('with rand nnk cov diag = ',np.diagonal(cov_nnk))
    print('nnk zeta var ratio = ',np.diagonal(cov_nnk)/var_nnk_rr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nnk))-np.log(var_nnk_rr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_nnk)), np.log(var_nnk_rr), atol=0.8*tol_factor)
    cov_nnk = nnk_dr.estimate_cov('jackknife')
    print('with rand/dr nnk cov diag = ',np.diagonal(cov_nnk))
    print('nnk zeta var ratio = ',np.diagonal(cov_nnk)/var_nnk_dr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nnk))-np.log(var_nnk_dr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_nnk)), np.log(var_nnk_dr), atol=0.8*tol_factor)

    cov_nkn = nkn_rr.estimate_cov('jackknife')
    print('with rand nkn cov diag = ',np.diagonal(cov_nkn))
    print('nkn zeta var ratio = ',np.diagonal(cov_nkn)/var_nkn_rr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nkn))-np.log(var_nkn_rr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_nkn)), np.log(var_nkn_rr), atol=0.7*tol_factor)
    cov_nkn = nkn_dr.estimate_cov('jackknife')
    print('with rand/dr nkn cov diag = ',np.diagonal(cov_nkn))
    print('nkn zeta var ratio = ',np.diagonal(cov_nkn)/var_nkn_dr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_nkn))-np.log(var_nkn_dr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_nkn)), np.log(var_nkn_dr), atol=0.8*tol_factor)

    cov_knn = knn_rr.estimate_cov('jackknife')
    print('with rand knn cov diag = ',np.diagonal(cov_knn))
    print('knn zeta var ratio = ',np.diagonal(cov_knn)/var_knn_rr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_knn))-np.log(var_knn_rr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_knn)), np.log(var_knn_rr), atol=0.7*tol_factor)
    cov_knn = knn_dr.estimate_cov('jackknife')
    print('with rand/dr knn cov diag = ',np.diagonal(cov_knn))
    print('knn zeta var ratio = ',np.diagonal(cov_knn)/var_knn_dr)
    print('max log(ratio) = ',np.max(np.abs(np.log(np.diagonal(cov_knn))-np.log(var_knn_dr))))
    np.testing.assert_allclose(
        np.log(np.diagonal(cov_knn)), np.log(var_knn_dr), atol=0.7*tol_factor)


if __name__ == '__main__':
    test_direct_logruv_cross()
    test_direct_logruv_cross21()
    test_varzeta_logruv()
    test_direct_logsas_cross()
    test_direct_logsas_cross21()
    test_direct_logmultipole_cross()
    test_direct_logmultipole_cross21()
    test_nnk_logsas()
    test_varzeta()
    test_nnk_logsas_jk()

# Copyright (c) 2003-2019 by Mike Jarvis
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
import os
import coord
import time
import fitsio
import treecorr

from test_helper import assert_raises, do_pickle, timer, get_from_wiki, CaptureLog
from test_patch import clear_save, generate_shear_field

@timer
def test_brute_jk():
    # With bin_slop = 0, the jackknife calculation from patches should match a
    # brute force calcaulation where we literally remove one patch at a time to make
    # the vectors.
    if __name__ == '__main__':
        nside = 100
        nsource = 500
        npatch = 16
        rand_factor = 5
    else:
        nside = 100
        nsource = 300
        npatch = 8
        rand_factor = 5

    rng = np.random.RandomState(8675309)
    x, y, g1, g2, k = generate_shear_field(nside)
    # randomize positions slightly, since with grid, can get v=0 exactly, which is ambiguous
    # as to +- sign for v.  So complicates verification of equal results.
    x += rng.normal(0,0.01,len(x))
    y += rng.normal(0,0.01,len(y))

    indx = rng.choice(range(len(x)),nsource,replace=False)
    source_cat_nopatch = treecorr.Catalog(x=x[indx], y=y[indx],
                                          g1=g1[indx], g2=g2[indx], k=k[indx])
    source_cat = treecorr.Catalog(x=x[indx], y=y[indx],
                                  g1=g1[indx], g2=g2[indx], k=k[indx],
                                  npatch=npatch)
    print('source_cat patches = ',np.unique(source_cat.patch))
    print('len = ',source_cat.nobj, source_cat.ntot)
    assert source_cat.nobj == nsource

    # Start with KKK, since relatively simple.
    kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk1.process(source_cat_nopatch)

    kkk = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                  min_u=0.8, max_u=1.0, nubins=1,
                                  min_v=0., max_v=0.2, nvbins=1,
                                  var_method='jackknife')
    kkk.process(source_cat)
    np.testing.assert_allclose(kkk.zeta, kkk1.zeta)

    kkk_zeta_list = []
    for i in range(npatch):
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i],
                                       g1=source_cat.g1[source_cat.patch != i],
                                       g2=source_cat.g2[source_cat.patch != i])
        kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                       min_u=0.8, max_u=1.0, nubins=1,
                                       min_v=0., max_v=0.2, nvbins=1)
        kkk1.process(source_cat1)
        print('zeta = ',kkk1.zeta.ravel())
        kkk_zeta_list.append(kkk1.zeta.ravel())

    kkk_zeta_list = np.array(kkk_zeta_list)
    cov = np.cov(kkk_zeta_list.T, bias=True) * (len(kkk_zeta_list)-1)
    varzeta = np.diagonal(np.cov(kkk_zeta_list.T, bias=True)) * (len(kkk_zeta_list)-1)
    print('KKK: treecorr jackknife varzeta = ',kkk.varzeta.ravel())
    print('KKK: direct jackknife varzeta = ',varzeta)
    np.testing.assert_allclose(kkk.varzeta.ravel(), varzeta)

    # Now GGG
    ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    ggg1.process(source_cat_nopatch)

    ggg = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                  min_u=0.8, max_u=1.0, nubins=1,
                                  min_v=0., max_v=0.2, nvbins=1,
                                  var_method='jackknife')
    ggg.process(source_cat)
    np.testing.assert_allclose(ggg.gam0, ggg1.gam0)
    np.testing.assert_allclose(ggg.gam1, ggg1.gam1)
    np.testing.assert_allclose(ggg.gam2, ggg1.gam2)
    np.testing.assert_allclose(ggg.gam3, ggg1.gam3)

    ggg_gam0_list = []
    ggg_gam1_list = []
    ggg_gam2_list = []
    ggg_gam3_list = []
    ggg_map3_list = []
    for i in range(npatch):
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i],
                                       k=source_cat.k[source_cat.patch != i],
                                       g1=source_cat.g1[source_cat.patch != i],
                                       g2=source_cat.g2[source_cat.patch != i])
        ggg1 = treecorr.GGGCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                       min_u=0.8, max_u=1.0, nubins=1,
                                       min_v=0., max_v=0.2, nvbins=1)
        ggg1.process(source_cat1)
        ggg_gam0_list.append(ggg1.gam0.ravel())
        ggg_gam1_list.append(ggg1.gam1.ravel())
        ggg_gam2_list.append(ggg1.gam2.ravel())
        ggg_gam3_list.append(ggg1.gam3.ravel())
        ggg_map3_list.append(ggg1.calculateMap3()[0])

    ggg_gam0_list = np.array(ggg_gam0_list)
    vargam0 = np.diagonal(np.cov(ggg_gam0_list.T, bias=True)) * (len(ggg_gam0_list)-1)
    print('GG: treecorr jackknife vargam0 = ',ggg.vargam0.ravel())
    print('GG: direct jackknife vargam0 = ',vargam0)
    np.testing.assert_allclose(ggg.vargam0.ravel(), vargam0)
    ggg_gam1_list = np.array(ggg_gam1_list)
    vargam1 = np.diagonal(np.cov(ggg_gam1_list.T, bias=True)) * (len(ggg_gam1_list)-1)
    print('GG: treecorr jackknife vargam1 = ',ggg.vargam1.ravel())
    print('GG: direct jackknife vargam1 = ',vargam1)
    np.testing.assert_allclose(ggg.vargam1.ravel(), vargam1)
    ggg_gam2_list = np.array(ggg_gam2_list)
    vargam2 = np.diagonal(np.cov(ggg_gam2_list.T, bias=True)) * (len(ggg_gam2_list)-1)
    print('GG: treecorr jackknife vargam2 = ',ggg.vargam2.ravel())
    print('GG: direct jackknife vargam2 = ',vargam2)
    np.testing.assert_allclose(ggg.vargam2.ravel(), vargam2)
    ggg_gam3_list = np.array(ggg_gam3_list)
    vargam3 = np.diagonal(np.cov(ggg_gam3_list.T, bias=True)) * (len(ggg_gam3_list)-1)
    print('GG: treecorr jackknife vargam3 = ',ggg.vargam3.ravel())
    print('GG: direct jackknife vargam3 = ',vargam3)
    np.testing.assert_allclose(ggg.vargam3.ravel(), vargam3)

    ggg_map3_list = np.array(ggg_map3_list)
    varmap3 = np.diagonal(np.cov(ggg_map3_list.T, bias=True)) * (len(ggg_map3_list)-1)
    covmap3 = treecorr.estimate_multi_cov([ggg], 'jackknife',
                                          lambda corrs: corrs[0].calculateMap3()[0])
    print('GG: treecorr jackknife varmap3 = ',np.diagonal(covmap3))
    print('GG: direct jackknife varmap3 = ',varmap3)
    np.testing.assert_allclose(np.diagonal(covmap3), varmap3)

    return
    # Finally, test NN, which is complicated, since several different combinations of randoms.
    # 1. (DD-RR)/RR
    # 2. (DD-2DR+RR)/RR
    # 3. (DD-2RD+RR)/RR
    # 4. (DD-DR-RD+RR)/RR

    rand_source_cat = treecorr.Catalog(x=rng.uniform(0,1000,nsource*rand_factor),
                                       y=rng.uniform(0,1000,nsource*rand_factor),
                                       patch_centers=source_cat.patch_centers)
    print('rand_source_cat patches = ',np.unique(rand_source_cat.patch))
    print('len = ',rand_source_cat.nobj, rand_source_cat.ntot)

    dd = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    dd.process(lens_cat, source_cat)
    rr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    rr.process(rand_lens_cat, rand_source_cat)
    rd = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    rd.process(rand_lens_cat, source_cat)
    dr = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0,
                                var_method='jackknife')
    dr.process(lens_cat, rand_source_cat)

    # Now do this using brute force calculation.
    xi1_list = []
    xi2_list = []
    xi3_list = []
    xi4_list = []
    for i in range(npatch):
        lens_cat1 = treecorr.Catalog(x=lens_cat.x[lens_cat.patch != i],
                                     y=lens_cat.y[lens_cat.patch != i])
        source_cat1 = treecorr.Catalog(x=source_cat.x[source_cat.patch != i],
                                       y=source_cat.y[source_cat.patch != i])
        rand_lens_cat1 = treecorr.Catalog(x=rand_lens_cat.x[rand_lens_cat.patch != i],
                                          y=rand_lens_cat.y[rand_lens_cat.patch != i])
        rand_source_cat1 = treecorr.Catalog(x=rand_source_cat.x[rand_source_cat.patch != i],
                                            y=rand_source_cat.y[rand_source_cat.patch != i])
        dd1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        dd1.process(lens_cat1, source_cat1)
        rr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        rr1.process(rand_lens_cat1, rand_source_cat1)
        rd1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        rd1.process(rand_lens_cat1, source_cat1)
        dr1 = treecorr.NNCorrelation(bin_size=0.3, min_sep=10., max_sep=30., bin_slop=0)
        dr1.process(lens_cat1, rand_source_cat1)
        xi1_list.append(dd1.calculateXi(rr1)[0])
        xi2_list.append(dd1.calculateXi(rr1,dr=dr1)[0])
        xi3_list.append(dd1.calculateXi(rr1,rd=rd1)[0])
        xi4_list.append(dd1.calculateXi(rr1,dr=dr1,rd=rd1)[0])

    print('(DD-RR)/RR')
    xi1_list = np.array(xi1_list)
    xi1, varxi1 = dd.calculateXi(rr)
    varxi = np.diagonal(np.cov(xi1_list.T, bias=True)) * (len(xi1_list)-1)
    print('treecorr jackknife varxi = ',varxi1)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-2DR+RR)/RR')
    xi2_list = np.array(xi2_list)
    xi2, varxi2 = dd.calculateXi(rr, dr=dr)
    varxi = np.diagonal(np.cov(xi2_list.T, bias=True)) * (len(xi2_list)-1)
    print('treecorr jackknife varxi = ',varxi2)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-2RD+RR)/RR')
    xi3_list = np.array(xi3_list)
    xi3, varxi3 = dd.calculateXi(rr, rd=rd)
    varxi = np.diagonal(np.cov(xi3_list.T, bias=True)) * (len(xi3_list)-1)
    print('treecorr jackknife varxi = ',varxi3)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

    print('(DD-DR-RD+RR)/RR')
    xi4_list = np.array(xi4_list)
    xi4, varxi4 = dd.calculateXi(rr, rd=rd, dr=dr)
    varxi = np.diagonal(np.cov(xi4_list.T, bias=True)) * (len(xi4_list)-1)
    print('treecorr jackknife varxi = ',varxi4)
    print('direct jackknife varxi = ',varxi)
    np.testing.assert_allclose(dd.varxi, varxi)

@timer
def test_finalize_false():

    nside = 100
    nsource = 100
    npatch = 8

    # Make three independent data sets
    rng = np.random.RandomState(8675309)
    x_1, y_1, g1_1, g2_1, k_1 = generate_shear_field(nside)
    indx = rng.choice(range(len(x_1)),nsource,replace=False)
    x_1 = x_1[indx]
    y_1 = y_1[indx]
    g1_1 = g1_1[indx]
    g2_1 = g2_1[indx]
    k_1 = k_1[indx]
    x_1 += rng.normal(0,0.01,nsource)
    y_1 += rng.normal(0,0.01,nsource)
    x_2, y_2, g1_2, g2_2, k_2 = generate_shear_field(nside)
    indx = rng.choice(range(len(x_2)),nsource,replace=False)
    x_2 = x_2[indx]
    y_2 = y_2[indx]
    g1_2 = g1_2[indx]
    g2_2 = g2_2[indx]
    k_2 = k_2[indx]
    x_2 += rng.normal(0,0.01,nsource)
    y_2 += rng.normal(0,0.01,nsource)
    x_3, y_3, g1_3, g2_3, k_3 = generate_shear_field(nside)
    indx = rng.choice(range(len(x_3)),nsource,replace=False)
    x_3 = x_3[indx]
    y_3 = y_3[indx]
    g1_3 = g1_3[indx]
    g2_3 = g2_3[indx]
    k_3 = k_3[indx]
    x_3 += rng.normal(0,0.01,nsource)
    y_3 += rng.normal(0,0.01,nsource)

    # Make a single catalog with all three together
    cat = treecorr.Catalog(x=np.concatenate([x_1, x_2, x_3]),
                           y=np.concatenate([y_1, y_2, y_3]),
                           g1=np.concatenate([g1_1, g1_2, g1_3]),
                           g2=np.concatenate([g2_1, g2_2, g2_3]),
                           k=np.concatenate([k_1, k_2, k_3]),
                           npatch=npatch)

    # Now the three separately, using the same patch centers
    cat1 = treecorr.Catalog(x=x_1, y=y_1, g1=g1_1, g2=g2_1, k=k_1, patch_centers=cat.patch_centers)
    cat2 = treecorr.Catalog(x=x_2, y=y_2, g1=g1_2, g2=g2_2, k=k_2, patch_centers=cat.patch_centers)
    cat3 = treecorr.Catalog(x=x_3, y=y_3, g1=g1_3, g2=g2_3, k=k_3, patch_centers=cat.patch_centers)

    np.testing.assert_array_equal(cat1.patch, cat.patch[0:nsource])
    np.testing.assert_array_equal(cat2.patch, cat.patch[nsource:2*nsource])
    np.testing.assert_array_equal(cat3.patch, cat.patch[2*nsource:3*nsource])

    # KKK
    kkk1 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk1.process(cat)

    kkk2 = treecorr.KKKCorrelation(nbins=3, min_sep=30., max_sep=100., brute=True,
                                   min_u=0.8, max_u=1.0, nubins=1,
                                   min_v=0., max_v=0.2, nvbins=1)
    kkk2.process(cat1, initialize=True, finalize=False)
    kkk2.process(cat2, initialize=False, finalize=False)
    kkk2.process(cat3, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, initialize=False, finalize=False)
    kkk2.process(cat1, cat3, initialize=False, finalize=False)
    kkk2.process(cat2, cat1, initialize=False, finalize=False)
    kkk2.process(cat2, cat3, initialize=False, finalize=False)
    kkk2.process(cat3, cat1, initialize=False, finalize=False)
    kkk2.process(cat3, cat2, initialize=False, finalize=False)
    kkk2.process(cat1, cat2, cat3, initialize=False, finalize=True)

    np.testing.assert_allclose(kkk1.ntri, kkk2.ntri)
    np.testing.assert_allclose(kkk1.weight, kkk2.weight)
    np.testing.assert_allclose(kkk1.meand1, kkk2.meand1)
    np.testing.assert_allclose(kkk1.meand2, kkk2.meand2)
    np.testing.assert_allclose(kkk1.meand3, kkk2.meand3)
    np.testing.assert_allclose(kkk1.zeta, kkk2.zeta)


if __name__ == '__main__':
    test_brute_jk()
    test_finalize_false()

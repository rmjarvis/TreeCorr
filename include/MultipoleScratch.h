/* Copyright (c) 2003-2024 by Mike Jarvis
 *
 * TreeCorr is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef TreeCorr_MPScratch_H
#define TreeCorr_MPScratch_H

// The multipole algorithm needs some scratch arrays to store information during the
// computation.  Put them all together in a struct for ease of passing them around.
// Also, some of the details vary by DataType, so this is a base class we can use
// in generic situations, and we'll static_cast to the right then when we need
// the actual arrays.

#include <memory>

// make_unique is c++14, which we don't require.  It's simple enough to just roll our own...
// cf. https://stackoverflow.com/questions/7038357/make-unique-and-perfect-forwarding
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Base class has implementation of npairs, sumw, sumwr, sumwr, and Wn.
// Anything related to K or G values is in the relevant derived classes.
struct BaseMultipoleScratch
{
    BaseMultipoleScratch(int nbins, int nubins, bool use_ww) :
        ww(use_ww), n(nbins), Wnsize(nbins * (nubins+1)),
        Wn(Wnsize), npairs(n), sumw(n), sumwr(n), sumwlogr(n)
    {
        if (ww) {
            sumww.resize(n);
            sumwwr.resize(n);
            sumwwlogr.resize(n);
        }
    }

    BaseMultipoleScratch(const BaseMultipoleScratch& rhs) :
        ww(rhs.ww), n(rhs.n), Wnsize(rhs.Wnsize),
        Wn(rhs.Wn), npairs(rhs.npairs), sumw(rhs.sumw), sumwr(rhs.sumwr), sumwlogr(rhs.sumwlogr),
        sumww(rhs.sumww), sumwwr(rhs.sumwwr), sumwwlogr(rhs.sumwwlogr)
    {}

    virtual ~BaseMultipoleScratch() {}

    virtual void clear()
    {
        for (int i=0; i<Wnsize; ++i) Wn[i] = 0.;

        for (int i=0; i<n; ++i) {
            npairs[i] = 0.;
            sumw[i] = 0.;
            sumwr[i] = 0.;
            sumwlogr[i] = 0.;
        }

        if (ww) {
            for (int i=0; i<n; ++i) {
                sumww[i] = 0.;
                sumwwr[i] = 0.;
                sumwwlogr[i] = 0.;
            }
        }
    }

    virtual std::unique_ptr<BaseMultipoleScratch> duplicate() = 0;

    const bool ww;
    const int n;
    const int Wnsize;

    std::vector<std::complex<double> > Wn;
    std::vector<double> npairs;
    std::vector<double> sumw;
    std::vector<double> sumwr;
    std::vector<double> sumwlogr;
    std::vector<double> sumww;
    std::vector<double> sumwwr;
    std::vector<double> sumwwlogr;
};

template <int D1, int D2> struct MultipoleScratch;

// Specializations for different DataTypes:
//
// NData, NData doesn't need Gn at all, so the base class does everything already.
template <>
struct MultipoleScratch<NData, NData> : public BaseMultipoleScratch
{
    MultipoleScratch(int nbins, int nubins, bool use_ww) :
        BaseMultipoleScratch(nbins, nubins, use_ww)
    {}

    std::unique_ptr<BaseMultipoleScratch> duplicate()
    {
        return make_unique<MultipoleScratch>(*this);
    }
};

// KData, KData needs Gn and sumwwkk
template <>
struct MultipoleScratch<KData, KData> : public BaseMultipoleScratch
{
    MultipoleScratch(int nbins, int nubins, bool use_ww) :
        BaseMultipoleScratch(nbins, nubins, use_ww),
        Gn(Wnsize), sumwwkk(n)
    {}

    MultipoleScratch(const MultipoleScratch& rhs) :
        BaseMultipoleScratch(rhs),
        Gn(rhs.Gn), sumwwkk(rhs.sumwwkk)
    {}


    std::unique_ptr<BaseMultipoleScratch> duplicate()
    {
        return make_unique<MultipoleScratch>(*this);
    }

    void clear()
    {
        BaseMultipoleScratch::clear();
        for (int i=0; i<Wnsize; ++i) Gn[i] = 0.;
        if (ww) {
            for (int i=0; i<n; ++i) sumwwkk[i] = 0.;
        }
    }

    std::vector<std::complex<double> > Gn;
    std::vector<double> sumwwkk;
};

// GData, GData needs a bigger Gn and several extra ww arrays.
template <>
struct MultipoleScratch<GData, GData> : public BaseMultipoleScratch
{
    MultipoleScratch(int nbins, int nubins, bool use_ww) :
        BaseMultipoleScratch(nbins, nubins, use_ww),
        Gnsize(nbins * (2*nubins+3)), Gn(Gnsize)
    {
        if (ww) {
            sumwwgg0.resize(n);
            sumwwgg1.resize(n);
            sumwwgg2.resize(n);
        }
    }

    MultipoleScratch(const MultipoleScratch& rhs) :
        BaseMultipoleScratch(rhs),
        Gnsize(rhs.Gnsize), Gn(rhs.Gn),
        sumwwgg0(rhs.sumwwgg0), sumwwgg1(rhs.sumwwgg1), sumwwgg2(rhs.sumwwgg2)
    {}

    std::unique_ptr<BaseMultipoleScratch> duplicate()
    {
        return make_unique<MultipoleScratch>(*this);
    }

    void clear()
    {
        BaseMultipoleScratch::clear();
        for (int i=0; i<Gnsize; ++i) Gn[i] = 0.;
        if (ww) {
            for (int i=0; i<n; ++i) {
                sumwwgg0[i] = 0.;
                sumwwgg1[i] = 0.;
                sumwwgg2[i] = 0.;
            }
        }
    }
 
    const int Gnsize;

    std::vector<std::complex<double> > Gn;
    std::vector<std::complex<double> > sumwwgg0;
    std::vector<std::complex<double> > sumwwgg1;
    std::vector<std::complex<double> > sumwwgg2;
};

#endif

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
    BaseMultipoleScratch(int _nbins, int _maxn, bool use_ww, int _wbuffer=0) :
        ww(use_ww), nbins(_nbins), maxn(_maxn), wbuffer(_wbuffer),
        Wnsize(nbins * (maxn+1+wbuffer)),
        Wn(Wnsize), npairs(nbins), sumw(nbins), sumwr(nbins), sumwlogr(nbins)
    {
        if (ww) {
            sumww.resize(nbins);
            sumwwr.resize(nbins);
            sumwwlogr.resize(nbins);
        }
    }

    BaseMultipoleScratch(const BaseMultipoleScratch& rhs) :
        ww(rhs.ww), nbins(rhs.nbins), maxn(rhs.maxn), wbuffer(rhs.wbuffer), Wnsize(rhs.Wnsize),
        Wn(rhs.Wn), npairs(rhs.npairs), sumw(rhs.sumw), sumwr(rhs.sumwr), sumwlogr(rhs.sumwlogr),
        sumww(rhs.sumww), sumwwr(rhs.sumwwr), sumwwlogr(rhs.sumwwlogr)
    {}

    virtual ~BaseMultipoleScratch() {}

    virtual void clear()
    {
        for (int i=0; i<Wnsize; ++i) Wn[i] = 0.;

        for (int i=0; i<nbins; ++i) {
            npairs[i] = 0.;
            sumw[i] = 0.;
            sumwr[i] = 0.;
            sumwlogr[i] = 0.;
        }

        if (ww) {
            for (int i=0; i<nbins; ++i) {
                sumww[i] = 0.;
                sumwwr[i] = 0.;
                sumwwlogr[i] = 0.;
            }
        }
    }

    virtual std::unique_ptr<BaseMultipoleScratch> duplicate() = 0;

    inline int Windex(int k, int n=0)
    { return k * (maxn+1+wbuffer) + n; }

    virtual int Gindex(int k, int n=0) = 0;

    virtual std::complex<double> Gn(int index, int n=0) = 0;

    virtual double correction0r(int k) = 0;
    virtual std::complex<double> correction0(int k) = 0;
    virtual std::complex<double> correction1(int k) = 0;
    virtual std::complex<double> correction2(int k) = 0;

    template <int C>
    void calculateGn(
        const BaseCell<C>& c1, const BaseCell<C>& c2,
        double rsq, double r, int k, double w)
    { doCalculateGn(c1, c2, rsq, r, k, w); }

    const bool ww;
    const int nbins;
    const int maxn;
    const int wbuffer;
    const int Wnsize;

    std::vector<std::complex<double> > Wn;
    std::vector<double> npairs;
    std::vector<double> sumw;
    std::vector<double> sumwr;
    std::vector<double> sumwlogr;
    std::vector<double> sumww;
    std::vector<double> sumwwr;
    std::vector<double> sumwwlogr;

protected:

    virtual void doCalculateGn(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
        double rsq, double r, int k, double w) = 0;
    virtual void doCalculateGn(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
        double rsq, double r, int k, double w) = 0;
    virtual void doCalculateGn(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
        double rsq, double r, int k, double w) = 0;

};

template <int D> struct MultipoleScratch;

// Specializations for different DataTypes:
//
// NData doesn't need Gn at all, so the base class does everything already.
template <>
struct MultipoleScratch<NData> : public BaseMultipoleScratch
{
    MultipoleScratch(int nbins, int maxn, bool use_ww, int buffer) :
        BaseMultipoleScratch(nbins, maxn, use_ww, buffer)
    {
        if (use_ww && buffer) {
            sumwwzz.resize(nbins);
        }
    }

    std::unique_ptr<BaseMultipoleScratch> duplicate()
    {
        return make_unique<MultipoleScratch>(*this);
    }

    void clear()
    {
        BaseMultipoleScratch::clear();
        if (sumwwzz.size()) {
            for (int i=0; i<nbins; ++i) sumwwzz[i] = 0.;
        }
    }

    int Gindex(int k, int n=0)
    { return Windex(k, n); }

    std::complex<double> Gn(int index, int n=0)
    {
        if (n >= 0) {
            XAssert(index+n < Wnsize);
            XAssert(index+n >= 0);
        } else {
            XAssert(index-n < Wnsize);
            XAssert(index-n >= 0);
        }
        return n >= 0 ? Wn[index+n] : std::conj(Wn[index-n]);
    }

    double correction0r(int k)
    {
        XAssert(k >= 0 && k < sumww.size());
        return sumww[k];
    }

    std::complex<double> correction0(int k)
    {
        XAssert(k >= 0 && k < (wbuffer ? sumwwzz.size() : sumww.size()));
        return wbuffer ? sumwwzz[k] : sumww[k];
    }

    std::complex<double> correction1(int k)
    { XAssert(false); return sumww[k]; }

    std::complex<double> correction2(int k)
    { XAssert(false); return sumww[k]; }

    template <int C>
    void calculateGn(
        const BaseCell<C>& c1, const Cell<NData,C>& c2,
        double rsq, double r, int k, double w);

    std::vector<std::complex<double> > sumwwzz;

protected:
    void doCalculateGn(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<NData,Flat>&>(c2), rsq, r, k, w); }
    void doCalculateGn(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<NData,Sphere>&>(c2), rsq, r, k, w); }
    void doCalculateGn(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<NData,ThreeD>&>(c2), rsq, r, k, w); }
};

// KData needs Gn and sumwwkk
template <>
struct MultipoleScratch<KData> : public BaseMultipoleScratch
{
    MultipoleScratch(int nbins, int maxn, bool use_ww, int _buffer) :
        BaseMultipoleScratch(nbins, maxn, use_ww), buffer(_buffer),
        Gnsize(nbins * (maxn+1+buffer)), _Gn(Gnsize)
    {
        if (ww) {
            sumwwkk.resize(nbins);
        }
    }

    MultipoleScratch(const MultipoleScratch& rhs) :
        BaseMultipoleScratch(rhs), buffer(rhs.buffer),
        _Gn(rhs._Gn), sumwwkk(rhs.sumwwkk)
    {}


    std::unique_ptr<BaseMultipoleScratch> duplicate()
    {
        return make_unique<MultipoleScratch>(*this);
    }

    void clear()
    {
        BaseMultipoleScratch::clear();
        for (int i=0; i<Gnsize; ++i) _Gn[i] = 0.;
        if (ww) {
            for (int i=0; i<nbins; ++i) sumwwkk[i] = 0.;
        }
    }

    int Gindex(int k, int n=0)
    { return k * (maxn+1+buffer) + n; }

    std::complex<double> Gn(int index, int n=0)
    {
        if (n >= 0) {
            XAssert(index+n < Gnsize);
            XAssert(index+n >= 0);
        } else {
            XAssert(index-n < Gnsize);
            XAssert(index-n >= 0);
        }
        return n >= 0 ? _Gn[index+n] : std::conj(_Gn[index-n]);
    }

    double correction0r(int k)
    { return sumwwkk[k].real(); }

    std::complex<double> correction0(int k)
    { return sumwwkk[k]; }

    std::complex<double> correction1(int k)
    { XAssert(false); return sumwwkk[k]; }

    std::complex<double> correction2(int k)
    { XAssert(false); return sumwwkk[k]; }

    int buffer, Gnsize;
    std::vector<std::complex<double> > _Gn;
    std::vector<std::complex<double> > sumwwkk;

    template <int C>
    void calculateGn(
        const BaseCell<C>& c1, const Cell<KData,C>& c2,
        double rsq, double r, int k, double w);

protected:
    void doCalculateGn(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<KData,Flat>&>(c2), rsq, r, k, w); }
    void doCalculateGn(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<KData,Sphere>&>(c2), rsq, r, k, w); }
    void doCalculateGn(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<KData,ThreeD>&>(c2), rsq, r, k, w); }
};

// GData needs a bigger Gn and several extra ww arrays.
template <>
struct MultipoleScratch<GData> : public BaseMultipoleScratch
{
    MultipoleScratch(int nbins, int maxn, bool use_ww, int _buffer) :
        BaseMultipoleScratch(nbins, maxn, use_ww), buffer(_buffer),
        Gnsize(nbins * (2*(maxn+buffer)+1)), _Gn(Gnsize)
    {
        if (ww) {
            sumwwgg0.resize(nbins);
            sumwwgg1.resize(nbins);
            sumwwgg2.resize(nbins);
        }
    }

    MultipoleScratch(const MultipoleScratch& rhs) :
        BaseMultipoleScratch(rhs), buffer(rhs.buffer),
        Gnsize(rhs.Gnsize), _Gn(rhs._Gn),
        sumwwgg0(rhs.sumwwgg0), sumwwgg1(rhs.sumwwgg1), sumwwgg2(rhs.sumwwgg2)
    {}

    std::unique_ptr<BaseMultipoleScratch> duplicate()
    {
        return make_unique<MultipoleScratch>(*this);
    }

    void clear()
    {
        BaseMultipoleScratch::clear();
        for (int i=0; i<Gnsize; ++i) _Gn[i] = 0.;
        if (ww) {
            for (int i=0; i<nbins; ++i) {
                sumwwgg0[i] = 0.;
                sumwwgg1[i] = 0.;
                sumwwgg2[i] = 0.;
            }
        }
    }

    int Gindex(int k, int n=0)
    { return (2*k+1)*(maxn+buffer) + k + n; }
    // 2*k*maxn + 2*k*buffer + maxn + buffer + k

    std::complex<double> Gn(int index, int n=0)
    {
        XAssert(index+n < Gnsize);
        XAssert(index+n >= 0);
        return _Gn[index+n];
    }

    double correction0r(int k)
    { XAssert(false); return sumwwgg0[k].real(); }

    std::complex<double> correction0(int k)
    { return sumwwgg0[k]; }

    std::complex<double> correction1(int k)
    { return sumwwgg1[k]; }

    std::complex<double> correction2(int k)
    { return sumwwgg2[k]; }

    int buffer;
    const int Gnsize;

    std::vector<std::complex<double> > _Gn;
    std::vector<std::complex<double> > sumwwgg0;
    std::vector<std::complex<double> > sumwwgg1;
    std::vector<std::complex<double> > sumwwgg2;

    template <int C>
    void calculateGn(
        const BaseCell<C>& c1, const Cell<GData,C>& c2,
        double rsq, double r, int k, double w);

protected:
    void doCalculateGn(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<GData,Flat>&>(c2), rsq, r, k, w); }
    void doCalculateGn(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<GData,Sphere>&>(c2), rsq, r, k, w); }
    void doCalculateGn(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
        double rsq, double r, int k, double w)
    { calculateGn(c1, static_cast<const Cell<GData,ThreeD>&>(c2), rsq, r, k, w); }
};

#endif

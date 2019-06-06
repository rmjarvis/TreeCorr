/* Copyright (c) 2003-2019 by Mike Jarvis
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

// Uncomment this to enable xassert, usually more time-consuming assert statements
// Also to turn on dbg<< messages.
//#define DEBUGLOGGING

#include "dbg.h"
#include "BinnedCorr2.h"
#include "Split.h"
#include "ProjectHelper.h"
#include "Metric.h"
#include <vector>
#include <set>
#include <map>

#ifdef _OPENMP
#include "omp.h"
#endif

// When we need a compile-time max, use this rather than std::max, which only became valid
// for compile-time constexpr in C++14, which we don't require.
#define MAX(a,b) (a > b ? a : b)

template <int D1, int D2, int B>
BinnedCorr2<D1,D2,B>::BinnedCorr2(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double minrpar, double maxrpar, double xp, double yp, double zp,
    double* xi0, double* xi1, double* xi2, double* xi3,
    double* meanr, double* meanlogr, double* weight, double* npairs) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minrpar(minrpar), _maxrpar(maxrpar), _xp(xp), _yp(yp), _zp(zp),
    _coords(-1), _owns_data(false),
    _xi(xi0,xi1,xi2,xi3), _meanr(meanr), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
{
    dbg<<"BinnedCorr2 constructor\n";
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _bsq = _b * _b;
    _fullmaxsep = BinTypeHelper<B>::calculateFullMaxSep(minsep, maxsep, nbins, binsize);
    _fullmaxsepsq = _fullmaxsep*_fullmaxsep;
    dbg<<"minsep, maxsep = "<<_minsep<<"  "<<_maxsep<<std::endl;
    dbg<<"nbins = "<<_nbins<<std::endl;
    dbg<<"binsize = "<<_binsize<<std::endl;
    dbg<<"b = "<<_b<<std::endl;
    dbg<<"minrpar, maxrpar = "<<_minrpar<<"  "<<_maxrpar<<std::endl;
    dbg<<"period = "<<_xp<<"  "<<_yp<<"  "<<_zp<<std::endl;
}

template <int D1, int D2, int B>
BinnedCorr2<D1,D2,B>::BinnedCorr2(const BinnedCorr2<D1,D2,B>& rhs, bool copy_data) :
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minrpar(rhs._minrpar), _maxrpar(rhs._maxrpar),
    _xp(rhs._xp), _yp(rhs._yp), _zp(rhs._zp),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq), _bsq(rhs._bsq),
    _fullmaxsep(rhs._fullmaxsep), _fullmaxsepsq(rhs._fullmaxsepsq),
    _coords(rhs._coords), _owns_data(true),
    _xi(0,0,0,0), _weight(0)
{
    dbg<<"BinnedCorr2 copy constructor\n";
    _xi.new_data(_nbins);
    _meanr = new double[_nbins];
    _meanlogr = new double[_nbins];
    _weight = new double[_nbins];
    _npairs = new double[_nbins];

    if (copy_data) *this = rhs;
    else clear();
}

template <int D1, int D2, int B>
BinnedCorr2<D1,D2,B>::~BinnedCorr2()
{
    dbg<<"BinnedCorr2 destructor\n";
    if (_owns_data) {
        _xi.delete_data(_nbins);
        delete [] _meanr; _meanr = 0;
        delete [] _meanlogr; _meanlogr = 0;
        delete [] _weight; _weight = 0;
        delete [] _npairs; _npairs = 0;
    }
}

// BinnedCorr2::process2 is invalid if D1 != D2, so this helper struct lets us only call
// process2 when D1 == D2.
template <int D1, int D2, int B, int C, int M>
struct ProcessHelper
{
    static void process2(BinnedCorr2<D1,D2,B>& , const Cell<D1,C>&, const MetricHelper<M>& ) {}
};

template <int D, int B, int C, int M>
struct ProcessHelper<D,D,B,C,M>
{
    static void process2(BinnedCorr2<D,D,B>& b, const Cell<D,C>& c12, const MetricHelper<M>& m)
    { b.template process2<C,M>(c12, m); }
};

template <int D1, int D2, int B>
void BinnedCorr2<D1,D2,B>::clear()
{
    _xi.clear(_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _weight[i] = 0.;
    for (int i=0; i<_nbins; ++i) _npairs[i] = 0.;
    _coords = -1;
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::process(const Field<D1,C>& field, bool dots)
{
    xdbg<<"Start process (auto): M,C = "<<M<<"  "<<C<<std::endl;
    Assert(D1 == D2);
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field.getNTopLevel();
    dbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<D1,D2,B> bc2(*this,false);
#else
        BinnedCorr2<D1,D2,B>& bc2 = *this;
#endif

        // Inside the omp parallel, so each thread has its own MetricHelper.
        MetricHelper<M> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                if (dots) std::cout<<'.'<<std::flush;
            }
            const Cell<D1,C>& c1 = *field.getCells()[i];
            ProcessHelper<D1,D2,B,C,M>::process2(bc2, c1, metric);
            for (int j=i+1;j<n1;++j) {
                const Cell<D1,C>& c2 = *field.getCells()[j];
                bc2.process11<C,M>(c1, c2, metric, BinTypeHelper<B>::doReverse());
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc2;
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::process(const Field<D1,C>& field1, const Field<D2,C>& field2,
                                   bool dots)
{
    xdbg<<"Start process (cross): M,C = "<<M<<"  "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    dbg<<"field1 has "<<n1<<" top level nodes\n";
    dbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<D1,D2,B> bc2(*this,false);
#else
        BinnedCorr2<D1,D2,B>& bc2 = *this;
#endif

        MetricHelper<M> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                if (dots) std::cout<<'.'<<std::flush;
            }
            const Cell<D1,C>& c1 = *field1.getCells()[i];
            for (int j=0;j<n2;++j) {
                const Cell<D2,C>& c2 = *field2.getCells()[j];
                bc2.process11<C,M>(c1, c2, metric, false);
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc2;
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::processPairwise(
    const SimpleField<D1,C>& field1, const SimpleField<D2,C>& field2, bool dots)
{
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long nobj = field1.getNObj();
    const long nobj2 = field2.getNObj();
    xdbg<<"field1 has "<<nobj<<" objects\n";
    xdbg<<"field2 has "<<nobj2<<" objects\n";
    Assert(nobj > 0);
    Assert(nobj == nobj2);

    const long sqrtn = long(sqrt(double(nobj)));

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<D1,D2,B> bc2(*this,false);
#else
        BinnedCorr2<D1,D2,B>& bc2 = *this;
#endif

        MetricHelper<M> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long i=0;i<nobj;++i) {
            // Let the progress dots happen every sqrt(n) iterations.
            if (dots && (i % sqrtn == 0)) {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
#ifdef _OPENMP
                    xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                    std::cout<<'.'<<std::flush;
                }
            }
            const Cell<D1,C>& c1 = *field1.getCells()[i];
            const Cell<D2,C>& c2 = *field2.getCells()[i];
            const Position<C>& p1 = c1.getPos();
            const Position<C>& p2 = c2.getPos();
            double s=0.;
            const double rsq = metric.DistSq(p1, p2, s, s);
            if (BinTypeHelper<B>::isRSqInRange(rsq, p1, p2,
                                               _minsep, _minsepsq, _maxsep, _maxsepsq)) {
                bc2.template directProcess11(c1,c2,rsq,false);
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc2;
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::process2(const Cell<D1,C>& c12, const MetricHelper<M>& metric)
{
    if (c12.getW() == 0.) return;
    if (c12.getSize() <= _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2<C,M>(*c12.getLeft(), metric);
    process2<C,M>(*c12.getRight(), metric);
    process11<C,M>(*c12.getLeft(), *c12.getRight(), metric, BinTypeHelper<B>::doReverse());
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::process11(const Cell<D1,C>& c1, const Cell<D2,C>& c2,
                                     const MetricHelper<M>& metric, bool do_reverse)
{
    //set_verbose(2);
    xdbg<<"Start process11 for "<<c1.getPos()<<",  "<<c2.getPos()<<"   ";
    xdbg<<"w = "<<c1.getW()<<", "<<c2.getW()<<std::endl;
    if (c1.getW() == 0. || c2.getW() == 0.) return;

    const Position<C>& p1 = c1.getPos();
    const Position<C>& p2 = c2.getPos();
    double s1 = c1.getSize(); // May be modified by DistSq function.
    double s2 = c2.getSize(); // "
    xdbg<<"s1,s2 = "<<s1<<','<<s2<<std::endl;
    xdbg<<"M,C = "<<M<<"  "<<C<<std::endl;
    const double rsq = metric.DistSq(p1,p2,s1,s2);
    xdbg<<"rsq = "<<rsq<<std::endl;
    xdbg<<"s1,s2 => "<<s1<<','<<s2<<std::endl;
    const double s1ps2 = s1+s2;

    double rpar = 0; // Gets set to correct value by this function if appropriate
    if (metric.isRParOutsideRange(p1, p2, s1ps2, rpar)) {
        return;
    }
    xdbg<<"RPar in range\n";

    if (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, _minsep, _minsepsq) &&
        metric.tooSmallDist(p1, p2, rsq, rpar, s1ps2, _minsep, _minsepsq)) {
        return;
    }
    xdbg<<"Not too small separation\n";

    if (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq) &&
        metric.tooLargeDist(p1, p2, rsq, rpar, s1ps2, _fullmaxsep, _fullmaxsepsq)) {
        return;
    }
    xdbg<<"Not too large separation\n";

    // Now check if these cells are small enough that it is ok to drop into a single bin.
    int k=-1;
    double r=0,logr=0;  // If singleBin is true, these values are set for use by directProcess11
    if (metric.isRParInsideRange(p1, p2, s1ps2, rpar) &&
        BinTypeHelper<B>::singleBin(rsq, s1ps2, p1, p2, _binsize, _b, _bsq,
                                    _minsep, _maxsep, _logminsep, k, r, logr))
    {
        xdbg<<"Drop into single bin.\n";
        if (BinTypeHelper<B>::isRSqInRange(rsq, p1, p2, _minsep, _minsepsq, _maxsep, _maxsepsq)) {
            directProcess11(c1,c2,rsq,do_reverse,k,r,logr);
        }
    } else {
        xdbg<<"Need to split.\n";
        bool split1=false, split2=false;
        double bsq_eff = BinTypeHelper<B>::getEffectiveBSq(rsq,_bsq);
        xdbg<<"bsq_eff = "<<bsq_eff<<std::endl;
        CalcSplitSq(split1,split2,s1,s2,s1ps2,bsq_eff);
        xdbg<<"rsq = "<<rsq<<", s1ps2 = "<<s1ps2<<"  ";
        xdbg<<"s1ps2 / r = "<<s1ps2 / sqrt(rsq)<<", b = "<<_b<<"  ";
        xdbg<<"split = "<<split1<<','<<split2<<std::endl;

        if (split1 && split2) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<C,M>(*c1.getLeft(),*c2.getLeft(),metric,do_reverse);
            process11<C,M>(*c1.getLeft(),*c2.getRight(),metric,do_reverse);
            process11<C,M>(*c1.getRight(),*c2.getLeft(),metric,do_reverse);
            process11<C,M>(*c1.getRight(),*c2.getRight(),metric,do_reverse);
        } else if (split1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11<C,M>(*c1.getLeft(),c2,metric,do_reverse);
            process11<C,M>(*c1.getRight(),c2,metric,do_reverse);
        } else {
            Assert(split2);
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<C,M>(c1,*c2.getLeft(),metric,do_reverse);
            process11<C,M>(c1,*c2.getRight(),metric,do_reverse);
        }
    }
}


// We also set up a helper class for doing the direct processing
template <int D1, int D2>
struct DirectHelper;

template <>
struct DirectHelper<NData,NData>
{
    template <int C>
    static void ProcessXi(
        const Cell<NData,C>& , const Cell<NData,C>& , const double ,
        XiData<NData,NData>& , int, int )
    {}
};

template <>
struct DirectHelper<NData,KData>
{
    template <int C>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<KData,C>& c2, const double ,
        XiData<NData,KData>& xi, int k, int )
    { xi.xi[k] += c1.getW() * c2.getData().getWK(); }
};

template <>
struct DirectHelper<NData,GData>
{
    template <int C>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<GData,C>& c2, const double rsq,
        XiData<NData,GData>& xi, int k, int )
    {
        std::complex<double> g2;
        ProjectHelper<C>::ProjectShear(c1,c2,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getW();
        xi.xi[k] += real(g2);
        xi.xi_im[k] += imag(g2);
    }
};

template <>
struct DirectHelper<KData,KData>
{
    template <int C>
    static void ProcessXi(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2, const double ,
        XiData<KData,KData>& xi, int k, int k2)
    {
        double wkk = c1.getData().getWK() * c2.getData().getWK();
        xi.xi[k] += wkk;
        if (k2 != -1) xi.xi[k2] += wkk;
    }
};

template <>
struct DirectHelper<KData,GData>
{
    template <int C>
    static void ProcessXi(
        const Cell<KData,C>& c1, const Cell<GData,C>& c2, const double rsq,
        XiData<KData,GData>& xi, int k, int )
    {
        std::complex<double> g2;
        ProjectHelper<C>::ProjectShear(c1,c2,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getData().getWK();
        xi.xi[k] += real(g2);
        xi.xi_im[k] += imag(g2);
    }
};

template <>
struct DirectHelper<GData,GData>
{
    template <int C>
    static void ProcessXi(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2, const double rsq,
        XiData<GData,GData>& xi, int k, int k2)
    {
        std::complex<double> g1, g2;
        ProjectHelper<C>::ProjectShears(c1,c2,g1,g2);

        // The complex products g1 g2 and g1 g2* share most of the calculations,
        // so faster to do this manually.
        double g1rg2r = g1.real() * g2.real();
        double g1rg2i = g1.real() * g2.imag();
        double g1ig2r = g1.imag() * g2.real();
        double g1ig2i = g1.imag() * g2.imag();

        xi.xip[k] += g1rg2r + g1ig2i;       // g1 * conj(g2)
        xi.xip_im[k] += g1ig2r - g1rg2i;
        xi.xim[k] += g1rg2r - g1ig2i;       // g1 * g2
        xi.xim_im[k] += g1ig2r + g1rg2i;

        if (k2 != -1) {
            xi.xip[k2] += g1rg2r + g1ig2i;       // g1 * conj(g2)
            xi.xip_im[k2] += g1ig2r - g1rg2i;
            xi.xim[k2] += g1rg2r - g1ig2i;       // g1 * g2
            xi.xim_im[k2] += g1ig2r + g1rg2i;
        }
    }
};

template <int D1, int D2, int B> template <int C>
void BinnedCorr2<D1,D2,B>::directProcess11(
    const Cell<D1,C>& c1, const Cell<D2,C>& c2, const double rsq, bool do_reverse,
    int k, double r, double logr)
{
    xdbg<<"DirectProcess11: rsq = "<<rsq<<std::endl;
    XAssert(rsq >= _minsepsq);
    XAssert(rsq < _fullmaxsepsq);
    // Note that most of these XAsserts around are still hardcoded for Log binning and Euclidean
    // metric.  If turning on verbose>=3, these could fail.
    XAssert(c1.getSize()+c2.getSize() < sqrt(rsq)*_b + 0.0001);

    XAssert(_binsize != 0.);
    const Position<C>& p1 = c1.getPos();
    const Position<C>& p2 = c2.getPos();
    if (k < 0) {
        r = sqrt(rsq);
        logr = log(r);
        Assert(logr >= _logminsep);
        k = BinTypeHelper<B>::calculateBinK(p1, p2, r, logr, _binsize,
                                            _minsep, _maxsep, _logminsep);
    } else {
        XAssert(std::abs(r - sqrt(rsq)) < 1.e-10*r);
        XAssert(std::abs(logr - 0.5*log(rsq)) < 1.e-10);
        XAssert(k == BinTypeHelper<B>::calculateBinK(p1, p2, r, logr, _binsize,
                                                     _minsep, _maxsep, _logminsep));
    }
    Assert(k >= 0);
    Assert(k < _nbins);
    xdbg<<"r,logr,k = "<<r<<','<<logr<<','<<k<<std::endl;

    double nn = double(c1.getN()) * double(c2.getN());
    _npairs[k] += nn;

    double ww = double(c1.getW()) * double(c2.getW());
    _meanr[k] += ww * r;
    _meanlogr[k] += ww * logr;
    _weight[k] += ww;
    xdbg<<"n,w = "<<nn<<','<<ww<<" ==>  "<<_npairs[k]<<','<<_weight[k]<<std::endl;

    int k2 = -1;
    if (do_reverse) {
        k2 = BinTypeHelper<B>::calculateBinK(p2, p1, r, logr, _binsize,
                                             _minsep, _maxsep, _logminsep);
        Assert(k2 >= 0);
        Assert(k2 < _nbins);
        _npairs[k2] += nn;
        _meanr[k2] += ww * r;
        _meanlogr[k2] += ww * logr;
        _weight[k2] += ww;
    }

    DirectHelper<D1,D2>::template ProcessXi<C>(c1,c2,rsq,_xi,k,k2);
}

template <int D1, int D2, int B>
void BinnedCorr2<D1,D2,B>::operator=(const BinnedCorr2<D1,D2,B>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.copy(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template <int D1, int D2, int B>
void BinnedCorr2<D1,D2,B>::operator+=(const BinnedCorr2<D1,D2,B>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.add(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] += rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] += rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] += rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] += rhs._npairs[i];
}

template <int D1, int D2, int B> template <int C, int M>
long BinnedCorr2<D1,D2,B>::samplePairs(
    const Field<D1, C>& field1, const Field<D2, C>& field2,
    double minsep, double maxsep, long* i1, long* i2, double* sep, int n)
{
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    dbg<<"field1 has "<<n1<<" top level nodes\n";
    dbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

    MetricHelper<M> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

    double minsepsq = minsep*minsep;
    double maxsepsq = maxsep*maxsep;

    long k=0;
    for (int i=0;i<n1;++i) {
        const Cell<D1,C>& c1 = *field1.getCells()[i];
        for (int j=0;j<n2;++j) {
            const Cell<D2,C>& c2 = *field2.getCells()[j];
            samplePairs(c1, c2, metric, minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        }
    }
    return k;
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::samplePairs(
    const Cell<D1, C>& c1, const Cell<D2, C>& c2, const MetricHelper<M>& metric,
    double minsep, double minsepsq, double maxsep, double maxsepsq,
    long* i1, long* i2, double* sep, int n, long& k)
{
    // This tracks process11, but we only select pairs at the end, not call directProcess11
    xdbg<<"Start samplePairs for "<<c1.getPos()<<",  "<<c2.getPos()<<"   ";
    xdbg<<"w = "<<c1.getW()<<", "<<c2.getW()<<std::endl;
    if (c1.getW() == 0. || c2.getW() == 0.) return;

    const Position<C>& p1 = c1.getPos();
    const Position<C>& p2 = c2.getPos();
    double s1 = c1.getSize(); // May be modified by DistSq function.
    double s2 = c2.getSize(); // "
    xdbg<<"s1,s2 = "<<s1<<','<<s2<<std::endl;
    xdbg<<"M,C = "<<M<<"  "<<C<<std::endl;
    const double rsq = metric.DistSq(p1, p2, s1, s2);
    xdbg<<"rsq = "<<rsq<<std::endl;
    xdbg<<"s1,s2 => "<<s1<<','<<s2<<std::endl;
    const double s1ps2 = s1+s2;

    double rpar = 0; // Gets set to correct value by this function if appropriate
    if (metric.isRParOutsideRange(p1, p2, s1ps2, rpar)) {
        return;
    }
    xdbg<<"RPar in range\n";

    if (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, minsep, minsepsq) &&
        metric.tooSmallDist(p1, p2, rsq, rpar, s1ps2, minsep, minsepsq)) {
        return;
    }
    xdbg<<"Not too small separation\n";

    if (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, maxsep, maxsepsq) &&
        metric.tooLargeDist(p1, p2, rsq, rpar, s1ps2, maxsep, maxsepsq)) {
        return;
    }
    xdbg<<"Not too large separation\n";

    // Now check if these cells are small enough that it is ok to drop into a single bin.
    int kk=-1;
    double r=0,logr=0;  // If singleBin is true, these values are set for use by directProcess11
    if (metric.isRParInsideRange(p1, p2, s1ps2, rpar) &&
        BinTypeHelper<B>::singleBin(rsq, s1ps2, p1, p2, _binsize, _b, _bsq,
                                    _minsep, _maxsep, _logminsep, kk, r, logr))
    {
        xdbg<<"Drop into single bin.\n";
        xdbg<<"rsq = "<<rsq<<std::endl;
        xdbg<<"minsepsq = "<<minsepsq<<std::endl;
        xdbg<<"maxsepsq = "<<maxsepsq<<std::endl;
        if (BinTypeHelper<B>::isRSqInRange(rsq, p1, p2, minsep, minsepsq, maxsep, maxsepsq)) {
            sampleFrom(c1,c2,rsq,r,i1,i2,sep,n,k);
        }
    } else {
        xdbg<<"Need to split.\n";
        bool split1=false, split2=false;
        double bsq_eff = BinTypeHelper<B>::getEffectiveBSq(rsq,_bsq);
        CalcSplitSq(split1,split2,s1,s2,s1ps2,bsq_eff);
        xdbg<<"rsq = "<<rsq<<", s1ps2 = "<<s1ps2<<"  ";
        xdbg<<"s1ps2 / r = "<<s1ps2 / sqrt(rsq)<<", b = "<<_b<<"  ";
        xdbg<<"split = "<<split1<<','<<split2<<std::endl;

        if (split1 && split2) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            samplePairs<C,M>(*c1.getLeft(), *c2.getLeft(), metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<C,M>(*c1.getLeft(), *c2.getRight(), metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<C,M>(*c1.getRight(), *c2.getLeft(), metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<C,M>(*c1.getRight(), *c2.getRight(), metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        } else if (split1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            samplePairs<C,M>(*c1.getLeft(), c2, metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<C,M>(*c1.getRight(), c2, metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        } else {
            Assert(split2);
            Assert(c2.getLeft());
            Assert(c2.getRight());
            samplePairs<C,M>(c1, *c2.getLeft(), metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<C,M>(c1, *c2.getRight(), metric,
                             minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        }
    }
}

void SelectRandomFrom(long m, std::vector<long>& selection)
{
    xdbg<<"SelectRandomFrom("<<m<<", "<<selection.size()<<")\n";
    long n = selection.size();
    // There are two algorithms here.
    // Floyd's algorithm is efficient for m >> n.
    // Fisher-Yates is efficient for m not much > n.
    // I don't know exactly the transition for when Floyd's becomes the better choice.
    // So I'm somewhat arbitrarily picking 3*n.
    if (m > 3*n) {
        dbg<<"Floyd's algorithm\n";
        // Floyd's algorithm
        // O(n) in memory, but slightly more than O(n) in time, especially as m ~ n.
        std::set<long> selected;
        while (long(selected.size()) < n) {
            double urd = rand();
            urd /= RAND_MAX;        // 0 < urd < 1
            long j = long(urd * m); // 0 <= j < m
            if (j == m) j = m-1;    // Just in case.
            std::pair<std::set<long>::iterator,bool> ret = selected.insert(j);
            if (ret.second) {
                // Then insertion happened.  I.e. not already present.
                // Also write it to the end of the selection list
                selection[selected.size()-1] = j;
            }
        }
    } else {
        dbg<<"Fisher-Yates\n";
        // Fisher-Yates shuffle up through n elements
        // O(n) in time, but O(m) in memory.
        std::vector<long> full(m);
        for (long i=0; i<m; ++i) full[i] = i;
        for (long i=0; i<n; ++i) {
            double urd = rand();
            urd /= RAND_MAX;            // 0 < urd < 1
            long j = long(urd * (m-i)); // 0 <= j < m-i
            j += i;                     // i <= j < m
            if (j == m) j = m-1;        // Just in case.
            std::swap(full[i], full[j]);
        }
        std::copy(full.begin(), full.begin()+n, selection.begin());
    }
}

template <int D1, int D2, int B> template <int C>
void BinnedCorr2<D1,D2,B>::sampleFrom(
    const Cell<D1, C>& c1, const Cell<D2, C>& c2, double rsq, double r,
    long* i1, long* i2, double* sep, int n, long& k)
{
    // At the start, k pairs will already have been considered for selection.
    // Of these min(k,n) will have been selected for inclusion in the lists.

    // If we consider each pair one at a time, then the algorithm to end up with a
    // uniform probability that any given pair is selected is as follows.
    // 1. When k < n, always select the next pair.
    // 2. When k >= n, select the next pair with probability n/(k+1).
    // 3. If selected, replace a random pair currently in the list with the new pair.

    // Proof that for all k >= n, all pairs fromm i=1..k have P(selected) = n/k:
    //
    // 1. If k==n, then all items are selected with probability 1 = n/k.
    // 2. If true for some k>=n, then for step k+1:
    //    - probability for i=k+1 is n/(k+1)
    //    - probability for i<=k is
    //      P(already selected) * P(remains in list | alread selected)
    //      = n/k * (P(i=k+1 not selected) + P(i=k+1 is selected, but given pair is not replaced))
    //      = n/k * ( (1 - n/(k+1)) + n/(k+1) * (n-1)/n )
    //      = n/k * ( (k+1 - n + n-1)/(k+1) )
    //      = n/k * ( k/(k+1) )
    //      = n/(k+1)
    // QED

    // In practice, we usually have many pairs to add, not just one at a time.  So we want to
    // try to do the net result of m passes of the above algorithm.
    // Consider 3 cases:
    // 1. k+m<=n:           Take all m pairs
    // 2. m<=n:             Just do m passes of the above algorithm.
    // 3. k+m>n and m>n:    Choose n pairs randomly without replacement from full k+m set.
    //                      For the selected pairs with k<=i<k+m, either place in the next spot
    //                      (if k<n) or randomly replace one of the ones that was already there.

    long n1 = c1.getN();
    long n2 = c2.getN();
    long m = n1 * n2;

    std::vector<const Cell<D1,C>*> leaf1 = c1.getAllLeaves();
    std::vector<const Cell<D2,C>*> leaf2 = c2.getAllLeaves();

    if (r == 0.) {
        r = sqrt(rsq);
    } else {
        XAssert(std::abs(r - sqrt(rsq)) < 1.e-10*r);
    }
    xdbg<<"sampleFrom: "<<c1.getN()<<"  "<<c2.getN()<<"  "<<rsq<<"  "<<r<<std::endl;
    xdbg<<"   n1,n2,k,n,m = "<<n1<<','<<n2<<','<<k<<','<<n<<','<<m<<std::endl;

    if (k + m <= n) {
        // Case 1
        xdbg<<"Case 1: take all pairs\n";
        for (size_t p1=0; p1<leaf1.size(); ++p1) {
            int nn1 = leaf1[p1]->getN();
            for (int q1=0; q1<nn1; ++q1) {
                int index1;
                if (nn1 == 1) index1 = leaf1[p1]->getInfo().index;
                else index1 = (*leaf1[p1]->getListInfo().indices)[q1];
                for (size_t p2=0; p2<leaf2.size(); ++p2) {
                    int nn2 = leaf2[p2]->getN();
                    for (int q2=0; q2<nn2; ++q2) {
                        int index2;
                        if (nn2 == 1) index2 = leaf2[p2]->getInfo().index;
                        else index2 = (*leaf2[p2]->getListInfo().indices)[q2];
                        i1[k] = index1;
                        i2[k] = index2;
                        sep[k] = r;
                        ++k;
                    }
                }
            }
        }
    } else if (m <= n) {
        // Case 2
        xdbg<<"Case 2: Check one at a time.\n";
        for (size_t p1=0; p1<leaf1.size(); ++p1) {
            int nn1 = leaf1[p1]->getN();
            for (int q1=0; q1<nn1; ++q1) {
                int index1;
                if (nn1 == 1) index1 = leaf1[p1]->getInfo().index;
                else index1 = (*leaf1[p1]->getListInfo().indices)[q1];
                for (size_t p2=0; p2<leaf2.size(); ++p2) {
                    int nn2 = leaf2[p2]->getN();
                    for (int q2=0; q2<nn2; ++q2) {
                        int index2;
                        if (nn2 == 1) index2 = leaf2[p2]->getInfo().index;
                        else index2 = (*leaf2[p2]->getListInfo().indices)[q2];
                        int j = k;  // j is where in the lists we will place this
                        if (k >= n) {
                            double urd = rand();
                            urd /= RAND_MAX;  // 0 < urd < 1
                            j = int(urd * (k+1)); // 0 <= j < k+1
                        }
                        if (j < n)  {
                            i1[j] = index1;
                            i2[j] = index2;
                            sep[j] = r;
                        }
                        ++k;
                    }
                }
            }
        }
    } else {
        // Case 3
        xdbg<<"Case 3: Select n without replacement\n";
        std::vector<long> selection(n);
        SelectRandomFrom(k+m, selection);
        // If any items in k<=i<n are from the original set, put them in their original place.
        for(int i=k;i<n;++i) {
            int j = selection[i];
            if (j < n) std::swap(selection[i], selection[j]);
        }

        // Now any items with j >= k can replace the value at their i in this list.
        std::map<long, long> places;
        for(int i=0;i<n;++i) {
            int j = selection[i];
            if (j >= k) places[j] = i;
        }
        if (places.size() == 0) {
            // If nothing selected from the new set, then we're done.
            k += m;
            return;
        }

        std::map<long, long>::iterator next = places.begin();

        int i=k;  // When i is in the map, place it at places[i]
        for (size_t p1=0; p1<leaf1.size(); ++p1) {
            int nn1 = leaf1[p1]->getN();
            for (int q1=0; q1<nn1; ++q1) {
                xdbg<<"i = "<<i<<", next = "<<next->first<<"  "<<next->second<<std::endl;
                Assert(i <= next->first);
                if (next->first > i + n2) {
                    // Then skip this loop through the second vector
                    i += n2;
                    continue;
                }
                int index1;
                if (nn1 == 1) index1 = leaf1[p1]->getInfo().index;
                else index1 = (*leaf1[p1]->getListInfo().indices)[q1];
                for (size_t p2=0; p2<leaf2.size(); ++p2) {
                    int nn2 = leaf2[p2]->getN();
                    for (int q2=0; q2<nn2; ++q2,++i) {
                        if (i == next->first) {
                            xdbg<<"Use i = "<<i<<std::endl;
                            int index2;
                            if (nn2 == 1) index2 = leaf2[p2]->getInfo().index;
                            else index2 = (*leaf2[p2]->getListInfo().indices)[q2];
                            long j = next->second;
                            i1[j] = index1;
                            i2[j] = index2;
                            sep[j] = r;
                            ++next;
                        }
                        if (next == places.end()) break;
                    }
                    if (next == places.end()) break;
                }
                if (next == places.end()) break;
            }
            if (next == places.end()) break;
        }
        xdbg<<"Done: i = "<<i<<", k+m = "<<k+m<<std::endl;
        k += m;
    }
}


//
//
// The C interface for python
//
//

extern "C" {
#include "BinnedCorr2_C.h"
}

template <int D1, int D2>
void* BuildCorr2b(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar, double xp, double yp, double zp,
                  double* xi0, double* xi1, double* xi2, double* xi3,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    switch(bin_type) {
      case Log:
           return static_cast<void*>(new BinnedCorr2<D1,D2,Log>(
                   minsep, maxsep, nbins, binsize, b, minrpar, maxrpar, xp, yp, zp,
                   xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs));
           break;
      case Linear:
           return static_cast<void*>(new BinnedCorr2<D1,D2,Linear>(
                   minsep, maxsep, nbins, binsize, b, minrpar, maxrpar, xp, yp, zp,
                   xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs));
           break;
      case TwoD:
           return static_cast<void*>(new BinnedCorr2<D1,D2,TwoD>(
                   minsep, maxsep, nbins, binsize, b, minrpar, maxrpar, xp, yp, zp,
                   xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs));
           break;
      default:
           Assert(false);
    }
    return 0;
}

template <int D1>
void* BuildCorr2a(int d2, int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar, double xp, double yp, double zp,
                  double* xi0, double* xi1, double* xi2, double* xi3,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    // Note: we only ever call this with d2 >= d1, so the MAX bit below is equivalent to
    // just using d2 for the cases that actually get called, but doing this saves some
    // compile time and some size in the final library from not instantiating templates
    // that aren't needed.
    switch(d2) {
      case NData:
           return BuildCorr2b<D1,MAX(D1,NData)>(bin_type,
                                                minsep, maxsep, nbins, binsize, b,
                                                minrpar, maxrpar, xp, yp, zp,
                                                xi0, xi1, xi2, xi3,
                                                meanr, meanlogr, weight, npairs);
           break;
      case KData:
           return BuildCorr2b<D1,MAX(D1,KData)>(bin_type,
                                                minsep, maxsep, nbins, binsize, b,
                                                minrpar, maxrpar, xp, yp, zp,
                                                xi0, xi1, xi2, xi3,
                                                meanr, meanlogr, weight, npairs);
           break;
      case GData:
           return BuildCorr2b<D1,MAX(D1,GData)>(bin_type,
                                                minsep, maxsep, nbins, binsize, b,
                                                minrpar, maxrpar, xp, yp, zp,
                                                xi0, xi1, xi2, xi3,
                                                meanr, meanlogr, weight, npairs);
           break;
      default:
           Assert(false);
    }
    return 0;
}


void* BuildCorr2(int d1, int d2, int bin_type,
                 double minsep, double maxsep, int nbins, double binsize, double b,
                 double minrpar, double maxrpar, double xp, double yp, double zp,
                 double* xi0, double* xi1, double* xi2, double* xi3,
                 double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildCorr2: "<<d1<<" "<<d2<<" "<<bin_type<<std::endl;
    void* corr=0;
    switch(d1) {
      case NData:
           corr = BuildCorr2a<NData>(d2, bin_type,
                                     minsep, maxsep, nbins, binsize, b,
                                     minrpar, maxrpar, xp, yp, zp,
                                     xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs);
           break;
      case KData:
           corr = BuildCorr2a<KData>(d2, bin_type,
                                     minsep, maxsep, nbins, binsize, b,
                                     minrpar, maxrpar, xp, yp, zp,
                                     xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs);
           break;
      case GData:
           corr = BuildCorr2a<GData>(d2, bin_type,
                                     minsep, maxsep, nbins, binsize, b,
                                     minrpar, maxrpar, xp, yp, zp,
                                     xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs);
           break;
      default:
           Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

template <int D1, int D2>
void DestroyCorr2b(void* corr, int bin_type)
{
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<D1,D2,Log>*>(corr);
           break;
      case Linear:
           delete static_cast<BinnedCorr2<D1,D2,Linear>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<D1,D2,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}

template <int D1>
void DestroyCorr2a(void* corr, int d2, int bin_type)
{
    switch(d2) {
      case NData:
           DestroyCorr2b<D1,MAX(D1,NData)>(corr, bin_type);
           break;
      case KData:
           DestroyCorr2b<D1,MAX(D1,KData)>(corr, bin_type);
           break;
      case GData:
           DestroyCorr2b<D1,MAX(D1,GData)>(corr, bin_type);
           break;
      default:
           Assert(false);
    }
}

void DestroyCorr2(void* corr, int d1, int d2, int bin_type)
{
    dbg<<"Start DestroyCorr2: "<<d1<<" "<<d2<<" "<<bin_type<<std::endl;
    xdbg<<"corr = "<<corr<<std::endl;
    switch(d1) {
      case NData:
           DestroyCorr2a<NData>(corr, d2, bin_type);
           break;
      case KData:
           DestroyCorr2a<KData>(corr, d2, bin_type);
           break;
      case GData:
           DestroyCorr2a<GData>(corr, d2, bin_type);
           break;
      default:
           Assert(false);
    }
}

template <int M, int D, int B>
void ProcessAuto2d(BinnedCorr2<D,D,B>* corr, void* field, int dots, int coords)
{
    switch(coords) {
      case Flat:
           Assert(MetricHelper<M>::_Flat == int(Flat));
           corr->template process<MetricHelper<M>::_Flat, M>(
               *static_cast<Field<D,MetricHelper<M>::_Flat>*>(field), dots);
           break;
      case Sphere:
           Assert(MetricHelper<M>::_Sphere == int(Sphere));
           corr->template process<MetricHelper<M>::_Sphere, M>(
               *static_cast<Field<D,MetricHelper<M>::_Sphere>*>(field), dots);
           break;
      case ThreeD:
           Assert(MetricHelper<M>::_ThreeD == int(ThreeD));
           corr->template process<MetricHelper<M>::_ThreeD, M>(
               *static_cast<Field<D,MetricHelper<M>::_ThreeD>*>(field), dots);
           break;
      default:
           Assert(false);
    }
}

template <int D, int B>
void ProcessAuto2c(BinnedCorr2<D,D,B>* corr, void* field, int dots,
                   int coords, int metric)
{
    switch(metric) {
      case Euclidean:
           ProcessAuto2d<Euclidean>(corr, field, dots, coords);
           break;
      case Rperp:
           ProcessAuto2d<Rperp>(corr, field, dots, coords);
           break;
      case OldRperp:
           ProcessAuto2d<OldRperp>(corr, field, dots, coords);
           break;
      case Rlens:
           ProcessAuto2d<Rlens>(corr, field, dots, coords);
           break;
      case Arc:
           ProcessAuto2d<Arc>(corr, field, dots, coords);
           break;
      case Periodic:
           ProcessAuto2d<Periodic>(corr, field, dots, coords);
           break;
      default:
           Assert(false);
    }
}

template <int D>
void ProcessAuto2b(void* corr, void* field, int dots, int coords, int bin_type, int metric)
{
    switch(bin_type) {
      case Log:
           ProcessAuto2c(static_cast<BinnedCorr2<D,D,Log>*>(corr), field, dots, coords, metric);
           break;
      case Linear:
           ProcessAuto2c(static_cast<BinnedCorr2<D,D,Linear>*>(corr), field, dots, coords, metric);
           break;
      case TwoD:
           ProcessAuto2c(static_cast<BinnedCorr2<D,D,TwoD>*>(corr), field, dots, coords, metric);
           break;
      default:
           Assert(false);
    }
}

void ProcessAuto2(void* corr, void* field, int dots,
                  int d, int coords, int bin_type, int metric)
{
    dbg<<"Start ProcessAuto2: "<<d<<" "<<coords<<" "<<bin_type<<" "<<metric<<std::endl;

    switch(d) {
      case NData:
           ProcessAuto2b<NData>(corr, field, dots, coords, bin_type, metric);
           break;
      case KData:
           ProcessAuto2b<KData>(corr, field, dots, coords, bin_type, metric);
           break;
      case GData:
           ProcessAuto2b<GData>(corr, field, dots, coords, bin_type, metric);
           break;
      default:
           Assert(false);
    }
}

template <int M, int D1, int D2, int B>
void ProcessCross2d(BinnedCorr2<D1,D2,B>* corr, void* field1, void* field2, int dots, int coords)
{
    switch(coords) {
      case Flat:
           Assert(MetricHelper<M>::_Flat == int(Flat));
           corr->template process<MetricHelper<M>::_Flat, M>(
               *static_cast<Field<D1,MetricHelper<M>::_Flat>*>(field1),
               *static_cast<Field<D2,MetricHelper<M>::_Flat>*>(field2), dots);
           break;
      case Sphere:
           Assert(MetricHelper<M>::_Sphere == int(Sphere));
           corr->template process<MetricHelper<M>::_Sphere, M>(
               *static_cast<Field<D1,MetricHelper<M>::_Sphere>*>(field1),
               *static_cast<Field<D2,MetricHelper<M>::_Sphere>*>(field2), dots);
           break;
      case ThreeD:
           Assert(MetricHelper<M>::_ThreeD == int(ThreeD));
           corr->template process<MetricHelper<M>::_ThreeD, M>(
               *static_cast<Field<D1,MetricHelper<M>::_ThreeD>*>(field1),
               *static_cast<Field<D2,MetricHelper<M>::_ThreeD>*>(field2), dots);
           break;
      default:
           Assert(false);
    }
}

template <int D1, int D2, int B>
void ProcessCross2c(BinnedCorr2<D1,D2,B>* corr, void* field1, void* field2, int dots,
                    int coords, int metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCross2d<Euclidean>(corr, field1, field2, dots, coords);
           break;
      case Rperp:
           ProcessCross2d<Rperp>(corr, field1, field2, dots, coords);
           break;
      case OldRperp:
           ProcessCross2d<OldRperp>(corr, field1, field2, dots, coords);
           break;
      case Rlens:
           ProcessCross2d<Rlens>(corr, field1, field2, dots, coords);
           break;
      case Arc:
           ProcessCross2d<Arc>(corr, field1, field2, dots, coords);
           break;
      case Periodic:
           ProcessCross2d<Periodic>(corr, field1, field2, dots, coords);
           break;
      default:
           Assert(false);
    }
}

template <int D1, int D2>
void ProcessCross2b(void* corr, void* field1, void* field2, int dots,
                    int coords, int bin_type, int metric)
{
    switch(bin_type) {
      case Log:
           ProcessCross2c(static_cast<BinnedCorr2<D1,D2,Log>*>(corr), field1, field2, dots,
                          coords, metric);
           break;
      case Linear:
           ProcessCross2c(static_cast<BinnedCorr2<D1,D2,Linear>*>(corr), field1, field2, dots,
                          coords, metric);
           break;
      case TwoD:
           ProcessCross2c(static_cast<BinnedCorr2<D1,D2,TwoD>*>(corr), field1, field2, dots,
                          coords, metric);
           break;
      default:
           Assert(false);
    }
}

template <int D1>
void ProcessCross2a(void* corr, void* field1, void* field2, int dots,
                    int d2, int coords, int bin_type, int metric)
{
    // Note: we only ever call this with d2 >= d1, so the MAX bit below is equivalent to
    // just using d2 for the cases that actually get called, but doing this saves some
    // compile time and some size in the final library from not instantiating templates
    // that aren't needed.
    Assert(d2 >= D1);
    switch(d2) {
      case NData:
           ProcessCross2b<D1,MAX(D1,NData)>(corr, field1, field2, dots,
                                            coords, bin_type, metric);
           break;
      case KData:
           ProcessCross2b<D1,MAX(D1,KData)>(corr, field1, field2, dots,
                                            coords, bin_type, metric);
           break;
      case GData:
           ProcessCross2b<D1,MAX(D1,GData)>(corr, field1, field2, dots,
                                            coords, bin_type, metric);
           break;
      default:
           Assert(false);
    }
}

void ProcessCross2(void* corr, void* field1, void* field2, int dots,
                   int d1, int d2, int coords, int bin_type, int metric)
{
    dbg<<"Start ProcessCross2: "<<d1<<" "<<d2<<" "<<coords<<" "<<bin_type<<" "<<metric<<std::endl;

    switch(d1) {
      case NData:
           ProcessCross2a<NData>(corr, field1, field2, dots,
                                 d2, coords, bin_type, metric);
           break;
      case KData:
           ProcessCross2a<KData>(corr, field1, field2, dots,
                                 d2, coords, bin_type, metric);
           break;
      case GData:
           ProcessCross2a<GData>(corr, field1, field2, dots,
                                 d2, coords, bin_type, metric);
           break;
      default:
           Assert(false);
    }
}

template <int M, int D1, int D2, int B>
void ProcessPair2d(BinnedCorr2<D1,D2,B>* corr, void* field1, void* field2, int dots, int coords)
{
    switch(coords) {
      case Flat:
           Assert(MetricHelper<M>::_Flat == int(Flat));
           corr->template processPairwise<MetricHelper<M>::_Flat, M>(
               *static_cast<SimpleField<D1,MetricHelper<M>::_Flat>*>(field1),
               *static_cast<SimpleField<D2,MetricHelper<M>::_Flat>*>(field2), dots);
           break;
      case Sphere:
           Assert(MetricHelper<M>::_Sphere == int(Sphere));
           corr->template processPairwise<MetricHelper<M>::_Sphere, M>(
               *static_cast<SimpleField<D1,MetricHelper<M>::_Sphere>*>(field1),
               *static_cast<SimpleField<D2,MetricHelper<M>::_Sphere>*>(field2), dots);
           break;
      case ThreeD:
           Assert(MetricHelper<M>::_ThreeD == int(ThreeD));
           corr->template processPairwise<MetricHelper<M>::_ThreeD, M>(
               *static_cast<SimpleField<D1,MetricHelper<M>::_ThreeD>*>(field1),
               *static_cast<SimpleField<D2,MetricHelper<M>::_ThreeD>*>(field2), dots);
           break;
      default:
           Assert(false);
    }
}
    template <int D1, int D2, int B>
    void ProcessPair2c(BinnedCorr2<D1,D2,B>* corr, void* field1, void* field2, int dots,
                       int coords, int metric)
{
    switch(metric) {
      case Euclidean:
           ProcessPair2d<Euclidean>(corr, field1, field2, dots, coords);
           break;
      case Rperp:
           ProcessPair2d<Rperp>(corr, field1, field2, dots, coords);
           break;
      case OldRperp:
           ProcessPair2d<OldRperp>(corr, field1, field2, dots, coords);
           break;
      case Rlens:
           ProcessPair2d<Rlens>(corr, field1, field2, dots, coords);
           break;
      case Arc:
           ProcessPair2d<Arc>(corr, field1, field2, dots, coords);
           break;
      case Periodic:
           ProcessPair2d<Periodic>(corr, field1, field2, dots, coords);
           break;
      default:
           Assert(false);
    }
}

template <int D1, int D2>
void ProcessPair2b(void* corr, void* field1, void* field2, int dots,
                   int coords, int bin_type, int metric)
{
    switch(bin_type) {
      case Log:
           ProcessPair2c(static_cast<BinnedCorr2<D1,D2,Log>*>(corr), field1, field2, dots,
                         coords, metric);
           break;
      case Linear:
           ProcessPair2c(static_cast<BinnedCorr2<D1,D2,Linear>*>(corr), field1, field2, dots,
                         coords, metric);
           break;
      case TwoD:
           ProcessPair2c(static_cast<BinnedCorr2<D1,D2,TwoD>*>(corr), field1, field2, dots,
                         coords, metric);
           break;
      default:
           Assert(false);
    }
}

template <int D1>
void ProcessPair2a(void* corr, void* field1, void* field2, int dots,
                   int d2, int coords, int bin_type, int metric)
{
    Assert(d2 >= D1);
    switch(d2) {
      case NData:
           ProcessPair2b<D1,MAX(D1,NData)>(corr, field1, field2, dots,
                                           coords, bin_type, metric);
           break;
      case KData:
           ProcessPair2b<D1,MAX(D1,KData)>(corr, field1, field2, dots,
                                           coords, bin_type, metric);
           break;
      case GData:
           ProcessPair2b<D1,MAX(D1,GData)>(corr, field1, field2, dots,
                                           coords, bin_type, metric);
           break;
      default:
           Assert(false);
    }
}

void ProcessPair(void* corr, void* field1, void* field2, int dots,
                 int d1, int d2, int coords, int bin_type, int metric)
{
    dbg<<"Start ProcessPair: "<<d1<<" "<<d2<<" "<<coords<<" "<<bin_type<<" "<<metric<<std::endl;

    switch(d1) {
      case NData:
           ProcessPair2a<NData>(corr, field1, field2, dots,
                                d2, coords, bin_type, metric);
           break;
      case KData:
           ProcessPair2a<KData>(corr, field1, field2, dots,
                                d2, coords, bin_type, metric);
           break;
      case GData:
           ProcessPair2a<GData>(corr, field1, field2, dots,
                                d2, coords, bin_type, metric);
           break;
      default:
           Assert(false);
    }
}


int SetOMPThreads(int num_threads)
{
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <int M, int D1, int D2, int B>
long SamplePairs2d(BinnedCorr2<D1,D2,B>* corr, void* field1, void* field2,
                   double minsep, double maxsep,
                   int coords, long* i1, long* i2, double* sep, int n)
{
    switch(coords) {
      case Flat:
           Assert(MetricHelper<M>::_Flat == int(Flat));
           return corr->template samplePairs<MetricHelper<M>::_Flat, M>(
               *static_cast<Field<D1,MetricHelper<M>::_Flat>*>(field1),
               *static_cast<Field<D2,MetricHelper<M>::_Flat>*>(field2),
               minsep, maxsep, i1, i2, sep, n);
           break;
      case Sphere:
           Assert(MetricHelper<M>::_Sphere == int(Sphere));
           return corr->template samplePairs<MetricHelper<M>::_Sphere, M>(
               *static_cast<Field<D1,MetricHelper<M>::_Sphere>*>(field1),
               *static_cast<Field<D2,MetricHelper<M>::_Sphere>*>(field2),
               minsep, maxsep, i1, i2, sep, n);
           break;
      case ThreeD:
           Assert(MetricHelper<M>::_ThreeD == int(ThreeD));
           return corr->template samplePairs<MetricHelper<M>::_ThreeD, M>(
               *static_cast<Field<D1,MetricHelper<M>::_ThreeD>*>(field1),
               *static_cast<Field<D2,MetricHelper<M>::_ThreeD>*>(field2),
               minsep, maxsep, i1, i2, sep, n);
           break;
      default:
           Assert(false);
    }
    return 0;
}
    template <int D1, int D2, int B>
    long SamplePairs2c(BinnedCorr2<D1,D2,B>* corr, void* field1, void* field2,
                       double minsep, double maxsep,
                       int coords, int metric, long* i1, long* i2, double* sep, int n)
{
    switch(metric) {
      case Euclidean:
           return SamplePairs2d<Euclidean>(corr, field1, field2, minsep, maxsep,
                                           coords, i1, i2, sep, n);
           break;
      case Rperp:
           return SamplePairs2d<Rperp>(corr, field1, field2, minsep, maxsep,
                                       coords, i1, i2, sep, n);
           break;
      case OldRperp:
           return SamplePairs2d<OldRperp>(corr, field1, field2, minsep, maxsep,
                                          coords, i1, i2, sep, n);
           break;
      case Rlens:
           return SamplePairs2d<Rlens>(corr, field1, field2, minsep, maxsep,
                                       coords, i1, i2, sep, n);
           break;
      case Arc:
           return SamplePairs2d<Arc>(corr, field1, field2, minsep, maxsep,
                                     coords, i1, i2, sep, n);
           break;
      case Periodic:
           return SamplePairs2d<Periodic>(corr, field1, field2, minsep, maxsep,
                                          coords, i1, i2, sep, n);
           break;
      default:
           Assert(false);
    }
    return 0;
}

template <int D1, int D2>
long SamplePairs2b(void* corr, void* field1, void* field2, double minsep, double maxsep,
                   int coords, int bin_type, int metric,
                   long* i1, long* i2, double* sep, int n)
{
    switch(bin_type) {
      case Log:
           return SamplePairs2c(static_cast<BinnedCorr2<D1,D2,Log>*>(corr),
                                field1, field2, minsep, maxsep,
                                coords, metric, i1, i2, sep, n);
           break;
      case Linear:
           return SamplePairs2c(static_cast<BinnedCorr2<D1,D2,Linear>*>(corr),
                                field1, field2, minsep, maxsep,
                                coords, metric, i1, i2, sep, n);
           break;
      case TwoD:
           // TwoD not implemented.
           break;
      default:
           Assert(false);
    }
    return 0;
}

template <int D1>
long SamplePairs2a(void* corr, void* field1, void* field2, double minsep, double maxsep,
                   int d2, int coords, int bin_type, int metric,
                   long* i1, long* i2, double* sep, int n)
{
    Assert(d2 >= D1);
    switch(d2) {
      case NData:
           return SamplePairs2b<D1,MAX(D1,NData)>(corr, field1, field2, minsep, maxsep,
                                                  coords, bin_type, metric, i1, i2, sep, n);
           break;
      case KData:
           return SamplePairs2b<D1,MAX(D1,KData)>(corr, field1, field2, minsep, maxsep,
                                                  coords, bin_type, metric, i1, i2, sep, n);
           break;
      case GData:
           return SamplePairs2b<D1,MAX(D1,GData)>(corr, field1, field2, minsep, maxsep,
                                                  coords, bin_type, metric, i1, i2, sep, n);
           break;
      default:
           Assert(false);
    }
    return 0;
}

long SamplePairs(void* corr, void* field1, void* field2, double minsep, double maxsep,
                 int d1, int d2, int coords, int bin_type, int metric,
                 long* i1, long* i2, double* sep, int n)
{
    dbg<<"Start SamplePairs: "<<d1<<" "<<d2<<" "<<coords<<" "<<bin_type<<" "<<metric<<std::endl;

    switch(d1) {
      case NData:
           return SamplePairs2a<NData>(corr, field1, field2, minsep, maxsep,
                                       d2, coords, bin_type, metric, i1, i2, sep, n);
           break;
      case KData:
           return SamplePairs2a<KData>(corr, field1, field2, minsep, maxsep,
                                       d2, coords, bin_type, metric, i1, i2, sep, n);
           break;
      case GData:
           return SamplePairs2a<GData>(corr, field1, field2, minsep, maxsep,
                                       d2, coords, bin_type, metric, i1, i2, sep, n);
           break;
      default:
           Assert(false);
    }
    return 0;
}

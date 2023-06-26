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

#include "PyBind11Helper.h"

#include <vector>
#include <set>
#include <map>

#include "dbg.h"
#include "Corr2.h"
#include "Split.h"
#include "ProjectHelper.h"
#include "Metric.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// When we need a compile-time max, use this rather than std::max, which only became valid
// for compile-time constexpr in C++14, which we don't require.
#define MAX(a,b) (a > b ? a : b)

double CalculateFullMaxSep(BinType bin_type, double minsep, double maxsep, int nbins,
                           double binsize)
{
    switch(bin_type) {
      case Log:
           return BinTypeHelper<Log>::calculateFullMaxSep(minsep, maxsep, nbins, binsize);
      case Linear:
           return BinTypeHelper<Linear>::calculateFullMaxSep(minsep, maxsep, nbins, binsize);
      case TwoD:
           return BinTypeHelper<TwoD>::calculateFullMaxSep(minsep, maxsep, nbins, binsize);
      default:
           Assert(false);
    }
    return 0.;
}

BaseCorr2::BaseCorr2(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
    double minrpar, double maxrpar, double xp, double yp, double zp) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minrpar(minrpar), _maxrpar(maxrpar), _xp(xp), _yp(yp), _zp(zp), _coords(-1)
{
    dbg<<"Corr2 constructor\n";
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _bsq = _b * _b;
    _fullmaxsep = CalculateFullMaxSep(bin_type, minsep, maxsep, nbins, binsize);
    _fullmaxsepsq = _fullmaxsep*_fullmaxsep;
    dbg<<"minsep, maxsep = "<<_minsep<<"  "<<_maxsep<<std::endl;
    dbg<<"nbins = "<<_nbins<<std::endl;
    dbg<<"binsize = "<<_binsize<<std::endl;
    dbg<<"b = "<<_b<<std::endl;
    dbg<<"minrpar, maxrpar = "<<_minrpar<<"  "<<_maxrpar<<std::endl;
    dbg<<"period = "<<_xp<<"  "<<_yp<<"  "<<_zp<<std::endl;
}


template <int D1, int D2>
Corr2<D1,D2>::Corr2(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
    double minrpar, double maxrpar, double xp, double yp, double zp,
    double* xi0, double* xi1, double* xi2, double* xi3,
    double* meanr, double* meanlogr, double* weight, double* npairs) :
    BaseCorr2(bin_type, minsep, maxsep, nbins, binsize, b, minrpar, maxrpar, xp, yp, zp),
    _owns_data(false),
    _xi(xi0,xi1,xi2,xi3), _meanr(meanr), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
{}

BaseCorr2::BaseCorr2(const BaseCorr2& rhs) :
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minrpar(rhs._minrpar), _maxrpar(rhs._maxrpar),
    _xp(rhs._xp), _yp(rhs._yp), _zp(rhs._zp),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq), _bsq(rhs._bsq),
    _fullmaxsep(rhs._fullmaxsep), _fullmaxsepsq(rhs._fullmaxsepsq),
    _coords(rhs._coords)
{}

template <int D1, int D2>
Corr2<D1,D2>::Corr2(const Corr2<D1,D2>& rhs, bool copy_data) :
    BaseCorr2(rhs), _owns_data(true), _xi(0,0,0,0), _weight(0)
{
    dbg<<"Corr2 copy constructor\n";
    _xi.new_data(_nbins);
    _meanr = new double[_nbins];
    _meanlogr = new double[_nbins];
    _weight = new double[_nbins];
    _npairs = new double[_nbins];

    if (copy_data) *this = rhs;
    else clear();
}

template <int D1, int D2>
Corr2<D1,D2>::~Corr2()
{
    dbg<<"Corr2 destructor\n";
    if (_owns_data) {
        _xi.delete_data(_nbins);
        delete [] _meanr; _meanr = 0;
        delete [] _meanlogr; _meanlogr = 0;
        delete [] _weight; _weight = 0;
        delete [] _npairs; _npairs = 0;
    }
}

// Corr2::process2 is invalid if D1 != D2, so this helper struct lets us only call
// process2 when D1 == D2.
template <int D1, int D2, int B, int M, int P, int C>
struct ProcessHelper
{
    static void process2(Corr2<D1,D2>& , const BaseCell<C>&, const MetricHelper<M,P>& ) {}
};

template <int D, int B, int M, int P, int C>
struct ProcessHelper<D,D,B,M,P,C>
{
    static void process2(Corr2<D,D>& b, const BaseCell<C>& c12, const MetricHelper<M,P>& m)
    { b.template process2<B,M,P>(c12, m); }
};

template <int D1, int D2>
void Corr2<D1,D2>::clear()
{
    _xi.clear(_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _weight[i] = 0.;
    for (int i=0; i<_nbins; ++i) _npairs[i] = 0.;
    _coords = -1;
}

template <int D1, int D2> template <int B, int M, int P, int C>
void Corr2<D1,D2>::process(const Field<D1,C>& field, bool dots)
{
    xdbg<<"Start process (auto): M,P,C = "<<M<<"  "<<P<<"  "<<C<<std::endl;
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
        Corr2<D1,D2> bc2(*this,false);
#else
        Corr2<D1,D2>& bc2 = *this;
#endif

        // Inside the omp parallel, so each thread has its own MetricHelper.
        MetricHelper<M,P> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                if (dots) std::cout<<'.'<<std::flush;
            }
            const BaseCell<C>& c1 = *field.getCells()[i];
            ProcessHelper<D1,D2,B,M,P,C>::process2(bc2, c1, metric);
            for (long j=i+1;j<n1;++j) {
                const BaseCell<C>& c2 = *field.getCells()[j];
                bc2.process11<B,M,P,BinTypeHelper<B>::do_reverse>(c1, c2, metric);
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

template <int D1, int D2> template <int B, int M, int P, int C>
void Corr2<D1,D2>::process(const Field<D1,C>& field1, const Field<D2,C>& field2, bool dots)
{
    xdbg<<"Start process (cross): M,P,C = "<<M<<"  "<<P<<"  "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;

    // Check if we can early exit.
    MetricHelper<M,P> metric1(_minrpar, _maxrpar, _xp, _yp, _zp);
    const Position<C>& p1 = field1.getCenter();
    const Position<C>& p2 = field2.getCenter();
    double s1 = field1.getSize();
    double s2 = field2.getSize();
    const double rsq = metric1.DistSq(p1, p2, s1, s2);
    double s1ps2 = s1 + s2;
    double rpar = 0; // Gets set to correct value by isRParOutsideRange if appropriate
    if (metric1.isRParOutsideRange(p1, p2, s1ps2, rpar) ||
        (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, _minsep, _minsepsq) &&
         metric1.tooSmallDist(p1, p2, rsq, rpar, s1ps2, _minsep, _minsepsq)) ||
        (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq) &&
         metric1.tooLargeDist(p1, p2, rsq, rpar, s1ps2, _fullmaxsep, _fullmaxsepsq))) {
        dbg<<"Fields have no relevant coverage.  Early exit.\n";
        return;
    }

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
        Corr2<D1,D2> bc2(*this,false);
#else
        Corr2<D1,D2>& bc2 = *this;
#endif

        MetricHelper<M,P> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                if (dots) std::cout<<'.'<<std::flush;
            }
            const BaseCell<C>& c1 = *field1.getCells()[i];
            for (long j=0;j<n2;++j) {
                const BaseCell<C>& c2 = *field2.getCells()[j];
                bc2.process11<B,M,P,false>(c1, c2, metric);
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

template <int D1, int D2> template <int B, int M, int P, int C>
void Corr2<D1,D2>::process2(const BaseCell<C>& c12, const MetricHelper<M,P>& metric)
{
    if (c12.getW() == 0.) return;
    if (c12.getSize() <= _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2<B,M,P>(*c12.getLeft(), metric);
    process2<B,M,P>(*c12.getRight(), metric);
    process11<B,M,P,BinTypeHelper<B>::do_reverse>(*c12.getLeft(), *c12.getRight(), metric);
}

template <int D1, int D2> template <int B, int M, int P, int R, int C>
void Corr2<D1,D2>::process11(const BaseCell<C>& c1, const BaseCell<C>& c2,
                             const MetricHelper<M,P>& metric)
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
            directProcess11<B,R>(c1,c2,rsq,k,r,logr);
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
            process11<B,M,P,R>(*c1.getLeft(),*c2.getLeft(),metric);
            process11<B,M,P,R>(*c1.getLeft(),*c2.getRight(),metric);
            process11<B,M,P,R>(*c1.getRight(),*c2.getLeft(),metric);
            process11<B,M,P,R>(*c1.getRight(),*c2.getRight(),metric);
        } else if (split1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11<B,M,P,R>(*c1.getLeft(),c2,metric);
            process11<B,M,P,R>(*c1.getRight(),c2,metric);
        } else {
            Assert(split2);
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<B,M,P,R>(c1,*c2.getLeft(),metric);
            process11<B,M,P,R>(c1,*c2.getRight(),metric);
        }
    }
}


// We also set up a helper class for doing the direct processing
template <int D1, int D2>
struct DirectHelper;

template <>
struct DirectHelper<NData,NData>
{
    template <int R, int C>
    static void ProcessXi(
        const Cell<NData,C>& , const Cell<NData,C>& , const double ,
        XiData<NData,NData>& , int, int )
    {}
};

template <>
struct DirectHelper<NData,KData>
{
    template <int R, int C>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<KData,C>& c2, const double ,
        XiData<NData,KData>& xi, int k, int )
    { xi.xi[k] += c1.getW() * c2.getData().getWK(); }
};

template <>
struct DirectHelper<NData,GData>
{
    template <int R, int C>
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
    template <int R, int C>
    static void ProcessXi(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2, const double ,
        XiData<KData,KData>& xi, int k, int k2)
    {
        double wkk = c1.getData().getWK() * c2.getData().getWK();
        xi.xi[k] += wkk;
        if (R) {
            xi.xi[k2] += wkk;
        }
    }
};

template <>
struct DirectHelper<KData,GData>
{
    template <int R, int C>
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
    template <int R, int C>
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

        if (R) {
            xi.xip[k2] += g1rg2r + g1ig2i;       // g1 * conj(g2)
            xi.xip_im[k2] += g1ig2r - g1rg2i;
            xi.xim[k2] += g1rg2r - g1ig2i;       // g1 * g2
            xi.xim_im[k2] += g1ig2r + g1rg2i;
        }
    }
};

template <int D1, int D2> template <int B, int R, int C>
void Corr2<D1,D2>::directProcess11(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const double rsq,
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
    Assert(k <= _nbins);
    // It's possible for k to be == _nbins here if the r is very close to the top of the
    // last bin, but numerical rounding in the log calculation bumped k into the next
    // (non-existent) bin.
    if (k == _nbins) {
        XAssert(BinTypeHelper<B>::calculateBinK(p2, p1, r, logr-1.e-10, _binsize,
                                                _minsep, _maxsep, _logminsep) == _nbins-1);
        --k;
    }
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
    if (R) {
        k2 = BinTypeHelper<B>::calculateBinK(p2, p1, r, logr, _binsize,
                                             _minsep, _maxsep, _logminsep);
        if (k2 == _nbins) --k2;  // As before, this can (rarely) happen.
        Assert(k2 >= 0);
        Assert(k2 < _nbins);
        _npairs[k2] += nn;
        _meanr[k2] += ww * r;
        _meanlogr[k2] += ww * logr;
        _weight[k2] += ww;
    }

    DirectHelper<D1,D2>::template ProcessXi<R,C>(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        rsq,_xi,k,k2);
}

template <int D1, int D2>
void Corr2<D1,D2>::operator=(const Corr2<D1,D2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.copy(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template <int D1, int D2>
void Corr2<D1,D2>::operator+=(const Corr2<D1,D2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.add(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] += rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] += rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] += rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] += rhs._npairs[i];
}

template <int B, int M, int C>
bool BaseCorr2::triviallyZero(Position<C> p1, Position<C> p2, double s1, double s2)
{
    // Ignore any min/max rpar for this calculation.
    double minrpar = -std::numeric_limits<double>::max();
    double maxrpar = std::numeric_limits<double>::max();
    MetricHelper<M,0> metric(minrpar, maxrpar, _xp, _yp, _zp);
    const double rsq = metric.DistSq(p1, p2, s1, s2);
    double s1ps2 = s1 + s2;
    double rpar = 0;
    return (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq) &&
            metric.tooLargeDist(p1, p2, rsq, rpar, s1ps2, _fullmaxsep, _fullmaxsepsq));
}

template <int B, int M, int P, int C>
long BaseCorr2::samplePairs(
    const BaseField<C>& field1, const BaseField<C>& field2,
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

    MetricHelper<M,P> metric(_minrpar, _maxrpar, _xp, _yp, _zp);

    double minsepsq = minsep*minsep;
    double maxsepsq = maxsep*maxsep;

    long k=0;
    for (long i=0;i<n1;++i) {
        const BaseCell<C>& c1 = *field1.getCells()[i];
        for (long j=0;j<n2;++j) {
            const BaseCell<C>& c2 = *field2.getCells()[j];
            samplePairs<B>(c1, c2, metric, minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        }
    }
    return k;
}

template <int B, int M, int P, int C>
void BaseCorr2::samplePairs(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const MetricHelper<M,P>& metric,
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
            sampleFrom<B>(c1,c2,rsq,r,i1,i2,sep,n,k);
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
            samplePairs<B>(*c1.getLeft(), *c2.getLeft(), metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<B>(*c1.getLeft(), *c2.getRight(), metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<B>(*c1.getRight(), *c2.getLeft(), metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<B>(*c1.getRight(), *c2.getRight(), metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        } else if (split1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            samplePairs<B>(*c1.getLeft(), c2, metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<B>(*c1.getRight(), c2, metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
        } else {
            Assert(split2);
            Assert(c2.getLeft());
            Assert(c2.getRight());
            samplePairs<B>(c1, *c2.getLeft(), metric,
                           minsep, minsepsq, maxsep, maxsepsq, i1, i2, sep, n, k);
            samplePairs<B>(c1, *c2.getRight(), metric,
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
            double urd = urand();
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
            double urd = urand();
            long j = long(urd * (m-i)); // 0 <= j < m-i
            j += i;                     // i <= j < m
            if (j == m) j = m-1;        // Just in case.
            std::swap(full[i], full[j]);
        }
        std::copy(full.begin(), full.begin()+n, selection.begin());
    }
}

template <int B, int C>
void BaseCorr2::sampleFrom(
    const BaseCell<C>& c1, const BaseCell<C>& c2, double rsq, double r,
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

    std::vector<const BaseCell<C>*> leaf1 = c1.getAllLeaves();
    std::vector<const BaseCell<C>*> leaf2 = c2.getAllLeaves();

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
            long nn1 = leaf1[p1]->getN();
            for (long q1=0; q1<nn1; ++q1) {
                long index1;
                if (nn1 == 1) index1 = leaf1[p1]->getInfo().index;
                else index1 = (*leaf1[p1]->getListInfo().indices)[q1];
                for (size_t p2=0; p2<leaf2.size(); ++p2) {
                    long nn2 = leaf2[p2]->getN();
                    for (long q2=0; q2<nn2; ++q2) {
                        long index2;
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
            long nn1 = leaf1[p1]->getN();
            for (long q1=0; q1<nn1; ++q1) {
                long index1;
                if (nn1 == 1) index1 = leaf1[p1]->getInfo().index;
                else index1 = (*leaf1[p1]->getListInfo().indices)[q1];
                for (size_t p2=0; p2<leaf2.size(); ++p2) {
                    long nn2 = leaf2[p2]->getN();
                    for (long q2=0; q2<nn2; ++q2) {
                        long index2;
                        if (nn2 == 1) index2 = leaf2[p2]->getInfo().index;
                        else index2 = (*leaf2[p2]->getListInfo().indices)[q2];
                        long j = k;  // j is where in the lists we will place this
                        if (k >= n) {
                            double urd = urand(); // 0 < urd < 1
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
        for(long i=k;i<n;++i) {
            long j = selection[i];
            if (j < n) std::swap(selection[i], selection[j]);
        }

        // Now any items with j >= k can replace the value at their i in this list.
        std::map<long, long> places;
        for(long i=0;i<n;++i) {
            long j = selection[i];
            if (j >= k) places[j] = i;
        }
        if (places.size() == 0) {
            // If nothing selected from the new set, then we're done.
            k += m;
            return;
        }

        std::map<long, long>::iterator next = places.begin();

        long i=k;  // When i is in the map, place it at places[i]
        for (size_t p1=0; p1<leaf1.size(); ++p1) {
            long nn1 = leaf1[p1]->getN();
            for (long q1=0; q1<nn1; ++q1) {
                xdbg<<"i = "<<i<<", next = "<<next->first<<"  "<<next->second<<std::endl;
                Assert(i <= next->first);
                if (next->first > i + n2) {
                    // Then skip this loop through the second vector
                    i += n2;
                    continue;
                }
                long index1;
                if (nn1 == 1) index1 = leaf1[p1]->getInfo().index;
                else index1 = (*leaf1[p1]->getListInfo().indices)[q1];
                for (size_t p2=0; p2<leaf2.size(); ++p2) {
                    long nn2 = leaf2[p2]->getN();
                    for (long q2=0; q2<nn2; ++q2,++i) {
                        if (i == next->first) {
                            xdbg<<"Use i = "<<i<<std::endl;
                            long index2;
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
// The functions we call from Python.
//
//

template <int D1, int D2>
Corr2<D1,D2>* BuildCorr2(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
    double minrpar, double maxrpar, double xp, double yp, double zp,
    py::array_t<double>& xi0p, py::array_t<double>& xi1p,
    py::array_t<double>& xi2p, py::array_t<double>& xi3p,
    py::array_t<double>& meanrp, py::array_t<double>& meanlogrp,
    py::array_t<double>& weightp, py::array_t<double>& npairsp)
{
    double* xi0 = xi0p.size() == 0 ? 0 : static_cast<double*>(xi0p.mutable_data());
    double* xi1 = xi1p.size() == 0 ? 0 : static_cast<double*>(xi1p.mutable_data());
    double* xi2 = xi2p.size() == 0 ? 0 : static_cast<double*>(xi2p.mutable_data());
    double* xi3 = xi3p.size() == 0 ? 0 : static_cast<double*>(xi3p.mutable_data());
    double* meanr = static_cast<double*>(meanrp.mutable_data());
    double* meanlogr = static_cast<double*>(meanlogrp.mutable_data());
    double* weight = static_cast<double*>(weightp.mutable_data());
    double* npairs = static_cast<double*>(npairsp.mutable_data());

    return new Corr2<D1,D2>(
            bin_type, minsep, maxsep, nbins, binsize, b, minrpar, maxrpar, xp, yp, zp,
            xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs);
}

template <int B, int M, int D, int C>
void ProcessAuto2d(Corr2<D,D>* corr, Field<D,C>* field, bool dots)
{
    const bool P = corr->nontrivialRPar();
    dbg<<"ProcessAuto: coords = "<<C<<", metric = "<<M<<", P = "<<P<<std::endl;

    // Only call the actual function if the M/C combination is valid.
    // The Assert checks that this only gets called when the combination is actually valid.
    // But other combinations are nominally instantiated, and give various compiler errors
    // when they are tried.  So this _M business make sure such combinations aren't instantiated.
    // (Plus they would add code that is never used, leading to code bloat.)
    Assert((ValidMC<M,C>::_M == M));
    if (P) {
        Assert(C == ThreeD);
        corr->template process<B, ValidMC<M,C>::_M, (C==ThreeD)>(*field, dots);
    } else {
        corr->template process<B, ValidMC<M,C>::_M, false>(*field, dots);
    }
}

template <int B, int D, int C>
void ProcessAuto2c(Corr2<D,D>* corr, Field<D,C>* field, bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessAuto2d<B,Euclidean>(corr, field, dots);
           break;
      case Rperp:
           ProcessAuto2d<B,Rperp>(corr, field, dots);
           break;
      case OldRperp:
           ProcessAuto2d<B,OldRperp>(corr, field, dots);
           break;
      case Rlens:
           ProcessAuto2d<B,Rlens>(corr, field, dots);
           break;
      case Arc:
           ProcessAuto2d<B,Arc>(corr, field, dots);
           break;
      case Periodic:
           ProcessAuto2d<B,Periodic>(corr, field, dots);
           break;
      default:
           Assert(false);
    }
}

template <int D, int C>
void ProcessAuto(Corr2<D,D>* corr, Field<D,C>* field, bool dots, BinType bin_type, Metric metric)
{
    switch(bin_type) {
      case Log:
           ProcessAuto2c<Log>(corr, field, dots, metric);
           break;
      case Linear:
           ProcessAuto2c<Linear>(corr, field, dots, metric);
           break;
      case TwoD:
           ProcessAuto2c<TwoD>(corr, field, dots, metric);
           break;
      default:
           Assert(false);
    }
}

template <int B, int M, int D1, int D2, int C>
void ProcessCross2d(Corr2<D1,D2>* corr, Field<D1,C>* field1, Field<D2,C>* field2, bool dots)
{
    const bool P = corr->nontrivialRPar();
    dbg<<"ProcessCross: coords = "<<C<<", metric = "<<M<<", P = "<<P<<std::endl;

    Assert((ValidMC<M,C>::_M == M));
    if (P) {
        Assert(C == ThreeD);
        corr->template process<B, ValidMC<M,C>::_M, C==ThreeD>(*field1, *field2, dots);
    } else {
        corr->template process<B, ValidMC<M,C>::_M, false>(*field1, *field2, dots);
    }
}

template <int B, int D1, int D2, int C>
void ProcessCross2c(Corr2<D1,D2>* corr, Field<D1,C>* field1, Field<D2,C>* field2,
                    bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCross2d<B,Euclidean>(corr, field1, field2, dots);
           break;
      case Rperp:
           ProcessCross2d<B,Rperp>(corr, field1, field2, dots);
           break;
      case OldRperp:
           ProcessCross2d<B,OldRperp>(corr, field1, field2, dots);
           break;
      case Rlens:
           ProcessCross2d<B,Rlens>(corr, field1, field2, dots);
           break;
      case Arc:
           ProcessCross2d<B,Arc>(corr, field1, field2, dots);
           break;
      case Periodic:
           ProcessCross2d<B,Periodic>(corr, field1, field2, dots);
           break;
      default:
           Assert(false);
    }
}

template <int D1, int D2, int C>
void ProcessCross(Corr2<D1,D2>* corr, Field<D1,C>* field1, Field<D2,C>* field2,
                  bool dots, BinType bin_type, Metric metric)
{
    dbg<<"Start ProcessCross: "<<D1<<" "<<D2<<" "<<bin_type<<" "<<metric<<std::endl;
    switch(bin_type) {
      case Log:
           ProcessCross2c<Log>(corr, field1, field2, dots, metric);
           break;
      case Linear:
           ProcessCross2c<Linear>(corr, field1, field2, dots, metric);
           break;
      case TwoD:
           ProcessCross2c<TwoD>(corr, field1, field2, dots, metric);
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

int GetOMPThreads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <int B, int M, int C>
long SamplePairs2d(BaseCorr2* corr, BaseField<C>* field1, BaseField<C>* field2,
                   double minsep, double maxsep,
                   long* i1, long* i2, double* sep, int n)
{
    const bool P = corr->nontrivialRPar();
    dbg<<"SamplePairs: coords = "<<C<<", metric = "<<M<<", P = "<<P<<std::endl;

    Assert((ValidMC<M,C>::_M == M));
    if (P) {
        Assert(C == ThreeD);
        return corr->template samplePairs<B, ValidMC<M,C>::_M, C==ThreeD>(
            *field1, *field2, minsep, maxsep, i1, i2, sep, n);
    } else {
        return corr->template samplePairs<B, ValidMC<M,C>::_M, false>(
            *field1, *field2, minsep, maxsep, i1, i2, sep, n);
    }
}

template <int B, int C>
long SamplePairs2c(BaseCorr2* corr, BaseField<C>* field1, BaseField<C>* field2,
                   double minsep, double maxsep, Metric metric,
                   long* i1, long* i2, double* sep, int n)
{
    switch(metric) {
      case Euclidean:
           return SamplePairs2d<B,Euclidean>(corr, field1, field2, minsep, maxsep, i1, i2, sep, n);
           break;
      case Rperp:
           return SamplePairs2d<B,Rperp>(corr, field1, field2, minsep, maxsep, i1, i2, sep, n);
           break;
      case OldRperp:
           return SamplePairs2d<B,OldRperp>(corr, field1, field2, minsep, maxsep, i1, i2, sep, n);
           break;
      case Rlens:
           return SamplePairs2d<B,Rlens>(corr, field1, field2, minsep, maxsep, i1, i2, sep, n);
           break;
      case Arc:
           return SamplePairs2d<B,Arc>(corr, field1, field2, minsep, maxsep, i1, i2, sep, n);
           break;
      case Periodic:
           return SamplePairs2d<B,Periodic>(corr, field1, field2, minsep, maxsep, i1, i2, sep, n);
           break;
      default:
           Assert(false);
    }
    return 0;
}

template <int C>
long SamplePairs(BaseCorr2* corr, BaseField<C>* field1, BaseField<C>* field2,
                 double minsep, double maxsep, BinType bin_type, Metric metric,
                 py::array_t<long>& i1p, py::array_t<long>& i2p, py::array_t<double>& sepp)
{
    long n = i1p.size();
    Assert(i2p.size() == n);
    Assert(sepp.size() == n);

    long* i1 = static_cast<long*>(i1p.mutable_data());
    long* i2 = static_cast<long*>(i2p.mutable_data());
    double* sep = static_cast<double*>(sepp.mutable_data());

    dbg<<"Start SamplePairs: "<<bin_type<<" "<<metric<<std::endl;

    switch(bin_type) {
      case Log:
           return SamplePairs2c<Log>(corr, field1, field2, minsep, maxsep,
                                     metric, i1, i2, sep, n);
           break;
      case Linear:
           return SamplePairs2c<Linear>(corr, field1, field2, minsep, maxsep,
                                        metric, i1, i2, sep, n);
           break;
      case TwoD:
           // TwoD not implemented.
           break;
      default:
           Assert(false);
    }
    return 0;
}

template <int B, int M, int C>
int TriviallyZero2e(BaseCorr2* corr,
                    double x1, double y1, double z1, double s1,
                    double x2, double y2, double z2, double s2)
{
    Position<C> p1(x1,y1,z1);
    Position<C> p2(x2,y2,z2);
    return corr->template triviallyZero<B,M>(p1, p2, s1, s2);
}

template <int B, int M>
int TriviallyZero2d(BaseCorr2* corr, Coord coords,
                    double x1, double y1, double z1, double s1,
                    double x2, double y2, double z2, double s2)
{
    switch(coords) {
      case Flat:
           Assert((MetricHelper<M,0>::_Flat == int(Flat)));
           return TriviallyZero2e<B,M,MetricHelper<M,0>::_Flat>(
               corr, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Sphere:
           Assert((MetricHelper<M,0>::_Sphere == int(Sphere)));
           return TriviallyZero2e<B,M,MetricHelper<M,0>::_Sphere>(
               corr, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case ThreeD:
           Assert((MetricHelper<M,0>::_ThreeD == int(ThreeD)));
           return TriviallyZero2e<B,M,MetricHelper<M,0>::_ThreeD>(
               corr, x1, y1, z1, s1, x2, y2, z2, s2);
      default:
           Assert(false);
    }
    return 0;
}

template <int B>
int TriviallyZero2c(BaseCorr2* corr, Metric metric, Coord coords,
                    double x1, double y1, double z1, double s1,
                    double x2, double y2, double z2, double s2)
{
    switch(metric) {
      case Euclidean:
           return TriviallyZero2d<B,Euclidean>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Rperp:
           return TriviallyZero2d<B,Rperp>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case OldRperp:
           return TriviallyZero2d<B,OldRperp>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Rlens:
           return TriviallyZero2d<B,Rlens>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Arc:
           return TriviallyZero2d<B,Arc>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Periodic:
           return TriviallyZero2d<B,Periodic>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      default:
           Assert(false);
    }
    return 0;
}

int TriviallyZero(BaseCorr2* corr, BinType bin_type, Metric metric, Coord coords,
                  double x1, double y1, double z1, double s1,
                  double x2, double y2, double z2, double s2)
{
    switch(bin_type) {
      case Log:
           return TriviallyZero2c<Log>(corr, metric, coords,
                                       x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Linear:
           return TriviallyZero2c<Linear>(corr, metric, coords,
                                          x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case TwoD:
           return TriviallyZero2c<TwoD>(corr, metric, coords,
                                        x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      default:
           Assert(false);
    }
    return 0;
}

// Export the above functions using pybind11

template <int D1, int D2, int C>
struct WrapAuto
{
    template <typename W>
    static void run(W& corr2) {}
};

template <int D, int C>
struct WrapAuto<D,D,C>
{
    template <typename W>
    static void run(W& corr2)
    {
        typedef void (*auto_type)(Corr2<D,D>* corr, Field<D,C>* field,
                                  bool dots, BinType bin_type, Metric metric);
        corr2.def("processAuto", auto_type(&ProcessAuto));
    }
};

template <int D1, int D2, int C, typename W1, typename W2>
void WrapCross(py::module& _treecorr, W1& corr2, W2& base_corr2)
{
    typedef void (*cross_type)(Corr2<D1,D2>* corr,
                               Field<D1,C>* field1, Field<D2,C>* field2,
                               bool dots, BinType bin_type, Metric metric);

    corr2.def("processCross", cross_type(&ProcessCross));
    WrapAuto<D1,D2,C>::run(corr2);
}

template <int D1, int D2, typename W>
void WrapCorr2(py::module& _treecorr, std::string prefix, W& base_corr2)
{
    typedef Corr2<D1,D2>* (*init_type)(
        BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
        double minrpar, double maxrpar, double xp, double yp, double zp,
        py::array_t<double>& xi0p, py::array_t<double>& xi1p,
        py::array_t<double>& xi2p, py::array_t<double>& xi3p,
        py::array_t<double>& meanrp, py::array_t<double>& meanlogrp,
        py::array_t<double>& weightp, py::array_t<double>& npairsp);

    py::class_<Corr2<D1,D2>, BaseCorr2> corr2(_treecorr, (prefix + "Corr").c_str());
    corr2.def(py::init(init_type(&BuildCorr2)));

    WrapCross<D1,D2,Flat>(_treecorr, corr2, base_corr2);
    WrapCross<D1,D2,Sphere>(_treecorr, corr2, base_corr2);
    WrapCross<D1,D2,ThreeD>(_treecorr, corr2, base_corr2);
}

template <int C, typename W>
void WrapSample(py::module& _treecorr, W& base_corr2)
{
    typedef long (*sample_type)(BaseCorr2* corr,
                                BaseField<C>* field1, BaseField<C>* field2,
                                double minsep, double maxsep,
                                BinType bin_type, Metric metric,
                                py::array_t<long>& i1p, py::array_t<long>& i2p,
                                py::array_t<double>& sepp);

    base_corr2.def("samplePairs", sample_type(&SamplePairs));
}

void pyExportCorr2(py::module& _treecorr)
{
    py::class_<BaseCorr2> base_corr2(_treecorr, "BaseCorr2");
    base_corr2.def("triviallyZero", &TriviallyZero);

    WrapSample<Flat>(_treecorr, base_corr2);
    WrapSample<Sphere>(_treecorr, base_corr2);
    WrapSample<ThreeD>(_treecorr, base_corr2);

    WrapCorr2<NData,NData>(_treecorr, "NN", base_corr2);
    WrapCorr2<NData,KData>(_treecorr, "NK", base_corr2);
    WrapCorr2<NData,GData>(_treecorr, "NG", base_corr2);
    WrapCorr2<KData,KData>(_treecorr, "KK", base_corr2);
    WrapCorr2<KData,GData>(_treecorr, "KG", base_corr2);
    WrapCorr2<GData,GData>(_treecorr, "GG", base_corr2);

    _treecorr.def("SetOMPThreads", &SetOMPThreads);
    _treecorr.def("GetOMPThreads", &GetOMPThreads);

    // Also wrap all the enums we want to have in the Python layer.
    py::enum_<BinType>(_treecorr, "BinType")
        .value("Log", Log)
        .value("Linear", Linear)
        .value("TwoD", TwoD)
        .export_values();

    py::enum_<Coord>(_treecorr, "Coord")
        .value("Flat", Flat)
        .value("Sphere", Sphere)
        .value("ThreeD", ThreeD)
        .export_values();

    py::enum_<Metric>(_treecorr, "Metric")
        .value("Euclidean", Euclidean)
        .value("Rperp", Rperp)
        .value("Rlens", Rlens)
        .value("Arc", Arc)
        .value("OldRperp", OldRperp)
        .value("Periodic", Periodic)
        .export_values();

    py::enum_<SplitMethod>(_treecorr, "SplitMethod")
        .value("Middle", Middle)
        .value("Median", Median)
        .value("Mean", Mean)
        .value("Random", Random)
        .export_values();
}

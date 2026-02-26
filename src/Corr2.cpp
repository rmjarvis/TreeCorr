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

// Uncomment this to enable xassert, usually more time-consuming assert statements
// Also to turn on dbg<< messages.
//#define DEBUGLOGGING

#include "PyBind11Helper.h"

#include <vector>
#include <unordered_set>

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
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
    double b, double a,
    double minrpar, double maxrpar, double xp, double yp, double zp) :
    _bin_type(bin_type),
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b), _a(a),
    _minrpar(minrpar), _maxrpar(maxrpar), _xp(xp), _yp(yp), _zp(zp), _coords(-1)
{
    dbg<<"Corr2 constructor\n";
    xdbg<<bin_type<<std::endl;
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _bsq = _b * _b;
    _asq = _a * _a;
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
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
    double b, double a,
    double minrpar, double maxrpar, double xp, double yp, double zp,
    double* xi0, double* xi1, double* xi2, double* xi3,
    double* meanr, double* meanlogr, double* weight, double* npairs) :
    BaseCorr2(bin_type, minsep, maxsep, nbins, binsize, b, a, minrpar, maxrpar, xp, yp, zp),
    _owns_data(false),
    _xi(xi0,xi1,xi2,xi3), _meanr(meanr), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
{}

BaseCorr2::BaseCorr2(const BaseCorr2& rhs) :
    _bin_type(rhs._bin_type),
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minrpar(rhs._minrpar), _maxrpar(rhs._maxrpar),
    _xp(rhs._xp), _yp(rhs._yp), _zp(rhs._zp),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq), _bsq(rhs._bsq), _asq(rhs._asq),
    _fullmaxsep(rhs._fullmaxsep), _fullmaxsepsq(rhs._fullmaxsepsq),
    _coords(rhs._coords)
{}

template <int D1, int D2>
Corr2<D1,D2>::Corr2(const Corr2<D1,D2>& rhs, bool copy_data) :
    BaseCorr2(rhs), _owns_data(true), _xi(0,0,0,0), _weight(0)
{
    xdbg<<"Corr2 copy constructor\n";
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
    xdbg<<"Corr2 destructor\n";
    if (_owns_data) {
        _xi.delete_data(_nbins);
        delete [] _meanr; _meanr = 0;
        delete [] _meanlogr; _meanlogr = 0;
        delete [] _weight; _weight = 0;
        delete [] _npairs; _npairs = 0;
    }
}

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

template <int B, int M, int P, int C>
void BaseCorr2::process(const BaseField<C>& field, bool dots, bool quick)
{
    xdbg<<"Start process (auto): M,P,C = "<<M<<"  "<<P<<"  "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field.getNTopLevel();
    dbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);

    const std::vector<const BaseCell<C>*>& cells = field.getCells();

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr2> bc2p = duplicate();
        BaseCorr2& bc2 = *bc2p;
#else
        BaseCorr2& bc2 = *this;
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
            const BaseCell<C>& c1 = *cells[i];
            bc2.template process2<B,M,P>(c1, metric, quick);
            for (long j=i+1;j<n1;++j) {
                const BaseCell<C>& c2 = *cells[j];
                const int R = BinTypeHelper<B>::do_reverse;
                if (quick) {
                    bc2.process11<B,M,P,true,R>(c1, c2, metric);
                } else {
                    bc2.process11<B,M,P,false,R>(c1, c2, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(bc2);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int P, int C>
void BaseCorr2::process(const BaseField<C>& field1, const BaseField<C>& field2,
                        bool dots, bool quick)
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

    const std::vector<const BaseCell<C>*>& c1list = field1.getCells();
    const std::vector<const BaseCell<C>*>& c2list = field2.getCells();

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr2> bc2p = duplicate();
        BaseCorr2& bc2 = *bc2p;
#else
        BaseCorr2& bc2 = *this;
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
            const BaseCell<C>& c1 = *c1list[i];
            for (long j=0;j<n2;++j) {
                const BaseCell<C>& c2 = *c2list[j];
                // R is always false here.
                if (quick) {
                    bc2.process11<B,M,P,true,false>(c1, c2, metric);
                } else {
                    bc2.process11<B,M,P,false,false>(c1, c2, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(bc2);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int P, int C>
void BaseCorr2::process2(const BaseCell<C>& c12, const MetricHelper<M,P>& metric, bool quick)
{
    if (c12.getW() == 0.) return;
    if (c12.getSize() <= _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2<B,M,P>(*c12.getLeft(), metric, quick);
    process2<B,M,P>(*c12.getRight(), metric, quick);
    const int R = BinTypeHelper<B>::do_reverse;
    if (quick) {
        process11<B,M,P,true,R>(*c12.getLeft(), *c12.getRight(), metric);
    } else {
        process11<B,M,P,false,R>(*c12.getLeft(), *c12.getRight(), metric);
    }
}

template <int B, int M, int P, int Q, int R, int C>
void BaseCorr2::process11(const BaseCell<C>& c1, const BaseCell<C>& c2,
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
        BinTypeHelper<B>::singleBin(rsq, s1ps2, p1, p2, _binsize, _b, _bsq, _a, _asq,
                                    _minsep, _maxsep, _logminsep, k, r, logr))
    {
        xdbg<<"Drop into single bin.\n";
        if (BinTypeHelper<B>::isRSqInRange(rsq, p1, p2, _minsep, _minsepsq, _maxsep, _maxsepsq)) {
            directProcess11<B,Q,R>(c1,c2,rsq,k,r,logr);
        }
    } else {
        xdbg<<"Need to split.\n";
        bool split1=false, split2=false;
        double bsq_eff = BinTypeHelper<B>::getEffectiveBSq(rsq,_bsq,_asq);
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
            process11<B,M,P,Q,R>(*c1.getLeft(),*c2.getLeft(),metric);
            process11<B,M,P,Q,R>(*c1.getLeft(),*c2.getRight(),metric);
            process11<B,M,P,Q,R>(*c1.getRight(),*c2.getLeft(),metric);
            process11<B,M,P,Q,R>(*c1.getRight(),*c2.getRight(),metric);
        } else if (split1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11<B,M,P,Q,R>(*c1.getLeft(),c2,metric);
            process11<B,M,P,Q,R>(*c1.getRight(),c2,metric);
        } else {
            Assert(split2);
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<B,M,P,Q,R>(c1,*c2.getLeft(),metric);
            process11<B,M,P,Q,R>(c1,*c2.getRight(),metric);
        }
    }
}

// There are a smallish number of algorithms, each of which still may use D1 and/or D2
// for projections, but are otherwise the same for mutliple D combinations.
// So separate them out into a separate structure.
template <int algo>
struct DirectHelper;

// NN
template <>
struct DirectHelper<0>
{
    template <int R, int C>
    static void ProcessXi(
        const Cell<NData,C>& , const Cell<NData,C>& , const double ,
        XiData<NData,NData>& , int, int )
    {}
};

// both real
template <>
struct DirectHelper<1>
{
    template <int R, int D1, int D2, int C>
    static void ProcessXi(
        const Cell<D1,C>& c1, const Cell<D2,C>& c2, const double ,
        XiData<D1,D2>& xi, int k, int k2)
    { 
        double wkk = getWK(c1) * getWK(c2);
        xi.xi[k] += wkk;
        if (R) xi.xi[k2] += wkk;
    }
};

// real, complex
template <>
struct DirectHelper<2>
{
    template <int R, int D1, int D2, int C>
    static void ProcessXi(
        const Cell<D1,C>& c1, const Cell<D2,C>& c2, const double rsq,
        XiData<D1,D2>& xi, int k, int )
    {
        std::complex<double> g2 = c2.getWZ();
        ProjectHelper<C>::Project(c1,c2,g2);
        // For GData only, we multiply by -1, because the standard thing to accumulate is
        // the tangential shear, rather than radial.  Everyone else accumulates the radial
        // value (which is what Project returns).
        if (D2 == GData) g2 *= -getWK(c1);
        else g2 *= getWK(c1);
        xi.xi[k] += g2.real();
        xi.xi_im[k] += g2.imag();
    }
};

// both complex
template <>
struct DirectHelper<3>
{
    template <int R, int D1, int D2, int C>
    static void ProcessXi(
        const Cell<D1,C>& c1, const Cell<D2,C>& c2, const double rsq,
        XiData<D1,D2>& xi, int k, int k2)
    {
        std::complex<double> g1 = c1.getWZ();
        std::complex<double> g2 = c2.getWZ();
        ProjectHelper<C>::Project(c1,c2,g1,g2);

        // The complex products g1 g2 and g1 g2* share most of the calculations,
        // so faster to do this manually, which we do in both_complex_prod.
        std::complex<double> xip, xim;
        tie(xim, xip) = both_complex_prod(g2, g1);

        xi.xip[k] += xip.real();
        xi.xip_im[k] += xip.imag();
        xi.xim[k] += xim.real();
        xi.xim_im[k] += xim.imag();

        if (R) {
            xi.xip[k2] += xip.real();
            xi.xip_im[k2] += xip.imag();
            xi.xim[k2] += xim.real();
            xi.xim_im[k2] += xim.imag();
        }
    }
};

template <int B, int Q, int R, int C>
void BaseCorr2::directProcess11(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const double rsq,
    int k, double r, double logr)
{
    xdbg<<"DirectProcess11: rsq = "<<rsq<<"  r = "<<sqrt(rsq)<<std::endl;
    xdbg<<"p1 = "<<c1.getPos()<<"  p2 = "<<c2.getPos()<<std::endl;
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

    int k2 = -1;
    if (R) {
        k2 = BinTypeHelper<B>::calculateBinK(p2, p1, r, logr, _binsize,
                                             _minsep, _maxsep, _logminsep);
        if (k2 == _nbins) --k2;  // As before, this can (rarely) happen.
    }

    finishProcess<Q,R>(c1, c2, rsq, r, logr, k, k2);
}

template <int D1, int D2> template <int Q, int R, int C>
void Corr2<D1,D2>::finishProcess(const BaseCell<C>& c1, const BaseCell<C>& c2,
                                 double rsq, double r, double logr, int k, int k2)
{
    double ww = double(c1.getW()) * double(c2.getW());
    _weight[k] += ww;
    if (R) _weight[k2] += ww;

    if (!Q) {
        double nn = double(c1.getN()) * double(c2.getN());
        _npairs[k] += nn;

        double wwr = ww * r;
        double wwlogr = ww * logr;
        _meanr[k] += wwr;
        _meanlogr[k] += wwlogr;
        xdbg<<"n,w = "<<nn<<','<<ww<<" ==>  "<<_npairs[k]<<','<<_weight[k]<<std::endl;

        if (R) {
            Assert(k2 >= 0);
            Assert(k2 < _nbins);
            _npairs[k2] += nn;
            _meanr[k2] += wwr;
            _meanlogr[k2] += wwlogr;
        }
    }

    const int algo =
        (D1 == NData && D2 == NData) ? 0 :      // NN (nothing to do)
        (D1 <= KData && D2 <= KData) ? 1 :      // NK, KN, KK
        (D1 <= KData && D2 >= ZData) ? 2 :      // NZ, KZ
        (D1 >= ZData && D2 >= ZData) ? 3 : -1;  // ZZ

    // Note: R=1 is only possible if D1 == D2, so multiply by (D1==D2)
    // to force the others to only instantiate R=0.
    DirectHelper<algo>::template ProcessXi<R*(D1==D2)>(
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
    return (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq) &&
            metric.tooLargeDist(p1, p2, rsq, 0, s1ps2, _fullmaxsep, _fullmaxsepsq));
}

Sampler::Sampler(const BaseCorr2& base_corr2, double minsep, double maxsep,
                 long* i1, long* i2, double* sep, int n) :
    BaseCorr2(base_corr2), _i1(i1), _i2(i2), _sep(sep), _n(n), _k(0)
{
    dbg<<"Sampler constructor\n";
    xdbg<<"Initial minsep/maxsep = "<<_minsep<<", "<<_maxsep<<std::endl;
    // Update the minsep/maxsep values.
    _minsep = minsep;
    _maxsep = maxsep;
    xdbg<<"New minsep/maxsep = "<<_minsep<<", "<<_maxsep<<std::endl;
    // And recompute the relevant derived quantities.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _fullmaxsep = CalculateFullMaxSep(base_corr2.getBinType(), _minsep, _maxsep, _nbins, _binsize);
    _fullmaxsepsq = _fullmaxsep*_fullmaxsep;
}

void SelectRandomFrom(long m, std::vector<long>& selection)
{
    xdbg<<"SelectRandomFrom("<<m<<", "<<selection.size()<<")\n";
    long n = long(selection.size());
    // There are two algorithms here.
    // Floyd's algorithm is efficient for m >> n.
    // Fisher-Yates is efficient for m not much > n.
    // I don't know exactly the transition for when Floyd's becomes the better choice.
    // So I'm somewhat arbitrarily picking 3*n.
    if (m > 3*n) {
        dbg<<"Floyd's algorithm\n";
        // Floyd's algorithm
        // O(n) in memory, but slightly more than O(n) in time, especially as m ~ n.
        std::unordered_set<long> selected;
        selected.reserve(n);
        while (long(selected.size()) < n) {
            double urd = urand();
            long j = long(urd * m); // 0 <= j < m
            if (j == m) j = m-1;    // Just in case.
            std::pair<std::unordered_set<long>::iterator,bool> ret = selected.insert(j);
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

template <int Q, int R, int C>
void Sampler::finishProcess(
    const BaseCell<C>& c1, const BaseCell<C>& c2,
    double rsq, double r, double logr, int kk, int kk2)
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
    xdbg<<"   n1,n2,k,n,m = "<<n1<<','<<n2<<','<<_k<<','<<_n<<','<<m<<std::endl;

    if (_k + m <= _n) {
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
                        _i1[_k] = index1;
                        _i2[_k] = index2;
                        _sep[_k] = r;
                        ++_k;
                    }
                }
            }
        }
    } else if (m <= _n) {
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
                        long j = _k;  // j is where in the lists we will place this
                        if (_k >= _n) {
                            double urd = urand(); // 0 < urd < 1
                            j = int(urd * (_k+1)); // 0 <= j < k+1
                        }
                        if (j < _n)  {
                            _i1[j] = index1;
                            _i2[j] = index2;
                            _sep[j] = r;
                        }
                        ++_k;
                    }
                }
            }
        }
    } else {
        // Case 3
        xdbg<<"Case 3: Select n without replacement\n";
        std::vector<long> selection(_n);
        SelectRandomFrom(_k+m, selection);
        // If any items in k<=i<n are from the original set, put them in their original place.
        for(long i=_k;i<_n;++i) {
            long j = selection[i];
            if (j < _n) std::swap(selection[i], selection[j]);
        }

        // Now any items with j >= k can replace the value at their i in this list.
        std::vector<std::pair<long, long> > places;
        places.reserve(_n);
        for(long i=0;i<_n;++i) {
            long j = selection[i];
            if (j >= _k) places.emplace_back(j, i);
        }
        if (places.empty()) {
            // If nothing selected from the new set, then we're done.
            _k += m;
            return;
        }
        std::sort(places.begin(), places.end());

        size_t next = 0;

        long i=_k;  // When i is in the map, place it at places[i]
        for (size_t p1=0; p1<leaf1.size(); ++p1) {
            long nn1 = leaf1[p1]->getN();
            for (long q1=0; q1<nn1; ++q1) {
                xdbg<<"i = "<<i<<", next = "<<places[next].first<<"  "<<places[next].second<<std::endl;
                Assert(i <= places[next].first);
                if (places[next].first > i + n2) {
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
                        if (i == places[next].first) {
                            xdbg<<"Use i = "<<i<<std::endl;
                            long index2;
                            if (nn2 == 1) index2 = leaf2[p2]->getInfo().index;
                            else index2 = (*leaf2[p2]->getListInfo().indices)[q2];
                            long j = places[next].second;
                            _i1[j] = index1;
                            _i2[j] = index2;
                            _sep[j] = r;
                            ++next;
                        }
                        if (next == places.size()) break;
                    }
                    if (next == places.size()) break;
                }
                if (next == places.size()) break;
            }
            if (next == places.size()) break;
        }
        xdbg<<"Done: i = "<<i<<", k+m = "<<_k+m<<std::endl;
        _k += m;
    }
}


//
//
// The functions we call from Python.
//
//

template <int D1, int D2>
Corr2<D1,D2>* BuildCorr2(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
    double b, double a,
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
            bin_type, minsep, maxsep, nbins, binsize, b, a, minrpar, maxrpar, xp, yp, zp,
            xi0, xi1, xi2, xi3, meanr, meanlogr, weight, npairs);
}

template <int B, int M, int C>
void ProcessAuto2(BaseCorr2& corr, BaseField<C>& field, bool dots, bool quick)
{
    const bool P = corr.nontrivialRPar();
    dbg<<"ProcessAuto: coords = "<<C<<", metric = "<<M<<", P = "<<P<<std::endl;

    // Only call the actual function if the M/C combination is valid.
    // The Assert checks that this only gets called when the combination is actually valid.
    // But other combinations are nominally instantiated, and give various compiler errors
    // when they are tried.  So this _M business make sure such combinations aren't instantiated.
    // (Plus they would add code that is never used, leading to code bloat.)
    Assert((ValidMC<M,C>::_M == M));
    if (P) {
        Assert(C == ThreeD);
        corr.template process<B, ValidMC<M,C>::_M, (C==ThreeD)>(field, dots, quick);
    } else {
        corr.template process<B, ValidMC<M,C>::_M, false>(field, dots, quick);
    }
}

template <int B, int C>
void ProcessAuto1(BaseCorr2& corr, BaseField<C>& field, bool dots, bool quick, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessAuto2<B,Euclidean>(corr, field, dots, quick);
           break;
      case Rperp:
           ProcessAuto2<B,Rperp>(corr, field, dots, quick);
           break;
      case OldRperp:
           ProcessAuto2<B,OldRperp>(corr, field, dots, quick);
           break;
      case Rlens:
           ProcessAuto2<B,Rlens>(corr, field, dots, quick);
           break;
      case Arc:
           ProcessAuto2<B,Arc>(corr, field, dots, quick);
           break;
      case Periodic:
           ProcessAuto2<B,Periodic>(corr, field, dots, quick);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessAuto(BaseCorr2& corr, BaseField<C>& field, bool dots, bool quick, Metric metric)
{
    switch(corr.getBinType()) {
      case Log:
           ProcessAuto1<Log>(corr, field, dots, quick, metric);
           break;
      case Linear:
           ProcessAuto1<Linear>(corr, field, dots, quick, metric);
           break;
      case TwoD:
           ProcessAuto1<TwoD>(corr, field, dots, quick, metric);
           break;
      default:
           Assert(false);
    }
}

template <int B, int M, int C>
void ProcessCross2(BaseCorr2& corr, BaseField<C>& field1, BaseField<C>& field2,
                   bool dots, bool quick)
{
    const bool P = corr.nontrivialRPar();
    dbg<<"ProcessCross: coords = "<<C<<", metric = "<<M<<", P = "<<P<<std::endl;

    Assert((ValidMC<M,C>::_M == M));
    if (P) {
        Assert(C == ThreeD);
        corr.template process<B, ValidMC<M,C>::_M, C==ThreeD>(field1, field2, dots, quick);
    } else {
        corr.template process<B, ValidMC<M,C>::_M, false>(field1, field2, dots, quick);
    }
}

template <int B, int C>
void ProcessCross1(BaseCorr2& corr, BaseField<C>& field1, BaseField<C>& field2,
                   bool dots, bool quick, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCross2<B,Euclidean>(corr, field1, field2, dots, quick);
           break;
      case Rperp:
           ProcessCross2<B,Rperp>(corr, field1, field2, dots, quick);
           break;
      case OldRperp:
           ProcessCross2<B,OldRperp>(corr, field1, field2, dots, quick);
           break;
      case Rlens:
           ProcessCross2<B,Rlens>(corr, field1, field2, dots, quick);
           break;
      case Arc:
           ProcessCross2<B,Arc>(corr, field1, field2, dots, quick);
           break;
      case Periodic:
           ProcessCross2<B,Periodic>(corr, field1, field2, dots, quick);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessCross(BaseCorr2& corr, BaseField<C>& field1, BaseField<C>& field2,
                  bool dots, bool quick, Metric metric)
{
    dbg<<"Start ProcessCross: "<<corr.getBinType()<<" "<<metric<<std::endl;
    switch(corr.getBinType()) {
      case Log:
           ProcessCross1<Log>(corr, field1, field2, dots, quick, metric);
           break;
      case Linear:
           ProcessCross1<Linear>(corr, field1, field2, dots, quick, metric);
           break;
      case TwoD:
           ProcessCross1<TwoD>(corr, field1, field2, dots, quick, metric);
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

template <int C>
long SamplePairs(BaseCorr2& corr, BaseField<C>& field1, BaseField<C>& field2,
                 double minsep, double maxsep, Metric metric, long long seed,
                 py::array_t<long>& i1p, py::array_t<long>& i2p, py::array_t<double>& sepp)
{
    long n = long(i1p.size());
    Assert(i2p.size() == n);
    Assert(sepp.size() == n);

    urand(seed);  // Make sure rand is seeded properly.

    long* i1 = static_cast<long*>(i1p.mutable_data());
    long* i2 = static_cast<long*>(i2p.mutable_data());
    double* sep = static_cast<double*>(sepp.mutable_data());

    dbg<<"Start SamplePairs: "<<corr.getBinType()<<" "<<metric<<std::endl;

    Sampler sampler(corr, minsep, maxsep, i1, i2, sep, n);

    // I don't know how to do the sampling safely in parallel, so temporarily set num_threads=1.
    int old_num_threads = SetOMPThreads(1);
    ProcessCross(sampler, field1, field2, false, false, metric);
    SetOMPThreads(old_num_threads);

    return sampler.getK();
}

template <int B, int M, int C>
int TriviallyZero3(BaseCorr2& corr,
                   double x1, double y1, double z1, double s1,
                   double x2, double y2, double z2, double s2)
{
    Position<C> p1(x1,y1,z1);
    Position<C> p2(x2,y2,z2);
    return corr.template triviallyZero<B,M>(p1, p2, s1, s2);
}

template <int B, int M>
int TriviallyZero2(BaseCorr2& corr, Coord coords,
                   double x1, double y1, double z1, double s1,
                   double x2, double y2, double z2, double s2)
{
    switch(coords) {
      case Flat:
           Assert((MetricHelper<M,0>::_Flat == int(Flat)));
           return TriviallyZero3<B,M,MetricHelper<M,0>::_Flat>(
               corr, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Sphere:
           Assert((MetricHelper<M,0>::_Sphere == int(Sphere)));
           return TriviallyZero3<B,M,MetricHelper<M,0>::_Sphere>(
               corr, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case ThreeD:
           Assert((MetricHelper<M,0>::_ThreeD == int(ThreeD)));
           return TriviallyZero3<B,M,MetricHelper<M,0>::_ThreeD>(
               corr, x1, y1, z1, s1, x2, y2, z2, s2);
      default:
           Assert(false);
    }
    return 0;
}

template <int B>
int TriviallyZero1(BaseCorr2& corr, Metric metric, Coord coords,
                   double x1, double y1, double z1, double s1,
                   double x2, double y2, double z2, double s2)
{
    switch(metric) {
      case Euclidean:
           return TriviallyZero2<B,Euclidean>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Rperp:
           return TriviallyZero2<B,Rperp>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case OldRperp:
           return TriviallyZero2<B,OldRperp>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Rlens:
           return TriviallyZero2<B,Rlens>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Arc:
           return TriviallyZero2<B,Arc>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Periodic:
           return TriviallyZero2<B,Periodic>(corr, coords, x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      default:
           Assert(false);
    }
    return 0;
}

int TriviallyZero(BaseCorr2& corr, Metric metric, Coord coords,
                  double x1, double y1, double z1, double s1,
                  double x2, double y2, double z2, double s2)
{
    switch(corr.getBinType()) {
      case Log:
           return TriviallyZero1<Log>(corr, metric, coords,
                                      x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case Linear:
           return TriviallyZero1<Linear>(corr, metric, coords,
                                         x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      case TwoD:
           return TriviallyZero1<TwoD>(corr, metric, coords,
                                       x1, y1, z1, s1, x2, y2, z2, s2);
           break;
      default:
           Assert(false);
    }
    return 0;
}

// Export the above functions using pybind11

template <int D1, int D2>
void WrapCorr2(py::module& _treecorr, std::string prefix)
{
    typedef Corr2<D1,D2>* (*init_type)(
        BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
        double b, double a,
        double minrpar, double maxrpar, double xp, double yp, double zp,
        py::array_t<double>& xi0p, py::array_t<double>& xi1p,
        py::array_t<double>& xi2p, py::array_t<double>& xi3p,
        py::array_t<double>& meanrp, py::array_t<double>& meanlogrp,
        py::array_t<double>& weightp, py::array_t<double>& npairsp);

    py::class_<Corr2<D1,D2>, BaseCorr2> corr2(_treecorr, (prefix + "Corr").c_str());
    corr2.def(py::init(init_type(&BuildCorr2)));
}

template <int C, typename W>
void WrapProcess(py::module& _treecorr, W& base_corr2)
{
    typedef void (*auto_type)(BaseCorr2& corr, BaseField<C>& field,
                              bool dots, bool quick, Metric metric);
    base_corr2.def("processAuto", auto_type(&ProcessAuto));

    typedef void (*cross_type)(BaseCorr2& corr, BaseField<C>& field1, BaseField<C>& field2,
                               bool dots, bool quick, Metric metric);
    base_corr2.def("processCross", cross_type(&ProcessCross));

    typedef long (*sample_type)(BaseCorr2& corr,
                                BaseField<C>& field1, BaseField<C>& field2,
                                double minsep, double maxsep,
                                Metric metric, long long seed,
                                py::array_t<long>& i1p, py::array_t<long>& i2p,
                                py::array_t<double>& sepp);
    base_corr2.def("samplePairs", sample_type(&SamplePairs));
}

void pyExportCorr2(py::module& _treecorr)
{
    py::class_<BaseCorr2> base_corr2(_treecorr, "BaseCorr2");
    base_corr2.def("triviallyZero", &TriviallyZero);

    WrapProcess<Flat>(_treecorr, base_corr2);
    WrapProcess<Sphere>(_treecorr, base_corr2);
    WrapProcess<ThreeD>(_treecorr, base_corr2);

    WrapCorr2<NData,NData>(_treecorr, "NN");
    WrapCorr2<NData,KData>(_treecorr, "NK");
    WrapCorr2<KData,KData>(_treecorr, "KK");

    WrapCorr2<NData,ZData>(_treecorr, "NZ");
    WrapCorr2<KData,ZData>(_treecorr, "KZ");
    WrapCorr2<ZData,ZData>(_treecorr, "ZZ");

    WrapCorr2<NData,VData>(_treecorr, "NV");
    WrapCorr2<KData,VData>(_treecorr, "KV");
    WrapCorr2<VData,VData>(_treecorr, "VV");

    WrapCorr2<NData,GData>(_treecorr, "NG");
    WrapCorr2<KData,GData>(_treecorr, "KG");
    WrapCorr2<GData,GData>(_treecorr, "GG");

    WrapCorr2<NData,TData>(_treecorr, "NT");
    WrapCorr2<KData,TData>(_treecorr, "KT");
    WrapCorr2<TData,TData>(_treecorr, "TT");

    WrapCorr2<NData,QData>(_treecorr, "NQ");
    WrapCorr2<KData,QData>(_treecorr, "KQ");
    WrapCorr2<QData,QData>(_treecorr, "QQ");

    _treecorr.def("SetOMPThreads", &SetOMPThreads);
    _treecorr.def("GetOMPThreads", &GetOMPThreads);

    // Also wrap all the enums we want to have in the Python layer.
    py::enum_<BinType>(_treecorr, "BinType")
        .value("Log", Log)
        .value("Linear", Linear)
        .value("TwoD", TwoD)
        .value("LogRUV", LogRUV)
        .value("LogSAS", LogSAS)
        .value("LogMultipole", LogMultipole)
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

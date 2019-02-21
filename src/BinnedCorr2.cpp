/* Copyright (c) 2003-2015 by Mike Jarvis
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

#ifdef _OPENMP
#include "omp.h"
#endif

template <int D1, int D2, int B>
BinnedCorr2<D1,D2,B>::BinnedCorr2(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double minrpar, double maxrpar,
    double* xi0, double* xi1, double* xi2, double* xi3,
    double* meanr, double* meanlogr, double* weight, double* npairs) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minrpar(minrpar), _maxrpar(maxrpar),
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
}

template <int D1, int D2, int B>
BinnedCorr2<D1,D2,B>::BinnedCorr2(const BinnedCorr2<D1,D2,B>& rhs, bool copy_data) :
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minrpar(rhs._minrpar), _maxrpar(rhs._maxrpar),
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
    static void process2(BinnedCorr2<D1,D2,B>& , const Cell<D1,C>& ) {}
};

template <int D, int B, int C, int M>
struct ProcessHelper<D,D,B,C,M>
{
    static void process2(BinnedCorr2<D,D,B>& b, const Cell<D,C>& c12)
    { b.template process2<C,M>(c12); }
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
            ProcessHelper<D1,D2,B,C,M>::process2(bc2,c1);
            for (int j=i+1;j<n1;++j) {
                const Cell<D1,C>& c2 = *field.getCells()[j];
                bc2.process11<C,M>(c1,c2, BinTypeHelper<B>::doReverse());
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
                bc2.process11<C,M>(c1,c2,false);
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
            double s=0.;
            const double dsq = MetricHelper<M>::DistSq(c1.getPos(),c2.getPos(),s,s);
            if (BinTypeHelper<B>::isDSqInRange(dsq, c1.getPos(), c2.getPos(),
                                               _minsep, _minsepsq, _maxsep, _maxsepsq)) {
                bc2.template directProcess11<C,M>(c1,c2,dsq,false);
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
void BinnedCorr2<D1,D2,B>::process2(const Cell<D1,C>& c12)
{
    if (c12.getW() == 0.) return;
    if (c12.getSize() <= _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2<C,M>(*c12.getLeft());
    process2<C,M>(*c12.getRight());
    process11<C,M>(*c12.getLeft(),*c12.getRight(), BinTypeHelper<B>::doReverse());
}

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::process11(const Cell<D1,C>& c1, const Cell<D2,C>& c2, bool do_reverse)
{
    //set_verbose(2);
    xdbg<<"Start process11 for "<<c1.getPos()<<",  "<<c2.getPos()<<"   ";
    xdbg<<"w = "<<c1.getW()<<", "<<c2.getW()<<std::endl;
    if (c1.getW() == 0. || c2.getW() == 0.) return;

    double s1 = c1.getSize(); // May be modified by DistSq function.
    double s2 = c2.getSize(); // "
    xdbg<<"s1,s2 = "<<s1<<','<<s2<<std::endl;
    const double dsq = MetricHelper<M>::DistSq(c1.getPos(),c2.getPos(),s1,s2);
    xdbg<<"s1,s2 => "<<s1<<','<<s2<<std::endl;
    const double s1ps2 = s1+s2;

    double rpar = 0; // Gets set to correct value by this function if appropriate
    if (MetricHelper<M>::isRParOutsideRange(c1.getPos(), c2.getPos(), s1ps2, _minrpar, _maxrpar,
                                            rpar))
        return;
    xdbg<<"RPar in range\n";

    if (BinTypeHelper<B>::tooSmallDist(dsq, s1ps2, _minsep, _minsepsq) &&
        MetricHelper<M>::tooSmallDist(c1.getPos(), c2.getPos(), dsq, rpar, s1ps2, _minsepsq))
        return;
    xdbg<<"Not too small separation\n";

    if (BinTypeHelper<B>::tooLargeDist(dsq, s1ps2, _maxsep, _maxsepsq) &&
        MetricHelper<M>::tooLargeDist(c1.getPos(), c2.getPos(), dsq, rpar, s1ps2, _fullmaxsepsq))
        return;
    xdbg<<"Not too large separation\n";

    // Now check if these cells are small enough that it is ok to drop into a single bin.
    int k;
    double logr;  // If singleBin is true, these values are set for use by directProcess11
    if (MetricHelper<M>::isRParInsideRange(rpar, s1ps2, _minrpar, _maxrpar) &&
        BinTypeHelper<B>::singleBin(dsq, s1ps2, c1.getPos(), c2.getPos(),
                                    _binsize, _b, _bsq,
                                    _logminsep, _minsep, _maxsep,
                                    k, logr))
    {
        xdbg<<"Drop into single bin.\n";
        if (BinTypeHelper<B>::isDSqInRange(dsq, c1.getPos(), c2.getPos(),
                                           _minsep, _minsepsq, _maxsep, _maxsepsq)) {
            directProcess11<C,M>(c1,c2,dsq,do_reverse);
        }
    } else {
        xdbg<<"Need to split.\n";
        bool split1=false, split2=false;
        double bsq_eff = BinTypeHelper<B>::getEffectiveBSq(dsq,_bsq);
        CalcSplitSq(split1,split2,s1,s2,s1ps2,bsq_eff);
        xdbg<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<"  ";
        xdbg<<"s1ps2 / d = "<<s1ps2 / sqrt(dsq)<<", b = "<<_b<<"  ";
        xdbg<<"split = "<<split1<<','<<split2<<std::endl;

        if (split1 && split2) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<C,M>(*c1.getLeft(),*c2.getLeft(),do_reverse);
            process11<C,M>(*c1.getLeft(),*c2.getRight(),do_reverse);
            process11<C,M>(*c1.getRight(),*c2.getLeft(),do_reverse);
            process11<C,M>(*c1.getRight(),*c2.getRight(),do_reverse);
        } else if (split1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11<C,M>(*c1.getLeft(),c2,do_reverse);
            process11<C,M>(*c1.getRight(),c2,do_reverse);
        } else {
            Assert(split2);
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<C,M>(c1,*c2.getLeft(),do_reverse);
            process11<C,M>(c1,*c2.getRight(),do_reverse);
        }
    }
}


// We also set up a helper class for doing the direct processing
template <int D1, int D2>
struct DirectHelper;

template <>
struct DirectHelper<NData,NData>
{
    template <int C, int M>
    static void ProcessXi(
        const Cell<NData,C>& , const Cell<NData,C>& , const double ,
        XiData<NData,NData>& , int, int )
    {}
};

template <>
struct DirectHelper<NData,KData>
{
    template <int C, int M>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<KData,C>& c2, const double ,
        XiData<NData,KData>& xi, int k, int )
    { xi.xi[k] += c1.getW() * c2.getData().getWK(); }
};

template <>
struct DirectHelper<NData,GData>
{
    template <int C, int M>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<GData,C>& c2, const double dsq,
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
    template <int C, int M>
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
    template <int C, int M>
    static void ProcessXi(
        const Cell<KData,C>& c1, const Cell<GData,C>& c2, const double dsq,
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
    template <int C, int M>
    static void ProcessXi(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2, const double dsq,
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

template <int D1, int D2, int B> template <int C, int M>
void BinnedCorr2<D1,D2,B>::directProcess11(
    const Cell<D1,C>& c1, const Cell<D2,C>& c2, const double dsq, bool do_reverse,
    int k, double logr)
{
    xdbg<<"DirectProcess11: dsq = "<<dsq<<std::endl;
    XAssert(dsq >= _minsepsq);
    XAssert(dsq < _fullmaxsepsq);
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*_b + 0.0001);

    XAssert(_binsize != 0.);
    if (k < 0) {
        logr = 0.5 * log(dsq);
        Assert(logr >= _logminsep);
        k = BinTypeHelper<B>::calculateBinK(c1.getPos(), c2.getPos(),
                                            logr, _logminsep, _binsize,
                                            _minsep, _maxsep);
    } else {
        XAssert(std::abs(logr - 0.5*log(dsq)) < 1.e-10);
        XAssert(k == BinTypeHelper<B>::calculateBinK(c1.getPos(), c2.getPos(),
                                                     logr, _logminsep, _binsize,
                                                     _minsep, _maxsep));
    }
    Assert(k >= 0);
    Assert(k < _nbins);
    double r = sqrt(dsq);
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
        k2 = BinTypeHelper<B>::calculateBinK(c2.getPos(), c1.getPos(),
                                             logr, _logminsep, _binsize,
                                             _minsep, _maxsep);
        Assert(k2 >= 0);
        Assert(k2 < _nbins);
        _npairs[k2] += nn;
        _meanr[k2] += ww * r;
        _meanlogr[k2] += ww * logr;
        _weight[k2] += ww;
    }

    DirectHelper<D1,D2>::template ProcessXi<C,M>(c1,c2,dsq,_xi,k,k2);
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

//
//
// The C interface for python
//
//

extern "C" {
#include "BinnedCorr2_C.h"
}

void* BuildNNCorr(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNNCorr\n";
    void* corr=0;
    switch(bin_type) {
      case Log:
           corr = static_cast<void*>(new BinnedCorr2<NData,NData,Log>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   0, 0, 0, 0,
                   meanr, meanlogr, weight, npairs));
            break;
      case TwoD:
           corr = static_cast<void*>(new BinnedCorr2<NData,NData,TwoD>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   0, 0, 0, 0,
                   meanr, meanlogr, weight, npairs));
            break;
      default:
            Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNKCorr(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNKCorr\n";
    void* corr=0;
    switch(bin_type) {
      case Log:
           corr = static_cast<void*>(new BinnedCorr2<NData,KData,Log>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, 0, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      case TwoD:
           corr = static_cast<void*>(new BinnedCorr2<NData,KData,TwoD>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, 0, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      default:
           Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNGCorr(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi, double* xi_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNGCorr\n";
    void* corr=0;
    switch(bin_type) {
      case Log:
           corr = static_cast<void*>(new BinnedCorr2<NData,GData,Log>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, xi_im, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      case TwoD:
           corr = static_cast<void*>(new BinnedCorr2<NData,GData,TwoD>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, xi_im, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      default:
           Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKCorr(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKKCorr\n";
    void* corr=0;
    switch(bin_type) {
      case Log:
           corr = static_cast<void*>(new BinnedCorr2<KData,KData,Log>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, 0, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      case TwoD:
           corr = static_cast<void*>(new BinnedCorr2<KData,KData,TwoD>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, 0, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      default:
           Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKGCorr(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi, double* xi_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKGCorr\n";
    void* corr=0;
    switch(bin_type) {
      case Log:
           corr = static_cast<void*>(new BinnedCorr2<KData,GData,Log>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, xi_im, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      case TwoD:
           corr = static_cast<void*>(new BinnedCorr2<KData,GData,TwoD>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xi, xi_im, 0, 0,
                   meanr, meanlogr, weight, npairs));
           break;
      default:
           Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildGGCorr(int bin_type,
                  double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xip, double* xip_im, double* xim, double* xim_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildGGCorr\n";
    void* corr=0;
    switch(bin_type) {
      case Log:
           corr = static_cast<void*>(new BinnedCorr2<GData,GData,Log>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xip, xip_im, xim, xim_im,
                   meanr, meanlogr, weight, npairs));
           break;
      case TwoD:
           corr = static_cast<void*>(new BinnedCorr2<GData,GData,TwoD>(
                   minsep, maxsep, nbins, binsize, b,
                   minrpar, maxrpar,
                   xip, xip_im, xim, xim_im,
                   meanr, meanlogr, weight, npairs));
           break;
      default:
           Assert(false);
    }
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void DestroyNNCorr(void* corr, int bin_type)
{
    dbg<<"Start DestroyNNCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<NData,NData,Log>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<NData,NData,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}

void DestroyNKCorr(void* corr, int bin_type)
{
    dbg<<"Start DestroyNKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<NData,KData,Log>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<NData,KData,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}

void DestroyNGCorr(void* corr, int bin_type)
{
    dbg<<"Start DestroyNGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<NData,GData,Log>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<NData,GData,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}

void DestroyKKCorr(void* corr, int bin_type)
{
    dbg<<"Start DestroyKKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<KData,KData,Log>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<KData,KData,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}

void DestroyKGCorr(void* corr, int bin_type)
{
    dbg<<"Start DestroyKGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<KData,GData,Log>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<KData,GData,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}

void DestroyGGCorr(void* corr, int bin_type)
{
    dbg<<"Start DestroyGGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    switch(bin_type) {
      case Log:
           delete static_cast<BinnedCorr2<GData,GData,Log>*>(corr);
           break;
      case TwoD:
           delete static_cast<BinnedCorr2<GData,GData,TwoD>*>(corr);
           break;
      default:
           Assert(false);
    }
}


template <int D, int B>
void ProcessAuto2b(BinnedCorr2<D,D,B>& corr, void* field, int dots, int coord, int metric)
{
    if (coord == Flat) {
        switch(metric) {
          case Euclidean:
            corr.template process<Flat,Euclidean>(*static_cast<Field<D,Flat>*>(field),dots);
            break;
          default:
            Assert(false);
        }
    } else {
        switch(metric) {
          case Euclidean:
            corr.template process<ThreeD,Euclidean>(*static_cast<Field<D,ThreeD>*>(field),dots);
            break;
          case Perp:
            corr.template process<ThreeD,Perp>(*static_cast<Field<D,ThreeD>*>(field),dots);
            break;
          case Lens:
            corr.template process<ThreeD,Lens>(*static_cast<Field<D,ThreeD>*>(field),dots);
            break;
          case Arc:
            corr.template process<Sphere,Arc>(*static_cast<Field<D,Sphere>*>(field),dots);
            break;
          default:
            Assert(false);
        }
    }
}

template <int D>
void ProcessAuto2(void* corr, void* field, int dots, int coord, int bin_type, int metric)
{
    switch(bin_type) {
      case Log:
           ProcessAuto2b(*(static_cast<BinnedCorr2<D,D,Log>*>(corr)), field, dots, coord, metric);
           break;
      case TwoD:
           ProcessAuto2b(*(static_cast<BinnedCorr2<D,D,TwoD>*>(corr)), field, dots, coord, metric);
           break;
      default:
           Assert(false);
    }
}

void ProcessAutoNN(void* corr, void* field, int dots, int coord, int bin_type, int metric)
{
    dbg<<"Start ProcessAutoNN\n";
    ProcessAuto2<NData>(corr, field, dots, coord, bin_type, metric);
}

void ProcessAutoKK(void* corr, void* field, int dots, int coord, int bin_type, int metric)
{
    dbg<<"Start ProcessAutoKK\n";
    ProcessAuto2<KData>(corr, field, dots, coord, bin_type, metric);
}

void ProcessAutoGG(void* corr, void* field, int dots, int coord, int bin_type, int metric)
{
    dbg<<"Start ProcessAutoGG\n";
    ProcessAuto2<GData>(corr, field, dots, coord, bin_type, metric);
}

template <int D1, int D2, int B>
void ProcessCross2b(BinnedCorr2<D1,D2,B>& corr, void* field1, void* field2,
                    int dots, int coord, int metric)
{
    if (coord == Flat) {
        switch(metric) {
          case Euclidean:
            corr.template process<Flat,Euclidean>(
                *static_cast<Field<D1,Flat>*>(field1),
                *static_cast<Field<D2,Flat>*>(field2),dots);
            break;
          default:
            Assert(false);
        }
    } else {
        switch(metric) {
          case Euclidean:
            corr.template process<ThreeD,Euclidean>(
                *static_cast<Field<D1,ThreeD>*>(field1),
                *static_cast<Field<D2,ThreeD>*>(field2),dots);
            break;
          case Perp:
            corr.template process<ThreeD,Perp>(
                *static_cast<Field<D1,ThreeD>*>(field1),
                *static_cast<Field<D2,ThreeD>*>(field2),dots);
            break;
          case Lens:
            corr.template process<ThreeD,Lens>(
                *static_cast<Field<D1,ThreeD>*>(field1),
                *static_cast<Field<D2,ThreeD>*>(field2),dots);
            break;
          case Arc:
            corr.template process<Sphere,Arc>(
                *static_cast<Field<D1,Sphere>*>(field1),
                *static_cast<Field<D2,Sphere>*>(field2),dots);
            break;
          default:
            Assert(false);
        }
    }
}

template <int D1, int D2>
void ProcessCross2(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    switch(bin_type) {
      case Log:
           ProcessCross2b(*static_cast<BinnedCorr2<D1,D2,Log>*>(corr), field1, field2,
                          dots, coord, metric);
           break;
      case TwoD:
           ProcessCross2b(*static_cast<BinnedCorr2<D1,D2,TwoD>*>(corr), field1, field2,
                          dots, coord, metric);
           break;
      default:
           Assert(false);
    }
}

void ProcessCrossNN(void* corr, void* field1, void* field2, int dots, int coord,
                    int bin_type, int metric)
{
    dbg<<"Start ProcessCrossNN\n";
    ProcessCross2<NData,NData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessCrossNK(void* corr, void* field1, void* field2, int dots, int coord,
                    int bin_type, int metric)
{
    dbg<<"Start ProcessCrossNK\n";
    ProcessCross2<NData,KData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessCrossNG(void* corr, void* field1, void* field2, int dots, int coord,
                    int bin_type, int metric)
{
    dbg<<"Start ProcessCrossNG\n";
    ProcessCross2<NData,GData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessCrossKK(void* corr, void* field1, void* field2, int dots, int coord,
                    int bin_type, int metric)
{
    dbg<<"Start ProcessCrossKK\n";
    ProcessCross2<KData,KData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessCrossKG(void* corr, void* field1, void* field2, int dots, int coord,
                    int bin_type, int metric)
{
    dbg<<"Start ProcessCrossKG\n";
    ProcessCross2<KData,GData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessCrossGG(void* corr, void* field1, void* field2, int dots, int coord,
                    int bin_type, int metric)
{
    dbg<<"Start ProcessCrossGG\n";
    ProcessCross2<GData,GData>(corr, field1, field2, dots, coord, bin_type, metric);
}

template <int D1, int D2, int B>
void ProcessPair2b(BinnedCorr2<D1,D2,B>& corr, void* field1, void* field2,
                   int dots, int coord, int metric)
{
    if (coord == Flat) {
        switch(metric) {
          case Euclidean:
            corr.template processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<D1,Flat>*>(field1),
                *static_cast<SimpleField<D2,Flat>*>(field2),dots);
            break;
          default:
            Assert(false);
        }
    } else {
        switch(metric) {
          case Euclidean:
            corr.template processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<D1,ThreeD>*>(field1),
                *static_cast<SimpleField<D2,ThreeD>*>(field2),dots);
            break;
          case Perp:
            corr.template processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<D1,ThreeD>*>(field1),
                *static_cast<SimpleField<D2,ThreeD>*>(field2),dots);
            break;
          case Lens:
            corr.template processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<D1,ThreeD>*>(field1),
                *static_cast<SimpleField<D2,ThreeD>*>(field2),dots);
            break;
          case Arc:
            corr.template processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<D1,Sphere>*>(field1),
                *static_cast<SimpleField<D2,Sphere>*>(field2),dots);
            break;
          default:
            Assert(false);
        }
    }
}

template <int D1, int D2>
void ProcessPair2(void* corr, void* field1, void* field2, int dots, int coord,
                  int bin_type, int metric)
{
    switch(bin_type) {
      case Log:
           ProcessPair2b(*static_cast<BinnedCorr2<D1,D2,Log>*>(corr), field1, field2,
                         dots, coord, metric);
           break;
      case TwoD:
           ProcessPair2b(*static_cast<BinnedCorr2<D1,D2,TwoD>*>(corr), field1, field2,
                         dots, coord, metric);
           break;
      default:
           Assert(false);
    }
}

void ProcessPairNN(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    dbg<<"Start ProcessPairNN\n";
    ProcessPair2<NData,NData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessPairNK(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    dbg<<"Start ProcessPairNK\n";
    ProcessPair2<NData,KData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessPairNG(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    dbg<<"Start ProcessPairNG\n";
    ProcessPair2<NData,GData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessPairKK(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    dbg<<"Start ProcessPairKK\n";
    ProcessPair2<KData,KData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessPairKG(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    dbg<<"Start ProcessPairKG\n";
    ProcessPair2<KData,GData>(corr, field1, field2, dots, coord, bin_type, metric);
}

void ProcessPairGG(void* corr, void* field1, void* field2, int dots, int coord,
                   int bin_type, int metric)
{
    dbg<<"Start ProcessPairGG\n";
    ProcessPair2<GData,GData>(corr, field1, field2, dots, coord, bin_type, metric);
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



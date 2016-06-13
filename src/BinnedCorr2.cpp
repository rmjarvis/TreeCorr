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

#include "dbg.h"
#include "BinnedCorr2.h"
#include "Split.h"
#include "ProjectHelper.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Switch these for more time-consuming Assert statements
//#define XAssert(x) Assert(x)
#define XAssert(x)

template <int D1, int D2>
BinnedCorr2<D1,D2>::BinnedCorr2(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double minrpar, double maxrpar,
    double* xi0, double* xi1, double* xi2, double* xi3,
    double* meanr, double* meanlogr, double* weight, double* npairs) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minrpar(minrpar), _maxrpar(maxrpar),
    _coords(-1), _owns_data(false),
    _xi(xi0,xi1,xi2,xi3), _meanr(meanr), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
{
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _bsq = _b * _b;
}

template <int D1, int D2>
BinnedCorr2<D1,D2>::BinnedCorr2(const BinnedCorr2<D1,D2>& rhs, bool copy_data) :
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minrpar(rhs._minrpar), _maxrpar(rhs._maxrpar),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq), _bsq(rhs._bsq),
    _coords(rhs._coords), _owns_data(true),
    _xi(0,0,0,0), _weight(0)
{
    _xi.new_data(_nbins);
    _meanr = new double[_nbins];
    _meanlogr = new double[_nbins];
    _weight = new double[_nbins];
    _npairs = new double[_nbins];

    if (copy_data) *this = rhs;
    else clear();
}

template <int D1, int D2>
BinnedCorr2<D1,D2>::~BinnedCorr2()
{
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
template <int D1, int D2, int C, int M>
struct ProcessHelper
{
    static void process2(BinnedCorr2<D1,D2>& , const Cell<D1,C>& ) {}
};

template <int D, int C, int M>
struct ProcessHelper<D,D,C,M>
{
    static void process2(BinnedCorr2<D,D>& b, const Cell<D,C>& c12)
    { b.template process2<C,M>(c12); }
};

template <int D1, int D2>
void BinnedCorr2<D1,D2>::clear()
{
    _xi.clear(_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _weight[i] = 0.;
    for (int i=0; i<_nbins; ++i) _npairs[i] = 0.;
    _coords = -1;
}

template <int D1, int D2> template <int C, int M>
void BinnedCorr2<D1,D2>::process(const Field<D1,C>& field, bool dots)
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
        BinnedCorr2<D1,D2> bc2(*this,false);
#else
        BinnedCorr2<D1,D2>& bc2 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                //dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                if (dots) std::cout<<'.'<<std::flush;
            }
            const Cell<D1,C>& c1 = *field.getCells()[i];
            ProcessHelper<D1,D2,C,M>::process2(bc2,c1);
            for (int j=i+1;j<n1;++j) {
                const Cell<D1,C>& c2 = *field.getCells()[j];
                bc2.process11<C,M>(c1,c2);
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

template <int D1, int D2> template <int C, int M>
void BinnedCorr2<D1,D2>::process(const Field<D1,C>& field1, const Field<D2,C>& field2,
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
        BinnedCorr2<D1,D2> bc2(*this,false);
#else
        BinnedCorr2<D1,D2>& bc2 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                //dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                if (dots) std::cout<<'.'<<std::flush;
            }
            const Cell<D1,C>& c1 = *field1.getCells()[i];
            for (int j=0;j<n2;++j) {
                const Cell<D2,C>& c2 = *field2.getCells()[j];
                bc2.process11<C,M>(c1,c2);
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

template <int D1, int D2> template <int C, int M>
void BinnedCorr2<D1,D2>::processPairwise(
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
        BinnedCorr2<D1,D2> bc2(*this,false);
#else
        BinnedCorr2<D1,D2>& bc2 = *this;
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
                    //xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                    std::cout<<'.'<<std::flush;
                }
            }
            const Cell<D1,C>& c1 = *field1.getCells()[i];
            const Cell<D2,C>& c2 = *field2.getCells()[i];
            double s=0.;
            const double dsq = MetricHelper<M>::DistSq(c1.getPos(),c2.getPos(),s,s);
            if (dsq >= _minsepsq && dsq < _maxsepsq) {
                bc2.template directProcess11<C,M>(c1,c2,dsq);
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

template <int D1, int D2> template <int C, int M>
void BinnedCorr2<D1,D2>::process2(const Cell<D1,C>& c12)
{
    if (c12.getW() == 0.) return;
    if (c12.getSize() < _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2<C,M>(*c12.getLeft());
    process2<C,M>(*c12.getRight());
    process11<C,M>(*c12.getLeft(),*c12.getRight());
}

template <int D1, int D2> template <int C, int M>
void BinnedCorr2<D1,D2>::process11(const Cell<D1,C>& c1, const Cell<D2,C>& c2)
{
    //dbg<<"Start process11 for "<<c1.getPos()<<",  "<<c2.getPos()<<"   ";
    //dbg<<"w = "<<c1.getW()<<", "<<c2.getW()<<std::endl;
    if (c1.getW() == 0. || c2.getW() == 0.) return;

    double s1 = c1.getSize();
    double s2 = c2.getSize();
    //dbg<<"s1,s2 = "<<s1<<','<<s2<<std::endl;
    const double dsq = MetricHelper<M>::DistSq(c1.getPos(),c2.getPos(),s1,s2);
    //dbg<<"s1,s2 => "<<s1<<','<<s2<<std::endl;
    const double s1ps2 = s1+s2;

    //dbg<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<std::endl;
    if (MetricHelper<M>::TooSmallDist(c1.getPos(), c2.getPos(), s1ps2, dsq, _minsep, _minsepsq,
                                      _minrpar))
        return;
    //dbg<<"Not too small\n";
    if (MetricHelper<M>::TooLargeDist(c1.getPos(), c2.getPos(), s1ps2, dsq, _maxsep, _maxsepsq,
                                      _maxrpar))
        return;
    //dbg<<"Not too large\n";

    // See if need to split:
    bool split1=false, split2=false;
    CalcSplitSq(split1,split2,dsq,s1,s2,_bsq);
    //dbg<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<"  ";
    //dbg<<"s1ps2 / d = "<<s1ps2 / sqrt(dsq)<<", b = "<<_b<<"  ";
    //dbg<<"split = "<<split1<<','<<split2<<std::endl;

    if (split1) {
        if (split2) {
            if (!c1.getLeft()) {
                std::cerr<<"minsep = "<<_minsep<<", maxsep = "<<_maxsep<<std::endl;
                std::cerr<<"minsepsq = "<<_minsepsq<<", maxsepsq = "<<_maxsepsq<<std::endl;
                std::cerr<<"c1.Size = "<<c1.getSize()<<", c2.Size = "<<c2.getSize()<<std::endl;
                std::cerr<<"c1.SizeSq = "<<c1.getSizeSq()<<
                    ", c2.SizeSq = "<<c2.getSizeSq()<<std::endl;
                std::cerr<<"c1.N = "<<c1.getN()<<", c2.N = "<<c2.getN()<<std::endl;
                std::cerr<<"c1.Pos = "<<c1.getPos();
                std::cerr<<", c2.Pos = "<<c2.getPos()<<std::endl;
                std::cerr<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<std::endl;
            }
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<C,M>(*c1.getLeft(),*c2.getLeft());
            process11<C,M>(*c1.getLeft(),*c2.getRight());
            process11<C,M>(*c1.getRight(),*c2.getLeft());
            process11<C,M>(*c1.getRight(),*c2.getRight());
        } else {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11<C,M>(*c1.getLeft(),c2);
            process11<C,M>(*c1.getRight(),c2);
        }
    } else {
        if (split2) {
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11<C,M>(c1,*c2.getLeft());
            process11<C,M>(c1,*c2.getRight());
        } else if (dsq >= _minsepsq && dsq < _maxsepsq) {
            XAssert(NoSplit(c1,c2,sqrt(dsq),_b));
            directProcess11<C,M>(c1,c2,dsq);
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
        XiData<NData,NData>& , int )
    {}
};

template <>
struct DirectHelper<NData,KData>
{
    template <int C, int M>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<KData,C>& c2, const double ,
        XiData<NData,KData>& xi, int k)
    { xi.xi[k] += c1.getW() * c2.getData().getWK(); }
};

template <>
struct DirectHelper<NData,GData>
{
    template <int C, int M>
    static void ProcessXi(
        const Cell<NData,C>& c1, const Cell<GData,C>& c2, const double dsq,
        XiData<NData,GData>& xi, int k)
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
        XiData<KData,KData>& xi, int k)
    { xi.xi[k] += c1.getData().getWK() * c2.getData().getWK(); }
};

template <>
struct DirectHelper<KData,GData>
{
    template <int C, int M>
    static void ProcessXi(
        const Cell<KData,C>& c1, const Cell<GData,C>& c2, const double dsq,
        XiData<KData,GData>& xi, int k)
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
        XiData<GData,GData>& xi, int k)
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
    }
};

template <int D1, int D2> template <int C, int M>
void BinnedCorr2<D1,D2>::directProcess11(
    const Cell<D1,C>& c1, const Cell<D2,C>& c2, const double dsq)
{
    //dbg<<"DirectProcess11: dsq = "<<dsq<<std::endl;
    XAssert(dsq >= _minsepsq);
    XAssert(dsq < _maxsepsq);
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*_b + 0.0001);

    const double r = sqrt(dsq);
    const double logr = log(dsq)/2.;
    XAssert(logr >= _logminsep);

    XAssert(_binsize != 0.);
    const int k = int((logr - _logminsep)/_binsize);
    XAssert(k >= 0);
    XAssert(k < _nbins);
    //dbg<<"r,logr,k = "<<r<<','<<logr<<','<<k<<std::endl;

    double nn = double(c1.getN()) * double(c2.getN());
    _npairs[k] += nn;

    double ww = double(c1.getW()) * double(c2.getW());
    _meanr[k] += ww * r;
    _meanlogr[k] += ww * logr;
    _weight[k] += ww;
    //dbg<<"n,w = "<<nn<<','<<ww<<" ==>  "<<_npairs[k]<<','<<_weight[k]<<std::endl;

    DirectHelper<D1,D2>::template ProcessXi<C,M>(c1,c2,dsq,_xi,k);
}

template <int D1, int D2>
void BinnedCorr2<D1,D2>::operator=(const BinnedCorr2<D1,D2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.copy(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template <int D1, int D2>
void BinnedCorr2<D1,D2>::operator+=(const BinnedCorr2<D1,D2>& rhs)
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

void* BuildNNCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNNCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,NData>(
            minsep, maxsep, nbins, binsize, b,
            minrpar, maxrpar,
            0, 0, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,KData>(
            minsep, maxsep, nbins, binsize, b,
            minrpar, maxrpar,
            xi, 0, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi, double* xi_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,GData>(
            minsep, maxsep, nbins, binsize, b,
            minrpar, maxrpar,
            xi, xi_im, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<KData,KData>(
            minsep, maxsep, nbins, binsize, b,
            minrpar, maxrpar,
            xi, 0, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xi, double* xi_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<KData,GData>(
            minsep, maxsep, nbins, binsize, b,
            minrpar, maxrpar,
            xi, xi_im, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double minrpar, double maxrpar,
                  double* xip, double* xip_im, double* xim, double* xim_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildGGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<GData,GData>(
            minsep, maxsep, nbins, binsize, b,
            minrpar, maxrpar,
            xip, xip_im, xim, xim_im,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void DestroyNNCorr(void* corr)
{
    dbg<<"Start DestroyNNCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<NData,NData>*>(corr);
}

void DestroyNKCorr(void* corr)
{
    dbg<<"Start DestroyNKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<NData,KData>*>(corr);
}

void DestroyNGCorr(void* corr)
{
    dbg<<"Start DestroyNGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<NData,GData>*>(corr);
}

void DestroyKKCorr(void* corr)
{
    dbg<<"Start DestroyKKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<KData,KData>*>(corr);
}

void DestroyKGCorr(void* corr)
{
    dbg<<"Start DestroyKGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<KData,GData>*>(corr);
}

void DestroyGGCorr(void* corr)
{
    dbg<<"Start DestroyGGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<GData,GData>*>(corr);
}


void ProcessAutoNN(void* corr, void* field, int dots, int coord, int metric)
{
    dbg<<"Start ProcessAutoNN\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field),dots);
        else
            Assert(false);
    }
}

void ProcessAutoKK(void* corr, void* field, int dots, int coord, int metric)
{
    dbg<<"Start ProcessAutoKK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<KData,Flat>*>(field),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<KData,ThreeD>*>(field),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<KData,ThreeD>*>(field),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<KData,ThreeD>*>(field),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<KData,Sphere>*>(field),dots);
        else
            Assert(false);
    }
}

void ProcessAutoGG(void* corr, void* field, int dots, int coord, int metric)
{
    dbg<<"Start ProcessAutoGG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<GData,Flat>*>(field),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<GData,ThreeD>*>(field),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<GData,ThreeD>*>(field),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<GData,ThreeD>*>(field),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<GData,Sphere>*>(field),dots);
        else
            Assert(false);
    }
}

void ProcessCrossNN(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessCrossNN\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field1),
                *static_cast<Field<NData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field1),
                *static_cast<Field<NData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessCrossNK(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessCrossNK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field1),
                *static_cast<Field<KData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field1),
                *static_cast<Field<KData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessCrossNG(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessCrossNG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field1),
                *static_cast<Field<GData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field1),
                *static_cast<Field<GData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessCrossKK(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessCrossKK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<KData,Flat>*>(field1),
                *static_cast<Field<KData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<KData,Sphere>*>(field1),
                *static_cast<Field<KData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessCrossKG(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessCrossKG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<KData,Flat>*>(field1),
                *static_cast<Field<GData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<KData,Sphere>*>(field1),
                *static_cast<Field<GData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessCrossGG(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessCrossGG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<GData,Flat>*>(field1),
                *static_cast<Field<GData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<GData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<GData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<GData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<GData,Sphere>*>(field1),
                *static_cast<Field<GData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessPairNN(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessPairNN\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<NData,Flat>*>(field1),
                *static_cast<SimpleField<NData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<NData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<NData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<NData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<NData,Sphere>*>(field1),
                *static_cast<SimpleField<NData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessPairNK(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessPairNK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<NData,Flat>*>(field1),
                *static_cast<SimpleField<KData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<KData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<KData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<KData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<NData,Sphere>*>(field1),
                *static_cast<SimpleField<KData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessPairNG(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessPairNG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<NData,Flat>*>(field1),
                *static_cast<SimpleField<GData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<NData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<NData,Sphere>*>(field1),
                *static_cast<SimpleField<GData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessPairKK(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessPairKK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<KData,Flat>*>(field1),
                *static_cast<SimpleField<KData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<KData,ThreeD>*>(field1),
                *static_cast<SimpleField<KData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<KData,ThreeD>*>(field1),
                *static_cast<SimpleField<KData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<KData,ThreeD>*>(field1),
                *static_cast<SimpleField<KData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<KData,Sphere>*>(field1),
                *static_cast<SimpleField<KData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessPairKG(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessPairKG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<KData,Flat>*>(field1),
                *static_cast<SimpleField<GData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<KData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<KData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<KData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<KData,Sphere>*>(field1),
                *static_cast<SimpleField<GData,Sphere>*>(field2),dots);
        else
            Assert(false);
    }
}

void ProcessPairGG(void* corr, void* field1, void* field2, int dots, int coord, int metric)
{
    dbg<<"Start ProcessPairGG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise<Flat,Euclidean>(
                *static_cast<SimpleField<GData,Flat>*>(field1),
                *static_cast<SimpleField<GData,Flat>*>(field2),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise<ThreeD,Euclidean>(
                *static_cast<SimpleField<GData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise<ThreeD,Perp>(
                *static_cast<SimpleField<GData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise<ThreeD,Lens>(
                *static_cast<SimpleField<GData,ThreeD>*>(field1),
                *static_cast<SimpleField<GData,ThreeD>*>(field2),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise<Sphere,Arc>(
                *static_cast<SimpleField<GData,Sphere>*>(field1),
                *static_cast<SimpleField<GData,Sphere>*>(field2),dots);
        else
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



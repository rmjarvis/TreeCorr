/* Copyright (c) 2003-2014 by Mike Jarvis
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
#include "MetricHelper.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Switch these for more time-consuming Assert statements
//#define XAssert(x) Assert(x)
#define XAssert(x)

template <int DC1, int DC2>
BinnedCorr2<DC1,DC2>::BinnedCorr2(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double* xi0, double* xi1, double* xi2, double* xi3,
    double* meanr, double* meanlogr, double* weight, double* npairs) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _metric(-1), _owns_data(false),
    _xi(xi0,xi1,xi2,xi3), _meanr(meanr), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
{
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _bsq = _b * _b;
}

template <int DC1, int DC2>
BinnedCorr2<DC1,DC2>::BinnedCorr2(const BinnedCorr2<DC1,DC2>& rhs, bool copy_data) :
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq), _bsq(rhs._bsq),
    _metric(rhs._metric), _owns_data(true),
    _xi(0,0,0,0), _weight(0)
{
    _xi.new_data(_nbins);
    _meanr = new double[_nbins];
    _meanlogr = new double[_nbins];
    if (rhs._weight) _weight = new double[_nbins];
    _npairs = new double[_nbins];

    if (copy_data) *this = rhs;
    else clear();
}

template <int DC1, int DC2>
BinnedCorr2<DC1,DC2>::~BinnedCorr2()
{
    if (_owns_data) {
        _xi.delete_data(_nbins);
        delete [] _meanr; _meanr = 0;
        delete [] _meanlogr; _meanlogr = 0;
        if (_weight) delete [] _weight; _weight = 0;
        delete [] _npairs; _npairs = 0;
    }
}

// BinnedCorr2::process2 is invalid if DC1 != DC2, so this helper struct lets us only call 
// process2 when DC1 == DC2.
template <int DC1, int DC2, int M>
struct ProcessHelper
{
    static void process2(BinnedCorr2<DC1,DC2>& , const Cell<DC1,M>& ) {}
};

    
template <int DC, int M>
struct ProcessHelper<DC,DC,M>
{
    static void process2(BinnedCorr2<DC,DC>& b, const Cell<DC,M>& c12) { b.process2(c12); }
};

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::clear()
{
    _xi.clear(_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = 0.;
    if (_weight) for (int i=0; i<_nbins; ++i) _weight[i] = 0.;
    for (int i=0; i<_nbins; ++i) _npairs[i] = 0.;
    _metric = -1;
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process(const Field<DC1,M>& field, bool dots)
{
    Assert(DC1 == DC2);
    Assert(_metric == -1 || _metric == M);
    _metric = M;
    const long n1 = field.getNTopLevel();
    xdbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);
#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this,false);
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
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
            const Cell<DC1,M>& c1 = *field.getCells()[i];
            ProcessHelper<DC1,DC2,M>::process2(bc2,c1);
            for (int j=i+1;j<n1;++j) {
                const Cell<DC1,M>& c2 = *field.getCells()[j];
                bc2.process11(c1,c2);
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

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process(const Field<DC1,M>& field1, const Field<DC2,M>& field2,
                                   bool dots)
{
    Assert(_metric == -1 || _metric == M);
    _metric = M;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    xdbg<<"field1 has "<<n1<<" top level nodes\n";
    xdbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this,false);
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
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
            const Cell<DC1,M>& c1 = *field1.getCells()[i];
            for (int j=0;j<n2;++j) {
                const Cell<DC2,M>& c2 = *field2.getCells()[j];
                bc2.process11(c1,c2);
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

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::processPairwise(
    const SimpleField<DC1,M>& field1, const SimpleField<DC2,M>& field2, bool dots)
{ 
    Assert(_metric == -1 || _metric == M);
    _metric = M;
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
        BinnedCorr2<DC1,DC2> bc2(*this,false);
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
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
            const Cell<DC1,M>& c1 = *field1.getCells()[i];
            const Cell<DC2,M>& c2 = *field2.getCells()[i];
            const double dsq = DistSq(c1.getData().getPos(),c2.getData().getPos());
            if (dsq >= _minsepsq && dsq < _maxsepsq) {
                bc2.directProcess11(c1,c2,dsq);
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

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process2(const Cell<DC1,M>& c12)
{
    if (c12.getSize() < _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2(*c12.getLeft());
    process2(*c12.getRight());
    process11(*c12.getLeft(),*c12.getRight());
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process11(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2)
{
    const double dsq = DistSq(c1.getData().getPos(),c2.getData().getPos());
    const double s1ps2 = c1.getAllSize()+c2.getAllSize();

    
    if (TooSmallDist(c1.getData().getPos(), c2.getData().getPos(), s1ps2, dsq, _minsep, _minsepsq))
        return;
    if (TooLargeDist(c1.getData().getPos(), c2.getData().getPos(), s1ps2, dsq, _maxsep, _maxsepsq))
        return;

    // See if need to split:
    bool split1=false, split2=false;
    CalcSplitSq(split1,split2,c1,c2,dsq,s1ps2,_bsq);

    if (split1) {
        if (split2) {
            if (!c1.getLeft()) {
                std::cerr<<"minsep = "<<_minsep<<", maxsep = "<<_maxsep<<std::endl;
                std::cerr<<"minsepsq = "<<_minsepsq<<", maxsepsq = "<<_maxsepsq<<std::endl;
                std::cerr<<"c1.Size = "<<c1.getSize()<<", c2.Size = "<<c2.getSize()<<std::endl;
                std::cerr<<"c1.SizeSq = "<<c1.getSizeSq()<<
                    ", c2.SizeSq = "<<c2.getSizeSq()<<std::endl;
                std::cerr<<"c1.N = "<<c1.getData().getN()<<", c2.N = "<<c2.getData().getN()<<std::endl;
                std::cerr<<"c1.Pos = "<<c1.getData().getPos();
                std::cerr<<", c2.Pos = "<<c2.getData().getPos()<<std::endl;
                std::cerr<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<std::endl;
            }
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11(*c1.getLeft(),*c2.getLeft());
            process11(*c1.getLeft(),*c2.getRight());
            process11(*c1.getRight(),*c2.getLeft());
            process11(*c1.getRight(),*c2.getRight());
        } else {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11(*c1.getLeft(),c2);
            process11(*c1.getRight(),c2);
        }
    } else {
        if (split2) {
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11(c1,*c2.getLeft());
            process11(c1,*c2.getRight());
        } else if (dsq >= _minsepsq && dsq < _maxsepsq) {
            XAssert(NoSplit(c1,c2,sqrt(dsq),_b));
            directProcess11(c1,c2,dsq);
        }
    }
}


// We also set up a helper class for doing the direct processing
template <int DC1, int DC2>
struct DirectHelper;

template <>
struct DirectHelper<NData,NData>
{
    template <int M>
    static void ProcessXi(
        const Cell<NData,M>& , const Cell<NData,M>& , const double ,
        XiData<NData,NData>& , int )
    {}
};
 
template <>
struct DirectHelper<NData,KData>
{
    template <int M>
    static void ProcessXi(
        const Cell<NData,M>& c1, const Cell<KData,M>& c2, const double ,
        XiData<NData,KData>& xi, int k)
    { xi.xi[k] += c1.getData().getW() * c2.getData().getWK(); }
};
 
template <>
struct DirectHelper<NData,GData>
{
    template <int M>
    static void ProcessXi(
        const Cell<NData,M>& c1, const Cell<GData,M>& c2, const double dsq,
        XiData<NData,GData>& xi, int k)
    {
        std::complex<double> g2;
        MetricHelper<M>::ProjectShear(c1,c2,dsq,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getData().getW();
        xi.xi[k] += real(g2);
        xi.xi_im[k] += imag(g2);

    }
};

template <>
struct DirectHelper<KData,KData>
{
    template <int M>
    static void ProcessXi(
        const Cell<KData,M>& c1, const Cell<KData,M>& c2, const double ,
        XiData<KData,KData>& xi, int k)
    { xi.xi[k] += c1.getData().getWK() * c2.getData().getWK(); }
};
 
template <>
struct DirectHelper<KData,GData>
{
    template <int M>
    static void ProcessXi(
        const Cell<KData,M>& c1, const Cell<GData,M>& c2, const double dsq,
        XiData<KData,GData>& xi, int k)
    {
        std::complex<double> g2;
        MetricHelper<M>::ProjectShear(c1,c2,dsq,g2);
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
    template <int M>
    static void ProcessXi(
        const Cell<GData,M>& c1, const Cell<GData,M>& c2, const double dsq,
        XiData<GData,GData>& xi, int k)
    {
        std::complex<double> g1, g2;
        MetricHelper<M>::ProjectShears(c1,c2,dsq,g1,g2);

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

// The way meanr, meanlogr and weight are processed is the same for everything except NN.
// So do this as a separate template specialization:
template <int DC1, int DC2>
struct DirectHelper2
{
    template <int M>
    static void ProcessWeight(
        const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, 
        const double r,  const double logr, const double ,
        double* meanr, double* meanlogr, double* weight, int k)
    {
        double ww = double(c1.getData().getW()) * double(c2.getData().getW());
        meanr[k] += ww * r;
        meanlogr[k] += ww * logr;
        weight[k] += ww;
    }
};
            
template <>
struct DirectHelper2<NData, NData>
{
    template <int M>
    static void ProcessWeight(
        const Cell<NData,M>& , const Cell<NData,M>& ,
        const double r, const double logr, const double nn,
        double* meanr, double* meanlogr, double* , int k)
    { 
        meanr[k] += nn * r; 
        meanlogr[k] += nn * logr; 
    }
};

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::directProcess11(
    const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const double dsq)
{
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

    double nn = double(c1.getData().getN()) * double(c2.getData().getN());
    _npairs[k] += nn;

    DirectHelper<DC1,DC2>::ProcessXi(c1,c2,dsq,_xi,k);

    DirectHelper2<DC1,DC2>::ProcessWeight(c1,c2,r,logr,nn,_meanr,_meanlogr,_weight,k);
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.copy(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] = rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    if (_weight) for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator+=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.add(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanr[i] += rhs._meanr[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] += rhs._meanlogr[i];
    if (_weight) for (int i=0; i<_nbins; ++i) _weight[i] += rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] += rhs._npairs[i];
}

//
//
// The C interface for python
//
//

void* BuildNNCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* meanr, double* meanlogr, double* npairs)
{
    dbg<<"Start BuildNNCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,NData>(
            minsep, maxsep, nbins, binsize, b,
            0, 0, 0, 0,
            meanr, meanlogr, 0, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,KData>(
            minsep, maxsep, nbins, binsize, b,
            xi, 0, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi, double* xi_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,GData>(
            minsep, maxsep, nbins, binsize, b,
            xi, xi_im, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<KData,KData>(
            minsep, maxsep, nbins, binsize, b,
            xi, 0, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi, double* xi_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<KData,GData>(
            minsep, maxsep, nbins, binsize, b,
            xi, xi_im, 0, 0,
            meanr, meanlogr, weight, npairs));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xip, double* xip_im, double* xim, double* xim_im,
                  double* meanr, double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildGGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<GData,GData>(
            minsep, maxsep, nbins, binsize, b,
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


void ProcessAutoNNFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNNFlat\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field),dots);
}
void ProcessAutoNN3D(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNN3D\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field),dots);
}
void ProcessAutoNNPerp(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNNPerp\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Perp>*>(field),dots);
}

void ProcessAutoKKFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoKKFlat\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Flat>*>(field),dots);
}
void ProcessAutoKK3D(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoKK3D\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Sphere>*>(field),dots);
}
void ProcessAutoKKPerp(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoKKPerp\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Perp>*>(field),dots);
}

void ProcessAutoGGFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoGGFlat\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Flat>*>(field),dots);
}
void ProcessAutoGG3D(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoGG3D\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Sphere>*>(field),dots);
}
void ProcessAutoGGPerp(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoGGPerp\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Perp>*>(field),dots);
}

void ProcessCrossNNFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNNFlat\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field1),
        *static_cast<Field<NData,Flat>*>(field2),dots);
}
void ProcessCrossNN3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNN3D\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field1),
        *static_cast<Field<NData,Sphere>*>(field2),dots);
}
void ProcessCrossNNPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNNPerp\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Perp>*>(field1),
        *static_cast<Field<NData,Perp>*>(field2),dots);
}

void ProcessCrossNKFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNKFlat\n";
    static_cast<BinnedCorr2<NData,KData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field1),
        *static_cast<Field<KData,Flat>*>(field2),dots);
}
void ProcessCrossNK3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNK3D\n";
    static_cast<BinnedCorr2<NData,KData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field1),
        *static_cast<Field<KData,Sphere>*>(field2),dots);
}
void ProcessCrossNKPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNKPerp\n";
    static_cast<BinnedCorr2<NData,KData>*>(corr)->process(
        *static_cast<Field<NData,Perp>*>(field1),
        *static_cast<Field<KData,Perp>*>(field2),dots);
}

void ProcessCrossNGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNGFlat\n";
    static_cast<BinnedCorr2<NData,GData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field1),
        *static_cast<Field<GData,Flat>*>(field2),dots);
}
void ProcessCrossNG3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNG3D\n";
    static_cast<BinnedCorr2<NData,GData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field1),
        *static_cast<Field<GData,Sphere>*>(field2),dots);
}
void ProcessCrossNGPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNGPerp\n";
    static_cast<BinnedCorr2<NData,GData>*>(corr)->process(
        *static_cast<Field<NData,Perp>*>(field1),
        *static_cast<Field<GData,Perp>*>(field2),dots);
}

void ProcessCrossKKFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKKFlat\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Flat>*>(field1),
        *static_cast<Field<KData,Flat>*>(field2),dots);
}
void ProcessCrossKK3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKK3D\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Sphere>*>(field1),
        *static_cast<Field<KData,Sphere>*>(field2),dots);
}
void ProcessCrossKKPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKKPerp\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Perp>*>(field1),
        *static_cast<Field<KData,Perp>*>(field2),dots);
}

void ProcessCrossKGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKGFlat\n";
    static_cast<BinnedCorr2<KData,GData>*>(corr)->process(
        *static_cast<Field<KData,Flat>*>(field1),
        *static_cast<Field<GData,Flat>*>(field2),dots);
}
void ProcessCrossKG3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKG3D\n";
    static_cast<BinnedCorr2<KData,GData>*>(corr)->process(
        *static_cast<Field<KData,Sphere>*>(field1),
        *static_cast<Field<GData,Sphere>*>(field2),dots);
}
void ProcessCrossKGPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKGPerp\n";
    static_cast<BinnedCorr2<KData,GData>*>(corr)->process(
        *static_cast<Field<KData,Perp>*>(field1),
        *static_cast<Field<GData,Perp>*>(field2),dots);
}

void ProcessCrossGGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossGGFlat\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Flat>*>(field1),
        *static_cast<Field<GData,Flat>*>(field2),dots);
}
void ProcessCrossGG3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossGG3D\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Sphere>*>(field1),
        *static_cast<Field<GData,Sphere>*>(field2),dots);
}
void ProcessCrossGGPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossGGPerp\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Perp>*>(field1),
        *static_cast<Field<GData,Perp>*>(field2),dots);
}

void ProcessPairwiseNNFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNNFlat\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Flat>*>(field1),
        *static_cast<SimpleField<NData,Flat>*>(field2),dots);
}
void ProcessPairwiseNN3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNN3D\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Sphere>*>(field1),
        *static_cast<SimpleField<NData,Sphere>*>(field2),dots);
}
void ProcessPairwiseNNPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNNPerp\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Perp>*>(field1),
        *static_cast<SimpleField<NData,Perp>*>(field2),dots);
}

void ProcessPairwiseNKFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNKFlat\n";
    static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Flat>*>(field1),
        *static_cast<SimpleField<KData,Flat>*>(field2),dots);
}
void ProcessPairwiseNK3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNK3D\n";
    static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Sphere>*>(field1),
        *static_cast<SimpleField<KData,Sphere>*>(field2),dots);
}
void ProcessPairwiseNKPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNKPerp\n";
    static_cast<BinnedCorr2<NData,KData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Perp>*>(field1),
        *static_cast<SimpleField<KData,Perp>*>(field2),dots);
}

void ProcessPairwiseNGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNGFlat\n";
    static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Flat>*>(field1),
        *static_cast<SimpleField<GData,Flat>*>(field2),dots);
}
void ProcessPairwiseNG3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNG3D\n";
    static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Sphere>*>(field1),
        *static_cast<SimpleField<GData,Sphere>*>(field2),dots);
}
void ProcessPairwiseNGPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNGPerp\n";
    static_cast<BinnedCorr2<NData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<NData,Perp>*>(field1),
        *static_cast<SimpleField<GData,Perp>*>(field2),dots);
}

void ProcessPairwiseKKFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKKFlat\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise(
        *static_cast<SimpleField<KData,Flat>*>(field1),
        *static_cast<SimpleField<KData,Flat>*>(field2),dots);
}
void ProcessPairwiseKK3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKK3D\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise(
        *static_cast<SimpleField<KData,Sphere>*>(field1),
        *static_cast<SimpleField<KData,Sphere>*>(field2),dots);
}
void ProcessPairwiseKKPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKKPerp\n";
    static_cast<BinnedCorr2<KData,KData>*>(corr)->processPairwise(
        *static_cast<SimpleField<KData,Perp>*>(field1),
        *static_cast<SimpleField<KData,Perp>*>(field2),dots);
}

void ProcessPairwiseKGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKGFlat\n";
    static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<KData,Flat>*>(field1),
        *static_cast<SimpleField<GData,Flat>*>(field2),dots);
}
void ProcessPairwiseKG3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKG3D\n";
    static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<KData,Sphere>*>(field1),
        *static_cast<SimpleField<GData,Sphere>*>(field2),dots);
}
void ProcessPairwiseKGPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKGPerp\n";
    static_cast<BinnedCorr2<KData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<KData,Perp>*>(field1),
        *static_cast<SimpleField<GData,Perp>*>(field2),dots);
}

void ProcessPairwiseGGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseGGFlat\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<GData,Flat>*>(field1),
        *static_cast<SimpleField<GData,Flat>*>(field2),dots);
}
void ProcessPairwiseGG3D(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseGG3D\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<GData,Sphere>*>(field1),
        *static_cast<SimpleField<GData,Sphere>*>(field2),dots);
}
void ProcessPairwiseGGPerp(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseGGPerp\n";
    static_cast<BinnedCorr2<GData,GData>*>(corr)->processPairwise(
        *static_cast<SimpleField<GData,Perp>*>(field1),
        *static_cast<SimpleField<GData,Perp>*>(field2),dots);
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
        


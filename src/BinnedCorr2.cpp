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
    double* meanlogr, double* weight, double* npairs) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _metric(-1), _owns_data(false),
    _xi(xi0,xi1,xi2,xi3), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
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
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = 0.;
    if (_weight) for (int i=0; i<_nbins; ++i) _weight[i] = 0.;
    for (int i=0; i<_nbins; ++i) _npairs[i] = 0.;
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
    CalcSplitSq(split1,split2,c1,c2,dsq,_bsq);

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


// For the direct processing, we need a helper struct to handle some of the manipulations
// we need to do to the shear values.
template <int M>
struct MetricHelper;

template <>
struct MetricHelper<Flat>
{
    template <int DC1>
    static void ProjectShear(
        const Cell<DC1,Flat>& c1, const Cell<GData,Flat>& c2,
        double dsq, std::complex<double>& g2)
    {
        // Project given shear to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        Assert(dsq != 0.);
        std::complex<double> expm2iarg = conj(cr*cr)/dsq;
        g2 = c2.getData().getWG() * expm2iarg;
    }

    static void ProjectShears(
        const Cell<GData,Flat>& c1, const Cell<GData,Flat>& c2,
        double dsq, std::complex<double>& g1, std::complex<double>& g2)
    {
        // Project given shears to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        Assert(dsq != 0.);
        std::complex<double> expm2iarg = conj(cr*cr)/dsq;
        g1 = c1.getData().getWG() * expm2iarg;
        g2 = c2.getData().getWG() * expm2iarg;
    }
};

template <>
struct MetricHelper<Sphere>
{
    static void ProjectShear2(
        const Position<Sphere>& p1, const Position<Sphere>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g2)
    {
        // For spherical triangles, it's a bit trickier, since the angles aren't equal.
        // In this function we just project the shear at p2.
        // This will be used by both NG and GG projections below.

        // We need the angle at each point between north and the line connecting the 
        // two points.
        //
        // Use the spherical law of cosines:
        //
        // cos(a) = cos(b) cos(c) + sin(b) sin(c) cos(A)
        //
        // In our case:
        //   a = distance from pole to p1 = Pi/2 - dec1
        //   b = distance from pole to p2 = Pi/2 - dec2
        //   c = the great circle distance between the two points.
        //   A = angle between c and north at p2
        //   B = angle between c and north at p1
        //   C = angle between meridians = ra1 - ra2
        // 

        // cos(C) = cos(ra1 - ra2) = cos(ra1)cos(ra2) + sin(ra1)sin(ra2)
        //        = (x1/cos(dec1)) (x2/cos(dec2)) + (y1/cos(dec1)) (y2/cos(dec2))
        //        = (x1 x2 + y1 y2) / (cos(dec1) cos(dec2))
        // sin(C) = sin(ra1 - ra2) = sin(ra1)cos(ra2) - cos(ra1)sin(ra2)
        //        = (y1/cos(dec1)) (x2/cos(dec2)) - (x1/cos(dec1)) (y2/cos(dec2))
        //        = (y1 x2 - x1 y2) / (cos(dec1) cos(dec2))

        // cos(A) = (sin(dec1) - sin(dec2) cos(c)) / (cos(dec2) sin(c))
        //
        // The division is fairly unstable if cos(dec2) or sin(c) is small.
        // And sin(c) is often small, so we want to manipulate this a bit.
        // cos(c) = sin(dec1) sin(dec2) + cos(dec1) cos(dec2) cos(C)
        // cos(A) = (sin(dec1) cos(dec2)^2 - sin(dec2) cos(dec1) cos(dec2) cos(C)) /
        //                (cos(dec2) sin(c))
        //        = (sin(dec1) cos(dec2) - sin(dec2) cos(dec1) cos(C)) / sin(c)
        //        = (sin(dec1) cos(dec2)^2 - sin(dec2) (x1 x2 + y1 y2)) / (cos(dec2) sin(c))
        //        = (z1 (1-z2^2) - z2 (x1 x2 + y1 y2)) / (cos(dec2) sin(c))
        //        = (z1 - z2 (x1 x2 + y1 y2 + z1 z2)) / (cos(dec2) sin(c))
        //
        // Note:  dsq = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
        //            = (x1^2+y1^2+z1^2) + (x2^2+y2^2+z2^2) -2x1x2 -2y1y2 - 2z1z2
        //            = 2 - 2 (x1 x2 + y1 y2 + z1 z2)
        //            = 2 - 2 dot  ^^^^ This is the dot product of the positions.
        //
        // cos(A) = ( z1 - z2*dot ) / (cos(dec2) sin(c))
        //        = ( (z1-z2) + z2*(1-dot) ) / (cos(dec2) sin(c))
        //        = ( (z1-z2) + z2*(dsq/2) ) / (cos(dec2) sin(c))
        //
        // sin(A) / sin(a) = sin(C) / sin(c)
        // sin(A) = cos(dec1) sin(C) / sin(c)
        //        = (y1 x2 - x1 y2) / (cos(dec2) sin(c))
        //
        // We ignore the denominator for now, and then figure out what it is from
        // sin(A)^2 + cos(A)^2 = 1
        double z1 = p1.getZ();
        double z2 = p2.getZ();
        double cosA = (z1-z2) + 0.5*z2*dsq;  // These are unnormalized.
        double sinA = cross;
        double cosAsq = cosA*cosA;
        double sinAsq = crosssq;
        double normAsq = cosAsq + sinAsq;
        Assert(normAsq > 0.);
        double cos2A = (cosAsq - sinAsq) / normAsq; // These are now correct.
        double sin2A = 2.*sinA*cosA / normAsq;

        // In fact, A is not really the angles by which we want to rotate the shear.
        // We really want to rotae by the angle between due _east_ and c, not _north_.
        //
        // exp(-2ialpha) = exp(-2i (A - Pi/2) )
        //               = exp(iPi) * exp(-2iA)
        //               = - exp(-2iA)

        std::complex<double> expm2ialpha(-cos2A,sin2A);
        g2 *= expm2ialpha;
    }

    static void ProjectShear1(
        const Position<Sphere>& p1, const Position<Sphere>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g1)
    {
        // It is similar for the shear at p1:

        // cos(B) = (sin(dec2) - sin(dec1) cos(c)) / (cos(dec1) sin(c))
        //        = (sin(dec2) cos(dec1)^2 - sin(dec1) (x1 x2 + y1 y2)) / (cos(dec1) sin(c))
        //        = (z2 (1-z1^2) - z1 (x1 x2 + y1 y2)) / (cos(dec1) sin(c))
        //        = (z2 - z1 (x1 x2 + y1 y2 + z1 z2)) / (cos(dec1) sin(c))
        //        = (z2-z1 + z1*dsq/2) / (cos(dec1) sin(c))
        // sin(B) / sin(b) = sin(C) / sin(c)
        // sin(B) = cos(dec2) sin(C) / sin(c)
        //        = (y1 x2 - x1 y2) / (cos(dec1) sin(c))
        double z1 = p1.getZ();
        double z2 = p2.getZ();
        double cosB = (z2-z1) + 0.5*z1*dsq;  // These are unnormalized.
        double sinB = cross;  
        double cosBsq = cosB*cosB;
        double sinBsq = crosssq;
        double normBsq = cosBsq + sinBsq;
        Assert(normBsq != 0.);
        double cos2B = (cosBsq - sinBsq) / normBsq;
        double sin2B = 2.*sinB*cosB / normBsq;

        // exp(-2ibeta)  = exp(-2i (Pi/2 - B) )
        //               = exp(-iPi) * exp(2iB)
        //               = - exp(2iB)

        std::complex<double> expm2ibeta(-cos2B,-sin2B); 
        g1 *= expm2ibeta;
    }

    template <int DC1>
    static void ProjectShear(
        const Cell<DC1,Sphere>& c1, const Cell<GData,Sphere>& c2,
        double dsq, std::complex<double>& g2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }

    static void ProjectShears(
        const Cell<GData,Sphere>& c1, const Cell<GData,Sphere>& c2,
        double dsq, std::complex<double>& g1, std::complex<double>& g2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear1(p1,p2,dsq,cross,crosssq,g1);
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }
};

// The projections for Perp are the same as for Sphere.
template <>
struct MetricHelper<Perp>
{
    static void ProjectShear2(
        const Position<Perp>& p1, const Position<Perp>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g2)
    { MetricHelper<Sphere>::ProjectShear2(p1,p2,dsq,cross,crosssq,g2); }

    static void ProjectShear1(
        const Position<Perp>& p1, const Position<Perp>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g1)
    { MetricHelper<Sphere>::ProjectShear1(p1,p2,dsq,cross,crosssq,g1); }

    template <int DC1>
    static void ProjectShear(
        const Cell<DC1,Perp>& c1, const Cell<GData,Perp>& c2,
        double dsq, std::complex<double>& g2)
    {
        const Position<Perp>& p1 = c1.getData().getPos();
        const Position<Perp>& p2 = c2.getData().getPos();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }

    static void ProjectShears(
        const Cell<GData,Perp>& c1, const Cell<GData,Perp>& c2,
        double dsq, std::complex<double>& g1, std::complex<double>& g2)
    {
        const Position<Perp>& p1 = c1.getData().getPos();
        const Position<Perp>& p2 = c2.getData().getPos();
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear1(p1,p2,dsq,cross,crosssq,g1);
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }
};

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

// The way meanlogr and weight are processed is the same for everything except NN.
// So do this as a separate template specialization:
template <int DC1, int DC2>
struct DirectHelper2
{
    template <int M>
    static void ProcessWeight(
        const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const double logr, const double ,
        double* meanlogr, double* weight, int k)
    {
        double ww = double(c1.getData().getW()) * double(c2.getData().getW());
        meanlogr[k] += ww * logr;
        weight[k] += ww;
    }
};
            
template <>
struct DirectHelper2<NData, NData>
{
    template <int M>
    static void ProcessWeight(
        const Cell<NData,M>& , const Cell<NData,M>& , const double logr, const double nn,
        double* meanlogr, double* , int k)
    { meanlogr[k] += nn * logr; }
};

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::directProcess11(
    const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const double dsq)
{
    XAssert(dsq >= _minsepsq);
    XAssert(dsq < _maxsepsq);
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*_b + 0.0001);

    const double logr = log(dsq)/2.;
    XAssert(logr >= _logminsep);

    XAssert(_binsize != 0.);
    const int k = int((logr - _logminsep)/_binsize);
    XAssert(k >= 0); 
    XAssert(k < _nbins);

    double nn = double(c1.getData().getN()) * double(c2.getData().getN());
    _npairs[k] += nn;

    DirectHelper<DC1,DC2>::ProcessXi(c1,c2,dsq,_xi,k);

    DirectHelper2<DC1,DC2>::ProcessWeight(c1,c2,logr,nn,_meanlogr,_weight,k);
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.copy(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    if (_weight) for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator+=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.add(rhs._xi,_nbins);
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
                  double* meanlogr, double* npairs)
{
    dbg<<"Start BuildNNCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,NData>(
            minsep, maxsep, nbins, binsize, b,
            0, 0, 0, 0,
            meanlogr, 0, npairs));
    dbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi,
                  double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,KData>(
            minsep, maxsep, nbins, binsize, b,
            xi, 0, 0, 0,
            meanlogr, weight, npairs));
    dbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi, double* xi_im,
                  double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildNGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<NData,GData>(
            minsep, maxsep, nbins, binsize, b,
            xi, xi_im, 0, 0,
            meanlogr, weight, npairs));
    dbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi,
                  double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<KData,KData>(
            minsep, maxsep, nbins, binsize, b,
            xi, 0, 0, 0,
            meanlogr, weight, npairs));
    dbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xi, double* xi_im,
                  double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildKGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<KData,GData>(
            minsep, maxsep, nbins, binsize, b,
            xi, xi_im, 0, 0,
            meanlogr, weight, npairs));
    dbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* xip, double* xip_im, double* xim, double* xim_im,
                  double* meanlogr, double* weight, double* npairs)
{
    dbg<<"Start BuildGGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr2<GData,GData>(
            minsep, maxsep, nbins, binsize, b,
            xip, xip_im, xim, xim_im,
            meanlogr, weight, npairs));
    dbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void DestroyNNCorr(void* corr)
{
    dbg<<"Start DestroyNNCorr\n";
    dbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<NData,NData>*>(corr);
}

void DestroyNKCorr(void* corr)
{
    dbg<<"Start DestroyNKCorr\n";
    dbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<NData,KData>*>(corr);
}

void DestroyNGCorr(void* corr)
{
    dbg<<"Start DestroyNGCorr\n";
    dbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<NData,GData>*>(corr);
}

void DestroyKKCorr(void* corr)
{
    dbg<<"Start DestroyKKCorr\n";
    dbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<KData,KData>*>(corr);
}

void DestroyKGCorr(void* corr)
{
    dbg<<"Start DestroyKGCorr\n";
    dbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<KData,GData>*>(corr);
}

void DestroyGGCorr(void* corr)
{
    dbg<<"Start DestroyGGCorr\n";
    dbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr2<GData,GData>*>(corr);
}


void ProcessAutoNNFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNNFlat\n";
    static_cast<BinnedCorr2<NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field),dots);
}
void ProcessAutoNNSphere(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNNSphere\n";
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
void ProcessAutoKKSphere(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoKKSphere\n";
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
void ProcessAutoGGSphere(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoGGSphere\n";
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
void ProcessCrossNNSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNNSphere\n";
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
void ProcessCrossNKSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNKSphere\n";
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
void ProcessCrossNGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNGSphere\n";
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
void ProcessCrossKKSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKKSphere\n";
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
void ProcessCrossKGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKGSphere\n";
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
void ProcessCrossGGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossGGSphere\n";
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
void ProcessPairwiseNNSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNNSphere\n";
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
void ProcessPairwiseNKSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNKSphere\n";
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
void ProcessPairwiseNGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseNGSphere\n";
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
void ProcessPairwiseKKSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKKSphere\n";
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
void ProcessPairwiseKGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseKGSphere\n";
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
void ProcessPairwiseGGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessPairwiseGGSphere\n";
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
        


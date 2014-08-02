
#include "dbg.h"
#include "BinnedCorr2.h"
#include "Split.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Switch these for more time-consuming Assert statements
//#define XAssert(x) Assert(x)
#define XAssert(x)

template <typename T>
inline T SQR(T x) { return x * x; }

template <int DC1, int DC2>
BinnedCorr2<DC1,DC2>::BinnedCorr2(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double* xi, double* xi1, double* xi2, double* xi3,
    double* varxi, double* meanlogr, double* weight, double* npairs) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _metric(-1),
    _xi(xi,xi1,xi2,xi3), _varxi(varxi), _meanlogr(meanlogr), _weight(weight), _npairs(npairs)
{
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _bsq = _b * _b;
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

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process(const Field<DC1,M>& field)
{
    Assert(DC1 == DC2);
    Assert(_metric == -1 || _metric == M);
    _metric = M;
    const int n1 = field.getN();
    xdbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);
#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this);
        bc2.clearData();
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
                dbg<<'.';
            }
            const Cell<DC1,M>& c1 = *field.getCells()[i];
            ProcessHelper<DC1,DC2,M>::process2(*this,c1);
            for (int j=i+1;j<n1;++j) {
                const Cell<DC1,M>& c2 = *field.getCells()[j];
                process11(c1,c2);
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            Assert(_metric == -1 || bc2._metric == -1 || _metric == bc2._metric);
            if (bc2._metric != -1) _metric = bc2._metric;
            *this += bc2;
        }
    }
#endif
    dbg<<std::endl;
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process(const Field<DC1,M>& field1, const Field<DC2,M>& field2)
{
    Assert(_metric == -1 || _metric == M);
    _metric = M;
    const int n1 = field1.getN();
    const int n2 = field2.getN();
    xdbg<<"field1 has "<<n1<<" top level nodes\n";
    xdbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this);
        bc2.clearData();
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
                dbg<<'.';
            }
            for (int j=0;j<n2;++j) {
                const Cell<DC1,M>& c1 = *field1.getCells()[i];
                const Cell<DC2,M>& c2 = *field2.getCells()[j];
                process11(c1,c2);
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            Assert(_metric == -1 || bc2._metric == -1 || _metric == bc2._metric);
            if (bc2._metric != -1) _metric = bc2._metric;
            *this += bc2;
        }
    }
#endif
    dbg<<std::endl;
}

#if 0
template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::doProcessPairwise(const InputFile& file1, const InputFile& file2)
{
    std::vector<CellData<DC1,M>*> celldata1;
    std::vector<CellData<DC2,M>*> celldata2;
    Field<DC1>::BuildCellData(file1,celldata1);
    Field<DC2>::BuildCellData(file2,celldata2);

    _metric = M;

    const int n = celldata1.size();
    const int sqrtn = int(sqrt(double(n)));

#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this);
        bc2.clearData();
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i=0;i<n;++i) {
            // Let the progress dots happen every sqrt(n) iterations.
            if (i % sqrtn == 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    //xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                    dbg<<'.';
                }
            }
            // Note: This transfers ownership of the pointer to the Cell,
            // and the data is deleted when the Cell goes out of scope.
            // TODO: I really should switch to using shared_ptr, so these
            // memory issues are more seamless...
            Cell<DC1,M> c1(celldata1[i]);
            Cell<DC2,M> c2(celldata2[i]);
            const double dsq = DistSq(c1.getData().getPos(),c2.getData().getPos());
            if (dsq >= _minsepsq && dsq < _maxsepsq) {
                bc2.directProcess11(c1,c2,dsq);
            }
        }
        dbg<<std::endl;
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc2;
        }
    }
#endif
}
#endif

#if 0
template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::processPairwise(const InputFile& file1, const InputFile& file2)
{
    dbg<<"Starting processPairwise for 2 files: "<<
        file1.getFileName()<<"  "<<file2.getFileName()<<std::endl;
    const int n = file1.getNTot();
    const int n2 = file2.getNTot();
    Assert(n > 0);
    Assert(n == n2);
    xdbg<<"files have "<<n<<" objects\n";

    if (file1.useRaDec()) {
        Assert(file2.useRaDec());
        doProcessPairwise<Sphere>(file1,file2);
    } else {
        Assert(!file2.useRaDec());
        doProcessPairwise<Flat>(file1,file2);
    }
}
#endif

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

    if (dsq < _minsepsq && s1ps2 < _minsep && dsq < SQR(_minsep - s1ps2)) return;
    if (dsq >= _maxsepsq && dsq >= SQR(_maxsep + s1ps2)) return;

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
    { xi.xi[k] += c1.getData().getW() * c2.getData().getK(); }
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

        xi.xip[k] += g1rg2r - g1ig2i;
        xi.xip_im[k] += g1ig2r + g1rg2i;
        xi.xim[k] += g1rg2r + g1ig2i;
        xi.xim_im[k] += g1ig2r - g1rg2i;
    }
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
    double ww = double(c1.getData().getW()) * double(c2.getData().geWN());
    _meanlogr[k] += ww * logr;
    _weight[k] += ww;

    DirectHelper<DC1,DC2>::ProcessXi(c1,c2,dsq,_xi,k);
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.Copy(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _varxi[i] = rhs._varxi[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator+=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._nbins == _nbins);
    _xi.Add(rhs._xi,_nbins);
    for (int i=0; i<_nbins; ++i) _varxi[i] = rhs._varxi[i];
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = rhs._meanlogr[i];
    for (int i=0; i<_nbins; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_nbins; ++i) _npairs[i] = rhs._npairs[i];
}

template class BinnedCorr2<NData,NData>;
template class BinnedCorr2<NData,GData>;
template class BinnedCorr2<GData,GData>;
template class BinnedCorr2<KData,KData>;
template class BinnedCorr2<NData,KData>;
template class BinnedCorr2<KData,GData>;


#include "dbg.h"
#include "BinData2.h"

// First a helper struct to do the right thing with shears depending on the metric.
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
        xxdbg<<"expm2ibeta = "<<expm2iarg<<std::endl;
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

        xxdbg<<"p1 = "<<p1<<", p2 = "<<p2<<std::endl;
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
        xxdbg<<"A = "<<atan2(sinA,cosA)*180./M_PI<<std::endl;
        double cosAsq = cosA*cosA;
        double sinAsq = crosssq;
        double normAsq = cosAsq + sinAsq;
        xxdbg<<"normAsq = "<<cosAsq<<" + "<<sinAsq<<" = "<<normAsq<<std::endl;
        Assert(normAsq > 0.);
        double cos2A = (cosAsq - sinAsq) / normAsq; // These are now correct.
        double sin2A = 2.*sinA*cosA / normAsq;
        xxdbg<<"2A = "<<atan2(sin2A,cos2A)*180./M_PI<<std::endl;

        // In fact, A is not really the angles by which we want to rotate the shear.
        // We really want to rotae by the angle between due _east_ and c, not _north_.
        //
        // exp(-2ialpha) = exp(-2i (A - Pi/2) )
        //               = exp(iPi) * exp(-2iA)
        //               = - exp(-2iA)

        std::complex<double> expm2ialpha(-cos2A,sin2A);
        xxdbg<<"expm2ialpha = "<<expm2ialpha<<std::endl;
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
        xxdbg<<"B = "<<atan2(sinB,cosB)*180./M_PI<<std::endl;
        double cosBsq = cosB*cosB;
        double sinBsq = crosssq;
        double normBsq = cosBsq + sinBsq;
        xxdbg<<"normBsq = "<<cosBsq<<" + "<<sinBsq<<" = "<<normBsq<<std::endl;
        Assert(normBsq != 0.);
        double cos2B = (cosBsq - sinBsq) / normBsq;
        double sin2B = 2.*sinB*cosB / normBsq;
        xxdbg<<"2B = "<<atan2(sin2B,cos2B)*180./M_PI<<std::endl;

        // exp(-2ibeta)  = exp(-2i (Pi/2 - B) )
        //               = exp(-iPi) * exp(2iB)
        //               = - exp(2iB)

        std::complex<double> expm2ibeta(-cos2B,-sin2B); 
        xxdbg<<"expm2ibeta = "<<expm2ibeta<<std::endl;
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


// 
// NN
//

template <int M>
void BinData2<NData,NData>::directProcess11(
    const Cell<NData,M>& c1, const Cell<NData,M>& c2, double dsq, double logr)
{
    xxdbg<<"DirectProcess11: p1 = "<<c1.getData().getPos()<<", p2 = "<<c2.getData().getPos()<<std::endl;
    double nn = double(c1.getData().getN()) * double(c2.getData().getN());
    meanlogr += nn * logr;
    npair += nn;
}

void BinData2<NData,NData>::finalize(double, double)
{
    xxdbg<<"Finalize NN:\n";
    xxdbg<<"   "<<meanlogr<<"  "<<npair<<std::endl;
    if (npair == 0.) {
        meanlogr = 0.;
    } else {
        meanlogr /= npair;
    }
    xxdbg<<"=> "<<meanlogr<<std::endl;
}

//
// NG
//

template <int M>
void BinData2<NData,GData>::directProcess11(
    const Cell<NData,M>& c1, const Cell<GData,M>& c2,
    double dsq, double logr)
{
    xxdbg<<"DirectProcess11: p1 = "<<c1.getData().getPos()<<", p2 = "<<c2.getData().getPos()<<std::endl;
    std::complex<double> g2;
    MetricHelper<M>::ProjectShear(c1,c2,dsq,g2);

    double nw = double(c1.getData().getN()) * c2.getData().getW();
    double nn = double(c1.getData().getN()) * double(c2.getData().getN());

    // The minus sign here is to make it accumulate tangential shear, rather than radial.
    // g2 from the above ProjectShear is measured along the connecting line, not tangent.
    meangammat -= double(c1.getData().getN()) * g2;
    xxdbg<<"<gt> => "<<meangammat<<std::endl;
    weight += nw;
    meanlogr += nw*logr;
    npair += nn;

#if 0
    static bool foundnan = false;
    if (!foundnan && (IsNan(meangammat))) {
        foundnan = true;
        dbg<<"Found nan\n";
        dbgout = &std::cout;
        directProcess11(c1,c2,dsq,logr);
        myerror("Found nan");
        exit(1);
    }
#endif
}

void BinData2<NData,GData>::finalize(double, double varg)
{
    xxdbg<<"Finalize: weight = "<<weight<<"  varg = "<<varg<<std::endl;
    xxdbg<<"   "<<meangammat<<"  "<<meanlogr<<"  "<<vargammat<<std::endl;
#if 0
    double temp = real(meangammat);
    xdbg<<"temp = "<<temp<<std::endl;
    xdbg<<"(temp != temp) = "<<(temp != temp)<<std::endl;
    xdbg<<"(temp*temp >= 0) = "<<(temp*temp >= 0.)<<std::endl;
    xdbg<<"!(temp*temp >= 0) = "<<!(temp*temp >= 0.)<<std::endl;
    xdbg<<"IsNan(temp) = "<<IsNan(temp)<<std::endl;
    xdbg<<"IsNan(meangammat) = "<<IsNan(meangammat)<<std::endl;
    if (IsNan(meangammat) || IsNan(ximinus)) {
        dbgout = &std::cout;
        dbg<<"Found nan\n";
        myerror("Found nan");
        exit(1);
    }
#endif
    if (weight == 0.) {
        meangammat = meanlogr = vargammat = 0.;
    } else {
        meangammat /= weight;
        meanlogr /= weight;
        Assert(npair != 0.);
        vargammat = varg/npair;
    }
    xxdbg<<"=> "<<meangammat<<"  "<<meanlogr<<"  "<<vargammat<<std::endl;
}

//
// GG
//

template <class T>
static bool IsNan(T x) { return (x != x) || !(x*x >= T(0)); }
template <class T>
static bool IsNan(std::complex<T> z) { return IsNan(real(z)) || IsNan(imag(z)); }


template <int M>
void BinData2<GData,GData>::directProcess11(
    const Cell<GData,M>& c1, const Cell<GData,M>& c2,
    double dsq, double logr)
{
    xxdbg<<"DirectProcess11: p1 = "<<c1.getData().getPos()<<", p2 = "<<c2.getData().getPos()<<std::endl;
    xxdbg<<"dsq = "<<dsq<<std::endl;
    xxdbg<<"logr = "<<logr<<std::endl;
    xxdbg<<"raw g1,g2 = "<<c1.getData().getWG()<<", "<<c2.getData().getWG()<<std::endl;
    std::complex<double> g1, g2;
    MetricHelper<M>::ProjectShears(c1,c2,dsq,g1,g2);
    xxdbg<<"g1,g2 = "<<g1<<','<<g2<<std::endl;

    // The complex products g1 g2 and g1 g2* share most of the calculations,
    // so faster to do this manually.
    double g1rg2r = g1.real() * g2.real();
    double g1rg2i = g1.real() * g2.imag();
    double g1ig2r = g1.imag() * g2.real();
    double g1ig2i = g1.imag() * g2.imag();

    const std::complex<double> ee ( g1rg2r - g1ig2i , g1ig2r + g1rg2i );
    const std::complex<double> eec( g1rg2r + g1ig2i , g1ig2r - g1rg2i );
    xxdbg<<"ee,eec = "<<ee<<" "<<eec<<std::endl;

    double ww = c1.getData().getW() * c2.getData().getW();
    double nn = double(c1.getData().getN()) * double(c2.getData().getN());

    xiplus += eec;
    ximinus += ee;
    xxdbg<<"xi => "<<xiplus<<" "<<ximinus<<std::endl;
    weight += ww;
    meanlogr += ww*logr;
    npair += nn;

#if 0
    static bool foundnan = false;
    if (!foundnan && (IsNan(xiplus) || IsNan(ximinus))) {
        foundnan = true;
        dbg<<"Found nan\n";
        dbgout = &std::cout;
        directProcess11(c1,c2,dsq,logr);
        myerror("Found nan");
        exit(1);
    }
#endif
}

void BinData2<GData,GData>::finalize(double varg1, double varg2)
{
    xxdbg<<"Finalize: weight = "<<weight<<"  varg = "<<varg1<<", "<<varg2<<std::endl;
    xxdbg<<"   "<<xiplus<<"  "<<ximinus<<"  "<<meanlogr<<"  "<<varxi<<std::endl;
#if 0
    double temp = real(xiplus);
    xdbg<<"temp = "<<temp<<std::endl;
    xdbg<<"(temp != temp) = "<<(temp != temp)<<std::endl;
    xdbg<<"(temp*temp >= 0) = "<<(temp*temp >= 0.)<<std::endl;
    xdbg<<"!(temp*temp >= 0) = "<<!(temp*temp >= 0.)<<std::endl;
    xdbg<<"IsNan(temp) = "<<IsNan(temp)<<std::endl;
    xdbg<<"IsNan(xiplus) = "<<IsNan(xiplus)<<std::endl;
    if (IsNan(xiplus) || IsNan(ximinus)) {
        dbgout = &std::cout;
        dbg<<"Found nan\n";
        myerror("Found nan");
        exit(1);
    }
#endif
    if (weight == 0.) {
        xiplus = ximinus = meanlogr = varxi = 0.;
    } else {
        xiplus /= weight;
        ximinus /= weight;
        meanlogr /= weight;
        Assert(npair != 0.);
        varxi = varg1*varg2/npair;
    }
    xxdbg<<"=> "<<xiplus<<"  "<<ximinus<<"  "<<meanlogr<<"  "<<varxi<<std::endl;
}


//
// NK
//

template <int M>
void BinData2<NData,KData>::directProcess11(
    const Cell<NData,M>& c1, const Cell<KData,M>& c2,
    double dsq, double logr)
{
    xxdbg<<"DirectProcess11: p1 = "<<c1.getData().getPos()<<", p2 = "<<c2.getData().getPos()<<std::endl;
    double nw = double(c1.getData().getN()) * c2.getData().getW();
    double nn = double(c1.getData().getN()) * double(c2.getData().getN());

    meankappa += double(c1.getData().getN()) * c2.getData().getWK();
    weight += nw;
    meanlogr += nw*logr;
    npair += nn;
}

void BinData2<NData,KData>::finalize(double, double vark)
{
    xxdbg<<"Finalize: weight = "<<weight<<"  vark = "<<vark<<std::endl;
    xxdbg<<"   "<<meankappa<<"  "<<meanlogr<<"  "<<varkappa<<std::endl;
    if (weight == 0.) {
        meankappa = meanlogr = varkappa = 0.;
    } else {
        meankappa /= weight;
        meanlogr /= weight;
        Assert(npair != 0.);
        varkappa = vark/npair;
    }
    xxdbg<<"=> "<<meankappa<<"  "<<meanlogr<<"  "<<varkappa<<std::endl;
}

//
// KK
//

template <int M>
void BinData2<KData,KData>::directProcess11(
    const Cell<KData,M>& c1, const Cell<KData,M>& c2,
    double dsq, double logr)
{
    xxdbg<<"DirectProcess11: p1 = "<<c1.getData().getPos()<<", p2 = "<<c2.getData().getPos()<<std::endl;
    xxdbg<<"dsq = "<<dsq<<std::endl;
    xxdbg<<"logr = "<<logr<<std::endl;

    double ww = c1.getData().getW() * c2.getData().getW();
    double nn = double(c1.getData().getN()) * double(c2.getData().getN());

    xi += c1.getData().getWK() * c2.getData().getWK();
    weight += ww;
    meanlogr += ww*logr;
    npair += nn;
}

void BinData2<KData,KData>::finalize(double vark1, double vark2)
{
    xxdbg<<"Finalize: weight = "<<weight<<"  vark = "<<vark1<<", "<<vark2<<std::endl;
    xxdbg<<"   "<<xi<<"  "<<meanlogr<<"  "<<varxi<<std::endl;
    if (weight == 0.) {
        xi = meanlogr = varxi = 0.;
    } else {
        xi /= weight;
        meanlogr /= weight;
        Assert(npair != 0.);
        varxi = vark1*vark2/npair;
    }
    xxdbg<<"=> "<<xi<<"  "<<meanlogr<<"  "<<varxi<<std::endl;
}


//
// KG
//

template <int M>
void BinData2<KData,GData>::directProcess11(
    const Cell<KData,M>& c1, const Cell<GData,M>& c2,
    double dsq, double logr)
{
    xxdbg<<"DirectProcess11: p1 = "<<c1.getData().getPos()<<", p2 = "<<c2.getData().getPos()<<std::endl;
    xxdbg<<"dsq = "<<dsq<<std::endl;
    xxdbg<<"logr = "<<logr<<std::endl;
    xxdbg<<"k1 = "<<c1.getData().getWK();
    xxdbg<<"raw g2 = "<<c2.getData().getWG()<<std::endl;
    std::complex<double> g2;
    MetricHelper<M>::ProjectShear(c1,c2,dsq,g2);
    xxdbg<<"g2 = "<<g2<<std::endl;

    // The minus sign here is to make it accumulate tangential shear, rather than radial.
    // g2 from the above ProjectShear is measured along the connecting line, not tangent.
    meankgammat -= c1.getData().getWK() * g2;
    double ww = c1.getData().getW() * c2.getData().getW();
    double nn = double(c1.getData().getN()) * double(c2.getData().getN());
    weight += ww;
    meanlogr += ww*logr;
    npair += nn;
}

void BinData2<KData,GData>::finalize(double vark, double varg)
{
    xxdbg<<"Finalize: weight = "<<weight<<"  vark = "<<vark<<", varg = "<<varg<<std::endl;
    xxdbg<<"   "<<meankgammat<<"  "<<meanlogr<<"  "<<varkgammat<<std::endl;
    if (weight == 0.) {
        meankgammat = meanlogr = varkgammat = 0.;
    } else {
        meankgammat /= weight;
        meanlogr /= weight;
        Assert(npair != 0.);
        varkgammat = vark*varg/npair;
    }
    xxdbg<<"=> "<<meankgammat<<"  "<<meanlogr<<"  "<<varkgammat<<std::endl;
}


template void BinData2<NData,NData>::directProcess11(
    const Cell<NData,Flat>& c1, const Cell<NData,Flat>& c2,
    double dsq, double logr);
template void BinData2<NData,NData>::directProcess11(
    const Cell<NData,Sphere>& c1, const Cell<NData,Sphere>& c2,
    double dsq, double logr);
template void BinData2<NData,GData>::directProcess11(
    const Cell<NData,Flat>& c1, const Cell<GData,Flat>& c2,
    double dsq, double logr);
template void BinData2<NData,GData>::directProcess11(
    const Cell<NData,Sphere>& c1, const Cell<GData,Sphere>& c2,
    double dsq, double logr);
template void BinData2<GData,GData>::directProcess11(
    const Cell<GData,Flat>& c1, const Cell<GData,Flat>& c2,
    double dsq, double logr);
template void BinData2<GData,GData>::directProcess11(
    const Cell<GData,Sphere>& c1, const Cell<GData,Sphere>& c2,
    double dsq, double logr);
template void BinData2<NData,KData>::directProcess11(
    const Cell<NData,Flat>& c1, const Cell<KData,Flat>& c2,
    double dsq, double logr);
template void BinData2<NData,KData>::directProcess11(
    const Cell<NData,Sphere>& c1, const Cell<KData,Sphere>& c2,
    double dsq, double logr);
template void BinData2<KData,KData>::directProcess11(
    const Cell<KData,Flat>& c1, const Cell<KData,Flat>& c2,
    double dsq, double logr);
template void BinData2<KData,KData>::directProcess11(
    const Cell<KData,Sphere>& c1, const Cell<KData,Sphere>& c2,
    double dsq, double logr);
template void BinData2<KData,GData>::directProcess11(
    const Cell<KData,Flat>& c1, const Cell<GData,Flat>& c2,
    double dsq, double logr);
template void BinData2<KData,GData>::directProcess11(
    const Cell<KData,Sphere>& c1, const Cell<GData,Sphere>& c2,
    double dsq, double logr);
